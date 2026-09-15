#!/usr/bin/env python3
"""Run the frozen A-D ablation with BF16 and fail-fast diagnostics.

This orchestration layer deliberately leaves the model and allocator source
untouched.  It wraps the existing TrainerAuxiliary/EvaluatorAuxiliary entry
points with BF16 autocast, keeps the already-forensically-identified global
Hydra/Mamba FP32 boundary, and writes reproducibility/step diagnostics beside
each run.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hydra"))

from libs import load_opt  # noqa: E402
from libs.dist_utils import get_rank, print0  # noqa: E402
from libs.train_utils import AverageMeter  # noqa: E402
from libs.worker import EvaluatorAuxiliary, TrainerAuxiliary  # noqa: E402


CONFIGS = {
    "A": ROOT / "opts/research_ablation_A_baseline.yaml",
    "B": ROOT / "opts/research_ablation_B_query_modulation.yaml",
    "C": ROOT / "opts/research_ablation_C_boundary_importance.yaml",
    "D": ROOT / "opts/research_ablation_D_adaptive_anchor.yaml",
}


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (torch.dtype,)):
        return str(value)
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    raise TypeError("not JSON serializable: {}".format(type(value).__name__))


def _write_json(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")


def _finite(value: Any) -> bool:
    if torch.is_tensor(value):
        return not (value.is_floating_point() or value.is_complex()) or bool(
            torch.isfinite(value).all()
        )
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite(item) for item in value)
    return True


def _cast_float(value: Any):
    if torch.is_tensor(value):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, tuple):
        return tuple(_cast_float(item) for item in value)
    if isinstance(value, list):
        return [_cast_float(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _cast_float(item) for key, item in value.items()}
    return value


def _install_named_fp32_boundaries(model: nn.Module, predicate):
    """Disable autocast only inside modules selected by ``predicate``.

    The original module object and parameter names are preserved, so EMA and
    checkpoint state dictionaries remain byte-for-byte compatible with the
    normal model implementation.
    """
    installed = []
    for name, module in model.named_modules():
        if not predicate(name, module):
            continue
        if getattr(module, "_formal_fp32_boundary", False):
            continue
        original_forward = module.forward

        def forward_fp32(*args, _original=original_forward, **kwargs):
            with torch.autocast(device_type="cuda", enabled=False):
                args32 = tuple(_cast_float(arg) for arg in args)
                kwargs32 = {key: _cast_float(value) for key, value in kwargs.items()}
                output = _original(*args32, **kwargs32)
            return _cast_float(output)

        module.forward = forward_fp32
        module._formal_fp32_boundary = True
        installed.append(name)
    return installed


def _install_global_fp32_boundary(model: nn.Module):
    return _install_named_fp32_boundaries(
        model, lambda name, _module: name.endswith("global_encoder")
    )


def install_precision_boundaries(model: nn.Module, policy: str):
    """Install one of the explicit D-recovery precision policies.

    P2 remains the default formal-training behavior. P1 and P3 are exposed for
    controlled numerical forensics without changing module or parameter names.
    """
    normalized = policy.strip().lower()
    if normalized == "p1":
        predicate = lambda name, _module: (
            name == "vid_net.branch.2.global_encoder"
        )
    elif normalized == "p2":
        predicate = lambda name, _module: name.endswith("global_encoder")
    elif normalized == "p3":
        predicate = lambda name, _module: (
            name.endswith("global_encoder")
            or name.endswith("norm_global")
            or name.endswith("norm_global_in")
            or name.endswith("query_modulation_mlp")
        )
    else:
        raise ValueError("unknown precision boundary policy: {}".format(policy))
    return _install_named_fp32_boundaries(model, predicate)


def _unwrap(model, distributed=False):
    return model.module if distributed and hasattr(model, "module") else model


def _tensor_stats(value: torch.Tensor, mask: Optional[torch.Tensor] = None):
    if value is None:
        return {"count": 0, "nonfinite": 0}
    tensor = value.detach().float()
    if mask is not None:
        mask = mask.to(device=tensor.device, dtype=torch.bool)
        if tensor.ndim == mask.ndim + 1:
            if tuple(tensor.shape[:mask.ndim]) == tuple(mask.shape):
                mask = mask.unsqueeze(-1).expand_as(tensor)
            elif tensor.shape[0] == mask.shape[0] and tensor.shape[-1] == mask.shape[-1]:
                mask = mask.unsqueeze(1).expand_as(tensor)
            else:
                raise RuntimeError(
                    "statistics mask shape {} does not match tensor {}".format(
                        tuple(mask.shape), tuple(tensor.shape)
                    )
                )
        elif tuple(tensor.shape) != tuple(mask.shape):
            raise RuntimeError(
                "statistics mask shape {} does not match tensor {}".format(
                    tuple(mask.shape), tuple(tensor.shape)
                )
            )
        tensor = tensor[mask]
    if tensor.numel() == 0:
        return {"count": 0, "nonfinite": 0}
    finite = torch.isfinite(tensor)
    finite_values = tensor[finite]
    result = {
        "count": int(tensor.numel()),
        "nonfinite": int((~finite).sum().item()),
    }
    if finite_values.numel():
        result.update({
            "min": float(finite_values.min().item()),
            "max": float(finite_values.max().item()),
            "mean": float(finite_values.mean().item()),
            "std": float(finite_values.std(unbiased=False).item()),
            "p50": float(torch.quantile(finite_values, 0.50).item()),
            "p90": float(torch.quantile(finite_values, 0.90).item()),
            "p95": float(torch.quantile(finite_values, 0.95).item()),
            "p99": float(torch.quantile(finite_values, 0.99).item()),
        })
    return result


def _mask2(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 3:
        if mask.size(1) != 1:
            raise RuntimeError("expected singleton mask channel")
        return mask[:, 0]
    if mask.ndim != 2:
        raise RuntimeError("expected (B,T) or (B,1,T) mask")
    return mask


def _config_flatten(value: Any, prefix=""):
    if isinstance(value, Mapping):
        result = {}
        for key in sorted(value):
            child = "{}.{}".format(prefix, key) if prefix else str(key)
            result.update(_config_flatten(value[key], child))
        return result
    if isinstance(value, (tuple, list)):
        return {prefix: json.dumps(value, sort_keys=True)}
    return {prefix: value}


def generate_config_diff(output: Path):
    raw = {label: yaml.safe_load(path.read_text()) for label, path in CONFIGS.items()}
    normalized = {label: load_opt(str(path), is_training=True) for label, path in CONFIGS.items()}
    flat = {label: _config_flatten(value) for label, value in normalized.items()}
    keys = sorted(set().union(*(values.keys() for values in flat.values())))
    lines = [
        "A-D formal ablation configuration diff",
        "Generated from normalized load_opt() configurations.",
        "git_available=false (repository has no .git directory).",
        "",
    ]
    for key in keys:
        values = {label: flat[label].get(key) for label in CONFIGS}
        if len({json.dumps(value, sort_keys=True, default=str) for value in values.values()}) > 1:
            lines.append("[DIFF] {}".format(key))
            for label in CONFIGS:
                lines.append("  {} = {}".format(label, values[label]))
    lines.extend([
        "",
        "Expected research-variable differences:",
        "  B: model.vid_net.query_modulation=true",
        "  C: B + model.vid_net.query_boundary_importance=true and auxiliary importance loss enabled",
        "  D: C + adaptive_anchor=true, explicit eight 0.5 keep ratios, assignment_acc=true",
        "  D: progressive/coarse-to-fine/soft grouping/STE remain disabled",
        "",
        "Raw YAML snapshots:",
    ])
    for label, path in CONFIGS.items():
        lines.append("  {}: {}".format(label, path))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    return raw, normalized


def source_fingerprint():
    digest = hashlib.sha256()
    files = []
    for directory in (ROOT / "libs", ROOT / "tools", ROOT / "opts", ROOT / "tests"):
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix not in {".py", ".yaml", ".yml"}:
                continue
            relative = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(data)
            files.append(relative)
    return {"git_unavailable": True, "sha256": digest.hexdigest(), "files": files}


class FormalTrainer(TrainerAuxiliary):
    """Existing trainer with BF16/finite checks and side-channel diagnostics."""

    def __init__(
        self,
        opt,
        label: str,
        run_root: Path,
        max_steps: Optional[int] = None,
        precision_policy: str = "p2",
    ):
        self.formal_label = label
        self.run_root = run_root
        self.max_steps = max_steps
        self.precision_policy = precision_policy.strip().lower()
        self._last_capture = {}
        self._failed = None
        super().__init__(opt)
        self.run_root.mkdir(parents=True, exist_ok=True)
        model_ref = _unwrap(self.model, bool(opt.get("_distributed", False)))
        self.fp32_boundary_modules = install_precision_boundaries(
            model_ref, self.precision_policy
        )
        self.fp32_loss_modules = []
        for _loss_name in ("query_boundary_importance_loss", "boundary_supervision_loss", "allocation_supervision_loss", "ranking_allocation_loss"):
            _loss_module = getattr(self, _loss_name, None)
            if _loss_module is None:
                continue
            _original_loss_forward = _loss_module.forward
            def _forward_fp32(*args, _original=_original_loss_forward, **kwargs):
                with torch.autocast(device_type="cuda", enabled=False):
                    args32 = tuple(_cast_float(arg) for arg in args)
                    kwargs32 = {key: _cast_float(value) for key, value in kwargs.items()}
                    return _original(*args32, **kwargs32)
            _loss_module.forward = _forward_fp32
            self.fp32_loss_modules.append(_loss_name)
        self._capture_handle = model_ref.register_forward_hook(self._capture_outputs)
        self._metrics_handle = (self.run_root / "training_metrics.jsonl").open("a")
        self._diagnostic_summary = {
            "label": label,
            "precision": "bf16",
            "precision_policy": self.precision_policy,
            "grad_scaler": False,
            "fp32_boundary_modules": self.fp32_boundary_modules,
            "fp32_loss_modules": self.fp32_loss_modules,
            "steps": 0,
            "status": "initialized",
        }
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("formal A-D training requires CUDA BF16 support")
        self._write_jsonl({
            "event": "run_start",
            "label": label,
            "precision": "bf16",
            "precision_policy": self.precision_policy,
            "grad_scaler": False,
            "seed": int(opt.get("seed", -1)),
            "cuda_device": torch.cuda.get_device_name(torch.cuda.current_device()),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "fp32_boundary_modules": self.fp32_boundary_modules,
            "fp32_loss_modules": self.fp32_loss_modules,
        })

    def _write_jsonl(self, record):
        self._metrics_handle.write(json.dumps(record, default=_json_default) + "\n")
        self._metrics_handle.flush()

    def _capture_outputs(self, _module, _inputs, output):
        if not isinstance(output, (tuple, list)) or len(output) < 8:
            return
        self._last_capture = {
            "fpn_masks": tuple(item.detach() for item in output[3]),
            "sequence_masks": tuple(item.detach() for item in output[5]),
            "anchor_masks": tuple(item.detach() for item in output[7]),
        }
        if self.adaptive_anchor and self.assignment_acc:
            extras = output[8:]
            if self.return_importance_debug:
                if len(extras) >= 2:
                    self._last_capture["assignments"] = tuple(
                        item.detach() for item in extras[1]
                    )
            elif extras:
                self._last_capture["assignments"] = tuple(
                    item.detach() for item in extras[0]
                )

    def _gradient_forensics(self):
        records = []
        nonfinite = []
        total_sq = None
        for name, parameter in self.model.named_parameters():
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach()
            finite = bool(torch.isfinite(grad).all())
            record = {
                "name": name,
                "dtype": str(grad.dtype),
                "finite": finite,
                "finite_ratio": float(torch.isfinite(grad).float().mean().item()),
                "min": float(grad.float().min().item()),
                "max": float(grad.float().max().item()),
                "norm": float(grad.float().norm().item()),
            }
            records.append(record)
            if not finite:
                nonfinite.append(record)
            else:
                contribution = grad.float().square().sum()
                total_sq = contribution if total_sq is None else total_sq + contribution
        return {
            "finite": not nonfinite,
            "first_nonfinite": nonfinite[0] if nonfinite else None,
            "parameters": records,
            "norm": float(torch.sqrt(total_sq).item()) if total_sq is not None else 0.0,
        }

    def _check_optimizer_state(self):
        bad = []
        for index, state in enumerate(self.optimizer.state.values()):
            for key, value in state.items():
                if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all():
                    bad.append({"state_index": index, "key": key})
        return {"finite": not bad, "nonfinite": bad}

    def _check_temporal(self):
        if not self.adaptive_anchor:
            return {"adaptive": False, "finite": True}
        model_ref = _unwrap(self.model, bool(self.opt.get("_distributed", False)))
        metadata = getattr(model_ref.vid_net, "last_temporal_metadata", None)
        if metadata is None:
            raise RuntimeError("adaptive forward did not expose temporal metadata")
        sequence_masks = self._last_capture.get("sequence_masks")
        anchor_masks = self._last_capture.get("anchor_masks")
        assignments = self._last_capture.get("assignments")
        if sequence_masks is None or anchor_masks is None:
            raise RuntimeError("adaptive forward did not expose masks")
        levels = []
        for level, (geometry, sequence_mask, anchor_mask) in enumerate(
            zip(metadata, sequence_masks, anchor_masks)
        ):
            valid = _mask2(sequence_mask)
            anchors = _mask2(anchor_mask)
            if geometry.ndim != 3 or geometry.size(-1) != 5:
                raise RuntimeError("invalid temporal metadata shape at level {}".format(level))
            if not torch.isfinite(geometry).all():
                raise RuntimeError("nonfinite temporal metadata at level {}".format(level))
            values = geometry.float()
            valid_values = values[valid]
            if valid_values.numel():
                if bool((valid_values[:, 0] > valid_values[:, 2]).any()) or bool((valid_values[:, 2] > valid_values[:, 1]).any()):
                    raise RuntimeError("metadata start/center/end invariant failed at level {}".format(level))
                if bool((valid_values[:, 3] <= 0).any()) or bool((valid_values[:, 4] <= 0).any()):
                    raise RuntimeError("metadata span/scale invariant failed at level {}".format(level))
                for row in range(values.size(0)):
                    count = int(valid[row].sum().item())
                    if count > 1 and bool((values[row, 1:count, 2] <= values[row, :count - 1, 2]).any()):
                        raise RuntimeError("metadata centers are not strictly monotonic at level {}".format(level))
            level_record = {
                "level": level,
                "metadata": _tensor_stats(values, valid),
                "input_tokens": [int(item) for item in valid.sum(dim=-1).detach().cpu().tolist()],
                "anchor_tokens": [int(item) for item in anchors.sum(dim=-1).detach().cpu().tolist()],
            }
            if assignments is not None:
                assignment = assignments[level].float()
                if assignment.ndim != 3 or assignment.size(0) != valid.size(0):
                    raise RuntimeError("invalid assignment shape at level {}".format(level))
                membership = assignment > 0.5
                if bool((membership & ~valid[:, None, :]).any()):
                    raise RuntimeError("padding token assigned at level {}".format(level))
                coverage = (membership & valid[:, None, :]).sum(dim=1)
                if bool((coverage[valid] != 1).any()):
                    raise RuntimeError("valid token coverage failed at level {}".format(level))
                member_counts = membership.sum(dim=-1)
                if bool((member_counts[anchors] == 0).any()) or bool((member_counts[~anchors] != 0).any()):
                    raise RuntimeError("empty/padded anchor invariant failed at level {}".format(level))
                group_sizes = member_counts[anchors]
                level_record["group_size"] = _tensor_stats(group_sizes)
                level_record["assignment_nonzero"] = int(membership.sum().item())
            levels.append(level_record)
        return {"adaptive": True, "finite": True, "levels": levels}

    def _region_diagnostics(self, data_list):
        if not self.adaptive_anchor:
            return None
        model_ref = _unwrap(self.model, bool(self.opt.get("_distributed", False)))
        metadata = getattr(model_ref.vid_net, "last_temporal_metadata", None)
        masks = self._last_capture.get("sequence_masks")
        if metadata is None or masks is None:
            return None
        divisor = 1 if self.adaptive_anchor else self.vid_stride
        target = torch.cat([item["target"] / divisor for item in data_list]).to(metadata[0].device)
        output = {"boundary": [], "foreground": [], "background": []}
        for geometry, mask in zip(metadata, masks):
            valid = _mask2(mask)
            rows = min(geometry.size(0), target.size(0))
            geometry = geometry[:rows].float()
            valid = valid[:rows]
            centers, spans = geometry[..., 2], geometry[..., 3]
            for row in range(rows):
                valid_row = valid[row]
                if not bool(valid_row.any()):
                    continue
                start, end = target[row, 0], target[row, 1]
                center = centers[row]
                span = spans[row]
                boundary = ((center - start).abs() <= span) | ((center - end).abs() <= span)
                foreground = (center >= start) & (center <= end) & ~boundary
                background = ~(boundary | foreground)
                for name, selector in (("boundary", boundary), ("foreground", foreground), ("background", background)):
                    selected = valid_row & selector
                    if bool(selected.any()):
                        output[name].append({
                            "span": _tensor_stats(span, selected),
                            "group_size": None,
                        })
        return output

    def _grouping_difference(self):
        assignments = self._last_capture.get("assignments")
        if not assignments:
            return None
        result = []
        for level, assignment in enumerate(assignments):
            if assignment.size(0) < 2:
                continue
            cuts = assignment.float().argmax(dim=1)[:, 1:] != assignment.float().argmax(dim=1)[:, :-1]
            a = cuts[0]
            b = cuts[1]
            union = (a | b).sum().item()
            inter = (a & b).sum().item()
            result.append({
                "level": level,
                "cut_jaccard": float(inter / union) if union else 1.0,
                "different_cut_count": int((a != b).sum().item()),
                "assignment_identical": bool(torch.equal(assignment[0], assignment[1])),
            })
        return result

    def _diagnostics(self, data_list, loss_dict, gradient, elapsed, memory):
        temporal = self._check_temporal()
        model_ref = _unwrap(self.model, bool(self.opt.get("_distributed", False)))
        record = {
            "event": "step",
            "label": self.formal_label,
            "epoch": int(self.epoch),
            "iteration": int(self.itr + 1),
            "precision": "bf16",
            "grad_scaler": False,
            "loss": {key: float(value.detach().item()) for key, value in loss_dict.items()},
            "learning_rate": float(self.scheduler.get_last_lr()[0]),
            "gradient_norm": gradient["norm"],
            "gradient_finite": gradient["finite"],
            "parameter_finite": all(bool(torch.isfinite(parameter).all()) for parameter in self.model.parameters()),
            "optimizer_state_finite": self._check_optimizer_state(),
            "iteration_latency_sec": elapsed,
            "peak_allocated_bytes": int(memory[0]),
            "peak_reserved_bytes": int(memory[1]),
            "temporal": temporal,
            "importance": None,
            "adaptive_layers": temporal.get("levels") if temporal.get("adaptive") else None,
            "region": self._region_diagnostics(data_list),
            "query_grouping": self._grouping_difference(),
        }
        importance_debug = getattr(model_ref.vid_net, "last_importance_debug", None)
        if importance_debug:
            importance_records = []
            for level, debug in enumerate(importance_debug):
                level_record = {}
                level_mask = _mask2(self._last_capture["sequence_masks"][level])
                for key, value in debug.items():
                    if not torch.is_tensor(value):
                        continue
                    try:
                        level_record[key] = _tensor_stats(value, level_mask)
                        level_record[key]["mask_applied"] = True
                    except RuntimeError:
                        level_record[key] = _tensor_stats(value)
                        level_record[key]["mask_applied"] = False
                    level_record[key]["source_shape"] = list(value.shape)
                importance_records.append(level_record)
            record["importance"] = importance_records
        return record

    def _crash(self, exc):
        self._failed = repr(exc)
        payload = {
            "label": self.formal_label,
            "status": "failed",
            "epoch": int(self.epoch),
            "iteration": int(self.itr),
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
            "last_capture": {key: str(type(value).__name__) for key, value in self._last_capture.items()},
        }
        _write_json(self.run_root / "crash_diagnostics.json", payload)
        try:
            self._write_jsonl({"event": "crash", **payload})
        except Exception:
            pass

    def run(self):
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA BF16 is required for formal training")
        print0("Formal BF16 training started: {}".format(self.formal_label))
        self._diagnostic_summary["status"] = "running"
        try:
            while self.epoch < self.num_epochs:
                self.dataset.set_epoch(self.epoch)
                if self.opt["_distributed"]:
                    self.sampler.set_epoch(self.epoch)
                for data_list in self.dataloader:
                    if self.max_steps is not None and self.itr >= self.max_steps:
                        self.epoch = self.num_epochs
                        break
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    self.optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss_dict = self.forward_backward(data_list)
                    if not _finite(loss_dict):
                        raise FloatingPointError("nonfinite BF16 loss")
                    gradient = self._gradient_forensics()
                    if not gradient["finite"]:
                        raise FloatingPointError("nonfinite BF16 gradient: {}".format(gradient["first_nonfinite"]))
                    if self.clip_grad_norm:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.clip_grad_norm, error_if_nonfinite=True
                        )
                    self.optimizer.step()
                    if not all(bool(torch.isfinite(parameter).all()) for parameter in self.model.parameters()):
                        raise FloatingPointError("nonfinite parameter after optimizer.step")
                    state_finite = self._check_optimizer_state()
                    if not state_finite["finite"]:
                        raise FloatingPointError("nonfinite optimizer state")
                    self.scheduler.step()
                    self.itr += 1
                    self._ema_update()
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    memory = (torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
                    record = self._diagnostics(data_list, loss_dict, gradient, elapsed, memory)
                    self._write_jsonl(record)
                    for key, value in loss_dict.items():
                        if key not in self.loss_meters:
                            self.loss_meters[key] = AverageMeter()
                        self.loss_meters[key].update(value.detach())
                    self.timer.update(elapsed)
                    if self.itr == 1 or self.itr % self.log_interval == 0:
                        self.log()
                self.epoch += 1
                self.checkpoint()
                self._write_jsonl({"event": "checkpoint", "epoch": int(self.epoch), "iteration": int(self.itr)})
            self._diagnostic_summary.update({"status": "completed", "steps": int(self.itr), "epoch": int(self.epoch)})
            _write_json(self.run_root / "training_summary.json", self._diagnostic_summary)
            print0("Formal BF16 training completed: {}".format(self.formal_label))
        except Exception as exc:
            self._crash(exc)
            self._diagnostic_summary.update({"status": "failed", "steps": int(self.itr), "epoch": int(self.epoch), "exception": repr(exc)})
            _write_json(self.run_root / "training_summary.json", self._diagnostic_summary)
            raise
        finally:
            if hasattr(self, "_capture_handle"):
                self._capture_handle.remove()
            if hasattr(self, "_metrics_handle"):
                self._metrics_handle.close()


class FormalEvaluator(EvaluatorAuxiliary):
    """Official evaluator under the same BF16/global-encoder precision policy."""

    @staticmethod
    def _load_model_state(model, state_dict):
        """Match the checkpoint compatibility path used by TrainerOriginal."""
        load_compatible = getattr(model, "load_compatible_state_dict", None)
        if load_compatible is None:
            return model.load_state_dict(state_dict)
        return load_compatible(state_dict)

    def __init__(self, opt, precision_policy: str = "p2"):
        super().__init__(opt)
        _batched_nms = self.batched_nms
        self.batched_nms = lambda segs, scores: _batched_nms(
            segs.float(), scores.float()
        )
        model_ref = _unwrap(self.model, bool(opt.get("_distributed", False)))
        self.fp32_boundary_modules = install_precision_boundaries(
            model_ref, precision_policy
        )
        self.fp32_loss_modules = []
        for _loss_name in ("query_boundary_importance_loss", "boundary_supervision_loss", "allocation_supervision_loss", "ranking_allocation_loss"):
            _loss_module = getattr(self, _loss_name, None)
            if _loss_module is None:
                continue
            _original_loss_forward = _loss_module.forward
            def _forward_fp32(*args, _original=_original_loss_forward, **kwargs):
                with torch.autocast(device_type="cuda", enabled=False):
                    args32 = tuple(_cast_float(arg) for arg in args)
                    kwargs32 = {key: _cast_float(value) for key, value in kwargs.items()}
                    return _original(*args32, **kwargs32)
            _loss_module.forward = _forward_fp32
            self.fp32_loss_modules.append(_loss_name)

    def predict(self, data):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return super().predict(data)


def _prepare_run(label: str, seed: int, root: Path):
    opt = load_opt(str(CONFIGS[label]), is_training=True)
    opt["seed"] = int(seed)
    opt["_root"] = str(root)
    opt["_resume"] = False
    opt["_distributed"] = False
    opt["_world_size"] = 1
    root.mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(exist_ok=True)
    (root / "states").mkdir(exist_ok=True)
    shutil.copyfile(CONFIGS[label], root / "config_source.yaml")
    (root / "opt.yaml").write_text(yaml.dump(opt, sort_keys=False))
    manifest = {
        "label": label,
        "seed": seed,
        "config_source": str(CONFIGS[label]),
        "config_snapshot": str(root / "opt.yaml"),
        "precision": "bf16",
        "grad_scaler": False,
        "cuda_device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "source": source_fingerprint(),
    }
    _write_json(root / "run_manifest.json", manifest)
    return opt


def run_one(
    label: str,
    seed: int,
    root: Path,
    max_steps: Optional[int] = None,
    evaluate: bool = True,
    precision_policy: str = "p2",
):
    summary_path = root / "training_summary.json"
    checkpoint_path = root / "models" / "last.pth"
    training_completed = False
    if max_steps is None and summary_path.exists() and checkpoint_path.exists():
        summary = json.loads(summary_path.read_text())
        training_completed = (
            summary.get("status") == "completed"
            and summary.get("precision_policy", "p2") == precision_policy
        )

    if training_completed:
        print0("Reusing completed {} seed {} training checkpoint".format(label, seed))
    else:
        opt = _prepare_run(label, seed, root)
        trainer = FormalTrainer(
            opt,
            label,
            root,
            max_steps=max_steps,
            precision_policy=precision_policy,
        )
        trainer.run()

    if evaluate and max_steps is None:
        evaluation_path = root / "evaluation_summary.json"
        if evaluation_path.exists():
            evaluation = json.loads(evaluation_path.read_text())
            print0("Reusing completed {} seed {} evaluation".format(label, seed))
            return evaluation.get("metrics", {})
        eval_opt = load_opt(str(root / "opt.yaml"), is_training=False)
        eval_opt["_root"] = str(root)
        eval_opt["_ckpt"] = "last"
        evaluator = FormalEvaluator(eval_opt, precision_policy=precision_policy)
        evaluator.run()
        prediction_path = root / "predictions_last.json"
        metrics = {}
        if prediction_path.exists():
            payload = json.loads(prediction_path.read_text())
            metrics = payload.get("summary", {}).get("overall_recall_at_iou", {})
        _write_json(evaluation_path, {
            "precision": "bf16",
            "precision_policy": precision_policy,
            "checkpoint": "last",
            "checkpoint_selection_policy": "last (identical for A-D)",
            "metrics": metrics,
        })
        return metrics
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-root", default="experiments/formal_ablation")
    parser.add_argument("--only", default="A,B,C,D", help="comma-separated configurations")
    parser.add_argument("--max-steps", type=int, default=None, help="debug/sanity limit; omit for formal run")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--config-diff", default="experiment_config_diff.txt")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for formal BF16 ablation")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("CUDA BF16 is not supported on this device")

    diff_path = ROOT / args.config_diff
    generate_config_diff(diff_path)
    print0("Wrote {}".format(diff_path))
    labels = [item.strip().upper() for item in args.only.split(",") if item.strip()]
    unknown = sorted(set(labels) - set(CONFIGS))
    if unknown:
        raise SystemExit("unknown configurations: {}".format(unknown))
    output_root = ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    common_manifest = {
        "seed": args.seed,
        "precision": "bf16",
        "grad_scaler": False,
        "configs": labels,
        "formal_training_gate": "open_for_A_D_only",
        "long_training_started": args.max_steps is None,
        "source": source_fingerprint(),
        "config_diff": str(diff_path),
    }
    _write_json(output_root / "manifest.json", common_manifest)
    all_metrics = {}
    for label in labels:
        root = output_root / "seed_{}".format(args.seed) / label
        print0("Starting {} seed {}".format(label, args.seed))
        all_metrics[label] = run_one(label, args.seed, root, max_steps=args.max_steps, evaluate=not args.no_eval)
    _write_json(output_root / "seed_{}_results.json".format(args.seed), {
        "seed": args.seed,
        "metrics": all_metrics,
        "status": "completed" if args.max_steps is None else "sanity_completed",
        "formal_training_gate": "open_for_A_D_only",
        "long_training_started": args.max_steps is None,
    })


if __name__ == "__main__":
    main()

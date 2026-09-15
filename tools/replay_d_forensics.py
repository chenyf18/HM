#!/usr/bin/env python3
"""Replay D from the last normal checkpoint without mutating the source run.

This tool intentionally keeps all recovery state in memory. It advances the
same seeded DataLoader sampler to epoch 6, then runs a bounded number of
optimization steps under P1/P2/P3 precision boundaries while recording compact
batch, activation, gradient, and optimizer diagnostics.
"""

from __future__ import annotations

import argparse
from functools import partial
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hydra"))

from libs import load_opt
from libs.data.data_utils import trivial_batch_collator, worker_init_reset_seed
from libs.worker import TrainerAuxiliary
from tools.run_formal_ablation import install_precision_boundaries


def finite_tree(value: Any) -> bool:
    if torch.is_tensor(value):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_tree(v) for v in value)
    return True


def cast_float(value: Any):
    if torch.is_tensor(value):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, tuple):
        return tuple(cast_float(v) for v in value)
    if isinstance(value, list):
        return [cast_float(v) for v in value]
    if isinstance(value, dict):
        return {k: cast_float(v) for k, v in value.items()}
    return value


def install_loss_boundaries(trainer):
    installed = []
    for name in ("query_boundary_importance_loss", "boundary_supervision_loss"):
        module = getattr(trainer, name, None)
        if module is None:
            continue
        original = module.forward

        def forward_fp32(*args, _original=original, **kwargs):
            with torch.autocast(device_type="cuda", enabled=False):
                return _original(
                    *(cast_float(arg) for arg in args),
                    **{key: cast_float(value) for key, value in kwargs.items()},
                )

        module.forward = forward_fp32
        installed.append(name)
    return installed


def stats(value: torch.Tensor | None) -> dict[str, Any]:
    if value is None:
        return {"count": 0, "nonfinite": 0}
    t = value.detach()
    record = {"shape": list(t.shape), "dtype": str(t.dtype), "device": str(t.device)}
    if not (t.is_floating_point() or t.is_complex()):
        record["finite_ratio"] = 1.0
        return record
    tf = t.float().reshape(-1)
    finite = torch.isfinite(tf)
    record["count"] = int(tf.numel())
    record["nonfinite"] = int((~finite).sum().item())
    record["finite_ratio"] = float(finite.float().mean().item()) if tf.numel() else 1.0
    vals = tf[finite]
    if vals.numel():
        record.update({
            "min": float(vals.min().item()),
            "max": float(vals.max().item()),
            "mean": float(vals.mean().item()),
            "std": float(vals.std(unbiased=False).item()),
            "abs_max": float(vals.abs().max().item()),
            "l2": float(torch.linalg.vector_norm(vals).item()),
        })
    return record


def masked_stats(value: torch.Tensor, mask: torch.Tensor | None) -> dict[str, Any]:
    if mask is None:
        return stats(value)
    v = value.detach()
    m = mask.to(device=v.device, dtype=torch.bool)
    if v.ndim == m.ndim + 1 and v.shape[:m.ndim] == m.shape:
        v = v[m]
    elif v.ndim == m.ndim + 1 and v.shape[0] == m.shape[0] and v.shape[-1] == m.shape[-1]:
        v = v.transpose(1, 2)[m]
    elif tuple(v.shape) != tuple(m.shape):
        return stats(value)
    else:
        v = v[m]
    return stats(v)


def batch_metadata(data_list, dataset, detailed=False):
    result = []
    for item in data_list:
        text_ids = list(item.get("text_ids", ()))
        record = {
            "vid_id": item.get("vid_id"),
            "text_ids": text_ids,
            "vid_shape": list(item["vid"].shape),
            "text_shapes": [list(t.shape) for t in item["text"]]
            if isinstance(item.get("text"), (tuple, list)) else list(item["text"].shape),
            "target_grid": item["target"].detach().cpu().tolist(),
        }
        if detailed:
            record.update({
                "sentences": list(item.get("sentences", ())),
                "vid_stats": stats(item["vid"]),
                "gt_segment_seconds": [
                    np.asarray(dataset.text_dict[text_id]["segment"]).tolist()
                    for text_id in text_ids
                ],
                "duration": float(item["duration"]),
                "fps": float(item["fps"]),
            })
        result.append(record)
    return result


def optimizer_target_state(trainer):
    names = dict(trainer.model.named_parameters())
    target = names["vid_net.branch.2.global_encoder.in_proj.weight"]
    state = trainer.optimizer.state[target]
    result = {"parameter": "vid_net.branch.2.global_encoder.in_proj.weight"}
    for key, value in state.items():
        result[key] = stats(value) if torch.is_tensor(value) else value
    return result


def temporal_metadata(trainer, capture):
    model_ref = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    geometries = getattr(model_ref.vid_net, "last_temporal_metadata", None)
    if geometries is None:
        return None
    masks = capture.get("sequence_masks")
    anchors = capture.get("anchor_masks")
    levels = []
    for i, geometry in enumerate(geometries):
        mask = masks[i] if masks is not None and i < len(masks) else None
        valid = mask[:, 0] if mask is not None and mask.ndim == 3 else mask
        g = geometry.detach().float()
        row = {
            "level": i,
            "geometry": masked_stats(g, valid),
            "span": masked_stats(g[..., 3], valid),
            "regression_scale": masked_stats(g[..., 4], valid),
        }
        if valid is not None:
            row["input_tokens"] = [int(x) for x in valid.sum(-1).detach().cpu().tolist()]
        if anchors is not None and i < len(anchors):
            am = anchors[i]
            am = am[:, 0] if am.ndim == 3 else am
            row["anchor_tokens"] = [int(x) for x in am.sum(-1).detach().cpu().tolist()]
        levels.append(row)
    return levels


class FixedOrderSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def replay_epoch_order(trainer, epochs: int):
    """Recreate the persistent-worker RandomSampler order for one epoch."""
    generator = trainer.dataloader.generator
    sample_count = len(trainer.dataset)
    torch.empty((), dtype=torch.int64).random_(generator=generator)
    order = None
    for _ in range(epochs + 1):
        order = torch.randperm(sample_count, generator=generator).tolist()
        # RandomSampler also evaluates the zero-length remainder permutation.
        torch.randperm(sample_count, generator=generator)
    return order


def install_replay_loader(trainer, order):
    num_workers = int(trainer.opt["train"]["num_workers"])
    trainer.dataloader = DataLoader(
        trainer.dataset,
        batch_size=trainer.batch_size,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator,
        worker_init_fn=partial(worker_init_reset_seed, num_workers, 0),
        sampler=FixedOrderSampler(order),
        shuffle=False,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    return iter(trainer.dataloader)


def load_trainer(source_root: Path, work_root: Path, policy: str):
    opt = load_opt(str(source_root / "opt.yaml"), is_training=True)
    opt["_root"] = str(work_root)
    opt["_resume"] = False
    opt["_distributed"] = False
    opt["_world_size"] = 1
    work_root.mkdir(parents=True, exist_ok=True)
    trainer = TrainerAuxiliary(opt)
    model_ckpt = torch.load(source_root / "models/last.pth", map_location="cpu", weights_only=False)
    state_ckpt = torch.load(source_root / "states/last.pth", map_location="cpu", weights_only=False)
    trainer._load_model_state(trainer.model, model_ckpt["model"])
    trainer._load_model_state(trainer.model_ema, model_ckpt["model_ema"])
    trainer.optimizer.load_state_dict(state_ckpt["optimizer"])
    trainer.scheduler.load_state_dict(state_ckpt["scheduler"])
    trainer.epoch = int(state_ckpt["epoch"])
    trainer.itr = int(state_ckpt["itr"])
    trainer.model.train()
    trainer.model_ema.eval()
    installed = install_precision_boundaries(trainer.model, policy)
    loss_boundaries = install_loss_boundaries(trainer)
    return trainer, installed, loss_boundaries


def install_hooks(trainer, current):
    handles = []
    model_ref = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    capture = {}

    def pre(name):
        def hook(_module, inputs):
            if current.get("detail") and inputs and torch.is_tensor(inputs[0]):
                current["modules"].setdefault(name, {})["input"] = stats(inputs[0])
        return hook

    def post(name):
        def hook(_module, _inputs, output):
            if not current.get("detail"):
                return
            current["modules"].setdefault(name, {})["output"] = stats(output)
            if torch.is_tensor(output) and output.requires_grad:
                def output_grad(grad, module_name=name):
                    current["backward_events"].append({
                        "order": len(current["backward_events"]),
                        "module": module_name + ".output",
                        "gradient": stats(grad),
                    })
                    return grad
                output.register_hook(output_grad)
        return hook

    for i in range(4):
        prefix = "vid_net.branch.{}".format(i)
        encoder = dict(model_ref.named_modules())[prefix + ".global_encoder"]
        handles.extend([encoder.register_forward_pre_hook(pre(prefix + ".global_encoder")),
                        encoder.register_forward_hook(post(prefix + ".global_encoder"))])
        norm_name = prefix + ".norm_global"
        if norm_name in dict(model_ref.named_modules()):
            norm = dict(model_ref.named_modules())[norm_name]
            handles.extend([norm.register_forward_pre_hook(pre(norm_name)),
                            norm.register_forward_hook(post(norm_name))])
        proj_name = prefix + ".global_encoder.in_proj"
        if proj_name in dict(model_ref.named_modules()):
            proj = dict(model_ref.named_modules())[proj_name]
            handles.extend([proj.register_forward_pre_hook(pre(proj_name)),
                            proj.register_forward_hook(post(proj_name))])
        parameter = dict(model_ref.named_parameters())[prefix + ".global_encoder.in_proj.weight"]
        def parameter_grad(grad, module_name=prefix + ".global_encoder.in_proj.weight"):
            if current.get("detail"):
                current["backward_events"].append({
                    "order": len(current["backward_events"]),
                    "module": module_name,
                    "gradient": stats(grad),
                })
            return grad
        handles.append(parameter.register_hook(parameter_grad))

    def model_post(_module, _inputs, output):
        if not current.get("detail"):
            return
        if not isinstance(output, (tuple, list)) or len(output) < 8:
            return
        capture["sequence_masks"] = tuple(x.detach() for x in output[5])
        capture["anchor_masks"] = tuple(x.detach() for x in output[7])
        capture["fpn_masks"] = tuple(x.detach() for x in output[3])
        extras = output[8:]
        if len(extras) >= 2:
            assignments = extras[-1]
            if isinstance(assignments, (tuple, list)):
                capture["assignments"] = tuple(x.detach() for x in assignments)
    handles.append(model_ref.register_forward_hook(model_post))
    return capture, handles


def gradients(trainer, detailed):
    parameters = [p for p in trainer.model.parameters() if p.grad is not None]
    target_name = "vid_net.branch.2.global_encoder.in_proj.weight"
    named_parameters = dict(trainer.model.named_parameters())
    target_grad = named_parameters[target_name].grad
    raw_target_stats = stats(target_grad) if target_grad is not None else None
    selected = {target_name: raw_target_stats} if detailed and raw_target_stats else {}
    max_norm = trainer.clip_grad_norm
    max_norm = float("inf") if max_norm is None else float(max_norm)
    try:
        raw_norm = nn.utils.clip_grad_norm_(
            parameters, max_norm, error_if_nonfinite=True, foreach=True
        )
        return {
            "finite": True,
            "bad": [],
            "norm": float(raw_norm.item()),
            "selected": selected,
            "clipped": trainer.clip_grad_norm is not None,
        }
    except RuntimeError:
        bad = []
        for name, parameter in trainer.model.named_parameters():
            if parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item()):
                continue
            bad.append(name)
            selected[name] = stats(parameter.grad)
        return {
            "finite": False,
            "bad": bad,
            "norm": float("inf"),
            "selected": selected,
            "clipped": False,
        }


def tensor_collection_finite(tensors):
    groups = {}
    cpu_flags = []
    for tensor in tensors:
        if not torch.is_tensor(tensor) or not tensor.is_floating_point():
            continue
        if tensor.is_cuda:
            groups.setdefault((tensor.device, tensor.dtype), []).append(tensor)
        else:
            cpu_flags.append(torch.isfinite(tensor).all())
    if cpu_flags and not bool(torch.stack(cpu_flags).all().item()):
        return False
    for (device, _dtype), group in groups.items():
        found_inf = torch.zeros((), device=device)
        inv_scale = torch.ones((), device=device)
        torch._amp_foreach_non_finite_check_and_unscale_(
            group, found_inf, inv_scale
        )
        if bool(found_inf.item()):
            return False
    return True



def run(policy: str, source_root: Path, output: Path, steps: int,
        skip_epochs: int, start_offset: int, detail_start: int,
        detail_end: int, compute_precision: str):
    work_root = output.parent / (output.stem + "_work")
    trainer, installed, loss_boundaries = load_trainer(source_root, work_root, policy)
    current = {}
    capture, handles = install_hooks(trainer, current)
    epoch_order = replay_epoch_order(trainer, skip_epochs)
    if start_offset < 0 or start_offset >= len(epoch_order):
        raise ValueError("start_offset outside epoch: {}".format(start_offset))
    loader_iter = install_replay_loader(trainer, epoch_order[start_offset:])
    records = []
    progress_path = output.with_suffix(output.suffix + ".jsonl")
    progress_handle = progress_path.open("w")
    initial_optimizer_state = optimizer_target_state(trainer)

    def append_record(record):
        records.append(record)
        progress_handle.write(json.dumps(record, sort_keys=True) + "\n")
        progress_handle.flush()
        if len(records) == 1 or len(records) % 100 == 0:
            print("replay {} step {} status={} grad_norm={:.6g}".format(
                policy, len(records), record.get("status"),
                float(record.get("gradient_norm", float("nan"))),
            ), flush=True)

    try:
        for attempt in range(steps):
            batch_offset = start_offset + attempt
            current.clear()
            capture.clear()
            current["modules"] = {}
            current["backward_events"] = []
            current["capture"] = capture
            current["detail"] = detail_start <= batch_offset < detail_end
            current["itr_before"] = int(trainer.itr)
            data_list = next(loader_iter)
            current["batch"] = batch_metadata(
                data_list, trainer.dataset, current["detail"]
            )
            t0 = time.perf_counter()
            record = {
                "event": "step_attempt",
                "policy": policy,
                "compute_precision": compute_precision,
                "batch_offset": batch_offset,
                "logical_source_step": 31992 + batch_offset,
                "iteration_before": int(trainer.itr),
                "epoch": int(trainer.epoch),
                "batch": current["batch"],
            }
            trainer.optimizer.zero_grad(set_to_none=True)
            try:
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=(compute_precision == "bf16"),
                ):
                    losses = trainer.forward_backward(data_list)
                record["forward_loss_finite"] = finite_tree(losses)
                record["losses"] = {k: float(v.detach().item()) for k, v in losses.items()}
                record["forward_backward_sec"] = time.perf_counter() - t0
                phase_start = time.perf_counter()
                record["temporal"] = (
                    temporal_metadata(trainer, capture) if current["detail"] else None
                )
                record["forward_modules"] = (
                    current["modules"] if current["detail"] else None
                )
                grad_info = gradients(trainer, current["detail"])
                record["gradient_norm"] = grad_info["norm"]
                record["gradient_finite"] = grad_info["finite"]
                record["gradient_parameters"] = grad_info["selected"]
                record["gradient_check_clip_sec"] = time.perf_counter() - phase_start
                record["backward_events"] = current["backward_events"]
                record["optimizer_state_before"] = (
                    optimizer_target_state(trainer) if current["detail"] else None
                )
                if not grad_info["finite"]:
                    record["status"] = "nonfinite_gradient"
                    append_record(record)
                    break
                trainer.optimizer.step()
                trainer.scheduler.step()
                trainer.itr += 1
                trainer._ema_update()
                record["optimizer_ema_sec"] = time.perf_counter() - phase_start
                record["parameter_finite_after"] = tensor_collection_finite(
                    trainer.model.parameters()
                )
                record["optimizer_state_finite_after"] = tensor_collection_finite(
                    value
                    for state in trainer.optimizer.state.values()
                    for value in state.values()
                )
                record["optimizer_state_after"] = (
                    optimizer_target_state(trainer) if current["detail"] else None
                )
                if not record["parameter_finite_after"]:
                    raise FloatingPointError("nonfinite parameter after optimizer.step")
                if not record["optimizer_state_finite_after"]:
                    raise FloatingPointError("nonfinite optimizer state after optimizer.step")
                record["status"] = "ok"
            except Exception as exc:
                record["status"] = "exception"
                record["exception"] = repr(exc)
                record["forward_modules"] = (
                    current.get("modules", {}) if current.get("detail") else None
                )
                record["backward_events"] = current.get("backward_events", [])
                record["temporal"] = (
                    temporal_metadata(trainer, capture)
                    if current.get("detail") else None
                )
                append_record(record)
                break
            record["elapsed_sec"] = time.perf_counter() - t0
            append_record(record)
    finally:
        progress_handle.close()
        for handle in handles:
            handle.remove()
        report = {
            "source_root": str(source_root),
            "policy": policy,
            "compute_precision": compute_precision,
            "skip_epochs": skip_epochs,
            "start_offset": start_offset,
            "checkpoint_iteration": 31992,
            "installed_fp32_modules": installed,
            "installed_fp32_loss_modules": loss_boundaries,
            "detail_start": detail_start,
            "detail_end": detail_end,
            "initial_optimizer_state": initial_optimizer_state,
            "progress_jsonl": str(progress_path),
            "steps_requested": steps,
            "steps_recorded": len(records),
            "records": records,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True))
        del trainer
        torch.cuda.empty_cache()
    print(json.dumps({
        "policy": policy,
        "steps_recorded": len(records),
        "last_status": records[-1]["status"] if records else None,
        "last_iteration_before": records[-1]["iteration_before"] if records else None,
        "output": str(output),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="experiments/formal_ablation/seed_1/D")
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy", choices=("p1", "p2", "p3"), required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--skip-epochs", type=int, default=6)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--detail-start", type=int, default=0)
    parser.add_argument("--detail-end", type=int, default=0)
    parser.add_argument("--compute-precision", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args()
    run(
        args.policy,
        ROOT / args.source_root,
        ROOT / args.output,
        args.steps,
        args.skip_epochs,
        args.start_offset,
        args.detail_start,
        args.detail_end,
        args.compute_precision,
    )


if __name__ == "__main__":
    main()


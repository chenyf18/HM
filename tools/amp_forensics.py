#!/usr/bin/env python3
"""One-step precision and gradient forensic for the research validator.

This is intentionally separate from the model and validator implementation. It
constructs the same real Ego4D batch and loss graph, records forward outputs,
gradient tensors, optimizer state, and backward hook order, and never writes a
checkpoint or changes model source code.
"""

import argparse
import copy
import json
import math
import os
import sys
import types

import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "hydra"))

from libs import load_opt
from libs.modeling import MultiScaleMaskedContrastive, make_optimizer
from libs.modeling.model import make_models_net
import tools.research_validate_adaptive as rv


def _finite_tree(value):
    if torch.is_tensor(value):
        if value.is_floating_point() or value.is_complex():
            return bool(torch.isfinite(value).all())
        return True
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite_tree(item) for item in value)
    return True


def _tensor_summary(value):
    if not torch.is_tensor(value):
        return {"type": type(value).__name__, "finite": True}
    record = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "finite": True,
    }
    if value.is_floating_point() or value.is_complex():
        finite = torch.isfinite(value)
        record["finite"] = bool(finite.all())
        if finite.any():
            vals = value.detach().float()[finite]
            record.update({
                "min": float(vals.min()),
                "max": float(vals.max()),
                "mean_abs": float(vals.abs().mean()),
            })
        record["finite_ratio"] = float(finite.float().mean())
    return record


def _tree_summary(value):
    if torch.is_tensor(value):
        return _tensor_summary(value)
    if isinstance(value, dict):
        return {str(k): _tree_summary(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_tree_summary(v) for v in value]
    return {"type": type(value).__name__, "finite": True}


def _cast_tree(value, dtype):
    if torch.is_tensor(value):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, tuple):
        return tuple(_cast_tree(v, dtype) for v in value)
    if isinstance(value, list):
        return [_cast_tree(v, dtype) for v in value]
    if isinstance(value, dict):
        return {k: _cast_tree(v, dtype) for k, v in value.items()}
    return value


class _FP32Encoder(nn.Module):
    """Diagnostic wrapper that runs one existing encoder outside autocast."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        with torch.autocast(device_type="cuda", enabled=False):
            args = tuple(
                arg.float() if torch.is_tensor(arg) and arg.is_floating_point()
                else arg
                for arg in args
            )
            kwargs = {
                key: value.float()
                if torch.is_tensor(value) and value.is_floating_point()
                else value
                for key, value in kwargs.items()
            }
            output = self.module(*args, **kwargs)
        return _cast_tree(output, None)


def _force_global_fp32(model):
    names = []
    for parent_name, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            if child_name != "global_encoder":
                continue
            setattr(parent, child_name, _FP32Encoder(child))
            names.append((parent_name + "." + child_name).strip("."))
    return names


def _add_acc_optimizer_params(optimizer, acc_module, opt):
    if acc_module is None:
        return
    decay, no_decay = [], []
    for name, parameter in acc_module.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if name.endswith("bias") or parameter.ndim <= 1 else decay).append(parameter)
    optimizer_opt = opt["train"]["optimizer"]
    if decay:
        optimizer.add_param_group({
            "params": decay,
            "weight_decay": optimizer_opt["weight_decay"],
            "lr": optimizer_opt["lr"],
        })
    if no_decay:
        optimizer.add_param_group({
            "params": no_decay,
            "weight_decay": 0.0,
            "lr": optimizer_opt["lr"],
        })


def _make_acc(opt, adaptive):
    acc_opt = opt["train"].get("loss_aux", {}).get("ds_contrast", {})
    if not acc_opt.get("enable", False):
        return None
    expected = "assignment_acc" if adaptive else "legacy_acc"
    if acc_opt.get("acc_mode", "legacy_acc") != expected:
        raise ValueError("unexpected ACC mode")
    return MultiScaleMaskedContrastive(
        acc_opt, opt["model"]["vid_net"]["embd_dim"]
    ).cuda().train()


def _register_forensics(model):
    module_outputs = {}
    backward_events = []
    handles = []

    def forward_hook(name):
        def hook(_module, _inputs, output):
            summary = _tree_summary(output)
            module_outputs[name] = summary

            def tensor_hook(grad):
                backward_events.append({
                    "order": len(backward_events),
                    "module": name,
                    "gradient": _tensor_summary(grad),
                })
                return grad

            def attach(value):
                if torch.is_tensor(value) and value.requires_grad:
                    value.register_hook(tensor_hook)
                elif isinstance(value, (tuple, list)):
                    for item in value:
                        attach(item)
                elif isinstance(value, dict):
                    for item in value.values():
                        attach(item)

            attach(output)
        return hook

    # Keep this broad enough to identify the first bad stage while recording
    # only forward values, not copies of the full activation tensors.
    for name, module in model.named_modules():
        if name:
            handles.append(module.register_forward_hook(forward_hook(name)))
    return module_outputs, backward_events, handles


def _gradient_forensics(owner):
    records = []
    first_nonfinite = None
    for name, parameter in owner.named_parameters():
        if parameter.grad is None:
            records.append({"name": name, "requires_grad": bool(parameter.requires_grad), "grad": None})
            continue
        grad = parameter.grad.detach()
        summary = _tensor_summary(grad)
        record = {"name": name, "requires_grad": bool(parameter.requires_grad), "grad": summary}
        records.append(record)
        if first_nonfinite is None and not summary["finite"]:
            first_nonfinite = record
    return {
        "first_nonfinite_parameter": first_nonfinite,
        "nonfinite_parameters": [
            record["name"] for record in records
            if isinstance(record.get("grad"), dict) and record["grad"].get("finite") is False
        ],
        "missing_gradients": [
            record["name"] for record in records
            if record.get("requires_grad") and record.get("grad") is None
        ],
        "parameters": records,
    }


def _optimizer_state_finite(optimizer):
    bad = []
    for index, state in enumerate(optimizer.state.values()):
        for key, value in state.items():
            if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all():
                bad.append({"state_index": index, "key": key})
    return {"finite": not bad, "nonfinite": bad}


def run_variant(opt, batch, label, autocast_dtype=None, scaler_enabled=False, force_fp32=False, seed=20260813):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = make_models_net(opt).cuda().train()
    forced_names = _force_global_fp32(model) if force_fp32 else []
    adaptive = bool(opt["model"]["vid_net"].get("adaptive_anchor", False))
    importance = bool(opt["model"]["vid_net"].get("query_boundary_importance", False) or adaptive)
    acc_module = _make_acc(opt, adaptive)
    optimizer_owner = model
    optimizer = make_optimizer(model, opt["train"]["optimizer"])
    if acc_module is not None:
        optimizer_owner = nn.Module()
        optimizer_owner.add_module("model", model)
        optimizer_owner.add_module("acc_module", acc_module)
        _add_acc_optimizer_params(optimizer, acc_module, opt)
    scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    module_outputs, backward_events, handles = _register_forensics(model)
    optimizer.zero_grad(set_to_none=True)
    result = {
        "variant": label,
        "autocast_dtype": str(autocast_dtype) if autocast_dtype else None,
        "grad_scaler_enabled": scaler_enabled,
        "forced_global_fp32": forced_names,
    }
    try:
        autocast_enabled = autocast_dtype is not None
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_enabled):
            outputs = rv.model_forward(opt, model, batch, importance=importance, assignments=adaptive)
            points = rv.build_points(opt, model, outputs["masks"])
            cls_loss, reg_loss, positive_count = rv.grounding_losses(opt, points, outputs)
            zero = outputs["logits"][0].float().sum() * 0.0
            acc_loss = zero
            if acc_module is not None:
                acc_loss = acc_module(
                    outputs["fpn"], outputs["sequence_masks"],
                    outputs["anchors"], outputs["anchor_masks"],
                    assignment_matrices=(outputs["assignments"] if adaptive else None),
                ) / float(opt["train"].get("loss_norm", 1.0))
            total = cls_loss + float(opt["train"].get("loss_weight", 1.0)) * reg_loss + acc_loss
        with torch.autocast(device_type="cuda", enabled=False):
            auxiliary = rv.auxiliary_losses(opt, outputs, points)
        if auxiliary["importance"] is not None:
            total = total + auxiliary["importance_weight"] * auxiliary["importance"]
        if auxiliary["boundary"] is not None:
            total = total + auxiliary["boundary_weight"] * auxiliary["boundary"]
        result.update({
            "forward_finite": bool(rv.tensor_tree_finite(outputs)),
            "loss_finite": bool(rv.tensor_tree_finite((cls_loss, reg_loss, acc_loss, auxiliary["importance"], auxiliary["boundary"], total))),
            "losses": {"cls": float(cls_loss.detach()), "reg": float(reg_loss.detach()), "acc": float(acc_loss.detach()), "total": float(total.detach())},
            "positive_count": int(positive_count),
            "output_summary": {"logits": _tree_summary(outputs["logits"]), "offsets": _tree_summary(outputs["offsets"]), "fpn": _tree_summary(outputs["fpn"])},
        })
        scaled = scaler.scale(total) if scaler_enabled else total
        result["scaled_loss"] = _tensor_summary(scaled.detach())
        scaled.backward()
        result["backward_events"] = backward_events[:]
        result["scaled_gradient_forensics"] = _gradient_forensics(optimizer_owner)
        if scaler_enabled:
            scaler.unscale_(optimizer)
        result["unscaled_gradient_forensics"] = _gradient_forensics(optimizer_owner)
        gradient_finite = not result["unscaled_gradient_forensics"]["nonfinite_parameters"]
        result["backward_finite"] = bool(gradient_finite)
        scale_before = float(scaler.get_scale())
        before = [parameter.detach().clone() for parameter in optimizer_owner.parameters()]
        if scaler_enabled:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        result["scaler_before"] = scale_before
        result["scaler_after"] = float(scaler.get_scale())
        result["parameter_delta_norm"] = float(torch.sqrt(sum((after.detach().float() - before_i.float()).square().sum() for after, before_i in zip(optimizer_owner.parameters(), before))))
        result["optimizer_state"] = _optimizer_state_finite(optimizer)
        result["optimizer_step_finite"] = bool(all(torch.isfinite(parameter).all() for parameter in optimizer_owner.parameters()) and result["optimizer_state"]["finite"])
    except Exception as exc:
        result["exception"] = repr(exc)
        result.setdefault("backward_events", backward_events[:])
        result.setdefault("module_outputs", module_outputs)
    finally:
        result["module_outputs"] = module_outputs
        result["nonfinite_output_modules"] = [name for name, summary in module_outputs.items() if not _finite_tree(summary)]
        for handle in handles:
            handle.remove()
        del optimizer, model, acc_module
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--output", default="experiments/research_validation/amp_precision_matrix.json")
    parser.add_argument("--only", default=None, help="comma-separated variant labels")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    baseline_path = os.path.join(ROOT, "opts/research_ablation_A_baseline.yaml")
    adaptive_path = os.path.join(ROOT, "opts/research_ablation_D_adaptive_anchor.yaml")
    baseline_opt = load_opt(baseline_path, is_training=True)
    adaptive_opt = load_opt(adaptive_path, is_training=True)
    samples, _ = rv.select_real_samples(baseline_opt, 2, args.seed)
    batch = rv.batchify(samples[:1], baseline_opt)
    matrix = []
    variants = [
        ("baseline_fp32", baseline_opt, None, False, False),
        ("baseline_fp16_gradscaler", baseline_opt, torch.float16, True, False),
        ("baseline_bf16", baseline_opt, torch.bfloat16, False, False),
        ("baseline_fp16_global_fp32", baseline_opt, torch.float16, True, True),
        ("adaptive_fp32", adaptive_opt, None, False, False),
        ("adaptive_fp16_gradscaler", adaptive_opt, torch.float16, True, False),
        ("adaptive_bf16", adaptive_opt, torch.bfloat16, False, False),
        ("adaptive_fp16_global_fp32", adaptive_opt, torch.float16, True, True),
    ]
    selected = set(args.only.split(",")) if args.only else None
    for label, opt, dtype, scaler, force in variants:
        if selected is not None and label not in selected:
            continue
        print("running", label, flush=True)
        matrix.append(run_variant(opt, batch, label, dtype, scaler, force, args.seed))
    report = {
        "device": torch.cuda.get_device_name(),
        "batch_video_ids": batch["video_ids"],
        "batch_query_count": len(batch["queries"]),
        "variants": matrix,
    }
    path = os.path.join(ROOT, args.output)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps({
        "device": report["device"],
        "results": [{
            "variant": item["variant"],
            "forward_finite": item.get("forward_finite"),
            "loss_finite": item.get("loss_finite"),
            "backward_finite": item.get("backward_finite"),
            "optimizer_step_finite": item.get("optimizer_step_finite"),
            "exception": item.get("exception"),
            "first_nonfinite": item.get("unscaled_gradient_forensics", {}).get("first_nonfinite_parameter"),
            "nonfinite_output_modules": item.get("nonfinite_output_modules"),
        } for item in matrix]
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

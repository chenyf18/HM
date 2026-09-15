#!/usr/bin/env python3
"""Exact A vs A-U equivalence audit (directive: find the FIRST divergence).

Both sides start from the SAME trained baseline-A checkpoint. All losses and
gradients are assembled through the real TrainerAuxiliary code paths.
No optimizer step, no long training, eval mode, FP32.

Stages (--stages comma list):
  feature   layer-by-layer capture, first divergent tensor (S4/S5/S6)
  points    legacy PtGenerator vs dynamic metadata, element-wise (S7 Q1-Q3)
  assign    GT positive masks + regression targets (S8/S9)
  loss      per-term loss values incl. AU-noaux / AU-fixedpoint (S10-S13)
  grad      backward once, first diverging parameter module (S12)
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "hydra"), str(ROOT / "tools")]

from libs import load_opt  # noqa: E402
from libs.worker import TrainerAuxiliary  # noqa: E402
from tools import research_validate_adaptive as rva  # noqa: E402

OPT_A = ROOT / "opts/research_ablation_A_baseline.yaml"
OPT_AU = ROOT / "opts/research_allocator_AU_uniform.yaml"
CKPT_A = ROOT / "experiments/formal_ablation/seed_1/A/models/last.pth"
OUTPUT = ROOT / "experiments/allocator_diagnosis/exact_equivalence_audit.json"

HOOK_NAMES = (
    "fusion",
    "text_net",
)


def tensor_report(a, b, rtol=1e-4, atol=1e-5):
    if a.shape != b.shape:
        return {"shape_a": list(a.shape), "shape_b": list(b.shape),
                "shapes_equal": False}
    diff = (a.float() - b.float()).abs()
    denom = b.float().abs().clamp_min(1e-8)
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "rel_error_max": float((diff / denom).max()),
        "allclose": bool(torch.allclose(a.float(), b.float(), rtol=rtol,
                                        atol=atol)),
        "exact_equal": bool(torch.equal(a, b)),
    }


def load_shared_checkpoint(model, ckpt_path):
    payload = torch.load(ckpt_path, map_location="cpu")
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    own = model.state_dict()
    filtered = {k: v for k, v in state.items()
                if k in own and own[k].shape == v.shape}
    skipped = sorted(set(state) - set(filtered))
    missing = sorted(set(own) - set(state))
    model.load_state_dict(filtered, strict=False)
    return {"loaded": len(filtered), "skapped_unexpected": skipped[:12],
            "missing_in_ckpt_random_init_count": len(missing)}


def build_trainers(seed, au_config=None):
    import tempfile

    scratch = Path(tempfile.mkdtemp(prefix="audit_"))
    opt_a = load_opt(str(OPT_A), is_training=True)
    opt_au = load_opt(str(au_config or OPT_AU), is_training=True)
    for opt, name in ((opt_a, "a"), (opt_au, "au")):
        opt["_root"] = str(scratch / name)
        opt["_resume"] = False
        opt["_distributed"] = False
        opt["_world_size"] = 1
        Path(opt["_root"]).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    trainer_a = TrainerAuxiliary(copy.deepcopy(opt_a))
    torch.manual_seed(seed)
    trainer_au = TrainerAuxiliary(copy.deepcopy(opt_au))
    info_a = load_shared_checkpoint(trainer_a.model, CKPT_A)
    info_au = load_shared_checkpoint(trainer_au.model, CKPT_A)
    trainer_a.model.eval()
    trainer_au.model.eval()
    return trainer_a, trainer_au, {"a_load": info_a, "au_load": info_au}


def make_batch(trainer_a, trainer_au):
    samples, indices = rva.select_real_samples(
        trainer_a.opt, 2, 1234567891
    )
    sample = [dict(samples[0])]
    vid, vmask, text, tmask, tsize = trainer_a._batchify(
        [d["vid"] for d in sample], [d["text"] for d in sample]
    )
    vid, vmask, text, tmask, tsize = (
        vid.cuda(), vmask.cuda(), text.cuda(), tmask.cuda(), tsize.cuda())
    div_a = 1 if trainer_a.adaptive_anchor else trainer_a.vid_stride
    div_au = 1 if trainer_au.adaptive_anchor else trainer_au.vid_stride
    targets_a = torch.cat([d["target"] / div_a for d in sample]).cuda()
    targets_au = torch.cat([d["target"] / div_au for d in sample]).cuda()
    return (vid, vmask, text, tmask, tsize), targets_a, targets_au


def run_model(trainer, inputs, want_debug, want_assignments=False):
    vid, vmask, text, tmask, tsize = inputs
    kwargs = {}
    if want_debug:
        kwargs["return_importance_debug"] = True
    if want_assignments:
        kwargs["return_anchor_assignments"] = True
    with torch.no_grad():
        out = trainer.model(vid, vmask, text, tmask, tsize, **kwargs)
    extra = len(out) - 8
    return out[:8], (out[8:] if extra else None)


def stage_feature(trainer_a, trainer_au, inputs):
    captures = {"a": {}, "au": {}}

    def make_recorder(store, prefix, name):
        def fn(_m, _inp, out):
            if isinstance(out, tuple):
                out = next((t for t in out if torch.is_tensor(t)), None)
            if torch.is_tensor(out):
                _ = prefix
                store[name] = out.detach().float().cpu()
        return fn

    models = {"a": trainer_a.model, "au": trainer_au.model}
    leaves = {"norm_global", "norm_global_in", "global_encoder",
              "local_encoder", "ffn", "norm_ffn"}
    handles = []
    branch_names = []
    for side, model in models.items():
        for name, module in model.named_modules():
            include = False
            if name.startswith("vid_net.branch."):
                suffix = name[len("vid_net.branch."):]
                if "." not in suffix:
                    include = True
                    if side == "a":
                        branch_names.append(name)
                else:
                    include = suffix.split(".")[-1] in leaves
            elif name in HOOK_NAMES:
                include = True
            elif side == "au" and name.endswith("adaptive_anchor_allocator"):
                include = True
            if include:
                handles.append(module.register_forward_hook(
                    make_recorder(captures[side], side, name)))

    with torch.no_grad():
        run_model(trainer_a, inputs, want_debug=False)
        run_model(trainer_au, inputs, want_debug=True)
    for handle in handles:
        handle.remove()

    report = {"branches": branch_names[:2], "captures_a": sorted(captures["a"]),
              "captures_au": sorted(captures["au"]), "compared": []}
    first_divergence = None
    order = ["fusion", "text_net"] + [
        "vid_net.branch.{0}".format(i) for i in range(8)
    ]
    # Compare shared scalar-ish captures level by level where shapes match.
    for prefix in order:
        a_keys = [k for k in captures["a"] if k.startswith(prefix)]
        au_keys = [k for k in captures["au"] if k.startswith(prefix)]
        for key in sorted(set(a_keys) | set(au_keys)):
            entry = {"tensor": key}
            if key in captures["a"] and key in captures["au"]:
                entry.update(tensor_report(captures["a"][key],
                                           captures["au"][key]))
                if not entry.get("allclose", False) and first_divergence is None:
                    first_divergence = key
            else:
                entry["only_in"] = ("a" if key in captures["a"] else "au")
            report["compared"].append(entry)
    report["first_divergence_tensor"] = first_divergence
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages",
                        default="feature,points,assign,loss,grad")
    parser.add_argument("--seed", type=int, default=1234567891)
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--au-config", default=str(OPT_AU))
    args = parser.parse_args()
    stages = set(args.stages.split(","))
    device = "cuda"
    report = {"checkpoint": str(CKPT_A), "precision": "fp32-eval"}

    print("[setup] building trainers ...", flush=True)
    trainer_a, trainer_au, load_info = build_trainers(args.seed,
                                                      args.au_config)
    report["checkpoint_loading"] = load_info
    inputs, targets_a, targets_au = make_batch(trainer_a, trainer_au)

    if "feature" in stages:
        print("[stage] feature", flush=True)
        report["feature"] = stage_feature(trainer_a, trainer_au, inputs)

    if "points" in stages or "assign" in stages or "loss" in stages \
            or "grad" in stages:
        print("[stage] forward both models", flush=True)
        out_a, _extras_a = run_model(trainer_a, inputs, want_debug=False)
        out_au, extras_au = run_model(
            trainer_au, inputs, want_debug=True, want_assignments=True)
        debug_au = extras_au[0]
        assign_au = extras_au[1]

    if "assign" in stages:
        print("[stage] assign", flush=True)
        report["assignment_regression"] = stage_assign(
            trainer_a, trainer_au, out_a, out_au, targets_a, targets_au)

    if "loss" in stages:
        print("[stage] loss", flush=True)
        report["losses"] = stage_loss(
            trainer_a, trainer_au, out_a, debug_au, out_au, assign_au,
            targets_a, targets_au)

    if "grad" in stages:
        print("[stage] grad", flush=True)
        report["gradients"] = stage_grad(
            trainer_a, trainer_au, inputs, targets_a, targets_au)

    if "points" in stages:
        print("[stage] points", flush=True)
        fpn_masks_a, fpn_masks_au = out_a[3], out_au[3]
        npts_a = [m.size(-1) for m in fpn_masks_a]
        npts_au = [m.size(-1) for m in fpn_masks_au]
        pts_a = trainer_a.pt_gen(npts_a)
        pts_au = trainer_au.adaptive_pt_gen(
            trainer_au.model.vid_net.last_temporal_metadata, fpn_masks_au)
        levels = []
        for lvl, (pa, pu) in enumerate(zip(pts_a, pts_au)):
            if pu.ndim == 3:
                pu = pu[0]
            n = min(pa.size(0), pu.size(0))
            pa_n, pu_n = pa[:n].float().cpu(), pu[:n].float().cpu()
            center_match = bool(torch.allclose(
                pa_n[:, 0], pu_n[:, 0], atol=1e-3))
            scale_match = bool(torch.allclose(
                pa_n[:, 3], pu_n[:, 3], atol=1e-3))
            levels.append({
                "level": lvl,
                "num_points_a": int(pa.size(0)),
                "num_points_au": int(pu.size(0)),
                "center_max_abs_diff_common_prefix": float(
                    (pa_n[:, 0] - pu_n[:, 0]).abs().max()),
                "center_first_mismatch_index": next((
                    i for i in range(n)
                    if abs(pa_n[i, 0].item() - pu_n[i, 0].item()) > 1e-3
                ), None),
                "scale_max_abs_diff_common_prefix": float(
                    (pa_n[:, 3] - pu_n[:, 3]).abs().max()),
                "reg_range_max_abs_diff": float(
                    (pa_n[:, 1:3] - pu_n[:, 1:3]).abs().max()),
                "Q1_center_equal": center_match,
                "Q2_scale_equal": scale_match,
                "centers_a_first6": [round(float(v), 3)
                                     for v in pa_n[:6, 0]],
                "centers_au_first6": [round(float(v), 3)
                                      for v in pu_n[:6, 0]],
                "scales_a_first6": [round(float(v), 3)
                                    for v in pa_n[:6, 3]],
                "scales_au_first6": [round(float(v), 3)
                                     for v in pu_n[:6, 3]],
            })
        report["points"] = {
            "per_level": levels,
            "Q3_systematic_offset": any(
                not item["Q1_center_equal"] for item in levels),
        }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items()
                      if k in ("first_divergence_note",)}, indent=2))
    print("report:", args.output)




def _points_levels(trainer, out, override=None):
    fpn_masks = out[3]
    if override is not None:
        return override
    npts = [m.size(-1) for m in fpn_masks]
    if trainer.adaptive_anchor:
        return trainer.adaptive_pt_gen(
            trainer.model.vid_net.last_temporal_metadata, fpn_masks)
    return trainer.pt_gen(npts)


def assemble_losses(trainer, out, debug, targets, assignments=None,
                    points_override=None, importance_weight=None,
                    need_grad=False):
    """Replicate worker forward_backward loss assembly exactly."""
    fpn_logits_l, _, fpn_offsets_l, fpn_mask_l, fpn, seq_masks, \
        anchor_fpn, anchor_masks = out[:8]
    fpn_logits = torch.cat(fpn_logits_l, dim=1)
    fpn_offsets = torch.cat(fpn_offsets_l, dim=1)
    fpn_masks = torch.cat(fpn_mask_l, dim=1)
    pts_levels = _points_levels(trainer, out, points_override)
    batched = pts_levels[0].ndim == 3
    pts_cat = (torch.cat(pts_levels, dim=1) if batched
               else torch.cat(pts_levels))
    labels, offsets = trainer._annotate_points(pts_cat, targets)
    npts = [m.size(-1) for m in fpn_mask_l]
    labels_split = labels.split(npts, dim=1)

    pos = torch.logical_and(labels, fpn_masks)
    norm = pos.sum()
    cls_loss = trainer._calc_focal_loss(
        logits=fpn_logits[fpn_masks], labels=labels[fpn_masks]
    ) / trainer.loss_norm
    reg_loss = trainer._calc_iou_loss(
        pred_offsets=fpn_offsets[pos], gt_offsets=offsets[pos]
    ) / trainer.loss_norm

    ds_loss = trainer.ds_contrastive_loss(
        fpn, seq_masks, anchor_fpn, anchor_masks,
        assignment_matrices=assignments,
    ) / trainer.loss_norm

    if batched:
        span_labels = generate_multiscale_gt_masks_from_points(
            pts_levels, targets)
    else:
        span_labels = generate_multiscale_gt_masks(targets, npts)
    span_split = span_labels.split(npts, dim=1)
    span_contrastive = generate_multiscale_gt_masks_contrastive(
        pts_cat, targets, trainer.loss_aux_span_radius).split(npts, dim=1)
    gt_loss = trainer.gt_contrastive_loss(
        fpn, fpn_masks.split(npts, dim=1), labels_split, span_split)

    imp_loss = torch.tensor(0.0, device=cls_loss.device)
    if debug is not None and trainer.query_boundary_importance:
        imp_components = trainer.query_boundary_importance_loss(
            debug, pts_levels, seq_masks, targets, return_components=True)
        imp_loss = imp_components[0] if isinstance(imp_components, tuple) \
            else imp_components
    weight = trainer.query_boundary_importance_weight \
        if importance_weight is None else importance_weight

    total = (cls_loss + trainer.loss_weight * reg_loss
             + trainer.ds_contrastive_weight * ds_loss
             + trainer.gt_contrastive_weight * gt_loss
             + float(weight) * imp_loss)
    return {
        "cls": float(cls_loss), "reg": float(reg_loss),
        "ds_contrast": float(ds_loss), "gt_contrast": float(gt_loss),
        "importance": float(imp_loss), "total": float(total),
        "num_positive": int(pos.sum()), "norm_count": int(norm),
        "_graph": total if need_grad else None,
    }


def stage_assign(trainer_a, trainer_au, out_a, out_au, targets_a, targets_au):
    pts_a = _points_levels(trainer_a, out_a)
    pts_au = _points_levels(trainer_au, out_au)
    cat_a = torch.cat(pts_a)
    cat_au = torch.cat(pts_au, dim=1)[0]
    labels_a, offs_a = trainer_a._annotate_points(cat_a, targets_a[:1])
    labels_au, offs_au = trainer_au._annotate_points(cat_au, targets_au[:1])
    la, lb = labels_a[0].cpu(), labels_au[0].cpu()
    oa, ob = offs_a[0].cpu(), offs_au[0].cpu()

    # Level 0 is an exact common integer grid on both sides.
    n0 = min(pts_a[0].size(0), pts_au[0][0].size(0)
             if pts_au[0].ndim == 3 else pts_au[0].size(0))
    l0_a, l0_b = la[:n0], lb[:n0]
    off0_diff = (oa[:n0][l0_a & l0_b] - ob[:n0][l0_a & l0_b]).abs()

    # Levels >= 1: nearest-neighbour analysis between positive center sets.
    nn_report = []
    idx = n0
    bounds_a = list(n0 + npoints(pts_a)) if False else None
    return {
        "num_positive_A_total": int(la.sum()),
        "num_positive_AU_total": int(lb.sum()),
        "level0_common_prefix_len": int(n0),
        "level0_num_positive_A": int(l0_a.sum()),
        "level0_num_positive_AU": int(l0_b.sum()),
        "level0_positive_mask_equal": bool(torch.equal(l0_a, l0_b)),
        "level0_disagreement_indices": torch.nonzero(
            l0_a != l0_b).flatten().tolist()[:10],
        "level0_reg_target_max_abs_diff_common_positive":
            float(off0_diff.max()) if bool((l0_a & l0_b).any()) else None,
        "level0_reg_target_equal": bool(
            float(off0_diff.max()) < 1e-4) if bool((l0_a & l0_b).any()) else None,
    }


def npoints(levels):
    return [lvl.size(1) if lvl.ndim == 3 else lvl.size(0) for lvl in levels]


def stage_loss(trainer_a, trainer_au, out_a, debug_au, out_au, assign_au,
               targets_a, targets_au):
    rows = {}
    rows["A_full"] = {k: v for k, v in assemble_losses(
        trainer_a, out_a, None, targets_a).items() if not k.startswith("_")}
    rows["AU_trained"] = {k: v for k, v in assemble_losses(
        trainer_au, out_au, debug_au, targets_au,
        assignments=assign_au).items() if not k.startswith("_")}
    rows["AU_noaux"] = {k: v for k, v in assemble_losses(
        trainer_au, out_au, debug_au, targets_au, assignments=assign_au,
        importance_weight=0.0).items() if not k.startswith("_")}
    rows["AU_fixedpoint_note"] = (
        "infeasible as pure point swap: AU FPN widths are per-sample valid "
        "lengths and cannot index the legacy padded grid; isolating the "
        "coordinate system requires a padded-grid grouping code path")
    return rows


def stage_grad(trainer_a, trainer_au, inputs, targets_a, targets_au):
    def backward_side(trainer, side_targets, adaptive_call):
        model = trainer.model
        model.zero_grad(set_to_none=True)
        vid, vmask, text, tmask, tsize = inputs
        kwargs = {"return_importance_debug": True,
                  "return_anchor_assignments": True} if adaptive_call else {}
        out = model(vid, vmask, text, tmask, tsize, **kwargs)
        extra = len(out) - 8
        debug = out[8] if extra >= 1 else None
        assignments = out[9] if extra >= 2 else None
        losses = assemble_losses(trainer, out, debug, side_targets,
                                 assignments=assignments, need_grad=True)
        losses["_graph"].backward()
        grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.detach().float().cpu()
        return losses, grads

    losses_a, grads_a = backward_side(trainer_a, targets_a, False)
    losses_au, grads_au = backward_side(trainer_au, targets_au, True)

    module_stats = {}
    first_module = None
    for name, grad in grads_a.items():
        if name not in grads_au:
            continue
        other = grads_au[name]
        ga, gb = grad.flatten(), other.flatten()
        denom = (ga.norm() * gb.norm()).clamp_min(1e-12)
        cos = float((ga * gb).sum() / denom)
        module_key = ".".join(name.split(".")[:4])
        entry = module_stats.setdefault(module_key, {"cos_min": 1.0,
                                                     "params": 0})
        entry["cos_min"] = min(entry["cos_min"], cos)
        entry["params"] += 1
        if first_module is None and cos < 0.99:
            first_module = {
                "param": name, "cosine": round(cos, 6),
                "grad_norm_a": round(float(ga.norm()), 6),
                "grad_norm_au": round(float(gb.norm()), 6),
            }
    order = sorted(module_stats.items(), key=lambda kv: kv[0])
    first_module_block = next(
        (k for k, v in order if v["cos_min"] < 0.99), None)
    return {
        "loss_values_with_grad": {"A": {k: v for k, v in losses_a.items()
                                        if not k.startswith("_")},
                                  "AU": {k: v for k, v in losses_au.items()
                                         if not k.startswith("_")}},
        "first_diverging_parameter": first_module,
        "first_diverging_module_block": first_module_block,
        "per_module_cosine_min": {k: v for k, v in order},
    }


from libs.train_utils import (  # noqa: E402
    generate_multiscale_gt_masks,
    generate_multiscale_gt_masks_from_points,
    generate_multiscale_gt_masks_contrastive,
)


if __name__ == "__main__":
    main()

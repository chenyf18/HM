#!/usr/bin/env python3
"""GMUA Stage-1: counterfactual marginal-utility data collection (HM-ALLOC-MU).

For each train sample, at configured shallow levels:
  L_base  : grounding loss (cls + lambda_reg * reg) under uniform partition
  L_split(i): same but uniform group i split into 1+1
  L_merge(i,i+1): two adjacent uniform groups merged into size-4
All forwards run under no_grad (utility target is a detached teacher).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "hydra"), str(ROOT / "tools")]

from libs import load_opt  # noqa: E402
from libs.modeling.model import make_models_net  # noqa: E402
from libs.modeling.adaptive_anchor import (  # noqa: E402
    QueryBoundaryAdaptiveAnchorAllocator as Q,
)
from tools import research_validate_adaptive as rva  # noqa: E402
import tools.exact_equivalence_audit_au as aud  # noqa: E402


def grounding_loss(trainer, out, targets):
    """cls + lambda_reg * reg exactly as the worker weights them."""
    fpn_logits_l, _, fpn_offsets_l, fpn_mask_l = out[:4]
    fpn_logits = torch.cat(fpn_logits_l, dim=1)
    fpn_offsets = torch.cat(fpn_offsets_l, dim=1)
    fpn_masks = torch.cat(fpn_mask_l, dim=1)
    pts = aud._points_levels(trainer, out)
    cat = (torch.cat(pts, dim=1) if pts[0].ndim == 3 else torch.cat(pts))
    labels, offsets = trainer._annotate_points(cat, targets)
    pos = torch.logical_and(labels, fpn_masks)
    cls = trainer._calc_focal_loss(
        logits=fpn_logits[fpn_masks], labels=labels[fpn_masks])
    reg = trainer._calc_iou_loss(
        pred_offsets=fpn_offsets[pos], gt_offsets=offsets[pos])
    return (cls + trainer.loss_weight * reg).detach().float(), \
        cls.detach().float(), reg.detach().float()


def uniform_offsets(vc, tc):
    # Must match the allocator's _uniform_offsets (ceil form): the short
    # group sits at the valid tail, never in the middle.
    tc = max(tc, 1)
    steps = torch.arange(1, tc)
    return torch.div(steps * vc + (tc - 1), tc, rounding_mode='floor')


def build_level_offsets(caps, level, b, split_at=None, merge_pair=None,
                        k_dummy=None):
    """(offsets, counts) per level; single split adds +1 anchor, merge -1."""
    x, mask = caps[level]
    T = x.shape[-1]
    B = x.shape[0]
    rows = []
    for bi in range(B):
        vm = mask[bi, 0].bool().cpu()
        vc = int(vm.sum())
        tc = min(vc, max(1, -(-vc // 2)))
        uoff = uniform_offsets(vc, tc)
        cuts = set(int(c) for c in uoff)
        starts = [0] + sorted(cuts)
        ends = [starts[g + 1] for g in range(len(starts) - 1)] + [vc]
        if split_at is not None:
            # one interior cut inside the size-2 group -> 1 + 1
            cuts.add(int(starts[split_at]) + 1)
        if merge_pair is not None:
            g, h = merge_pair
            cuts.discard(int(starts[h]))
        final = sorted(cuts)
        rows.append((final, tc + (1 if split_at is not None else 0)
                     - (1 if merge_pair is not None else 0)))
    W = max(len(r[0]) for r in rows)
    out = torch.full((B, W), T, dtype=torch.long)
    counts = torch.zeros(B, dtype=torch.long)
    for bi, (r, c) in enumerate(rows):
        out[bi, :len(r)] = torch.tensor(r)
        counts[bi] = c
    return out, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(
        ROOT / "opts/research_allocator_AUC_clean.yaml"))
    ap.add_argument("--ckpt", default=str(
        ROOT / "experiments/allocator_diagnosis/seed_1/AUC/models/last.pth"))
    ap.add_argument("--output", default=str(
        ROOT / "experiments/allocator_diagnosis/marginal_utility.jsonl"))
    ap.add_argument("--num-samples", type=int, default=200)
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234567891)
    args = ap.parse_args()

    opt = load_opt(str(args.config), is_training=True)
    torch.manual_seed(args.seed)
    model = make_models_net(opt)
    aud.load_shared_checkpoint(model, str(args.ckpt))
    model = model.cuda().eval()

    from libs.worker import TrainerAuxiliary
    scratch = Path("/tmp/mu_collect")
    scratch.mkdir(exist_ok=True)
    opt["_root"] = str(scratch); opt["_resume"] = False
    opt["_distributed"] = False; opt["_world_size"] = 1
    torch.manual_seed(args.seed)
    trainer = TrainerAuxiliary(copy_opt := opt)
    # Reuse the already-loaded eval model so last_temporal_metadata and the
    # loss/annotation helpers operate on the exact same graph.
    trainer.model = model
    trainer.model_ema = model

    samples, _ = rva.select_real_samples(opt, args.num_samples, args.seed)
    rng = random.Random(args.seed)
    out_path = Path(args.output)
    fout = out_path.open("w")
    stats = {"split": [], "merge": []}
    n_done = 0
    for sample in samples:
        batch = rva.batchify([sample], opt)
        args_t = tuple(t.cuda() for t in (
            batch["video"], batch["video_mask"], batch["text"],
            batch["text_mask"], batch["text_size"]))
        targets = rva.targets_for_model(batch, opt).float().cuda()
        caps = []
        blocks = [m for n, m in model.named_modules()
                  if n.startswith("vid_net.branch.")
                  and "." not in n[len("vid_net.branch."):]]
        hooks = [m.register_forward_pre_hook(
            lambda m, a, i=i: caps.append((a[0].detach(), a[1].detach())))
            for i, m in enumerate(blocks)]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            base_out = model(*args_t, return_importance_debug=True)
        for h in hooks:
            h.remove()
        L_base, cls_b, reg_b = grounding_loss(trainer, base_out, targets)
        imp_levels = [d["importance"] for d in base_out[8]]
        meta_levels = list(model.vid_net.last_temporal_metadata)

        rec_base = {
            "vid_id": sample.get("vid_id"),
            "level_base": True,
            "loss": float(L_base), "cls": float(cls_b), "reg": float(reg_b),
        }
        fout.write(json.dumps(rec_base) + "\n")

        for level in args.levels:
            x, mask = caps[level]
            B, _, T = x.shape
            vm = mask[0, 0].bool().cpu()
            vc = int(vm.sum())
            tc = min(vc, max(1, -(-vc // 2)))
            starts = [0] + sorted(int(c) for c in uniform_offsets(vc, tc))
            if tc > 1 and len(starts) >= tc:
                ends = [starts[g + 1] for g in range(tc - 1)] + [vc]
            else:
                ends = [starts[g + 1] for g in range(len(starts) - 1)] + [vc]
            full = [g for g in range(tc) if ends[g] - starts[g] == 2
                    and g + 1 < tc]
            splits = rng.sample(full, min(args.k, len(full))) if full else []
            pairs = [(g, g + 1) for g in full if (g + 1) in full]
            rng.shuffle(pairs)
            merges, used = [], set()
            for g, h in pairs:
                if g in used or h in used or h - 1 in used or h + 1 in used:
                    continue
                merges.append((g, h)); used.update((g, h))
                if len(merges) >= args.k:
                    break
            for g in splits:
                built = [build_level_offsets(caps, l, 0) if l != level else
                         build_level_offsets(caps, l, 0, split_at=g)
                         for l in range(len(caps))]
                offs = tuple(o[0].cuda() for o in built)
                cnts = tuple(o[1].cuda() for o in built)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    o = model(*args_t, forced_cut_offsets=offs,
                              forced_target_counts=cnts,
                              return_importance_debug=True)
                L_c, cls_c, reg_c = grounding_loss(trainer, o, targets)
                stats["split"].append(float(L_base - L_c))
                imp = float(imp_levels[level][0, starts[g]:min(
                    starts[g + 1] if g + 1 < len(starts) else vc, T)]
                    .mean()) if level < len(imp_levels) else None
                fout.write(json.dumps({
                    "vid_id": sample.get("vid_id"), "level": level,
                    "type": "split", "group": g,
                    "base_loss": float(L_base), "cand_loss": float(L_c),
                    "utility": float(L_base - L_c),
                    "d_cls": float(cls_c - cls_b), "d_reg": float(reg_c - reg_b),
                    "importance_mean": imp,
                }) + "\n")
            for g, h in merges:
                built = [build_level_offsets(caps, l, 0) if l != level else
                         build_level_offsets(caps, l, 0, merge_pair=(g, h))
                         for l in range(len(caps))]
                offs = tuple(o[0].cuda() for o in built)
                cnts = tuple(o[1].cuda() for o in built)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    o = model(*args_t, forced_cut_offsets=offs,
                              forced_target_counts=cnts,
                              return_importance_debug=True)
                L_c, cls_c, reg_c = grounding_loss(trainer, o, targets)
                stats["merge"].append(float(L_base - L_c))
                fout.write(json.dumps({
                    "vid_id": sample.get("vid_id"), "level": level,
                    "type": "merge", "pair": [g, h],
                    "base_loss": float(L_base), "cand_loss": float(L_c),
                    "utility": float(L_base - L_c),
                    "d_cls": float(cls_c - cls_b), "d_reg": float(reg_c - reg_b),
                }) + "\n")
        n_done += 1
        if n_done % 25 == 0:
            print(f"processed {n_done} samples", flush=True)
    fout.close()

    def q(vals, p):
        v = sorted(vals)
        return v[min(int(p * len(v)), len(v) - 1)] if v else None

    for kind in ("split", "merge"):
        v = stats[kind]
        if v:
            print(f"{kind}: n={len(v)} mean={sum(v)/len(v):.5f} std="
                  f"{(sum((x-sum(v)/len(v))**2 for x in v)/len(v))**0.5:.5f} "
                  f"p50={q(v,0.5):.5f} p90={q(v,0.9):.5f} p99={q(v,0.99):.5f}")
    print("records:", out_path)


if __name__ == "__main__":
    main()

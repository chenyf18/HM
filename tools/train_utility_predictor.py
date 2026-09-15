#!/usr/bin/env python3
"""GMUA Stage-2: offline utility-predictor training and quality report.

Trains the existing per-level QueryBoundaryImportancePredictor (levels 0/1)
to regress counterfactual grounding utility of split/merge candidates.
Prediction for a candidate = mean predictor importance over its tokens
(the exact aggregation the balanced allocator uses for selection).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "hydra"), str(ROOT / "tools")]

from libs import load_opt  # noqa: E402
from libs.modeling.model import make_models_net  # noqa: E402
from tools import research_validate_adaptive as rva  # noqa: E402
import tools.exact_equivalence_audit_au as aud  # noqa: E402
from tools.collect_marginal_utility import uniform_offsets  # noqa: E402


def group_spans(vc, tc):
    starts = [0] + sorted(int(c) for c in uniform_offsets(vc, tc))
    ends = [starts[g + 1] for g in range(len(starts) - 1)] + [vc]
    return [(starts[g], ends[g]) for g in range(tc)]


def pearson(x, y):
    mx, my = sum(x) / len(x), sum(y) / len(y)
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    return cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else float("nan")


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return pearson(rank(list(x)), rank(list(y)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(
        ROOT / "opts/research_allocator_AUC_clean.yaml"))
    ap.add_argument("--ckpt", default=str(
        ROOT / "experiments/allocator_diagnosis/seed_1/AUC/models/last.pth"))
    ap.add_argument("--data", default=str(
        ROOT / "experiments/allocator_diagnosis/marginal_utility.jsonl"))
    ap.add_argument("--save", default=str(
        ROOT / "experiments/allocator_diagnosis/utility_predictor_levels01.pt"))
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clamp", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=1234567891)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.data)]
    by_vid = defaultdict(list)
    for r in records:
        if r.get("type") in ("split", "merge") and r.get("level") in args.levels:
            by_vid[r["vid_id"]].append(r)
    vids = sorted(by_vid)
    rng = random.Random(args.seed)
    rng.shuffle(vids)
    n_val = max(4, int(0.3 * len(vids)))
    val_vids, train_vids = set(vids[:n_val]), vids[n_val:]
    train_u = [r["utility"] for v in train_vids for r in by_vid[v]]
    scale = math.sqrt(sum((u - sum(train_u) / len(train_u)) ** 2
                          for u in train_u) / max(len(train_u), 1))
    print(f"samples train={len(train_vids)} val={len(val_vids)} "
          f"records train={len(train_u)} target_scale(std)={scale:.4f}")

    opt = load_opt(str(args.config), is_training=True)
    torch.manual_seed(args.seed)
    model = make_models_net(opt)
    aud.load_shared_checkpoint(model, str(args.ckpt))
    model = model.cuda()
    blocks = [m for n, m in model.named_modules()
              if n.startswith("vid_net.branch.")
              and "." not in n[len("vid_net.branch."):]]
    pred_params = []
    for lvl in args.levels:
        pred_params += list(blocks[lvl].importance_predictor.parameters())
    for p in model.parameters():
        p.requires_grad_(False)
    for p in pred_params:
        p.requires_grad_(True)
    optim = torch.optim.Adam(pred_params, lr=args.lr)
    samples, _ = rva.select_real_samples(opt, 200, args.seed)
    by_vid_samples = {s.get("vid_id"): s for s in samples}

    def forward_batch(sample):
        batch = rva.batchify([sample], opt)
        caps = []
        hooks = [m.register_forward_pre_hook(
            lambda m, a: caps.append((a[0].detach(), a[1].detach())))
            for m in blocks]
        args_t = tuple(t.cuda() for t in (
            batch["video"], batch["video_mask"], batch["text"],
            batch["text_mask"], batch["text_size"]))
        out = model(*args_t, return_importance_debug=True)
        for h in hooks:
            h.remove()
        imp = [d["importance"] for d in out[8]]
        return imp

    def targets_and_preds(imp, recs, level, compute_grad):
        ctx = torch.enable_grad() if compute_grad else torch.no_grad()
        with ctx:
            pass
        x_mask_vc = None
        losses, preds, tgts = [], [], []
        for r in recs:
            if r["level"] != level:
                continue
            imp_l = imp[level]
            vc = imp_l.shape[-1]
            tc = min(vc, max(1, -(-vc // 2)))
            spans = group_spans(vc, tc)
            if r["type"] == "split":
                lo, hi = spans[r["group"]]
            else:
                g, h = r["pair"]
                lo, hi = spans[g][0], spans[h][1]
            pred = imp_l[0, lo:hi].mean()
            tgt = float(r["utility"]) / scale
            tgt = max(-args.clamp, min(args.clamp, tgt))
            preds.append(float(pred.detach()))
            tgts.append(tgt * scale)
            losses.append((pred, torch.tensor(tgt, device=pred.device)))
        return losses, preds, tgts

    for epoch in range(args.epochs):
        total, count = 0.0, 0
        rng.shuffle(train_vids)
        for vid in train_vids:
            sample = by_vid_samples.get(vid)
            if sample is None:
                continue
            recs = by_vid[vid]
            imp = forward_batch(sample)
            optim.zero_grad()
            loss_sum = None
            for level in args.levels:
                losses, _, _ = targets_and_preds(imp, recs, level, True)
                for pred, tgt in losses:
                    l = F.smooth_l1_loss(pred.unsqueeze(0), tgt.unsqueeze(0))
                    loss_sum = l if loss_sum is None else loss_sum + l
            if loss_sum is not None:
                loss_sum.backward()
                optim.step()
                total += float(loss_sum)
                count += 1
        print(f"epoch {epoch}: mean loss/candidate-group-sample "
              f"{total / max(count, 1):.5f} ({count} samples)")

    # held-out quality
    preds_all, tgts_all = [], []
    rank_correct, rank_total = 0, 0
    for vid in sorted(val_vids):
        sample = by_vid_samples.get(vid)
        if sample is None:
            continue
        recs = by_vid[vid]
        with torch.no_grad():
            imp = forward_batch(sample)
        for level in args.levels:
            _, preds, tgts = targets_and_preds(imp, recs, level, False)
            preds_all += preds
            tgts_all += tgts
        # pairwise ranking within (sample, level, type)
        for level in args.levels:
            sub = [r for r in recs if r["level"] == level]
            preds_l = []
            tgts_l = []
            for r in sub:
                vc = imp[level].shape[-1]
                spans = group_spans(vc, min(vc, max(1, -(-vc // 2))))
                if r["type"] == "split":
                    lo, hi = spans[r["group"]]
                else:
                    g, h = r["pair"]
                    lo, hi = spans[g][0], spans[h][1]
                preds_l.append(float(imp[level][0, lo:hi].mean()))
                tgts_l.append(float(r["utility"]))
            for i in range(len(preds_l)):
                for j in range(i + 1, len(preds_l)):
                    if abs(tgts_l[i] - tgts_l[j]) > 0.05:
                        rank_total += 1
                        if (preds_l[i] > preds_l[j]) == (
                                tgts_l[i] > tgts_l[j]):
                            rank_correct += 1
    pe = pearson(preds_all, tgts_all) if len(preds_all) > 2 else float("nan")
    sp = spearman(preds_all, tgts_all) if len(preds_all) > 2 else float("nan")
    ra = rank_correct / max(rank_total, 1)
    print(f"HELD-OUT: n={len(preds_all)} pearson={pe:.4f} "
          f"spearman={sp:.4f} ranking_acc={ra:.4f} (pairs={rank_total})")
    state = {f"level{lvl}": blocks[lvl].importance_predictor.state_dict()
             for lvl in args.levels}
    torch.save({"state": state, "scale": scale, "clamp": args.clamp,
                "levels": args.levels,
                "quality": {"pearson": pe, "spearman": sp,
                            "ranking_acc": ra, "n_val": len(preds_all)}},
               args.save)
    print("saved:", args.save)


if __name__ == "__main__":
    main()

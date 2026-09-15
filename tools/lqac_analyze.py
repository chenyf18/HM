#!/usr/bin/env python3
"""Offline analysis for the LQAC frozen probe (HM-LQAC-022).

Consumes experiments/lqac_probe/val_rescore.npz (full val pool scalars
with q_hat from both heads) and lqac_heads.pt (LC-only parameters).
Produces Gate-1 quality-predictability metrics, rescoring tables
(S0-S3, Q-Base / Q-Level / LC-only / LCxQ), per-level diagnostics and
duration analysis. Writes lqac_results.json.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "hydra"), str(ROOT / "tools")]

from tools.audit_ranking_bottleneck import (  # noqa: E402
    iou_1d, rankdata_avg, corr, soft_nms,
)
from tools.lqac_probe import (  # noqa: E402
    LQAC_ROOT, VAL_RESCORE_NPZ, HEADS_PT, RESULTS_JSON,
)

METRICS = ("R1@0.3", "R1@0.5", "R5@0.3", "R5@0.5")


def _rk_from_scores(pool_sorted, scores, ious_sorted, to_sec, gt):
    """Pool already top-2000 by cls; apply scoring, soft-NMS, R@K."""
    top = pool_sorted
    k_segs, k_scores = soft_nms(
        np.stack([top[:, 7], top[:, 8]], axis=1), scores)
    seg_sec = np.stack(
        [to_sec(k_segs[:, 0]), to_sec(k_segs[:, 1])], axis=1)
    q_i = iou_1d(seg_sec[:, 0], seg_sec[:, 1], gt[0], gt[1])
    top1 = q_i[0] if len(q_i) else 0.0
    best5 = q_i.max() if len(q_i) else 0.0
    return (top1 >= 0.3, top1 >= 0.5, best5 >= 0.3, best5 >= 0.5), top1


def run_analyze():
    data = np.load(VAL_RESCORE_NPZ, allow_pickle=False)
    heads = None
    if HEADS_PT.exists():
        import torch
        heads = torch.load(HEADS_PT, map_location="cpu",
                           weights_only=False)
    n = len(data["gt_start"])
    pool = data["pool"]
    offs = data["pool_offsets"]
    ious_all = data["ious"]
    q_base_all = data["q_base"]
    q_level_all = data["q_level"]
    gt_s, gt_e = data["gt_start"], data["gt_end"]
    clip_stride, clip_size = data["clip_stride"], data["clip_size"]
    fps, duration = data["fps"], data["duration"]

    lc_a = np.array(heads["lc"]["a"]) if heads else np.ones(8)
    lc_b = np.array(heads["lc"]["b"]) if heads else np.zeros(8)

    # ---------- assemble per-query structures ----------
    per_q = []
    for i in range(n):
        sl = slice(offs[i], offs[i + 1])
        p = pool[sl]
        if p.shape[0]:
            order = np.argsort(-p[:, 1], kind="stable")
            p = p[order]
        p = p[:2000]
        iou = ious_all[sl]
        if iou.shape[0]:
            iou = iou[order][:2000]
        qb = q_base_all[sl]
        ql = q_level_all[sl]
        if qb.shape[0]:
            qb = qb[order][:2000]
            ql = ql[order][:2000]
        per_q.append({
            "p": p, "iou": iou, "qb": qb, "ql": ql,
            "gt": (gt_s[i], gt_e[i]),
            "gt_dur": gt_e[i] - gt_s[i],
            "cs": clip_stride[i], "cc": clip_size[i],
            "fps": fps[i], "dur": duration[i],
        })

    def to_sec_fn(rec):
        def f(tok):
            return np.clip(
                (tok * rec["cs"] + 0.5 * rec["cc"]) / rec["fps"],
                0.0, rec["dur"])
        return f

    # ---------- Stage 2: quality predictability ----------
    qb_all = np.concatenate([r["qb"] for r in per_q if r["qb"].size])
    ql_all = np.concatenate([r["ql"] for r in per_q if r["ql"].size])
    iou_flat = np.concatenate([r["iou"] for r in per_q if r["iou"].size])
    cls_flat = np.concatenate([r["p"][:, 1] for r in per_q
                               if r["p"].size])

    def chunked_spearman(x, y, chunk=200000):
        vals = []
        for s in np.array_split(np.arange(len(x)),
                                max(1, len(x) // chunk)):
            if len(s) < 3:
                continue
            v = corr(rankdata_avg(x[s]), rankdata_avg(y[s]))
            if not math.isnan(v):
                vals.append(v)
        return float(np.mean(vals)), float(np.std(vals))

    sp_qb_g, sp_qb_gsd = chunked_spearman(qb_all, iou_flat)
    sp_ql_g, sp_ql_gsd = chunked_spearman(ql_all, iou_flat)
    pe_qb_g = corr(qb_all, iou_flat)
    pe_ql_g = corr(ql_all, iou_flat)
    mae_qb = float(np.abs(qb_all - iou_flat).mean())
    mae_ql = float(np.abs(ql_all - iou_flat).mean())
    sl1_qb = float(np.where(
        np.abs(qb_all - iou_flat) < 0.1,
        0.5 * (qb_all - iou_flat) ** 2 / 0.1,
        np.abs(qb_all - iou_flat) - 0.05).mean())

    # per-query correlations top20/50/100 + pairwise accuracy
    def per_query_stats(key):
        sp = {20: [], 50: [], 100: []}
        pe = {20: [], 50: [], 100: []}
        pw = {20: [], 50: []}
        for r in per_q:
            qv = r[key]
            iv = r["iou"]
            if r["p"].shape[0] < 3:
                continue
            for K in (20, 50, 100):
                k = min(K, len(qv))
                v = corr(rankdata_avg(qv[:k]), rankdata_avg(iv[:k]))
                if not math.isnan(v):
                    sp[K].append(v)
                v2 = corr(qv[:k], iv[:k])
                if not math.isnan(v2):
                    pe[K].append(v2)
            for K in (20, 50):
                k = min(K, len(qv))
                ii, jj = np.triu_indices(k, k=1)
                d = iv[ii] - iv[jj]
                ok = np.abs(d) > 1e-6
                if ok.sum() == 0:
                    continue
                agree = np.sign(
                    qv[ii][ok] - qv[jj][ok]) == np.sign(d[ok])
                pw[K].append(float(agree.mean()))
        return {
            f"top{K}": {
                "spearman_mean": float(np.mean(sp[K])),
                "spearman_median": float(np.median(sp[K])),
                "spearman_std": float(np.std(sp[K])),
                "pearson_mean": float(np.mean(pe[K])),
                "n_valid": len(sp[K]),
            } for K in (20, 50, 100)
        } | {
            f"pairwise_top{K}": {
                "mean": float(np.mean(pw[K])),
                "median": float(np.median(pw[K])),
                "n_valid": len(pw[K]),
            } for K in (20, 50)
        }

    # calibration curve: mean q per true-IoU bin
    bins = np.array([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
    which = np.digitize(iou_flat, bins) - 1
    calib = {
        "bins": bins.tolist(),
        "q_base": [float(qb_all[which == b].mean())
                   if (which == b).any() else None
                   for b in range(len(bins) - 1)],
        "q_level": [float(ql_all[which == b].mean())
                    if (which == b).any() else None
                    for b in range(len(bins) - 1)],
        "count": [int((which == b).sum()) for b in range(len(bins) - 1)],
    }

    predictability = {
        "global": {
            "spearman_q_base": sp_qb_g, "spearman_q_base_std": sp_qb_gsd,
            "spearman_q_level": sp_ql_g, "spearman_q_level_std": sp_ql_gsd,
            "pearson_q_base": pe_qb_g, "pearson_q_level": pe_ql_g,
            "mae_q_base": mae_qb, "mae_q_level": mae_ql,
            "smoothl1_q_base": sl1_qb,
            "n_candidates": int(len(iou_flat)),
        },
        "per_query_q_base": per_query_stats("qb"),
        "per_query_q_level": per_query_stats("ql"),
        "calibration_curve": calib,
    }

    # ---------- Stage 3: rescoring ----------
    def scoring_variants(r):
        lv = r["p"][:, 2].astype(int)
        lc_score = 1.0 / (1.0 + np.exp(
            -(lc_a[lv] * r["p"][:, 0] + lc_b[lv])))
        cls = r["p"][:, 1]
        qb, ql = r["qb"], r["ql"]
        eps = 1e-8
        return {
            "S0_cls": cls,
            "S1_q_base": qb,
            "S2_cls*q_base": cls * qb,
            "S3_sqrt(cls*q_base)": np.sqrt(
                np.maximum(cls, 0) * np.maximum(qb, 0) + eps),
            "S1_q_level": ql,
            "S2_cls*q_level": cls * ql,
            "S3_sqrt(cls*q_level)": np.sqrt(
                np.maximum(cls, 0) * np.maximum(ql, 0) + eps),
            "LC_only": lc_score,
            "LCxQ_level": np.sqrt(
                np.maximum(lc_score, 0) * np.maximum(ql, 0) + eps),
        }

    names = list(scoring_variants(per_q[0]).keys()) if per_q else []
    counts = {nm: np.zeros(4) for nm in names}
    rescue = {nm: {K: 0 for K in (5, 10, 20)} for nm in names}
    best_rank = {nm: [] for nm in names}
    spear_var = {nm: [] for nm in names}
    s0_top1 = np.zeros(n)

    for i, r in enumerate(per_q):
        variants = scoring_variants(r)
        to_sec = to_sec_fn(r)
        for nm in names:
            hit, top1 = _rk_from_scores(
                r["p"], variants[nm], r["iou"], to_sec, r["gt"])
            counts[nm] += hit
            if nm == "S0_cls":
                s0_top1[i] = top1
            iou = r["iou"]
            if iou.size:
                order = np.argsort(-variants[nm], kind="stable")
                rank_of_iou = np.empty(len(iou), dtype=int)
                rank_of_iou[order] = np.arange(len(iou))
                bi = np.where(iou >= iou.max() - 1e-9)[0]
                best_rank[nm].append(int(rank_of_iou[bi].min()) + 1)
                for K in (5, 10, 20):
                    k = min(K, len(iou))
                    if top1 < 0.5 and iou[order[:k]].max() >= 0.5:
                        rescue[nm][K] += 1
                if len(iou) >= 20:
                    v = corr(rankdata_avg(variants[nm][:20]),
                             rankdata_avg(iou[:20]))
                    if not math.isnan(v):
                        spear_var[nm].append(v)

    rescoring = {}
    for nm in names:
        m = counts[nm] / n
        rescoring[nm] = {
            "R1@0.3": float(m[0]), "R1@0.5": float(m[1]),
            "R5@0.3": float(m[2]), "R5@0.5": float(m[3]),
            "Mean": float(m.mean()),
            "rescue@5@0.5": rescue[nm][5] / max(
                int((s0_top1 < 0.5).sum()), 1),
            "rescue@10@0.5": rescue[nm][10] / max(
                int((s0_top1 < 0.5).sum()), 1),
            "rescue@20@0.5": rescue[nm][20] / max(
                int((s0_top1 < 0.5).sum()), 1),
            "best_iou_rank_median": float(np.median(best_rank[nm])),
            "best_iou_rank_le10": float(
                (np.array(best_rank[nm]) <= 10).mean()),
            "spearman_top20_mean": float(np.mean(spear_var[nm])),
        }

    # official P32 anchor for S0 comparison
    official = json.loads((ROOT / "experiments/allocator_diagnosis"
                           "/seed_1/AUC_P32/evaluation_summary.json"
                           ).read_text())["metrics"]
    off_mean = 100 * sum(official.values()) / 4

    # ---------- level diagnostics ----------
    level_diag = []
    lv_all = np.concatenate([r["p"][:, 2] for r in per_q
                             if r["p"].size]).astype(int)
    for lv in range(8):
        m = lv_all == lv
        if not m.any():
            continue
        sp_cls = corr(rankdata_avg(cls_flat[m]), rankdata_avg(iou_flat[m]))
        sp_qb = corr(rankdata_avg(qb_all[m]), rankdata_avg(iou_flat[m]))
        sp_ql = corr(rankdata_avg(ql_all[m]), rankdata_avg(iou_flat[m]))
        fused = cls_flat[m] * ql_all[m]
        sp_f = corr(rankdata_avg(fused), rankdata_avg(iou_flat[m]))
        level_diag.append({
            "level": lv,
            "n": int(m.sum()),
            "mean_cls": float(cls_flat[m].mean()),
            "mean_iou": float(iou_flat[m].mean()),
            "mean_q_base": float(qb_all[m].mean()),
            "mean_q_level": float(ql_all[m].mean()),
            "spearman_cls": sp_cls, "spearman_q_base": sp_qb,
            "spearman_q_level": sp_ql, "spearman_fused_clsq": sp_f,
        })

    # ---------- duration analysis ----------
    durs = np.array([r["gt_dur"] for r in per_q])
    t1, t2 = np.percentile(durs, [100 / 3, 200 / 3])
    best_name = max(
        [nm for nm in names], key=lambda nm: rescoring[nm]["Mean"])
    duration = {"terciles": [float(t1), float(t2)], "best": best_name}
    # recompute per-bucket R@K for S0 and best variant
    for nm in ("S0_cls", best_name):
        per_b = {}
        for bname, mask in (
            ("short", durs <= t1), ("medium", (durs > t1) & (durs <= t2)),
            ("long", durs > t2),
        ):
            idxs = np.where(mask)[0]
            c = np.zeros(4)
            for i in idxs:
                r = per_q[i]
                hit, _ = _rk_from_scores(
                    r["p"], scoring_variants(r)[nm], r["iou"],
                    to_sec_fn(r), r["gt"])
                c += hit
            m = c / max(len(idxs), 1)
            per_b[bname] = {
                "n": int(len(idxs)),
                "R1@0.3": float(m[0]), "R1@0.5": float(m[1]),
                "R5@0.3": float(m[2]), "R5@0.5": float(m[3]),
                "Mean": float(m.mean()),
            }
        duration[nm] = per_b

    out = {
        "n_queries": n,
        "official_P32": {k: 100 * v for k, v in official.items()},
        "official_P32_mean": off_mean,
        "predictability": predictability,
        "rescoring": rescoring,
        "level_diagnostics": level_diag,
        "duration_analysis": duration,
    }
    RESULTS_JSON.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    # ---------- console summary ----------
    print(f"\n=== GATE 1: quality predictability (global chunked) ===")
    print(f"Q-Base : Spearman={sp_qb_g:.4f}±{sp_qb_gsd:.3f} "
          f"Pearson={pe_qb_g:.4f} MAE={mae_qb:.4f}")
    print(f"Q-Level: Spearman={sp_ql_g:.4f}±{sp_ql_gsd:.3f} "
          f"Pearson={pe_ql_g:.4f} MAE={mae_ql:.4f}")
    print(f"pairwise(Q-Base, top20) = "
          f"{predictability['per_query_q_base']['pairwise_top20']['mean']:.4f}"
          f"  (cls was 0.507)")
    print(f"\n=== RESCORING (official P32 Mean = {off_mean:.2f}) ===")
    print(f"{'scoring':26s} R1@0.3 R1@0.5 R5@0.3 R5@0.5  Mean   dMean")
    for nm in names:
        r = rescoring[nm]
        print(f"{nm:26s} {100*r['R1@0.3']:6.2f} {100*r['R1@0.5']:6.2f} "
              f"{100*r['R5@0.3']:6.2f} {100*r['R5@0.5']:6.2f} "
              f"{100*r['Mean']:6.2f} {100*r['Mean']-off_mean:+6.2f}")
    print("\nresults ->", RESULTS_JSON)


if __name__ == "__main__":
    run_analyze()

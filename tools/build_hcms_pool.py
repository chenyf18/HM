#!/usr/bin/env python3
"""HM-HCMS-001 stage 2/3: build + audit the hard-negative pool.

Selection rules per query (all in seconds):
  1. same video (by construction)
  2. IoU(candidate, GT) < 0.1
  3. |len(cand) - len(GT)| / len(GT) < 0.1
  4. candidate cls score in the query's top-20% by score
  5. |center_cand - center_GT| > max(len_GT, len_cand)   (non-neighbor)
Output: one JSON line per query with its hard-negative list.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "hydra"), str(ROOT / "tools")]

DUMP = ROOT / "experiments/hcms_pool/train_candidates.jsonl"
POOL = ROOT / "experiments/hcms_pool/hard_negative_pool.jsonl"
AUDIT = ROOT / "experiments/hcms_pool/pool_audit.json"

IOU_MAX = 0.1
DUR_TOL = 0.1
TOP_FRAC = 0.2


def main():
    queries = defaultdict(list)
    gt = {}
    with open(DUMP) as f:
        for line in f:
            r = json.loads(line)
            key = (r["vid_id"], r["query_idx"])
            queries[key].append(r)
    # GT comes from the dataset (seconds); rebuild via annotations
    sys.path[:0] = [str(ROOT), str(ROOT / "hydra")]
    import libs.worker as W
    from libs import load_opt
    from libs.data import make_dataset
    opt = load_opt(str(
        ROOT / "experiments/allocator_diagnosis/seed_1/AUC/opt.yaml"),
        is_training=False)
    opt["eval"]["data"]["split"] = "train"
    ds = make_dataset(opt["eval"]["data"], is_training=False)
    for i in range(len(ds)):
        item = ds[i]
        for q, seg in enumerate(item["segment"]):
            gt[(str(item["vid_id"]), q)] = (float(seg[0]), float(seg[1]))

    n_q = len(gt)
    pools = {}
    stats = {
        "n_queries": n_q,
        "coverage": 0,
        "avg_neg_per_query": 0.0,
        "dur_ratio": [], "iou": [], "score": [], "score_gap": [],
    }
    examples = []
    for key, cands in queries.items():
        if key not in gt:
            continue
        g0, g1 = gt[key]
        glen = max(g1 - g0, 1e-6)
        gctr = 0.5 * (g0 + g1)
        if not cands:
            continue
        scores = np.array([c["score"] for c in cands])
        thr = np.quantile(scores, 1 - TOP_FRAC)
        # best GT-overlapping candidate score for the gap metric
        gt_scores = [c["score"] for c in cands if c["iou"] >= 0.5]
        score_gt = max(gt_scores) if gt_scores else np.nan
        negs = []
        for c in cands:
            dur = c["duration"]
            if c["iou"] >= IOU_MAX:
                continue
            if abs(dur - glen) / glen >= DUR_TOL:
                continue
            if c["score"] < thr:
                continue
            if abs(c["center"] - gctr) <= max(glen, dur):
                continue
            negs.append({
                "neg_start": c["start"], "neg_end": c["end"],
                "neg_score": c["score"], "neg_iou": c["iou"],
                "duration_ratio": dur / glen, "rank": c["rank"],
            })
        if negs:
            negs.sort(key=lambda x: -x["neg_score"])
            pools[key] = negs
            stats["coverage"] += 1
            stats["avg_neg_per_query"] += len(negs)
            stats["dur_ratio"] += [n["duration_ratio"] for n in negs]
            stats["iou"] += [n["neg_iou"] for n in negs]
            stats["score"] += [n["neg_score"] for n in negs]
            if not math_isnan(score_gt):
                stats["score_gap"] += [
                    score_gt - n["neg_score"] for n in negs]
            if len(examples) < 5 and gt_scores:
                examples.append({
                    "vid": key[0], "gt": [g0, g1],
                    "n_negs": len(negs),
                    "top_neg": negs[0],
                    "score_gt": score_gt})
    stats["coverage"] /= n_q
    stats["avg_neg_per_query"] /= max(len(pools), 1)
    audit = {
        "n_queries": n_q,
        "coverage": stats["coverage"],
        "avg_neg_per_query": stats["avg_neg_per_query"],
        "dur_ratio_pct": pct(stats["dur_ratio"]),
        "iou_pct": pct(stats["iou"]),
        "score_pct": pct(stats["score"]),
        "score_gap_pct": pct(stats["score_gap"]),
        "consistency_with_prior_audit":
            "selection reuses the 028 HN definition (IoU<0.1 here vs "
            "<0.3 there; top-20% score replaces rank<=20; adds explicit "
            "non-neighbor + duration-similarity rules)",
        "examples": examples,
    }
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n")
    with open(POOL, "w") as f:
        for key, negs in pools.items():
            f.write(json.dumps({
                "vid_id": key[0], "query_idx": key[1],
                "hard_negatives": negs[:20]}) + "\n")
    print(json.dumps({k: v for k, v in audit.items()
                      if k != "examples"}, indent=2))
    print("->", POOL)


def math_isnan(x):
    return isinstance(x, float) and np.isnan(x)


def pct(arr, qs=(0.05, 0.25, 0.5, 0.75, 0.95)):
    if not arr:
        return None
    a = np.asarray(arr)
    return {f"p{int(q*100)}": float(np.quantile(a, q)) for q in qs}


if __name__ == "__main__":
    main()

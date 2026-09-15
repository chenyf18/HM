#!/usr/bin/env python3
"""R@5 failure analysis between two prediction files (directive section 7).

Classifies why queries that the baseline retrieves inside top-5 are missed by
the treatment run:
  candidate_missing      no top-5 segment overlaps GT and none is near GT
  wrong_scale_near_gt    segments exist near GT but IoU == 0 (duration ratio
                         collapses or explodes)
  boundary_too_coarse    best IoU is positive but below the threshold
  low_ranked_beyond_top5 not detectable: only post-NMS top-5 is saved

Usage:
  python tools/analyze_r5_failures.py \
    --baseline experiments/formal_ablation/seed_1/A/predictions_last.json \
    --treatment <AL predictions_last.json> \
    --output experiments/allocator_diagnosis/seed_1/r5_failure_analysis.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def iou(a, b):
    inter = min(a[1], b[1]) - max(a[0], b[0])
    if inter <= 0:
        return 0.0
    union = max(a[1], b[1]) - min(a[0], b[0])
    return float(inter) / union if union > 0 else 0.0


def query_iter(predictions):
    for vid_id, entry in predictions["videos"].items():
        for index, query in enumerate(entry["queries"]):
            yield (vid_id, index), query


def build_index(path):
    payload = json.loads(Path(path).read_text())
    return {
        key: {
            "gt": [float(x) for x in query["ground_truth"]],
            "preds": [
                [float(s["segment"][0]), float(s["segment"][1]), float(s.get("score", 0.0))]
                for s in query["predictions"]
            ],
        }
        for key, query in query_iter(payload)
    }


def classify(entry, thresh):
    gt = entry["gt"]
    duration = max(gt[1] - gt[0], 1e-6)
    ious = [iou(seg, gt) for seg in entry["preds"]]
    best = max(ious) if ious else 0.0
    if best >= thresh:
        return "success", best, None
    if best <= 0.0:
        near = any(
            (seg[0] + seg[1]) / 2 >= gt[0] - duration
            and (seg[0] + seg[1]) / 2 <= gt[1] + duration
            for seg in entry["preds"]
        )
        if not near:
            return "candidate_missing", best, None
        ratios = [
            max((seg[1] - seg[0]) / duration, duration / max(seg[1] - seg[0], 1e-6))
            for seg in entry["preds"]
        ]
        return "wrong_scale_near_gt", best, min(ratios)
    return "boundary_too_coarse", best, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iou-thresh", type=float, default=0.3)
    parser.add_argument("--sample-n", type=int, default=12)
    args = parser.parse_args()

    base = build_index(args.baseline)
    treat = build_index(args.treatment)
    shared = sorted(set(base) & set(treat))
    missing_rows = len(set(base) ^ set(treat))

    categories = Counter()
    lost_queries = []
    gained = 0
    lost_best_iou_baseline = []
    lost_duration_ratios = []
    for key in shared:
        base_status, base_best, _ = classify(base[key], args.iou_thresh)
        treat_status, treat_best, ratio = classify(treat[key], args.iou_thresh)
        if base_status == "success" and treat_status != "success":
            categories[treat_status] += 1
            lost_queries.append((key, base_best, treat_best, ratio))
            lost_best_iou_baseline.append(base_best)
            if ratio:
                lost_duration_ratios.append(ratio)
        elif base_status != "success" and treat_status == "success":
            gained += 1
            categories["gained_by_treatment"] += 1

    lost_queries.sort(key=lambda item: item[2] - item[1])
    samples = []
    for key, base_best, treat_best, ratio in lost_queries[: args.sample_n]:
        samples.append({
            "video_id": key[0],
            "query_id": key[1],
            "gt": base[key]["gt"],
            "baseline_best_iou": round(base_best, 4),
            "treatment_best_iou": round(treat_best, 4),
            "duration_ratio_min": round(ratio, 4) if ratio else None,
            "baseline_top5": [
                [round(seg[0], 2), round(seg[1], 2)] for seg in base[key]["preds"]
            ],
            "treatment_top5": [
                [round(seg[0], 2), round(seg[1], 2)]
                for seg in treat[key]["preds"]
            ],
        })

    report = {
        "baseline": str(args.baseline),
        "treatment": str(args.treatment),
        "iou_threshold": args.iou_thresh,
        "shared_queries": len(shared),
        "queries_missing_in_one_file": missing_rows,
        "lost_at_rank5": len(lost_queries),
        "gained_at_rank5": gained,
        "lost_categories": dict(categories),
        "low_ranked_beyond_top5_note": (
            "only post-NMS top-5 candidates are saved; 'candidate exists but "
            "ranked >5' cannot be separated from 'missing' without a "
            "larger max_num_segs evaluation"
        ),
        "lost_baseline_best_iou_hist": {
            "lt_0.1": sum(v < 0.1 for v in lost_best_iou_baseline),
            "0.1_to_0.2": sum(0.1 <= v < 0.2 for v in lost_best_iou_baseline),
            "ge_0.2": sum(v >= 0.2 for v in lost_best_iou_baseline),
        },
        "lost_duration_ratio_median": (
            sorted(lost_duration_ratios)[len(lost_duration_ratios) // 2]
            if lost_duration_ratios
            else None
        ),
        "samples_worst_treatment_drop": samples,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in (
        "shared_queries", "lost_at_rank5", "gained_at_rank5",
        "lost_categories", "lost_baseline_best_iou_hist",
    )}, indent=2))
    print("report:", output)


if __name__ == "__main__":
    main()

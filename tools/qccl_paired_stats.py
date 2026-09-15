#!/usr/bin/env python3
"""Archived paired statistics for HM-QCCL-030 (post-hoc, no forward).

Video-level clustered bootstrap over the fixed step-8000 predictions of
the three arms (identical resampling across arms, query-weighted
metrics). Intervals describe validation-set sampling uncertainty of
these three fixed training runs; they do NOT capture across-seed
training uncertainty.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]


def iou_seg(a, b, gs, ge):
    inter = max(0.0, min(b, ge) - max(a, gs))
    union = (b - a) + (ge - gs) - inter
    return inter / union if union > 0 else 0.0


def load_per_query(arm):
    dump = np.load(
        ROOT / "experiments/audit_ranking_bottleneck/prenms_dump.npz")
    vid_order = list(dict.fromkeys(dump["vid_ids"].tolist()))
    gt = {}
    for v in vid_order:
        m = dump["vid_ids"] == v
        gt[v] = list(zip(dump["gt_start"][m].tolist(),
                         dump["gt_end"][m].tolist()))
    preds = json.load(open(
        ROOT / f"experiments/qccl_arms/{arm}/evals/step8000/"
        f"predictions_step8000.json"))["videos"]
    per_video = []
    for v in vid_order:
        hits = {k: [] for k in ("r1_3", "r1_5", "r5_3", "r5_5")}
        for q, (gs, ge) in zip(preds[v]["queries"], gt[v]):
            segs = [p["segment"] for p in q["predictions"]]
            if not segs:
                for k in hits:
                    hits[k].append(0.0)
                continue
            ious = [iou_seg(a, b, gs, ge) for a, b in segs]
            hits["r1_3"].append(ious[0] >= 0.3)
            hits["r1_5"].append(ious[0] >= 0.5)
            hits["r5_3"].append(max(ious) >= 0.3)
            hits["r5_5"].append(max(ious) >= 0.5)
        per_video.append({k: float(np.mean(v2)) for k, v2 in hits.items()})
    return per_video


def main():
    arms = ("G0", "G1", "G2")
    per_video = {a: load_per_query(a) for a in arms}
    keys = ("r1_3", "r1_5", "r5_3", "r5_5")
    n_videos = len(per_video["G0"])

    def video_mean(a, idxs, k):
        if k == "mean":
            return float(np.mean([
                np.mean([per_video[a][i][kk] for kk in keys])
                for i in idxs]))
        return float(np.mean([per_video[a][i][k] for i in idxs]))

    rng = np.random.default_rng(20260910)
    B = 2000
    print(f"videos: {n_videos}; bootstrap B={B} (video-clustered, "
          f"shared draws across arms; query-weighted within video)")
    out = {}
    for tag, (a, b) in {"G1-G0": ("G1", "G0"),
                        "G2-G0": ("G2", "G0"),
                        "G2-G1": ("G2", "G1")}.items():
        out[tag] = {}
        for k in keys + ("mean",):
            d0 = 100 * (video_mean(a, range(n_videos), k)
                        - video_mean(b, range(n_videos), k))
            bs = []
            for _ in range(B):
                idxs = rng.integers(0, n_videos, n_videos)
                bs.append(100 * (video_mean(a, idxs, k)
                                 - video_mean(b, idxs, k)))
            out[tag][k] = {"delta": d0,
                           "ci95": [float(np.percentile(bs, 2.5)),
                                    float(np.percentile(bs, 97.5))]}
            print(f"{tag} {k:5s}: {d0:+.2f} "
                  f"[{out[tag][k]['ci95'][0]:+.2f}, "
                  f"{out[tag][k]['ci95'][1]:+.2f}]")

    dest = ROOT / "experiments/qccl_arms/paired_stats_step8000.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print("->", dest)

    # archive hashes
    hashes = {}
    for p in [ROOT / "experiments/allocator_diagnosis/seed_1/AUC/models/last.pth"] + \
            [ROOT / f"experiments/qccl_arms/{a}/evals/step8000/models/step8000.pth"
             for a in arms] + \
            [ROOT / f"experiments/qccl_arms/{a}/evals/step8000/predictions_step8000.json"
             for a in arms]:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        hashes[str(p.relative_to(ROOT))] = h.hexdigest()
    (ROOT / "experiments/qccl_arms/archive_hashes.json").write_text(
        json.dumps(hashes, indent=2) + "\n")
    print("-> experiments/qccl_arms/archive_hashes.json")


if __name__ == "__main__":
    main()

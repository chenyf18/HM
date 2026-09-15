#!/usr/bin/env python3
"""HM-QCCL-030 numeric reconciliation (archive-quality, no forward).

Single source of truth for the step-8000 three-arm table:
  1. recompute each arm's four official metrics + Mean from the stored
     top-5 predictions using the OFFICIAL scoring rule (query-weighted
     global average; R1 = IoU of the score-ranked 1st, R5 = best IoU
     within the score-ranked top-5; no re-NMS);
  2. observed_difference = full-sample metric differences (also derived
     from the per-query hit matrix; both computations must agree);
  3. video-clustered bootstrap CIs (shared draws across arms) with the
     SAME query-weighted metric (resampled pool = concatenated queries
     of resampled videos). Bootstrap quantiles never replace observed
     differences.

Verifies prediction hashes against archive_hashes.json and asserts the
full query set (320 videos / 3498 queries) with no silent intersections.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

ARMS = ("G0", "G1", "G2")
KEYS = ("r1_3", "r1_5", "r5_3", "r5_5")
NAMES = {"r1_3": "Rank@1_IoU@0.3", "r1_5": "Rank@1_IoU@0.5",
         "r5_3": "Rank@5_IoU@0.3", "r5_5": "Rank@5_IoU@0.5"}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iou_seg(a, b, gs, ge):
    inter = max(0.0, min(b, ge) - max(a, gs))
    union = (b - a) + (ge - gs) - inter
    return inter / union if union > 0 else 0.0


def load_arm(arm, vid_order, gt, hashes):
    pred_path = ROOT / f"experiments/qccl_arms/{arm}/evals/step8000/" \
        f"predictions_step8000.json"
    digest = sha256(pred_path)
    assert digest == hashes[str(pred_path.relative_to(ROOT))], \
        f"{arm} prediction hash mismatch"
    preds = json.load(open(pred_path))["videos"]
    assert set(preds.keys()) == set(vid_order), \
        f"{arm}: video set mismatch"
    rows = []       # per query: dict of hits
    video_of = []   # video index per query
    for vi, v in enumerate(vid_order):
        qs = preds[v]["queries"]
        gts = gt[v]
        assert len(qs) == len(gts), \
            f"{arm} {v}: {len(qs)} predictions vs {len(gts)} GT queries"
        for q, (gs, ge) in zip(qs, gts):
            segs = [p["segment"] for p in q["predictions"]]
            # predictions are stored score-ranked (top-5)
            if not segs:
                rows.append({k: 0.0 for k in KEYS})
            else:
                ious = [iou_seg(a, b, gs, ge) for a, b in segs[:5]]
                rows.append({
                    "r1_3": float(ious[0] >= 0.3),
                    "r1_5": float(ious[0] >= 0.5),
                    "r5_3": float(max(ious) >= 0.3),
                    "r5_5": float(max(ious) >= 0.5),
                })
            video_of.append(vi)
    return rows, np.array(video_of)


def main():
    dump = np.load(
        ROOT / "experiments/audit_ranking_bottleneck/prenms_dump.npz")
    vid_order = list(dict.fromkeys(dump["vid_ids"].tolist()))
    gt = {}
    for v in vid_order:
        m = dump["vid_ids"] == v
        gt[v] = list(zip(dump["gt_start"][m].tolist(),
                         dump["gt_end"][m].tolist()))
    hashes = json.loads(
        (ROOT / "experiments/qccl_arms/archive_hashes.json").read_text())

    data = {}
    for arm in ARMS:
        rows, video_of = load_arm(arm, vid_order, gt, hashes)
        data[arm] = (rows, video_of)
    n_q = len(data["G0"][0])
    n_v = len(vid_order)
    assert all(len(data[a][0]) == n_q for a in ARMS)
    per_video_nq = np.array([
        sum(1 for x in data["G0"][1] if x == vi) for vi in range(n_v)])
    print(f"query set: {n_v} videos / {n_q} queries "
          f"(identical across arms; hashes verified)")

    # ---- 1. official query-weighted metrics per arm ----
    metrics = {}
    for arm in ARMS:
        rows = data[arm][0]
        m = {k: 100 * float(np.mean([r[k] for r in rows]))
             for k in KEYS}
        m["Mean"] = float(np.mean([m[k] for k in KEYS]))
        metrics[arm] = m
        # cross-check against the evaluator's own metrics.json
        ref = json.loads((ROOT / f"experiments/qccl_arms/{arm}/evals/"
                         f"step8000/metrics.json").read_text())
        for k, name in NAMES.items():
            assert abs(m[k] - 100 * ref[name]) < 5e-3, \
                (arm, k, m[k], 100 * ref[name])
        assert abs(m["Mean"] - 100 * ref["Mean"]) < 5e-3
        print(f"{arm}: " + " ".join(
            f"{NAMES[k]}={m[k]:.2f}" for k in KEYS)
            + f" Mean={m['Mean']:.2f}  [matches metrics.json]")

    # ---- 2. observed differences: direct vs hit-matrix ----
    observed = {}
    for tag, (a, b) in {"G1-G0": ("G1", "G0"), "G2-G0": ("G2", "G0"),
                        "G2-G1": ("G2", "G1")}.items():
        obs = {}
        for k in KEYS:
            direct = metrics[a][k] - metrics[b][k]
            ha = np.array([r[k] for r in data[a][0]])
            hb = np.array([r[k] for r in data[b][0]])
            from_matrix = 100 * float((ha - hb).mean())
            assert abs(direct - from_matrix) < 1e-9, (tag, k)
            obs[k] = direct
        obs["Mean"] = metrics[a]["Mean"] - metrics[b]["Mean"]
        ha = np.mean([ [data[a][0][i][k] for k in KEYS] for i in range(n_q)], axis=0)
        hb = np.mean([ [data[b][0][i][k] for k in KEYS] for i in range(n_q)], axis=0)
        assert abs(obs["Mean"] - 100 * float((ha - hb).mean())) < 1e-9
        observed[tag] = obs
        print(f"{tag}: " + " ".join(
            f"{NAMES[k]}={obs[k]:+.2f}" for k in KEYS)
            + f" Mean={obs['Mean']:+.2f}")

    # ---- 3. video-clustered bootstrap, query-weighted ----
    rng = np.random.default_rng(20260910)
    B = 2000
    H = {arm: {k: np.array([r[k] for r in data[arm][0]]) for k in KEYS}
         for arm in ARMS}
    starts = np.concatenate([[0], np.cumsum(per_video_nq)[:-1]])
    cis = {}
    for tag, (a, b) in {"G1-G0": ("G1", "G0"), "G2-G0": ("G2", "G0"),
                        "G2-G1": ("G2", "G1")}.items():
        cis[tag] = {}
        for k in KEYS:
            diffs = []
            for _ in range(B):
                vids = rng.integers(0, n_v, n_v)
                idx = np.concatenate(
                    [np.arange(starts[v], starts[v] + per_video_nq[v])
                     for v in vids])
                # query-weighted over the resampled pool
                diffs.append(100 * float(
                    (H[a][k][idx] - H[b][k][idx]).mean()))
            cis[tag][k] = [float(np.percentile(diffs, 2.5)),
                           float(np.percentile(diffs, 97.5))]
        # Mean CI from per-query four-key means
        ma = np.mean(np.stack([H[a][k] for k in KEYS]), axis=0)
        mb = np.mean(np.stack([H[b][k] for k in KEYS]), axis=0)
        diffs = []
        for _ in range(B):
            vids = rng.integers(0, n_v, n_v)
            idx = np.concatenate(
                [np.arange(starts[v], starts[v] + per_video_nq[v])
                 for v in vids])
            diffs.append(100 * float((ma[idx] - mb[idx]).mean()))
        cis[tag]["Mean"] = [float(np.percentile(diffs, 2.5)),
                            float(np.percentile(diffs, 97.5))]
        print(f"{tag} CI95: " + " ".join(
            f"{NAMES[k]}=[{cis[tag][k][0]:+.2f},{cis[tag][k][1]:+.2f}]"
            for k in KEYS)
            + f" Mean=[{cis[tag]['Mean'][0]:+.2f},"
            f"{cis[tag]['Mean'][1]:+.2f}]")

    out = {
        "scoring": "official query-weighted; R1/R5 from stored top-5 "
                   "score-ranked predictions; no re-NMS",
        "n_videos": n_v, "n_queries": n_q,
        "arm_metrics": metrics,
        "observed_difference": observed,
        "bootstrap_ci95_video_clustered": cis,
        "bootstrap_B": B,
        "note": "CI describes validation-set sampling uncertainty of "
                "these fixed runs, not across-seed training uncertainty",
    }
    dest = ROOT / "experiments/qccl_arms/reconciliation_step8000.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print("->", dest)


if __name__ == "__main__":
    main()

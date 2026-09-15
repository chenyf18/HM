#!/usr/bin/env python3
"""Top-100 blind rerank dump for the QHNA recheck (HM-QHNA-028-R).

One val pass on the retrained A-U-clean checkpoint: for every query keep
the TRUE top-100 candidates by cls (no positive injection, no GT-based
selection), with FP32-decoded segments, cls logit/score, and the frozen
probe score computed on the fly from inference-available inputs
(F1 fused feature + value-projected pooled query). GT is never used in
the forward path; it is stored only for post-hoc evaluation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "hydra"), str(ROOT / "tools")]

import tools.qc_audit as qc  # noqa: E402
from tools.qc_audit import build_evaluator  # noqa: E402
from tools.qhna_audit import iou_seconds  # noqa: E402

RECHECK = ROOT / "experiments/qhna_recheck"
OUT_NPZ = RECHECK / "val_top100.npz"
PROBE_PT = RECHECK / "probe.pt"
TOPK = 100

_STASH = {}


def _pi(rec, pool):
    _STASH["pool"] = pool
    return iou_seconds(rec, pool)


def _bs(rng, ious):
    pool = _STASH.get("pool")
    if pool is None or pool.shape[0] == 0:
        return np.zeros(0, dtype=int)
    order = np.argsort(-pool[:, 1], kind="stable")
    return np.sort(order[:TOPK])


def main():
    RECHECK.mkdir(parents=True, exist_ok=True)
    qc.QROOT = RECHECK
    models_link = RECHECK / "models"
    if models_link.is_symlink():
        models_link.unlink()
    models_link.symlink_to(
        ROOT / "experiments/allocator_diagnosis/seed_1/AUC/models",
        target_is_directory=True)
    for split in ("val",):
        f = RECHECK / f"opt_{split}.yaml"
        if f.exists():
            f.unlink()
        import shutil
        shutil.copyfile(
            ROOT / "experiments/allocator_diagnosis/seed_1/AUC_P32/opt.yaml",
            f)

    qc.iou_seconds = _pi
    qc.balanced_select = _bs

    probe = torch.nn.Sequential(
        torch.nn.Linear(768, 128), torch.nn.GELU(),
        torch.nn.Linear(128, 1)).cuda().eval()
    probe.load_state_dict(
        torch.load(PROBE_PT, map_location="cpu",
                   weights_only=False)["state"])

    ev = build_evaluator("val")
    ev.run()
    recs = ev.records
    n = len(recs)
    lengths = [r["sample_pool"].shape[0] for r in recs]
    arrays = {
        "rows": np.concatenate([r["sample_pool"] for r in recs]),
        "ious": np.concatenate([r["sample_ious"] for r in recs]),
        "qi": np.concatenate([
            np.full(lengths[i], i, dtype=np.int64)
            for i in range(n)]),
    }
    probe_scores = []
    for i, r in enumerate(recs):
        if r["sample_F1"].shape[0] == 0:
            probe_scores.append(np.zeros(0, np.float32))
            continue
        with torch.no_grad():
            f1 = torch.tensor(r["sample_F1"], dtype=torch.float32).cuda()
            q = torch.tensor(r["q384"], dtype=torch.float32).cuda()
            q = q[None].expand(f1.size(0), -1)
            x = torch.cat([f1, q], dim=1)
            probe_scores.append(
                torch.sigmoid(probe(x)).squeeze(-1).cpu().numpy()
                .astype(np.float32))
    arrays["probe_score"] = np.concatenate(probe_scores)
    for k in ("gt_start", "gt_end", "clip_stride", "clip_size", "fps",
              "duration"):
        arrays[k] = np.array([r[k] for r in recs], dtype=np.float64)
    np.savez_compressed(OUT_NPZ, **arrays)
    print(f"top100 dump: {sum(lengths)} rows / {n} queries -> {OUT_NPZ}")
    print(f"pool oracle@0.5 = {float((arrays['ious'] >= 0.5).any() and np.mean([ (r>=0.5).any() for r in [arrays['ious'][arrays['qi']==i] for i in range(n)]])):.4f}")


if __name__ == "__main__":
    main()

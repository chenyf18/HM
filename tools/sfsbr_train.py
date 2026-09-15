#!/usr/bin/env python3
"""SFSBR B/C training + full-val evaluation (HM-SFSBR-031 rev1).

Locked protocol:
  - single pass over the same train query set, query-level loss
    normalisation (empty-set terms contribute zero; queries with no
    reachable endpoint still contribute the unreachable regulariser)
  - AdamW lr=1e-4, wd=0 (refiner only), seed=0, data order = loader
  - arms B/C share ONE identical initial parameter copy (state_dict
    clone), not just the same seed
  - loss: SmoothL1(delta/R, target/R) on reachable endpoints
    + 0.05 * mean((delta/R)^2) on unreachable endpoints
  - final model = single-pass end (no best-checkpoint selection)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "hydra"), str(ROOT / "tools")]

from libs.modeling.sfsbr import (  # noqa: E402
    SFSBRRefiner, gather_window, supervision_targets,
)
from tools.sfsbr_lib import (  # noqa: E402
    build_frozen_evaluator, fine_matrix, TOPK_REFINE, SROOT,
)

R = 2.0
REG = 0.05
LR = 1e-4
WD = 0.0
SEED = 0


def query_loss(delta, m_s, m_e, ds, de):
    """Per-query normalised loss (scalar) using per-endpoint masks."""
    dn = delta / R
    reach = torch.stack([m_s, m_e], dim=1).bool().to(delta.device)
    tgt = (torch.stack([ds, de], dim=1) / R).to(delta.device)
    if reach.any():
        l1 = torch.nn.functional.smooth_l1_loss(dn[reach], tgt[reach])
    else:
        l1 = None
    if (~reach).any():
        l2 = (dn[~reach] ** 2).mean()
    else:
        l2 = None
    loss = 0.0 * dn.sum()
    if l1 is not None:
        loss = loss + l1
    if l2 is not None:
        loss = loss + REG * l2
    return loss, int(reach.sum()), int((~reach).sum())


def run_train(arm):
    torch.manual_seed(SEED)
    ev = build_frozen_evaluator("train", arm)
    refiner = SFSBRRefiner().cuda().train()
    init_path = SROOT / "shared_init.pt"
    if arm == "B":
        torch.save(refiner.state_dict(), init_path)
    else:
        refiner.load_state_dict(torch.load(init_path,
                                           weights_only=False))
    opt = torch.optim.AdamW(refiner.parameters(), lr=LR,
                            weight_decay=WD)
    n_steps = 0
    n_query = 0
    reach_eps = unreach_eps = 0
    delta_abs = []
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    for data_list in ev.dataloader:
        data = data_list[0]
        ev.predict(data)
        for rec in ev.records:
            pool = rec["pool"]
            n_query += 1
            if pool.shape[0] == 0:
                continue
            order = np.argsort(-pool[:, 0], kind="stable")[:TOPK_REFINE]
            centers = pool[order, 2]
            scales = pool[order, 3]
            st = torch.tensor(centers - pool[order, 4] * scales)
            en = torch.tensor(centers + pool[order, 5] * scales)
            gt_s = torch.full_like(st, (
                rec["gt"][0] * rec["fps"]
                - 0.5 * rec["clip_size"]) / rec["clip_stride"])
            gt_e = torch.full_like(en, (
                rec["gt"][1] * rec["fps"]
                - 0.5 * rec["clip_size"]) / rec["clip_stride"])
            m_s, m_e, ds, de = supervision_targets(st, en, gt_s, gt_e)
            fine = fine_matrix(rec["vid_id"]) if arm == "C" else None
            sw = gather_window(st * 2.0, rec["raw_vid"],
                               fine).cuda()
            ew = gather_window(en * 2.0, rec["raw_vid"],
                               fine).cuda()
            q = torch.from_numpy(np.broadcast_to(
                rec["q384"], (len(order), 384)).copy()).cuda()
            opt.zero_grad(set_to_none=True)
            delta = refiner(rec["F1"][order].cuda(), q, sw, ew)
            loss, nr, nu = query_loss(delta, m_s, m_e, ds, de)
            loss.backward()
            opt.step()
            n_steps += 1
            reach_eps += nr
            unreach_eps += nu
            delta_abs.append(float(delta.detach().abs().mean()))
        ev.records = []
    out_dir = SROOT / arm
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(refiner.state_dict(), out_dir / "refiner_final.pt")
    stats = {
        "arm": arm, "queries": n_query, "optimizer_steps": n_steps,
        "reachable_endpoints": reach_eps,
        "unreachable_endpoints": unreach_eps,
        "mean_delta_abs_final20": float(np.mean(delta_abs[-20:])),
        "wall_time_s": time.time() - t0,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "locked": {"lr": LR, "wd": WD, "seed": SEED, "R": R, "reg": REG,
                   "topk": TOPK_REFINE, "budget": "single pass"},
    }
    (out_dir / "train_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["B", "C"], required=True)
    run_train(ap.parse_args().arm)

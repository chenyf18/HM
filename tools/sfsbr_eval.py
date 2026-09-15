#!/usr/bin/env python3
"""SFSBR full-val evaluation (HM-SFSBR-031 rev1).

Applies the trained refiner to the cls top-50 of each query's pool
(unchanged pool/scores/NMS), then runs the official Soft-NMS pipeline
unchanged. Reports official four metrics + Mean for A (=R1 archived, no
forward), B, C, plus auxiliary per-candidate diagnostics.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "hydra"), str(ROOT / "tools")]

from libs.modeling.sfsbr import (  # noqa: E402
    SFSBRRefiner, gather_window, apply_delta,
)
from tools.sfsbr_lib import (  # noqa: E402
    build_frozen_evaluator, fine_matrix, TOPK_REFINE, SROOT,
)
from tools.audit_ranking_bottleneck import soft_nms, iou_1d  # noqa: E402


def to_sec(tok, rec):
    cs, cc, fps, dur = (rec["clip_stride"], rec["clip_size"],
                        rec["fps"], rec["duration"])
    return np.clip((tok * cs + 0.5 * cc) / fps, 0, dur)


def run_eval(arm):
    ev = build_frozen_evaluator("val", arm)
    refiner = SFSBRRefiner().cuda().eval()
    refiner.load_state_dict(torch.load(
        SROOT / arm / "refiner_final.pt", weights_only=False))
    counts = np.zeros(4)
    n_q = 0
    aux = {"n_refined": 0, "iou_before": [], "iou_after": [],
           "degraded": 0, "delta_abs": [], "short_before": [],
           "short_after": [], "reach_at_eval": 0, "endpoints": 0}
    t0 = time.time()
    hit_rows = []
    torch.cuda.reset_peak_memory_stats()
    for data_list in ev.dataloader:
        data = data_list[0]
        results = ev.predict(data)          # official pipeline (untouched)
        for i, rec in enumerate(ev.records):
            pool = rec["pool"]
            n_q += 1
            result = results[i]
            segs = result["segments"].clone()
            scores = result["scores"].clone()
            if pool.shape[0] == 0:
                # no candidates: keep official output as-is
                counts += evaluate(segs, scores, rec)
                continue
            order = np.argsort(-pool[:, 0], kind="stable")
            centers = pool[:, 2]
            scales = pool[:, 3]
            st_all = torch.tensor(centers - pool[:, 4] * scales)
            en_all = torch.tensor(centers + pool[:, 5] * scales)
            top = order[:TOPK_REFINE]
            fine = fine_matrix(rec["vid_id"]) if arm == "C" else None
            with torch.no_grad():
                sw = gather_window(st_all[top] * 2.0, rec["raw_vid"],
                                   fine).cuda()
                ew = gather_window(en_all[top] * 2.0, rec["raw_vid"],
                                   fine).cuda()
                q = torch.from_numpy(np.broadcast_to(
                    rec["q384"], (len(top), 384)).copy()).cuda()
                delta = refiner(rec["F1"][top].cuda(), q, sw, ew).cpu()
            t_valid = float(rec["raw_vid"].size(-1))
            s2, e2 = apply_delta(st_all[top], en_all[top], delta,
                                 t_valid)
            # per-candidate diagnostics on the refined top-50
            iou_b = iou_1d(to_sec(st_all[top].numpy(), rec),
                           to_sec(en_all[top].numpy(), rec),
                           rec["gt"][0], rec["gt"][1])
            iou_a = iou_1d(to_sec(s2.numpy(), rec),
                           to_sec(e2.numpy(), rec),
                           rec["gt"][0], rec["gt"][1])
            aux["n_refined"] += len(top)
            aux["iou_before"].append(iou_b.max() if len(iou_b) else 0.0)
            aux["iou_after"].append(iou_a.max() if len(iou_a) else 0.0)
            aux["degraded"] += int((iou_a < iou_b - 1e-6).sum())
            aux["delta_abs"].append(float(delta.abs().mean()))
            gt_dur = rec["gt"][1] - rec["gt"][0]
            if gt_dur <= 2.3:      # short tercile boundary (028)
                aux["short_before"].append(
                    iou_b.max() if len(iou_b) else 0.0)
                aux["short_after"].append(
                    iou_a.max() if len(iou_a) else 0.0)
            # rerun the NMS pipeline on the pool with top-50 boundaries
            # replaced (scores unchanged, same NMS parameters)
            pool_st = st_all.numpy().copy()
            pool_en = en_all.numpy().copy()
            pool_st[top] = s2.numpy()
            pool_en[top] = e2.numpy()
            g_order = np.argsort(-pool[:, 0], kind="stable")[:2000]
            k_segs, k_scores = soft_nms(np.stack(
                [pool_st[g_order], pool_en[g_order]], axis=1),
                torch.tensor(pool[g_order, 0]))
            # token -> seconds before scoring (official convention)
            sec = np.stack([to_sec(k_segs[:, 0], rec),
                            to_sec(k_segs[:, 1], rec)], axis=1)
            c4 = evaluate(
                torch.from_numpy(sec), torch.from_numpy(k_scores),
                rec)
            counts += c4
            hit_rows.append(c4)
        ev.records = []
    metrics = {
        "Rank@1_IoU@0.3": float(counts[0] / n_q),
        "Rank@1_IoU@0.5": float(counts[1] / n_q),
        "Rank@5_IoU@0.3": float(counts[2] / n_q),
        "Rank@5_IoU@0.5": float(counts[3] / n_q),
    }
    metrics["Mean"] = sum(metrics.values()) / 4
    aux_out = {
        "n_queries": n_q,
        "n_refined_top50": aux["n_refined"],
        "degraded_candidates": aux["degraded"],
        "mean_delta_abs": float(np.mean(aux["delta_abs"])),
        "oracle_iou_top50_before": float(
            np.mean(aux["iou_before"])),
        "oracle_iou_top50_after": float(np.mean(aux["iou_after"])),
        "short_oracle_before": float(np.mean(aux["short_before"]))
        if aux["short_before"] else None,
        "short_oracle_after": float(np.mean(aux["short_after"]))
        if aux["short_after"] else None,
        "wall_time_s": time.time() - t0,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    print(f"[{arm}] " + " ".join(
        f"{k}={100*v:.2f}" for k, v in metrics.items()))
    print(json.dumps(aux_out, indent=2))
    np.savez_compressed(SROOT / arm / "per_query_hits.npz",
                        hits=np.stack(hit_rows))  # (n_q, 4)
    out = SROOT / arm / "eval_final.json"
    out.write_text(json.dumps(
        {"metrics": metrics, "aux": aux_out}, indent=2) + "\n")
    return metrics, aux_out


def evaluate(segs, scores, rec):
    """Official R@K counting for one query."""
    if len(segs) == 0:
        return np.zeros(4)
    order = np.argsort(-scores.numpy(), kind="stable")[:5]
    s = segs.numpy()[order]
    iou = iou_1d(s[:, 0], s[:, 1], rec["gt"][0], rec["gt"][1])
    c = np.zeros(4)
    c[0] = (iou[0] >= 0.3) if len(iou) else 0
    c[1] = (iou[0] >= 0.5) if len(iou) else 0
    c[2] = (iou.max() >= 0.3) if len(iou) else 0
    c[3] = (iou.max() >= 0.5) if len(iou) else 0
    return c


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["B", "C"], required=True)
    run_eval(ap.parse_args().arm)

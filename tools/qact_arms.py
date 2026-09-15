#!/usr/bin/env python3
"""HM-QACT-033 three-arm experiment: quality-aware cls targets.

G0 binary / G1 decoded_iou / G2 metric_utility. The ONLY difference is
opt['train']['cls_target_type']. Common start = R1 model_ema branch,
constant lr 1e-4 (8000-step preregistered short protocol), identical
data order/seed/optimizer, eval at 0/2000/4000/8000 with per-eval
candidate-level diagnostics (six metrics) + training-time cls-target
monitoring (histogram / positive mass / cls-head grad norm).
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "hydra"), str(ROOT / "tools")]

import tools.run_formal_ablation as rfa  # noqa: E402
from tools.audit_ranking_bottleneck import rankdata_avg, corr  # noqa: E402
from libs.modeling.temporal_coordinates import decode_offsets  # noqa: E402

R1_OPT = ROOT / "experiments/allocator_diagnosis/seed_1/AUC/opt.yaml"
R1_CKPT = ROOT / "experiments/allocator_diagnosis/seed_1/AUC/models/last.pth"
AROOT = ROOT / "experiments/qact_arms"
EVAL_STEPS = (0, 2000, 4000, 6000, 8000)
MAX_STEPS = 8000
LOCKED_LR = 1e-4
MON_EVERY = 250
HIST_EDGES = (0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.01)


def build_arm_trainer(arm):
    from libs import load_opt
    root = AROOT / arm
    root.mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(exist_ok=True)
    (root / "states").mkdir(exist_ok=True)
    opt = load_opt(str(R1_OPT), is_training=True)
    opt["train"]["cls_target_type"] = {
        "G0": "binary", "G1": "decoded_iou",
        "G2": "metric_utility"}[arm]
    opt["seed"] = 1
    opt["_root"] = str(root)
    opt["_resume"] = False
    opt["_distributed"] = False
    opt["_world_size"] = 1

    monitor = {"records": [], "grad_norms": []}

    class QactTrainer(rfa.FormalTrainer):
        def __init__(self, opt, label, root, max_steps):
            self._monitor = monitor
            super().__init__(opt, label, root, max_steps=max_steps,
                             precision_policy="p3")

        def _install(self):
            ckpt = torch.load(R1_CKPT, map_location="cpu",
                              weights_only=False)
            self.model.load_compatible_state_dict(ckpt["model_ema"])
            self._ema_init()
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lambda e: 1.0)
            for g in self.optimizer.param_groups:
                g["lr"] = LOCKED_LR
                g["initial_lr"] = LOCKED_LR
            orig_focal = self._calc_focal_loss
            self._orig_focal = orig_focal
            trainer = self

            def focal_rec(logits, labels):
                ret = orig_focal(logits, labels)
                if trainer.itr % MON_EVERY == 0:
                    lab = labels.detach().float()
                    trainer._monitor["records"].append({
                        "itr": trainer.itr,
                        "n": int(lab.numel()),
                        "positive_mass": float(lab.sum()),
                        "mean": float(lab.mean()),
                        "frac_gt": {
                            f"{HIST_EDGES[i]:.2f}-{HIST_EDGES[i+1]:.2f}":
                            float(((lab >= HIST_EDGES[i])
                                   & (lab < HIST_EDGES[i + 1])).float()
                                  .mean())
                            for i in range(len(HIST_EDGES) - 1)},
                    })
                return ret

            self._calc_focal_loss = focal_rec
            orig_step = self.optimizer.step

            def step_rec(*a, **k):
                out = orig_step(*a, **k)
                if trainer.itr % MON_EVERY == 0:
                    g2 = math.sqrt(sum(
                        (p.grad.detach().float() ** 2).sum()
                        for n, p in self.model.named_parameters()
                        if p.grad is not None and n.startswith("cls_head")))
                    trainer._monitor["grad_norms"].append(
                        {"itr": trainer.itr, "cls_head_grad_norm": g2})
                return out

            self.optimizer.step = step_rec
            orig_jsonl = self._write_jsonl

            def jsonl_rec(record):
                orig_jsonl(record)
                if trainer.itr in EVAL_STEPS and trainer.itr > 0:
                    trainer.eval_at(trainer.itr)

            self._write_jsonl = jsonl_rec

        def eval_at(self, step):
            mroot = Path(self.opt["_root"])
            ckpt_path = mroot / "models" / f"step{step}.pth"
            torch.save({"model": self._unwrap(self.model).state_dict(),
                        "model_ema": self.model_ema.state_dict()},
                       ckpt_path)
            eroot = mroot / "evals" / f"step{step}"
            (eroot / "models").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ckpt_path,
                            eroot / "models" / f"step{step}.pth")
            opt = load_opt(str(R1_OPT), is_training=False)
            opt["_root"] = str(eroot)
            opt["_ckpt"] = f"step{step}"
            metrics = run_diag_eval(opt, eroot)
            (eroot / "metrics.json").write_text(
                json.dumps(metrics, indent=2) + "\n")
            print(f"[{self.formal_label}] step {step}: "
                  + " ".join(f"{k}={100*v:.2f}" for k, v in
                             metrics["official"].items()),
                  flush=True)

    t = QactTrainer(opt, f"QACT_{arm}", root, max_steps=MAX_STEPS)
    t._install()
    return t, monitor


# ---------------------------------------------------------------------------
# diagnostic evaluation (official metrics + candidate dump + 6 metrics)
# ---------------------------------------------------------------------------

def run_diag_eval(opt, eroot):
    class DiagEval(rfa.FormalEvaluator):
        def __init__(self, opt):
            self.recs = []
            self._collect_log = []
            super().__init__(opt, precision_policy="p3")
            orig_collect = self._collect_segments

            def wrap(fpn_points, fpn_logits, fpn_offsets, fpn_masks,
                     ext_scores=None, return_levels=False,
                     fpn_feats=None, query_idx=0):
                res = orig_collect(
                    fpn_points, fpn_logits, fpn_offsets, fpn_masks,
                    ext_scores, return_levels=return_levels,
                    fpn_feats=fpn_feats, query_idx=query_idx)
                per_level = []
                for level in range(len(fpn_logits)):
                    pts = fpn_points[level]
                    if pts.ndim == 3:
                        pts = pts[0]
                    lg = fpn_logits[level]
                    while lg.ndim > 1:
                        lg = lg[0]
                    of = fpn_offsets[level]
                    while of.ndim > 2:
                        of = of[0]
                    mk = fpn_masks[level]
                    while mk.ndim > 1:
                        mk = mk[0]
                    sc = torch.sigmoid(lg) * mk.float()
                    keep = sc > self.pre_nms_thresh
                    per_level.append({
                        "level": level,
                        "score": sc[keep].detach().float().cpu(),
                        "points": pts[keep].detach().float().cpu(),
                        "offset": of[keep].detach().float().cpu(),
                    })
                self._collect_log.append(per_level)
                return res

            self._collect_segments = wrap

        def predict(self, data):
            tokens = data["text"]
            if not isinstance(tokens, tuple):
                tokens = (tokens, )
            self._collect_log = []
            results = super().predict(data)
            if len(self._collect_log) != len(results):
                raise RuntimeError("collect/queries mismatch")
            for q, res in enumerate(results):
                per_level = self._collect_log[q]
                rows = []
                for pl in per_level:
                    n_k = pl["score"].numel()
                    e = np.empty((n_k, 5), dtype=np.float32)
                    e[:, 0] = pl["score"].numpy()
                    e[:, 1] = pl["level"]
                    e[:, 2] = pl["points"][:, 0].numpy()
                    e[:, 3] = pl["points"][:, 1].numpy()   # reg_min
                    e[:, 4] = pl["points"][:, 2].numpy()   # reg_max
                    rows.append((e, pl["points"], pl["offset"]))
                gt = np.asarray(data["segment"][q], dtype=np.float64)
                # token-frame GT (target convention: no vid_stride div)
                tgt_tok = data["target"][q].float().numpy()
                self.recs.append({
                    "rows": rows, "gt": gt, "gt_tok": tgt_tok,
                    "gt_dur": gt[1] - gt[0],
                })
            return results

    ev = DiagEval(opt)
    ev.run()
    official = {
        f"Rank@{r}_IoU@{t:.1f}":
            float(ev.counts[i][j] / ev.text_cnt)
        for i, r in enumerate(ev.ranks)
        for j, t in enumerate(ev.iou_threshs)}
    official["Mean"] = sum(official.values()) / 4

    diag = compute_diagnostics(ev.recs)
    return {"official": official, "diagnostics": diag}


def binary_label(points_row, tgt, radius_mult=1.5):
    """Trainer rule replay: center-in-radius-window AND in reg range."""
    c, rmin, rmax = points_row[0], points_row[1], points_row[2]
    ctr = 0.5 * (tgt[0] + tgt[1])
    radius = points_row[3] * radius_mult if len(points_row) > 3 else rmax
    radius = radius * radius_mult  # points col3 is scale
    t_min = max(ctr - radius, tgt[0])
    t_max = min(ctr + radius, tgt[1])
    inside = (c - t_min) > 0 and (t_max - c) > 0
    d = max(tgt[0] - c, c - tgt[1]) if False else max(
        tgt[0] - c, tgt[1] - c) * -1  # placeholder, fixed below
    pt2start = c - tgt[0]
    pt2end = tgt[1] - c
    max_reg = max(pt2start, pt2end)
    in_range = rmin <= max_reg < rmax
    return bool(inside and in_range)


def compute_diagnostics(recs):
    spear_qs = []
    best_rank = []
    top1_ious = []
    conflictA = conflictB = 0
    tot = 0
    bin_pos = 0
    hn_pairs_win = hn_pairs_tot = 0
    hn_pairs_band = {}
    calib_scores = []
    calib_ious = []
    for rec in recs:
        tgt = rec["gt_tok"]
        all_rows = []
        for e, pts, offs in rec["rows"]:
            n_k = e.shape[0]
            if n_k == 0:
                continue
            seg = decode_offsets(pts, offs).numpy()
            s1 = np.maximum(0, np.minimum(seg[:, 0], tgt[1])
                            - 0)  # full overlap calc below
            inter = np.maximum(0, np.minimum(seg[:, 1], tgt[1])
                               - np.maximum(seg[:, 0], tgt[0]))
            union = (seg[:, 1] - seg[:, 0]) + (tgt[1] - tgt[0]) - inter
            iou = np.where(union > 0, inter / np.maximum(union, 1e-12),
                           0.0)
            pr = pts.numpy()
            scale = pr[:, 3]
            ctr = 0.5 * (tgt[0] + tgt[1])
            rad = scale * 1.5
            t_min = np.maximum(ctr - rad, tgt[0])
            t_max = np.minimum(ctr + rad, tgt[1])
            inside = (pr[:, 0] - t_min > 0) & (t_max - pr[:, 0] > 0)
            maxreg = np.maximum(tgt[0] - pr[:, 0], tgt[1] - pr[:, 0])
            inrange = (pr[:, 1] <= maxreg) & (maxreg < pr[:, 2])
            binlbl = inside & inrange
            all_rows.append(np.stack([
                e[:, 0], iou, binlbl.astype(np.float32),
                seg[:, 1] - seg[:, 0]], axis=1))
        if not all_rows:
            continue
        R = np.concatenate(all_rows)
        if R.shape[0] < 3:
            continue
        sc, iou, bl = R[:, 0], R[:, 1], R[:, 2]
        order = np.argsort(-sc, kind="stable")
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        # 1) per-query spearman
        sp = corr(rankdata_avg(sc), rankdata_avg(iou))
        if not math.isnan(sp):
            spear_qs.append(sp)
        # 2) best-IoU rank
        bi = np.where(iou >= iou.max() - 1e-9)[0]
        best_rank.append(int(rank[bi].min()) + 1)
        # 3) top-1 quality
        top1_ious.append(float(iou[order[0]]))
        # 4) conflict rates
        conflictA += int(((bl == 0) & (iou >= 0.5)).sum())
        conflictB += int(((bl == 1) & (iou < 0.3)).sum())
        bin_pos += int(bl.sum())
        tot += len(bl)
        # 5) HN pairwise (028 口径: pos∈top50 IoU>=.5 vs HN rank<=20 IoU<.3)
        top50 = order[:50]
        pos_i = top50[iou[top50] >= 0.5]
        hn_i = order[:20][iou[order[:20]] < 0.3]
        band = ("short" if rec["gt_dur"] <= 2.3 else
                "medium" if rec["gt_dur"] <= 6.5 else "long")
        hb = hn_pairs_band.setdefault(band, [0, 0])
        for a in pos_i:
            for b in hn_i:
                hn_pairs_tot += 1
                hb[1] += 1
                w = float(sc[a] > sc[b]) + 0.5 * float(sc[a] == sc[b])
                hn_pairs_win += w
                hb[0] += w
        # 6) calibration
        calib_scores.append(sc)
        calib_ious.append(iou)
    top1 = np.array(top1_ious)
    br = np.array(best_rank)
    all_sc = np.concatenate(calib_scores)
    all_io = np.concatenate(calib_ious)
    calib = []
    b_idx = np.argsort(all_sc, kind="stable")
    splits = np.array_split(b_idx, 10)
    for si, idxs in enumerate(splits):
        if len(idxs) == 0:
            continue
        calib.append({
            "bucket": si,
            "mean_score": float(all_sc[idxs].mean()),
            "mean_iou": float(all_io[idxs].mean()),
            "n": int(len(idxs))})
    # bootstrap over queries for spearman
    rng = np.random.default_rng(0)
    sq = np.array(spear_qs)
    boots = [float(sq[rng.choice(len(sq), len(sq), replace=True)].mean())
             for _ in range(500)] if len(sq) else [float("nan")]
    return {
        "spearman_query_mean": float(np.mean(sq)) if len(sq) else None,
        "spearman_boot_ci95": [float(np.percentile(boots, 2.5)),
                               float(np.percentile(boots, 97.5))],
        "best_iou_rank_mean": float(br.mean()),
        "best_iou_rank_median": float(np.median(br)),
        "top1_iou_mean": float(top1.mean()),
        "top1_iou_median": float(np.median(top1)),
        "top1_R03": float((top1 >= 0.3).mean()),
        "top1_R05": float((top1 >= 0.5).mean()),
        "conflictA_lbl0_iouGe.5": conflictA / max(tot, 1),
        "conflictB_lbl1_iouLt.3": conflictB / max(tot, 1),
        "binary_pos_rate": bin_pos / max(tot, 1),
        "hn_pairwise_acc": hn_pairs_win / max(hn_pairs_tot, 1),
        "hn_pairwise_n": hn_pairs_tot,
        "hn_pairwise_by_band": {
            k: v[0] / max(v[1], 1) for k, v in hn_pairs_band.items()},
        "calibration": calib,
        "n_queries": len(recs),
    }


def run_train(arm):
    t, monitor = build_arm_trainer(arm)
    t.model.train()
    t.eval_at(0)
    t.run()
    out = Path(t.opt["_root"])
    (out / "monitor.json").write_text(json.dumps({
        "cls_target_records": monitor["records"],
        "cls_grad_norms": monitor["grad_norms"]}, indent=2) + "\n")
    print(f"arm {arm} done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["G0", "G1", "G2"], required=True)
    args = ap.parse_args()
    run_train(args.arm)


if __name__ == "__main__":
    main()

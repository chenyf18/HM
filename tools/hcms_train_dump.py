#!/usr/bin/env python3
"""HM-HCMS-001 stage 1: pre-NMS candidate dump over the TRAIN split.

Uses the frozen A-U-clean-R1 backbone with the same collect wrapper
family as prior audits (FP32 decode, official thresholds), storing for
every query the top-500 candidates by cls score BEFORE NMS with all
required fields. GT is used only to compute the stored IoU column.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "hydra"), str(ROOT / "tools")]

import tools.run_formal_ablation as rfa  # noqa: E402
from libs.modeling.temporal_coordinates import decode_offsets  # noqa: E402
from tools.audit_ranking_bottleneck import iou_1d  # noqa: E402

OUT = ROOT / "experiments/hcms_pool/train_candidates.jsonl"
TOP_DUMP = 500


def main():
    import shutil
    from libs import load_opt

    work = ROOT / "experiments/hcms_pool/backbone_train"
    work.mkdir(parents=True, exist_ok=True)
    models_link = work / "models"
    if models_link.is_symlink() or models_link.exists():
        models_link.unlink()
    models_link.symlink_to(
        ROOT / "experiments/allocator_diagnosis/seed_1/AUC/models",
        target_is_directory=True)
    opt_path = work / "opt_train.yaml"
    if opt_path.exists():
        opt_path.unlink()
    shutil.copyfile(
        ROOT / "experiments/allocator_diagnosis/seed_1/AUC/opt.yaml",
        opt_path)
    opt = load_opt(str(opt_path), is_training=False)
    opt["eval"]["data"]["split"] = "train"
    opt["_root"] = str(work)
    opt["_ckpt"] = "last"

    class DumpEval(rfa.FormalEvaluator):
        def __init__(self, opt):
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
            # window offset 0 (single window; Ego4D grid == target frame)
            for q, per_level in enumerate(self._collect_log):
                rows = []
                for pl in per_level:
                    n_k = pl["score"].numel()
                    if n_k == 0:
                        continue
                    pts = pl["points"]
                    seg = decode_offsets(pts, pl["offset"])
                    cs = float(data["clip_stride"])
                    cc = float(data["clip_size"])
                    fps = float(data["fps"])
                    dur = float(data["duration"])
                    s_sec = np.asarray(np.clip(
                        (seg[:, 0].numpy() * cs + 0.5 * cc) / fps, 0, dur), dtype=float)
                    e_sec = np.asarray(np.clip(
                        (seg[:, 1].numpy() * cs + 0.5 * cc) / fps, 0, dur), dtype=float)
                    gt = np.asarray(data["segment"][q], dtype=np.float64)
                    iou = iou_1d(s_sec, e_sec, gt[0], gt[1])
                    for j in range(n_k):
                        rows.append((float(pl["score"][j]), s_sec[j],
                                     e_sec[j], float(iou[j]),
                                     int(pl["level"]),
                                     float(pts[j, 0])))
                rows.sort(key=lambda r: -r[0])
                with open(OUT, "a") as f:
                    for rank, r in enumerate(rows[:TOP_DUMP]):
                        f.write(json.dumps({
                            "vid_id": str(data["vid_id"]),
                            "query_idx": q,
                            "score": r[0], "start": r[1], "end": r[2],
                            "center": 0.5 * (r[1] + r[2]),
                            "duration": r[2] - r[1],
                            "iou": r[3], "level": r[4],
                            "center_grid": r[5], "rank": rank,
                        }) + "\n")
            return results

    ev = DumpEval(opt)
    ev.run()
    print("dump ->", OUT)


if __name__ == "__main__":
    main()

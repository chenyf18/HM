#!/usr/bin/env python3
"""HM-HCMS-001 stage 5: counterfactual equivariance diagnostic.

For a held-out probe set, run the SAME model on (V, q) and (V_cf, q)
where V_cf moves the GT event to a hard-negative window. Records:
  - center shift error: pred_shift - GT_shift
  - boundary shift error (start/end)
  - IoU(pred_cf, moved_GT) vs IoU(pred_orig, orig_GT)
  - prediction shift distance vs GT shift distance
Model untouched (frozen eval); GT only builds the swap + scores.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "hydra"), str(ROOT / "tools")]

from tools.audit_ranking_bottleneck import iou_1d, soft_nms  # noqa: E402


def run_equivariance(ckpt_dir, ckpt_name, out_json, n_videos=60,
                     device="cuda"):
    import tools.run_formal_ablation as rfa
    from libs import load_opt
    import shutil

    work = ROOT / "experiments/hcms_pool/equiv_probe"
    (work / "models").mkdir(parents=True, exist_ok=True)
    dst = work / "models" / f"{ckpt_name}.pth"
    src = ckpt_dir / "models" / f"{ckpt_name}.pth"
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if not src.exists():
        raise FileNotFoundError(src)
    dst.symlink_to(src.resolve())
    opt_path = work / "opt_val.yaml"
    if opt_path.exists():
        opt_path.unlink()
    shutil.copyfile(
        ROOT / "experiments/allocator_diagnosis/seed_1/AUC/opt.yaml",
        opt_path)
    opt = load_opt(str(opt_path), is_training=False)
    opt["_root"] = str(work)
    opt["_ckpt"] = ckpt_name

    rng = random.Random(0)
    recs = []
    ev = rfa.FormalEvaluator(opt, precision_policy="p3")
    # wrap _collect_segments to capture pools, and after each predict
    # run a swapped forward manually on CPU-side features? we instead
    # replay: for each video take one query, build V_cf at feature
    # level from the SAME loaded features via the dataset
    from libs.data import make_dataset
    opt2 = load_opt(str(opt_path), is_training=False)
    opt2["eval"]["data"]["split"] = "val"
    ds = make_dataset(opt2["eval"]["data"], is_training=False)
    from libs.data.hcms import select_negative_random, moment_swap
    n_done = 0
    for i in range(len(ds)):
        if n_done >= n_videos:
            break
        item = ds[i]
        vid = item["vid"]                    # (C, T) grid feats
        for q in range(len(item["text"])):
            gt = item["target"][q].tolist()
            others = [item["target"][j].tolist()
                      for j in range(len(item["text"])) if j != q]
            neg = select_negative_random(
                gt, vid.size(1), others, len_mode="relaxed", rng=rng)
            if neg is None:
                continue
            vid_cf, tgt_cf, ok, ratio = moment_swap(
                vid[None], item["target"].clone(), q,
                (neg[0], neg[1]), neg[2])
            if not ok:
                continue
            # forward both through the frozen model
            with torch.no_grad(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16):
                def top1_seg(v, t_q):
                    m = ev.model
                    text = item["text"][q][None].float().cuda()
                    tm = torch.ones(1, 1, text.size(-1),
                                    dtype=torch.bool).cuda()
                    T, TM = m.encode_text(text, tm)
                    v = v.float().cuda()[None]        # (1, C_raw, T)
                    vm = torch.ones(1, 1, v.size(-1),
                                    dtype=torch.bool).cuda()
                    # early fusion path (official eval order)
                    wp, wpm = m.vid_proj(v, vm)
                    fused, fm_ = m.fusion(wp, wpm, T, TM, None)
                    qf = T if m.query_conditioned else None
                    qm = TM if m.query_conditioned else None
                    fpn, fm, _, _ = m.encode_video(
                        fused, fm_, query_feat=qf, query_mask=qm)
                    lg, _, off, _ = m.fuse_and_predict(fpn, fm, T, TM)
                    lg = torch.cat(lg, 1)[0]
                    off = torch.cat(off, 1)[0]
                    sc = torch.sigmoid(lg)
                    j = int(sc.argmax())
                    from libs.modeling.temporal_coordinates import (
                        decode_offsets as dec)
                    if ev.adaptive_anchor:
                        mref = ev.model
                        pts = ev.adaptive_pt_gen(
                            mref.vid_net.last_temporal_metadata, fm)
                        P = torch.cat(pts, 1)[0][j]
                    else:
                        pts = ev.pt_gen([mm.size(-1) for mm in fm])
                        P = torch.cat(pts, 0)[j]
                    seg = dec(P[None], off[j][None])[0]
                    cs = float(item["clip_stride"])
                    cc = float(item["clip_size"])
                    fps = float(item["fps"])
                    s = float(np.clip(
                        (seg[0].item() * cs + 0.5 * cc) / fps, 0,
                        item["duration"]))
                    e = float(np.clip(
                        (seg[1].item() * cs + 0.5 * cc) / fps, 0,
                        item["duration"]))
                    return s, e
                s0, e0 = top1_seg(vid, gt)
                s1, e1 = top1_seg(vid_cf[0], tgt_cf[q])
            cs = float(item["clip_stride"])
            cc = float(item["clip_size"])
            fps = float(item["fps"])
            to_sec = lambda t0, t1: (
                np.clip((t0 * cs + 0.5 * cc) / fps, 0, item["duration"]),
                np.clip((t1 * cs + 0.5 * cc) / fps, 0, item["duration"]))
            g0s, g0e = to_sec(gt[0], gt[1])
            g1s, g1e = to_sec(tgt_cf[q][0].item(),
                              tgt_cf[q][1].item())
            gt_shift = 0.5 * (g1s + g1e) - 0.5 * (g0s + g0e)
            pred_shift = 0.5 * (s1 + e1) - 0.5 * (s0 + e0)
            recs.append({
                "vid": str(item["vid_id"])[:12], "q": q,
                "iou_orig": float(iou_1d(np.array(s0), np.array(e0),
                                         g0s, g0e)),
                "iou_cf": float(iou_1d(np.array(s1), np.array(e1),
                                       g1s, g1e)),
                "gt_shift": float(gt_shift), "pred_shift":
                    float(pred_shift),
                "center_err": float(pred_shift - gt_shift),
                "start_err": float((s1 - s0) - (g1s - g0s)),
                "end_err": float((e1 - e0) - (g1e - g0e)),
            })
            break
        n_done += 1
    a = {k: float(np.mean([abs(r[k]) for r in recs]))
         for k in ("center_err", "start_err", "end_err")}
    out = {
        "n_probes": len(recs),
        "mean_abs_center_err_s": a["center_err"],
        "mean_abs_start_err_s": a["start_err"],
        "mean_abs_end_err_s": a["end_err"],
        "mean_iou_orig": float(np.mean([r["iou_orig"] for r in recs])),
        "mean_iou_cf": float(np.mean([r["iou_cf"] for r in recs])),
        "corr_predshift_gtshift": float(np.corrcoef(
            [r["pred_shift"] for r in recs],
            [r["gt_shift"] for r in recs])[0, 1]) if len(recs) > 2 else None,
        "records_sample": recs[:10],
    }
    Path(out_json).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items()
                      if k != "records_sample"}, indent=2))
    print("->", out_json)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--ckpt", default="last")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-videos", type=int, default=60)
    args = ap.parse_args()
    run_equivariance(Path(args.ckpt_dir), args.ckpt, args.out,
                     n_videos=args.n_videos)

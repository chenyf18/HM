#!/usr/bin/env python3
"""Eval-only MomentRankHead rerank for ALMR (plan A decisive step).

Rebuilds the training-time cross-level candidate pool per query, then ranks
it three ways under identical decode+NMS: sigmoid baseline, trained head,
and a blend. Internally consistent A/B on the same candidates.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "hydra"), str(ROOT / "tools")]

from libs import load_opt  # noqa: E402
from libs.modeling.model import make_models_net  # noqa: E402
from libs.modeling.temporal_coordinates import (  # noqa: E402
    AdaptiveTemporalPointGenerator, decode_offsets,
)
from libs.data import make_dataset  # noqa: E402
import tools.exact_equivalence_audit_au as aud  # noqa: E402


def iou(a, b):
    inter = min(a[1], b[1]) - max(a[0], b[0])
    if inter <= 0:
        return 0.0
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def nms(segs, scores, iou_thresh=0.3, topk=5):
    order = sorted(range(len(segs)), key=lambda i: -scores[i])
    keep = []
    for i in order:
        if all(iou(segs[i], segs[j]) < iou_thresh for j in keep):
            keep.append(i)
        if len(keep) >= topk:
            break
    return [segs[i] for i in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(
        ROOT / "experiments/allocator_diagnosis/seed_1/ALMR"))
    ap.add_argument("--per-level-topk", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", default=str(
        ROOT / "experiments/allocator_diagnosis/seed_1/ALMR/rank_head_eval.json"))
    args = ap.parse_args()
    run = Path(args.run_dir)
    opt = load_opt(str(run / "opt.yaml"), is_training=False)
    torch.manual_seed(1234567891)
    model = make_models_net(opt)
    aud.load_shared_checkpoint(model, str(run / "models/last.pth"))
    model = model.cuda().eval()
    ptg = AdaptiveTemporalPointGenerator(
        max_seq_len=opt['pt_gen']['max_seq_len'],
        num_fpn_levels=opt['pt_gen']['num_fpn_levels'],
        regression_range=opt['pt_gen'].get('regression_range', 4),
        sigma=opt['pt_gen'].get('sigma', 1),
        input_stride=opt['model']['vid_net'].get('stride', 1),
    ).cuda()

    data_opt = dict(opt['train']['data'])
    data_opt.update(dict(opt['eval']['data']))
    data_opt['split'] = 'val'
    data_opt['crop_ratio'] = None
    dataset = make_dataset(data_opt, num_epochs=1, is_training=False)

    hits = {k: [0, 0, 0] for k in (10, 11, 50, 51)}  # base, head, blend
    n_q = 0
    for di in range(len(dataset)):
        sample = dataset[di]
        clip_stride = sample.get('clip_stride', 1.0)
        clip_size = sample.get('clip_size', 1.0)
        fps = sample.get('fps', 1.0)
        vid = sample['vid']
        for qi in range(len(sample['text'])):
            gt_tok = sample['target'][qi].tolist()
            gt = ((gt_tok[0] * clip_stride + 0.5 * clip_size) / fps,
                  (gt_tok[1] * clip_stride + 0.5 * clip_size) / fps)
            window = vid.unsqueeze(0).cuda()
            wmask = torch.ones(1, 1, window.size(-1), dtype=torch.bool,
                               device='cuda')
            text = sample['text'][qi].unsqueeze(0).cuda()
            tmask = torch.ones(1, 1, text.size(-1), dtype=torch.bool,
                               device='cuda')
            tsize = torch.ones(1, dtype=torch.long, device='cuda')
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                if model.early_fusion:
                    w, wm = model.project_video(window, wmask)
                    w, wm = model.fusion(w, wm, text, tmask, tsize)
                else:
                    w, wm = window, wmask
                fpn, fpn_masks, _, _ = model.encode_video(
                    w, wm, query_feat=text, query_mask=tmask, text_size=tsize)
                logits, _, offsets, _ = model.fuse_and_predict(
                    fpn, fpn_masks, text, tmask, tsize)
                meta = list(model.vid_net.last_temporal_metadata)
                pts = ptg(meta, fpn_masks)
            cands = []   # (sec_a, sec_b, sigmoid, head)
            pooled_all, geo_all, lvl_all = [], [], []
            raw = []
            for lvl, (lg, off, p, fpn_l, msk) in enumerate(
                    zip(logits, offsets, pts, fpn, fpn_masks)):
                seg = decode_offsets(p, off)
                prob = torch.sigmoid(lg).float()
                valid = (msk if msk.ndim == 2 else msk.squeeze(-2)).bool()[0]
                v = prob[0][valid]
                if v.numel() == 0:
                    continue
                k = min(args.per_level_topk, v.numel())
                pos = valid.nonzero(as_tuple=False).flatten()[
                    v.topk(k).indices]
                centers = p[0, :, 0]
                for t_idx in pos.tolist():
                    s0, e0 = seg[0, t_idx].tolist()
                    lo, hi = min(s0, e0), max(s0, e0)
                    inside = (centers >= lo) & (centers <= hi) & valid
                    if not bool(inside.any()):
                        inside = valid
                    pooled_all.append(fpn_l[0, :, inside].float().mean(-1))
                    dur = hi - lo + 1e-6
                    ref = float(p.size(1)) + 1.0
                    geo_all.append(torch.tensor(
                        [math.log(dur), p[0, t_idx, 0].item() / ref,
                         lo / ref, hi / ref]))
                    lvl_all.append(lvl)
                    raw.append((
                        (lo * clip_stride + 0.5 * clip_size) / fps,
                        (hi * clip_stride + 0.5 * clip_size) / fps,
                        float(prob[0, t_idx])))
            if not raw:
                continue
            with torch.no_grad():
                hs = model.moment_rank_head(
                    torch.stack(pooled_all).cuda(),
                    torch.stack(geo_all).cuda(),
                    torch.tensor(lvl_all, device='cuda')).cpu()
            smax = max(r[2] for r in raw) + 1e-6
            hmax = float(hs.max()) + 1e-6
            variants = (
                [r[2] for r in raw],                       # base sigmoid
                [float(h) for h in hs],                    # head only
                [0.5 * r[2] / smax + 0.5 * float(h) / hmax
                 for r, h in zip(raw, hs)],                # blend
            )
            segs = [(r[0], r[1]) for r in raw]
            for vi, scores in enumerate(variants):
                top = nms(segs, scores)
                for kk, thr in ((10, 0.3), (11, 0.5)):
                    hits[kk][vi] += int(any(iou(s, gt) >= thr for s in top[:1]))
                for kk, thr in ((50, 0.3), (51, 0.5)):
                    hits[kk][vi] += int(any(iou(s, gt) >= thr for s in top[:5]))
            n_q += 1
            if n_q % 300 == 0:
                print(f"{n_q} q | base R1@.3={hits[10][0]/n_q:.4f} "
                      f"head={hits[10][1]/n_q:.4f} "
                      f"blend={hits[10][2]/n_q:.4f}", flush=True)
        if args.limit and di + 1 >= args.limit:
            break

    names = ("base_sigmoid", "rank_head", "blend")
    out = {"queries": n_q}
    for vi, name in enumerate(names):
        out[name] = {
            "R1@0.3": hits[10][vi] / n_q, "R1@0.5": hits[11][vi] / n_q,
            "R5@0.3": hits[50][vi] / n_q, "R5@0.5": hits[51][vi] / n_q,
        }
        m = out[name]
        out[name]["Mean"] = sum(m[k] for k in
                                ("R1@0.3", "R1@0.5", "R5@0.3", "R5@0.5")) / 4
    print(json.dumps(out, indent=2))
    Path(args.output).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Option B screening: Test-time Temporal Zoom on a trained clean checkpoint.

For each validation query:
  1. take top-5 coarse candidates from the existing predictions_last.json;
  2. crop the raw feature window around each candidate (with margin);
  3. re-encode at native resolution through the dynamic pipeline + decode;
  4. fuse candidate sets and re-rank; recompute official R@1/5 metrics.

Zero training. Both baseline-replay and zoom metrics use the same final
ranking code, so the delta isolates the zoom effect.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "hydra"), str(ROOT / "tools")]

from libs import load_opt  # noqa: E402
from libs.modeling.model import make_models_net  # noqa: E402
from libs.modeling.temporal_coordinates import decode_offsets  # noqa: E402
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
    return [segs[i] for i in keep], [scores[i] for i in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(
        ROOT / "experiments/allocator_diagnosis/seed_1/AUC"))
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--margin", type=float, default=0.5,
                    help="crop margin as a fraction of candidate duration")
    ap.add_argument("--min-window", type=int, default=64)
    ap.add_argument("--zoom-topn", type=int, default=50)
    ap.add_argument("--limit-videos", type=int, default=0)
    ap.add_argument("--output", default="")
    ap.add_argument("--mode", default="rrf", choices=("refine", "rrf"))
    args = ap.parse_args()
    run = Path(args.run_dir)
    preds = json.loads((run / "predictions_last.json").read_text())

    opt = load_opt(str(run / "opt.yaml"), is_training=False)
    torch.manual_seed(1234567891)
    model = make_models_net(opt)
    aud.load_shared_checkpoint(model, str(run / "models/last.pth"))
    model = model.cuda().eval()
    adaptive_pt_gen = __import__(
        "libs.modeling.temporal_coordinates", fromlist=["AdaptiveTemporalPointGenerator"]
    ).AdaptiveTemporalPointGenerator(
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
    print(f"val dataset entries: {len(dataset)}")

    def zoom_candidates(vid_feat, text_tok, text_mask, clip_stride, clip_size,
                        fps, s_tok, e_tok, _model=model,
                        _ptg=adaptive_pt_gen, _topn=args.zoom_topn,
                        _minw=args.min_window):
        model = _model; adaptive_pt_gen = _ptg
        zoom_topn = _topn; min_window = _minw
        T0, T1 = int(s_tok), int(e_tok)
        window = vid_feat[:, T0:T1]
        if window.size(-1) < min_window:
            pad = min_window - window.size(-1)
            window = F.pad(window, (0, pad))
        wmask = torch.ones(1, 1, window.size(-1), dtype=torch.bool,
                           device='cuda')
        window = window.unsqueeze(0).cuda()
        text = text_tok.unsqueeze(0).cuda()
        tmask = text_mask.cuda()
        tsize = torch.ones(1, dtype=torch.long, device='cuda')
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            pt = torch.cat([text], dim=0) if text.ndim == 2 else text
            pm = tmask
            if model.early_fusion:
                w, wm = model.project_video(window, wmask)
                w, wm = model.fusion(w, wm, pt, pm, tsize)
            else:
                w, wm = window, wmask
            fpn, fpn_masks, _, _ = model.encode_video(
                w, wm, query_feat=pt, query_mask=pm, text_size=tsize)
            logits, _, offsets, _ = model.fuse_and_predict(
                fpn, fpn_masks, pt, pm, tsize)
            meta = list(model.vid_net.last_temporal_metadata)
            pts = adaptive_pt_gen(meta, fpn_masks)
        cands = []
        for lvl, (lg, off, p) in enumerate(zip(logits, offsets, pts)):
            seg = decode_offsets(p, off)          # (B, T, 2) local tokens
            sc = torch.sigmoid(lg).flatten()
            top = sc.topk(min(zoom_topn, sc.numel()))
            for s_val, idx in zip(top.values.tolist(), top.indices.tolist()):
                a, b = seg[0, idx].tolist()
                ga = (a + T0) * clip_stride + 0.5 * clip_size
                gb = (b + T0) * clip_stride + 0.5 * clip_size
                cands.append((ga / fps, gb / fps, float(s_val)))
        return cands

    hits_base = {k: 0 for k in (10, 11, 50, 51)}
    hits_zoom = {k: 0 for k in (10, 11, 50, 51)}
    n_q = 0
    out_records = []
    for di in range(len(dataset)):
        sample = dataset[di]
        vid = sample['vid']
        vid_id = sample.get('vid_id')
        entry = preds['videos'].get(vid_id)
        if entry is None:
            continue
        clip_stride = sample.get('clip_stride', 1.0)
        clip_size = sample.get('clip_size', 1.0)
        fps = sample.get('fps', 1.0)
        for qi, q in enumerate(entry['queries']):
            if qi >= len(sample['text']):
                break
            gt = q['ground_truth']
            coarse = [(p['segment'][0], p['segment'][1], p['score'])
                      for p in q['predictions'][:args.topk]]
            if not coarse:
                continue
            segs_b, _ = nms(coarse, [c[2] for c in coarse], topk=5)
            # zoom: keep candidates per coarse window for calibration-free fusion
            zoom_by_cand = []
            for (a, b, sc) in coarse:
                dur = max(b - a, 1e-6)
                m = args.margin * dur
                to_seconds = fps / max(clip_stride, 1e-6)
                s_tok = max(0, int(((a - m) * to_seconds - 0.5 * clip_size)))
                e_tok = min(vid.size(-1), int(
                    ((b + m) * to_seconds - 0.5 * clip_size)) + 1)
                if e_tok - s_tok < 8:
                    zoom_by_cand.append([])
                    continue
                text_tok = sample['text'][qi]
                tmask = torch.ones(1, 1, text_tok.size(-1), dtype=torch.bool)
                zoom_by_cand.append(zoom_candidates(
                    vid, text_tok, tmask, clip_stride, clip_size, fps,
                    s_tok, e_tok))
            if args.mode == 'refine':
                # boundary refinement: keep coarse ranking, snap each span to
                # the best-overlap high-score zoom candidate in its window
                refined = []
                for (a, b, sc), wins in zip(coarse, zoom_by_cand):
                    best, best_key = (a, b), -1.0
                    for (za, zb, zs) in wins:
                        ov = iou((a, b), (za, zb))
                        key = zs * (0.5 + 0.5 * ov)
                        if key > best_key and ov > 0.3:
                            best, best_key = (za, zb), key
                    refined.append((best[0], best[1], sc))
                segs_z, _ = nms(refined, [c[2] for c in refined], topk=5)
            else:  # rrf: reciprocal-rank fusion, immune to score scales
                rrf = {}
                def add(items, k=60):
                    for rank, (a, b, _) in enumerate(
                            sorted(items, key=lambda c: -c[2])[:10]):
                        rrf.setdefault((round(a, 2), round(b, 2)), 0.0)
                        rrf[(round(a, 2), round(b, 2))] += 1.0 / (k + rank + 1)
                add(coarse)
                for wins in zoom_by_cand:
                    add(wins)
                top = sorted(rrf.items(), key=lambda kv: -kv[1])[:5]
                segs_z = [kv[0] for kv in top]
            for tag, segs, hits in (("base", segs_b, hits_base),
                                    ("zoom", segs_z, hits_zoom)):
                for k, thr in ((10, 0.3), (11, 0.5)):
                    hits[k] += int(any(iou(s, gt) >= thr for s in segs[:1]))
                for k, thr in ((50, 0.3), (51, 0.5)):
                    hits[k] += int(any(iou(s, gt) >= thr for s in segs[:5]))
            n_q += 1
            if n_q % 200 == 0:
                print(f"{n_q} queries | base R1@.3={hits_base[10]/n_q:.4f} "
                      f"zoom={hits_zoom[10]/n_q:.4f}", flush=True)
        if args.limit_videos and di + 1 >= args.limit_videos:
            break

    def fmt(h):
        return {"R1@0.3": h[10] / n_q, "R1@0.5": h[11] / n_q,
                "R5@0.3": h[50] / n_q, "R5@0.5": h[51] / n_q}
    result = {"queries": n_q, "baseline_replay": fmt(hits_base),
              "zoom": fmt(hits_zoom)}
    print(json.dumps(result, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

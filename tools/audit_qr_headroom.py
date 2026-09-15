#!/usr/bin/env python3
"""QR-TG Step-0 audit on an existing checkpoint (zero training).

Quantifies on single-window val queries:
  1. per-level top-score distributions (cross-level calibration mismatch);
  2. final top-1 source level under raw scores vs per-level z-norm;
  3. ranking headroom: best-IoU among top-5 by raw global score vs by
     per-level-normalized score vs oracle top-5.
"""
import sys, json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "hydra"), str(ROOT / "tools")]

from libs import load_opt
from libs.modeling.model import make_models_net
from libs.modeling.temporal_coordinates import (
    AdaptiveTemporalPointGenerator, decode_offsets)
from libs.data import make_dataset
import tools.exact_equivalence_audit_au as aud


def iou(a, b):
    inter = min(a[1], b[1]) - max(a[0], b[0])
    if inter <= 0:
        return 0.0
    u = max(a[1], b[1]) - min(a[0], b[0])
    return inter / u


def main():
    run = ROOT / "experiments/allocator_diagnosis/seed_1/AUC"
    opt = load_opt(str(run / "opt.yaml"), is_training=False)
    torch.manual_seed(1234567891)
    model = make_models_net(opt)
    aud.load_shared_checkpoint(model, str(run / "models/last.pth"))
    model = model.cuda().eval()
    ptg = AdaptiveTemporalPointGenerator(
        max_seq_len=opt['pt_gen']['max_seq_len'],
        num_fpn_levels=opt['pt_gen']['num_fpn_levels'],
        regression_range=4, sigma=1,
        input_stride=opt['model']['vid_net'].get('stride', 1)).cuda()
    do = dict(opt['train']['data']); do.update(dict(opt['eval']['data']))
    do['split'] = 'val'; do['crop_ratio'] = None
    ds = make_dataset(do, num_epochs=1, is_training=False)

    lvl_top_scores = [[] for _ in range(8)]
    lvl_top1_raw = [0] * 8
    lvl_top1_zn = [0] * 8
    best5_raw, best5_zn, best5_oracle = [], [], []
    n = 0
    for di in range(len(ds)):
        s = ds[di]
        if s['vid'].size(-1) > 2300:
            continue
        for qi in range(len(s['text'])):
            window = s['vid'].unsqueeze(0).cuda()
            wmask = torch.ones(1, 1, window.size(-1), dtype=torch.bool,
                               device='cuda')
            text = s['text'][qi].unsqueeze(0).cuda()
            tmask = torch.ones(1, 1, text.size(-1), dtype=torch.bool,
                               device='cuda')
            tsize = torch.ones(1, dtype=torch.long, device='cuda')
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                w, wm = model.project_video(window, wmask)
                w, wm = model.fusion(w, wm, text, tmask, tsize)
                fpn, fpn_masks, _, _ = model.encode_video(
                    w, wm, query_feat=text, query_mask=tmask, text_size=tsize)
                logits, _, offsets, _ = model.fuse_and_predict(
                    fpn, fpn_masks, text, tmask, tsize)
                pts = ptg(list(model.vid_net.last_temporal_metadata), fpn_masks)
            gt = s['target'][qi].tolist()
            pool = []          # (iou, score_raw, level)
            lvl_stats = {}
            for lvl in range(len(logits)):
                seg = decode_offsets(pts[lvl], offsets[lvl])[0]
                prob = torch.sigmoid(logits[lvl][0]).float()
                valid = fpn_masks[lvl].squeeze()[:prob.numel()].bool()
                v = prob[valid]
                if v.numel() == 0:
                    continue
                k = min(10, v.numel())
                top = v.topk(k)
                pos = valid.nonzero().flatten()[top.indices]
                lvl_top_scores[lvl].append(float(top.values.max()))
                mu = float(top.values.mean()); sd = float(top.values.std()) + 1e-6
                lvl_stats[lvl] = (mu, sd)
                for sv, ti in zip(top.values.tolist(), pos.tolist()):
                    a, b = seg[ti].tolist()
                    pool.append((iou((min(a, b), max(a, b)), gt), sv, lvl))
            if not pool:
                continue
            # raw ranking
            pool_raw = sorted(pool, key=lambda x: -x[1])[:5]
            best5_raw.append(max(x[0] for x in pool_raw))
            lvl_top1_raw[pool_raw[0][2]] += 1
            # z-norm per level ranking
            pool_z = [(i0, (sv - lvl_stats[lv][0]) / lvl_stats[lv][1], lv)
                      for (i0, sv, lv) in pool]
            top_z = sorted(pool_z, key=lambda x: -x[1])[:5]
            best5_zn.append(max(x[0] for x in top_z))
            lvl_top1_zn[top_z[0][2]] += 1
            best5_oracle.append(sorted(x[0] for x in pool)[-1])
            n += 1
            if n >= 150:
                break
        if n >= 150:
            break

    m = lambda v: sum(v) / max(len(v), 1)
    print(f"queries={n}")
    for lvl in range(8):
        if lvl_top_scores[lvl]:
            v = lvl_top_scores[lvl]
            print(f"level{lvl} top-score mean={m(v):.4f} std_over_queries="
                  f"{(sum((x-m(v))**2 for x in v)/len(v))**0.5:.4f} "
                  f"min={min(v):.3f} max={max(v):.3f}")
    print("top-1 level attribution RAW :", lvl_top1_raw)
    print("top-1 level attribution ZNORM:", lvl_top1_zn)
    print(f"bestIoU@top5: raw={m(best5_raw):.4f} znorm={m(best5_zn):.4f} "
          f"oracle={m(best5_oracle):.4f}")


if __name__ == "__main__":
    main()

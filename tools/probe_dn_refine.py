#!/usr/bin/env python3
"""Plan C probe: does trained MomentDenoiser refinement improve candidate IoU?

Single-window videos, training-identical candidate construction; measures
mean best-IoU / candidate IoU before vs after T refinement steps.
"""
import sys, math
from pathlib import Path
import torch
import torch.nn.functional as F

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
    run = ROOT / "experiments/allocator_diagnosis/seed_1/ALDN"
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

    before, after1, after2, n = [], [], [], 0
    for di in range(len(ds)):
        s = ds[di]
        if s['vid'].size(-1) > 2300:      # single-window videos only
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
                if model.early_fusion:
                    w, wm = model.project_video(window, wmask)
                    w, wm = model.fusion(w, wm, text, tmask, tsize)
                else:
                    w, wm = window, wmask
                enc_t, enc_m = model.encode_text(text.float(), tmask)
                qr = (enc_t * enc_m[:, None, :].to(enc_t.dtype)).sum(-1) / \
                    enc_m[:, None, :].to(enc_t.dtype).sum(-1).clamp_min(1)
                fpn, fpn_masks, _, _ = model.encode_video(
                    w, wm, query_feat=text, query_mask=tmask, text_size=tsize)
                logits, _, offsets, _ = model.fuse_and_predict(
                    fpn, fpn_masks, text, tmask, tsize)
                pts = ptg(list(model.vid_net.last_temporal_metadata), fpn_masks)
            gt = s['target'][qi].tolist()
            cands = []   # (s, e, score, level, t_idx)
            for lvl in range(len(logits)):
                seg = decode_offsets(pts[lvl], offsets[lvl])[0]
                prob = torch.sigmoid(logits[lvl][0])
                valid = fpn_masks[lvl].squeeze()[:prob.numel()].bool()
                v = prob[valid]
                if v.numel() == 0:
                    continue
                pos = valid.nonzero().flatten()[v.topk(min(10, v.numel())).indices]
                centers = pts[lvl][0, :, 0]
                for ti in pos.tolist():
                    a, b = seg[ti].tolist()
                    lo, hi = min(a, b), max(a, b)
                    cands.append((lo, hi, float(prob[ti]), lvl, ti, centers))
            if not cands:
                continue
            cands_k = list(cands)
            with torch.no_grad():
                def refine(spans, steps):
                    nonlocal cands_k
                    out = torch.tensor(spans, dtype=torch.float32,
                                       device='cuda')
                    for _ in range(steps):
                        pooled, geo = [], []
                        ref = 2400.0
                        for (lo, hi, _, lvl, ti, centers) in cands_k:
                            span = max(hi - lo, 1e-6)
                            geo.append([math.log(span),
                                        0.5 * (lo + hi) / ref,
                                        lo / ref, hi / ref])
                        geo_t = torch.tensor(geo, device='cuda')
                        for i, (lo, hi, _, lvl, ti, centers) in enumerate(cands_k):
                            inside = (centers >= lo) & (centers <= hi)
                            if not bool(inside.any()):
                                inside = torch.ones_like(centers, dtype=torch.bool)
                            pooled.append(fpn[lvl][0, :, :centers.numel()][
                                :, inside[:fpn[lvl].size(-1)]].float().mean(-1))
                        pooled_t = torch.stack(pooled)
                        qr_flat = qr.reshape(-1, qr.size(-1)).float()
                        pred = model.moment_denoiser(
                            pooled_t,
                            qr_flat[:1].expand(pooled_t.size(0), -1),
                            geo_t)
                        span = (out[:, 1] - out[:, 0]).clamp_min(1e-6)
                        out = out.clone()
                        out[:, 0] = out[:, 0] + pred[:, 0] * span
                        out[:, 1] = out[:, 1] + pred[:, 1] * span
                        # update cands_k spans for next iter pooling
                        cands_k = [(float(out[i, 0]), float(out[i, 1]),
                                    c[2], c[3], c[4], c[5])
                                   for i, c in enumerate(cands_k)]
                    return out

                base_spans = [(c[0], c[1]) for c in cands]
                r1 = refine(base_spans, 1)
                r2 = refine([(c[0], c[1]) for c in cands], 2)
            before.append(max(iou((c[0], c[1]), gt) for c in cands))
            after1.append(max(iou((float(r[0]), float(r[1])), gt)
                              for r in r1))
            after2.append(max(iou((float(r[0]), float(r[1])), gt)
                              for r in r2))
            n += 1
            if n >= 120:
                break
        if n >= 120:
            break
    m = lambda v: sum(v) / max(len(v), 1)
    print(f"n={n} bestIoU: base={m(before):.4f} refine1={m(after1):.4f} "
          f"refine2={m(after2):.4f}")


if __name__ == "__main__":
    main()

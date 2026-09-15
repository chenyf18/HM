#!/usr/bin/env python3
"""Query Injection Audit (HM-QINJ-AUDIT-025).

Controlled frozen probes: does query injection BEFORE the temporal
(Mamba) encoder produce query-conditioned localization signal in the
encoder output?

Context: A-U-clean already fuses the query at the input level (xattn
after vid_proj, before encode_video) - that product is the F0 baseline
(GT AUC ~ 0.50, QC-AUDIT-024). This audit replaces the injection
mechanism with three canonical frozen forms and re-runs the FROZEN
encoder:

  F0 baseline : official pipeline (xattn early fusion + encoder)
  ProbeA_Add  : x_t + q        (q = value-projected pooled query, frozen)
  ProbeB_Gate : x_t * sigmoid(q)
  ProbeC_Tok  : [q, x_0..x_T]  (query token prepended, state propagates
                through the Mamba hierarchy; outputs read at video
                positions, i.e. [:, 1:] per level)

Per variant measured on candidates: query cosine, inside-GT AUC,
frozen linear IoU probe, candidate ranking probe. No model weights are
modified; no loss; no training of the backbone.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "hydra"), str(ROOT / "tools")]

from tools.audit_ranking_bottleneck import (  # noqa: E402
    AU_ROOT, iou_1d, rankdata_avg, corr, soft_nms,
)
from tools.anchor_audit import bucket_id, balanced_select  # noqa: E402
from tools.qc_audit import (  # noqa: E402
    AUC_P32_ROOT, make_rank_head, iou_seconds, center_seconds,
)

QINJ_ROOT = ROOT / "experiments/qinj_audit"
TRAIN_NPZ = QINJ_ROOT / "train_candidates.npz"
VAL_NPZ = QINJ_ROOT / "val_candidates.npz"
VAL_RANK_NPZ = QINJ_ROOT / "val_rank.npz"
PROBES_PT = QINJ_ROOT / "qinj_probes.pt"
RESULTS_JSON = QINJ_ROOT / "qinj_results.json"

VARIANTS = ("F0", "A_add", "B_gate", "C_tok")
FEAT_DIM = 384
LOW_CAP = 60


def build_evaluator(split, limit_videos=None, sample=True, probes=None):
    import tools.run_formal_ablation as rfa
    from libs import load_opt

    QINJ_ROOT.mkdir(parents=True, exist_ok=True)
    opt_path = QINJ_ROOT / f"opt_{split}.yaml"
    if not opt_path.exists():
        shutil.copyfile(AUC_P32_ROOT / "opt.yaml", opt_path)
    opt = load_opt(str(opt_path), is_training=False)
    opt["eval"]["data"]["split"] = split
    opt["_root"] = str(QINJ_ROOT)
    opt["_ckpt"] = "last"
    models_link = QINJ_ROOT / "models"
    if not models_link.exists():
        models_link.symlink_to(AU_ROOT / "models", target_is_directory=True)

    rank_heads = None
    if probes is not None:
        rank_heads = {}
        for name, sd in probes["rank_heads"].items():
            h = make_rank_head().cuda().eval()
            h.load_state_dict(sd)
            rank_heads[name] = h

    class QinjEvaluator(rfa.FormalEvaluator):
        def __init__(self, opt, precision_policy="p2"):
            self.records = []
            self._collect_log = []
            self._text_pool = None
            self.rng = np.random.default_rng(31337)
            self.input_norms = []
            super().__init__(opt, precision_policy=precision_policy)
            layer0 = self.model.fusion.layers[0]
            self.proj_ln = layer0.ln_xattn_kv
            self.proj_val = layer0.xattn.xattn.value
            original_collect = self._collect_segments

            def collect_wrapper(fpn_points, fpn_logits, fpn_offsets,
                                fpn_masks, ext_scores=None,
                                return_levels=False, fpn_feats=None,
                                query_idx=0):
                result = original_collect(
                    fpn_points, fpn_logits, fpn_offsets, fpn_masks,
                    ext_scores, return_levels=return_levels,
                    fpn_feats=fpn_feats, query_idx=query_idx,
                )
                per_level = []
                for level in range(len(fpn_logits)):
                    points = fpn_points[level]
                    if points.ndim == 3:
                        points = points[0]
                    logits = fpn_logits[level]
                    while logits.ndim > 1:
                        logits = logits[0]
                    offsets = fpn_offsets[level]
                    while offsets.ndim > 2:
                        offsets = offsets[0]
                    mask = fpn_masks[level]
                    while mask.ndim > 1:
                        mask = mask[0]
                    scores = torch.sigmoid(logits) * mask.float()
                    keep = scores > self.pre_nms_thresh
                    per_level.append({
                        "level": level,
                        "pos": keep.nonzero(as_tuple=False).flatten().cpu(),
                        "logit": logits[keep].detach().float().cpu(),
                        "score": scores[keep].detach().float().cpu(),
                        "center": points[:, 0][keep].detach().float().cpu(),
                        "scale": points[:, 3][keep].detach().float().cpu(),
                        "offset": offsets[keep].detach().float().cpu(),
                    })
                self._collect_log.append(per_level)
                return result

            self._collect_segments = collect_wrapper
            orig_encode_text2 = self.model.encode_text2

            def text2_recorder(text, text_masks, text_size):
                out = orig_encode_text2(text, text_masks, text_size)
                self._text_pool = (out[0].detach(), out[1].detach())
                return out

            self.model.encode_text2 = text2_recorder

        def _query_vecs(self):
            t, m = self._text_pool
            with torch.no_grad():
                v = self.proj_val(self.proj_ln(t.float()))
                mf = m.float()
                q384 = (v * mf).sum(-1) / mf.sum(-1).clamp_min(1.0)
                q384 = torch.nn.functional.normalize(q384, dim=-1)
                return q384

        def _probe_forwards(self, window, window_mask, q384, text,
                            text_masks, text_size):
            """Run the frozen encoder under the three injection probes.

            window: raw video features (1, C_in, T); mirrors the official
            pre-fusion steps (vid_proj) then applies the probe injection.
            Returns dict variant -> per-level fpn tuple.
            """
            m = self.model
            outs = {}
            with torch.no_grad():
                wp, wp_mask = m.vid_proj(window, window_mask)
                q = q384
                nq = q.size(0)
                wp_rep = wp.repeat_interleave(nq, dim=0)
                wp_mask_rep = wp_mask.repeat_interleave(nq, dim=0)
                if len(self.input_norms) < 8:
                    self.input_norms.append([
                        float(wp_rep.float().norm(dim=1).mean()),
                        float(q.float().norm(dim=-1).mean()),
                    ])
                kw = dict(
                    query_feat=text if self.query_conditioned else None,
                    query_mask=text_masks if self.query_conditioned
                    else None,
                    text_size=text_size,
                )
                # A: additive broadcast
                xA = wp_rep + q.view(nq, FEAT_DIM, 1).to(wp_rep.dtype)
                outs["A_add"] = m.encode_video(xA, wp_mask_rep, **kw)[0]
                # B: channel gate
                g = torch.sigmoid(q.float()).to(wp_rep.dtype)
                xB = wp_rep * g.view(nq, FEAT_DIM, 1)
                outs["B_gate"] = m.encode_video(xB, wp_mask_rep, **kw)[0]
                # C: query token prepended (pad to stride multiple)
                pad = self.min_chunk_size * self.vid_stride
                t_len = wp_rep.size(-1) + 1
                padded = ((t_len + pad - 1) // pad) * pad
                xC = wp_rep.new_zeros(nq, FEAT_DIM, padded)
                xC[:, :, 0:1] = q.view(nq, FEAT_DIM, 1).to(wp_rep.dtype)
                xC[:, :, 1:wp_rep.size(-1) + 1] = wp_rep
                maskC = torch.zeros(
                    nq, 1, padded, dtype=wp_mask_rep.dtype,
                    device=wp_mask_rep.device)
                maskC[:, :, :t_len] = True
                outs["C_tok"] = m.encode_video(xC, maskC, **kw)[0]
            return outs

        def predict(self, data):
            tokens = data["text"]
            if not isinstance(tokens, tuple):
                tokens = (tokens, )
            vid = data["vid"]
            vid_len = vid.size(-1)
            window_size = min(self.window_size or vid_len, vid_len)
            window_stride = self.window_stride or window_size
            n_win = vid_len - window_size
            offsets_w = []
            idx = 0
            while idx <= n_win:
                offsets_w.append(idx)
                idx += window_stride
            if n_win > 0 and n_win % window_stride > 0:
                offsets_w.append(n_win)

            self._collect_log = []
            # capture the official fused input to encode_video for F0
            f0_store = {}
            orig_encode = self.model.encode_video

            def encode_recorder(vid_in, mask_in, **kwargs):
                out = orig_encode(vid_in, mask_in, **kwargs)
                if 0 not in f0_store:
                    f0_store[0] = out[0]
                return out

            self.model.encode_video = encode_recorder
            results = super().predict(data)
            self.model.encode_video = orig_encode

            n_q = len(results)
            n_w = len(offsets_w)
            if len(self._collect_log) != n_w * n_q:
                raise RuntimeError("collect calls != windows x queries")

            # official text batch (same as predict built internally)
            with torch.no_grad():
                text, text_masks, text_size = self._batchify_text2(
                    text_list=[tokens])
                text = text.cuda()
                text_masks = text_masks.cuda()
                text_size = text_size.cuda()
                text, text_masks = self.model.encode_text2(
                    text, text_masks, text_size)
            q384 = self._query_vecs()

            stride = self.min_chunk_size * self.vid_stride
            input_vid_len = (window_size + stride - 1) // stride * stride
            # probe forwards (single window assumed for Ego4D; assert)
            probe_outs = None
            if n_w == 1:
                window = vid[..., 0:window_size]
                window = torch.nn.functional.pad(
                    window, (0, input_vid_len - window_size))[None].cuda()
                window_mask = (
                    torch.arange(input_vid_len).view(1, 1, -1).cuda()
                    < window_size)
                with torch.autocast(device_type="cuda",
                                    dtype=torch.bfloat16):
                    probe_outs = self._probe_forwards(
                        window, window_mask, q384, text, text_masks,
                        text_size)
            segments = data["segment"]
            for q in range(n_q):
                per_level = self._collect_log[q]
                if n_w != 1:
                    per_level = self._collect_log[q]
                merged = None
                rows = {v: [] for v in VARIANTS}
                qn = q384[q]
                qn_n = torch.nn.functional.normalize(qn.float(), dim=-1)
                for pl in per_level:
                    n_k = pl["score"].numel()
                    entry = np.empty((n_k, 9), dtype=np.float64)
                    entry[:, 0] = pl["logit"].numpy()
                    entry[:, 1] = pl["score"].numpy()
                    entry[:, 2] = pl["level"]
                    entry[:, 3] = pl["center"].numpy()
                    entry[:, 4] = pl["scale"].numpy()
                    entry[:, 5] = pl["offset"][:, 0].numpy()
                    entry[:, 6] = pl["offset"][:, 1].numpy()
                    centers = entry[:, 3] + offsets_w[0]
                    entry[:, 7] = centers - entry[:, 5] * entry[:, 4]
                    entry[:, 8] = centers + entry[:, 6] * entry[:, 4]
                    merged = (entry if merged is None
                              else np.concatenate([merged, entry]))
                    lvl = int(pl["level"])
                    pos = pl["pos"].cuda()
                    f0 = f0_store[0][lvl][q][:, pos].transpose(0, 1)
                    rows["F0"].append(f0.detach().float().cpu())
                    if probe_outs is not None:
                        rows["A_add"].append(
                            probe_outs["A_add"][lvl][q][:, pos]
                            .transpose(0, 1).detach().float().cpu())
                        rows["B_gate"].append(
                            probe_outs["B_gate"][lvl][q][:, pos]
                            .transpose(0, 1).detach().float().cpu())
                        # probe C position mapping: the prepended query
                        # token shifts level-0 outputs by +1; at deeper
                        # levels the pooling grid shifts by only half an
                        # anchor (one input token), so index t_l remains
                        # the best match for the official span.
                        c_out = probe_outs["C_tok"][lvl][q]
                        c_pos = pos + 1 if lvl == 0 else pos
                        rows["C_tok"].append(
                            c_out[:, c_pos]
                            .transpose(0, 1).detach().float().cpu())
                if merged is None:
                    merged = np.zeros((0, 9), dtype=np.float64)
                gt = np.asarray(segments[q], dtype=np.float64)
                rec = {
                    "vid_id": str(data["vid_id"]),
                    "query_idx": q,
                    "gt_start": float(gt[0]),
                    "gt_end": float(gt[1]),
                    "clip_stride": float(data["clip_stride"]),
                    "clip_size": float(data["clip_size"]),
                    "fps": float(data["fps"]),
                    "duration": float(data["duration"]),
                    "pool": merged,
                    "official_segments": results[q][
                        "segments"].detach().cpu().numpy().astype(
                            np.float64),
                    "official_scores": results[q][
                        "scores"].detach().cpu().numpy().astype(np.float64),
                }
                ious = iou_seconds(rec, merged) if merged.shape[0] \
                    else np.zeros(0)
                cos = {}
                feats = {}
                for v in VARIANTS:
                    ft = (torch.cat(rows[v]) if rows[v]
                          else torch.zeros(0, FEAT_DIM))
                    feats[v] = ft
                    if ft.shape[0]:
                        fn = torch.nn.functional.normalize(
                            ft.cuda().float(), dim=-1)
                        cos[v] = (fn @ qn_n.cuda()).cpu().numpy().astype(
                            np.float32)
                    else:
                        cos[v] = np.zeros(0, np.float32)
                rec["cos"] = cos
                if rank_heads is not None and merged.shape[0]:
                    geo = np.stack([
                        merged[:, 5], merged[:, 6],
                        np.log1p(np.maximum(
                            merged[:, 8] - merged[:, 7], 0.0)),
                    ], axis=1).astype(np.float32)
                    g = torch.from_numpy(geo).cuda()
                    lv = torch.from_numpy(merged[:, 2]).long().cuda()
                    for v in VARIANTS:
                        with torch.no_grad():
                            rec[f"score_{v}"] = torch.sigmoid(
                                rank_heads[v](
                                    feats[v].cuda().float(), g, lv)
                            ).cpu().numpy().astype(np.float32)
                if sample and merged.shape[0]:
                    sel = balanced_select(self.rng, ious)
                    rec["sample_pool"] = merged[sel].astype(np.float32)
                    rec["sample_ious"] = ious[sel].astype(np.float32)
                    for v in VARIANTS:
                        rec[f"sample_feat_{v}"] = (
                            feats[v].numpy()[sel].astype(np.float16))
                self.records.append(rec)
            return results

    ev = QinjEvaluator(opt, precision_policy="p3")
    if limit_videos:
        ev.dataloader = list(itertools.islice(
            iter(ev.dataloader), limit_videos))
        ev.num_itrs = len(ev.dataloader)
    return ev


def run_dump(split, limit_videos=None):
    ev = build_evaluator(split, limit_videos=limit_videos)
    ev.run()
    recs = [r for r in ev.records]
    rows = np.concatenate([r["sample_pool"] for r in recs])
    ious = np.concatenate([r["sample_ious"] for r in recs])
    qi = np.concatenate([
        np.full(len(r["sample_ious"]), i, dtype=np.int64)
        for i, r in enumerate(recs)])
    inside = np.concatenate([
        (center_seconds(r, r["sample_pool"].astype(np.float64))
         >= r["gt_start"]) & (center_seconds(
             r, r["sample_pool"].astype(np.float64)) <= r["gt_end"])
        for r in recs])
    arrays = dict(rows=rows, ious=ious, qi=qi, inside_gt=inside)
    for v in VARIANTS:
        arrays[v] = np.concatenate([r[f"sample_feat_{v}"] for r in recs])
    if ev.input_norms:
        arrays["input_norms"] = np.array(ev.input_norms)
    out = TRAIN_NPZ if split == "train" else VAL_NPZ
    np.savez_compressed(out, **arrays)
    hist = np.histogram(ious, bins=[0, .1, .3, .5, 1.01])[0]
    print(f"[{split}] sampled {len(rows)}  buckets={hist.tolist()} -> {out}")
    if ev.input_norms:
        n = ev.input_norms[0]
        print(f"input norms: video {n[0]:.1f}, query {n[1]:.3f}")


def auroc_ovr(scores, labels, n_classes=4):
    aucs = []
    for c in range(n_classes):
        pos = labels == c
        npos, nneg = int(pos.sum()), int((~pos).sum())
        if npos == 0 or nneg == 0:
            continue
        r = rankdata_avg(scores[:, c])
        aucs.append((r[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))
    return float(np.mean(aucs))


def run_train_probes():
    tr = np.load(TRAIN_NPZ)
    va = np.load(VAL_NPZ)
    dev = "cuda:0"
    ytr = np.array([bucket_id(v) for v in tr["ious"]])
    yva = np.array([bucket_id(v) for v in va["ious"]])
    results = {}

    def pairwise(probs):
        accs = []
        for qid in np.unique(va["qi"]):
            m = va["qi"] == qid
            if m.sum() < 4:
                continue
            p, iv = probs[m], va["ious"][m]
            ii, jj = np.triu_indices(len(p), k=1)
            d = iv[ii] - iv[jj]
            ok = np.abs(d) > 1e-6
            if ok.sum() < 4:
                continue
            accs.append(float((np.sign(p[ii][ok] - p[jj][ok])
                               == np.sign(d[ok])).mean()))
        return float(np.mean(accs)), len(accs)

    for v in VARIANTS:
        torch.manual_seed(11)
        lin = torch.nn.Linear(FEAT_DIM, 4).to(dev)
        opt = torch.optim.Adam(lin.parameters(), lr=1e-3,
                               weight_decay=1e-4)
        X = torch.tensor(tr[v], dtype=torch.float32)
        Y = torch.tensor(ytr, dtype=torch.long)
        n = len(Y)
        for ep in range(12):
            perm = torch.randperm(n)
            for i in range(0, n, 8192):
                idx = perm[i:i + 8192]
                loss = torch.nn.functional.cross_entropy(
                    lin(X[idx].to(dev)), Y[idx].to(dev))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        with torch.no_grad():
            logits = torch.cat([
                lin(torch.tensor(va[v][i:i + (1 << 17)],
                                 dtype=torch.float32).to(dev)).cpu()
                for i in range(0, len(yva), 1 << 17)]).numpy()
        pred = logits.argmax(1)
        f1s = []
        for c in range(4):
            tp = float(((pred == c) & (yva == c)).sum())
            fp = float(((pred == c) & (yva != c)).sum())
            fn = float(((pred != c) & (yva == c)).sum())
            f1s.append(2 * tp / max(2 * tp + fp + fn, 1e-9))
        pw, n_pw = pairwise(pred.astype(np.float64))
        results[f"probe_{v}"] = {
            "accuracy": float((pred == yva).mean()),
            "macro_f1": float(np.mean(f1s)),
            "per_class_f1": f1s,
            "macro_auroc": auroc_ovr(logits, yva),
            "pairwise_within_query": pw,
            "pairwise_n_queries": n_pw,
        }
        r = results[f"probe_{v}"]
        print(f"Probe({v}): acc={r['accuracy']:.4f} "
              f"macroF1={r['macro_f1']:.4f} "
              f"AUROC={r['macro_auroc']:.4f} "
              f"pairwise={r['pairwise_within_query']:.4f}")

    # rank heads
    def geo_of(rows):
        return np.stack([
            rows[:, 5], rows[:, 6],
            np.log1p(np.maximum(rows[:, 8] - rows[:, 7], 0.0)),
        ], axis=1).astype(np.float32)

    rank_heads = {}
    curves = {}
    for v in VARIANTS:
        torch.manual_seed(7)
        head = make_rank_head().to(dev)
        opt = torch.optim.Adam(head.parameters(), lr=1e-3)
        F = torch.tensor(tr[v], dtype=torch.float32)
        G = torch.tensor(geo_of(tr["rows"]))
        L = torch.tensor(tr["rows"][:, 2], dtype=torch.long)
        Yv = torch.tensor(tr["ious"], dtype=torch.float32)
        n = len(Yv)
        best = {"sp": -2.0, "state": None}
        curve = []
        for ep in range(15):
            head.train()
            perm = torch.randperm(n)
            for i in range(0, n, 8192):
                idx = perm[i:i + 8192]
                qh = torch.sigmoid(head(
                    F[idx].to(dev), G[idx].to(dev), L[idx].to(dev)))
                loss = torch.nn.functional.binary_cross_entropy(
                    qh, Yv[idx].to(dev).clamp(1e-4, 1 - 1e-4))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            head.eval()
            with torch.no_grad():
                qs = torch.cat([
                    torch.sigmoid(head(
                        torch.tensor(va[v][i:i + (1 << 16)],
                                     dtype=torch.float32).to(dev),
                        torch.tensor(geo_of(
                            va["rows"][i:i + (1 << 16)])).to(dev),
                        torch.tensor(
                            va["rows"][i:i + (1 << 16), 2],
                            dtype=torch.long).to(dev))).cpu()
                    for i in range(0, len(va["ious"]), 1 << 16)]).numpy()
            sp = corr(rankdata_avg(qs), rankdata_avg(va["ious"]))
            curve.append(float(sp))
            if sp > best["sp"]:
                best = {"sp": float(sp), "state": {
                    k: v2.cpu().clone()
                    for k, v2 in head.state_dict().items()}}
        rank_heads[v] = best["state"]
        curves[v] = curve
        print(f"RankHead({v}): best val Spearman={best['sp']:.4f}")
    torch.save({"rank_heads": rank_heads, "curves": curves,
                "linear_probes": results}, PROBES_PT)
    print("probes saved ->", PROBES_PT)


def run_rank(limit_videos=None):
    probes = torch.load(PROBES_PT, map_location="cpu", weights_only=False)
    ev = build_evaluator("val", limit_videos=limit_videos,
                         sample=False, probes=probes)
    ev.run()
    recs = ev.records
    n = len(recs)
    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum([r["pool"].shape[0] for r in recs], out=offsets[1:])
    arrays = {
        "pool": np.concatenate([r["pool"] for r in recs]),
        "pool_offsets": offsets,
        "ious": np.concatenate([
            iou_seconds(r, r["pool"]) if r["pool"].shape[0]
            else np.zeros(0) for r in recs]),
    }
    for v in VARIANTS:
        arrays[f"score_{v}"] = np.concatenate([
            r.get(f"score_{v}", np.zeros(r["pool"].shape[0]))
            for r in recs])
        arrays[f"cos_{v}"] = np.concatenate([r["cos"][v] for r in recs])
    for k in ("gt_start", "gt_end", "clip_stride", "clip_size",
              "fps", "duration"):
        arrays[k] = np.array([r[k] for r in recs], dtype=np.float64)
    np.savez_compressed(VAL_RANK_NPZ, **arrays)
    print(f"val rank scalars ({int(offsets[-1])}) -> {VAL_RANK_NPZ}")


def run_analyze():
    va = np.load(VAL_NPZ)
    probes = torch.load(PROBES_PT, map_location="cpu",
                        weights_only=False)
    rk = np.load(VAL_RANK_NPZ)
    out = {"linear_probes": probes["linear_probes"]}

    lv = va["rows"][:, 2].astype(int)
    inside = va["inside_gt"]
    qn = None  # cos recomputed from stored feats? q384 not stored here.
    # use stored cos columns from the rank pass for alignment stats on the
    # full pool, and per-level sampled AUC recomputed from feats is not
    # possible without q; recompute via rank-pass cos aggregated per
    # query-sample is skipped - use full-pool stats below.
    pool_all = rk["pool"]
    offs = rk["pool_offsets"]
    ious_all = rk["ious"]
    gt_s, gt_e = rk["gt_start"], rk["gt_end"]
    cs_c, cc_c = rk["clip_stride"], rk["clip_size"]
    fps_c, dur_c = rk["fps"], rk["duration"]
    n = len(gt_s)

    def auc_binary(score, pos):
        p, ng = int(pos.sum()), int((~pos).sum())
        if p == 0 or ng == 0:
            return float("nan")
        r = rankdata_avg(score)
        return float((r[pos].sum() - p * (p + 1) / 2) / (p * ng))

    # inside-GT flags per candidate for the full pool
    center_tok = pool_all[:, 3]
    align = []
    for v in VARIANTS:
        cos_col = rk[f"cos_{v}"]
        ins = np.zeros(len(cos_col), dtype=bool)
        for i in range(n):
            sl = slice(offs[i], offs[i + 1])
            if pool_all[sl].shape[0] == 0:
                continue
            cs = (pool_all[sl, 3] * cs_c[i] + 0.5 * cc_c[i]) / fps_c[i]
            ins[sl] = (cs >= gt_s[i]) & (cs <= gt_e[i])
        align.append({
            "feature": v,
            "cos_mean": float(cos_col.mean()),
            "cos_median": float(np.median(cos_col)),
            "auc_inside_gt": auc_binary(cos_col, ins),
        })
    out["alignment"] = align

    # ranking probe
    variants = ["cls"] + [f"score_{v}" for v in VARIANTS]
    o10 = {v: 0.0 for v in variants}
    r15 = {v: 0.0 for v in variants}
    score_cols = {v: rk[f"score_{v}"] for v in VARIANTS}
    for i in range(n):
        sl = slice(offs[i], offs[i + 1])
        p = pool_all[sl]
        if p.shape[0] == 0:
            continue
        base = np.argsort(-p[:, 1], kind="stable")
        p = p[base][:2000]
        iou = ious_all[sl][base][:2000]
        scores = {"cls": p[:, 1]}
        for v in VARIANTS:
            scores[f"score_{v}"] = score_cols[v][sl][base][:2000]
        for v in variants:
            sc = scores[v]
            order = np.argsort(-sc, kind="stable")
            o10[v] += float(iou[order[:10]].max() >= 0.5)
            k_segs, _ = soft_nms(
                np.stack([p[:, 7], p[:, 8]], axis=1), sc)
            s = np.clip(
                (k_segs[:, 0] * cs_c[i] + 0.5 * cc_c[i]) / fps_c[i],
                0, dur_c[i])
            e = np.clip(
                (k_segs[:, 1] * cs_c[i] + 0.5 * cc_c[i]) / fps_c[i],
                0, dur_c[i])
            q_i = iou_1d(s, e, gt_s[i], gt_e[i])
            r15[v] += float((q_i[0] if len(q_i) else 0.0) >= 0.5)
    out["ranking_probe"] = {
        v: {"oracle@10@0.5": o10[v] / n, "R1@0.5": r15[v] / n}
        for v in variants}
    out["n_queries"] = n

    RESULTS_JSON.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print("\n=== alignment (full pool) ===")
    for a in align:
        print(f"{a['feature']:8s} cos_mean={a['cos_mean']:+.4f} "
              f"cos_median={a['cos_median']:+.4f} "
              f"inside-GT AUC={a['auc_inside_gt']:.4f}")
    print("\n=== ranking probe ===")
    for v in variants:
        r = out["ranking_probe"][v]
        print(f"{v:12s} Oracle@10@0.5={100*r['oracle@10@0.5']:.2f} "
              f"R1@0.5={100*r['R1@0.5']:.2f}")
    print("results ->", RESULTS_JSON)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=[
        "dump-train", "dump-val", "train-probes", "rank", "analyze"])
    ap.add_argument("--limit-videos", type=int, default=None)
    args = ap.parse_args()
    if args.stage == "dump-train":
        run_dump("train", args.limit_videos)
    elif args.stage == "dump-val":
        run_dump("val", args.limit_videos)
    elif args.stage == "train-probes":
        run_train_probes()
    elif args.stage == "rank":
        run_rank(args.limit_videos)
    else:
        run_analyze()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Query-Conditioned Representation Audit (HM-QC-AUDIT-024).

Frozen A-U-clean-P32. Localize the bottleneck: fusion stage vs prediction
head. Three feature families per candidate:

  F0: pre-head-fusion video feature   (encode_video fpn output)
  F1: post-fusion prediction feature  (xattn fusion output = head input)
  F2: prediction head hidden feature  (cls head conv/norm/ReLU stack,
      full temporal context, before the final 1-channel conv)

Stages:
  dump-train / dump-val : balanced IoU-bucket sample with F0/F1/F2 +
      per-query embeddings (768 pooled, 384 value-projected).
  rank    : val pass applying trained IoU-regression probes (one per
      feature family) to every pool candidate; stores scores + cosines.
  analyze : alignment (cos / inside-GT AUC), linear IoU probes
      (AUROC / macro-F1 / pairwise), ranking probes (Oracle@10@0.5,
      R1@0.5), verdict data.

No model/loss modification, no base-model training, no decoder.
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
from tools.lqac_probe import _FusionRecorder  # noqa: E402
from tools.anchor_audit import _EncodeRecorder, bucket_id, balanced_select  # noqa: E402

AUC_P32_ROOT = ROOT / "experiments/allocator_diagnosis/seed_1/AUC_P32"
QROOT = ROOT / "experiments/qc_audit"
TRAIN_NPZ = QROOT / "train_candidates.npz"
VAL_NPZ = QROOT / "val_candidates.npz"
VAL_RANK_NPZ = QROOT / "val_rank.npz"
PROBES_PT = QROOT / "qc_probes.pt"
RESULTS_JSON = QROOT / "qc_audit_results.json"

BUCKETS = ((0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 1.01))
LOW_CAP = 60
FEAT_DIM = 384


def iou_seconds(rec, pool):
    cs, cc, fps, dur = (rec["clip_stride"], rec["clip_size"],
                        rec["fps"], rec["duration"])
    s = np.clip((pool[:, 7] * cs + 0.5 * cc) / fps, 0, dur)
    e = np.clip((pool[:, 8] * cs + 0.5 * cc) / fps, 0, dur)
    return iou_1d(s, e, rec["gt_start"], rec["gt_end"])


def center_seconds(rec, pool):
    cs, cc, fps = rec["clip_stride"], rec["clip_size"], rec["fps"]
    return (pool[:, 3] * cs + 0.5 * cc) / fps


def build_evaluator(split, limit_videos=None, sample=True, probes=None):
    import tools.run_formal_ablation as rfa
    from libs import load_opt

    QROOT.mkdir(parents=True, exist_ok=True)
    opt_path = QROOT / f"opt_{split}.yaml"
    if not opt_path.exists():
        shutil.copyfile(AUC_P32_ROOT / "opt.yaml", opt_path)
    opt = load_opt(str(opt_path), is_training=False)
    opt["eval"]["data"]["split"] = split
    opt["_root"] = str(QROOT)
    opt["_ckpt"] = "last"
    models_link = QROOT / "models"
    if not models_link.exists():
        models_link.symlink_to(AU_ROOT / "models", target_is_directory=True)

    rank_heads = None
    if probes is not None:
        rank_heads = {}
        for name, sd in probes["rank_heads"].items():
            h = make_rank_head().cuda().eval()
            h.load_state_dict(sd)
            rank_heads[name] = h

    class QcEvaluator(rfa.FormalEvaluator):
        def __init__(self, opt, precision_policy="p2"):
            self.records = []
            self._collect_log = []
            self._enc_rec = None
            self._fusion_rec = None
            self._text_pool = None
            self.rng = np.random.default_rng(2024)
            self.f2_check = []
            super().__init__(opt, precision_policy=precision_policy)
            self._enc_rec = _EncodeRecorder(self.model)
            self._enc_rec.install()
            self._fusion_rec = _FusionRecorder(self.model.fusion)
            self._fusion_rec.install()
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
                        "pos": keep.nonzero(as_tuple=False).flatten(),
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
                mf = m.float()
                q768 = ((t.float()) * mf).sum(-1) / mf.sum(-1).clamp_min(1.0)
                v = self.proj_val(self.proj_ln(t.float()))
                q384 = (v * mf).sum(-1) / mf.sum(-1).clamp_min(1.0)
                q384 = torch.nn.functional.normalize(q384, dim=-1)
                return q768.cpu().numpy(), q384.cpu().numpy()

        def _cls_hidden(self, f1_level, mask_level):
            """Replicate the cls head hidden stack on the full sequence."""
            import torch.nn.functional as F
            x = f1_level
            for conv, norm in zip(self.model.cls_head.convs,
                                  self.model.cls_head.norms):
                x, _ = conv(x, mask_level)
                x = F.relu(norm(x), inplace=True)
            return x

        def predict(self, data):
            tokens = data["text"]
            if not isinstance(tokens, tuple):
                tokens = (tokens, )
            vid_len = data["vid"].size(-1)
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
            self._enc_rec.clear()
            self._fusion_rec.clear()
            results = super().predict(data)
            n_q = len(results)
            n_w = len(offsets_w)
            if len(self._collect_log) != n_w * n_q:
                raise RuntimeError("collect calls != windows x queries")
            if len(self._fusion_rec.outputs) != 2 * n_w:
                raise RuntimeError("fusion calls != 2 x windows")
            if len(self._enc_rec.outputs) != n_w:
                raise RuntimeError("encode calls != windows")
            q768, q384 = self._query_vecs()
            segments = data["segment"]
            for q in range(n_q):
                merged = None
                f0_rows, f1_rows, f2_rows = [], [], []
                qn = torch.from_numpy(q384[q]).cuda()
                with torch.no_grad(), torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16):
                    for w in range(n_w):
                        per_level = self._collect_log[w * n_q + q]
                        fpn, _ = self._enc_rec.outputs[w]
                        f1, f1_masks = self._fusion_rec.outputs[2 * w + 1]
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
                            centers = entry[:, 3] + offsets_w[w]
                            entry[:, 7] = centers - entry[:, 5] * entry[:, 4]
                            entry[:, 8] = centers + entry[:, 6] * entry[:, 4]
                            merged = (entry if merged is None
                                      else np.concatenate([merged, entry]))
                            lvl = int(pl["level"])
                            pos = pl["pos"].to(fpn[lvl].device)
                            f0_rows.append(
                                fpn[lvl][q][:, pos].transpose(0, 1)
                                .detach().float().cpu())
                            f1_full = f1[lvl]
                            mask_l = f1_masks[lvl]
                            f2_full = self._cls_hidden(f1_full, mask_l)
                            f1_rows.append(
                                f1_full[q][:, pos.to(f1_full.device)]
                                .transpose(0, 1).detach().float().cpu())
                            f2_rows.append(
                                f2_full[q][:, pos.to(f2_full.device)]
                                .transpose(0, 1).detach().float().cpu())
                            if len(self.f2_check) < 40:
                                final = self.model.cls_head.cls_head(
                                    f2_full, mask_l)[0][q][0]
                                got = final[pos]
                                want = pl["logit"].to(got.device)
                                self.f2_check.append(float(
                                    (got.float() - want.float())
                                    .abs().max()))
                if merged is None:
                    merged = np.zeros((0, 9), dtype=np.float64)
                z = torch.zeros(0, FEAT_DIM)
                f0 = torch.cat(f0_rows) if f0_rows else z
                f1 = torch.cat(f1_rows) if f1_rows else z
                f2 = torch.cat(f2_rows) if f2_rows else z
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
                    "q768": q768[q].astype(np.float32),
                    "q384": q384[q].astype(np.float32),
                }
                ious = iou_seconds(rec, merged) if merged.shape[0] \
                    else np.zeros(0)
                if merged.shape[0]:
                    cos = {}
                    for nm, ft in (("F0", f0), ("F1", f1), ("F2", f2)):
                        fn = torch.nn.functional.normalize(
                            ft.cuda().float(), dim=-1)
                        cos[nm] = (fn @ qn).cpu().numpy().astype(np.float32)
                    rec["cos"] = cos
                    if rank_heads is not None:
                        geo = np.stack([
                            merged[:, 5], merged[:, 6],
                            np.log1p(np.maximum(
                                merged[:, 8] - merged[:, 7], 0.0)),
                        ], axis=1).astype(np.float32)
                        g = torch.from_numpy(geo).cuda()
                        lv = torch.from_numpy(
                            merged[:, 2]).long().cuda()
                        for nm, ft in (("F0", f0), ("F1", f1), ("F2", f2)):
                            with torch.no_grad():
                                rec[f"score_{nm}"] = torch.sigmoid(
                                    rank_heads[nm](
                                        ft.cuda().float(), g, lv)
                                ).cpu().numpy().astype(np.float32)
                else:
                    rec["cos"] = {k: np.zeros(0, np.float32)
                                  for k in ("F0", "F1", "F2")}
                if sample and merged.shape[0]:
                    sel = balanced_select(self.rng, ious)
                    rec["sample_pool"] = merged[sel].astype(np.float32)
                    rec["sample_ious"] = ious[sel].astype(np.float32)
                    rec["sample_F0"] = f0.numpy()[sel].astype(np.float16)
                    rec["sample_F1"] = f1.numpy()[sel].astype(np.float16)
                    rec["sample_F2"] = f2.numpy()[sel].astype(np.float16)
                    for nm in ("F0", "F1", "F2"):
                        rec[f"sample_cos_{nm}"] = rec["cos"][nm][sel]
                self.records.append(rec)
            return results

    ev = QcEvaluator(opt, precision_policy="p3")
    if limit_videos:
        ev.dataloader = list(itertools.islice(
            iter(ev.dataloader), limit_videos))
        ev.num_itrs = len(ev.dataloader)
    return ev


def make_rank_head():
    import torch.nn as nn

    class RankHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.feat_proj = nn.Sequential(
                nn.Linear(FEAT_DIM, 128), nn.ReLU())
            self.geo_proj = nn.Sequential(nn.Linear(3, 32), nn.ReLU())
            self.out = nn.Linear(160, 1)

        def forward(self, f, geo, level):
            return self.out(torch.cat(
                [self.feat_proj(f), self.geo_proj(geo)], -1)).squeeze(-1)

    return RankHead()


def run_dump(split, limit_videos=None):
    ev = build_evaluator(split, limit_videos=limit_videos)
    ev.run()
    if ev.f2_check:
        print(f"[{split}] F2 recompute check: max|Δlogit| over "
              f"{len(ev.f2_check)} levels = {max(ev.f2_check):.5f} "
              f"(median {sorted(ev.f2_check)[len(ev.f2_check)//2]:.5f})")
    recs = ev.records
    rows = np.concatenate([r["sample_pool"] for r in recs])
    ious = np.concatenate([r["sample_ious"] for r in recs])
    feats = {k: np.concatenate([r[f"sample_{k}"] for r in recs])
             for k in ("F0", "F1", "F2")}
    q768 = np.stack([r["q768"] for r in recs])
    q384 = np.stack([r["q384"] for r in recs])
    gt = np.stack([[r["gt_start"], r["gt_end"]] for r in recs])
    qi = np.concatenate([
        np.full(len(r["sample_ious"]), i, dtype=np.int64)
        for i, r in enumerate(recs)])
    inside = np.concatenate([
        (center_seconds(r, r["sample_pool"].astype(np.float64))
         >= r["gt_start"]) & (center_seconds(
             r, r["sample_pool"].astype(np.float64)) <= r["gt_end"])
        for r in recs])
    out = TRAIN_NPZ if split == "train" else VAL_NPZ
    np.savez_compressed(
        out, rows=rows, ious=ious, qi=qi, inside_gt=inside,
        q768=q768, q384=q384, gt=gt,
        F0=feats["F0"], F1=feats["F1"], F2=feats["F2"])
    hist = np.histogram(ious, bins=[0, .1, .3, .5, 1.01])[0]
    print(f"[{split}] sampled {len(rows)}  buckets={hist.tolist()} -> {out}")


def run_train_probes():
    """Linear IoU probes + IoU-regression rank heads for F0/F1/F2."""
    tr = np.load(TRAIN_NPZ)
    va = np.load(VAL_NPZ)
    dev = "cuda:0"

    def lin_probe(name, xtr, xva):
        ytr = np.array([bucket_id(v) for v in tr["ious"]])
        yva = np.array([bucket_id(v) for v in va["ious"]])
        torch.manual_seed(11)
        lin = torch.nn.Linear(xtr.shape[1], 4).to(dev)
        opt = torch.optim.Adam(lin.parameters(), lr=1e-3,
                               weight_decay=1e-4)
        X = torch.tensor(xtr, dtype=torch.float32)
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
                lin(torch.tensor(xva[i:i + (1 << 17)],
                                 dtype=torch.float32).to(dev)).cpu()
                for i in range(0, len(yva), 1 << 17)]).numpy()
        pred = logits.argmax(1)
        f1s = []
        for c in range(4):
            tp = float(((pred == c) & (yva == c)).sum())
            fp = float(((pred == c) & (yva != c)).sum())
            fn = float(((pred != c) & (yva == c)).sum())
            f1s.append(2 * tp / max(2 * tp + fp + fn, 1e-9))
        # pairwise ranking accuracy within query (sampled rows)
        def pairwise(probs):
            accs = []
            for qid in np.unique(va["qi"]):
                m = va["qi"] == qid
                if m.sum() < 4:
                    continue
                p = probs[m]
                iv = va["ious"][m]
                ii, jj = np.triu_indices(len(p), k=1)
                d = iv[ii] - iv[jj]
                ok = np.abs(d) > 1e-6
                if ok.sum() < 4:
                    continue
                accs.append(float((np.sign(p[ii][ok] - p[jj][ok])
                                   == np.sign(d[ok])).mean()))
            return float(np.mean(accs)), len(accs)
        pw, n_pw = pairwise(pred.astype(np.float64))
        return {
            "accuracy": float((pred == yva).mean()),
            "macro_f1": float(np.mean(f1s)),
            "per_class_f1": f1s,
            "macro_auroc": auroc_ovr(logits, yva),
            "pairwise_within_query": pw,
            "pairwise_n_queries": n_pw,
        }

    def auroc_ovr(scores, labels, n_classes=4):
        aucs = []
        for c in range(n_classes):
            pos = labels == c
            npos, nneg = int(pos.sum()), int((~pos).sum())
            if npos == 0 or nneg == 0:
                continue
            r = rankdata_avg(scores[:, c])
            aucs.append((r[pos].sum() - npos * (npos + 1) / 2)
                        / (npos * nneg))
        return float(np.mean(aucs))

    results = {}
    for nm in ("F0", "F1", "F2"):
        results[f"probe_{nm}"] = lin_probe(nm, tr[nm], va[nm])
        r = results[f"probe_{nm}"]
        print(f"Probe({nm}): acc={r['accuracy']:.4f} "
              f"macroF1={r['macro_f1']:.4f} "
              f"AUROC={r['macro_auroc']:.4f} "
              f"pairwise={r['pairwise_within_query']:.4f} "
              f"(n={r['pairwise_n_queries']})")

    # ---- IoU-regression rank heads (for the §5 ranking pass) ----
    def geo_of(rows):
        return np.stack([
            rows[:, 5], rows[:, 6],
            np.log1p(np.maximum(rows[:, 8] - rows[:, 7], 0.0)),
        ], axis=1).astype(np.float32)

    rank_heads = {}
    curves = {}
    for nm in ("F0", "F1", "F2"):
        torch.manual_seed(7)
        head = make_rank_head().to(dev)
        opt = torch.optim.Adam(head.parameters(), lr=1e-3)
        F = torch.tensor(tr[nm], dtype=torch.float32)
        G = torch.tensor(geo_of(tr["rows"]))
        L = torch.tensor(tr["rows"][:, 2], dtype=torch.long)
        Y = torch.tensor(tr["ious"], dtype=torch.float32)
        n = len(Y)
        best = {"sp": -2.0, "state": None}
        curve = []
        for ep in range(15):
            head.train()
            perm = torch.randperm(n)
            for i in range(0, n, 8192):
                idx = perm[i:i + 8192]
                q = torch.sigmoid(head(
                    F[idx].to(dev), G[idx].to(dev), L[idx].to(dev)))
                loss = torch.nn.functional.binary_cross_entropy(
                    q, Y[idx].to(dev).clamp(1e-4, 1 - 1e-4))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            head.eval()
            with torch.no_grad():
                qs = torch.cat([
                    torch.sigmoid(head(
                        torch.tensor(va[nm][i:i + (1 << 16)],
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
                    k: v.cpu().clone()
                    for k, v in head.state_dict().items()}}
        rank_heads[nm] = best["state"]
        curves[nm] = curve
        print(f"RankHead({nm}): best val Spearman={best['sp']:.4f}")
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
    for nm in ("F0", "F1", "F2"):
        arrays[f"cos_{nm}"] = np.concatenate(
            [r["cos"][nm] for r in recs])
        arrays[f"score_{nm}"] = np.concatenate(
            [r.get(f"score_{nm}", np.zeros(r["pool"].shape[0]))
             for r in recs])
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

    # ---- §3 alignment ----
    lv = va["rows"][:, 2].astype(int)
    inside = va["inside_gt"]

    def auc_binary(score, pos):
        p, n = int(pos.sum()), int((~pos).sum())
        if p == 0 or n == 0:
            return float("nan")
        r = rankdata_avg(score)
        return float((r[pos].sum() - p * (p + 1) / 2) / (p * n))

    align = []
    qn_all = va["q384"][va["qi"]]          # (N, 384) per-candidate query vec
    qn_norm = qn_all / np.maximum(
        np.linalg.norm(qn_all, axis=-1, keepdims=True), 1e-8)
    for nm in ("F0", "F1", "F2"):
        fmat = va[nm].astype(np.float32)
        fnorm = fmat / np.maximum(
            np.linalg.norm(fmat, axis=-1, keepdims=True), 1e-8)
        cs = (fnorm * qn_norm).sum(-1)
        rows = []
        for l in range(8):
            m = lv == l
            if m.sum() < 100:
                continue
            rows.append({
                "level": l, "n": int(m.sum()),
                "cos_mean": float(cs[m].mean()),
                "cos_median": float(np.median(cs[m])),
                "auc_inside_gt": auc_binary(cs[m], inside[m]),
            })
        align.append({"feature": nm, "overall_cos_mean": float(cs.mean()),
                      "overall_auc": auc_binary(cs, inside), "per_level": rows})
    out["alignment"] = align

    # ---- §5 ranking ----
    pool_all = rk["pool"]
    offs = rk["pool_offsets"]
    ious_all = rk["ious"]
    cols = {nm: rk[f"cos_{nm}"] for nm in ("F0", "F1", "F2")}
    scols = {nm: rk[f"score_{nm}"] for nm in ("F0", "F1", "F2")}
    gt_s, gt_e = rk["gt_start"], rk["gt_end"]
    cs_c, cc_c = rk["clip_stride"], rk["clip_size"]
    fps_c, dur_c = rk["fps"], rk["duration"]
    n = len(gt_s)
    variants = (["cls"] + [f"score_{k}" for k in ("F0", "F1", "F2")]
                + [f"cos_{k}" for k in ("F0", "F1", "F2")])
    o10 = {v: 0.0 for v in variants}
    r15 = {v: 0.0 for v in variants}
    for i in range(n):
        sl = slice(offs[i], offs[i + 1])
        p = pool_all[sl]
        if p.shape[0] == 0:
            continue
        base = np.argsort(-p[:, 1], kind="stable")
        p = p[base][:2000]
        iou = ious_all[sl][base][:2000]
        scores = {"cls": p[:, 1]}
        for nm in ("F0", "F1", "F2"):
            scores[f"cos_{nm}"] = cols[nm][sl][base][:2000]
            scores[f"score_{nm}"] = scols[nm][sl][base][:2000]
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
    print("\n=== alignment (overall) ===")
    for a in align:
        print(f"{a['feature']}: cos_mean={a['overall_cos_mean']:+.4f} "
              f"inside-GT AUC={a['overall_auc']:.4f}")
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

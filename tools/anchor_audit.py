#!/usr/bin/env python3
"""Anchor Information Audit (HM-ANCHOR-AUDIT-023).

Frozen A-U-clean-P32 backbone. Question: does the pooled anchor
representation carry temporal-grounding information beyond the sequence
representation?

Stages:
  dump-train / dump-val : balanced IoU-bucket candidate sample with BOTH
      the raw sequence feature (encode_video fpn at the candidate point)
      and its parent anchor feature (anchor_fpn[level][pos//2]), plus
      cosine similarities to the frozen query projection.
  rank    : val pass storing sim_seq / sim_anchor for every pool
      candidate (on the fly), for ranking probes.
  probes  : frozen linear probes P1(seq) / P2(anchor) / P3(concat) ->
      4-way IoU bucket [0,.1) [.1,.3) [.3,.5) [.5,1]; acc / macro-F1 /
      macro AUROC on the val sample.
  analyze : alignment stats, probe table, ranking table, verdict data.

No model / loss modification, no base-model training, no decoder.
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

AUC_P32_ROOT = ROOT / "experiments/allocator_diagnosis/seed_1/AUC_P32"
AROOT = ROOT / "experiments/anchor_audit"
TRAIN_NPZ = AROOT / "train_candidates.npz"
VAL_NPZ = AROOT / "val_candidates.npz"
VAL_RANK_NPZ = AROOT / "val_rank.npz"
RESULTS_JSON = AROOT / "anchor_audit_results.json"

BUCKETS = ((0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 1.01))
LOW_CAP = 60
FEAT_DIM = 384


def bucket_id(v):
    if v < 0.1:
        return 0
    if v < 0.3:
        return 1
    if v < 0.5:
        return 2
    return 3


def balanced_select(rng, ious):
    ids = np.array([bucket_id(v) for v in ious])
    sel = []
    for b in range(len(BUCKETS)):
        idxs = np.where(ids == b)[0]
        if b == 0 and len(idxs) > LOW_CAP:
            idxs = rng.choice(idxs, LOW_CAP, replace=False)
        sel.append(idxs)
    return np.sort(np.concatenate(sel))


class _EncodeRecorder:
    """Wrap model.encode_video; record (fpn, anchor_fpn) per call."""

    def __init__(self, module):
        self.module = module
        self.outputs = []
        self._original = module.encode_video

    def install(self):
        rec = self

        def rec_forward(*args, **kwargs):
            out = rec._original(*args, **kwargs)
            rec.outputs.append((out[0], out[2]))
            return out

        self.module.encode_video = rec_forward

    def clear(self):
        self.outputs = []


def build_evaluator(split, limit_videos=None, sample=True):
    import tools.run_formal_ablation as rfa
    from libs import load_opt

    AROOT.mkdir(parents=True, exist_ok=True)
    opt_path = AROOT / f"opt_{split}.yaml"
    if not opt_path.exists():
        shutil.copyfile(AUC_P32_ROOT / "opt.yaml", opt_path)
    opt = load_opt(str(opt_path), is_training=False)
    opt["eval"]["data"]["split"] = split
    opt["_root"] = str(AROOT)
    opt["_ckpt"] = "last"
    models_link = AROOT / "models"
    if not models_link.exists():
        models_link.symlink_to(AU_ROOT / "models", target_is_directory=True)

    # frozen query projector: ln_xattn_kv + value conv of fusion layer 0
    fusion = None

    class AnchorEvaluator(rfa.FormalEvaluator):
        def __init__(self, opt, precision_policy="p2"):
            self.records = []
            self._collect_log = []
            self._enc_rec = None
            self._text_pool = None
            self.rng = np.random.default_rng(4321)
            super().__init__(opt, precision_policy=precision_policy)
            self._enc_rec = _EncodeRecorder(self.model)
            self._enc_rec.install()
            m = self.model
            layer0 = m.fusion.layers[0]
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

        def _query_vec(self):
            """Frozen query projection -> (Q, 384), L2-normalised."""
            t, m = self._text_pool          # (Q, 768, L), (Q, 1, L)
            with torch.no_grad():
                v = self.proj_val(self.proj_ln(t.float()))
                mf = m.float()
                q = (v * mf).sum(-1) / mf.sum(-1).clamp_min(1.0)
                return torch.nn.functional.normalize(q, dim=-1)

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
            results = super().predict(data)
            n_q = len(results)
            n_w = len(offsets_w)
            if len(self._collect_log) != n_w * n_q:
                raise RuntimeError("collect calls != windows x queries")
            if len(self._enc_rec.outputs) != n_w:
                raise RuntimeError("encode calls != windows")
            qvec = self._query_vec().cpu().numpy().astype(np.float32)
            if not hasattr(self, "shape_note") and self._enc_rec.outputs:
                fpn0, anc0 = self._enc_rec.outputs[0]
                self.shape_note = [
                    (lvl, tuple(fpn0[lvl].shape), tuple(anc0[lvl].shape))
                    for lvl in range(len(fpn0))
                ]
            segments = data["segment"]
            for q in range(n_q):
                merged = None
                seq_rows, anc_rows = [], []
                for w in range(n_w):
                    per_level = self._collect_log[w * n_q + q]
                    fpn, anchor_fpn = self._enc_rec.outputs[w]
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
                        pos = pl["pos"]
                        seq_rows.append(
                            fpn[lvl][q][:, pos.to(fpn[lvl].device)]
                            .transpose(0, 1).detach().float().cpu())
                        anc_idx = (pos // 2).clamp(
                            max=anchor_fpn[lvl].size(-1) - 1)
                        anc_rows.append(
                            anchor_fpn[lvl][q][
                                :, anc_idx.to(anchor_fpn[lvl].device)]
                            .transpose(0, 1).detach().float().cpu())
                if merged is None:
                    merged = np.zeros((0, 9), dtype=np.float64)
                seq_f = (torch.cat(seq_rows) if seq_rows
                         else torch.zeros(0, FEAT_DIM))
                anc_f = (torch.cat(anc_rows) if anc_rows
                         else torch.zeros(0, FEAT_DIM))
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
                if merged.shape[0]:
                    qn = torch.from_numpy(qvec[q])
                    sim_s = torch.nn.functional.cosine_similarity(
                        seq_f, qn[None], dim=-1).numpy()
                    sim_a = torch.nn.functional.cosine_similarity(
                        anc_f, qn[None], dim=-1).numpy()
                    rec["sim_seq"] = sim_s.astype(np.float32)
                    rec["sim_anchor"] = sim_a.astype(np.float32)
                else:
                    rec["sim_seq"] = np.zeros(0, np.float32)
                    rec["sim_anchor"] = np.zeros(0, np.float32)
                if sample and merged.shape[0]:
                    sel = balanced_select(self.rng, ious)
                    rec["sample_pool"] = merged[sel].astype(np.float32)
                    rec["sample_ious"] = ious[sel].astype(np.float32)
                    rec["sample_seq"] = seq_f.numpy()[sel].astype(np.float16)
                    rec["sample_anc"] = anc_f.numpy()[sel].astype(np.float16)
                    rec["sample_sim_seq"] = sim_s[sel]
                    rec["sample_sim_anchor"] = sim_a[sel]
                self.records.append(rec)
            return results

    ev = AnchorEvaluator(opt, precision_policy="p3")
    if limit_videos:
        ev.dataloader = list(itertools.islice(
            iter(ev.dataloader), limit_videos))
        ev.num_itrs = len(ev.dataloader)
    return ev


def iou_seconds(rec, pool):
    cs, cc, fps, dur = (rec["clip_stride"], rec["clip_size"],
                        rec["fps"], rec["duration"])
    s = np.clip((pool[:, 7] * cs + 0.5 * cc) / fps, 0, dur)
    e = np.clip((pool[:, 8] * cs + 0.5 * cc) / fps, 0, dur)
    return iou_1d(s, e, rec["gt_start"], rec["gt_end"])


def center_seconds(rec, pool):
    cs, cc, fps = rec["clip_stride"], rec["clip_size"], rec["fps"]
    return (pool[:, 3] * cs + 0.5 * cc) / fps


def run_dump(split, limit_videos=None):
    ev = build_evaluator(split, limit_videos=limit_videos)
    ev.run()
    if hasattr(ev, "shape_note"):
        print(f"[{split}] per-level shapes (fpn sequence, anchor):")
        for lvl, fs, as_ in ev.shape_note:
            print(f"  L{lvl}: fpn {fs}  anchor {as_}")
    recs = ev.records
    rows = np.concatenate([r["sample_pool"] for r in recs])
    ious = np.concatenate([r["sample_ious"] for r in recs])
    seq = np.concatenate([r["sample_seq"] for r in recs])
    anc = np.concatenate([r["sample_anc"] for r in recs])
    ssim = np.concatenate([r["sample_sim_seq"] for r in recs])
    asim = np.concatenate([r["sample_sim_anchor"] for r in recs])
    inside = np.concatenate([
        (center_seconds(r, r["sample_pool"].astype(np.float64))
         >= r["gt_start"]) & (center_seconds(
             r, r["sample_pool"].astype(np.float64)) <= r["gt_end"])
        for r in recs
    ])
    out = TRAIN_NPZ if split == "train" else VAL_NPZ
    np.savez_compressed(
        out, rows=rows, ious=ious, seq=seq, anc=anc,
        sim_seq=ssim, sim_anchor=asim, inside_gt=inside)
    hist = np.histogram(ious, bins=[0, .1, .3, .5, 1.01])[0]
    print(f"[{split}] sampled {len(rows)}  buckets={hist.tolist()} -> {out}")

    if split == "val":
        n = len(recs)
        offsets = np.zeros(n + 1, dtype=np.int64)
        np.cumsum([r["pool"].shape[0] for r in recs], out=offsets[1:])
        arrays = {
            "pool": np.concatenate([r["pool"] for r in recs]),
            "pool_offsets": offsets,
            "ious": np.concatenate([
                iou_seconds(r, r["pool"]) if r["pool"].shape[0]
                else np.zeros(0) for r in recs]),
            "sim_seq": np.concatenate([r["sim_seq"] for r in recs]),
            "sim_anchor": np.concatenate([r["sim_anchor"] for r in recs]),
        }
        for k in ("gt_start", "gt_end", "clip_stride", "clip_size",
                  "fps", "duration"):
            arrays[k] = np.array([r[k] for r in recs], dtype=np.float64)
        arrays["vid_ids"] = np.array([r["vid_id"] for r in recs])
        arrays["query_idx"] = np.array(
            [r["query_idx"] for r in recs], dtype=np.int64)
        np.savez_compressed(VAL_RANK_NPZ, **arrays)
        print(f"val full pool ({int(offsets[-1])}) -> {VAL_RANK_NPZ}")


# ---------------------------------------------------------------------------
# probes & analysis
# ---------------------------------------------------------------------------

def auroc_ovr(scores, labels, n_classes=4):
    """Macro one-vs-rest AUROC via rank statistic (no sklearn)."""
    aucs = []
    for c in range(n_classes):
        pos = labels == c
        neg = ~pos
        npos, nneg = int(pos.sum()), int(neg.sum())
        if npos == 0 or nneg == 0:
            continue
        r = rankdata_avg(scores[:, c])
        aucs.append(
            (r[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))
    return float(np.mean(aucs)) if aucs else float("nan")


def run_probes():
    tr = np.load(TRAIN_NPZ)
    va = np.load(VAL_NPZ)
    dev = "cuda:0"

    ytr = np.array([bucket_id(v) for v in tr["ious"]])
    yva = np.array([bucket_id(v) for v in va["ious"]])
    out = {}
    for name, feat_key in (
        ("P1_sequence", "seq"), ("P2_anchor", "anc"),
        ("P3_seq+anchor", "concat"),
    ):
        if feat_key == "concat":
            xtr = np.concatenate([tr["seq"], tr["anc"]], axis=1)
            xva = np.concatenate([va["seq"], va["anc"]], axis=1)
        else:
            xtr, xva = tr[feat_key], va[feat_key]
        torch.manual_seed(11)
        dim = xtr.shape[1]
        lin = torch.nn.Linear(dim, 4).to(dev)
        opt = torch.optim.Adam(lin.parameters(), lr=1e-3, weight_decay=1e-4)
        Xtr = torch.tensor(xtr, dtype=torch.float32)
        Ytr = torch.tensor(ytr, dtype=torch.long)
        n = len(Ytr)
        for ep in range(12):
            perm = torch.randperm(n)
            for i in range(0, n, 8192):
                idx = perm[i:i + 8192]
                loss = torch.nn.functional.cross_entropy(
                    lin(Xtr[idx].to(dev)), Ytr[idx].to(dev))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        with torch.no_grad():
            logits_va = torch.cat([
                lin(torch.tensor(xva[i:i + (1 << 17)],
                                 dtype=torch.float32).to(dev)).cpu()
                for i in range(0, len(yva), 1 << 17)]).numpy()
        pred = logits_va.argmax(1)
        acc = float((pred == yva).mean())
        f1s = []
        for c in range(4):
            tp = float(((pred == c) & (yva == c)).sum())
            fp = float(((pred == c) & (yva != c)).sum())
            fn = float(((pred != c) & (yva == c)).sum())
            f1s.append(2 * tp / max(2 * tp + fp + fn, 1e-9))
        out[name] = {
            "accuracy": acc,
            "macro_f1": float(np.mean(f1s)),
            "per_class_f1": f1s,
            "macro_auroc": auroc_ovr(logits_va, yva),
        }
        print(f"{name}: acc={acc:.4f} macroF1={out[name]['macro_f1']:.4f} "
              f"AUROC={out[name]['macro_auroc']:.4f} "
              f"F1s={[round(x,3) for x in f1s]}")
    return out


def run_analyze():
    va = np.load(VAL_NPZ)
    rk = np.load(VAL_RANK_NPZ)
    out = {"probes": run_probes()}

    # ---- §2 alignment analysis ----
    lv = va["rows"][:, 2].astype(int)
    inside = va["inside_gt"]
    align = {"per_level": []}
    for l in range(8):
        m = lv == l
        if m.sum() < 100:
            continue
        align["per_level"].append({
            "level": l, "n": int(m.sum()),
            "cos_seq_mean": float(va["sim_seq"][m].mean()),
            "cos_seq_median": float(np.median(va["sim_seq"][m])),
            "cos_anchor_mean": float(va["sim_anchor"][m].mean()),
            "cos_anchor_median": float(np.median(va["sim_anchor"][m])),
            "cos_seq_inside_gt": float(
                va["sim_seq"][m & inside].mean()),
            "cos_seq_outside_gt": float(
                va["sim_seq"][m & ~inside].mean()),
            "cos_anchor_inside_gt": float(
                va["sim_anchor"][m & inside].mean()),
            "cos_anchor_outside_gt": float(
                va["sim_anchor"][m & ~inside].mean()),
        })
    align["overall"] = {
        "cos_seq_mean": float(va["sim_seq"].mean()),
        "cos_anchor_mean": float(va["sim_anchor"].mean()),
        "inside_gt_rate": float(inside.mean()),
    }
    out["alignment"] = align

    # inside-GT discrimination AUC per level (does cos locate the GT?)
    def auc_binary(score, pos):
        p, n = pos.sum(), (~pos).sum()
        if p == 0 or n == 0:
            return float("nan")
        r = rankdata_avg(score)
        return float((r[pos].sum() - p * (p + 1) / 2) / (p * n))

    loc = []
    for l in range(8):
        m = lv == l
        if m.sum() < 100:
            continue
        loc.append({
            "level": l,
            "auc_seq": auc_binary(va["sim_seq"][m], inside[m]),
            "auc_anchor": auc_binary(va["sim_anchor"][m], inside[m]),
            "auc_seq_plus_anchor": auc_binary(
                va["sim_seq"][m] + va["sim_anchor"][m], inside[m]),
        })
    out["gt_localization_auc"] = loc

    # ---- §4 ranking probe ----
    # hoist all lazy npz arrays out of the per-query loop
    pool_all = rk["pool"]
    offs = rk["pool_offsets"]
    ious_all = rk["ious"]
    sim_seq_all = rk["sim_seq"]
    sim_anchor_all = rk["sim_anchor"]
    n = len(rk["gt_start"])
    gt_s, gt_e = rk["gt_start"], rk["gt_end"]
    cs, cc = rk["clip_stride"], rk["clip_size"]
    fps, dur = rk["fps"], rk["duration"]

    def rk_pipeline(order_scores, iou, pool_top, i):
        k_segs, k_scores = soft_nms(
            np.stack([pool_top[:, 7], pool_top[:, 8]], axis=1),
            order_scores)
        s = np.clip((k_segs[:, 0] * cs[i] + 0.5 * cc[i]) / fps[i], 0,
                    dur[i])
        e = np.clip((k_segs[:, 1] * cs[i] + 0.5 * cc[i]) / fps[i], 0,
                    dur[i])
        q_i = iou_1d(s, e, gt_s[i], gt_e[i])
        return (q_i[0] if len(q_i) else 0.0)

    variants = ("cls", "sim_seq", "sim_anchor", "sim_seq*sim_anchor")
    o10 = {v: 0 for v in variants}
    r1_5 = {v: 0.0 for v in variants}
    for i in range(n):
        sl = slice(offs[i], offs[i + 1])
        p = pool_all[sl]
        if p.shape[0] == 0:
            continue
        base = np.argsort(-p[:, 1], kind="stable")
        p = p[base][:2000]
        iou = ious_all[sl][base][:2000]
        ss = sim_seq_all[sl][base][:2000]
        sa = sim_anchor_all[sl][base][:2000]
        scores = {
            "cls": p[:, 1],
            "sim_seq": ss,
            "sim_anchor": sa,
            "sim_seq*sim_anchor": ss * sa,
        }
        for v in variants:
            sc = scores[v]
            order = np.argsort(-sc, kind="stable")
            o10[v] += float(iou[order[:10]].max() >= 0.5)
            top = p[:2000]
            r1_5[v] += float(rk_pipeline(sc, iou, top, i) >= 0.5)
    out["ranking_probe"] = {
        v: {"oracle@10@0.5": o10[v] / n, "R1@0.5": r1_5[v] / n}
        for v in variants
    }
    out["n_queries"] = n

    RESULTS_JSON.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print("\n=== ranking probe ===")
    for v in variants:
        r = out["ranking_probe"][v]
        print(f"{v:22s} Oracle@10@0.5={100*r['oracle@10@0.5']:.2f} "
              f"R1@0.5={100*r['R1@0.5']:.2f}")
    print("results ->", RESULTS_JSON)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=[
        "dump-train", "dump-val", "rank", "probes", "analyze"])
    ap.add_argument("--limit-videos", type=int, default=None)
    args = ap.parse_args()
    if args.stage == "dump-train":
        run_dump("train", args.limit_videos)
    elif args.stage == "dump-val":
        run_dump("val", args.limit_videos)
    elif args.stage == "rank":
        # ranking scalars are produced by dump-val's full-pool pass
        run_dump("val", args.limit_videos)
    elif args.stage == "probes":
        run_probes()
    else:
        run_analyze()


if __name__ == "__main__":
    main()

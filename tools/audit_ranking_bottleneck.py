#!/usr/bin/env python3
"""Proposal-Ranking Bottleneck Audit (HM-AUDIT-RANK-021).

No training, no architecture change, no retuning. Three modes:

  repro   : rerun the official FormalEvaluator protocol on the A-U-clean
            seed-1 checkpoint and compare against evaluation_summary.json.
  dump    : official predict pass with two recorders installed:
            (1) a wrapper around _collect_segments that recomputes the
                per-level point scores from the very same input tensors the
                official body consumes (bit-identical scores, no second
                forward), keeping the full pre-NMS pool per query;
            (2) a wrapper around batched_nms that records its exact inputs
                (global top-2000) and outputs per query.
  analyze : pure-numpy analysis over the dump.

Ranking conclusions are anchored by re-deriving official R@1/R@5 from the
dumped pool alone (pool -> top-2000 -> numpy soft-NMS replica -> top-5).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "hydra"), str(ROOT / "tools")]

AU_ROOT = ROOT / "experiments/allocator_diagnosis/seed_1/AUC"
AUDIT_ROOT = ROOT / "experiments/audit_ranking_bottleneck"
DUMP_PATH = AUDIT_ROOT / "prenms_dump.npz"
ANALYSIS_PATH = AUDIT_ROOT / "analysis_results.json"

METRIC_KEYS = ("Rank@1_IoU@0.3", "Rank@1_IoU@0.5",
               "Rank@5_IoU@0.3", "Rank@5_IoU@0.5")
POOL_COLS = ("score", "level", "center", "scale", "off_l", "off_r",
             "start_tok", "end_tok")


# ---------------------------------------------------------------------------
# modes: repro and dump (both need torch + CUDA)
# ---------------------------------------------------------------------------

def _limit(evaluator, limit_videos):
    """run() iterates the whole dataloader; truncate it for smoke tests."""
    evaluator.dataloader = list(
        itertools.islice(iter(evaluator.dataloader), limit_videos)
    )
    evaluator.num_itrs = len(evaluator.dataloader)


def build_evaluator(audit_root: Path, dump: bool):
    import shutil
    import torch
    from libs import load_opt
    import tools.run_formal_ablation as rfa

    audit_root.mkdir(parents=True, exist_ok=True)
    opt_path = audit_root / "opt.yaml"
    if not opt_path.exists():
        shutil.copyfile(AU_ROOT / "opt.yaml", opt_path)
    models_link = audit_root / "models"
    if not models_link.exists():
        models_link.symlink_to(AU_ROOT / "models",
                               target_is_directory=True)

    opt = load_opt(str(opt_path), is_training=False)
    opt["_root"] = str(audit_root)
    opt["_ckpt"] = "last"
    if dump:
        class DumpEvaluator(rfa.FormalEvaluator):
            def __init__(self, opt, precision_policy="p2"):
                self.dumped_records = []
                self._collect_log = []
                self._nms_log = []
                super().__init__(opt, precision_policy=precision_policy)
                original_collect = self._collect_segments
                original_nms = self.batched_nms

                def collect_wrapper(fpn_points, fpn_logits, fpn_offsets,
                                    fpn_masks, ext_scores=None,
                                    return_levels=False, fpn_feats=None,
                                    query_idx=0):
                    result = original_collect(
                        fpn_points, fpn_logits, fpn_offsets, fpn_masks,
                        ext_scores, return_levels=return_levels,
                        fpn_feats=fpn_feats, query_idx=query_idx,
                    )
                    # recompute the score pipeline on the same tensors:
                    # sigmoid(logits[0]) * mask, keep > pre_nms_thresh
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
                        scores = torch.sigmoid(logits)
                        scores = scores * mask.float()
                        keep = scores > self.pre_nms_thresh
                        per_level.append({
                            "level": level,
                            "score": scores[keep].detach().float().cpu(),
                            "center": points[:, 0][keep].detach().float().cpu(),
                            "scale": points[:, 3][keep].detach().float().cpu(),
                            "offset": offsets[keep].detach().float().cpu(),
                        })
                    self._collect_log.append(per_level)
                    return result
                def nms_wrapper(segs, scores):
                    out = original_nms(segs, scores)
                    self._nms_log.append((
                        segs.detach().float().cpu().numpy().copy(),
                        scores.detach().float().cpu().numpy().copy(),
                        out[0].detach().float().cpu().numpy().copy(),
                        out[1].detach().float().cpu().numpy().copy(),
                    ))
                    return out

                self._collect_segments = collect_wrapper
                self.batched_nms = nms_wrapper

            def predict(self, data):
                import torch  # noqa: F401
                tokens = data["text"]
                if not isinstance(tokens, tuple):
                    tokens = (tokens, )
                vid_len = data["vid"].size(-1)
                window_size = min(self.window_size or vid_len, vid_len)
                window_stride = self.window_stride or window_size
                n_windows = (
                    1 if vid_len <= window_size
                    else int(math.ceil((vid_len - window_size)
                                       / window_stride)) + 1
                )
                self._collect_log = []
                self._nms_log = []
                results = super().predict(data)
                n_q = len(results)
                if n_windows != 1:
                    raise RuntimeError(
                        "audit assumes single-window evaluation, got "
                        f"{n_windows}"
                    )
                if len(self._collect_log) != n_q:
                    raise RuntimeError(
                        "collect calls {} != queries {}".format(
                            len(self._collect_log), n_q
                        )
                    )
                if len(self._nms_log) != n_q:
                    raise RuntimeError(
                        "nms calls {} != queries {}".format(
                            len(self._nms_log), n_q
                        )
                    )
                segments = data["segment"]
                for q in range(n_q):
                    per_level = self._collect_log[q]
                    merged = None
                    for pl in per_level:
                        n = pl["score"].numel()
                        entry = np.empty((n, 8), dtype=np.float64)
                        entry[:, 0] = pl["score"].numpy()
                        entry[:, 1] = pl["level"]
                        entry[:, 2] = pl["center"].numpy()
                        entry[:, 3] = pl["scale"].numpy()
                        entry[:, 4] = pl["offset"][:, 0].numpy()
                        entry[:, 5] = pl["offset"][:, 1].numpy()
                        centers = entry[:, 2]
                        entry[:, 6] = centers - entry[:, 4] * entry[:, 3]
                        entry[:, 7] = centers + entry[:, 5] * entry[:, 3]
                        # official pre-NMS length filter (token units)
                        keep = (entry[:, 7] - entry[:, 6]) > float(
                            self.seg_len_thresh
                        )
                        entry = entry[keep]
                        merged = (entry if merged is None
                                  else np.concatenate([merged, entry]))
                    if merged is None:
                        merged = np.zeros((0, 8), dtype=np.float64)
                    nms_in_segs, nms_in_scores, nms_out_segs, nms_out_scores \
                        = self._nms_log[q]
                    gt = np.asarray(segments[q], dtype=np.float64)
                    self.dumped_records.append({
                        "vid_id": str(data["vid_id"]),
                        "query_idx": q,
                        "gt_start": float(gt[0]),
                        "gt_end": float(gt[1]),
                        "clip_stride": float(data["clip_stride"]),
                        "clip_size": float(data["clip_size"]),
                        "fps": float(data["fps"]),
                        "duration": float(data["duration"]),
                        "pool": merged,
                        "nms_in_segs": nms_in_segs,
                        "nms_in_scores": nms_in_scores,
                        "nms_out_segs": nms_out_segs,
                        "nms_out_scores": nms_out_scores,
                        "official_segments": results[q][
                            "segments"].detach().cpu().numpy().astype(
                                np.float64),
                        "official_scores": results[q][
                            "scores"].detach().cpu().numpy().astype(
                                np.float64),
                    })
                return results

        evaluator = DumpEvaluator(opt, precision_policy="p3")
    else:
        evaluator = rfa.FormalEvaluator(opt, precision_policy="p3")
    return evaluator


def run_repro(limit_videos=None):
    evaluator = build_evaluator(AUDIT_ROOT / "repro", dump=False)
    if limit_videos:
        _limit(evaluator, limit_videos)
    evaluator.run()
    metrics = {
        key: float(evaluator.counts[i][j] / evaluator.text_cnt)
        for i, rank in enumerate(evaluator.ranks)
        for j, key in enumerate(
            [f"Rank@{rank}_IoU@{t:.1f}" for t in evaluator.iou_threshs]
        )
    }
    official = json.loads(
        (AU_ROOT / "evaluation_summary.json").read_text()
    )["metrics"]
    print("\n=== REPRODUCTION CHECK (limited run)" if limit_videos
          else "\n=== REPRODUCTION CHECK")
    ok = True
    for key in METRIC_KEYS:
        diff = 100 * (metrics[key] - official[key])
        flag = "OK" if abs(diff) < 0.005 else "MISMATCH"
        if flag != "OK":
            ok = False
        print(f"{key}: repro={100*metrics[key]:.4f}  "
              f"official={100*official[key]:.4f}  diff={diff:+.4f}pp  {flag}")
    print("REPRO", "PASSED" if ok else "FAILED")
    return ok


def run_dump(limit_videos=None):
    import torch  # noqa: F401
    evaluator = build_evaluator(AUDIT_ROOT / "dump", dump=True)
    if limit_videos:
        _limit(evaluator, limit_videos)
    evaluator.run()

    official = json.loads(
        (AU_ROOT / "evaluation_summary.json").read_text()
    )["metrics"]
    print("\n=== OFFICIAL-METRIC CHECK DURING DUMP PASS")
    for i, rank in enumerate(evaluator.ranks):
        for j, t in enumerate(evaluator.iou_threshs):
            key = f"Rank@{rank}_IoU@{t:.1f}"
            val = float(evaluator.counts[i][j] / evaluator.text_cnt)
            print(f"{key}: {100*val:.4f} (official {100*official[key]:.4f})")

    records = evaluator.dumped_records
    n = len(records)
    arrays = {}
    arrays["n_queries"] = np.array([n])

    def pack(key, width):
        lengths = np.array([r[key].shape[0] for r in records],
                           dtype=np.int64)
        offs = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(lengths, out=offs[1:])
        if int(offs[-1]):
            mat = np.concatenate([r[key] for r in records], axis=0)
        else:
            mat = np.zeros((0, width))
        return mat, offs

    pool_mat, pool_offs_v = pack("pool", 8)
    nms_in_mat, nms_in_offs_v = pack("nms_in_segs", 2)
    arrays["pool"] = pool_mat
    arrays["pool_offsets"] = pool_offs_v
    arrays["nms_in_segs"] = nms_in_mat
    arrays["nms_in_offsets"] = nms_in_offs_v
    arrays["nms_in_scores"] = np.concatenate(
        [r["nms_in_scores"] for r in records]
    ) if n else np.zeros(0)
    nms_score_offs = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(
        np.array([len(r["nms_in_scores"]) for r in records]), out=nms_score_offs[1:]
    )
    arrays["nms_in_scores_offsets"] = nms_score_offs
    for key in ("gt_start", "gt_end", "clip_stride", "clip_size", "fps",
                "duration"):
        arrays[key] = np.array([r[key] for r in records], dtype=np.float64)
    max_off = max(len(r["official_segments"]) for r in records)
    off_segs = np.full((n, max_off, 2), np.nan)
    off_scores = np.full((n, max_off), np.nan)
    nout_segs = np.full((n, 5, 2), np.nan)
    nout_scores = np.full((n, 5), np.nan)
    for i, r in enumerate(records):
        k = len(r["official_segments"])
        if k:
            off_segs[i, :k] = r["official_segments"]
            off_scores[i, :k] = r["official_scores"]
            nout_segs[i, :k] = r["nms_out_segs"]
            nout_scores[i, :k] = r["nms_out_scores"]
    arrays["official_segments"] = off_segs
    arrays["official_scores"] = off_scores
    arrays["nms_out_segs"] = nout_segs
    arrays["nms_out_scores"] = nout_scores
    arrays["vid_ids"] = np.array([r["vid_id"] for r in records])
    arrays["query_idx"] = np.array(
        [r["query_idx"] for r in records], dtype=np.int64
    )
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DUMP_PATH, **arrays)
    lens = np.diff(arrays["pool_offsets"])
    print(f"\nDumped {n} queries, {arrays['pool'].shape[0]} pooled candidates "
          f"-> {DUMP_PATH}")
    print(f"pool size per query: mean={lens.mean():.1f} "
          f"median={np.median(lens):.0f} "
          f"p95={np.percentile(lens, 95):.0f} max={lens.max()} "
          f"zero={(lens == 0).sum()}")


# ---------------------------------------------------------------------------
# mode: analyze (numpy only)
# ---------------------------------------------------------------------------

def bf16_quantize(x):
    """Simulate torch's float32->bfloat16 cast (np.round is half-to-even).

    decode_offsets() casts centers/offsets to the offset dtype (bf16 under
    the official autocast eval), quantizing decoded coordinates onto the
    bf16 grid (spacing 2^(e-7) with e = floor(log2 |x|)).
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    nz = np.abs(x) > 0
    ax = np.abs(x[nz])
    spacing = np.exp2(np.floor(np.log2(ax)) - 7)
    out[nz] = np.round(x[nz] / spacing) * spacing
    return out


def iou_1d(a_start, a_end, b_start, b_end):
    inter = np.maximum(
        0.0, np.minimum(a_end, b_end) - np.maximum(a_start, b_start)
    )
    union = (a_end - a_start) + (b_end - b_start) - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def soft_nms(segs, scores, iou_thresh=0.1, sigma=0.9, voting_thresh=0.95,
             max_num_segs=5, min_score=0.001):
    """Gaussian soft-NMS + segment voting mirroring
    batched_nms(mode='soft_nms') -> nms_cpu.cpp softnms_1d_cpu(method=2)."""
    x = np.asarray(segs, dtype=np.float64).copy()
    s = np.asarray(scores, dtype=np.float64).copy()
    n = len(s)
    kept_segs = np.zeros((min(max_num_segs, max(n, 1)), 2))
    kept_scores = np.zeros(min(max_num_segs, max(n, 1)))
    n_kept = 0
    alive = np.ones(n, dtype=bool)
    while n_kept < max_num_segs and alive.any():
        cand = np.where(alive)[0]
        pick = cand[np.argmax(s[cand])]
        kept_segs[n_kept] = x[pick]
        kept_scores[n_kept] = s[pick]
        n_kept += 1
        alive[pick] = False
        rest = np.where(alive)[0]
        if len(rest) == 0:
            break
        inter = np.maximum(
            0.0, np.minimum(x[rest, 1], x[pick, 1])
            - np.maximum(x[rest, 0], x[pick, 0])
        )
        union = (x[rest, 1] - x[rest, 0]) + (x[pick, 1] - x[pick, 0]) - inter
        ov = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
        s[rest] = s[rest] * np.exp(-(ov * ov) / sigma)
        alive[rest] = s[rest] >= min_score
    kept_segs = kept_segs[:n_kept]
    kept_scores = kept_scores[:n_kept]
    if voting_thresh > 0 and n_kept > 0:
        ious = iou_1d(
            kept_segs[:, 0][:, None], kept_segs[:, 1][:, None],
            x[None, :, 0], x[None, :, 1]
        )
        w = (ious >= voting_thresh) * s[None, :]
        denom = w.sum(axis=1)
        good = denom > 0
        if good.any():
            w[good] = w[good] / denom[good, None]
            kept_segs[good] = w[good] @ x
    order = np.argsort(-kept_scores, kind="stable")
    return kept_segs[order], kept_scores[order]


def rankdata_avg(a):
    """Average-rank transform (ties share the mean rank)."""
    order = np.argsort(a, kind="stable")
    ranks = np.empty(len(a), dtype=np.float64)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3:
        return float("nan")
    xc = x - x.mean()
    yc = y - y.mean()
    denom = math.sqrt(float((xc * xc).sum()) * float((yc * yc).sum()))
    if denom <= 1e-12:
        return float("nan")
    return float((xc * yc).sum() / denom)


def run_analyze():
    data = np.load(DUMP_PATH, allow_pickle=False)
    n = int(data["n_queries"][0])
    pool = data["pool"]
    pool_offs = data["pool_offsets"]
    nms_in = data["nms_in_segs"]
    nms_in_offs = data["nms_in_offsets"]
    nms_in_sc = data["nms_in_scores"]
    nms_in_sc_offs = data["nms_in_scores_offsets"]
    gt_s = data["gt_start"]
    gt_e = data["gt_end"]
    clip_stride = data["clip_stride"]
    clip_size = data["clip_size"]
    fps = data["fps"]
    duration = data["duration"]
    off_segs = data["official_segments"]
    off_scores = data["official_scores"]
    nms_out_segs = data["nms_out_segs"]
    nms_out_scores = data["nms_out_scores"]

    out = {}

    def to_seconds(i, tok_vals):
        sec = (tok_vals * clip_stride[i] + 0.5 * clip_size[i]) / fps[i]
        return np.clip(sec, 0.0, duration[i])

    # ---------------- validation anchors ----------------
    official_top1 = np.zeros(n)
    official_top5_best = np.zeros(n)
    for i in range(n):
        segs = off_segs[i]
        scores = off_scores[i]
        valid = ~np.isnan(segs).any(axis=-1) & ~np.isnan(scores)
        segs, scores = segs[valid], scores[valid]
        if len(segs):
            order = np.argsort(-scores, kind="stable")
            segs, scores = segs[order[:5]], scores[order[:5]]
            q_i = iou_1d(segs[:, 0], segs[:, 1], gt_s[i], gt_e[i])
        else:
            q_i = np.zeros(0)
        official_top1[i] = q_i[0] if len(q_i) else 0.0
        official_top5_best[i] = q_i.max() if len(q_i) else 0.0

    # anchor A: numpy soft-NMS replica on the official NMS input vs output
    seg_match = score_match = 0
    for i in range(n):
        sl_s = slice(nms_in_offs[i], nms_in_offs[i + 1])
        sl_c = slice(nms_in_sc_offs[i], nms_in_sc_offs[i + 1])
        if nms_in[sl_s].shape[0] == 0:
            continue
        k_segs, k_scores = soft_nms(nms_in[sl_s], nms_in_sc[sl_c])
        o_segs = nms_out_segs[i]
        o_scores = nms_out_scores[i]
        o_segs = o_segs[~np.isnan(o_segs).any(axis=-1)]
        o_scores = o_scores[~np.isnan(o_scores)]
        m = min(len(k_segs), len(o_segs))
        if m == 0:
            continue
        if (np.abs(np.sort(k_scores[:m]) - np.sort(o_scores[:m]))
                .max() < 1e-6):
            score_match += 1
        kk = k_segs[np.argsort(-k_scores, kind="stable")][:m]
        oo = o_segs[np.argsort(-o_scores, kind="stable")][:m]
        if np.abs(kk - oo).max() < 1e-3:
            seg_match += 1

    # anchor B: end-to-end R@K purely from the dumped pool.
    # The official decode quantizes coordinates to the bf16 grid
    # (decode_offsets casts to the bf16 offset dtype under autocast); the
    # quantized pool is the official-faithful analysis basis, the fp32
    # pool is a free-fix counterfactual.
    pool_rk = {k: 0 for k in METRIC_KEYS}
    fp32_rk = {k: 0 for k in METRIC_KEYS}
    per_q = []
    full_pool_best = np.zeros(n)          # proposal existence (no top-2000 cap)
    for i in range(n):
        sl = slice(pool_offs[i], pool_offs[i + 1])
        p_raw = pool[sl]
        if p_raw.shape[0]:
            order = np.argsort(-p_raw[:, 0], kind="stable")
            p_raw = p_raw[order]
        p_f = p_raw[:2000]                          # fp32-decode variant
        # official-faithful variant: decode in bf16 arithmetic (per-op
        # rounding exactly as decode_offsets under autocast) + length filter
        p_q = p_raw.copy()
        if p_q.shape[0]:
            qc = bf16_quantize(p_q[:, 2])
            qs = bf16_quantize(p_q[:, 3])
            p_q[:, 6] = bf16_quantize(
                qc - bf16_quantize(p_q[:, 4] * qs))
            p_q[:, 7] = bf16_quantize(
                qc + bf16_quantize(p_q[:, 5] * qs))
            keep = (p_q[:, 7] - p_q[:, 6]) > 0.1
            p_q = p_q[keep]
            full_pool_best[i] = iou_1d(
                to_seconds(i, p_q[:, 6]), to_seconds(i, p_q[:, 7]),
                gt_s[i], gt_e[i]
            ).max()
            p_q = p_q[:2000]
        p = p_q
        n_c = p.shape[0]
        rec = {"p": p, "p_f": p_f, "n": n_c, "gt_dur": gt_e[i] - gt_s[i]}
        rec["ious"] = (
            iou_1d(to_seconds(i, p[:, 6]), to_seconds(i, p[:, 7]),
                   gt_s[i], gt_e[i]) if n_c else np.zeros(0)
        )
        # end-to-end R@K from the dumped pool (quantized & fp32 variants)
        for variant, arr, sink in (("q", p, pool_rk), ("f", p_f, fp32_rk)):
            if arr.shape[0] == 0:
                continue
            k_segs, k_scores = soft_nms(
                np.stack([arr[:, 6], arr[:, 7]], axis=1), arr[:, 0]
            )
            k_sec = np.stack(
                [to_seconds(i, k_segs[:, 0]), to_seconds(i, k_segs[:, 1])],
                axis=1,
            )
            q_i = iou_1d(k_sec[:, 0], k_sec[:, 1], gt_s[i], gt_e[i])
            sink["Rank@1_IoU@0.3"] += float(q_i[:1].max() >= 0.3
                                             if len(q_i) else 0.0)
            sink["Rank@1_IoU@0.5"] += float(q_i[:1].max() >= 0.5
                                             if len(q_i) else 0.0)
            sink["Rank@5_IoU@0.3"] += float(q_i.max() >= 0.3
                                            if len(q_i) else 0.0)
            sink["Rank@5_IoU@0.5"] += float(q_i.max() >= 0.5
                                            if len(q_i) else 0.0)
        per_q.append(rec)

    top1_score_equal = []
    for i, r in enumerate(per_q):
        s_off = off_scores[i]
        s_off = s_off[~np.isnan(s_off)]
        if r["n"] > 0 and len(s_off):
            top1_score_equal.append(
                abs(r["p"][0, 0] - s_off[0]) < 1e-9
            )
    out["validation"] = {
        "n_queries": n,
        "official_R1@0.3": float((official_top1 >= 0.3).mean()),
        "official_R1@0.5": float((official_top1 >= 0.5).mean()),
        "official_R5@0.3": float((official_top5_best >= 0.3).mean()),
        "official_R5@0.5": float((official_top5_best >= 0.5).mean()),
        "pool_only_R1@0.3": pool_rk["Rank@1_IoU@0.3"] / n,
        "pool_only_R1@0.5": pool_rk["Rank@1_IoU@0.5"] / n,
        "pool_only_R5@0.3": pool_rk["Rank@5_IoU@0.3"] / n,
        "pool_only_R5@0.5": pool_rk["Rank@5_IoU@0.5"] / n,
        "fp32_decode_R1@0.3": fp32_rk["Rank@1_IoU@0.3"] / n,
        "fp32_decode_R1@0.5": fp32_rk["Rank@1_IoU@0.5"] / n,
        "fp32_decode_R5@0.3": fp32_rk["Rank@5_IoU@0.3"] / n,
        "fp32_decode_R5@0.5": fp32_rk["Rank@5_IoU@0.5"] / n,
        "full_pool_best_ge0.3": float((full_pool_best >= 0.3).mean()),
        "full_pool_best_ge0.5": float((full_pool_best >= 0.5).mean()),
        "softnms_replica_score_match": score_match / n,
        "softnms_replica_seg_match": seg_match / n,
        "top1_score_equal_rate": float(np.mean(top1_score_equal)),
        "top1_iou_diff_gt_1e-6": float((np.abs(
            np.array([r["ious"][0] if r["n"] else 0.0 for r in per_q])
            - official_top1) > 1e-6).mean()),
        "top1_iou_diff_gt_0.01": float((np.abs(
            np.array([r["ious"][0] if r["n"] else 0.0 for r in per_q])
            - official_top1) > 0.01).mean()),
    }

    # ---------------- core audit ----------------
    K_LIST = (1, 5, 10, 20, 50, 100)
    oracle = {K: {"0.3": 0, "0.5": 0, "iou": 0.0} for K in K_LIST}
    rescue = {K: {"0.3": 0, "0.5": 0} for K in (5, 10, 20, 50)}
    fail_03 = official_top1 < 0.3
    fail_05 = official_top1 < 0.5
    best_rank = []
    spear_top = {20: [], 50: [], 100: []}
    pearson_top = {20: [], 50: [], 100: []}
    pair_acc = {20: [], 50: []}
    type_a_num = type_a_den = 0
    type_b_num = type_b_den = 0
    level_stats = [
        {"n": 0, "score_sum": 0.0, "iou_sum": 0.0, "best_iou": 0.0,
         "o3": 0, "o5": 0, "best_src": 0}
        for _ in range(8)
    ]
    noNMS_r1 = {"0.3": 0, "0.5": 0}
    noNMS_r5 = {"0.3": 0, "0.5": 0}
    oracle_rescore = {k: 0 for k in METRIC_KEYS}
    pool_best_iou = np.zeros(n)
    postnms_best_iou = official_top5_best.copy()
    nms_loss_q03 = nms_loss_q05 = 0

    for i, rec in enumerate(per_q):
        p, ious, n_c = rec["p"], rec["ious"], rec["n"]
        if n_c == 0:
            best_rank.append(np.nan)
            continue
        scores = p[:, 0]
        levels = p[:, 1].astype(int)

        for K in K_LIST:
            k = min(K, n_c)
            bi = ious[:k].max()
            oracle[K]["iou"] += bi
            oracle[K]["0.3"] += bi >= 0.3
            oracle[K]["0.5"] += bi >= 0.5
        for K in (5, 10, 20, 50):
            k = min(K, n_c)
            bi = ious[:k].max()
            if fail_03[i] and bi >= 0.3:
                rescue[K]["0.3"] += 1
            if fail_05[i] and bi >= 0.5:
                rescue[K]["0.5"] += 1

        pool_best_iou[i] = ious.max()
        cand = np.where(ious >= pool_best_iou[i] - 1e-9)[0]
        best_rank.append(int(cand.min()) + 1)

        for K in (20, 50, 100):
            k = min(K, n_c)
            sc, iv = scores[:k], ious[:k]
            sp = corr(rankdata_avg(sc), rankdata_avg(iv))
            pe = corr(sc, iv)
            if not math.isnan(sp):
                spear_top[K].append(sp)
            if not math.isnan(pe):
                pearson_top[K].append(pe)

        for K in (20, 50):
            k = min(K, n_c)
            iv, sc = ious[:k], scores[:k]
            ii, jj = np.triu_indices(k, k=1)
            if len(ii) == 0:
                continue
            diff = iv[ii] - iv[jj]
            ok = np.abs(diff) > 1e-6
            if ok.sum() == 0:
                continue
            agree = np.sign(sc[ii][ok] - sc[jj][ok]) == np.sign(diff[ok])
            pair_acc[K].append(float(agree.mean()))

        n10 = max(1, int(math.ceil(0.1 * n_c)))
        type_a_num += int((ious[:n10] < 0.3).sum())
        type_a_den += n10
        io_order = np.argsort(-ious, kind="stable")
        type_b_num += int((io_order[:n10] >= n10).sum())
        type_b_den += n10

        for lv in range(8):
            m = levels == lv
            st = level_stats[lv]
            st["n"] += int(m.sum())
            if m.any():
                st["score_sum"] += float(scores[m].sum())
                st["iou_sum"] += float(ious[m].sum())
                st["best_iou"] = max(st["best_iou"], float(ious[m].max()))
                st["o3"] += int((ious[m] >= 0.3).any())
                st["o5"] += int((ious[m] >= 0.5).any())
        level_stats[int(levels[int(np.argmax(ious))])]["best_src"] += 1

        k5 = min(5, n_c)
        noNMS_r1["0.3"] += float(ious[:1].max() >= 0.3)
        noNMS_r1["0.5"] += float(ious[:1].max() >= 0.5)
        noNMS_r5["0.3"] += float(ious[:k5].max() >= 0.3)
        noNMS_r5["0.5"] += float(ious[:k5].max() >= 0.5)

        top = p[:2000]
        t_s, t_e = to_seconds(i, top[:, 6]), to_seconds(i, top[:, 7])
        t_ious = iou_1d(t_s, t_e, gt_s[i], gt_e[i])
        k_segs, k_scores = soft_nms(
            np.stack([top[:, 6], top[:, 7]], axis=1), t_ious
        )
        k_sec = np.stack(
            [to_seconds(i, k_segs[:, 0]), to_seconds(i, k_segs[:, 1])],
            axis=1,
        )
        q_i = iou_1d(k_sec[:, 0], k_sec[:, 1], gt_s[i], gt_e[i])
        oracle_rescore["Rank@1_IoU@0.3"] += float((q_i[:1] >= 0.3).any())
        oracle_rescore["Rank@1_IoU@0.5"] += float((q_i[:1] >= 0.5).any())
        oracle_rescore["Rank@5_IoU@0.3"] += float((q_i[:5] >= 0.3).any())
        oracle_rescore["Rank@5_IoU@0.5"] += float((q_i[:5] >= 0.5).any())

        if pool_best_iou[i] >= 0.3 and postnms_best_iou[i] < 0.3:
            nms_loss_q03 += 1
        if pool_best_iou[i] >= 0.5 and postnms_best_iou[i] < 0.5:
            nms_loss_q05 += 1

    best_rank = np.array(best_rank, dtype=float)
    valid_rank = ~np.isnan(best_rank)
    out["oracle_at_k"] = {
        str(K): {
            "oracle@0.3": oracle[K]["0.3"] / n,
            "oracle@0.5": oracle[K]["0.5"] / n,
            "mean_best_iou": oracle[K]["iou"] / n,
        } for K in K_LIST
    }
    out["rescue_rate"] = {
        str(K): {
            "rescue@0.3": rescue[K]["0.3"] / max(int(fail_03.sum()), 1),
            "rescue@0.5": rescue[K]["0.5"] / max(int(fail_05.sum()), 1),
            "rescued_queries@0.3": rescue[K]["0.3"],
            "rescued_queries@0.5": rescue[K]["0.5"],
        } for K in (5, 10, 20, 50)
    }
    out["fail_counts"] = {
        "top1_below_0.3": int(fail_03.sum()),
        "top1_below_0.5": int(fail_05.sum()),
    }
    out["best_candidate_rank"] = {
        "mean": float(np.nanmean(best_rank)),
        "median": float(np.nanmedian(best_rank)),
        "p25": float(np.nanpercentile(best_rank, 25)),
        "p75": float(np.nanpercentile(best_rank, 75)),
        "p90": float(np.nanpercentile(best_rank, 90)),
        "frac_le5": float((best_rank[valid_rank] <= 5).mean()),
        "frac_le10": float((best_rank[valid_rank] <= 10).mean()),
        "frac_le20": float((best_rank[valid_rank] <= 20).mean()),
        "frac_le50": float((best_rank[valid_rank] <= 50).mean()),
    }
    out["correlation"] = {
        f"top{K}": {
            "spearman_mean": float(np.mean(spear_top[K])),
            "spearman_median": float(np.median(spear_top[K])),
            "spearman_std": float(np.std(spear_top[K])),
            "pearson_mean": float(np.mean(pearson_top[K])),
            "pearson_median": float(np.median(pearson_top[K])),
            "pearson_std": float(np.std(pearson_top[K])),
            "n_valid": len(spear_top[K]),
        } for K in (20, 50, 100)
    }
    out["pairwise_iou_ranking_accuracy"] = {
        f"top{K}": {
            "mean": float(np.mean(pair_acc[K])),
            "median": float(np.median(pair_acc[K])),
            "n_valid": len(pair_acc[K]),
        } for K in (20, 50)
    }
    out["type_mismatch"] = {
        "typeA_cls_top10pct_iou_lt_0.3": type_a_num / max(type_a_den, 1),
        "typeB_iou_top10pct_cls_rank_gt10": type_b_num / max(type_b_den, 1),
    }
    tot_cand = sum(st["n"] for st in level_stats)
    tot_src = sum(st["best_src"] for st in level_stats)
    out["per_level"] = [
        {
            "level": lv,
            "candidate_share": st["n"] / max(tot_cand, 1),
            "mean_cls": st["score_sum"] / max(st["n"], 1),
            "mean_iou": st["iou_sum"] / max(st["n"], 1),
            "best_iou_seen": st["best_iou"],
            "oracle@0.3": st["o3"] / n,
            "oracle@0.5": st["o5"] / n,
            "best_iou_source_share": st["best_src"] / max(tot_src, 1),
        } for lv, st in enumerate(level_stats)
    ]
    out["nms_analysis"] = {
        "no_nms_r1@0.3": noNMS_r1["0.3"] / n,
        "no_nms_r1@0.5": noNMS_r1["0.5"] / n,
        "no_nms_r5@0.3": noNMS_r5["0.3"] / n,
        "no_nms_r5@0.5": noNMS_r5["0.5"] / n,
        "mean_pool_best_iou": float(pool_best_iou.mean()),
        "mean_postnms_best_iou": float(postnms_best_iou.mean()),
        "q_pool_best_ge0.3_but_post5_lt0.3": nms_loss_q03 / n,
        "q_pool_best_ge0.5_but_post5_lt0.5": nms_loss_q05 / n,
    }
    out["oracle_rescoring"] = {k: v / n for k, v in oracle_rescore.items()}

    # ---------------- duration buckets ----------------
    durs = np.array([rec["gt_dur"] for rec in per_q])
    t1, t2 = np.percentile(durs, [100 / 3, 200 / 3])
    out["duration_analysis"] = {"tercile_bounds": [float(t1), float(t2)]}
    for name, m in (
        ("short", durs <= t1), ("medium", (durs > t1) & (durs <= t2)),
        ("long", durs > t2),
    ):
        idxs = np.where(m)[0]
        if len(idxs) == 0:
            continue
        o10_3 = o10_5 = 0.0
        sp = []
        for i in idxs:
            rec = per_q[i]
            if rec["n"] == 0:
                continue
            bi = rec["ious"][:min(10, rec["n"])].max()
            o10_3 += bi >= 0.3
            o10_5 += bi >= 0.5
            v = corr(rankdata_avg(rec["p"][:20, 0]),
                     rankdata_avg(rec["ious"][:20]))
            if not math.isnan(v):
                sp.append(v)
        fails = idxs[official_top1[idxs] < 0.5]
        res = float(np.mean([
            per_q[i]["ious"][:min(10, per_q[i]["n"])].max() >= 0.5
            for i in fails if per_q[i]["n"]
        ])) if len(fails) else 0.0
        out["duration_analysis"][name] = {
            "n": int(len(idxs)),
            "top1_r@0.3": float((official_top1[idxs] >= 0.3).mean()),
            "top1_r@0.5": float((official_top1[idxs] >= 0.5).mean()),
            "oracle@10@0.3": float(o10_3 / len(idxs)),
            "oracle@10@0.5": float(o10_5 / len(idxs)),
            "rescue@10@0.5": res,
            "spearman_top20_mean": float(np.mean(sp)) if sp else float("nan"),
        }

    # ---------------- examples ----------------
    def fmt_example(i):
        rec = per_q[i]
        p, ious = rec["p"], rec["ious"]
        rows = []
        for r in range(min(10, rec["n"])):
            rows.append({
                "rank": r + 1,
                "cls": round(float(p[r, 0]), 4),
                "start": round(float(to_seconds(i, p[r, 6])), 2),
                "end": round(float(to_seconds(i, p[r, 7])), 2),
                "iou": round(float(ious[r]), 4),
                "level": int(p[r, 1]),
            })
        return {
            "vid_id": str(data["vid_ids"][i]),
            "query_idx": int(data["query_idx"][i]),
            "gt": [round(float(gt_s[i]), 2), round(float(gt_e[i]), 2)],
            "gt_dur": round(float(gt_e[i] - gt_s[i]), 2),
            "official_top1_iou": round(float(official_top1[i]), 4),
            "best_iou_in_top50": (
                round(float(ious[:min(50, rec["n"])].max()), 4)
                if rec["n"] else 0.0
            ),
            "top10": rows,
        }

    rescuable, proposal_fail = [], []
    for i, rec in enumerate(per_q):
        if rec["n"] == 0:
            continue
        if (official_top1[i] < 0.5
                and rec["ious"][:min(20, rec["n"])].max() >= 0.5):
            rescuable.append(i)
        if rec["ious"][:min(50, rec["n"])].max() < 0.3:
            proposal_fail.append(i)
    rng = np.random.default_rng(0)
    pick_r = (rng.choice(rescuable, size=min(5, len(rescuable)),
                         replace=False) if rescuable else [])
    pick_p = (rng.choice(proposal_fail, size=min(5, len(proposal_fail)),
                         replace=False) if proposal_fail else [])
    out["examples"] = {
        "n_rescuable_top20": len(rescuable),
        "n_proposal_fail_top50": len(proposal_fail),
        "ranking_rescuable": [fmt_example(int(i)) for i in pick_r],
        "proposal_failure": [fmt_example(int(i)) for i in pick_p],
    }

    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    ANALYSIS_PATH.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out["validation"], indent=2, sort_keys=True))
    print("Analysis saved to", ANALYSIS_PATH)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["repro", "dump", "analyze"],
                    required=True)
    ap.add_argument("--limit-videos", type=int, default=None)
    args = ap.parse_args()
    if args.mode == "repro":
        run_repro(args.limit_videos)
    elif args.mode == "dump":
        run_dump(args.limit_videos)
    else:
        run_analyze()


if __name__ == "__main__":
    main()

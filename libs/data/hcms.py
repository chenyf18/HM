"""Hard-negative Counterfactual Moment Swap (HM-HCMS-001).

Dataset-level augmentation only. Operates on the (C, T) grid-unit
feature tensor and grid-unit targets AFTER _truncate_vid_feats, so
features and targets share one coordinate frame.

Modes (hcms_mode):
  "off"        - never swap (default)
  "random"     - G1: geometric negative (same video, non-overlap,
                 duration-similar) -> location equivariance
  "hardneg"    - G2: negative chosen by offline baseline score pool
                 (high cls score, low IoU); falls back to random when
                 no pool / no hit
  "bg_control" - G3: swap two background regions (no GT overlap),
                 targets unchanged (seam artifact control)

Length handling (hcms_len_mode):
  "strict"   - round(L_neg) == round(L_gt); equal-length swap
  "relaxed"  - |L_neg - L_gt| / L_gt <= dur_tol; swap the COMMON
               length only; moved GT covers the common window;
               effective ratio recorded. No feature resize ever.
"""
from __future__ import annotations

import math
import random
from typing import Optional, Sequence

import numpy as np
import torch

DUR_TOL = 0.1
IOU_MAX = 0.1


def _iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def candidate_starts(vid_len, length):
    return range(0, max(1, int(vid_len - length) + 1))


def select_negative_random(gt, vid_len, other_targets,
                           len_mode="strict", dur_tol=DUR_TOL,
                           iou_max=IOU_MAX, rng: Optional[random.Random] = None,
                           max_trials=200):
    """Return (s_neg, e_neg, comm_len) or None. Grid units, ints."""
    rng = rng or random
    L_gt = gt[1] - gt[0]
    if L_gt < 1:
        return None
    for _ in range(max_trials):
        if len_mode == "strict":
            L_neg = int(round(L_gt))
            if L_neg < 1:
                return None
        else:
            lo = max(1, int(math.floor(L_gt * (1 - dur_tol))))
            hi = max(lo, int(math.ceil(L_gt * (1 + dur_tol))))
            L_neg = rng.randint(lo, hi)
        s = rng.randint(0, max(0, int(vid_len) - L_neg))
        neg = (s, s + L_neg)
        if _iou(neg, (gt[0], gt[1])) > iou_max:
            continue
        if any(_iou(neg, tuple(o)) > iou_max for o in other_targets):
            continue
        comm = int(min(L_gt, L_neg))
        return s, s + L_neg, comm
    return None


def select_negative_from_pool(gt, vid_len, other_targets, score_pool,
                              len_mode="strict", dur_tol=DUR_TOL,
                              iou_max=IOU_MAX, topk=20):
    """G2: prefer windows with high offline baseline cls score.

    score_pool: (vid_len,) array of per-position baseline scores (or
    None). Windows are ranked by mean score among geometric-valid
    candidates; ties broken by distance to GT. Returns like random or
    None (caller falls back to random).
    """
    if score_pool is None or len(score_pool) == 0:
        return None
    L_gt = gt[1] - gt[0]
    if L_gt < 1:
        return None
    pool = np.asarray(score_pool, dtype=np.float32)
    cands = []
    if len_mode == "strict":
        L_neg = int(round(L_gt))
        lengths = [L_neg] if L_neg >= 1 else []
    else:
        lo = max(1, int(math.floor(L_gt * (1 - dur_tol))))
        hi = max(lo, int(math.ceil(L_gt * (1 + dur_tol))))
        lengths = list(range(lo, hi + 1))
    for L_neg in lengths:
        for s in candidate_starts(int(vid_len), L_neg):
            neg = (s, s + L_neg)
            if _iou(neg, (gt[0], gt[1])) > iou_max:
                continue
            if any(_iou(neg, tuple(o)) > iou_max for o in other_targets):
                continue
            mean_sc = float(pool[s:s + L_neg].mean())
            cands.append((mean_sc, abs(s - gt[0]), s, L_neg))
    if not cands:
        return None
    cands.sort(key=lambda x: (-x[0], x[1]))
    for _, _, s, L_neg in cands[:topk]:
        comm = int(min(L_gt, L_neg))
        return s, s + L_neg, comm
    return None


def background_pair(vid_len, all_targets, gt_len,
                    iou_max=IOU_MAX, rng: Optional[random.Random] = None,
                    max_trials=200):
    """Two equal-length regions with no GT overlap. None if unlikely."""
    rng = rng or random
    for _ in range(max_trials):
        s1 = rng.randint(0, max(0, int(vid_len) - gt_len))
        s2 = rng.randint(0, max(0, int(vid_len) - gt_len))
        r1, r2 = (s1, s1 + gt_len), (s2, s2 + gt_len)
        if abs(s1 - s2) < gt_len:
            continue
        if any(_iou(r, tuple(t)) > iou_max for t in all_targets for r in (r1, r2)):
            continue
        return r1, r2
    return None


def pool_to_grid_candidates(pool_negs, gt_sec, target_q, fps,
                            clip_stride):
    """Map pool negatives (seconds) into the CURRENT grid frame.

    target = sec*fps/clip_stride - offset, so subtracting two targets
    cancels the clip offset AND any truncation window shift:
    grid_delta = (sec_neg - sec_gt) * fps / clip_stride.
    Returns [(s_grid, e_grid, neg_score)] best-effort rounded.
    """
    if not pool_negs:
        return []
    out = []
    tg0, tg1 = float(target_q[0]), float(target_q[1])
    gctr = 0.5 * (tg0 + tg1)
    gsec_ctr = 0.5 * (gt_sec[0] + gt_sec[1])
    for n in pool_negs:
        nctr = 0.5 * (n["neg_start"] + n["neg_end"])
        nl = n["neg_end"] - n["neg_start"]
        d_grid = (nctr - gsec_ctr) * fps / clip_stride
        l_grid = nl * fps / clip_stride
        s = int(round(gctr + d_grid - l_grid / 2))
        e = int(round(gctr + d_grid + l_grid / 2))
        out.append((max(0, s), max(1, e), float(n["neg_score"])))
    return out


def select_hard_from_pool(grid_cands, gt, vid_len, other_targets,
                          len_mode="strict", iou_max=IOU_MAX):
    """Pick the highest-score valid pool candidate; None if none."""
    best = None
    L_gt = gt[1] - gt[0]
    for s, e, sc in sorted(grid_cands, key=lambda x: -x[2]):
        L = e - s
        if L < 1:
            continue
        if len_mode == "strict" and L != int(round(L_gt)):
            # allow one-column discretisation slack
            if abs(L - L_gt) > 1:
                continue
        if _iou((s, e), (gt[0], gt[1])) > iou_max:
            continue
        if any(_iou((s, e), tuple(o)) > iou_max for o in other_targets):
            continue
        if s < 0 or e > int(vid_len):
            continue
        best = (s, e, min(L, int(round(L_gt))))
        break
    return best


def moment_swap(feats: torch.Tensor, targets: torch.Tensor, q_idx: int,
                neg, comm_len):
    """Swap feats[:, gt:gt+comm] <-> feats[:, neg:neg+comm]; the moved
    GT for query q_idx becomes the neg common window. Returns NEW
    tensors; inputs untouched."""
    gt = targets[q_idx]
    s_gt = int(round(float(gt[0])))
    s_neg = neg[0]
    new_feats = feats.clone()
    new_targets = targets.clone()
    if s_gt < 0:
        s_gt = 0
    if s_gt + comm_len > feats.size(1):
        comm_len = feats.size(1) - s_gt
    if s_neg + comm_len > feats.size(1):
        comm_len = feats.size(1) - s_neg
    if comm_len <= 0:
        return feats, targets, False, 0.0
    tmp = new_feats[:, s_gt:s_gt + comm_len].clone()
    new_feats[:, s_gt:s_gt + comm_len] = new_feats[:, s_neg:s_neg + comm_len]
    new_feats[:, s_neg:s_neg + comm_len] = tmp
    new_targets[q_idx, 0] = float(s_neg)
    new_targets[q_idx, 1] = float(s_neg + comm_len)
    ratio = comm_len / max(float(gt[1] - gt[0]), 1e-6)
    return new_feats, new_targets, True, ratio


def background_swap(feats: torch.Tensor, r1, r2):
    """Swap two background regions; targets unchanged."""
    L = min(r1[1] - r1[0], r2[1] - r2[0])
    new_feats = feats.clone()
    tmp = new_feats[:, r1[0]:r1[0] + L].clone()
    new_feats[:, r1[0]:r1[0] + L] = new_feats[:, r2[0]:r2[0] + L]
    new_feats[:, r2[0]:r2[0] + L] = tmp
    return new_feats


def maybe_apply(feats, targets, mode, len_mode="strict",
                prob=0.0, rng: Optional[random.Random] = None,
                score_pools: Optional[Sequence] = None,
                query_pools=None, gt_secs=None, fps=30.0,
                clip_stride=16.0):
    """Entry point used by the dataset. Returns (feats, targets, info).

    info = {"activated": bool, "swapped_ratio": float,
            "neg": (s,e) or None, "kind": mode}
    A per-sample call applies the swap to at most ONE query (random
    among valid), leaving others untouched.
    """
    info = {"activated": False, "swapped_ratio": 0.0, "neg": None,
            "kind": mode}
    if mode == "off" or prob <= 0:
        return feats, targets, info
    rng = rng or random
    if rng.random() > prob:
        return feats, targets, info
    vid_len = feats.size(1)
    n_q = targets.size(0)
    # conservative: if ANY pair of training GTs in this sample
    # overlaps at IoU >= 0.1, skip entirely (do not move any target)
    if n_q > 1:
        tl = targets.tolist()
        for a in range(n_q):
            for b in range(a + 1, n_q):
                if _iou(tl[a], tl[b]) >= 0.1:
                    info["skipped_contamination"] = True
                    return feats, targets, info
    if mode == "bg_control":
        all_t = targets.tolist()
        gt_len = max(1, int(round(float(targets[0, 1] - targets[0, 0]))))
        pair = background_pair(vid_len, all_t, gt_len, rng=rng)
        if pair is None:
            return feats, targets, info
        feats = background_swap(feats, pair[0], pair[1])
        info.update(activated=True, neg=None)
        return feats, targets, info
    # random / hardneg: choose a query to move
    order = list(range(n_q))
    rng.shuffle(order)
    pool = None
    for q in order:
        gt = targets[q].tolist()
        others = [targets[i].tolist() for i in range(n_q) if i != q]
        neg = None
        used_hard = False
        if mode == "hard" and query_pools is not None \
                and q < len(query_pools) and query_pools[q]:
            grid_c = pool_to_grid_candidates(
                query_pools[q], gt_secs[q], targets[q], fps, clip_stride)
            neg = select_hard_from_pool(grid_c, gt, vid_len, others,
                                        len_mode=len_mode)
            used_hard = neg is not None
        if mode == "hardneg":
            if score_pools is not None and q < len(score_pools) \
                    and score_pools[q] is not None:
                neg = select_negative_from_pool(
                    gt, vid_len, others, score_pools[q],
                    len_mode=len_mode)
        if neg is None and mode in ("random", "hard", "hardneg"):
            neg = select_negative_random(
                gt, vid_len, others, len_mode=len_mode, rng=rng)
        if mode == "hard":
            info["used_hard"] = used_hard
        if neg is None:
            continue
        feats, targets, ok, ratio = moment_swap(
            feats, targets, q, (neg[0], neg[1]), neg[2])
        if ok:
            info.update(activated=True, swapped_ratio=ratio,
                        neg=(neg[0], neg[1]))
        return feats, targets, info
    return feats, targets, info

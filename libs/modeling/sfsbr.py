"""Sparse Fine-Sampling Boundary Refinement (HM-SFSBR-031), minimal.

Frozen HieraMamba backbone (A-U-clean-R1, eval mode, no grads). The
refiner only adjusts start/end of the top-K (50) candidates by original
cls score, right before the unchanged Soft-NMS. No scores are modified,
no candidates are added, NMS parameters untouched.

Local window: per boundary, 16 fixed positions on a fine-feature grid
(linear offsets -3.5..+3.5 coarse tokens, step 0.5, i.e. +-7 fine rows).
Feature source:
  mode="interp" (arm B): linear interpolation over the coarse samples
      the baseline actually kept (even fine rows) - no real fine rows.
  mode="fine"   (arm C): real rows of the original feature file.
Both arms share the same architecture/init/coords/rules; only the local
feature VALUES differ. All coordinate math FP32. Identity initialisation
(zero-init last layer) keeps predictions exactly equal to R1.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# fixed window: 16 offsets (coarse-token units), shared by both arms
BOUND_OFFSETS = torch.tensor(
    [(-7. + i) * 0.5 for i in range(16)], dtype=torch.float32)  # ±3.5 tok
MAX_SHIFT = 2.0     # coarse tokens; bounded, identity-preserving clamp
MIN_LEN = 0.5       # coarse tokens


class SFSBRRefiner(nn.Module):
    def __init__(self, feat_dim=384, q_dim=384, hidden=128,
                 window_dim=256):
        super().__init__()
        # window features are RAW file rows (256-d EgoVLP); candidate
        # features are query-fused F1 (384-d). Two distinct dims.
        self.local_proj = nn.Linear(window_dim + 1, hidden)  # +coord
        self.feat_proj = nn.Linear(feat_dim, hidden)
        self.q_proj = nn.Linear(q_dim, 64)
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 3 + 64, hidden), nn.GELU(),
            nn.Linear(hidden, 2))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, cand_feat, q, start_win, end_win):
        """
        cand_feat: (N, 384) F1 feature of the candidate point
        q:         (N, 384) pooled query (value-projected)
        start_win/end_win: (N, 16, 384) local window features
        returns delta (N, 2) in coarse tokens, bounded by tanh*MAX_SHIFT
        """
        n = cand_feat.size(0)
        offs = BOUND_OFFSETS.to(cand_feat.device).view(1, 16, 1)
        s_ctx = self.local_proj(torch.cat(
            [start_win, offs.expand(n, -1, -1)], dim=-1)).mean(dim=1)
        e_ctx = self.local_proj(torch.cat(
            [end_win, offs.expand(n, -1, -1)], dim=-1)).mean(dim=1)
        h = torch.cat([self.feat_proj(cand_feat.float()),
                       self.q_proj(q.float()),
                       s_ctx, e_ctx], dim=-1)
        return torch.tanh(self.mlp(h)) * MAX_SHIFT


def gather_window(fine_pos, vid_coarse, fine_mat, offsets=None):
    """Sample 16 local features around fine_pos (FP32).

    fine_pos:   (N,) positions in FINE-row coordinates (float)
    vid_coarse: (1, C, T) baseline coarse features (even fine rows)
    fine_mat:   mmap or ndarray (T_fine, C) or None (arm B uses only
                coarse; arm C passes the real fine matrix)
    arm B: linear interpolation between neighbouring even rows;
    arm C: nearest fine row from fine_mat (clamped).
    Returns (N, 16, C).
    """
    if offsets is None:
        offsets = BOUND_OFFSETS
    pos = fine_pos.view(-1, 1) + offsets.view(1, -1).to(
        fine_pos.device).float() * 2.0          # coarse->fine rows
    def lerp_rows(mat_rows, coords):
        """mat_rows: (M, C); coords: (N,16) continuous row positions.
        Linear interpolation between neighbouring rows, clamped at the
        edges. Shared by both arms - only the row source differs."""
        m_rows = mat_rows.shape[0]
        c = coords.clamp(0.0, float(m_rows - 1))
        lo = c.floor().long().clamp(0, m_rows - 2)
        alpha = (c - lo.float()).unsqueeze(-1)      # (N,16,1)
        N = coords.size(0)
        m = torch.as_tensor(np.asarray(mat_rows), dtype=torch.float32,
                            device=coords.device)
        a = m[lo.reshape(-1)].view(N, 16, -1)
        b = m[(lo + 1).reshape(-1)].view(N, 16, -1)
        return (1.0 - alpha) * a + alpha * b

    if fine_mat is None:
        # arm B: source rows are ONLY the coarse samples the baseline
        # kept (even fine rows); map fine coord -> coarse coord / 2
        v = vid_coarse[0].transpose(0, 1).contiguous()   # (T, C)
        return lerp_rows(v.cpu().numpy(), (pos / 2.0))
    # arm C: source rows are ALL real fine rows of the original file
    return lerp_rows(fine_mat, pos)


import numpy as np  # noqa: E402  (used by gather_window arm C)


def apply_delta(start_tok, end_tok, delta, t_valid):
    """Bounded, legal boundary update. Identity when delta == 0.

    start_tok/end_tok: (N,) FP32 coarse tokens (predicted)
    delta: (N, 2) from the refiner
    t_valid: scalar valid length (coarse tokens)
    """
    s = start_tok + delta[:, 0]
    e = end_tok + delta[:, 1]
    s = s.clamp(0.0, t_valid - MIN_LEN)
    e = e.clamp(MIN_LEN, float(t_valid))
    # strict identity: rows with zero delta return the original values
    # even if the original decoded coordinates lie outside the clamps
    zero = (delta == 0).all(dim=-1)
    s = torch.where(zero, start_tok, s)
    e = torch.where(zero, end_tok, e)
    # keep start < end - MIN_LEN without moving an identity pair
    bad = e - s < MIN_LEN
    if bool(bad.any()):
        mid = (start_tok[bad] + end_tok[bad]) / 2.0
        s = torch.where(bad, (mid - MIN_LEN / 2).clamp(0, None), s)
        e = torch.where(bad, (mid + MIN_LEN / 2).clamp(None, t_valid), e)
    return s, e


def supervision_targets(start_tok, end_tok, gt_start_tok, gt_end_tok):
    """Per-endpoint reachability (HM-SFSBR-031 rev1).

    An endpoint is supervised iff the candidate span overlaps GT AND
    that endpoint's error is within MAX_SHIFT. The two endpoints are
    independent: an unreachable far endpoint never discards the valid
    supervision of the near one.

    Returns mask_s (N,), mask_e (N,), ds (N,), de (N,) clamped targets
    (targets are zero where the respective mask is zero).
    """
    inter = (torch.minimum(end_tok, gt_end_tok)
             - torch.maximum(start_tok, gt_start_tok)).clamp_min(0)
    overlap = inter > 0
    ds_raw = (gt_start_tok - start_tok).clamp(-MAX_SHIFT, MAX_SHIFT)
    de_raw = (gt_end_tok - end_tok).clamp(-MAX_SHIFT, MAX_SHIFT)
    m_s = overlap & ((gt_start_tok - start_tok).abs()
                     <= MAX_SHIFT + 1e-6)
    m_e = overlap & ((gt_end_tok - end_tok).abs() <= MAX_SHIFT + 1e-6)
    return (m_s.float(), m_e.float(),
            ds_raw * m_s.float(), de_raw * m_e.float())

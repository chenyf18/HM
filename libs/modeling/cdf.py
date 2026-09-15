"""Conservative Detail Feedback neck (HM-CDF-032), first version.

Inserted after late fusion, before the original cls/reg heads. For
child levels L0/L1/L2 the ORIGINAL fused output of parent level l+1
feeds back in parallel (no recursion, no iteration):

    u_j   = W_out GELU( W_f LN(F_child_j) + W_p LN(F_parent_p(j)) )
    unc:  F'_j = F_j + u_j
    con:  F'_j = F_j + u_j - group_weighted_mean(u)_g(j)

Parent position p(j) is derived from the adaptive temporal metadata
(child centre falls inside the parent support span; nearest parent
centre as fallback), never from lengths. Group means use actual
masks: padded child positions never contribute; a group with a single
valid child gets exactly zero conservative update.

W_out is zero-initialised (no bias) so enabling the neck starts
exactly at the baseline; no extra zero-gate is multiplied on top, so
the gradient path is alive at init. The GELU nonlinearity sits between
the parent term and the group-mean subtraction, so the conservative
projection cannot cancel the parent contribution.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import LayerNorm

BOTTLENECK = 64


class CDFFeedbackUnit(nn.Module):
    def __init__(self, dim, bottleneck=BOTTLENECK):
        super().__init__()
        self.ln_c = LayerNorm(dim)
        self.ln_p = LayerNorm(dim)
        self.w_f = nn.Linear(dim, bottleneck, bias=False)
        self.w_p = nn.Linear(dim, bottleneck, bias=False)
        self.w_out = nn.Linear(bottleneck, dim, bias=False)
        nn.init.zeros_(self.w_out.weight)

    def forward(self, f_child, f_parent_pj):
        """Inputs and output are (B, C, T); LayerNorm runs on the
        channel dim via transpose (project convention)."""
        hc = self.ln_c(f_child).transpose(1, 2)     # (B, T, C)
        hp = self.ln_p(f_parent_pj).transpose(1, 2)
        h = self.w_f(hc) + self.w_p(hp)
        return self.w_out(torch.nn.functional.gelu(h)).transpose(1, 2)


def parent_index(child_center, parent_start, parent_end, parent_center):
    """(B, T_c) long: parent row whose [start,end] contains the child
    centre; nearest parent centre as fallback. All in input-token FP32.
    """
    B, Tc = child_center.shape
    Tp = parent_center.shape[1]
    idx = torch.searchsorted(
        parent_end.contiguous(), child_center.contiguous(), right=False)
    # searchsorted on end: first k with end_k > center -> candidate k;
    # verify start_k <= center, else nearest centre
    idx = idx.clamp(0, Tp - 1)
    inside = (parent_end.gather(1, idx) > child_center) & \
             (parent_start.gather(1, idx) <= child_center)
    dist = (parent_center.unsqueeze(1)
            - child_center.unsqueeze(2)).abs().argmin(dim=-1)
    return torch.where(inside, idx, dist)


class CDFNeck(nn.Module):
    """mode: "unconstrained" or "conservative". None/disabled models
    never construct this module (config gate)."""

    def __init__(self, dim=384, feedback_children=(0, 1, 2),
                 mode="conservative"):
        super().__init__()
        assert mode in ("unconstrained", "conservative")
        self.mode = mode
        self.children_levels = tuple(feedback_children)
        self.units = nn.ModuleDict({
            str(l): CDFFeedbackUnit(dim) for l in self.children_levels})
        self.last_stats = None

    def forward(self, fpn_fused, temporal_metadata, fpn_masks):
        """fpn_fused: tuple (B, C, T_l) query-fused level features.
        temporal_metadata: tuple (B, T_l, 5) adaptive geometry.
        fpn_masks: tuple (B, T_l) bool (fusion masks, query-expanded).
        Returns a NEW tuple; original tensors are never modified
        in-place; parents are the original fused outputs (no recursion).
        """
        out = list(fpn_fused)
        stats = {}
        for l in self.children_levels:
            p = l + 1
            f_c = fpn_fused[l]                       # (B, C, Tc)
            f_p = fpn_fused[p]                       # (B, C, Tp)
            meta_c = temporal_metadata[l].float()
            meta_p = temporal_metadata[p].float()
            mask_c = fpn_masks[l].bool() if fpn_masks[l].ndim == 2 \
                else fpn_masks[l][:, 0].bool()       # (B, Tc)
            mask_p = fpn_masks[p].bool() if fpn_masks[p].ndim == 2 \
                else fpn_masks[p][:, 0].bool()
            pj = parent_index(meta_c[..., 2], meta_p[..., 0],
                              meta_p[..., 1], meta_p[..., 2])   # (B,Tc)
            # gather parent features to child positions
            fpj = f_p.gather(
                2, pj.unsqueeze(1).expand(-1, f_p.size(1), -1).clamp(
                    max=f_p.size(2) - 1))            # (B, C, Tc)
            u = self.units[str(l)](f_c, fpj)         # (B, C, Tc)
            if self.mode == "conservative":
                B, C, Tc = u.shape
                valid = mask_c & mask_p.gather(1, pj)
                # group mean over valid children of each parent row
                counts = torch.zeros_like(u[:, 0, :])             # (B,Tc)
                sums = torch.zeros_like(u)
                idx = pj.clamp(max=f_p.size(2) - 1)
                # accumulate per-parent sums via index_add on dim -1
                flat_p = (torch.arange(B, device=u.device)
                          .unsqueeze(1) * f_p.size(2) + idx)      # (B,Tc)
                vp = valid.reshape(-1)
                flat_u = u.permute(0, 2, 1).reshape(-1, C)
                flat_g = flat_p.reshape(-1)
                m = vp
                gsum = torch.zeros((B * f_p.size(2), C),
                                   dtype=u.dtype, device=u.device)
                gcnt = torch.zeros((B * f_p.size(2), 1),
                                   dtype=u.dtype, device=u.device)
                gsum.index_add_(0, flat_g[m], flat_u[m])
                gcnt.index_add_(0, flat_g[m],
                                torch.ones((int(m.sum()), 1),
                                           dtype=u.dtype,
                                           device=u.device))
                mean_g = (gsum / gcnt.clamp_min(1.0)).reshape(
                    B, f_p.size(2), C).permute(0, 2, 1)           # (B,C,Tp)
                mean_child = mean_g.gather(
                    2, idx.unsqueeze(1).expand(-1, C, -1))        # (B,C,Tc)
                delta = u - mean_child * valid.unsqueeze(1).to(u.dtype)
                stats[f"L{l}"] = {
                    "n_valid_children": int(valid.sum()),
                    "single_child_groups_zero": True,
                }
            else:
                delta = u
            out[l] = fpn_fused[l] + delta
        self.last_stats = stats
        return tuple(out)

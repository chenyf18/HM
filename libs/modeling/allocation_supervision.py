"""Train-only GT supervision for the allocator importance predictor (A-S-clean).

The GT segment is used exclusively to build per-token allocation targets for
the auxiliary loss. Cuts are always produced by the predicted importance
scores; evaluation never receives targets (enforced by an allocator-side
fail-fast guard).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AllocationSupervisionLoss(nn.Module):
    """y_t = clamp(alpha * B_t + beta * F_t, 0, 1) over real coordinates."""

    def __init__(self, sigma: float = 2.0, alpha: float = 1.0,
                 beta: float = 0.5, eps: float = 1e-6):
        super().__init__()
        for name, value in (("sigma", sigma), ("alpha", alpha), ("beta", beta)):
            if not torch.isfinite(torch.tensor(float(value))) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if sigma == 0:
            raise ValueError("sigma must be positive")
        self.sigma = float(sigma)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)

    @torch.no_grad()
    def build_targets(self, temporal_metadata, targets):
        """Per-level allocation targets aligned to real token coordinates.

        temporal_metadata: tuple of (B, T, >=5) [start, end, center, span, scale]
        targets: (B, 2) GT segments in input-token coordinates.
        """
        gs = targets[:, 0].float()
        ge = targets[:, 1].float()
        level_targets = []
        for metadata in temporal_metadata:
            md = metadata.float()
            start, end = md[..., 0], md[..., 1]
            center, span = md[..., 2], md[..., 3].abs()
            valid = span > 0
            foreground = ((end >= gs[:, None]) &
                          (start <= ge[:, None])).float()
            sigma_t = (self.sigma * span).clamp_min(self.eps)
            b_start = torch.exp(
                -0.5 * ((center - gs[:, None]) / sigma_t) ** 2)
            b_end = torch.exp(
                -0.5 * ((center - ge[:, None]) / sigma_t) ** 2)
            boundary = b_start + b_end
            denom = boundary.amax(dim=1, keepdim=True).clamp_min(self.eps)
            boundary = boundary / denom
            y = (self.alpha * boundary + self.beta * foreground).clamp(0.0, 1.0)
            y = y * valid.float()
            level_targets.append(y)
        return level_targets

    def forward(self, importance_debug, temporal_metadata,
                sequence_masks, targets):
        """BCE-with-logits between predicted importance and allocation target.

        importance_debug: tuple of per-level dicts containing 'importance'
        probability maps (B, T).
        """
        level_targets = self.build_targets(temporal_metadata, targets)
        total = importance_debug[0]["importance"].float().sum() * 0.0
        terms = 0
        for debug, y, mask in zip(importance_debug, level_targets,
                                  sequence_masks):
            prob = debug["importance"].float()
            valid = mask.squeeze(1).bool() & (y > 0)
            if not bool(valid.any()):
                continue
            p = prob[valid].clamp(self.eps, 1.0 - self.eps)
            logit = torch.log(p / (1.0 - p))
            y_valid = y[valid]
            total = total + F.binary_cross_entropy_with_logits(
                logit, y_valid, reduction="mean")
            terms += 1
        if terms == 0:
            return total
        return total / terms


class RankingAllocationLoss(nn.Module):
    """Margin ranking: boundary > foreground > background (relative only)."""

    def __init__(self, boundary_radius: float = 1.5,
                 margin_bf: float = 0.1, margin_fb: float = 0.1,
                 eps: float = 1e-6):
        super().__init__()
        self.boundary_radius = float(boundary_radius)
        self.margin_bf = float(margin_bf)
        self.margin_fb = float(margin_fb)
        self.eps = float(eps)

    def _regions(self, metadata, targets):
        md = metadata.float()
        start, end = md[..., 0], md[..., 1]
        center, span = md[..., 2], md[..., 3].abs()
        valid = span > 0
        gs = targets[:, 0].float()[:, None]
        ge = targets[:, 1].float()[:, None]
        fg = (end >= gs) & (start <= ge) & valid
        radius = (span * self.boundary_radius).clamp_min(self.eps)
        bnd = fg & (((center - gs).abs() <= radius)
                    | ((center - ge).abs() <= radius))
        interior = fg & ~bnd
        background = valid & ~fg
        return bnd, interior, background

    def forward(self, importance_debug, temporal_metadata,
                sequence_masks, targets):
        total = importance_debug[0]["importance"].float().sum() * 0.0
        terms = 0
        for debug, metadata, mask in zip(importance_debug, temporal_metadata,
                                         sequence_masks):
            prob = debug["importance"].float()
            valid = mask.squeeze(1).bool()
            if not bool(valid.any()):
                continue
            bnd, interior, background = self._regions(metadata, targets)

            def masked_mean(region):
                region = region & valid
                if not bool(region.any()):
                    return None
                return prob[region].mean()

            sb = masked_mean(bnd)
            sf = masked_mean(interior)
            sbg = masked_mean(background)
            level_terms = []
            if sb is not None and sf is not None:
                level_terms.append(torch.relu(
                    self.margin_bf - sb + sf))
            if sf is not None and sbg is not None:
                level_terms.append(torch.relu(
                    self.margin_fb - sf + sbg))
            if level_terms:
                total = total + torch.stack(level_terms).mean()
                terms += 1
        if terms == 0:
            return total
        return total / terms

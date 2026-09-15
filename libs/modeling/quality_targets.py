"""Quality-aware classification targets (HM-QACT-033, phase 1).

Replaces the binary point-in-GT classification label with a continuous
target derived from the FP32-decoded proposal IoU. The decoded segments
are fully detached: the classification target never back-propagates into
the regression branch.

Modes:
  binary         - original behaviour (labels from _annotate_points)
  decoded_iou    - y_i = temporal_IoU(decode(points_i, offsets_i), GT)
  metric_utility - y_i = .5*sigmoid((IoU-.3)/tau) + .5*sigmoid((IoU-.5)/tau)

The regression assignment (positive masks) is NOT touched: callers keep
using the original binary labels for the reg loss.
"""
from __future__ import annotations

import torch

from .temporal_coordinates import decode_offsets


def temporal_iou_1d(segs, targets):
    """segs (B, P, 2), targets (B, 2) -> IoU (B, P), all FP32."""
    s1, e1 = segs[..., 0], segs[..., 1]
    s2 = targets[:, 0:1]
    e2 = targets[:, 1:2]
    inter = (torch.minimum(e1, e2) - torch.maximum(s1, s2)).clamp_min(0)
    union = (e1 - s1) + (e2 - s2) - inter
    return torch.where(union > 0, inter / union.clamp_min(1e-12),
                       torch.zeros_like(union))


def compute_quality_cls_targets(points, offsets, targets, mode, tau=0.05):
    """Return continuous cls targets (B, P) float32, fully detached.

    points  (B, P, >=4): adaptive point coords (center at col 0, scale
                         at col 3) - FP32
    offsets (B, P, 2):   predicted left/right offsets (any dtype; cast
                         to FP32, detached before decode)
    targets (B, 2):      GT segments in the same token coordinate frame
    """
    if mode == "binary":
        raise ValueError("binary mode uses the original label path")
    with torch.no_grad():
        pts = points.detach().float()
        off = offsets.detach().float()
        segs = decode_offsets(pts, off)                    # FP32 decode
        iou = temporal_iou_1d(segs, targets.detach().float())
        if mode == "decoded_iou":
            y = iou
        elif mode == "metric_utility":
            y = 0.5 * torch.sigmoid((iou - 0.3) / tau) \
                + 0.5 * torch.sigmoid((iou - 0.5) / tau)
        else:
            raise ValueError(f"unknown cls_target_type: {mode}")
        return y.clamp(0.0, 1.0)

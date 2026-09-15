"""Temporal Evidence Formation Module (HM-TEFM-038).

A per-level 1D temporal conv stack (Conv-GELU-Conv, zero-init last
layer => residual identity at start) inserted between query fusion and
the cls/reg heads. An auxiliary EvidenceHead produces per-position
temporal relevance logits supervised by the GT moment mask, teaching
the evidence features to carry temporal-boundary information.

No attention, no Transformer, no FiLM, no query modulation. Only local
temporal convolution (kernel=3) + direct temporal supervision.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import MaskedConv1D, LayerNorm


class TEFM(nn.Module):
    """E = F + Conv2(GELU(Conv1(F))), zero-init Conv2 => E=F at start."""

    def __init__(self, dim=384):
        super().__init__()
        self.conv1 = MaskedConv1D(dim, dim, kernel_size=3, padding=1,
                                  bias=False)
        self.norm1 = LayerNorm(dim)
        self.conv2 = MaskedConv1D(dim, dim, kernel_size=3, padding=1,
                                  bias=False)
        self.norm2 = LayerNorm(dim)
        nn.init.zeros_(self.conv2.conv.weight)

    def forward(self, fpn, fpn_masks):
        """fpn: tuple of (B,C,T_l); masks: tuple of (B,1,T_l) or (B,T_l).
        Returns new tuple; inputs not modified."""
        out = []
        for x, m in zip(fpn, fpn_masks):
            h, _ = self.conv1(x, m)
            h = F.gelu(self.norm1(h))
            h, _ = self.conv2(h, m)
            h = self.norm2(h)
            out.append(x + h)
        return tuple(out)


class EvidenceHead(nn.Module):
    """Per-position temporal relevance logit (1x1 conv)."""

    def __init__(self, dim=384):
        super().__init__()
        self.head = MaskedConv1D(dim, 1, kernel_size=1)

    def forward(self, fpn, fpn_masks):
        """Returns tuple of (B, T_l) relevance logits per level."""
        outs = []
        for x, m in zip(fpn, fpn_masks):
            logit, _ = self.head(x, m)
            outs.append(logit.squeeze(1))
        return tuple(outs)

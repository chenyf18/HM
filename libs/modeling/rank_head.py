"""Lightweight moment-level listwise ranking head (CLC-LMR, plan A)."""

import torch
import torch.nn as nn


class MomentRankHead(nn.Module):
    """Score a candidate moment from pooled fused features + geometry + level.

    Attached to the model (not the trainer) so optimizer / EMA / checkpoint
    handling is inherited automatically.
    """

    def __init__(self, feat_dim, num_levels=8, hidden_dim=256,
                 geo_dim=4, level_emb=8):
        super().__init__()
        self.level_embed = nn.Embedding(num_levels, level_emb)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim + geo_dim + level_emb, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.normal_(self.mlp[-1].weight, std=1e-3)

    def forward(self, pooled, geo, level_ids):
        """pooled (N, D), geo (N, 4), level_ids (N,) long -> scores (N,)."""
        emb = self.level_embed(level_ids)
        return self.mlp(torch.cat((pooled, geo, emb), dim=-1)).squeeze(-1)

"""DN-DETR-style span denoising for temporal grounding (plan C).

Training: reconstruct the true GT span from a noisy span's pooled features +
query embedding (multiple noise groups: center shifts and scale perturbs).
Inference: iterative refinement of candidate spans (same pathway).
"""

import torch
import torch.nn as nn


class MomentDenoiser(nn.Module):
    def __init__(self, feat_dim, query_dim, hidden_dim=256, geo_dim=4,
                 max_delta=2.0):
        super().__init__()
        self.max_delta = float(max_delta)
        self.net = nn.Sequential(
            nn.Linear(feat_dim + query_dim + geo_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        nn.init.zeros_(self.net[-1].bias)
        nn.init.normal_(self.net[-1].weight, std=1e-3)

    def forward(self, pooled, query_repr, geo):
        """pooled (N,D), query_repr (N,Q) broadcastable, geo (N,4)
        -> normalized (delta_start, delta_end) in [-max_delta, max_delta]."""
        if query_repr.size(0) != pooled.size(0):
            query_repr = query_repr.expand(pooled.size(0), -1)
        out = self.net(torch.cat((pooled, query_repr, geo), dim=-1))
        return self.max_delta * torch.tanh(out)

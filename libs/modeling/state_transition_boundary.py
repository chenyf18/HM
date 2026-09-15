"""QSTB (T1): query-conditioned state-transition boundary branch.

One implementation covers the §17 ablation via config:
  source: hydra (T1 / T1-state) | feature (Control-F)
  use_delta: true (T1 / feature-diff) | false (state-only)
Boundary predictor consumes gated transitions + current state; a zero-init
feature fusion injects the boundary embedding into the chosen FPN level.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StateTransitionBoundaryBranch(nn.Module):
    def __init__(self, d_model, query_dim, hidden_dim=256, levels=(1,),
                 source="hydra", use_delta=True, fusion_gamma=1.0):
        super().__init__()
        self.levels = tuple(levels)
        if source not in ("hydra", "feature"):
            raise ValueError("source must be hydra or feature")
        self.source = source
        self.use_delta = bool(use_delta)
        self.fusion_gamma = float(fusion_gamma)
        self.q_proj = nn.Linear(query_dim, d_model)
        self.gate_f = nn.Linear(d_model, d_model)
        self.gate_b = nn.Linear(d_model, d_model)
        in_dim = 3 * d_model if use_delta else d_model
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.start_head = nn.Linear(hidden_dim, 1)
        self.end_head = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.start_head.weight, std=1e-3)
        nn.init.zeros_(self.start_head.bias)
        nn.init.normal_(self.end_head.weight, std=1e-3)
        nn.init.zeros_(self.end_head.bias)
        self.fuse = nn.Linear(2, d_model)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)

    def forward(self, h, query_repr, seq_mask):
        """h (B,C,T) state/feature rows on the token grid;
        query_repr (B,Cq); seq_mask (B,T) bool. Returns logits (B,T,2),
        boundary embedding (B,C,T)."""
        B, C, T = h.shape
        x = h.transpose(1, 2)                       # (B,T,C)
        if self.use_delta:
            df = x - F.pad(x, (0, 0, 1, 0))[:, :-1]   # zero-pad left edge
            db = x - F.pad(x, (0, 0, 0, 1))[:, 1:]    # zero-pad right edge
            q = self.q_proj(query_repr).unsqueeze(1)
            zf = F.layer_norm(df, (C,)) * torch.sigmoid(self.gate_f(q))
            zb = F.layer_norm(db, (C,)) * torch.sigmoid(self.gate_b(q))
            z = torch.cat((zf, zb, x), dim=-1)
        else:
            q = self.q_proj(query_repr).unsqueeze(1)
            z = x * torch.sigmoid(self.gate_f(q))
        t = self.trunk(z)
        logits = torch.cat(
            (self.start_head(t), self.end_head(t)), dim=-1)   # (B,T,2)
        b_emb = self.fuse(
            torch.sigmoid(logits))                            # (B,T,C)
        out = b_emb.transpose(1, 2) * seq_mask.unsqueeze(1).to(
            b_emb.dtype)
        return logits, out

    def fused_fpn(self, fpn_level, b_emb):
        """Minimal-invasion fusion; zero-init fuse => baseline-equal start."""
        return fpn_level + self.fusion_gamma * b_emb

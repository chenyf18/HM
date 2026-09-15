"""Candidate Set Ranking Module (HM-CSRM-034).

Takes a query token and K candidate tokens (fused feature +
positional encoding + original logit + level), runs a small
Transformer encoder over the set, and outputs a ranking score per
candidate. Zero-initialised score head => enabling CSRM starts at
the baseline scorer's exact ranking (score=0 for all, stable under
any monotone sorting).

The module does NOT touch the video backbone, proposal heads, decode,
or NMS; it only re-scores the top-K candidates at inference.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def sincos_encoding(values, dim):
    """values: (N,) float; returns (N, dim)."""
    half = dim // 2
    freq = torch.exp(-math.log(10000.0) * torch.arange(
        half, dtype=torch.float32, device=values.device) / half)
    ang = values.unsqueeze(-1) * freq.unsqueeze(0)
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


class CandidateSetRankingModule(nn.Module):
    def __init__(self, feat_dim=384, q_dim=384, d_model=256, n_heads=4,
                 n_layers=2, topk=50, max_seq_len=2304, ff_mult=2):
        super().__init__()
        self.topk = topk
        self.d_model = d_model
        self.feat_proj = nn.Linear(feat_dim, d_model)
        self.q_proj = nn.Linear(q_dim, d_model)
        self.pos_dim = 64
        self.start_emb = nn.Linear(self.pos_dim * 2, d_model)
        self.logit_proj = nn.Linear(1, d_model)
        self.level_emb = nn.Embedding(8, d_model)
        self.query_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.query_token, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=0.1, activation='gelu',
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.score_head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.score_head.weight)
        nn.init.zeros_(self.score_head.bias)

    def forward(self, query, cand_feats, starts, ends, logits, levels):
        """
        query:      (B, 384) pooled query embedding (value-projected)
        cand_feats: (B, K, 384) F1 features at candidate positions
        starts:     (B, K) predicted start (grid tokens)
        ends:       (B, K) predicted end
        logits:     (B, K) original cls logits
        levels:     (B, K) pyramid level indices
        returns:    (B, K) CSRM ranking scores
        """
        B, K, _ = cand_feats.shape
        q = self.q_proj(query.float()).unsqueeze(1)           # (B,1,D)
        pe_s = sincos_encoding(starts.float().reshape(-1),
                               self.pos_dim).reshape(B, K, -1)
        pe_e = sincos_encoding(ends.float().reshape(-1),
                               self.pos_dim).reshape(B, K, -1)
        pos = torch.cat([pe_s, pe_e], dim=-1)                 # (B,K,2P)
        z = (self.feat_proj(cand_feats.float())
             + self.start_emb(pos)
             + self.logit_proj(logits.float().unsqueeze(-1))
             + self.level_emb(levels.long()))
        tokens = torch.cat([q, z], dim=1)                     # (B,K+1,D)
        tokens = tokens + self.query_token.unsqueeze(0).unsqueeze(0)
        out = self.encoder(tokens)                            # (B,K+1,D)
        cand_out = self.norm(out[:, 1:, :])                   # (B,K,D)
        return self.score_head(cand_out).squeeze(-1)          # (B,K)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

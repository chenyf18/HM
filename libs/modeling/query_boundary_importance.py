from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryBoundaryImportancePredictor(nn.Module):
    """Predict query relevance, temporal boundaries, and token change."""

    def __init__(
        self,
        video_dim: int,
        query_dim: int,
        hidden_dim: int = 0,
        alpha: float = 0.4,
        beta: float = 0.4,
        gamma: float = 0.2,
        eps: float = 1e-6,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim) if hidden_dim else int(video_dim)
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if min(alpha, beta, gamma) < 0:
            raise ValueError("importance weights must be non-negative")
        if alpha + beta + gamma <= 0:
            raise ValueError("at least one importance weight must be positive")

        self.video_dim = int(video_dim)
        self.query_dim = int(query_dim)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.eps = float(eps)

        self.relevance_video_proj = nn.Linear(video_dim, hidden_dim, bias=False)
        self.relevance_query_proj = nn.Linear(query_dim, hidden_dim, bias=False)

        self.boundary_video_proj = nn.Linear(video_dim, hidden_dim)
        self.boundary_query_proj = nn.Linear(query_dim, hidden_dim)
        self.boundary_predictor = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

        for module in self.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def _normalize_mask(mask: torch.Tensor, batch_size: int, seq_len: int):
        if mask.ndim == 3:
            if mask.size(1) != 1:
                raise ValueError("mask must have shape (B, 1, T) or (B, T)")
            mask = mask[:, 0]
        elif mask.ndim != 2:
            raise ValueError("mask must have shape (B, 1, T) or (B, T)")
        if mask.shape != (batch_size, seq_len):
            raise ValueError(
                "mask shape {} does not match video shape ({}, {})".format(
                    tuple(mask.shape), batch_size, seq_len
                )
            )
        return mask.to(dtype=torch.bool)

    @staticmethod
    def _neighbors(sequence: torch.Tensor, valid: torch.Tensor):
        previous = torch.cat((sequence[:, :1], sequence[:, :-1]), dim=1)
        following = torch.cat((sequence[:, 1:], sequence[:, -1:]), dim=1)

        previous_valid = torch.cat((valid[:, :1], valid[:, :-1]), dim=1)
        following_valid = torch.cat((valid[:, 1:], valid[:, -1:]), dim=1)
        previous = torch.where(previous_valid.unsqueeze(-1), previous, sequence)
        following = torch.where(following_valid.unsqueeze(-1), following, sequence)
        return previous, following, previous_valid

    def forward(
        self,
        video_feat: torch.Tensor,
        query_repr: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Return scores with shape (B, T) for video input (B, D, T)."""
        if video_feat.ndim != 3:
            raise ValueError("video_feat must have shape (B, D, T)")
        batch_size, channels, seq_len = video_feat.shape
        if seq_len == 0:
            raise ValueError("video sequence length must be positive")
        if channels != self.video_dim:
            raise ValueError(
                "expected video dimension {}, got {}".format(
                    self.video_dim, channels
                )
            )
        if query_repr.ndim != 2 or query_repr.shape != (batch_size, self.query_dim):
            raise ValueError(
                "query_repr must have shape ({}, {})".format(
                    batch_size, self.query_dim
                )
            )

        valid = self._normalize_mask(mask, batch_size, seq_len)
        valid = valid.to(device=video_feat.device)
        valid_float = valid.to(dtype=video_feat.dtype)
        sequence = video_feat.transpose(1, 2) * valid_float.unsqueeze(-1)
        query_repr = query_repr.to(
            device=video_feat.device, dtype=video_feat.dtype
        )

        video_rel = F.normalize(
            self.relevance_video_proj(sequence), p=2, dim=-1, eps=self.eps
        )
        query_rel = F.normalize(
            self.relevance_query_proj(query_repr), p=2, dim=-1, eps=self.eps
        )
        relevance = 0.5 * (
            (video_rel * query_rel.unsqueeze(1)).sum(dim=-1) + 1.0
        )

        previous, following, previous_valid = self._neighbors(sequence, valid)
        query_boundary = self.boundary_query_proj(query_repr).unsqueeze(1)
        query_boundary = query_boundary.expand(-1, seq_len, -1)
        boundary_input = torch.cat(
            (
                self.boundary_video_proj(previous),
                self.boundary_video_proj(sequence),
                self.boundary_video_proj(following),
                query_boundary,
            ),
            dim=-1,
        )
        boundary_logits = self.boundary_predictor(boundary_input)
        boundary_probs = torch.sigmoid(boundary_logits)
        start_score = boundary_probs[..., 0]
        end_score = boundary_probs[..., 1]
        boundary = torch.maximum(start_score, end_score)

        change_valid = valid & previous_valid
        change_valid[:, 0] = False
        temporal_change = torch.linalg.vector_norm(
            sequence - previous, ord=2, dim=-1
        )
        temporal_change = temporal_change * change_valid.to(temporal_change.dtype)
        max_change = temporal_change.amax(dim=1, keepdim=True).clamp(min=self.eps)
        temporal_change = temporal_change / max_change

        relevance = relevance * valid_float
        start_score = start_score * valid_float
        end_score = end_score * valid_float
        boundary = boundary * valid_float
        temporal_change = temporal_change * valid_float
        importance = (
            self.alpha * relevance
            + self.beta * boundary
            + self.gamma * temporal_change
        ) * valid_float

        return {
            "relevance": relevance,
            "start": start_score,
            "end": end_score,
            "start_prob": start_score,
            "end_prob": end_score,
            "boundary": boundary,
            "temporal_change": temporal_change,
            "importance": importance,
        }

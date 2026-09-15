import torch
import torch.nn as nn
import torch.nn.functional as F


class CoarseToFineTemporalRefiner(nn.Module):
    """Lightweight query-gated top-down refinement for temporal FPNs."""

    def __init__(
        self,
        d_model,
        query_dim,
        num_levels,
        refine_levels=3,
        initial_gate_bias=-4.0,
        eps=1e-6,
    ):
        super().__init__()

        self.d_model = int(d_model)
        self.query_dim = int(query_dim)
        self.num_levels = int(num_levels)
        try:
            parsed_refine_levels = int(refine_levels)
        except (TypeError, ValueError):
            raise ValueError("refine_levels must be an integer")
        if parsed_refine_levels != refine_levels:
            raise ValueError("refine_levels must be an integer")
        if self.d_model <= 0 or self.query_dim <= 0:
            raise ValueError("d_model and query_dim must be positive")
        if self.num_levels < 2:
            raise ValueError(
                "coarse-to-fine refinement requires at least two FPN levels"
            )
        if not 1 <= parsed_refine_levels < self.num_levels:
            raise ValueError(
                "refine_levels must be in [1, {}], got {}".format(
                    self.num_levels - 1, parsed_refine_levels
                )
            )

        self.refine_levels = parsed_refine_levels
        self.eps = float(eps)
        self.query_projection = nn.Linear(self.query_dim, self.d_model)
        self.coarse_projections = nn.ModuleList()
        self.gate_mlps = nn.ModuleList()
        for _ in range(self.refine_levels):
            coarse_projection = nn.Conv1d(
                self.d_model, self.d_model, kernel_size=1
            )
            gate_mlp = nn.Sequential(
                nn.Conv1d(3 * self.d_model, self.d_model, kernel_size=1),
                nn.GELU(),
                nn.Conv1d(self.d_model, 1, kernel_size=1),
            )
            nn.init.zeros_(coarse_projection.bias)
            nn.init.zeros_(gate_mlp[0].bias)
            nn.init.normal_(gate_mlp[-1].weight, std=1e-3)
            nn.init.constant_(gate_mlp[-1].bias, float(initial_gate_bias))
            self.coarse_projections.append(coarse_projection)
            self.gate_mlps.append(gate_mlp)

        nn.init.xavier_uniform_(self.query_projection.weight)
        nn.init.zeros_(self.query_projection.bias)
        self.last_gate_values = None
        self.last_alignment_modes = None

    @staticmethod
    def _normalize_mask(mask, batch_size, seq_len, device):
        if mask.ndim == 2:
            mask = mask.unsqueeze(1)
        if mask.shape != (batch_size, 1, seq_len):
            raise ValueError(
                "mask shape {} does not match ({}, 1, {})".format(
                    tuple(mask.shape), batch_size, seq_len
                )
            )
        return mask.to(device=device, dtype=torch.bool)

    def _pool_query(
        self,
        query_feat,
        query_mask,
        batch_size,
        dtype,
        device,
    ):
        if query_feat is None:
            return torch.zeros(
                (batch_size, self.d_model), dtype=dtype, device=device
            )
        if query_feat.ndim == 2:
            if query_feat.shape != (batch_size, self.query_dim):
                raise ValueError(
                    "pooled query_feat must have shape ({}, {})".format(
                        batch_size, self.query_dim
                    )
                )
            pooled = query_feat.to(device=device, dtype=dtype)
            return self.query_projection(pooled)

        if query_feat.ndim != 3 or query_feat.size(0) != batch_size:
            raise ValueError(
                "query_feat must have shape (B, Cq), (B, Cq, Lq), "
                "or (B, Lq, Cq)"
            )
        if query_feat.size(1) == self.query_dim:
            query_tokens = query_feat
        elif query_feat.size(2) == self.query_dim:
            query_tokens = query_feat.transpose(1, 2)
        else:
            raise ValueError(
                "query feature dimension does not match query_dim={}".format(
                    self.query_dim
                )
            )
        query_tokens = query_tokens.to(device=device, dtype=dtype)
        query_len = query_tokens.size(-1)

        if query_mask is None:
            valid = torch.ones(
                (batch_size, query_len), dtype=torch.bool, device=device
            )
        else:
            if query_mask.ndim == 3:
                if query_mask.size(1) != 1:
                    raise ValueError(
                        "query_mask must have shape (B, 1, Lq) or (B, Lq)"
                    )
                query_mask = query_mask[:, 0]
            if query_mask.shape != (batch_size, query_len):
                raise ValueError(
                    "query_mask shape {} does not match query length {}".format(
                        tuple(query_mask.shape), query_len
                    )
                )
            valid = query_mask.to(device=device, dtype=torch.bool)

        valid_float = valid.unsqueeze(1).to(dtype=dtype)
        denominator = valid_float.sum(dim=-1).clamp(min=1.0)
        pooled = (
            query_tokens * valid_float
        ).sum(dim=-1) / denominator
        return self.query_projection(pooled)

    def _align_with_assignment(
        self,
        coarse,
        coarse_mask,
        fine_mask,
        assignment,
    ):
        batch_size, channels, coarse_len = coarse.shape
        fine_len = fine_mask.size(-1)
        expected_shape = (batch_size, coarse_len, fine_len)
        if assignment.shape != expected_shape:
            raise ValueError(
                "assignment shape {} does not match {}".format(
                    tuple(assignment.shape), expected_shape
                )
            )
        weights = assignment.to(
            device=coarse.device, dtype=coarse.dtype
        ).clamp_min(0.0)
        weights = weights * coarse_mask[:, 0].unsqueeze(-1).to(
            dtype=coarse.dtype
        )
        weights = weights * fine_mask.to(dtype=coarse.dtype)
        denominator = weights.sum(dim=1, keepdim=True)
        weights = weights / denominator.clamp(min=self.eps)
        weights = weights * (denominator > 0).to(dtype=weights.dtype)
        aligned = torch.bmm(coarse, weights)
        return aligned * fine_mask.to(dtype=aligned.dtype)

    @staticmethod
    def _align_with_coordinates(
        coarse, coarse_metadata, coarse_mask, fine_metadata, fine_mask, eps=1e-6
    ):
        """Mask-aware linear interpolation in original temporal coordinates."""
        batch_size, channels, coarse_len = coarse.shape
        fine_len = fine_mask.size(-1)
        if coarse_metadata.shape != (batch_size, coarse_len, 5):
            raise ValueError("coarse temporal metadata shape mismatch")
        if fine_metadata.shape != (batch_size, fine_len, 5):
            raise ValueError("fine temporal metadata shape mismatch")
        aligned = coarse.new_zeros((batch_size, channels, fine_len))
        for batch_index in range(batch_size):
            cidx = torch.nonzero(coarse_mask[batch_index, 0], as_tuple=False).flatten()
            fidx = torch.nonzero(fine_mask[batch_index, 0], as_tuple=False).flatten()
            if cidx.numel() == 0 or fidx.numel() == 0:
                continue
            centers = coarse_metadata[batch_index, cidx, 2].float()
            samples = fine_metadata[batch_index, fidx, 2].float()
            values = coarse[batch_index, :, cidx]
            if cidx.numel() == 1:
                result = values.expand(-1, fidx.numel())
            else:
                right = torch.searchsorted(centers, samples).clamp(1, centers.numel() - 1)
                left = right - 1
                c0 = centers[left]
                c1 = centers[right]
                alpha = ((samples - c0) / (c1 - c0).clamp(min=eps)).clamp(0, 1)
                v0 = values[:, left]
                v1 = values[:, right]
                result = v0 + (v1 - v0) * alpha.to(
                    dtype=coarse.dtype
                ).unsqueeze(0)
            aligned[batch_index, :, fidx] = result
        return aligned * fine_mask.to(dtype=aligned.dtype)

    @staticmethod
    def _align_with_interpolation(coarse, coarse_mask, fine_mask):
        batch_size, channels, _ = coarse.shape
        fine_len = fine_mask.size(-1)
        samples = []
        for batch_index in range(batch_size):
            coarse_indices = torch.nonzero(
                coarse_mask[batch_index, 0], as_tuple=False
            ).flatten()
            fine_indices = torch.nonzero(
                fine_mask[batch_index, 0], as_tuple=False
            ).flatten()
            sample = coarse.new_zeros((channels, fine_len))
            if coarse_indices.numel() and fine_indices.numel():
                coarse_valid = coarse[batch_index].index_select(
                    1, coarse_indices
                )
                if coarse_indices.numel() == 1:
                    resized = coarse_valid.expand(
                        channels, fine_indices.numel()
                    )
                else:
                    resized = F.interpolate(
                        coarse_valid.unsqueeze(0),
                        size=fine_indices.numel(),
                        mode="linear",
                        align_corners=True,
                    )[0]
                sample = sample.index_copy(1, fine_indices, resized)
            samples.append(sample)
        return torch.stack(samples, dim=0)

    def forward(
        self,
        features,
        masks,
        query_feat=None,
        query_mask=None,
        assignments=None,
        temporal_metadata=None,
    ):
        if len(features) != self.num_levels or len(masks) != self.num_levels:
            raise ValueError(
                "expected {} FPN levels, got {} features and {} masks".format(
                    self.num_levels, len(features), len(masks)
                )
            )
        if assignments is not None and len(assignments) < self.refine_levels:
            raise ValueError(
                "expected at least {} assignment levels, got {}".format(
                    self.refine_levels, len(assignments)
                )
            )
        if temporal_metadata is not None and len(temporal_metadata) != self.num_levels:
            raise ValueError("temporal metadata must match all FPN levels")

        normalized_masks = []
        for feature, mask in zip(features, masks):
            if feature.ndim != 3 or feature.size(1) != self.d_model:
                raise ValueError(
                    "each feature must have shape (B, {}, T)".format(
                        self.d_model
                    )
                )
            normalized_masks.append(
                self._normalize_mask(
                    mask,
                    feature.size(0),
                    feature.size(-1),
                    feature.device,
                )
            )

        reference = features[0]
        pooled_query = self._pool_query(
            query_feat,
            query_mask,
            reference.size(0),
            reference.dtype,
            reference.device,
        )
        outputs = list(features)
        gate_values = [None] * self.refine_levels
        alignment_modes = [None] * self.refine_levels

        for level in reversed(range(self.refine_levels)):
            fine = outputs[level]
            coarse = outputs[level + 1]
            fine_mask = normalized_masks[level]
            coarse_mask = normalized_masks[level + 1]
            assignment = (
                None if assignments is None else assignments[level]
            )
            if temporal_metadata is not None:
                aligned = self._align_with_coordinates(
                    coarse,
                    temporal_metadata[level + 1],
                    coarse_mask,
                    temporal_metadata[level],
                    fine_mask,
                    eps=self.eps,
                )
                # Keep the legacy debug label while alignment uses true coordinates.
                alignment_modes[level] = "assignment" if assignment is not None else "coordinates"
            elif assignment is None:
                aligned = self._align_with_interpolation(
                    coarse, coarse_mask, fine_mask
                )
                alignment_modes[level] = "interpolation"
            else:
                aligned = self._align_with_assignment(
                    coarse,
                    coarse_mask,
                    fine_mask,
                    assignment,
                )
                alignment_modes[level] = "assignment"

            query = pooled_query.unsqueeze(-1).expand(
                -1, -1, fine.size(-1)
            )
            gate_input = torch.cat((fine, aligned, query), dim=1)
            gate = torch.sigmoid(self.gate_mlps[level](gate_input))
            delta = self.coarse_projections[level](aligned)
            outputs[level] = (
                fine + gate * delta
            ) * fine_mask.to(dtype=fine.dtype)
            gate_values[level] = gate.detach()

        self.last_gate_values = tuple(gate_values)
        self.last_alignment_modes = tuple(alignment_modes)
        return tuple(outputs)


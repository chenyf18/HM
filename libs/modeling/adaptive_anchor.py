import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F



def _balanced_cut_offsets(
    uniform_cut_offsets, valid_count, target_count, importance,
    reallocation_ratio, min_group_size_set=(1, 2, 4),
):
    """Pairwise split/merge on top of the uniform partition.

    Split: one full size-2 group -> two size-1 groups (+1 cut at its midpoint).
    Merge: two adjacent untouched size-2 groups -> one size-4 group (-1 cut).
    Returns sorted cut positions with exactly ``target_count - 1`` entries and
    all resulting group sizes in ``min_group_size_set``.
    """
    device = uniform_cut_offsets.device if torch.is_tensor(uniform_cut_offsets)         else "cpu"
    if target_count <= 1 or valid_count < 4:
        return torch.zeros(0, dtype=torch.long, device=device)
    cuts = sorted(int(c) for c in uniform_cut_offsets)
    starts = [0] + cuts
    sizes = [
        (starts[g + 1] - starts[g]) if g + 1 < len(starts) else
        (valid_count - starts[g])
        for g in range(len(starts))
    ]
    num_groups = target_count
    group_scores = []
    for g in range(num_groups):
        lo = starts[g]
        hi = starts[g + 1] if g + 1 < len(starts) else valid_count
        if importance is not None and hi > lo:
            seg = importance[lo:hi]
            group_scores.append(float(seg.float().mean()) if seg.numel() else 0.0)
        else:
            group_scores.append(0.0)
    full = [g for g in range(num_groups) if sizes[g] == 2]
    n_pairs = max(0, int(reallocation_ratio * num_groups) // 2)
    n_pairs = min(n_pairs, len(full) // 3) if len(full) >= 3 else 0

    ranked = sorted(full, key=lambda g: group_scores[g], reverse=True)
    split_set, merged = [], set()
    used = set()
    si = 0
    merges_needed = n_pairs
    merge_candidates = []
    for idx, g in enumerate(sorted(full)[:-1]):
        h = sorted(full)[idx + 1]
        if h == g + 1 and (g + 1) not in full:
            continue
        if h != g + 1:
            continue
        merge_candidates.append((group_scores[g] + group_scores[h], g, h))
    merge_candidates.sort(key=lambda x: x[0])
    used_groups = set()
    mi = 0
    while len(split_set) < n_pairs and si < len(ranked):
        g = ranked[si]
        si += 1
        used_groups.add(g)
        split_set.append(g)
    while mi < len(merge_candidates) and len(merged) < n_pairs:
        _, g, h = merge_candidates[mi]
        mi += 1
        if g in used_groups or h in used_groups:
            continue
        used_groups.add(g)
        used_groups.add(h)
        merged.add((g, h))
    # Pair down symmetrically: keep equal numbers of splits and merges.
    k = min(len(split_set), len(merged))
    split_set = split_set[:k]
    kept_merges = []
    used_groups = set(g for g in split_set)
    for _, g, h in merge_candidates:
        if len(kept_merges) >= k:
            break
        if g in used_groups or h in used_groups:
            continue
        used_groups.update((g, h))
        kept_merges.append((g, h))
    remove_cuts = {starts[h] for (_, h) in kept_merges}
    add_cuts = {starts[g] + 1 for g in split_set}
    final = sorted(set(cuts) - remove_cuts | add_cuts)
    # Fail-fast: every group size must be in the allowed set.
    bounds = [0] + final + [valid_count]
    for a, b in zip(bounds[:-1], bounds[1:]):
        if (b - a) not in min_group_size_set:
            raise ValueError(
                f"balanced reallocation produced group size {b - a}, "
                f"allowed {min_group_size_set}"
            )
    if len(final) != target_count - 1:
        raise ValueError(
            "balanced reallocation broke the anchor budget: "
            f"{len(final)} cuts != {target_count - 1}"
        )
    _ = device
    return torch.tensor(final, dtype=torch.long, device=device)


class QueryBoundaryAdaptiveAnchorAllocator(nn.Module):
    """Allocate a fixed-budget, temporally ordered set of adaptive anchors."""

    def __init__(
        self,
        target_keep_ratio: float = 0.75,
        importance_weighted_pooling: bool = True,
        allocator_policy: str = "learned",
        random_seed: int = 0,
        oracle_boundary_radius: float = 1.0,
        reallocation_ratio: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()
        target_keep_ratio = float(target_keep_ratio)
        if not 0.0 < target_keep_ratio <= 1.0:
            raise ValueError("target_keep_ratio must be in (0, 1]")
        self.target_keep_ratio = target_keep_ratio
        self.importance_weighted_pooling = bool(
            importance_weighted_pooling
        )
        allocator_policy = str(allocator_policy).strip().lower()
        if allocator_policy not in ("uniform", "random", "learned", "oracle",
                                    "balanced"):
            raise ValueError(
                "allocator_policy must be one of uniform, random, learned, oracle"
            )
        if not math.isfinite(float(oracle_boundary_radius)) or float(
            oracle_boundary_radius
        ) <= 0:
            raise ValueError("oracle_boundary_radius must be finite and positive")
        self.allocator_policy = allocator_policy
        self.random_seed = int(random_seed)
        if not torch.isfinite(torch.tensor(float(reallocation_ratio))) or not (
            0.0 <= float(reallocation_ratio) <= 1.0
        ):
            raise ValueError("reallocation_ratio must be in [0, 1] "
                             "(0 disables reallocation)")
        self.reallocation_ratio = float(reallocation_ratio)
        self.oracle_boundary_radius = float(oracle_boundary_radius)
        self.eps = float(eps)

    @staticmethod
    def _normalize_mask(mask, batch_size, seq_len, device):
        if mask.ndim == 3:
            if mask.size(1) != 1:
                raise ValueError("mask must have shape (B, 1, T) or (B, T)")
            mask = mask[:, 0]
        elif mask.ndim != 2:
            raise ValueError("mask must have shape (B, 1, T) or (B, T)")
        if mask.shape != (batch_size, seq_len):
            raise ValueError(
                "mask shape {} does not match input ({}, {})".format(
                    tuple(mask.shape), batch_size, seq_len
                )
            )
        return mask.to(device=device, dtype=torch.bool)

    @staticmethod
    def _normalize_importance(
        importance_score,
        batch_size,
        seq_len,
        dtype,
        device,
    ):
        if importance_score is None:
            return None
        if importance_score.ndim == 3:
            if importance_score.size(1) != 1:
                raise ValueError(
                    "importance_score must have shape (B, T) or (B, 1, T)"
                )
            importance_score = importance_score[:, 0]
        elif importance_score.ndim != 2:
            raise ValueError(
                "importance_score must have shape (B, T) or (B, 1, T)"
            )
        if importance_score.shape != (batch_size, seq_len):
            raise ValueError(
                "importance_score shape {} does not match input ({}, {})".format(
                    tuple(importance_score.shape), batch_size, seq_len
                )
            )
        return importance_score.to(device=device, dtype=dtype)

    def _target_count(self, valid_count: int) -> int:
        if valid_count == 0:
            return 0
        return min(
            valid_count,
            max(1, int(math.ceil(valid_count * self.target_keep_ratio))),
        )

    @staticmethod
    def _normalize_targets(targets, batch_size, dtype, device):
        if targets is None:
            return None
        if targets.ndim != 2 or targets.shape != (batch_size, 2):
            raise ValueError(
                "allocator_targets must have shape ({}, 2)".format(batch_size)
            )
        targets = targets.to(device=device, dtype=dtype)
        if not torch.isfinite(targets).all():
            raise ValueError("allocator_targets must be finite")
        return targets

    @staticmethod
    def _normalize_temporal_metadata(
        metadata, batch_size, seq_len, dtype, device
    ):
        if metadata is None:
            return None
        if metadata.ndim != 3 or metadata.size(0) != batch_size:
            raise ValueError("temporal_metadata must have shape (B, T, D)")
        if metadata.size(1) != seq_len or metadata.size(2) < 4:
            raise ValueError(
                "temporal_metadata shape {} does not match ({}, {}, >=4)".format(
                    tuple(metadata.shape), batch_size, seq_len
                )
            )
        metadata = metadata.to(device=device, dtype=dtype)
        if not torch.isfinite(metadata).all():
            raise ValueError("temporal_metadata must be finite")
        return metadata

    def _oracle_resolution_weights(
        self, temporal_metadata, targets, valid, dtype
    ):
        """Prioritize GT boundaries, then foreground, then background."""
        if temporal_metadata is None or targets is None:
            return None
        centers = temporal_metadata[..., 2]
        spans = temporal_metadata[..., 3].abs().clamp_min(self.eps)
        starts = torch.minimum(targets[:, 0], targets[:, 1]).unsqueeze(1)
        ends = torch.maximum(targets[:, 0], targets[:, 1]).unsqueeze(1)
        token_start = temporal_metadata[..., 0]
        token_end = temporal_metadata[..., 1]
        foreground = (token_end >= starts) & (token_start <= ends)
        boundary_radius = spans * self.oracle_boundary_radius
        boundary = (
            (torch.abs(centers - starts) <= boundary_radius)
            | (torch.abs(centers - ends) <= boundary_radius)
        )
        weights = torch.ones_like(centers, dtype=dtype)
        weights = torch.where(foreground, weights * 2.0, weights)
        weights = torch.where(boundary, weights * 2.0, weights)
        return weights * valid.to(dtype)

    @staticmethod
    def _uniform_offsets(valid_counts, target_counts, max_anchors):
        steps = torch.arange(
            1, max_anchors, dtype=torch.long, device=valid_counts.device
        ).unsqueeze(0)
        # Ceil places a possible short group at the padded tail, matching
        # fixed stride-2 grouping when keep_ratio is 0.5.
        numerator = steps * valid_counts.unsqueeze(1)
        denominator = target_counts.clamp_min(1).unsqueeze(1)
        return torch.div(
            numerator + denominator - 1,
            denominator,
            rounding_mode="floor",
        )

    def _random_offsets(self, valid_counts, target_counts, max_anchors, seq_len):
        if max_anchors <= 1 or seq_len <= 1:
            return torch.empty(
                (valid_counts.size(0), 0),
                dtype=torch.long,
                device=valid_counts.device,
            )
        generator = torch.Generator(device=valid_counts.device)
        generator.manual_seed(self.random_seed)
        scores = torch.rand(
            (valid_counts.size(0), seq_len - 1),
            device=valid_counts.device,
            generator=generator,
        )
        positions = torch.arange(seq_len - 1, device=valid_counts.device)
        scores = scores.masked_fill(
            positions.unsqueeze(0) >= (valid_counts - 1).unsqueeze(1),
            -torch.inf,
        )
        ranked = torch.argsort(scores, dim=-1, descending=True, stable=True)
        # Cut offsets must be temporally sorted: the reference path places
        # anchors directly from these offsets, while the tensorized path only
        # scatters markers, so sorting here keeps both paths consistent.
        return (ranked[:, : max_anchors - 1] + 1).sort(dim=-1).values

    def _oracle_offsets(self, valid_counts, target_counts, max_anchors, weights):
        uniform = self._uniform_offsets(valid_counts, target_counts, max_anchors)
        if weights is None or max_anchors <= 1:
            return uniform
        steps_long = torch.arange(
            1, max_anchors, dtype=torch.long, device=weights.device
        ).unsqueeze(0)
        cumulative = weights.float().cumsum(dim=-1)
        last_index = (valid_counts - 1).clamp_min(0).unsqueeze(1)
        total = cumulative.gather(1, last_index).clamp_min(self.eps)
        desired = (
            steps_long.float()
            * total
            / target_counts.clamp_min(1).float().unsqueeze(1)
        )
        offsets = torch.searchsorted(cumulative, desired, right=False) + 1
        offsets = torch.maximum(offsets, steps_long)
        offsets = (
            torch.cummax(offsets - steps_long, dim=1).values + steps_long
        )
        max_offsets = (
            valid_counts.unsqueeze(1)
            - target_counts.unsqueeze(1)
            + steps_long
        )
        return torch.minimum(offsets, max_offsets)

    @staticmethod
    def _select_cut_offsets(valid_count, target_count, importance, device):
        if target_count <= 1:
            return torch.empty(0, dtype=torch.long, device=device)
        if importance is None:
            steps = torch.arange(1, target_count, device=device)
            return torch.div(
                steps * valid_count,
                target_count,
                rounding_mode="floor",
            ).to(dtype=torch.long)

        boundary_score = torch.maximum(importance[:-1], importance[1:])
        ranked_boundaries = torch.argsort(
            boundary_score.detach(), descending=True, stable=True
        )
        return torch.sort(
            ranked_boundaries[:target_count - 1] + 1
        ).values

    def _forward_reference(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        importance_score: Optional[torch.Tensor] = None,
        allocator_targets: Optional[torch.Tensor] = None,
        temporal_metadata: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Correctness-first reference implementation for diagnostics."""
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape (B, D, T)")
        batch_size, channels, seq_len = tokens.shape
        valid = self._normalize_mask(
            mask, batch_size, seq_len, tokens.device
        )
        importance_score = self._normalize_importance(
            importance_score,
            batch_size,
            seq_len,
            tokens.dtype,
            tokens.device,
        )
        if importance_score is not None:
            importance_score = importance_score * valid.to(tokens.dtype)
        allocator_targets = self._normalize_targets(
            allocator_targets, batch_size, tokens.dtype, tokens.device
        )
        temporal_metadata = self._normalize_temporal_metadata(
            temporal_metadata,
            batch_size,
            seq_len,
            tokens.dtype,
            tokens.device,
        )
        oracle_weights = self._oracle_resolution_weights(
            temporal_metadata, allocator_targets, valid, tokens.dtype
        )
        if self.allocator_policy == "oracle" and oracle_weights is None:
            raise ValueError(
                "oracle allocator requires allocator_targets and temporal_metadata"
            )

        valid_counts = valid.sum(dim=-1)
        target_counts_list = [
            self._target_count(int(count.item())) for count in valid_counts
        ]
        target_counts = torch.tensor(
            target_counts_list, dtype=torch.long, device=tokens.device
        )
        max_anchors = max(target_counts_list, default=0)
        combined_len = seq_len + max_anchors

        assignment_matrix = tokens.new_zeros(
            (batch_size, max_anchors, seq_len)
        )
        anchor_positions = torch.full(
            (batch_size, max_anchors),
            -1,
            dtype=torch.long,
            device=tokens.device,
        )
        sequence_positions = torch.empty(
            (batch_size, seq_len), dtype=torch.long, device=tokens.device
        )

        sequence = tokens.transpose(1, 2)
        combined_samples = []
        anchor_samples = []
        expanded_masks = []
        for batch_index in range(batch_size):
            valid_indices = torch.nonzero(
                valid[batch_index], as_tuple=False
            ).flatten()
            invalid_indices = torch.nonzero(
                ~valid[batch_index], as_tuple=False
            ).flatten()
            valid_count = valid_indices.numel()
            target_count = target_counts_list[batch_index]
            valid_importance = (
                importance_score[batch_index].index_select(0, valid_indices)
                if (
                    importance_score is not None
                    and self.allocator_policy == "learned"
                )
                else None
            )

            if self.allocator_policy == "random":
                cut_offsets = self._random_offsets(
                    valid_counts[batch_index:batch_index + 1],
                    target_counts[batch_index:batch_index + 1],
                    target_count,
                    valid_count,
                )[0]
            elif self.allocator_policy == "oracle":
                sample_weights = (
                    oracle_weights[batch_index:batch_index + 1, :valid_count]
                    if oracle_weights is not None else None
                )
                cut_offsets = self._oracle_offsets(
                    valid_counts[batch_index:batch_index + 1],
                    target_counts[batch_index:batch_index + 1],
                    target_count,
                    sample_weights,
                )[0]
            elif self.allocator_policy == "uniform":
                cut_offsets = self._uniform_offsets(
                    valid_counts[batch_index:batch_index + 1],
                    target_counts[batch_index:batch_index + 1],
                    target_count,
                )[0]
            elif self.allocator_policy == "balanced":
                imp_b = (
                    importance_score[batch_index, :valid_count]
                    if importance_score is not None else None
                )
                u_off = self._uniform_offsets(
                    valid_counts[batch_index:batch_index + 1],
                    target_counts[batch_index:batch_index + 1],
                    target_count,
                )[0][:max(target_count - 1, 0)]
                cut_offsets = _balanced_cut_offsets(
                    u_off, valid_count, target_count, imp_b,
                    self.reallocation_ratio,
                )
            else:
                cut_offsets = self._select_cut_offsets(
                    valid_count,
                    target_count,
                    valid_importance,
                    tokens.device,
                )
            cut_markers = torch.zeros(
                valid_count, dtype=torch.long, device=tokens.device
            )
            if cut_offsets.numel():
                cut_markers.index_fill_(0, cut_offsets, 1)
            group_ids = torch.cumsum(cut_markers, dim=0)

            group_tokens = sequence[batch_index].index_select(
                0, valid_indices
            )
            if target_count:
                if (
                    self.importance_weighted_pooling
                    and self.allocator_policy == "learned"
                    and valid_importance is not None
                ):
                    weights = valid_importance.clamp_min(0.0) + self.eps
                else:
                    weights = torch.ones(
                        valid_count,
                        dtype=tokens.dtype,
                        device=tokens.device,
                    )
                weighted_tokens = group_tokens * weights.unsqueeze(-1)
                pooled_anchors = tokens.new_zeros(
                    (target_count, channels)
                ).index_add_(0, group_ids, weighted_tokens)
                weight_sums = tokens.new_zeros(target_count).index_add_(
                    0, group_ids, weights
                )
                pooled_anchors = pooled_anchors / weight_sums.unsqueeze(-1)

                assignment_matrix[batch_index, group_ids, valid_indices] = 1.0
                group_starts = torch.cat(
                    (
                        torch.zeros(1, dtype=torch.long, device=tokens.device),
                        cut_offsets,
                    )
                )
                anchor_pos = group_starts + torch.arange(
                    target_count, dtype=torch.long, device=tokens.device
                )
                sequence_pos = torch.arange(
                    valid_count, dtype=torch.long, device=tokens.device
                ) + group_ids + 1
                anchor_positions[batch_index, :target_count] = anchor_pos
                sequence_positions[batch_index].index_copy_(
                    0, valid_indices, sequence_pos
                )

                valid_combined = tokens.new_zeros(
                    (valid_count + target_count, channels)
                )
                valid_combined = valid_combined.index_copy(
                    0, anchor_pos, pooled_anchors
                )
                valid_combined = valid_combined.index_copy(
                    0, sequence_pos, group_tokens
                )
            else:
                pooled_anchors = tokens.new_zeros((0, channels))
                valid_combined = tokens.new_zeros((0, channels))

            invalid_tokens = sequence[batch_index].index_select(
                0, invalid_indices
            )
            invalid_positions = torch.arange(
                valid_count + target_count,
                valid_count + target_count + invalid_indices.numel(),
                dtype=torch.long,
                device=tokens.device,
            )
            sequence_positions[batch_index].index_copy_(
                0, invalid_indices, invalid_positions
            )
            sample_combined = torch.cat(
                (valid_combined, invalid_tokens), dim=0
            )

            anchor_padding = max_anchors - target_count
            if anchor_padding:
                sample_combined = F.pad(
                    sample_combined, (0, 0, 0, anchor_padding)
                )
                pooled_anchors = F.pad(
                    pooled_anchors, (0, 0, 0, anchor_padding)
                )
            if sample_combined.size(0) != combined_len:
                raise RuntimeError("adaptive interleave produced invalid length")

            valid_combined_len = valid_count + target_count
            expanded_masks.append(
                torch.arange(combined_len, device=tokens.device)
                < valid_combined_len
            )
            combined_samples.append(sample_combined)
            anchor_samples.append(pooled_anchors)

        anchor_mask = (
            torch.arange(max_anchors, device=tokens.device).unsqueeze(0)
            < target_counts.unsqueeze(1)
        ).unsqueeze(1)
        combined = torch.stack(combined_samples, dim=0)
        anchors = torch.stack(anchor_samples, dim=0).transpose(1, 2)
        expanded_mask = torch.stack(expanded_masks, dim=0).unsqueeze(-1)

        return {
            "combined": combined,
            "anchors": anchors,
            "anchor_positions": anchor_positions,
            "sequence_positions": sequence_positions,
            "expanded_mask": expanded_mask,
            "anchor_mask": anchor_mask,
            "sequence_mask": valid.unsqueeze(1),
            "assignment_matrix": assignment_matrix,
            "target_counts": target_counts,
        }

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        importance_score: Optional[torch.Tensor] = None,
        allocator_targets: Optional[torch.Tensor] = None,
        temporal_metadata: Optional[torch.Tensor] = None,
        forced_cut_offsets: Optional[torch.Tensor] = None,
        forced_target_counts: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Build anchors with batched tensor operations.

        The reference path above is intentionally retained for equivalence
        checks.  This path keeps hard contiguous cuts and the exact interleave
        layout, while doing cut ranking, grouping, pooling, and packing for the
        whole batch at once.
        """
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape (B, D, T)")
        batch_size, channels, seq_len = tokens.shape
        valid = self._normalize_mask(mask, batch_size, seq_len, tokens.device)
        importance_score = self._normalize_importance(
            importance_score,
            batch_size,
            seq_len,
            tokens.dtype,
            tokens.device,
        )
        if importance_score is not None:
            importance_score = importance_score * valid.to(tokens.dtype)
        allocator_targets = self._normalize_targets(
            allocator_targets, batch_size, tokens.dtype, tokens.device
        )
        temporal_metadata = self._normalize_temporal_metadata(
            temporal_metadata,
            batch_size,
            seq_len,
            tokens.dtype,
            tokens.device,
        )
        oracle_weights = self._oracle_resolution_weights(
            temporal_metadata, allocator_targets, valid, tokens.dtype
        )
        if self.allocator_policy == "oracle" and oracle_weights is None:
            raise ValueError(
                "oracle allocator requires allocator_targets and temporal_metadata"
            )
        if self.allocator_policy != "oracle" and allocator_targets is not None:
            raise ValueError(
                "allocator_targets are train-only GT supervision reserved for "
                "the oracle policy; refusing GT input to a non-oracle allocator"
            )

        valid_counts = valid.sum(dim=-1)
        if forced_target_counts is not None:
            # Offline counterfactual: a single split/merge shifts the anchor
            # count by +/-1 by definition, so the budget is overridden too.
            target_counts = forced_target_counts.to(
                device=tokens.device, dtype=torch.long
            ).clamp(min=0)
            target_counts = torch.minimum(target_counts, valid_counts)
        else:
            target_counts = torch.ceil(
                valid_counts.to(torch.float32) * self.target_keep_ratio
            ).to(torch.long)
        target_counts = torch.minimum(target_counts, valid_counts)
        target_counts = torch.where(
            valid_counts > 0,
            target_counts.clamp_min(1),
            torch.zeros_like(target_counts),
        )
        # One batch-level scalar is needed for the packed output shape.  There
        # are no per-sample scalar reads or host-side loops in the hot path.
        max_anchors = int(target_counts.max().detach()) if batch_size else 0
        combined_len = seq_len + max_anchors
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
        valid_positions = positions < valid_counts.unsqueeze(1)

        cut_markers = torch.zeros(
            (batch_size, seq_len), dtype=torch.long, device=tokens.device
        )
        if max_anchors > 1 and seq_len > 1:
            steps = torch.arange(
                1, max_anchors, dtype=torch.long, device=tokens.device
            ).unsqueeze(0)
            active_cuts = steps < target_counts.unsqueeze(1)
            if forced_cut_offsets is not None:
                offsets = forced_cut_offsets.to(
                    device=tokens.device, dtype=torch.long
                )
            elif (
                self.allocator_policy == "uniform"
                or (
                    self.allocator_policy == "learned"
                    and importance_score is None
                )
            ):
                offsets = self._uniform_offsets(
                    valid_counts, target_counts, max_anchors
                )
            elif self.allocator_policy == "random":
                offsets = self._random_offsets(
                    valid_counts, target_counts, max_anchors, seq_len
                )
            elif self.allocator_policy == "oracle":
                offsets = self._oracle_offsets(
                    valid_counts, target_counts, max_anchors, oracle_weights
                )
            elif self.allocator_policy == "balanced":
                if forced_cut_offsets is not None:
                    offsets = forced_cut_offsets
                else:
                    rows = []
                for b in range(batch_size):
                    vc = int(valid_counts[b])
                    tc = int(target_counts[b])
                    imp_b = (
                        importance_score[b, :vc]
                        if importance_score is not None else None
                    )
                    u_off = self._uniform_offsets(
                        valid_counts[b:b + 1], target_counts[b:b + 1],
                        max_anchors,
                    )[0][:max(tc - 1, 0)]
                    rows.append(_balanced_cut_offsets(
                        u_off, vc, tc, imp_b, self.reallocation_ratio,
                    ))
                width = max((r.numel() for r in rows), default=0)
                offsets = torch.full(
                    (batch_size, width), int(seq_len),
                    dtype=torch.long, device=tokens.device,
                )
                if forced_cut_offsets is None:
                    for b, r in enumerate(rows):
                        if r.numel():
                            offsets[b, :r.numel()] = r
            else:
                boundary_score = torch.maximum(
                    importance_score[:, :-1], importance_score[:, 1:]
                ).detach()
                boundary_valid = positions[:, :-1] < (
                    valid_counts - 1
                ).unsqueeze(1)
                ranked = torch.argsort(
                    boundary_score.masked_fill(
                        ~boundary_valid, -torch.inf
                    ),
                    dim=-1,
                    descending=True,
                    stable=True,
                )[:, :max_anchors - 1]
                offsets = ranked + 1
            cut_markers.scatter_add_(
                1,
                offsets.clamp(min=0, max=max(seq_len - 1, 0)),
                active_cuts.to(torch.long),
            )
            cut_markers.clamp_(max=1)

        group_ids = torch.cumsum(cut_markers, dim=-1)
        if max_anchors:
            group_index = group_ids.clamp(max=max_anchors - 1)
            anchor_valid = (
                torch.arange(max_anchors, device=tokens.device).unsqueeze(0)
                < target_counts.unsqueeze(1)
            )
            valid_float = valid.to(tokens.dtype)
            sequence = tokens.transpose(1, 2)
            if (
                self.importance_weighted_pooling
                and self.allocator_policy == "learned"
                and importance_score is not None
            ):
                weights = importance_score.clamp_min(0.0) + self.eps
            else:
                weights = torch.ones_like(valid_float)
            effective_weights = weights * valid_float
            weighted_tokens = sequence * effective_weights.unsqueeze(-1)
            pooled_anchors = tokens.new_zeros(
                (batch_size, max_anchors, channels)
            )
            pooled_anchors.scatter_add_(
                1,
                group_index.unsqueeze(-1).expand(-1, -1, channels),
                weighted_tokens,
            )
            weight_sums = tokens.new_zeros((batch_size, max_anchors))
            weight_sums.scatter_add_(1, group_index, effective_weights)
            safe_weight_sums = torch.where(
                anchor_valid, weight_sums, torch.ones_like(weight_sums)
            )
            pooled_anchors = pooled_anchors / safe_weight_sums.unsqueeze(-1)
            pooled_anchors = pooled_anchors * anchor_valid.unsqueeze(-1).to(
                pooled_anchors.dtype
            )

            # The minimum valid token index per group is its contiguous start.
            group_starts = torch.full(
                (batch_size, max_anchors),
                seq_len,
                dtype=torch.long,
                device=tokens.device,
            )
            group_starts.scatter_reduce_(
                1,
                group_index,
                torch.where(valid_positions, positions, torch.full_like(positions, seq_len)),
                reduce="amin",
                include_self=True,
            )
            anchor_positions = group_starts + torch.arange(
                max_anchors, device=tokens.device
            ).unsqueeze(0)
            anchor_positions = torch.where(
                anchor_valid, anchor_positions, torch.full_like(anchor_positions, -1)
            )
        else:
            group_index = torch.zeros_like(group_ids)
            anchor_valid = torch.zeros(
                (batch_size, 0), dtype=torch.bool, device=tokens.device
            )
            pooled_anchors = tokens.new_zeros((batch_size, 0, channels))
            anchor_positions = torch.empty(
                (batch_size, 0), dtype=torch.long, device=tokens.device
            )
            sequence = tokens.transpose(1, 2)

        sequence_positions = torch.where(
            valid_positions,
            positions + group_ids + 1,
            positions + target_counts.unsqueeze(1),
        )
        combined = tokens.new_zeros((batch_size, combined_len, channels))
        if max_anchors:
            anchor_indices = anchor_positions.clamp_min(0)
            combined.scatter_add_(
                1,
                anchor_indices.unsqueeze(-1).expand(-1, -1, channels),
                pooled_anchors * anchor_valid.unsqueeze(-1).to(tokens.dtype),
            )
        combined.scatter_add_(
            1,
            sequence_positions.unsqueeze(-1).expand(-1, -1, channels),
            sequence,
        )

        assignment_matrix = tokens.new_zeros(
            (batch_size, max_anchors, seq_len)
        )
        if max_anchors:
            assignment_matrix.scatter_(
                1,
                group_index.unsqueeze(1),
                valid_positions.to(tokens.dtype).unsqueeze(1),
            )
        expanded_mask = (
            torch.arange(combined_len, device=tokens.device).unsqueeze(0)
            < (valid_counts + target_counts).unsqueeze(1)
        ).unsqueeze(-1)
        return {
            "combined": combined,
            "anchors": pooled_anchors.transpose(1, 2),
            "anchor_positions": anchor_positions,
            "sequence_positions": sequence_positions,
            "expanded_mask": expanded_mask,
            "anchor_mask": anchor_valid.unsqueeze(1),
            "sequence_mask": valid.unsqueeze(1),
            "assignment_matrix": assignment_matrix,
            "target_counts": target_counts,
        }

    def reference_forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        importance_score: Optional[torch.Tensor] = None,
        allocator_targets: Optional[torch.Tensor] = None,
        temporal_metadata: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Expose the old implementation for fixed-seed equivalence tests."""
        return self._forward_reference(
            tokens,
            mask,
            importance_score,
            allocator_targets=allocator_targets,
            temporal_metadata=temporal_metadata,
        )

    @staticmethod
    def extract_outputs(
        processed_features: torch.Tensor,
        anchor_positions: torch.Tensor,
        sequence_positions: torch.Tensor,
        anchor_mask: torch.Tensor,
        sequence_mask: torch.Tensor,
    ):
        """Gather adaptive anchor and sequence streams from packed features."""
        batch_size, _, channels = processed_features.shape
        num_anchors = anchor_positions.size(1)
        seq_len = sequence_positions.size(1)

        if num_anchors:
            anchor_indices = anchor_positions.clamp_min(0).unsqueeze(-1)
            anchor_indices = anchor_indices.expand(
                batch_size, num_anchors, channels
            )
            anchor_out = processed_features.gather(
                1, anchor_indices
            ).transpose(1, 2)
            anchor_out = anchor_out * anchor_mask.to(anchor_out.dtype)
        else:
            anchor_out = processed_features.new_zeros(
                (batch_size, channels, 0)
            )

        sequence_indices = sequence_positions.unsqueeze(-1).expand(
            batch_size, seq_len, channels
        )
        sequence_out = processed_features.gather(
            1, sequence_indices
        ).transpose(1, 2)
        sequence_out = sequence_out * sequence_mask.to(sequence_out.dtype)
        return anchor_out, sequence_out

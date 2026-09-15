"""Temporal geometry for query-dependent adaptive FPNs.

The released point generator assumes a regular grid: a level-l point is at a
fixed 2**l stride.  Adaptive grouping breaks that assumption, so this module
keeps the physical support of every representation separate from the scale
used to parameterize regression offsets.
"""

import math

import torch
import torch.nn as nn


# Geometry carried by the backbone for every sequence/FPN representation.
TEMPORAL_METADATA_DIM = 5
METADATA_START = 0
METADATA_END = 1
METADATA_CENTER = 2
METADATA_SPAN = 3
METADATA_REGRESSION_SCALE = 4

# The first four fields deliberately preserve the legacy point interface.
POINT_CENTER = 0
POINT_REGRESSION_MIN = 1
POINT_REGRESSION_MAX = 2
POINT_REGRESSION_SCALE = 3
POINT_START = 4
POINT_END = 5
POINT_GEOMETRY_CENTER = 6
POINT_SPAN = 7
POINT_DIM = 8


def _normalize_mask(mask, batch_size, seq_len, device):
    if mask.ndim == 3:
        if mask.size(1) != 1:
            raise ValueError('mask must have shape (B, 1, T) or (B, T)')
        mask = mask[:, 0]
    elif mask.ndim != 2:
        raise ValueError('mask must have shape (B, 1, T) or (B, T)')
    if mask.shape != (batch_size, seq_len):
        raise ValueError(
            'mask shape {} does not match ({}, {})'.format(
                tuple(mask.shape), batch_size, seq_len
            )
        )
    return mask.to(device=device, dtype=torch.bool)


def _validate_prefix_mask(mask, name='mask'):
    """Require valid tokens to form a left-aligned prefix in every sample."""
    if mask.numel() == 0:
        return
    valid_counts = mask.sum(dim=-1)
    expected = torch.arange(
        mask.size(-1), device=mask.device
    ).unsqueeze(0) < valid_counts.unsqueeze(1)
    if not torch.equal(mask, expected):
        raise ValueError(
            '{} must be a valid-prefix mask; adaptive temporal grouping does '
            'not permit holes'.format(name)
        )


def _as_batched_points(points, batch_size, seq_len, dtype, device):
    if points.ndim == 2:
        if points.size(0) < seq_len or points.size(1) < 4:
            raise ValueError('points must have shape (T, D) with D >= 4')
        return points[:seq_len].to(
            device=device, dtype=dtype
        ).unsqueeze(0).expand(batch_size, -1, -1)
    if points.ndim == 3:
        if (
            points.size(0) != batch_size
            or points.size(1) < seq_len
            or points.size(2) < 4
        ):
            raise ValueError(
                'batched points must have shape (B, T, D) with D >= 4'
            )
        return points[:, :seq_len].to(device=device, dtype=dtype)
    raise ValueError('points must have shape (T, D) or (B, T, D)')


def point_supports(points, batch_size=None, seq_len=None, dtype=None, device=None):
    """Return start/end support coordinates, accepting legacy point tensors."""
    if points.ndim not in (2, 3):
        raise ValueError('points must have shape (T, D) or (B, T, D)')
    if batch_size is None:
        batch_size = 1 if points.ndim == 2 else points.size(0)
    if seq_len is None:
        seq_len = points.size(-2)
    if dtype is None:
        dtype = points.dtype
    if device is None:
        device = points.device
    points = _as_batched_points(
        points, batch_size, seq_len, dtype, device
    )
    if points.size(-1) >= POINT_DIM:
        return points[..., POINT_START], points[..., POINT_END]
    center = points[..., POINT_CENTER]
    scale = points[..., POINT_REGRESSION_SCALE].abs()
    return center - 0.5 * scale, center + 0.5 * scale


def encode_offsets(points, targets, eps=1e-6):
    """Encode [start, end] segments as normalized left/right offsets.

    The same per-point regression scale is used by :func:`decode_offsets`,
    making this parameterization an exact round trip independent of spacing.
    """
    if targets.ndim != 2 or targets.size(-1) != 2:
        raise ValueError('targets must have shape (B, 2)')
    batch_size = targets.size(0)
    seq_len = points.size(-2)
    points = _as_batched_points(
        points, batch_size, seq_len, targets.dtype, targets.device
    )
    centers = points[..., POINT_CENTER]
    scales = points[..., POINT_REGRESSION_SCALE].abs().clamp(min=eps)
    starts = torch.minimum(targets[:, 0], targets[:, 1]).unsqueeze(1)
    ends = torch.maximum(targets[:, 0], targets[:, 1]).unsqueeze(1)
    return torch.stack((centers - starts, ends - centers), dim=-1) / (
        scales.unsqueeze(-1)
    )


def decode_offsets(points, offsets):
    """Decode normalized left/right offsets into temporal segments.

    Temporal coordinate arithmetic is always performed in float32: under
    bf16 autocast the offsets arrive as bf16, and casting centers/scales
    down to the offset dtype quantizes decoded coordinates onto the bf16
    grid (spacing up to 4-8 tokens at large temporal indices), which
    collapses neighbouring candidates and degrades localization (see
    HM-AUDIT-RANK-021). Heads keep their training dtype; only the decode
    is promoted.
    """
    offsets = offsets.to(dtype=torch.float32)
    if points.ndim == 2:
        if offsets.ndim == 2:
            if offsets.shape != (points.size(0), 2):
                raise ValueError(
                    'offsets must have shape (T, 2) for shared points'
                )
            broadcast = False
        elif offsets.ndim == 3:
            if offsets.shape[1:] != (points.size(0), 2):
                raise ValueError(
                    'batched offsets must have shape (B, T, 2)'
                )
            broadcast = True
        centers = points[:, POINT_CENTER].to(
            device=offsets.device, dtype=torch.float32
        )
        scales = points[:, POINT_REGRESSION_SCALE].abs().to(
            device=offsets.device, dtype=torch.float32
        )
        if broadcast:
            centers = centers.unsqueeze(0)
            scales = scales.unsqueeze(0)
    elif points.ndim == 3:
        if (
            offsets.ndim != 3
            or offsets.shape[:2] != points.shape[:2]
            or offsets.size(-1) != 2
        ):
            raise ValueError('offsets must have shape (B, T, 2) for 3-D points')
        centers = points[..., POINT_CENTER].to(
            device=offsets.device, dtype=torch.float32
        )
        scales = points[..., POINT_REGRESSION_SCALE].abs().to(
            device=offsets.device, dtype=torch.float32
        )
    else:
        raise ValueError('points must have shape (T, D) or (B, T, D)')
    return torch.stack((
        centers - offsets[..., 0] * scales,
        centers + offsets[..., 1] * scales,
    ), dim=-1)


class AdaptiveTemporalPointGenerator(nn.Module):
    """Generate query-dependent FPN points from propagated temporal geometry.

    ``metadata`` stores physical geometry as
    ``[start, end, center, span, regression_scale]``.
    The generated point tensor keeps the legacy first four columns
    ``[center, regression_min, regression_max, regression_scale]`` and adds
    the complete geometry in columns 4--7.

    ``regression_scale`` is the local Voronoi-cell width induced by adjacent
    representation centers.  It is intentionally not the anchor support span.
    On a regular hierarchy it is exactly the released fixed stride.
    """

    def __init__(
        self,
        max_seq_len,
        num_fpn_levels,
        regression_range=4,
        sigma=1,
        input_stride=1.0,
        eps=1e-6,
    ):
        super().__init__()
        self.max_seq_len = int(max_seq_len)
        self.num_fpn_levels = int(num_fpn_levels)
        self.eps = float(eps)
        if self.max_seq_len <= 0 or self.num_fpn_levels <= 0:
            raise ValueError('max_seq_len and num_fpn_levels must be positive')
        if not math.isfinite(float(regression_range)) or regression_range <= 0:
            raise ValueError('regression_range must be finite and positive')
        if not 0 < float(sigma) <= 1:
            raise ValueError('sigma must be in (0, 1]')
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError('eps must be finite and positive')

        self.base_regression_range = float(regression_range)
        self.sigma = float(sigma)
        self.input_stride = float(input_stride)
        if not math.isfinite(self.input_stride) or self.input_stride <= 0:
            raise ValueError('input_stride must be finite and positive')

    @staticmethod
    def make_initial_metadata(mask, input_stride=1.0):
        """Create temporal cells in original input-token coordinates."""
        if mask.ndim == 3:
            if mask.size(1) != 1:
                raise ValueError('mask must have shape (B, 1, T) or (B, T)')
            mask = mask[:, 0]
        if mask.ndim != 2:
            raise ValueError('mask must have shape (B, 1, T) or (B, T)')
        batch_size, seq_len = mask.shape
        valid = _normalize_mask(mask, batch_size, seq_len, mask.device)
        _validate_prefix_mask(valid, 'initial temporal mask')
        input_stride = float(input_stride)
        if not math.isfinite(input_stride) or input_stride <= 0:
            raise ValueError('input_stride must be finite and positive')
        centers = torch.arange(
            seq_len, device=mask.device, dtype=torch.float32
        ) * input_stride
        span = torch.full_like(centers, input_stride)
        start = centers - 0.5 * span
        end = centers + 0.5 * span
        regression_scale = span.clone()
        metadata = torch.stack(
            (start, end, centers, span, regression_scale), dim=-1
        )
        metadata = metadata.unsqueeze(0).expand(batch_size, -1, -1).clone()
        return metadata * valid.unsqueeze(-1).to(metadata.dtype)

    @staticmethod
    def _propagate_reference(metadata, assignment_matrix, input_mask, anchor_mask, eps=1e-6):
        """Correctness-first metadata propagation retained for diagnostics."""
        if metadata.ndim != 3 or metadata.size(-1) != TEMPORAL_METADATA_DIM:
            raise ValueError('metadata must have shape (B, T, 5)')
        batch_size, seq_len, _ = metadata.shape
        input_valid = _normalize_mask(
            input_mask, batch_size, seq_len, metadata.device
        )
        _validate_prefix_mask(input_valid, 'input temporal mask')
        if assignment_matrix.ndim != 3 or assignment_matrix.shape[:1] != (batch_size,):
            raise ValueError('assignment_matrix must have shape (B, A, T)')
        if assignment_matrix.size(2) != seq_len:
            raise ValueError(
                'assignment input length {} does not match metadata {}'.format(
                    assignment_matrix.size(2), seq_len
                )
            )
        anchor_count = assignment_matrix.size(1)
        anchor_valid = _normalize_mask(
            anchor_mask, batch_size, anchor_count, metadata.device
        )
        _validate_prefix_mask(anchor_valid, 'anchor temporal mask')

        membership = assignment_matrix.to(
            device=metadata.device, dtype=torch.float32
        ) > 0
        if torch.any(membership & ~input_valid.unsqueeze(1)):
            raise ValueError('assignment_matrix assigns padding tokens')
        membership = membership & input_valid.unsqueeze(1)
        member_count = membership.sum(dim=-1)
        if torch.any(anchor_valid & (member_count == 0)):
            raise ValueError('valid adaptive anchor has an empty group')
        if torch.any(~anchor_valid & (member_count != 0)):
            raise ValueError('padded adaptive anchor has assigned tokens')
        coverage = membership.sum(dim=1)
        if torch.any(input_valid & (coverage != 1)):
            raise ValueError(
                'every valid input representation must belong to exactly one '
                'adaptive anchor'
            )

        for batch_index in range(batch_size):
            valid_anchor_count = int(anchor_valid[batch_index].sum().item())
            for anchor_index in range(valid_anchor_count):
                indices = torch.nonzero(
                    membership[batch_index, anchor_index], as_tuple=False
                ).flatten()
                if indices.numel() != int(
                    indices[-1].item() - indices[0].item() + 1
                ):
                    raise ValueError(
                        'adaptive anchor assignments must be temporally '
                        'contiguous'
                    )

        geometry = metadata.to(dtype=torch.float32)
        starts = geometry[..., METADATA_START].unsqueeze(1)
        ends = geometry[..., METADATA_END].unsqueeze(1)
        inf = torch.finfo(geometry.dtype).max
        out_start = torch.where(
            membership, starts, torch.full_like(starts, inf)
        ).amin(dim=-1)
        out_end = torch.where(
            membership, ends, torch.full_like(ends, -inf)
        ).amax(dim=-1)
        span = (out_end - out_start).clamp(min=float(eps))
        center = 0.5 * (out_start + out_end)

        regression_scale = torch.zeros_like(center)
        for batch_index in range(batch_size):
            valid_count = int(anchor_valid[batch_index].sum().item())
            if valid_count == 0:
                continue
            if valid_count == 1:
                parent_scales = geometry[
                    batch_index, :, METADATA_REGRESSION_SCALE
                ]
                weights = membership[batch_index, 0].to(
                    device=geometry.device, dtype=geometry.dtype
                )
                regression_scale[batch_index, 0] = (
                    parent_scales * weights
                ).sum().clamp(min=float(eps))
                continue
            local_centers = center[batch_index, :valid_count]
            deltas = local_centers[1:] - local_centers[:-1]
            if torch.any(deltas <= float(eps)):
                raise ValueError(
                    'adaptive representation centers must be strictly ordered'
                )
            regression_scale[batch_index, 0] = deltas[0]
            regression_scale[batch_index, valid_count - 1] = deltas[-1]
            if valid_count > 2:
                regression_scale[
                    batch_index, 1:valid_count - 1
                ] = 0.5 * (deltas[:-1] + deltas[1:])

        output = torch.stack(
            (out_start, out_end, center, span, regression_scale), dim=-1
        )
        output = output * anchor_valid.unsqueeze(-1).to(output.dtype)

        if anchor_count > 1:
            valid_pairs = anchor_valid[:, 1:] & anchor_valid[:, :-1]
            ordered = output[:, 1:, METADATA_CENTER] > (
                output[:, :-1, METADATA_CENTER] + float(eps)
            )
            if torch.any(valid_pairs & ~ordered):
                raise ValueError('adaptive anchors must preserve temporal order')
        return output

    @staticmethod
    def propagate(metadata, assignment_matrix, input_mask, anchor_mask, eps=1e-6):
        """Propagate support geometry with batched segmented reductions.

        The output remains ``[start, end, center, span, regression_scale]``.
        Regression scale is derived from adjacent output centers exactly as in
        the reference implementation; it is never replaced by support span.
        """
        if metadata.ndim != 3 or metadata.size(-1) != TEMPORAL_METADATA_DIM:
            raise ValueError('metadata must have shape (B, T, 5)')
        batch_size, seq_len, _ = metadata.shape
        input_valid = _normalize_mask(
            input_mask, batch_size, seq_len, metadata.device
        )
        if (
            assignment_matrix.ndim != 3
            or assignment_matrix.size(0) != batch_size
            or assignment_matrix.size(2) != seq_len
        ):
            raise ValueError('assignment_matrix must have shape (B, A, T)')
        anchor_count = assignment_matrix.size(1)
        anchor_valid = _normalize_mask(
            anchor_mask, batch_size, anchor_count, metadata.device
        )

        membership = assignment_matrix.to(
            device=metadata.device, dtype=torch.float32
        ) > 0
        member_count = membership.sum(dim=-1)
        coverage = membership.sum(dim=1)
        input_expected = (
            torch.arange(seq_len, device=metadata.device).unsqueeze(0)
            < input_valid.sum(dim=-1).unsqueeze(1)
        )
        anchor_expected = (
            torch.arange(anchor_count, device=metadata.device).unsqueeze(0)
            < anchor_valid.sum(dim=-1).unsqueeze(1)
        )
        positions = torch.arange(
            seq_len, device=metadata.device, dtype=torch.long
        ).view(1, 1, seq_len)
        first_member = torch.where(
            membership, positions, torch.full_like(positions, seq_len)
        ).amin(dim=-1)
        last_member = torch.where(
            membership, positions, torch.full_like(positions, -1)
        ).amax(dim=-1)
        contiguous = member_count == (last_member - first_member + 1)

        checks = torch.stack((
            torch.any(input_valid != input_expected),
            torch.any(anchor_valid != anchor_expected),
            torch.any(membership & ~input_valid.unsqueeze(1)),
            torch.any(anchor_valid & (member_count == 0)),
            torch.any(~anchor_valid & (member_count != 0)),
            torch.any(input_valid & (coverage != 1)),
            torch.any(anchor_valid & ~contiguous),
        ))
        if bool(checks.any()):
            failed = checks.detach().cpu().tolist()
            messages = (
                'input temporal mask must be a valid-prefix mask',
                'anchor temporal mask must be a valid-prefix mask',
                'assignment_matrix assigns padding tokens',
                'valid adaptive anchor has an empty group',
                'padded adaptive anchor has assigned tokens',
                'every valid input representation must belong to exactly one adaptive anchor',
                'adaptive anchor assignments must be temporally contiguous',
            )
            raise ValueError(messages[failed.index(True)])

        geometry = metadata.to(dtype=torch.float32)
        starts = geometry[..., METADATA_START].unsqueeze(1)
        ends = geometry[..., METADATA_END].unsqueeze(1)
        limit = torch.finfo(geometry.dtype).max
        raw_start = torch.where(
            membership, starts, torch.full_like(starts, limit)
        ).amin(dim=-1)
        raw_end = torch.where(
            membership, ends, torch.full_like(ends, -limit)
        ).amax(dim=-1)
        out_start = torch.where(
            anchor_valid, raw_start, torch.zeros_like(raw_start)
        )
        out_end = torch.where(
            anchor_valid, raw_end, torch.zeros_like(raw_end)
        )
        span = torch.where(
            anchor_valid,
            (out_end - out_start).clamp(min=float(eps)),
            torch.zeros_like(out_start),
        )
        center = 0.5 * (out_start + out_end)

        regression_scale = torch.zeros_like(center)
        valid_counts = anchor_valid.sum(dim=-1)
        if anchor_count > 1:
            deltas = center[:, 1:] - center[:, :-1]
            valid_pairs = anchor_valid[:, 1:] & anchor_valid[:, :-1]
            if bool(torch.any(valid_pairs & (deltas <= float(eps)))):
                raise ValueError(
                    'adaptive representation centers must be strictly ordered'
                )
            anchor_indices = torch.arange(
                anchor_count, device=metadata.device
            ).unsqueeze(0)
            left_indices = (anchor_indices - 1).clamp(
                min=0, max=anchor_count - 2
            )
            right_indices = anchor_indices.clamp(
                min=0, max=anchor_count - 2
            )
            left_delta = deltas.gather(1, left_indices.expand(batch_size, -1))
            right_delta = deltas.gather(1, right_indices.expand(batch_size, -1))
            multi_scale = 0.5 * (left_delta + right_delta)
            multi_scale = torch.where(
                anchor_indices == 0, right_delta, multi_scale
            )
            multi_scale = torch.where(
                anchor_indices == (valid_counts - 1).unsqueeze(1),
                left_delta,
                multi_scale,
            )
            regression_scale = torch.where(
                (valid_counts > 1).unsqueeze(1) & anchor_valid,
                multi_scale,
                regression_scale,
            )
        if anchor_count:
            parent_scales = geometry[
                :, :, METADATA_REGRESSION_SCALE
            ].unsqueeze(1)
            single_scale = (
                parent_scales * membership.to(geometry.dtype)
            ).sum(dim=-1).clamp(min=float(eps))
            regression_scale = torch.where(
                (valid_counts == 1).unsqueeze(1) & anchor_valid,
                single_scale,
                regression_scale,
            )
        regression_scale = regression_scale * anchor_valid.to(
            regression_scale.dtype
        )

        output = torch.stack(
            (out_start, out_end, center, span, regression_scale), dim=-1
        )
        return torch.where(
            anchor_valid.unsqueeze(-1), output, torch.zeros_like(output)
        )

    @staticmethod
    def reference_propagate(
        metadata, assignment_matrix, input_mask, anchor_mask, eps=1e-6
    ):
        """Expose the old propagation path for equivalence tests."""
        return AdaptiveTemporalPointGenerator._propagate_reference(
            metadata, assignment_matrix, input_mask, anchor_mask, eps
        )

    def _regression_scales(self, metadata, mask):
        batch_size, seq_len, _ = metadata.shape
        valid = _normalize_mask(mask, batch_size, seq_len, metadata.device)
        _validate_prefix_mask(valid, 'point temporal mask')
        scales = metadata[..., METADATA_REGRESSION_SCALE].to(
            dtype=torch.float32
        )
        if torch.any(valid & (scales <= self.eps)):
            raise ValueError('valid temporal metadata has non-positive scale')
        return scales * valid.to(dtype=scales.dtype)

    def forward(self, temporal_metadata, fpn_masks):
        if not (
            len(temporal_metadata) == len(fpn_masks) == self.num_fpn_levels
        ):
            raise ValueError(
                'temporal metadata and FPN masks must match {} levels'.format(
                    self.num_fpn_levels
                )
            )
        points = []
        for level, (metadata, mask) in enumerate(
            zip(temporal_metadata, fpn_masks)
        ):
            if (
                metadata.ndim != 3
                or metadata.size(-1) != TEMPORAL_METADATA_DIM
            ):
                raise ValueError('temporal metadata must have shape (B, T, 5)')
            batch_size, seq_len, _ = metadata.shape
            valid = _normalize_mask(mask, batch_size, seq_len, metadata.device)
            _validate_prefix_mask(valid, 'FPN temporal mask')
            geometry = metadata.to(dtype=torch.float32)
            scales = self._regression_scales(geometry, valid)
            point = torch.zeros(
                (batch_size, seq_len, POINT_DIM),
                dtype=torch.float32,
                device=metadata.device,
            )
            point[..., POINT_CENTER] = geometry[..., METADATA_CENTER]
            if level == 0:
                regression_min = torch.zeros_like(scales)
            else:
                regression_min = (
                    0.5
                    * self.base_regression_range
                    * self.sigma
                    * scales
                )
            regression_max = self.base_regression_range * scales
            if level == self.num_fpn_levels - 1:
                max_distance = torch.full_like(
                    regression_max,
                    (self.max_seq_len + 1.0) * self.input_stride,
                )
                regression_max = torch.maximum(
                    regression_max, max_distance
                )
            point[..., POINT_REGRESSION_MIN] = regression_min
            point[..., POINT_REGRESSION_MAX] = regression_max
            point[..., POINT_REGRESSION_SCALE] = scales
            point[..., POINT_START] = geometry[..., METADATA_START]
            point[..., POINT_END] = geometry[..., METADATA_END]
            point[..., POINT_GEOMETRY_CENTER] = geometry[..., METADATA_CENTER]
            point[..., POINT_SPAN] = geometry[..., METADATA_SPAN]
            points.append(point * valid.unsqueeze(-1).to(point.dtype))
        return tuple(points)

import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Tuple


class QueryBoundaryImportanceLoss(nn.Module):
    """Supervise query relevance and temporal boundaries on every FPN level."""

    def __init__(self, opt: dict):
        super().__init__()
        self.relevance_weight = float(opt.get('relevance_weight', 1.0))
        self.boundary_weight = float(opt.get('boundary_weight', 1.0))
        self.change_weight = float(opt.get('change_weight', 0.25))
        self.final_weight = float(opt.get('final_weight', 0.5))
        self.boundary_sigma = float(opt.get('boundary_sigma', 1.5))
        self.max_pos_weight = float(opt.get('max_pos_weight', 20.0))
        self.importance_alpha = float(opt.get('importance_alpha', 0.4))
        self.importance_beta = float(opt.get('importance_beta', 0.4))
        self.importance_gamma = float(opt.get('importance_gamma', 0.2))
        self.eps = float(opt.get('eps', 1e-6))

        component_weights = (
            self.relevance_weight,
            self.boundary_weight,
            self.change_weight,
            self.final_weight,
        )
        if min(component_weights) < 0:
            raise ValueError('importance loss weights must be non-negative')
        if sum(component_weights) <= 0:
            raise ValueError(
                'at least one importance loss weight must be positive'
            )
        if self.boundary_sigma <= 0:
            raise ValueError('boundary_sigma must be positive')
        if self.max_pos_weight < 1:
            raise ValueError('max_pos_weight must be at least 1')
        if min(
            self.importance_alpha,
            self.importance_beta,
            self.importance_gamma,
        ) < 0:
            raise ValueError('importance mixture weights must be non-negative')
        if (
            self.importance_alpha
            + self.importance_beta
            + self.importance_gamma
        ) <= 0:
            raise ValueError(
                'at least one importance mixture weight must be positive'
            )
        if self.eps <= 0:
            raise ValueError('eps must be positive')

    @staticmethod
    def _normalize_mask(mask: Tensor, batch_size: int, seq_len: int) -> Tensor:
        if mask.ndim == 3:
            if mask.size(1) != 1:
                raise ValueError('mask must have shape (B, 1, T) or (B, T)')
            mask = mask[:, 0]
        elif mask.ndim != 2:
            raise ValueError('mask must have shape (B, 1, T) or (B, T)')
        if mask.shape != (batch_size, seq_len):
            raise ValueError(
                'mask shape {} does not match score shape ({}, {})'.format(
                    tuple(mask.shape), batch_size, seq_len
                )
            )
        return mask.to(dtype=torch.bool)

    def build_targets(
        self,
        points: Tensor,
        mask: Tensor,
        targets: Tensor,
        dtype=None,
        device=None,
    ):
        """Build soft supervision maps with shape (B, T)."""
        if targets.ndim != 2 or targets.size(-1) != 2:
            raise ValueError('targets must have shape (B, 2)')
        batch_size = targets.size(0)
        seq_len = mask.size(-1)
        valid = self._normalize_mask(mask, batch_size, seq_len)
        if points.ndim not in (2, 3) or points.size(-1) < 4:
            raise ValueError('points must have shape (T, D) or (B, T, D)')
        if points.size(-2) < seq_len:
            raise ValueError(
                'points length {} is shorter than score length {}'.format(
                    points.size(-2), seq_len
                )
            )

        if device is None:
            device = targets.device
        if dtype is None:
            dtype = targets.dtype
        valid = valid.to(device=device)
        valid_float = valid.to(dtype=dtype)
        points = points[..., :seq_len, :].to(device=device, dtype=dtype)
        targets = targets.to(device=device, dtype=dtype)

        if points.ndim == 2:
            centers = points[:, 0][None, :]
            strides = points[:, 3].abs().clamp(min=self.eps)[None, :]
        else:
            if points.size(0) != batch_size:
                raise ValueError('batched points must match target batch')
            centers = points[:, :, 0]
            strides = points[:, :, 3].abs().clamp(min=self.eps)
        starts = torch.minimum(targets[:, 0], targets[:, 1])[:, None]
        ends = torch.maximum(targets[:, 0], targets[:, 1])[:, None]

        cell_start = centers - 0.5 * strides
        cell_end = centers + 0.5 * strides
        overlap = (
            torch.minimum(cell_end, ends)
            - torch.maximum(cell_start, starts)
        ).clamp(min=0)
        relevance = (overlap / strides).clamp(min=0, max=1) * valid_float

        start_distance = (centers - starts) / strides
        end_distance = (centers - ends) / strides
        sigma_sq = self.boundary_sigma ** 2
        start = (
            torch.exp(-0.5 * start_distance.square() / sigma_sq)
            * valid_float
        )
        end = (
            torch.exp(-0.5 * end_distance.square() / sigma_sq)
            * valid_float
        )
        boundary = torch.maximum(start, end)

        change = torch.zeros_like(relevance)
        if seq_len > 1:
            pair_valid = valid[:, 1:] & valid[:, :-1]
            change[:, 1:] = (
                relevance[:, 1:] - relevance[:, :-1]
            ).abs() * pair_valid.to(dtype=dtype)
        max_change = change.amax(dim=1, keepdim=True).clamp(min=self.eps)
        change = (
            (change / max_change).clamp(min=0, max=1) * valid_float
        )

        importance = (
            self.importance_alpha * relevance
            + self.importance_beta * boundary
            + self.importance_gamma * change
        ) * valid_float

        return {
            'relevance': relevance * valid_float,
            'start': start * valid_float,
            'end': end * valid_float,
            'boundary': boundary * valid_float,
            'temporal_change': change * valid_float,
            'importance': importance * valid_float,
            'valid': valid,
        }

    def _target_weights(self, target: Tensor, valid: Tensor) -> Tensor:
        valid_float = valid.to(dtype=target.dtype)
        positive_mass = (target * valid_float).sum()
        negative_mass = ((1.0 - target) * valid_float).sum()
        positive_weight = (
            negative_mass / positive_mass.clamp(min=self.eps)
        ).clamp(min=1.0, max=self.max_pos_weight)
        return (
            target * positive_weight + (1.0 - target)
        ) * valid_float

    def _balanced_bce(
        self, prediction: Tensor, target: Tensor, valid: Tensor
    ) -> Tensor:
        prediction = prediction.float().clamp(
            min=self.eps, max=1.0 - self.eps
        )
        target = target.float()
        weights = self._target_weights(target, valid)
        loss = F.binary_cross_entropy(
            prediction, target, reduction='none'
        )
        return (loss * weights).sum() / weights.sum().clamp(min=self.eps)

    def _balanced_smooth_l1(
        self,
        prediction: Tensor,
        target: Tensor,
        valid: Tensor,
        balance_target: Tensor = None,
    ) -> Tensor:
        prediction = prediction.float()
        target = target.float()
        if balance_target is None:
            balance_target = target.clamp(min=0, max=1)
        else:
            balance_target = balance_target.float()
        weights = self._target_weights(balance_target, valid)
        loss = F.smooth_l1_loss(prediction, target, reduction='none')
        return (loss * weights).sum() / weights.sum().clamp(min=self.eps)

    def forward(
        self,
        importance_debug: Tuple[dict],
        fpn_points: Tuple[Tensor],
        fpn_masks: Tuple[Tensor],
        targets: Tensor,
        return_components: bool = False,
    ):
        if importance_debug is None:
            raise ValueError('importance debug outputs are required')
        if len(importance_debug) == 0:
            raise ValueError('importance debug outputs cannot be empty')
        if not (
            len(importance_debug) == len(fpn_points) == len(fpn_masks)
        ):
            raise ValueError(
                'importance, points, and masks must have the same levels'
            )

        reference = importance_debug[0]['importance']
        components = {
            'relevance': reference.float().sum() * 0.0,
            'boundary': reference.float().sum() * 0.0,
            'temporal_change': reference.float().sum() * 0.0,
            'importance': reference.float().sum() * 0.0,
        }
        required_scores = (
            'relevance',
            'start',
            'end',
            'boundary',
            'temporal_change',
            'importance',
        )

        for debug, points, mask in zip(
            importance_debug, fpn_points, fpn_masks
        ):
            if debug is None:
                raise ValueError('importance debug output cannot be None')
            missing = [key for key in required_scores if key not in debug]
            if missing:
                raise ValueError(
                    'importance debug output is missing {}'.format(missing)
                )
            relevance = debug['relevance']
            if relevance.ndim != 2:
                raise ValueError('importance scores must have shape (B, T)')
            batch_size, seq_len = relevance.shape
            if targets.shape != (batch_size, 2):
                raise ValueError(
                    'targets shape {} does not match score batch {}'.format(
                        tuple(targets.shape), batch_size
                    )
                )
            valid = self._normalize_mask(mask, batch_size, seq_len)
            valid = valid.to(device=relevance.device)
            level_targets = self.build_targets(
                points,
                valid,
                targets,
                dtype=torch.float32,
                device=relevance.device,
            )
            for key in required_scores:
                if debug[key].shape != (batch_size, seq_len):
                    raise ValueError(
                        '{} score shape {} does not match ({}, {})'.format(
                            key,
                            tuple(debug[key].shape),
                            batch_size,
                            seq_len,
                        )
                    )

            components['relevance'] = components['relevance'] + (
                self._balanced_bce(
                    debug['relevance'], level_targets['relevance'], valid
                )
            )
            start_loss = self._balanced_bce(
                debug['start'], level_targets['start'], valid
            )
            end_loss = self._balanced_bce(
                debug['end'], level_targets['end'], valid
            )
            components['boundary'] = components['boundary'] + 0.5 * (
                start_loss + end_loss
            )
            components['temporal_change'] = (
                components['temporal_change']
                + self._balanced_smooth_l1(
                    debug['temporal_change'],
                    level_targets['temporal_change'],
                    valid,
                )
            )
            components['importance'] = components['importance'] + (
                self._balanced_smooth_l1(
                    debug['importance'],
                    level_targets['importance'],
                    valid,
                    balance_target=level_targets['boundary'],
                )
            )

        num_levels = float(len(importance_debug))
        components = {
            key: value / num_levels for key, value in components.items()
        }
        total = (
            self.relevance_weight * components['relevance']
            + self.boundary_weight * components['boundary']
            + self.change_weight * components['temporal_change']
            + self.final_weight * components['importance']
        )
        if return_components:
            return total, components
        return total


class BoundarySupervisionLoss(nn.Module):
    """Supervise start/end boundary probabilities on every FPN level."""

    def __init__(self, opt=None):
        super().__init__()
        opt = {} if opt is None else opt
        self.boundary_sigma = float(opt.get('boundary_sigma', 1.5))
        self.max_pos_weight = float(opt.get('max_pos_weight', 20.0))
        self.eps = float(opt.get('eps', 1e-6))
        if (
            not math.isfinite(self.boundary_sigma)
            or self.boundary_sigma <= 0
        ):
            raise ValueError('boundary_sigma must be finite and positive')
        if (
            not math.isfinite(self.max_pos_weight)
            or self.max_pos_weight < 1
        ):
            raise ValueError('max_pos_weight must be finite and at least 1')
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError('eps must be finite and positive')

    @staticmethod
    def _normalize_mask(mask, batch_size, seq_len):
        if mask.ndim == 3:
            if mask.size(1) != 1:
                raise ValueError(
                    'mask must have shape (B, 1, T) or (B, T)'
                )
            mask = mask[:, 0]
        elif mask.ndim != 2:
            raise ValueError(
                'mask must have shape (B, 1, T) or (B, T)'
            )
        if mask.shape != (batch_size, seq_len):
            raise ValueError(
                'mask shape {} does not match ({}, {})'.format(
                    tuple(mask.shape), batch_size, seq_len
                )
            )
        return mask.to(dtype=torch.bool)

    def _point_centers_and_strides(
        self, points, batch_size, seq_len, dtype, device
    ):
        if points.ndim == 2:
            if points.size(1) < 4 or points.size(0) < seq_len:
                raise ValueError(
                    'points must have shape (T, 4) with T >= score length'
                )
            points = points[:seq_len].to(device=device, dtype=dtype)
            centers = points[:, 0].unsqueeze(0).expand(batch_size, -1)
            strides = points[:, 3].abs().unsqueeze(0).expand(
                batch_size, -1
            )
        elif points.ndim == 3:
            if (
                points.size(0) != batch_size
                or points.size(2) < 4
                or points.size(1) < seq_len
            ):
                raise ValueError(
                    'batched points must have shape (B, T, 4) with '
                    'matching B and T >= score length'
                )
            points = points[:, :seq_len].to(device=device, dtype=dtype)
            centers = points[:, :, 0]
            strides = points[:, :, 3].abs()
        else:
            raise ValueError(
                'points must have shape (T, 4) or (B, T, 4)'
            )
        return centers, strides.clamp(min=self.eps)

    def build_targets(
        self, points, mask, targets, dtype=None, device=None
    ):
        """Build stride-normalized Gaussian start/end maps."""
        if targets.ndim != 2 or targets.size(-1) != 2:
            raise ValueError('targets must have shape (B, 2)')
        batch_size = targets.size(0)
        seq_len = mask.size(-1)
        valid = self._normalize_mask(mask, batch_size, seq_len)
        if device is None:
            device = targets.device
        if dtype is None:
            dtype = targets.dtype
        valid = valid.to(device=device)
        valid_float = valid.to(dtype=dtype)
        targets = targets.to(device=device, dtype=dtype)
        centers, strides = self._point_centers_and_strides(
            points, batch_size, seq_len, dtype, device
        )
        starts = torch.minimum(targets[:, 0], targets[:, 1]).unsqueeze(1)
        ends = torch.maximum(targets[:, 0], targets[:, 1]).unsqueeze(1)
        sigma_sq = self.boundary_sigma ** 2
        start_distance = (centers - starts) / strides
        end_distance = (centers - ends) / strides
        start_target = torch.exp(
            -0.5 * start_distance.square() / sigma_sq
        ) * valid_float
        end_target = torch.exp(
            -0.5 * end_distance.square() / sigma_sq
        ) * valid_float
        return {
            'start_target': start_target * valid_float,
            'end_target': end_target * valid_float,
            'valid': valid,
        }

    def _target_weights(self, target, valid):
        valid_float = valid.to(dtype=target.dtype)
        positive_mass = (target * valid_float).sum()
        negative_mass = ((1.0 - target) * valid_float).sum()
        positive_weight = (
            negative_mass / positive_mass.clamp(min=self.eps)
        ).clamp(min=1.0, max=self.max_pos_weight)
        return (
            target * positive_weight + (1.0 - target)
        ) * valid_float

    def _balanced_bce(self, prediction, target, valid):
        prediction = prediction.float().clamp(
            min=self.eps, max=1.0 - self.eps
        )
        target = target.float()
        weights = self._target_weights(target, valid)
        loss = F.binary_cross_entropy(
            prediction, target, reduction='none'
        )
        return (loss * weights).sum() / weights.sum().clamp(min=self.eps)

    @staticmethod
    def _get_prediction(debug, name, legacy_name):
        prediction = debug.get(name)
        if prediction is None:
            prediction = debug.get(legacy_name)
        if prediction is None:
            raise ValueError(
                'boundary debug output is missing {}'.format(name)
            )
        return prediction

    def forward(
        self,
        boundary_debug,
        fpn_points,
        fpn_masks,
        targets,
        return_components=False,
    ):
        if boundary_debug is None or len(boundary_debug) == 0:
            raise ValueError('boundary debug outputs cannot be empty')
        if not (
            len(boundary_debug) == len(fpn_points) == len(fpn_masks)
        ):
            raise ValueError(
                'boundary debug, points, and masks must have the same levels'
            )

        reference = self._get_prediction(
            boundary_debug[0], 'start_prob', 'start'
        )
        if reference.ndim != 2:
            raise ValueError('boundary probabilities must have shape (B, T)')
        zero = reference.float().sum() * 0.0
        components = {'start': zero, 'end': zero}

        for debug, points, mask in zip(
            boundary_debug, fpn_points, fpn_masks
        ):
            start_prob = self._get_prediction(
                debug, 'start_prob', 'start'
            )
            end_prob = self._get_prediction(debug, 'end_prob', 'end')
            if start_prob.ndim != 2:
                raise ValueError(
                    'boundary probabilities must have shape (B, T)'
                )
            batch_size, seq_len = start_prob.shape
            if end_prob.shape != (batch_size, seq_len):
                raise ValueError('start/end probability shapes must match')
            if targets.shape != (batch_size, 2):
                raise ValueError(
                    'targets shape {} does not match batch {}'.format(
                        tuple(targets.shape), batch_size
                    )
                )
            valid = self._normalize_mask(mask, batch_size, seq_len)
            valid = valid.to(device=start_prob.device)
            level_targets = self.build_targets(
                points,
                valid,
                targets,
                dtype=torch.float32,
                device=start_prob.device,
            )
            components['start'] = components['start'] + self._balanced_bce(
                start_prob,
                level_targets['start_target'],
                valid,
            )
            components['end'] = components['end'] + self._balanced_bce(
                end_prob,
                level_targets['end_target'],
                valid,
            )

        num_levels = float(len(boundary_debug))
        components = {
            key: value / num_levels for key, value in components.items()
        }
        components['boundary'] = (
            components['start'] + components['end']
        )
        if return_components:
            return components['boundary'], components
        return components['boundary']


def make_boundary_supervision_loss(opt=None):
    """Return (weight, loss) while making a zero weight a true no-op."""
    opt = {} if opt is None else opt
    weight = float(opt.get('boundary_loss_weight', 0.0))
    if not math.isfinite(weight) or weight < 0:
        raise ValueError(
            'boundary_loss_weight must be finite and non-negative'
        )
    if weight == 0.0:
        return weight, None
    return weight, BoundarySupervisionLoss(opt)

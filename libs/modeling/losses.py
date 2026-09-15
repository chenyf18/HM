import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Tuple, Callable, Optional

from libs.dist_utils import print0
from .temporal_coordinates import decode_offsets
from .contrastive_losses import (
    assignment_aware_anchor_contrastive,
    contrastive_subsample_negative_mp,
)

class CrossLevelQualityConsistency(nn.Module):
    """Regularize overlapping FPN positives with quality-aware consensus."""

    def __init__(self, opt: dict):
        super().__init__()
        self.boundary_weight = float(opt.get('boundary_weight', 1.0))
        self.rank_weight = float(opt.get('rank_weight', 0.25))
        self.quality_floor = float(opt.get('quality_floor', 0.05))
        self.score_floor = float(opt.get('score_floor', 0.05))
        self.score_power = float(opt.get('score_power', 1.0))
        self.quality_power = float(opt.get('quality_power', 1.0))
        self.rank_margin = float(opt.get('rank_margin', 0.05))
        self.min_levels = int(opt.get('min_levels', 2))
        self.max_level_gap = int(opt.get('max_level_gap', 1))
        self.boundary_norm = float(opt.get('boundary_norm', 4.0))
        print0(
            'Using cross-level quality consistency: '
            f'boundary_weight={self.boundary_weight}, '
            f'rank_weight={self.rank_weight}, '
            f'max_level_gap={self.max_level_gap}'
        )

    def forward(
        self,
        fpn_points: Tuple[Tensor],
        fpn_logits: Tuple[Tensor],
        fpn_offsets: Tuple[Tensor],
        fpn_masks: Tuple[Tensor],
        gt_labels: Tuple[Tensor],
        targets: Tensor,
    ) -> Tensor:
        """Align decoded boundaries and relative quality across adjacent levels."""
        prototypes = []
        level_quality = []
        level_scores = []
        level_valid = []

        target_segments = targets[:, None, :]
        for points, logits, offsets, masks, labels in zip(
            fpn_points, fpn_logits, fpn_offsets, fpn_masks, gt_labels
        ):
            positive = torch.logical_and(labels, masks)
            if points.ndim == 3:
                # Adaptive points are query-specific: (B, T, D).
                stride = points[..., 3].to(offsets.dtype)
                center = points[..., 0].to(offsets.dtype)
            else:
                stride = points[:, 3].to(offsets.dtype)[None, :]
                center = points[:, 0].to(offsets.dtype)[None, :]
            decoded = decode_offsets(points, offsets)

            intersection = (
                torch.minimum(decoded[..., 1], target_segments[..., 1])
                - torch.maximum(decoded[..., 0], target_segments[..., 0])
            ).clamp(min=0)
            union = (
                (decoded[..., 1] - decoded[..., 0])
                + (target_segments[..., 1] - target_segments[..., 0])
                - intersection
            )
            quality = (
                intersection / union.clamp(min=1e-6)
            ).detach().clamp(min=0, max=1)

            confidence = torch.sigmoid(logits).detach().clamp(
                min=self.score_floor, max=1.0
            )
            point_weights = positive.to(offsets.dtype)
            point_weights = point_weights * confidence.pow(self.score_power)
            point_weights = point_weights * (
                quality + self.quality_floor
            ).pow(self.quality_power)
            weight_sum = point_weights.sum(dim=1)
            valid = weight_sum > 0

            prototype = (
                decoded * point_weights[..., None]
            ).sum(dim=1) / weight_sum.clamp(min=1e-6)[..., None]
            mean_quality = (
                quality * point_weights
            ).sum(dim=1) / weight_sum.clamp(min=1e-6)
            mean_score = (
                logits * point_weights
            ).sum(dim=1) / weight_sum.clamp(min=1e-6)

            prototypes.append(prototype)
            level_quality.append(mean_quality)
            level_scores.append(mean_score)
            level_valid.append(valid)

        prototypes = torch.stack(prototypes, dim=1)
        level_quality = torch.stack(level_quality, dim=1).detach()
        level_scores = torch.stack(level_scores, dim=1)
        level_valid = torch.stack(level_valid, dim=1)
        enough_levels = level_valid.sum(dim=1) >= self.min_levels

        consensus_weights = level_valid.to(prototypes.dtype) * (
            level_quality + self.quality_floor
        ).pow(self.quality_power)
        consensus = (
            prototypes * consensus_weights[..., None]
        ).sum(dim=1) / consensus_weights.sum(dim=1).clamp(min=1e-6)[..., None]

        target_length = (
            targets[:, 1] - targets[:, 0]
        ).abs().clamp(min=self.boundary_norm)
        boundary_delta = (
            prototypes - consensus.detach()[:, None, :]
        ) / target_length[:, None, None]
        boundary_per_level = F.smooth_l1_loss(
            boundary_delta,
            torch.zeros_like(boundary_delta),
            reduction='none',
        ).mean(dim=-1)
        boundary_mask = level_valid & enough_levels[:, None]
        boundary_weights = (
            consensus_weights.detach() * boundary_mask.to(prototypes.dtype)
        )
        boundary_loss = (
            boundary_per_level * boundary_weights
        ).sum() / boundary_weights.sum().clamp(min=1e-6)

        rank_loss_sum = level_scores.sum() * 0.0
        rank_weight_sum = level_scores.sum().detach() * 0.0
        num_levels = len(fpn_points)
        for left_level in range(num_levels):
            right_end = min(
                num_levels, left_level + self.max_level_gap + 1
            )
            for right_level in range(left_level + 1, right_end):
                pair_mask = (
                    level_valid[:, left_level]
                    & level_valid[:, right_level]
                    & enough_levels
                )
                quality_delta = (
                    level_quality[:, left_level]
                    - level_quality[:, right_level]
                )
                informative = pair_mask & (
                    quality_delta.abs() >= self.rank_margin
                )
                direction = quality_delta.sign()
                score_delta = (
                    level_scores[:, left_level]
                    - level_scores[:, right_level]
                )
                pair_loss = F.softplus(-direction * score_delta)
                pair_weights = (
                    quality_delta.abs() * informative.to(level_scores.dtype)
                )
                rank_loss_sum = rank_loss_sum + (
                    pair_loss * pair_weights
                ).sum()
                rank_weight_sum = rank_weight_sum + pair_weights.sum()

        rank_loss = (
            rank_loss_sum / rank_weight_sum.clamp(min=1e-6)
        )
        return (
            self.boundary_weight * boundary_loss
            + self.rank_weight * rank_loss
        )



def build_single_level_loss(loss_type: str) -> Callable:
    if loss_type == 'contr_mp':
        return lambda *args, **kw: contrastive_subsample_negative_mp(*args, **kw)
    raise ValueError(f"Unsupported released contrastive type: {loss_type}")


class MultiScaleMaskedContrastive(nn.Module):
    def __init__(self, opt: dict, vid_embd_dim: int):
        super().__init__()
        self.acc_mode = opt.get('acc_mode', 'legacy_acc')
        if self.acc_mode not in (
            'legacy_acc', 'assignment_acc',
            'legacy_consistent_assignment_acc',
        ):
            raise ValueError(
                'acc_mode must be legacy_acc, assignment_acc, or '
                'legacy_consistent_assignment_acc, got '
                f'{self.acc_mode}'
            )
        contrastive_type = opt.get('contr_type', 'contr_mp')
        self.loss_fn = (
            build_single_level_loss(contrastive_type)
            if self.acc_mode == 'legacy_acc'
            else assignment_aware_anchor_contrastive
        )
        # Old assignment ACC kept for regression/audit only: it changes the
        # negative pool and is ~33x larger than legacy under uniform grouping.
        # Marked research-buggy per HM-AUDIT-002.
        if self.acc_mode == 'legacy_consistent_assignment_acc':
            from .contrastive_losses import legacy_consistent_assignment_acc
            self.loss_fn = legacy_consistent_assignment_acc
        self.temp = opt.get('temperature', 0.07)
        self.neg_ratio = opt.get('neg_ratio', 0.20)
        self.gap_ratio = opt.get('gap_ratio', 0.30)
        self.radius = opt.get('radius', 0)
        self.hard_neg = opt.get('hard_neg', False)
        self.cross_video_neg = opt.get('cross_video_neg', False)
        proj_outdim = opt.get('proj_outdim', 256)
        proj_expand = opt.get('proj_expand', 1.0)
        proj_num_layers = opt.get('proj_num_layers', 2)
        self.projector = LNProjector(
            in_dim=vid_embd_dim,
            out_dim=proj_outdim,
            expand=proj_expand,
            num_layers=proj_num_layers,
        )
        print0(
            f'Using ACC mode: {self.acc_mode}, '
            f'contrastive type: {contrastive_type}, '
            f'temp: {self.temp}, neg_ratio: {self.neg_ratio}, gap_ratio: {self.gap_ratio}, '
            f'radius: {self.radius}, hard_neg: {self.hard_neg}, '
            f'cross_video_neg: {self.cross_video_neg}, weight: {opt.get("weight", 1.0)}'
        )

    def forward(
        self,
        sequence_fpn: Tuple[Tensor],
        sequence_fpn_mask: Tuple[Tensor],
        anchor_fpn: Tuple[Tensor],
        anchor_fpn_mask: Tuple[Tensor],
        assignment_matrices: Optional[Tuple[Tensor]] = None,
    ) -> Tensor:
        num_levels = len(sequence_fpn)
        if not (
            num_levels
            == len(sequence_fpn_mask)
            == len(anchor_fpn)
            == len(anchor_fpn_mask)
        ):
            raise ValueError(
                'sequence and anchor FPN inputs must have the same levels'
            )
        if self.acc_mode == 'assignment_acc':
            if assignment_matrices is None:
                raise ValueError(
                    'assignment_acc requires assignment_matrices'
                )
            if len(assignment_matrices) != num_levels:
                raise ValueError(
                    'assignment_matrices must match the FPN levels'
                )

        total = sequence_fpn[0].float().sum() * 0.0
        uses_assignment = self.acc_mode in (
            'assignment_acc', 'legacy_consistent_assignment_acc'
        )
        for level, (seq, seq_m, anc, anc_m) in enumerate(zip(
            sequence_fpn,
            sequence_fpn_mask,
            anchor_fpn,
            anchor_fpn_mask,
        )):
            common_kwargs = {
                'anchors': anc,
                'seq_tokens': seq,
                'anchor_mask': anc_m,
                'seq_mask': seq_m,
                'projector': self.projector,
                'temperature': self.temp,
            }
            if uses_assignment:
                if assignment_matrices is None:
                    raise ValueError(
                        f'{self.acc_mode} requires assignment_matrices'
                    )
                total = total + self.loss_fn(
                    assignment_matrix=assignment_matrices[level],
                    **common_kwargs,
                )
                continue
            else:
                total = total + self.loss_fn(
                    neg_ratio=self.neg_ratio,
                    gap_ratio=self.gap_ratio,
                    radius=self.radius,
                    hard_neg=self.hard_neg,
                    cross_video_neg=self.cross_video_neg,
                    **common_kwargs,
                )
        return total


class MultiScaleMaskedGTPointContrastive(nn.Module):
    def __init__(self, opt: dict, vid_embd_dim: int):
        super().__init__()
        contrastive_type = opt.get('contr_type', 'point_gt_contr_pooled')
        if contrastive_type != 'point_gt_contr_pooled':
            raise ValueError(
                'Unsupported released GT contrastive type: '
                f'{contrastive_type}. The code release only supports point_gt_contr_pooled.'
            )

        self.temp = opt.get('temperature', 0.07)
        self.neg_ratio = opt.get('neg_ratio', 1.0)
        self.use_projector = opt.get('use_projector', True)
        if self.use_projector:
            proj_outdim = opt.get('proj_outdim', 256)
            proj_expand = opt.get('proj_expand', 1.0)
            proj_num_layers = opt.get('proj_num_layers', 2)
            self.projector = LNProjector(
                in_dim=vid_embd_dim,
                out_dim=proj_outdim,
                expand=proj_expand,
                num_layers=proj_num_layers,
            )
        else:
            self.projector = None
            print0('No projector used')

        print0(
            f'Using GT Contrastive Loss of type: {contrastive_type}, '
            f'temp: {self.temp}, neg_ratio: {self.neg_ratio}, '
            f'span_contr_gt: {opt.get("span_contr_gt", False)}, weight: {opt.get("weight", 1.0)}'
        )

    def forward(
        self,
        fpn_fused: Tuple[Tensor],
        fpn_fused_masks: Tuple[Tensor],
        gt_labels: Tuple[Tensor],
        gt_labels_span_list: Tuple[Tensor],
    ) -> Tensor:
        total = torch.tensor(0.0, device=fpn_fused[0].device)
        for fpn, fpn_mask, gt_label, gt_labels_span in zip(
            fpn_fused, fpn_fused_masks, gt_labels, gt_labels_span_list
        ):
            total = total + self._compute_infonce_loss_pooled(
                fpn, fpn_mask, gt_label, gt_labels_span
            )
        return total

    def _compute_infonce_loss_pooled(
        self,
        fpn: Tensor,
        fpn_mask: Tensor,
        gt_label: Tensor,
        gt_labels_span: Tensor,
    ) -> Tensor:
        device = fpn.device
        total_loss = torch.zeros(1, device=device)
        total_anchors = 0

        for batch_idx in range(fpn.shape[0]):
            valid_idx = fpn_mask[batch_idx].nonzero(as_tuple=True)[0]
            if valid_idx.numel() == 0:
                continue

            if self.use_projector and self.projector is not None:
                proj = self.projector(fpn[batch_idx, :, valid_idx].T).T
            else:
                proj = F.normalize(fpn[batch_idx, :, valid_idx], p=2, dim=0)
            proj = F.normalize(proj, p=2, dim=0)

            labels = gt_label[batch_idx, valid_idx]
            pos_idx = labels.nonzero(as_tuple=True)[0]
            neg_idx = torch.logical_and(
                ~labels,
                ~gt_labels_span[batch_idx, valid_idx],
            ).nonzero(as_tuple=True)[0]

            if pos_idx.numel() == 0 or neg_idx.numel() == 0:
                continue

            k_neg = int(pos_idx.numel() * self.neg_ratio)
            if neg_idx.numel() > k_neg:
                sel = torch.randperm(neg_idx.numel(), device=device)[:k_neg]
                neg_idx = neg_idx[sel]
            if neg_idx.numel() == 0:
                continue

            anchor = proj[:, pos_idx].mean(dim=1, keepdim=True)
            anchor = F.normalize(anchor, p=2, dim=0)

            sim_pos = (anchor.T @ proj[:, pos_idx]) / self.temp
            sim_neg = (anchor.T @ proj[:, neg_idx]) / self.temp
            log_pos = torch.logsumexp(sim_pos, dim=1)
            log_all = torch.logsumexp(torch.cat([sim_pos, sim_neg], dim=1), dim=1)
            total_loss += -(log_pos - log_all)
            total_anchors += 1

        if total_anchors == 0:
            return torch.tensor(0.0, device=device)
        return (total_loss / total_anchors).squeeze()


class LNProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256, expand: float = 2.0, num_layers: int = 2):
        super().__init__()
        if num_layers not in [1, 2, 3]:
            raise ValueError(f'num_layers must be 1, 2, or 3, got {num_layers}')

        self.num_layers = num_layers
        hid_dim = int(expand * in_dim)

        if num_layers == 1:
            self.fc = nn.Linear(in_dim, out_dim)
        elif num_layers == 2:
            self.fc1 = nn.Linear(in_dim, hid_dim, bias=False)
            self.ln1 = nn.LayerNorm(hid_dim)
            self.act1 = nn.GELU()
            self.fc2 = nn.Linear(hid_dim, out_dim)
            nn.init.ones_(self.ln1.weight)
            nn.init.zeros_(self.ln1.bias)
        else:
            self.fc1 = nn.Linear(in_dim, hid_dim, bias=False)
            self.ln1 = nn.LayerNorm(hid_dim)
            self.act1 = nn.GELU()
            self.fc2 = nn.Linear(hid_dim, hid_dim, bias=False)
            self.ln2 = nn.LayerNorm(hid_dim)
            self.act2 = nn.GELU()
            self.fc3 = nn.Linear(hid_dim, out_dim)
            nn.init.ones_(self.ln1.weight)
            nn.init.zeros_(self.ln1.bias)
            nn.init.ones_(self.ln2.weight)
            nn.init.zeros_(self.ln2.bias)

    def forward(self, x):
        if self.num_layers == 1:
            z = self.fc(x)
        elif self.num_layers == 2:
            z = self.fc2(self.act1(self.ln1(self.fc1(x))))
        else:
            h1 = self.act1(self.ln1(self.fc1(x)))
            h2 = self.act2(self.ln2(self.fc2(h1)))
            z = self.fc3(h2)
        return F.normalize(z, p=2, dim=1)

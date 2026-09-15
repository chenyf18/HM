from collections import OrderedDict
from copy import deepcopy
import math
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from .data import make_dataset, make_dataloader
from .dist_utils import get_rank, get_world_size, barrier, all_gather, print0
from .modeling import (
    PtGenerator, sigmoid_focal_loss, ctr_giou_loss, ctr_diou_loss,
    make_optimizer, make_scheduler, MultiScaleMaskedContrastive, MultiScaleMaskedGTPointContrastive,
    CrossLevelQualityConsistency,
)
from .nms import batched_nms
from .modeling.temporal_coordinates import AdaptiveTemporalPointGenerator, decode_offsets, encode_offsets
from .modeling.query_boundary_importance_loss import (
    QueryBoundaryImportanceLoss, make_boundary_supervision_loss,
)
from .modeling.allocation_supervision import (
    AllocationSupervisionLoss, RankingAllocationLoss,
)
from .modeling.rank_head import MomentRankHead
from .modeling.denoiser import MomentDenoiser
from .train_utils import Logger, AverageMeter, fix_random_seed, iou, time_str, generate_multiscale_gt_masks, generate_multiscale_gt_masks_from_points, generate_multiscale_gt_masks_contrastive

from .modeling.model import make_models_net
from torch.cuda.amp import autocast
import json

AUX_LOSS_REGISTRY = {
    'ds_contrastive': MultiScaleMaskedContrastive,
    'gt_point_contrastive': MultiScaleMaskedGTPointContrastive,
}



class TrainerOriginal:

    def __init__(self, opt):

        self.opt = opt

        # set random seed
        rng = fix_random_seed(opt.get('seed', 20))
        # rng = None
        # print('no seed set')
        # build model and EMA
        # self.model = PtTransformer(opt['model']).cuda()
        self.model = make_models_net(opt).cuda()
        self.listwise_ranking = bool(
            opt['train'].get('loss_aux', {}).get(
                'listwise_ranking', {}
            ).get('enable', False)
        )
        self.span_denoising = bool(
            opt['train'].get('loss_aux', {}).get(
                'span_denoising', {}
            ).get('enable', False)
        )
        self.model_ema = deepcopy(self.model).eval().requires_grad_(False)
        self.adaptive_anchor = bool(opt['model']['vid_net'].get('adaptive_anchor', False))
        self.pt_gen = PtGenerator(
            **opt['pt_gen'], allow_fixed_stride=not self.adaptive_anchor
        ).cuda()
        if self.adaptive_anchor:
            self.adaptive_pt_gen = AdaptiveTemporalPointGenerator(
                max_seq_len=opt['pt_gen']['max_seq_len'],
                num_fpn_levels=opt['pt_gen']['num_fpn_levels'],
                regression_range=opt['pt_gen'].get('regression_range', 4),
                sigma=opt['pt_gen'].get('sigma', 1),
                input_stride=opt['model']['vid_net'].get('stride', 1),
            ).cuda()
        else:
            self.adaptive_pt_gen = None
        self.ema_beta = opt['train'].get('ema_beta', 0.999)

        # prepare dataset
        self.num_epochs = opt['train']['epochs'] + opt['train']['warmup_epochs']
        self.dataset = make_dataset(
            opt['train']['data'], num_epochs=self.num_epochs, is_training=True
        )
        self.batch_size = batch_size = opt['train']['batch_size']
        self.dataloader, self.sampler = make_dataloader(
            self.dataset, generator=rng, is_training=True,
            batch_size=batch_size, num_workers=opt['train']['num_workers'],
            world_size=get_world_size(), rank=get_rank()
        )
        self.microbatch_size = opt['train'].get('microbatch_size', batch_size)
        self.num_microbatches = batch_size // self.microbatch_size
        assert batch_size % self.microbatch_size == 0

        # build training utilities
        self.itrs_per_epoch = opt['train']['scheduler']['itrs_per_epoch'] = len(self.dataloader)
        self.num_itrs = self.num_epochs * self.itrs_per_epoch
        self.epoch = self.itr = 0
        self.optimizer = make_optimizer(self.model, opt['train']['optimizer'])
        self.scheduler = make_scheduler(self.optimizer, opt['train']['scheduler'])
        self.clip_grad_norm = opt['train'].get('clip_grad_norm')

        # build logging utilities
        self.log_interval = opt['log'].get('log_interval', 100)
        self.checkpoint_epochs = opt['log'].get('checkpoint_epochs', (-1, ))
        if get_rank() == 0:
            self.logger = Logger(os.path.join(opt['_root'], 'log.txt'))
            self.tb_writer = SummaryWriter(os.path.join(opt['_root'], 'tensorboard'))
            self.loss_meters = OrderedDict()
            self.timer = AverageMeter()
        else:
            self.logger = self.tb_writer = self.loss_meters = self.timer = None

        # load model weights and training states
        if opt['_resume']:
            self.load()
            barrier()

        # set up distributed training
        if opt['_distributed']:
            query_conditioned = any(
                bool(opt['model']['vid_net'].get(key, False))
                for key in (
                    'query_modulation',
                    'query_aware_gate',
                    'query_boundary_importance',
                    'adaptive_anchor',
                    'coarse_to_fine_refine',
                )
            )
            self.model = DistributedDataParallel(
                self.model,
                [get_rank()],
                find_unused_parameters=query_conditioned,
            )
            self._ema_init()

        # register model hyperparameters
        self.max_vid_len = opt['model']['max_vid_len']
        self.max_text_len = opt['model']['max_text_len']
        self.vid_stride = opt['model'].get('vid_stride', 1)
        self.input_vid_len = self.max_vid_len * self.vid_stride

        # register annotation hyperparameters
        self.center_sampling = opt['train'].get('center_sampling', 'radius')
        self.center_sampling_radius = opt['train']['center_sampling_radius']

        # register optimization hyperparameters
        self.loss_norm_momentum = opt['train'].get('loss_norm_momentum', 0.9)
        self.loss_norm = opt['train']['loss_norm']
        self.loss_weight = opt['train'].get('loss_weight', 1.0)
        self.reg_loss = opt['train'].get('reg_loss', 'diou')

    def run(self):
        print0("Training started.")
        while self.epoch < self.num_epochs:
            self.dataset.set_epoch(self.epoch)
            if self.opt['_distributed']:
                self.sampler.set_epoch(self.epoch)
            for data_list in self.dataloader:
                # run one optimization step
                start_time = time.time()
                self.optimizer.zero_grad(set_to_none=True)
                loss_dict = self.forward_backward(data_list)
                if self.clip_grad_norm:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_grad_norm
                    )
                self.optimizer.step()
                self.scheduler.step()
                self.itr += 1
                self._ema_update()
                if get_rank() == 0:
                    # only track loss from rank 0 to avoid sync overhead
                    for k, v in loss_dict.items():
                        if k not in self.loss_meters:
                            self.loss_meters[k] = AverageMeter()
                        self.loss_meters[k].update(v.detach())
                    self.timer.update(time.time() - start_time)
                    if self.itr == 1 or self.itr % self.log_interval == 0:
                        self.log()
            self.epoch += 1
            self.checkpoint()
            barrier()
        print0("Training completed.")

    def _current_tefm_lambda(self):
        """HM-TEFM-039: scheduled evidence-loss coefficient."""
        import math
        if self.tefm_schedule == 'off':
            return 0.0
        if self.tefm_schedule == 'constant':
            return self.tefm_lambda
        if self.tefm_schedule == 'hard_off':
            return self.tefm_lambda if self.itr <= self.tefm_decay_start else 0.0
        if self.tefm_schedule == 'cosine_decay':
            if self.itr <= self.tefm_decay_start:
                return self.tefm_lambda
            p = (self.itr - self.tefm_decay_start) / max(
                self.tefm_total_steps - self.tefm_decay_start, 1)
            return self.tefm_lambda * 0.5 * (1 + math.cos(math.pi * p))
        return 0.0

    def forward_backward(self, data_list):
        cls_loss = reg_loss = total_loss = norm = 0
        for i in range(0, self.batch_size, self.microbatch_size):
            loss_dict = self._microbatch_forward_backward(
                data_list[i:i + self.microbatch_size],
                is_last=(i + self.microbatch_size >= self.batch_size)
            )
            cls_loss += loss_dict['cls']
            reg_loss += loss_dict['reg']
            total_loss += loss_dict['total']
            norm += loss_dict['norm']

        # update EMA loss norm
        all_norms = [torch.zeros_like(norm) for _ in range(get_world_size())]
        all_gather(all_norms, norm)
        self.loss_norm = (
            self.loss_norm_momentum * self.loss_norm
            + (1. - self.loss_norm_momentum) * max(sum(all_norms).item(), 1)
        )
        return {'cls': cls_loss, 'reg': reg_loss, 'total': total_loss}

    def _microbatch_forward_backward(self, data_list, is_last=False):
        # batch data
        vid, vid_masks, text, text_masks, text_size = self._batchify(
            vid_list=[d['vid'] for d in data_list], 
            text_list=[d['text'] for d in data_list]
        )
        vid = vid.cuda(non_blocking=True) # (bs, c_v, t)
        vid_masks = vid_masks.cuda(non_blocking=True)
        text = text.cuda(non_blocking=True) # (bs, num_queries, c_t, t)
        text_masks = text_masks.cuda(non_blocking=True)
        text_size = text_size.cuda(non_blocking=True)

        target_divisor = 1 if self.adaptive_anchor else self.vid_stride
        targets = torch.cat([d['target'] / target_divisor for d in data_list]) # (bs * num_queries, 2)
        targets = targets.cuda(non_blocking=True)
        
        # forward pass
        if is_last or not self.opt['_distributed']:
            fpn_logits, fpn_offsets, fpn_masks = \
                self.model(vid, vid_masks, text, text_masks, text_size)
        else:
            with self.model.no_sync():
                fpn_logits, fpn_offsets, fpn_masks = \
                    self.model(vid, vid_masks, text, text_masks, text_size)
        fpn_n_points = [m.size(-1) for m in fpn_masks]
        if self.adaptive_anchor:
            backbone = self.model.module if hasattr(self.model, 'module') else self.model
            fpn_points = backbone.vid_net.last_temporal_metadata
            fpn_points = self.adaptive_pt_gen(fpn_points, fpn_masks)
        else:
            fpn_points = self.pt_gen(fpn_n_points)

        # stitch model outputs
        fpn_logits = torch.cat(fpn_logits, dim=1)   # (bs * num_queries, p)
        fpn_offsets = torch.cat(fpn_offsets, dim=1) # (bs * num_queries, p, 2)
        fpn_masks = torch.cat(fpn_masks, dim=1)     # (bs * num_queries, p)
        points = (torch.cat(fpn_points, dim=1) if self.adaptive_anchor else torch.cat(fpn_points))

        # annotate points
        gt_labels, gt_offsets = self._annotate_points(points, targets)
        # gt_labels, gt_offsets = self._annotate_points_adaptive(points, targets)
        # gt_labels, gt_offsets = self._annotate_points_improved(points, targets)
        # gt_labels, gt_offsets = self._annotate_points_improved2(points, targets)
        
        # calculate point loss
        ## (1) loss norm
        pos_masks = torch.logical_and(gt_labels, fpn_masks)
        norm = pos_masks.sum()

        ## (2) classification loss on valid points
        qr_cfg = self.opt['train'].get('loss_aux', {}).get(
            'quality_ranking', {})
        if qr_cfg.get('enable', False):
            from .modeling.temporal_coordinates import (
                decode_offsets as _qr_decode,
            )
            gamma = float(qr_cfg.get('qf_gamma', 2.0))
            tau_qr = float(qr_cfg.get('tau', 0.5))
            topk_qr = int(qr_cfg.get('topk_per_level', 20))
            list_w = float(qr_cfg.get('list_weight', 0.2))
            rows_soft, rows_logit = {}, {}
            with torch.no_grad():
                q_maps = []
                for lvl, (pts_l, off_l, msk_l) in enumerate(zip(
                        fpn_points, fpn_offsets_levels, fpn_masks_levels)):
                    seg_l = _qr_decode(
                        pts_l if pts_l.ndim == 3 else pts_l.unsqueeze(0),
                        off_l,
                    )  # (B, T, 2)
                    s0 = seg_l[..., 0]; e0 = seg_l[..., 1]
                    lo = torch.minimum(s0, e0); hi = torch.maximum(s0, e0)
                    gs = targets[:, 0][:, None]; ge = targets[:, 1][:, None]
                    inter = (torch.minimum(hi, ge) - torch.maximum(lo, gs)
                             ).clamp_min(0)
                    union = ((torch.maximum(hi, ge)
                              - torch.minimum(lo, gs)) + 1e-6)
                    q_maps.append((inter / union))
            qf_terms = []
            for lvl, (lg_l, q_l, msk_l) in enumerate(zip(
                    fpn_logits_levels, q_maps, fpn_masks_levels)):
                valid = (msk_l if msk_l.ndim == 2
                         else msk_l.squeeze(-2)).bool()
                p = torch.sigmoid(lg_l.float())
                q = q_l.float()
                w = q * (1 - p).pow(gamma) * torch.log(p + 1e-9) + \
                    (1 - q) * p.pow(gamma) * torch.log(1 - p + 1e-9)
                qf_terms.append(-(w * valid).sum() /
                                valid.float().sum().clamp_min(1))
                # listwise candidates from the model's own logits
                with torch.no_grad():
                    sc = p.detach()
                for b in range(sc.size(0)):
                    v = sc[b][valid[b]]
                    if v.numel() == 0:
                        continue
                    k = min(topk_qr, v.numel())
                    pos = valid[b].nonzero().flatten()[v.topk(k).indices]
                    rows_soft.setdefault(b, []).append(q_l[b][pos].detach())
                    rows_logit.setdefault(b, []).append(lg_l[b][pos])
            cls_loss = torch.stack(qf_terms).mean() / self.loss_norm * \
                get_world_size()
            if rows_logit:
                losses = []
                for b, segs_lg in rows_logit.items():
                    lg = torch.cat(segs_lg)
                    soft = torch.cat(rows_soft[b])
                    if soft.numel() < 2:
                        continue
                    y = torch.softmax(soft / max(tau_qr, 1e-6), dim=-1)
                    losses.append(-(
                        y * torch.log_softmax(lg.float(), dim=-1)).sum())
                if losses:
                    cls_loss = cls_loss + list_w * torch.stack(losses).mean()
        else:
            cls_loss = self._calc_focal_loss(
                logits=fpn_logits[fpn_masks], labels=gt_labels[fpn_masks]
            ) / self.loss_norm * get_world_size()
        
        ## (3) regression loss on positive points
        reg_loss = self._calc_iou_loss(
            pred_offsets=fpn_offsets[pos_masks], gt_offsets=gt_offsets[pos_masks]
        ) / self.loss_norm * get_world_size()

        total_loss = cls_loss + self.loss_weight * reg_loss
        total_loss.backward()
        return {
            'cls': cls_loss.detach(),
            'reg': reg_loss.detach(),
            'total': total_loss.detach(),
            'norm': norm.detach(),
        }

    def _batchify_videos(self, vid_list):
        """
        Put video features and their masks in a batch.

        Args:
            vid_list (List[float tensor, (c1, t1)]): video features.

        Returns:
            vid (float tensor, (bs, c1, t1)): video feature sequences.
            vid_masks (bool tensor, (bs, t1)): video masks.
        """
        bs = len(vid_list)
        vid_dim = vid_list[0].size(0)
        vid_lens = [v.size(-1) for v in vid_list]
        vid = vid_list[0].new_full((bs, vid_dim, self.input_vid_len), 0.)
        for idx in range(bs):
            vid[idx, :, :vid_lens[idx]].copy_(vid_list[idx])
        vid_lens = torch.as_tensor(vid_lens)[:, None]
        vid_masks = torch.arange(self.input_vid_len)[None] < vid_lens
        return vid, vid_masks

    def _batchify_text(self, text_list):
        """
        Put text features and their masks in a batch.

        Args:
            text_list (List[float tensor, (c2, t2)]): token features.

        Returns:
            text (float tensor, (bs, c2, t2)): token feature sequences.
            text_masks (bool tensor, (bs, t2)): token masks.
        """
        bs = len(text_list)
        text_dim = text_list[0].size(0)
        text_lens = [t.size(-1) for t in text_list]
        text = text_list[0].new_full((bs, text_dim, self.max_text_len), 0.)
        for idx in range(bs):
            text[idx, :, :text_lens[idx]].copy_(text_list[idx])
        text_lens = torch.as_tensor(text_lens)[:, None]
        text_masks = torch.arange(self.max_text_len)[None] < text_lens
        return text, text_masks

    def _batchify(self, vid_list, text_list):
        assert len(vid_list) == len(text_list)
        bs = len(vid_list)

        # batch videos
        vid, vid_masks = self._batchify_videos(vid_list)

        # batch text
        if isinstance(text_list[0], tuple):
            # many text queries are associated with the same video
            b_text, b_text_masks = tuple(), tuple()
            n = tuple()
            for t in text_list:
                b_t, b_tm = self._batchify_text(t)
                b_text += (b_t, )
                b_text_masks += (b_tm, )
                n += (len(t), )
            n_max = max(n)      # max number of text queries

            # (bs, n, c, t)
            text_dim = b_text[0].size(1)
            text = b_text[0].new_full(
                (bs, n_max, text_dim, self.max_text_len), 0.
            )
            for idx in range(bs):
                text[idx, :n[idx]].copy_(b_text[idx])

            # (bs, n, t)
            text_masks = b_text_masks[0].new_full(
                (bs, n_max, self.max_text_len), 0, dtype=torch.bool
            )
            for idx in range(bs):
                text_masks[idx, :n[idx]].copy_(b_text_masks[idx])
        else:
            n = bs * (1, )
            text, text_masks = self._batchify_text(text_list)

        text_size = torch.as_tensor(n)

        # vid: (bs, c1, t1)
        # vid_masks: (bs, t1)
        # text: (bs, (n,) c2, t2)
        # text_masks (bs, (n,) t2)
        # text_size: (bs,)
        return vid, vid_masks, text, text_masks, text_size

    def _annotate_points(self, points, targets):
        """
        Assign ground-truth labels and offsets to candidate points.

        Args:
            fpn_points (List[float tensor, (p, 4)]): candidate points.
                (coordinate (1), regression range (2), stride(1))
            targets (float tensor, (bs, 2)): ground-truth segments.

        Returns:
            labels (bool tensor, (bs, p)): ground-truth binary labels.
            offsets (float tensor, (bs, p, 2)): ground-truth offsets.
        """
        labels_list, offsets_list = tuple(), tuple()
        for index, target in enumerate(targets):
            point_sample = points[index] if points.ndim == 3 else points
            labels, offsets = self._annotate_points_per_video(point_sample, target)
            labels_list += (labels, )
            offsets_list += (offsets, )
        labels = torch.stack(labels_list)
        offsets = torch.stack(offsets_list)
        return labels, offsets

    def _annotate_points_per_video(self, points, target):
        """
        Args:
            points (float tensor, (p, 4)): candidate points from all levels.
                (coordinate (1), regression range (2), stride (1))
            target (float tensor, (2,)): ground-truth segment.

        Returns:
            labels (bool tensor, (p,)): ground-truth binary labels.
            offsets (float tensor, (p, 2)): ground-truth offsets.
        """
        # point distance to segment boundaries
        pt2start = points[:, 0] - target[0]     # (p,)
        pt2end = target[1] - points[:, 0]       # (p,)

        # offsets rescaled by down-sampling stride
        offsets = encode_offsets(points, target.unsqueeze(0))[0]

        # (1) whether a point lies in given sampling window
        if self.center_sampling == 'radius':
            ctr = 0.5 * (target[0] + target[1])
            radius = points[:, 3] * self.center_sampling_radius
            t_min = (ctr - radius).clamp_(min=target[0])
            t_max = (ctr + radius).clamp_(max=target[1])
            # point distance to window boundaries
            pt2left = points[:, 0] - t_min  # (p,)
            pt2right = t_max - points[:, 0] # (p,)
            inside_window = torch.logical_and(pt2left > 0, pt2right > 0)
        else:
            inside_window = torch.logical_and(pt2start > 0, pt2end > 0)

        # (2) whether event is within regression range of a point
        max_reg_dist = torch.maximum(pt2start, pt2end)
        inside_range = torch.logical_and(
            max_reg_dist >= points[:, 1], max_reg_dist < points[:, 2]
        )

        # a point is positive only if it meets both criteria
        labels = torch.logical_and(inside_window, inside_range)

        return labels, offsets
    
    def _annotate_points_per_video_fine_scale_fix(self, points, target):
        """
        Conservative fine-scale fix that reduces regression loss.
        """
        pt2start = points[:, 0] - target[0]
        pt2end = target[1] - points[:, 0]
        offsets = encode_offsets(points, target.unsqueeze(0))[0]
        
        segment_length = target[1] - target[0]
        ctr = 0.5 * (target[0] + target[1])
        
        # (1) Standard center sampling
        if self.center_sampling == 'radius':
            radius = points[:, 3] * self.center_sampling_radius
            t_min = (ctr - radius).clamp_(min=target[0])
            t_max = (ctr + radius).clamp_(max=target[1])
            pt2left = points[:, 0] - t_min
            pt2right = t_max - points[:, 0]
            inside_window = torch.logical_and(pt2left > 0, pt2right > 0)
        else:
            inside_window = torch.logical_and(pt2start > 0, pt2end > 0)
        
        # (2) Conservative regression range adaptation
        max_reg_dist = torch.maximum(pt2start, pt2end)
        strides = points[:, 3]
        
        # More conservative approach: only minimal expansion for very short segments
        fine_scale_mask = strides <= 4
        very_short_segment = segment_length < 15  # Only help very short segments
        
        # Only expand upper bound, and only modestly
        adapted_reg_max = torch.where(
            fine_scale_mask & very_short_segment,
            points[:, 2] * 1.3,  # Only 30% expansion vs your 200%+ expansion
            points[:, 2]  # Keep original for others
        )
        
        # Keep original minimum to avoid too many low-quality positives
        inside_range = torch.logical_and(
            max_reg_dist >= points[:, 1],  # Original minimum
            max_reg_dist < adapted_reg_max  # Slightly expanded maximum
        )
        
        # Additional quality filter for very short segments
        if segment_length < 15:
            # Only keep points with reasonable symmetry to reduce regression difficulty
            asymmetry_ratio = torch.abs(pt2start - pt2end) / torch.maximum(pt2start, pt2end)
            symmetry_filter = asymmetry_ratio < 0.8  # Allow up to 80% asymmetry
            inside_range = torch.logical_and(inside_range, symmetry_filter)
        
        labels = torch.logical_and(inside_window, inside_range)
        return labels, offsets
    
    def _annotate_points_improved2(self, points, targets):
        """
        Improved annotation using multi-strategy approach.
        
        Args:
            points (float tensor, (p, 4)): candidate points.
            targets (float tensor, (bs, 2)): ground-truth segments.

        Returns:
            labels (bool tensor, (bs, p)): ground-truth binary labels.
            offsets (float tensor, (bs, p, 2)): ground-truth offsets.
        """
        labels_list, offsets_list = tuple(), tuple()
        for target in targets:
            labels, offsets = self._annotate_points_per_video_fine_scale_fix(points, target)
            labels_list += (labels, )
            offsets_list += (offsets, )
        labels = torch.stack(labels_list)
        offsets = torch.stack(offsets_list)
        return labels, offsets
    
    def _annotate_points_per_video_short_segments(self, points, target):
        """
        Improved annotation method for very short segments (e.g., length ~10 in videos of length ~900).
        Uses more conservative and targeted improvements.
        
        Args:
            points (float tensor, (p, 4)): candidate points from all levels.
                (coordinate (1), regression range (2), stride (1))
            target (float tensor, (2,)): ground-truth segment.

        Returns:
            labels (bool tensor, (p,)): ground-truth binary labels.
            offsets (float tensor, (p, 2)): ground-truth offsets.
        """
        # point distance to segment boundaries
        pt2start = points[:, 0] - target[0]     # (p,)
        pt2end = target[1] - points[:, 0]       # (p,)

        # offsets rescaled by down-sampling stride
        offsets = encode_offsets(points, target.unsqueeze(0))[0]

        # Calculate segment length and center
        segment_length = target[1] - target[0]
        ctr = 0.5 * (target[0] + target[1])

        # (1) Multi-scale aware center sampling
        if self.center_sampling == 'radius':
            base_radius = points[:, 3] * self.center_sampling_radius
            
            # Strategy 1: Scale-adaptive radius with conservative expansion
            stride_ratio = segment_length / points[:, 3]
            
            # Only boost for very fine scales where segment is smaller than 2x stride
            scale_boost = torch.where(
                stride_ratio < 1.0,  # Very short relative to stride
                torch.clamp(1.0 / torch.sqrt(stride_ratio), 1.0, 1.5),  # Conservative sqrt-based boost
                torch.ones_like(stride_ratio)
            )
            
            # Strategy 2: Ensure minimum effective radius but cap it
            min_effective_radius = torch.minimum(
                segment_length * 0.2,  # 20% of segment length
                points[:, 3] * 0.5     # Or half the stride, whichever is smaller
            )
            
            adaptive_radius = torch.maximum(base_radius * scale_boost, min_effective_radius)
            
            # Apply adaptive radius
            t_min = (ctr - adaptive_radius).clamp_(min=target[0])
            t_max = (ctr + adaptive_radius).clamp_(max=target[1])
            
            pt2left = points[:, 0] - t_min
            pt2right = t_max - points[:, 0]
            inside_window = torch.logical_and(pt2left > 0, pt2right > 0)
        else:
            inside_window = torch.logical_and(pt2start > 0, pt2end > 0)

        # (2) Conservative regression range adaptation
        max_reg_dist = torch.maximum(pt2start, pt2end)
        
        # Strategy 3: Gradual regression range relaxation based on segment/stride ratio
        if segment_length < points[:, 3].min() * 2:  # Only for very short segments
            # Conservative expansion: only 25% relaxation
            relaxation_factor = 1.25
            expanded_reg_min = points[:, 1] / relaxation_factor
            expanded_reg_max = points[:, 2] * relaxation_factor
            
            # Use weighted combination of original and relaxed ranges
            weight = torch.minimum(segment_length / (points[:, 3] * 2), torch.tensor(1.0))
            final_reg_min = weight * points[:, 1] + (1 - weight) * expanded_reg_min
            final_reg_max = weight * points[:, 2] + (1 - weight) * expanded_reg_max
            
            inside_range = torch.logical_and(
                max_reg_dist >= final_reg_min, max_reg_dist < final_reg_max
            )
        else:
            # Normal regression range constraint
            inside_range = torch.logical_and(
                max_reg_dist >= points[:, 1], max_reg_dist < points[:, 2]
            )

        # Strategy 4: Additional quality filtering for very short segments
        if segment_length < 20:
            # Prefer points closer to segment center for very short segments
            dist_to_center = torch.abs(points[:, 0] - ctr)
            center_weight = torch.exp(-dist_to_center / (segment_length * 0.5))
            
            # Only keep points with reasonable center alignment (top 80% by center weight)
            center_thresh = torch.quantile(center_weight[inside_window], 0.2) if inside_window.sum() > 5 else 0.0
            center_filter = center_weight >= center_thresh
            
            labels = torch.logical_and(
                torch.logical_and(inside_window, inside_range),
                center_filter
            )
        else:
            labels = torch.logical_and(inside_window, inside_range)

        return labels, offsets

    def _annotate_points_ultra_short(self, points, target):
        """
        Specialized handling for ultra-short segments (< 5 time units).
        """
        pt2start = points[:, 0] - target[0]
        pt2end = target[1] - points[:, 0]
        offsets = encode_offsets(points, target.unsqueeze(0))[0]
        
        segment_length = target[1] - target[0]
        ctr = 0.5 * (target[0] + target[1])
        
        # Very tight center sampling
        if self.center_sampling == 'radius':
            # Use fixed small radius for ultra-short segments
            tight_radius = torch.minimum(
                segment_length * 0.3,  # 30% of segment
                points[:, 3] * 0.3     # 30% of stride
            )
            
            t_min = (ctr - tight_radius).clamp_(min=target[0])
            t_max = (ctr + tight_radius).clamp_(max=target[1])
            
            pt2left = points[:, 0] - t_min
            pt2right = t_max - points[:, 0]
            inside_window = torch.logical_and(pt2left > 0, pt2right > 0)
        else:
            inside_window = torch.logical_and(pt2start > 0, pt2end > 0)
        
        # Strict regression range - only minimal relaxation
        max_reg_dist = torch.maximum(pt2start, pt2end)
        inside_range = torch.logical_and(
            max_reg_dist >= points[:, 1] * 0.8,  # Minimal relaxation
            max_reg_dist < points[:, 2] * 1.2
        )
        
        # Only keep the most relevant scale levels for ultra-short segments
        scale_filter = points[:, 3] <= segment_length * 2  # Only use fine scales
        
        labels = torch.logical_and(
            torch.logical_and(inside_window, inside_range),
            scale_filter
        )
        
        return labels, offsets

    def _annotate_points_multi_strategy(self, points, target):
        """
        Multi-strategy annotation that combines different approaches based on segment characteristics.
        """
        segment_length = target[1] - target[0]
        
        # Strategy selection based on segment length
        if segment_length < 5:
            # Ultra-short segments: very conservative, focus on highest quality points
            return self._annotate_points_ultra_short(points, target)
        elif segment_length < 20:
            # Short segments: use improved conservative method
            return self._annotate_points_per_video_short_segments(points, target)
        else:
            # Normal segments: use original method
            return self._annotate_points_per_video(points, target)

    def _annotate_points_improved(self, points, targets):
        """
        Improved annotation using multi-strategy approach.
        
        Args:
            points (float tensor, (p, 4)): candidate points.
            targets (float tensor, (bs, 2)): ground-truth segments.

        Returns:
            labels (bool tensor, (bs, p)): ground-truth binary labels.
            offsets (float tensor, (bs, p, 2)): ground-truth offsets.
        """
        labels_list, offsets_list = tuple(), tuple()
        for index, target in enumerate(targets):
            point_sample = points[index] if points.ndim == 3 else points
            labels, offsets = self._annotate_points_multi_strategy(point_sample, target)
            labels_list += (labels, )
            offsets_list += (offsets, )
        labels = torch.stack(labels_list)
        offsets = torch.stack(offsets_list)
        return labels, offsets

    def _calc_focal_loss(self, logits, labels, smoothing=0.2, alpha=0.5):
        labels = labels.to(logits.dtype) * (1.0 - smoothing) + smoothing / 2
        return sigmoid_focal_loss(logits, labels, alpha=alpha, reduction='sum')

    def _calc_iou_loss(self, pred_offsets, gt_offsets):
        iou_loss = ctr_diou_loss if self.reg_loss == 'diou' else ctr_giou_loss
        return iou_loss(pred_offsets, gt_offsets, reduction='sum')

    def _ema_init(self):
        for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
            p_ema.copy_(p.detach())
        for b, b_ema in zip(self.model.buffers(), self.model_ema.buffers()):
            b_ema.copy_(b.detach())

    @torch.no_grad()
    def _ema_update(self):
        for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
            p_ema.copy_(p.detach().lerp(p_ema, self.ema_beta))

    @staticmethod
    def _load_model_state(model, state_dict):
        load_compatible = getattr(
            model, 'load_compatible_state_dict', None
        )
        if load_compatible is None:
            return model.load_state_dict(state_dict)
        return load_compatible(state_dict)

    def load(self):
        model_path = os.path.join(self.opt['_root'], 'models', 'last.pth')
        state_path = os.path.join(self.opt['_root'], 'states', 'last.pth')
        model_ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        state_ckpt = torch.load(state_path, map_location='cpu', weights_only=False)
        self._load_model_state(self.model, model_ckpt['model'])
        self._load_model_state(self.model_ema, model_ckpt['model_ema'])
        self.optimizer.load_state_dict(state_ckpt['optimizer'])
        self.scheduler.load_state_dict(state_ckpt['scheduler'])
        self.epoch, self.itr = state_ckpt['epoch'], state_ckpt['itr']
        e, t = len(str(self.num_epochs)), len(str(self.num_itrs))
        print0(f"Loaded checkpoint [epoch {self.epoch:0{e}d} / itr {self.itr:0{t}d}]...")

    def _unwrap(self, model):
        return model.module if self.opt['_distributed'] else model

    def checkpoint(self):
        e, t = len(str(self.num_epochs)), len(str(self.num_itrs))
        print0(f"Checkpointing at [epoch {self.epoch:0{e}d} / itr {self.itr:0{t}d}]...")
        model_dir = os.path.join(self.opt['_root'], 'models')
        state_dir = os.path.join(self.opt['_root'], 'states')
        model_ckpt = {
            'model': self._unwrap(self.model).state_dict(),
            'model_ema': self.model_ema.state_dict(),
        }
        state_ckpt = {
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'itr': self.itr,
        }
        torch.save(model_ckpt, os.path.join(model_dir, 'last.pth'))
        torch.save(state_ckpt, os.path.join(state_dir, 'last.pth'))
        if self.epoch in self.checkpoint_epochs:
            shutil.copyfile(
                os.path.join(model_dir, 'last.pth'),
                os.path.join(model_dir, f"{self.epoch:0{e}d}.pth")
            )

    def log(self):
        t = len(str(self.num_itrs))
        log_str = f"[{self.itr:0{t}d}/{self.num_itrs:0{t}d}] "
        for k, v in self.loss_meters.items():
            log_str += f"{k} {v.item():.3f} | "
            self.tb_writer.add_scalar(k, v.item(), self.itr)
            v.reset()
        lr = self.scheduler.get_last_lr()[0]
        self.tb_writer.add_scalar('lr', lr, self.itr)
        log_str += time_str(self.timer.item() * self.log_interval)
        self.timer.reset()
        self.logger.write(log_str)
        self.tb_writer.flush()





class TrainerAuxiliary(TrainerOriginal):

    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        vid_embd_dim = opt['model']['vid_net']['embd_dim']
        self.ds_contrastive = opt['train']['loss_aux']['ds_contrast']['enable']
        self.gt_contrastive = opt['train']['loss_aux']['gt_contrast']['enable']
        cross_level_opt = opt['train'].get(
            'loss_aux', {}
        ).get('cross_level_consistency', {})
        self.cross_level_consistency = cross_level_opt.get(
            'enable', False
        )
        self.early_fusion = opt['model'].get('early_fusion', True)
        self.use_mst = opt['model'].get('use_mst', False)
        model_ref = (
            self.model.module if hasattr(self.model, 'module') else self.model
        )
        ds_contrast_opt = opt['train']['loss_aux']['ds_contrast']
        self.acc_mode = ds_contrast_opt.get('acc_mode', 'legacy_acc')
        self.cls_target_type = opt['train'].get('cls_target_type', 'binary')
        self.tefm_lambda = float(opt['train'].get('tefm_lambda', 0.0))
        self.tefm_schedule = opt['train'].get('tefm_schedule', 'off')
        self.tefm_decay_start = int(opt['train'].get('tefm_decay_start', 4000))
        self.tefm_total_steps = int(opt['train'].get('tefm_total_steps', 8000))
        if self.tefm_schedule not in ('off', 'constant', 'cosine_decay', 'hard_off'):
            raise ValueError(f'unknown tefm_schedule: {self.tefm_schedule}')
        self.metric_tau = float(opt['train'].get('metric_tau', 0.05))
        if self.cls_target_type not in (
                'binary', 'decoded_iou', 'metric_utility'):
            raise ValueError(
                'cls_target_type must be binary/decoded_iou/'
                f'metric_utility, got {self.cls_target_type}')
        self.assignment_acc = bool(
            self.ds_contrastive
            and self.acc_mode in (
                'assignment_acc', 'legacy_consistent_assignment_acc'
            )
        )
        if self.acc_mode not in (
            'legacy_acc', 'assignment_acc',
            'legacy_consistent_assignment_acc',
        ):
            raise ValueError(
                'acc_mode must be legacy_acc, assignment_acc, or '
                f'legacy_consistent_assignment_acc, got {self.acc_mode}'
            )
        if self.acc_mode in ('assignment_acc',
                             'legacy_consistent_assignment_acc') and not bool(
            getattr(getattr(model_ref, 'vid_net', None), 'adaptive_anchor', False)
        ):
            raise ValueError(
                f'acc_mode={self.acc_mode} requires '
                'model.vid_net.adaptive_anchor=true'
            )
        self.allocator_policy = getattr(model_ref, 'allocator_policy', 'learned')
        if self.allocator_policy == 'oracle' and not self.adaptive_anchor:
            raise ValueError(
                'oracle allocator_policy requires adaptive_anchor=true'
            )
        self.query_conditioned = bool(
            getattr(model_ref, 'query_conditioned', False)
        )
        importance_opt = deepcopy(
            opt['train'].get('loss_aux', {}).get(
                'query_boundary_importance', {}
            )
        )
        self.query_boundary_importance = bool(
            importance_opt.get('enable', False)
        )
        (
            self.boundary_loss_weight,
            boundary_supervision_loss,
        ) = make_boundary_supervision_loss(importance_opt)
        self.boundary_supervision = boundary_supervision_loss is not None
        self.return_importance_debug = bool(
            self.query_boundary_importance or self.boundary_supervision
        )
        if self.return_importance_debug and not bool(
            getattr(model_ref, 'importance_enabled', False)
        ):
            raise ValueError(
                'boundary supervision requires '
                'model.vid_net.query_boundary_importance=true '
                'or adaptive_anchor=true'
            )
        self.query_boundary_importance_weight = float(
            importance_opt.get('weight', 0.1)
        )
        alloc_opt = importance_opt.get('alloc_supervision', {})
        self.allocation_supervision = bool(alloc_opt.get('enable', False))
        if self.allocation_supervision:
            if not self.adaptive_anchor:
                raise ValueError(
                    'alloc_supervision requires adaptive_anchor=true'
                )
            self.allocation_supervision_loss = AllocationSupervisionLoss(
                sigma=alloc_opt.get('sigma', 2.0),
                alpha=alloc_opt.get('alpha', 1.0),
                beta=alloc_opt.get('beta', 0.5),
            )
            self.allocation_supervision_weight = float(
                alloc_opt.get('weight', 0.5)
            )
        rank_opt = importance_opt.get('ranking_supervision', {})
        self.ranking_supervision = bool(rank_opt.get('enable', False))
        if self.ranking_supervision:
            if not self.adaptive_anchor:
                raise ValueError(
                    'ranking_supervision requires adaptive_anchor=true'
                )
            self.ranking_allocation_loss = RankingAllocationLoss(
                boundary_radius=rank_opt.get('boundary_radius', 1.5),
                margin_bf=rank_opt.get('margin_bf', 0.1),
                margin_fb=rank_opt.get('margin_fb', 0.1),
            )
            self.ranking_supervision_weight = float(
                rank_opt.get('weight', 0.5)
            )
        self.return_importance_debug = bool(
            self.query_boundary_importance
            or self.boundary_supervision
            or self.allocation_supervision
            or self.ranking_supervision
        )
        if (
            self.query_boundary_importance
            and self.query_boundary_importance_weight < 0
        ):
            raise ValueError(
                'query boundary importance loss weight must be non-negative'
            )
        if self.query_boundary_importance:
            vid_net_opt = opt['model']['vid_net']
            for key, default in (
                ('importance_alpha', 0.4),
                ('importance_beta', 0.4),
                ('importance_gamma', 0.2),
            ):
                importance_opt.setdefault(
                    key, vid_net_opt.get(key, default)
                )
            self.query_boundary_importance_loss = (
                QueryBoundaryImportanceLoss(importance_opt).cuda()
            )
        else:
            self.query_boundary_importance_weight = 0.0
            self.query_boundary_importance_loss = None
        self.boundary_supervision_loss = (
            boundary_supervision_loss.cuda()
            if self.boundary_supervision
            else None
        )
        if self.early_fusion:
            self.logger.write("Early fusion enabled")

        if self.ds_contrastive:
            ds_loss_type = AUX_LOSS_REGISTRY[
                ds_contrast_opt.get('type', 'ds_contrastive')
            ]
            self.ds_contrastive_loss = ds_loss_type(
                ds_contrast_opt, vid_embd_dim
            ).cuda()
            self.ds_contrastive_weight = ds_contrast_opt['weight']
        else: 
            self.ds_contrastive_weight = 0.0
        
        if self.gt_contrastive:
            gt_loss_type = AUX_LOSS_REGISTRY[opt['train']['loss_aux']['gt_contrast'].get('type', 'gt_point_contrastive')]
            self.gt_contrastive_loss = gt_loss_type(opt['train']['loss_aux']['gt_contrast'], vid_embd_dim).cuda()
            self.gt_contrastive_weight = opt['train']['loss_aux']['gt_contrast']['weight']
            self.loss_aux_gt_type = opt['train']['loss_aux']['gt_contrast'].get('gt_type', 'point')
            self.loss_aux_span_radius = opt['train']['loss_aux']['gt_contrast'].get('span_radius', self.center_sampling_radius)
            self.span_contr_gt = opt['train']['loss_aux']['gt_contrast'].get('span_contr_gt', False)       
        else:
            self.gt_contrastive_weight = 0.0

        if self.cross_level_consistency:
            self.cross_level_consistency_loss = (
                CrossLevelQualityConsistency(cross_level_opt).cuda()
            )
            self.cross_level_consistency_weight = float(
                cross_level_opt.get('weight', 0.1)
            )
            self.cross_level_warmup_epochs = int(
                cross_level_opt.get('warmup_epochs', 0)
            )
        else:
            self.cross_level_consistency_weight = 0.0
            self.cross_level_warmup_epochs = 0

    def _current_tefm_lambda(self):
        """HM-TEFM-039: scheduled evidence-loss coefficient."""
        import math
        if self.tefm_schedule == 'off':
            return 0.0
        if self.tefm_schedule == 'constant':
            return self.tefm_lambda
        if self.tefm_schedule == 'hard_off':
            return self.tefm_lambda if self.itr <= self.tefm_decay_start else 0.0
        if self.tefm_schedule == 'cosine_decay':
            if self.itr <= self.tefm_decay_start:
                return self.tefm_lambda
            p = (self.itr - self.tefm_decay_start) / max(
                self.tefm_total_steps - self.tefm_decay_start, 1)
            return self.tefm_lambda * 0.5 * (1 + math.cos(math.pi * p))
        return 0.0

    def forward_backward(self, data_list):
        cls_loss = reg_loss = total_loss = norm = 0
        ds_contrast = gt_contrast = cross_level = 0
        importance_loss = importance_relevance = importance_boundary = 0
        importance_change = importance_final = 0
        boundary_loss = boundary_start = boundary_end = 0
        alloc_loss = 0
        rank_loss = 0
        list_loss = 0
        dn_loss = 0
        trans_loss = 0
        for i in range(0, self.batch_size, self.microbatch_size):
            loss_dict = self._microbatch_forward_backward(
                data_list[i:i + self.microbatch_size],
                is_last=(i + self.microbatch_size >= self.batch_size)
            )
            cls_loss += loss_dict['cls']
            reg_loss += loss_dict['reg']
            total_loss += loss_dict['total']
            norm += loss_dict['norm']
            ds_contrast += loss_dict['ds_contrast']
            gt_contrast += loss_dict['gt_contrast']
            cross_level += loss_dict['cross_level']
            importance_loss += loss_dict['importance_loss']
            importance_relevance += loss_dict['importance_relevance']
            importance_boundary += loss_dict['importance_boundary']
            importance_change += loss_dict['importance_change']
            importance_final += loss_dict['importance_final']
            boundary_loss += loss_dict['boundary_loss']
            boundary_start += loss_dict['boundary_start']
            boundary_end += loss_dict['boundary_end']
            alloc_loss = alloc_loss + loss_dict.get(
                'alloc_loss', loss_dict['total'] * 0.0
            )
            rank_loss = rank_loss + loss_dict.get(
                'rank_loss', loss_dict['total'] * 0.0
            )
            list_loss = list_loss + loss_dict.get(
                'list_loss', loss_dict['total'] * 0.0
            )
            dn_loss = dn_loss + loss_dict.get(
                'dn_loss', loss_dict['total'] * 0.0
            )
            trans_loss = trans_loss + loss_dict.get(
                'trans_loss', loss_dict['total'] * 0.0
            )
        
        # update EMA loss norm
        all_norms = [torch.zeros_like(norm) for _ in range(get_world_size())]
        all_gather(all_norms, norm)
        self.loss_norm = (
            self.loss_norm_momentum * self.loss_norm
            + (1. - self.loss_norm_momentum) * max(sum(all_norms).item(), 1)
        )
        return {
            'cls': cls_loss,
            'reg': reg_loss,
            'ds_contrast': ds_contrast,
            'gt_contrast': gt_contrast,
            'cross_level': cross_level,
            'importance_loss': importance_loss,
            'importance_relevance': importance_relevance,
            'importance_boundary': importance_boundary,
            'importance_change': importance_change,
            'importance_final': importance_final,
            'boundary_loss': boundary_loss,
            'boundary_start': boundary_start,
            'boundary_end': boundary_end,
            'alloc_loss': alloc_loss,
            'rank_loss': rank_loss,
            'list_loss': list_loss,
            'dn_loss': dn_loss,
            'trans_loss': trans_loss,
            'total': total_loss,
        }

    def _microbatch_forward_backward(self, data_list, is_last=False):
        # batch data
        vid, vid_masks, text, text_masks, text_size = self._batchify(
            vid_list=[d['vid'] for d in data_list], 
            text_list=[d['text'] for d in data_list]
        )
        vid = vid.cuda(non_blocking=True)
        vid_masks = vid_masks.cuda(non_blocking=True)
        text = text.cuda(non_blocking=True)
        text_masks = text_masks.cuda(non_blocking=True)
        text_size = text_size.cuda(non_blocking=True)

        target_divisor = 1 if self.adaptive_anchor else self.vid_stride
        targets = torch.cat([d['target'] / target_divisor for d in data_list])
        targets = targets.cuda(non_blocking=True)
        
        # forward pass
        model_kwargs = {}
        if self.return_importance_debug:
            model_kwargs['return_importance_debug'] = True
        if self.assignment_acc:
            model_kwargs['return_anchor_assignments'] = True
        if self.allocator_policy == 'oracle':
            # Diagnostic-only GT grouping resolution; targets never reach the
            # grounding heads. Rows align with the query-expanded video batch.
            model_kwargs['allocator_targets'] = targets
        if is_last or not self.opt['_distributed']:
            model_outputs = self.model(
                vid,
                vid_masks,
                text,
                text_masks,
                text_size,
                **model_kwargs
            )
        else:
            with self.model.no_sync():
                model_outputs = self.model(
                    vid,
                    vid_masks,
                    text,
                    text_masks,
                    text_size,
                    **model_kwargs
                )
        (
            fpn_logits,
            fpn_logits2,
            fpn_offsets,
            fpn_masks,
            fpn,
            sequence_fpn_masks,
            anchor_fpn,
            anchor_fpn_masks,
            *extra_outputs
        ) = model_outputs
        expected_extra = (
            int(self.return_importance_debug)
            + int(self.assignment_acc)
        )
        if len(extra_outputs) != expected_extra:
            raise RuntimeError(
                'expected {} auxiliary model outputs, got {}'.format(
                    expected_extra, len(extra_outputs)
                )
            )
        extra_index = 0
        importance_debug = None
        anchor_assignments = None
        if self.return_importance_debug:
            importance_debug = extra_outputs[extra_index]
            extra_index += 1
        if self.assignment_acc:
            anchor_assignments = extra_outputs[extra_index]
        fpn_logits_levels = fpn_logits
        fpn_offsets_levels = fpn_offsets
        fpn_masks_levels = fpn_masks
        fpn_n_points = [m.size(-1) for m in fpn_masks]
        if self.adaptive_anchor:
            model_ref = self.model.module if hasattr(self.model, 'module') else self.model
            fpn_points = self.adaptive_pt_gen(
                model_ref.vid_net.last_temporal_metadata, fpn_masks
            )
        else:
            fpn_points = self.pt_gen(fpn_n_points)

        zero_aux_loss = fpn_logits_levels[0].float().sum() * 0.0
        importance_loss = zero_aux_loss
        importance_components = {
            key: zero_aux_loss
            for key in (
                'relevance',
                'boundary',
                'temporal_change',
                'importance',
            )
        }
        if self.query_boundary_importance:
            importance_loss, importance_components = (
                self.query_boundary_importance_loss(
                    importance_debug,
                    fpn_points,
                    sequence_fpn_masks,
                    targets,
                    return_components=True,
                )
            )
        boundary_loss = zero_aux_loss
        boundary_components = {
            'start': zero_aux_loss,
            'end': zero_aux_loss,
            'boundary': zero_aux_loss,
        }
        if self.boundary_supervision:
            boundary_loss, boundary_components = (
                self.boundary_supervision_loss(
                    importance_debug,
                    fpn_points,
                    sequence_fpn_masks,
                    targets,
                    return_components=True,
                )
            )

        list_loss = zero_aux_loss
        if self.listwise_ranking:
            from .modeling.temporal_coordinates import decode_offsets
            lr_cfg = self.opt['train']['loss_aux']['listwise_ranking']
            topk_per_level = int(lr_cfg.get('per_level_topk', 20))
            tau_y = float(lr_cfg.get('tau', 0.5))
            pooled_all, geo_all, lvl_all, rows_all, iou_all = (
                [], [], [], [], []
            )
            for lvl, (lg, off, pts_l, fpn_l, msk_l) in enumerate(zip(
                fpn_logits_levels, fpn_offsets_levels, fpn_points, fpn,
                fpn_masks_levels,
            )):
                segs_l = decode_offsets(pts_l, off)        # (B, T, 2) tokens
                prob = torch.sigmoid(lg).float()
                valid = (msk_l if msk_l.ndim == 2
                         else msk_l.squeeze(-2)).bool()
                for b in range(prob.size(0)):
                    v = prob[b][valid[b]]
                    if v.numel() == 0:
                        continue
                    k = min(topk_per_level, v.numel())
                    sc_v, idx_v = v.topk(k)
                    pos = valid[b].nonzero(as_tuple=False).flatten()
                    if idx_v.numel() == 0 or int(idx_v.max()) >= pos.numel():
                        raise RuntimeError(
                            f"listwise shape mismatch: lg={tuple(lg.shape)} "
                            f"mask={tuple(msk_l.shape)} valid={tuple(valid.shape)} "
                            f"b={b} nonzero={pos.numel()} v={tuple(v.shape)} "
                            f"idxmax={int(idx_v.max()) if idx_v.numel() else -1}")
                    for s_val, t_idx in zip(sc_v.tolist(), pos.tolist()):
                        s0, e0 = segs_l[b, t_idx].tolist()
                        ctr = pts_l[b, t_idx, 0].item()
                        centers = pts_l[b, :, 0]
                        inside = (centers >= min(s0, e0)) & (
                            centers <= max(s0, e0)) & valid[b]
                        if not bool(inside.any()):
                            inside = valid[b]
                        pooled_all.append(
                            fpn_l[b, :, inside].float().mean(dim=-1))
                        dur = abs(e0 - s0) + 1e-6
                        span_ref = float(pts_l.size(1)) + 1.0
                        geo_all.append(torch.tensor(
                            [math.log(dur), ctr / span_ref,
                             min(s0, e0) / span_ref, max(s0, e0) / span_ref],
                            device=prob.device))
                        lvl_all.append(lvl)
                        rows_all.append(b)
                        t0, t1 = targets[b, 0].item(), targets[b, 1].item()
                        inter = max(0.0, min(e0, t1) - max(s0, t0))
                        union = (max(e0, t1) - min(s0, t0)) + 1e-6
                        iou_all.append(inter / union)
            if pooled_all:
                scores = self.model.moment_rank_head(
                    torch.stack(pooled_all), torch.stack(geo_all),
                    torch.tensor(lvl_all, device=prob.device))
                ious = torch.tensor(iou_all, device=scores.device)
                by_row = {}
                for i, r in enumerate(rows_all):
                    by_row.setdefault(r, []).append(i)
                losses = []
                for r, idxs in by_row.items():
                    if len(idxs) < 2:
                        continue
                    s = scores[idxs]
                    y = ious[idxs] / max(tau_y, 1e-6)
                    y = torch.softmax(y, dim=-1)
                    losses.append(-(
                        y * torch.log_softmax(s, dim=-1)).sum())
                if losses:
                    list_loss = torch.stack(losses).mean()
                self._last_listwise_debug = (
                    ious.detach(), scores.detach(),
                    torch.tensor([s for s in [
                        v for grp in [
                            prob[b][valid[b]] for b in range(prob.size(0))
                        ] for v in grp.topk(
                            min(topk_per_level, grp.numel())
                        ).values.tolist()
                    ]], device=scores.device) if False else None,
                )
        trans_loss = zero_aux_loss
        qstb_cfg = self.opt['train'].get('loss_aux', {}).get(
            'state_transition', {})
        if qstb_cfg.get('enable', False):
            _mref = (self.model.module if hasattr(self.model, 'module')
                     else self.model)
            dbg = getattr(_mref.vid_net, 'last_qstb', None) or {}
            meta_lv = _mref.vid_net.last_temporal_metadata
            sigma_factor = float(qstb_cfg.get('sigma_factor', 1.5))
            terms = []
            for lvl, rec in dbg.items():
                logits = rec['logits'].float()          # (B,T,2)
                msk = rec['mask']                       # (B,T)
                md = meta_lv[lvl].float()
                centers = md[..., 0] * 0.5 + md[..., 1] * 0.5
                centers = md[..., 2]
                span = md[..., 3].abs().clamp_min(1e-6)
                gs = targets[:, 0].float()[:, None]
                ge = targets[:, 1].float()[:, None]
                sig = (sigma_factor * span).clamp_min(1e-6)
                y_s = torch.exp(-0.5 * ((centers - gs) / sig) ** 2)
                y_e = torch.exp(-0.5 * ((centers - ge) / sig) ** 2)
                valid = msk & (span > 0)
                if bool(valid.any()):
                    lg = logits[valid]
                    terms.append(F.binary_cross_entropy_with_logits(
                        lg[:, 0], y_s[valid]) +
                        F.binary_cross_entropy_with_logits(
                            lg[:, 1], y_e[valid]))
            if terms:
                trans_loss = torch.stack(terms).mean()
        dn_loss = zero_aux_loss
        if self.span_denoising:
            dn_cfg = self.opt['train']['loss_aux']['span_denoising']
            n_groups = int(dn_cfg.get('groups', 4))
            with torch.no_grad():
                packed_text, packed_tmask = text, text_masks
                if packed_text.ndim == 4:
                    packed_text = torch.cat(
                        [tt[:k] for tt, k in zip(packed_text, text_size)])
                    packed_tmask = torch.cat(
                        [tm[:k] for tm, k in zip(packed_tmask, text_size)])
                enc_text, enc_tmask = self.model.encode_text(
                    packed_text.float(), packed_tmask)
                if enc_text.ndim == 4:      # (B, Q, C, L) -> (B*Q, C, L)
                    enc_text = enc_text.flatten(0, 1)
                if enc_tmask.ndim == 3:
                    enc_tmask = enc_tmask.flatten(0, 1)
                tmask_f = enc_tmask[:, None, :].to(enc_text.dtype)
                query_repr = (enc_text * tmask_f).sum(-1) / \
                    tmask_f.sum(-1).clamp_min(1.0)         # (N_q, Cq)
            fpn0, pts0 = fpn[0], fpn_points[0]
            if pts0.ndim == 2:
                pts0 = pts0.unsqueeze(0).expand(fpn0.size(0), -1, -1)
            centers0 = pts0[..., 0]
            valid0 = fpn_masks_levels[0]
            valid0 = (valid0 if valid0.ndim == 2
                      else valid0.squeeze(-2)).bool()
            pooled_list, geo_list, tgt_list, qrows = [], [], [], []
            gen = torch.Generator(device='cpu').manual_seed(
                int(self.itr) if hasattr(self, 'itr') else 0)
            for b in range(fpn0.size(0)):
                gs, ge = targets[b, 0].item(), targets[b, 1].item()
                dur = max(ge - gs, 1e-6)
                for g in range(n_groups):
                    shift = (torch.rand(1, generator=gen).item() - 0.5) \
                        * 0.5 * dur
                    scale = 0.5 + 1.5 * torch.rand(1, generator=gen).item()
                    ns = gs - dur * (scale - 1.0) / 2 + shift
                    ne = ge + dur * (scale - 1.0) / 2 + shift
                    ns = max(ns, float(centers0[b].min()))
                    ne = min(ne, float(centers0[b].max()))
                    if ne - ns < max(dur * 0.2, 1.0):
                        continue
                    inside = (centers0[b] >= ns) & (centers0[b] <= ne) \
                        & valid0[b]
                    if not bool(inside.any()):
                        continue
                    pooled_list.append(fpn0[b, :, inside].float().mean(-1))
                    ref = float(centers0.size(1)) + 1.0
                    geo_list.append(torch.tensor(
                        [math.log(ne - ns + 1e-6),
                         0.5 * (ns + ne) / ref, ns / ref, ne / ref]))
                    tgt_list.append([(gs - ns) / (ne - ns),
                                     (ge - ns) / (ne - ns)])
                    qrows.append(b)
            if pooled_list:
                pooled = torch.stack(pooled_list)
                geo = torch.stack(geo_list).to(pooled.device)
                qr = query_repr[torch.tensor(qrows, device=query_repr.device)]
                pred = self.model.moment_denoiser(pooled, qr, geo)
                tgt = torch.tensor(tgt_list, device=pred.device)
                dn_loss = torch.nn.functional.smooth_l1_loss(pred, tgt)
        alloc_loss = zero_aux_loss
        rank_loss = zero_aux_loss
        if self.ranking_supervision:
            _mref_r = (
                self.model.module
                if hasattr(self.model, 'module') else self.model
            )
            rank_loss = self.ranking_allocation_loss(
                importance_debug,
                _mref_r.vid_net.last_temporal_metadata,
                sequence_fpn_masks,
                targets,
            )
        if self.allocation_supervision:
            _model_ref = (
                self.model.module
                if hasattr(self.model, 'module') else self.model
            )
            alloc_loss = self.allocation_supervision_loss(
                importance_debug,
                _model_ref.vid_net.last_temporal_metadata,
                sequence_fpn_masks,
                targets,
            )

        # stitch model outputs
        fpn_logits = torch.cat(fpn_logits, dim=1)   # (bs, p)
        fpn_offsets = torch.cat(fpn_offsets, dim=1) # (bs, p, 2)
        fpn_masks = torch.cat(fpn_masks, dim=1)     # (bs, p)
        points = (torch.cat(fpn_points, dim=1) if self.adaptive_anchor else torch.cat(fpn_points))

        # annotate points
        gt_labels, gt_offsets = self._annotate_points(points, targets)
        if self.cls_target_type == 'binary':
            cls_targets = gt_labels.float()
        else:
            from .modeling.quality_targets import (
                compute_quality_cls_targets,
            )
            cls_targets = compute_quality_cls_targets(
                points, fpn_offsets, targets, self.cls_target_type,
                tau=self.metric_tau)
        gt_labels_split = gt_labels.split(fpn_n_points, dim=1)
        fpn_masks_split = fpn_masks_levels

        if self.cross_level_consistency:
            cross_level_loss = self.cross_level_consistency_loss(
                fpn_points,
                fpn_logits_levels,
                fpn_offsets_levels,
                fpn_masks_levels,
                gt_labels_split,
                targets,
            ) * get_world_size()
        else:
            cross_level_loss = None

        if self.ds_contrastive:
            ds_contrastive_loss = self.ds_contrastive_loss(
                fpn,
                sequence_fpn_masks,
                anchor_fpn,
                anchor_fpn_masks,
                assignment_matrices=anchor_assignments,
            ) / self.loss_norm * get_world_size()
            # Shared regular-FPN features keep the released normalization.
            if not self.early_fusion and not self.query_conditioned:
                ds_contrastive_loss = (
                    text_size.float().mean() * ds_contrastive_loss
                )
        else:
            ds_contrastive_loss = torch.tensor(0.0).cuda()

        if self.gt_contrastive:
            gt_labels_split = gt_labels.split(fpn_n_points, dim=1) # (B*num_queries, T_l)
            fpn_masks_split = fpn_masks.split(fpn_n_points, dim=1) # (B*num_queries, T_l)
            # fpn_logits_split = fpn_logits.split(fpn_n_points, dim=1)

            # gt labels SPAN
            if self.adaptive_anchor:
                gt_labels_span = generate_multiscale_gt_masks_from_points(
                    fpn_points, targets
                )
            else:
                gt_labels_span = generate_multiscale_gt_masks(
                    targets, fpn_n_points
                )
            gt_labels_span = gt_labels_span.split(fpn_n_points, dim=1) # (B*num_queries, T_l)
            
            # gt labels SPAN CONTRASTIVE
            gt_labels_span_contrastive = generate_multiscale_gt_masks_contrastive(points, targets, self.loss_aux_span_radius)
            gt_labels_span_contrastive = gt_labels_span_contrastive.split(fpn_n_points, dim=1) # (B*num_queries, T_l)
            
            # replace gt labels with contrastive gt labels that was sampled with configured radius
            if self.span_contr_gt:
                gt_labels_split = gt_labels_span_contrastive
            
            # Query-conditioned regular FPNs are already query-expanded.
            if not self.early_fusion and not self.query_conditioned:
                fpn_expanded = tuple(
                    torch.repeat_interleave(
                        fpn_layer, text_size, dim=0
                    )
                    for fpn_layer in fpn
                )
            else:
                fpn_expanded = fpn
            
            gt_contrastive_loss = self.gt_contrastive_loss(
                fpn_expanded, fpn_masks_split, gt_labels_split, gt_labels_span
            ) / self.loss_norm * get_world_size()

            # not masking entire gt span
            # gt_contrastive_loss = self.gt_contrastive_loss(
            #     fpn, fpn_masks_split, gt_labels_split, gt_labels_split
            # ) / self.loss_norm * get_world_size()
        else:
            gt_contrastive_loss = torch.tensor(0.0).cuda()

        # calculate point loss
        ## (1) loss norm
        pos_masks = torch.logical_and(gt_labels, fpn_masks)
        norm = pos_masks.sum()

        ## (2) classification loss on valid points
        cls_loss = self._calc_focal_loss(
            logits=fpn_logits[fpn_masks],
            labels=cls_targets[fpn_masks]
        ) / self.loss_norm * get_world_size()
        if self.use_mst:
            fpn_logits2 = torch.cat(fpn_logits2, dim=1) # (bs, p)
            cls_loss2 = self._calc_focal_loss(
                logits=fpn_logits2[fpn_masks], labels=gt_labels[fpn_masks]
            ) / self.loss_norm * get_world_size()
            cls_loss = (cls_loss + cls_loss2) / 2
        
        ## (3) regression loss on positive points
        reg_loss = self._calc_iou_loss(
            pred_offsets=fpn_offsets[pos_masks], gt_offsets=gt_offsets[pos_masks]
        ) / self.loss_norm * get_world_size()
        cross_level_weight = self.cross_level_consistency_weight
        if self.cross_level_warmup_epochs > 0:
            cross_level_weight *= min(
                1.0, float(self.epoch) / self.cross_level_warmup_epochs
            )

        # TEFM evidence loss (HM-TEFM-038)
        evidence_loss = zero_aux_loss
        _mref = self.model.module if hasattr(self.model, 'module') else self.model
        _ev = getattr(_mref, 'last_evidence_logits', None)
        _lam = self._current_tefm_lambda()
        if _ev is not None and _lam > 0:
            ev_logits = torch.cat(_ev, dim=1)        # (bs, p_total)
            # plain center-in-GT target (no center_sampling / range)
            _c = points[..., 0]                       # (bs, p_total)
            ev_target = ((_c >= targets[:, 0:1]) &
                         (_c <= targets[:, 1:2])).float()
            evidence_loss = torch.nn.functional \
                .binary_cross_entropy_with_logits(
                    ev_logits[fpn_masks], ev_target[fpn_masks]
                ) * _lam

        total_loss = cls_loss + self.loss_weight * reg_loss + \
            self.ds_contrastive_weight * ds_contrastive_loss + \
            evidence_loss + \
            self.gt_contrastive_weight * gt_contrastive_loss
        if self.cross_level_consistency:
            total_loss = total_loss + cross_level_weight * cross_level_loss
        if self.query_boundary_importance:
            total_loss = total_loss + (
                self.query_boundary_importance_weight * importance_loss
            )
        if self.boundary_supervision:
            total_loss = total_loss + (
                self.boundary_loss_weight * boundary_loss
            )
        if self.allocation_supervision:
            total_loss = total_loss + (
                self.allocation_supervision_weight * alloc_loss
            )
        if self.ranking_supervision:
            total_loss = total_loss + (
                self.ranking_supervision_weight * rank_loss
            )
        if self.listwise_ranking:
            total_loss = total_loss + (
                float(self.opt['train']['loss_aux']['listwise_ranking']
                      .get('weight', 0.1)) * list_loss
            )
        if self.span_denoising:
            total_loss = total_loss + (
                float(self.opt['train']['loss_aux']['span_denoising']
                      .get('weight', 0.1)) * dn_loss
            )
        if qstb_cfg.get('enable', False):
            total_loss = total_loss + (
                float(qstb_cfg.get('weight', 0.5)) * trans_loss
            )
        total_loss.backward()

        return {
            'cls': cls_loss.detach(),
            'reg': reg_loss.detach(),
            'total': total_loss.detach(),
            'norm': norm.detach(),
            'ds_contrast': ds_contrastive_loss.detach(),
            'gt_contrast': gt_contrastive_loss.detach(),
            'cross_level': (
                cross_level_loss.detach()
                if cross_level_loss is not None else cls_loss.detach() * 0.0
            ),
            'importance_loss': importance_loss.detach(),
            'alloc_loss': alloc_loss.detach(),
            'rank_loss': rank_loss.detach(),
            'list_loss': list_loss.detach(),
            'dn_loss': dn_loss.detach(),
            'trans_loss': trans_loss.detach(),
            'importance_relevance': importance_components['relevance'].detach(),
            'importance_boundary': importance_components['boundary'].detach(),
            'importance_change': importance_components['temporal_change'].detach(),
            'importance_final': importance_components['importance'].detach(),
            'boundary_loss': boundary_loss.detach(),
            'boundary_start': boundary_components['start'].detach(),
            'boundary_end': boundary_components['end'].detach(),
        }

    def log(self):
        t = len(str(self.num_itrs))
        log_str = f"[{self.itr:0{t}d}/{self.num_itrs:0{t}d}] "
        for k, v in self.loss_meters.items():
            if k in (
                'ds_contrast',
                'gt_contrast',
                'cross_level',
                'importance_loss',
                'importance_relevance',
                'importance_boundary',
                'importance_change',
                'importance_final',
                'boundary_loss',
                'boundary_start',
                'boundary_end',
            ):
                log_str += f"{k} {float(v.item()):.6f} | "
            else:
                log_str += f"{k} {v.item():.3f} | "
            self.tb_writer.add_scalar(k, v.item(), self.itr)
            v.reset()
        lr = self.scheduler.get_last_lr()[0]
        self.tb_writer.add_scalar('lr', lr, self.itr)
        log_str += time_str(self.timer.item() * self.log_interval)
        self.timer.reset()
        self.logger.write(log_str)
        self.tb_writer.flush()




class EvaluatorOriginal:

    def __init__(self, opt):

        self.opt = opt

        # set random seed
        rng = fix_random_seed(opt.get('seed', 2022))

        # prepare dataset
        dataset = make_dataset(opt['eval']['data'], is_training=False)
        self.dataloader, _ = make_dataloader(
            dataset, is_training=False, generator=rng, batch_size=1, num_workers=0
        )
        self.num_itrs = len(self.dataloader)
        self.itr = self.text_cnt = 0

        # load model
        # self.model = PtTransformer(opt['model']).cuda()
        self.model = make_models_net(opt).cuda()
        self.load_model()
        self.model.eval().requires_grad_(False)
        self.adaptive_anchor = bool(opt['model']['vid_net'].get('adaptive_anchor', False))
        self.pt_gen = PtGenerator(
            **opt['pt_gen'], allow_fixed_stride=not self.adaptive_anchor
        ).cuda()
        if self.adaptive_anchor:
            self.adaptive_pt_gen = AdaptiveTemporalPointGenerator(
                max_seq_len=opt['pt_gen']['max_seq_len'],
                num_fpn_levels=opt['pt_gen']['num_fpn_levels'],
                regression_range=opt['pt_gen'].get('regression_range', 4),
                sigma=opt['pt_gen'].get('sigma', 1),
                input_stride=opt['model']['vid_net'].get('stride', 1),
            ).cuda()
        else:
            self.adaptive_pt_gen = None

        # build logging utilities
        self.log_interval = self.num_itrs // 10
        self.logger = Logger(os.path.join(opt['_root'], f"eval_{opt['_ckpt']}.txt"))
        
        # initialize prediction storage
        self.predictions = {}

        # register model hyperparameters
        self.max_vid_len = opt['model']['max_vid_len']
        self.vid_stride = opt['model'].get('vid_stride', 1)
        self.input_vid_len = self.max_vid_len * self.vid_stride

        num_fpn_levels = opt['model']['num_fpn_levels']
        mha_win_size = opt['model']['mha_win_size']
        ds_strides = [2 ** i for i in range(num_fpn_levels)]
        min_chunk_size = 1
        for idx in range(num_fpn_levels):
            stride = ds_strides[idx]
            if mha_win_size > 0:
                stride *= (mha_win_size // 2) * 2
            min_chunk_size = max(min_chunk_size, stride)
        assert self.max_vid_len % min_chunk_size == 0, (
            f"max video length must be a multiple of {min_chunk_size}"
        )
        self.min_chunk_size = min_chunk_size

        # register evaluation hyperparameters
        self.ranks = opt['eval'].get('ranks', (1, 5))
        self.topk = max(self.ranks)
        self.iou_threshs = np.array(opt['eval'].get('iou_threshs', (0.3, 0.5)))
        self.counts = np.zeros((len(self.ranks), len(self.iou_threshs)))

        self.window_size = opt['eval'].get('window_size')
        self.window_stride = opt['eval'].get('window_stride')

        self.batched_nms = lambda segs, scores: batched_nms(
            segs, scores, **opt['eval']['nms']
        )
        self.pre_nms_topk = opt['eval']['pre_nms_topk']
        self.export_raw_candidates = opt['eval'].get(
            'export_raw_candidates', False
        )
        self.raw_topk = opt['eval'].get('raw_topk', 200)
        self._last_raw_candidates = None
        self.pre_nms_thresh = opt['eval']['pre_nms_thresh']
        self.seg_len_thresh = opt['eval']['seg_len_thresh']
        self.max_text_len = opt['eval'].get('max_text_len', 24)
        self.batchify_text_queries = opt['eval'].get('batchify_text_queries', True)
        self.text_batch_size = opt['eval'].get('text_batch_size', 0)
        if self.batchify_text_queries:
            print("Batchify text queries for evaluation")
        else:
            print("Single text query processing for evaluation")

    def load_model(self):
        filename = os.path.join(
            self.opt['_root'], 'models', f"{self.opt['_ckpt']}.pth"
        )
        ckpt = torch.load(filename, map_location='cpu', weights_only=False)
        self._load_model_state(self.model, ckpt['model_ema'])
        print0(f"Loaded checkpoint [epoch {self.opt['_ckpt']}]...")

    @torch.no_grad()
    def run(self):
        print0("Evaluation started.")
        start_time = time.time()
        for data_list in self.dataloader:
            data = data_list[0]
            results = self.predict(data)
            targets = data['segment']
            vid_id = data['vid_id']
            assert len(results) == len(targets)

            # Store predictions for this video
            if vid_id not in self.predictions:
                self.predictions[vid_id] = {
                    'queries': [],
                    'recall_at_iou': {}
                }

            video_iou_counts = np.zeros((len(self.ranks), len(self.iou_threshs)))
            
            for query_idx, (result, target) in enumerate(zip(results, targets)):
                segs, scores = result['segments'], result['scores']
                idx = scores.argsort(descending=True)
                segs, scores = segs[idx[:self.topk]], scores[idx[:self.topk]]
                target = torch.as_tensor(target, dtype=torch.float)
                target = target.expand(len(segs), -1)
                
                iou_topk = iou(segs, target)
                iou_n = np.array([iou_topk[:i].max().item() for i in self.ranks])
                self.counts += (iou_n[:, None] >= self.iou_threshs[None])
                video_iou_counts += (iou_n[:, None] >= self.iou_threshs[None])
                
                # Store query predictions (top-5 predictions)
                top5_segs = segs[:5] if len(segs) >= 5 else segs
                top5_scores = scores[:5] if len(scores) >= 5 else scores
                
                query_data = {
                    'query_id': query_idx,
                    'ground_truth': target[0].cpu().numpy().tolist(),
                    'predictions': []
                }
                
                for seg, score in zip(top5_segs, top5_scores):
                    query_data['predictions'].append({
                        'segment': seg.cpu().numpy().tolist(),
                        'score': score.item()
                    })
                if 'raw_candidates' in result:
                    query_data['raw_candidates'] = result['raw_candidates']
                
                self.predictions[vid_id]['queries'].append(query_data)
            
            # Calculate recall at IoU for this video
            video_metrics = video_iou_counts / len(targets)
            for i, rank in enumerate(self.ranks):
                for j, thresh in enumerate(self.iou_threshs):
                    key = f"Rank@{rank}_IoU@{thresh:.1f}"
                    self.predictions[vid_id]['recall_at_iou'][key] = video_metrics[i, j].item()
            
            self.text_cnt += len(targets)
            self.itr += 1

            if self.itr == 1 or self.itr % self.log_interval == 0:
                self.log()
        end_time = time.time()
        self.log(is_last=True)
        completion_msg = f"Evaluation completed in {time_str(time.time() - start_time)}."
        print0(completion_msg)
        self.logger.write(completion_msg)
        
        # Save predictions to JSON file
        self.save_predictions()

    def predict(self, data):
        """ Predict event segments given a single video and an arbitrary
        number of text queries. This function assumes single-GPU evaluation.
        """
        # parse text
        tokens = data['text'] # all text queries for the single video
        if not isinstance(tokens, tuple):
            tokens = (tokens, )
        # parse video
        vid = data['vid']
        vid_len = vid.size(-1)
        with torch.no_grad():
            if self.batchify_text_queries:
                text, text_masks, text_size = self._batchify_text2(
                    text_list=[tokens]
                )
                text = text.cuda(non_blocking=True) # (bs, num_queries, c_t, t)
                text_masks = text_masks.cuda(non_blocking=True) # (bs, num_queries, t)
                text_size = text_size.cuda(non_blocking=True)

                # batched_text_encoded: (num_queries, c_t, t), batched_text_mask_encoded: (num_queries, 1,t)
                text, text_masks = self.model.encode_text2(text, text_masks, text_size)
            else:
                text_list, text_mask_list = tuple(), tuple()
                for text in tokens:
                    text = text[None]
                    text_mask = text.new_full(
                        (1, 1, text.size(-1)), 1, dtype=torch.bool
                    )
                    text = text.cuda(non_blocking=True)
                    text_mask = text_mask.cuda(non_blocking=True)

                    text, text_mask = self.model.encode_text(text, text_mask)
                    text_list += (text, )
                    text_mask_list += (text_mask, )
        # external scores (n, t)
        ext_scores = data['ext_scores']
        if ext_scores is not None and ext_scores.ndim == 1:
            ext_scores = ext_scores[None]

        # sliding-window evaluation
        window_size = min(self.window_size or vid_len, vid_len)
        window_stride = self.window_stride or window_size

        n = vid_len - window_size
        windows, window_offsets, window_ext_scores = tuple(), tuple(), tuple()
        
        idx = 0
        while idx <= n:
            windows += (vid[..., idx:idx + window_size], )
            window_offsets += (idx, )
            if ext_scores is not None:
                window_ext_scores += (ext_scores[..., idx:idx + window_size], )
            else:
                window_ext_scores += (None, )
            idx += window_stride
        
        if n > 0 and n % window_stride > 0:
            # backpad last window
            windows += (vid[..., -window_size:], )
            window_offsets += (n, )
            if ext_scores is not None:
                window_ext_scores += (ext_scores[..., -window_size:], )
            else:
                window_ext_scores += (None, )

        # Calculate adaptive input_vid_len based on actual window size
        # This ensures we use the minimum padding needed for the FPN constraints
        stride = self.min_chunk_size * self.vid_stride
        input_vid_len = (window_size + (stride - 1)) // stride * stride

        segs_list, scores_list = tuple(), tuple()
        levels_list = tuple() if self.export_raw_candidates else None
        for window, window_offset, window_ext in \
            zip(windows, window_offsets, window_ext_scores):
            window = F.pad(window, (0, input_vid_len - window_size))[None]
            window_mask = torch.arange(input_vid_len).view(1, 1, -1) < window_size
            window = window.cuda(non_blocking=True)
            window_mask = window_mask.cuda(non_blocking=True)
            if window_ext is not None:
                window_ext = F.pad(window_ext, (0, input_vid_len - window_size))
                window_ext = window_ext.cuda(non_blocking=True)
            
            with torch.no_grad():
                fpn, fpn_masks = self.model.encode_video(window, window_mask)
                fpn_logits_list, fpn_offsets_list = tuple(), tuple()
                if self.batchify_text_queries:
                    fpn_logits, fpn_offsets, _ = self.model.fuse_and_predict(fpn, fpn_masks, text, text_masks, text_size)
                    for query_idx in range(len(tokens)):
                        # Extract this query's results from each layer
                        query_logits = tuple(layer_tensor[query_idx:query_idx+1] for layer_tensor in fpn_logits)
                        query_offsets = tuple(layer_tensor[query_idx:query_idx+1] for layer_tensor in fpn_offsets)
                        fpn_logits_list += (query_logits,)
                        fpn_offsets_list += (query_offsets,)
                else:
                    for text, text_mask in zip(text_list, text_mask_list):
                        fpn_logits, fpn_offsets, _ = \
                            self.model.fuse_and_predict(fpn, fpn_masks, text, text_mask)
                        fpn_logits_list += (fpn_logits, )
                        fpn_offsets_list += (fpn_offsets, )

            fpn_n_points = [m.size(-1) for m in fpn_masks]
            if self.adaptive_anchor:
                model_ref = self.model.module if hasattr(self.model, 'module') else self.model
                fpn_points = self.adaptive_pt_gen(
                    model_ref.vid_net.last_temporal_metadata, fpn_masks
                )
            else:
                fpn_points = self.pt_gen(fpn_n_points)
            fpn_masks = [m.squeeze(1) for m in fpn_masks]

            # collect segments and their scores
            window_segs_list, window_scores_list = tuple(), tuple()
            window_levels_list = tuple() if self.export_raw_candidates else None
            for idx, (fpn_logits, fpn_offsets) in \
                enumerate(zip(fpn_logits_list, fpn_offsets_list)):
                collected = self._collect_segments(
                    self._points_for_query(fpn_points, idx), fpn_logits, fpn_offsets, tuple(mask[idx:idx + 1] if mask.size(0) > 1 else mask for mask in fpn_masks), 
                    window_ext[idx] if window_ext is not None else None,
                    return_levels=self.export_raw_candidates,
                    fpn_feats=(fpn if getattr(self, 'rank_head_mode', 'off')
                               != 'off' else None),
                    query_idx=idx,
                )
                if self.export_raw_candidates:
                    window_segs, window_scores, window_levels = collected
                else:
                    window_segs, window_scores = collected
                window_segs += (window_offset if self.adaptive_anchor else window_offset / self.vid_stride)
                window_segs_list += (window_segs.cpu(), )
                window_scores_list += (window_scores.cpu(), )
                if self.export_raw_candidates:
                    window_levels_list += (window_levels.cpu(), )

            segs_list += (window_segs_list, )
            scores_list += (window_scores_list, )
            if self.export_raw_candidates:
                levels_list += (window_levels_list, )

        segs_list = [torch.cat(x) for x in zip(*segs_list)]     # [bs x (n, 2)]
        if self.export_raw_candidates:
            levels_list = [torch.cat(x) for x in zip(*levels_list)]
        scores_list = [torch.cat(x) for x in zip(*scores_list)] # [bs x (n,)]

        results = tuple()
        for query_idx, (segs, scores) in enumerate(zip(segs_list, scores_list)):
            raw_candidates = None
            if self.export_raw_candidates:
                levels = levels_list[query_idx]
                raw_n_topk = min(len(segs), self.raw_topk)
                raw_idx = scores.argsort(descending=True)[:raw_n_topk]
                raw_candidates = self._format_raw_candidates(
                    segs[raw_idx],
                    scores[raw_idx],
                    levels[raw_idx],
                    data,
                )

            # only keep top-k scoring boxes
            n_topk = min(len(segs), self.pre_nms_topk)
            idx = scores.argsort(descending=True)[:n_topk]

            # NMS
            segs, scores = self.batched_nms(segs[idx], scores[idx])

            # convert segments to timestamps in seconds
            if len(segs) > 0:
                clip_stride = data['clip_stride']
                clip_size = data['clip_size']
                fps = data['fps']
                duration = data['duration']

                if not self.adaptive_anchor:
                    segs *= self.vid_stride
                segs = (segs * clip_stride + 0.5 * clip_size) / fps
                segs = torch.clamp(segs, min=0, max=duration)

            result = {'segments': segs, 'scores': scores}
            if self.export_raw_candidates:
                result['raw_candidates'] = raw_candidates
            results += (result, )

        return results

    def _batchify_text(self, text_list):
        """
        Put text features and their masks in a batch.

        Args:
            text_list (List[float tensor, (c2, t2)]): token features.

        Returns:
            text (float tensor, (bs, c2, t2)): token feature sequences.
            text_masks (bool tensor, (bs, t2)): token masks.
        """
        bs = len(text_list)
        text_dim = text_list[0].size(0)
        text_lens = [min(t.size(-1), self.max_text_len) for t in text_list]
        text = text_list[0].new_full((bs, text_dim, self.max_text_len), 0.)
        for idx in range(bs):
            text[idx, :, :text_lens[idx]].copy_(
                text_list[idx][..., :text_lens[idx]]
            )
        text_lens = torch.as_tensor(text_lens)[:, None]
        text_masks = torch.arange(self.max_text_len)[None] < text_lens
        return text, text_masks
    
    def _batchify_text2(self, text_list):
        bs = len(text_list)

        # batch text
        if isinstance(text_list[0], tuple):
            # many text queries are associated with the same video
            b_text, b_text_masks = tuple(), tuple()
            n = tuple()
            for t in text_list:
                b_t, b_tm = self._batchify_text(t)
                b_text += (b_t, )
                b_text_masks += (b_tm, )
                n += (len(t), )
            n_max = max(n)      # max number of text queries

            # (bs, n, c, t)
            text_dim = b_text[0].size(1)
            text = b_text[0].new_full(
                (bs, n_max, text_dim, self.max_text_len), 0.
            )
            for idx in range(bs):
                text[idx, :n[idx]].copy_(b_text[idx])

            # (bs, n, t)
            text_masks = b_text_masks[0].new_full(
                (bs, n_max, self.max_text_len), 0, dtype=torch.bool
            )
            for idx in range(bs):
                text_masks[idx, :n[idx]].copy_(b_text_masks[idx])
        else:
            n = bs * (1, )
            text, text_masks = self._batchify_text(text_list)

        text_size = torch.as_tensor(n)

        # vid: (bs, c1, t1)
        # vid_masks: (bs, t1)
        # text: (bs, (n,) c2, t2)
        # text_masks (bs, (n,) t2)
        # text_size: (bs,)
        return text, text_masks, text_size
    
    def _format_raw_candidates(self, segs, scores, levels, data):
        if len(segs) == 0:
            return []
        clip_stride = data['clip_stride']
        clip_size = data['clip_size']
        fps = data['fps']
        duration = data['duration']
        raw_segs = segs.clone()
        if not self.adaptive_anchor:
            raw_segs *= self.vid_stride
        raw_segs = (raw_segs * clip_stride + 0.5 * clip_size) / fps
        raw_segs = torch.clamp(raw_segs, min=0, max=duration)
        return [
            {
                'segment': seg.cpu().numpy().tolist(),
                'score': score.item(),
                'level': int(level.item()),
            }
            for seg, score, level in zip(
                raw_segs, scores, levels
            )
        ]

    @staticmethod
    def _points_for_query(fpn_points, query_index):
        return tuple(
            points[query_index] if points.ndim == 3 else points
            for points in fpn_points
        )

    def _collect_segments(
        self,
        fpn_points,     # List[(p, 4) or (B, p, 8) * #levels]
        fpn_logits,     # List[(1, p) * #levels]
        fpn_offsets,    # List[(1, p, 2) * #levels]
        fpn_masks,      # List[(1, p) * #levels]
        ext_scores,     # (p, )
        return_levels=False,
        fpn_feats=None,  # optional fused features for rank-head rerank
        query_idx=0,
    ):
        points_list, scores_list, offsets_list = tuple(), tuple(), tuple()
        levels_list = tuple() if return_levels else None

        # loop over all FPN levels
        base_ext_scores = ext_scores
        for level, (points, logits, offsets, masks) in enumerate(zip(
            fpn_points, fpn_logits, fpn_offsets, fpn_masks
        )):
            logits, offsets, masks = logits[0], offsets[0], masks[0]
            if points.ndim == 3:
                if points.size(0) != 1:
                    raise ValueError(
                        'segment collection expects points for one query'
                    )
                points = points[0]

            # compute point scores
            scores = torch.sigmoid(logits)
            if base_ext_scores is not None:
                if self.adaptive_anchor:
                    coordinates = points[:, 0].to(base_ext_scores.dtype)
                    left = coordinates.floor().long().clamp(
                        min=0, max=base_ext_scores.numel() - 1
                    )
                    right = (left + 1).clamp(
                        max=base_ext_scores.numel() - 1
                    )
                    alpha = (coordinates - left.to(coordinates.dtype)).clamp(
                        min=0, max=1
                    )
                    point_scores = (
                        base_ext_scores[left] * (1 - alpha)
                        + base_ext_scores[right] * alpha
                    )
                    scores *= point_scores
                else:
                    scores *= ext_scores
                    ext_scores = F.max_pool1d(
                        ext_scores[None, None],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    )[0, 0]
            scores *= masks.float()

            if fpn_feats is not None and getattr(
                    self, 'rank_head_mode', 'off') != 'off':
                # MomentRankHead rerank on the top-K points of this level
                # (vectorized span pooling; identical construction to the
                # listwise training objective).
                import math as _m
                K = min(100, int(scores.numel()))
                top_idx = scores.topk(K).indices
                from .modeling.temporal_coordinates import (
                    decode_offsets as _dec,
                )
                segs_k = _dec(points, offsets)[top_idx]        # (K, 2)
                centers = points[:, 0]
                lo = segs_k[:, 0].clamp_min(0)
                hi = segs_k[:, 1]
                span_mask = (
                    (centers[None, :] >= lo[:, None])
                    & (centers[None, :] <= hi[:, None])
                    & masks[None, :]
                )
                feats = fpn_feats[level][query_idx].float()    # (C, T)
                counts = span_mask.float().sum(-1).clamp_min(1.0)
                pooled = (
                    span_mask.float() @ feats.transpose(0, 1)
                ) / counts[:, None]                            # (K, C)
                dur = (hi - lo).clamp_min(1e-6)
                ref = float(points.size(0)) + 1.0
                geo = torch.stack((
                    torch.log(dur),
                    points[top_idx, 0] / ref,
                    lo / ref,
                    hi / ref,
                ), dim=-1)
                lvl_ids = torch.full((K,), level, dtype=torch.long,
                                     device=points.device)
                head_sc = torch.sigmoid(self.rank_head(
                    pooled, geo, lvl_ids))
                base_sc = scores[top_idx]
                blended = base_sc
                if self.rank_head_mode == 'blend':
                    blended = 0.5 * base_sc + 0.5 * head_sc
                elif self.rank_head_mode == 'head':
                    blended = head_sc
                scores = scores.clone()
                scores[top_idx] = blended

            # clean up predictions before NMS for efficiency
            ## (1) filter points by confidence threshold
            idx = scores > self.pre_nms_thresh
            points_list += (points[idx], )
            scores_list += (scores[idx], )
            if return_levels:
                levels_list += (
                    torch.full(
                        (int(idx.sum().item()), ), level,
                        dtype=torch.long, device=points.device
                    ),
                )
            offsets_list += (offsets[idx], )

        if not points_list:
            empty = fpn_logits[0].new_empty((0, 2))
            if return_levels:
                return empty, empty.new_empty((0,)), empty.new_empty((0,), dtype=torch.long)
            return empty, empty.new_empty((0,))
        points = torch.cat(points_list)
        scores = torch.cat(scores_list)
        offsets = torch.cat(offsets_list)
        if return_levels:
            levels = torch.cat(levels_list)

        ## (2) only keep top-k scoring boxes
        n_topk = min(len(points), self.pre_nms_topk)
        idx = scores.argsort(descending=True)[:n_topk]
        if return_levels:
            levels = levels[idx]
        points, scores, offsets = points[idx], scores[idx], offsets[idx]

        ## (3) assemble predicted segments
        segs = decode_offsets(points, offsets)


        ## (4) filter segments by length threshold
        seg_lens = segs[:, 1] - segs[:, 0]
        idx = seg_lens > self.seg_len_thresh
        if return_levels:
            levels = levels[idx]
        segs, scores = segs[idx], scores[idx]

        if return_levels:
            return segs, scores, levels
        return segs, scores

    def log(self, is_last=False):
        metrics = self.counts / self.text_cnt
        log_str = "\nFinal:" if is_last else f"\n[{self.itr}/{self.num_itrs}]"
        for i, rank in enumerate(self.ranks):
            log_str += "\n-----"
            for j, thresh in enumerate(self.iou_threshs):
                log_str += (
                    f"\nRank@{rank}, IoU@{thresh:.1f}: "
                    f"{(metrics[i, j] * 100):.2f}"
                )
        self.logger.write(log_str)
    
    def save_predictions(self):
        """Save predictions to JSON file"""
        predictions_file = os.path.join(self.opt['_root'], f"predictions_{self.opt['_ckpt']}.json")
        
        # Add overall metrics to the predictions
        overall_metrics = self.counts / self.text_cnt
        summary = {
            'overall_recall_at_iou': {},
            'total_queries': int(self.text_cnt),
            'total_videos': len(self.predictions)
        }
        
        for i, rank in enumerate(self.ranks):
            for j, thresh in enumerate(self.iou_threshs):
                key = f"Rank@{rank}_IoU@{thresh:.1f}"
                summary['overall_recall_at_iou'][key] = overall_metrics[i, j].item()
        
        output_data = {
            'summary': summary,
            'videos': self.predictions
        }
        
        with open(predictions_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print0(f"Predictions saved to {predictions_file}")
        self.logger.write(f"Predictions saved to {predictions_file}")






class EvaluatorAuxiliary(EvaluatorOriginal):
    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        import os as _os
        # Rank-head rerank gate: RANK_HEAD_MODE=blend|head|off (default off).
        self.rank_head_mode = _os.environ.get('RANK_HEAD_MODE', 'off')
        self.dn_refine = _os.environ.get('DN_REFINE', '0') != '0'
        self.dn_steps = int(_os.environ.get('DN_STEPS', '2'))
        self.moment_denoiser = getattr(self.model, 'moment_denoiser', None)
        self.rank_head = getattr(self.model, 'moment_rank_head', None)
        if self.rank_head_mode != 'off' and self.rank_head is None:
            raise RuntimeError(
                'RANK_HEAD_MODE set but model has no moment_rank_head')
        self.early_fusion = opt['model'].get('early_fusion', True)
        self.use_mst = opt['model'].get('use_mst', False)
        self.query_conditioned = bool(
            getattr(self.model, 'query_conditioned', False)
        )
        self.allocator_policy = getattr(self.model, 'allocator_policy', 'learned')
        if self.early_fusion:
            self.logger.write("Early fusion enabled")

    def _oracle_allocator_targets(
        self, data, window_offset, window_len, query_batch, query_index=None
    ):
        """Windowed GT segments used only to decide grouping resolution.

        This is the diagnostic A-O upper bound: GT never reaches cls/reg
        heads. Targets are in input-token grid units like training targets.
        """
        targets = data['target'].detach().float().cpu()
        if targets.ndim != 2 or targets.size(-1) != 2:
            raise ValueError(
                'oracle evaluation requires per-query GT target segments'
            )
        if query_index is not None:
            targets = targets[query_index:query_index + 1]
        if targets.size(0) != query_batch:
            raise ValueError(
                'oracle targets cover {} queries but the encoded video '
                'batch is {}'.format(targets.size(0), query_batch)
            )
        targets = targets - float(window_offset)
        return torch.clamp(targets, 0.0, float(window_len)).cuda()
        
    def predict(self, data):
        """ Predict event segments given a single video and an arbitrary
        number of text queries. This function assumes single-GPU evaluation.
        """
        # parse text
        tokens = data['text']
        if not isinstance(tokens, tuple):
            tokens = (tokens, )
        # parse video
        vid = data['vid']
        vid_len = vid.size(-1)
        with torch.no_grad():
            if self.batchify_text_queries:
                text, text_mask, text_size = self._batchify_text2(
                    text_list=[tokens]
                )
                text = text.cuda(non_blocking=True) # (bs, num_queries, c_t, t)
                text_mask = text_mask.cuda(non_blocking=True) # (bs, num_queries, t)
                text_size = text_size.cuda(non_blocking=True)

                # batched_text_encoded: (num_queries, c_t, t), batched_text_mask_encoded: (num_queries, 1,t)
                text, text_mask = self.model.encode_text2(text, text_mask, text_size)
            else:
                text_list, text_mask_list = tuple(), tuple()
                for text in tokens:
                    text = text[None]
                    text_mask = text.new_full(
                        (1, 1, text.size(-1)), 1, dtype=torch.bool
                    )
                    text = text.cuda(non_blocking=True)
                    text_mask = text_mask.cuda(non_blocking=True)

                    text, text_mask = self.model.encode_text(text, text_mask)
                    text_list += (text, )
                    text_mask_list += (text_mask, )


        # external scores (n, t)
        ext_scores = data['ext_scores']
        if ext_scores is not None and ext_scores.ndim == 1:
            ext_scores = ext_scores[None]

        # sliding-window evaluation
        window_size = min(self.window_size or vid_len, vid_len)
        window_stride = self.window_stride or window_size

        n = vid_len - window_size
        windows, window_offsets, window_ext_scores = tuple(), tuple(), tuple()
        
        idx = 0
        while idx <= n:
            windows += (vid[..., idx:idx + window_size], )
            window_offsets += (idx, )
            if ext_scores is not None:
                window_ext_scores += (ext_scores[..., idx:idx + window_size], )
            else:
                window_ext_scores += (None, )
            idx += window_stride
        
        if n > 0 and n % window_stride > 0:
            # backpad last window
            windows += (vid[..., -window_size:], )
            window_offsets += (n, )
            if ext_scores is not None:
                window_ext_scores += (ext_scores[..., -window_size:], )
            else:
                window_ext_scores += (None, )

        # Calculate adaptive input_vid_len based on actual window size
        # This ensures we use the minimum padding needed for the FPN constraints
        stride = self.min_chunk_size * self.vid_stride
        input_vid_len = (window_size + (stride - 1)) // stride * stride

        segs_list, scores_list = tuple(), tuple()
        levels_list = tuple() if self.export_raw_candidates else None
        for window, window_offset, window_ext in \
            zip(windows, window_offsets, window_ext_scores):
            window = F.pad(window, (0, input_vid_len - window_size))[None]
            window_mask = torch.arange(input_vid_len).view(1, 1, -1) < window_size
            window = window.cuda(non_blocking=True)
            window_mask = window_mask.cuda(non_blocking=True)
            if window_ext is not None:
                window_ext = F.pad(window_ext, (0, input_vid_len - window_size))
                window_ext = window_ext.cuda(non_blocking=True)



            fpn_logits_list, fpn_offsets_list = tuple(), tuple()
            query_temporal_metadata = []
            with torch.no_grad():
                if self.batchify_text_queries:
                    if self.early_fusion:
                        window, window_mask = self.model.vid_proj(window, window_mask)
                        window, window_mask = self.model.fusion(window, window_mask, text, text_mask, text_size)
                    encode_kwargs = {}
                    if self.allocator_policy == 'oracle':
                        encode_kwargs['allocator_targets'] = (
                            self._oracle_allocator_targets(
                                data,
                                window_offset,
                                input_vid_len,
                                int(text_size.sum().item()),
                            )
                        )
                    fpn, fpn_masks, _, _ = self.model.encode_video(
                        window,
                        window_mask,
                        query_feat=text if self.query_conditioned else None,
                        query_mask=(
                            text_mask if self.query_conditioned else None
                        ),
                        text_size=text_size,
                        **encode_kwargs
                    )
                    
                    if self.use_mst:
                        fpn_logits, fpn_logits2, fpn_offsets, _ = \
                            self.model.fuse_and_predict_mst(fpn, fpn_masks, text, text_mask, text_size)
                    else:
                        fpn_logits, _, fpn_offsets, _ = \
                            self.model.fuse_and_predict(fpn, fpn_masks, text, text_mask, text_size)

                    for query_idx in range(len(tokens)):
                        # Extract this query's results from each layer
                        query_logits = tuple(layer_tensor[query_idx:query_idx+1] for layer_tensor in fpn_logits)
                        # query_logits = tuple((fl1[query_idx:query_idx+1] + fl2[query_idx:query_idx+1]) / 2 for fl1, fl2 in zip(fpn_logits, fpn_logits2))
                        # query_logits = tuple(torch.maximum(fl1[query_idx:query_idx+1], fl2[query_idx:query_idx+1]) for fl1, fl2 in zip(fpn_logits, fpn_logits2))
                        query_offsets = tuple(layer_tensor[query_idx:query_idx+1] for layer_tensor in fpn_offsets)
                        fpn_logits_list += (query_logits,)
                        fpn_offsets_list += (query_offsets,)
                else:
                    window_orig = window.clone()
                    window_mask_orig = window_mask.clone()
                    for query_idx, (text, text_mask) in enumerate(
                        zip(text_list, text_mask_list)
                    ):
                        # window: (1, dim, T)
                        # window_mask: (1, 1, T)
                        if self.early_fusion:
                            window, window_mask = self.model.vid_proj(window_orig, window_mask_orig)
                            window, window_mask = self.model.fusion(window, window_mask, text, text_mask)
                        encode_kwargs = {}
                        if self.allocator_policy == 'oracle':
                            encode_kwargs['allocator_targets'] = (
                                self._oracle_allocator_targets(
                                    data,
                                    window_offset,
                                    input_vid_len,
                                    1,
                                    query_index=query_idx,
                                )
                            )
                        fpn, fpn_masks, _, _ = self.model.encode_video(
                            window,
                            window_mask,
                            query_feat=(
                                text if self.query_conditioned else None
                            ),
                            query_mask=(
                                text_mask if self.query_conditioned else None
                            ),
                            **encode_kwargs
                        )
                        model_ref = self.model.module if hasattr(self.model, 'module') else self.model
                        query_temporal_metadata.append(
                            model_ref.vid_net.last_temporal_metadata
                        )
                        if self.use_mst:
                            fpn_logits, fpn_logits2, fpn_offsets, _ = \
                                self.model.fuse_and_predict_mst(fpn, fpn_masks, text, text_mask)
                        else:
                            fpn_logits, _, fpn_offsets, _ = \
                                self.model.fuse_and_predict(fpn, fpn_masks, text, text_mask)
                        fpn_logits_list += (fpn_logits, )
                        fpn_offsets_list += (fpn_offsets, )
            # fpn, fpn_masks = self.model.encode_video(window, window_mask)
            fpn_n_points = [m.size(-1) for m in fpn_masks]
            if self.adaptive_anchor:
                model_ref = self.model.module if hasattr(self.model, 'module') else self.model
                if self.batchify_text_queries:
                    fpn_points = self.adaptive_pt_gen(
                        model_ref.vid_net.last_temporal_metadata, fpn_masks
                    )
                else:
                    fpn_points = tuple(
                        tuple(
                            self.adaptive_pt_gen(metadata, fpn_masks)[level]
                            for level in range(len(fpn_masks))
                        )
                        for metadata in query_temporal_metadata
                    )
            else:
                fpn_points = self.pt_gen(fpn_n_points)
            fpn_masks = [m.squeeze(1) for m in fpn_masks]

            # collect segments and their scores
            window_levels_list = tuple() if self.export_raw_candidates else None
            window_segs_list, window_scores_list = tuple(), tuple()
            for idx, (fpn_logits, fpn_offsets) in \
                enumerate(zip(fpn_logits_list, fpn_offsets_list)):
                collected = self._collect_segments(
                    (fpn_points[idx] if self.adaptive_anchor and not self.batchify_text_queries else self._points_for_query(fpn_points, idx)), fpn_logits, fpn_offsets, tuple(mask[idx:idx + 1] if mask.size(0) > 1 else mask for mask in fpn_masks), 
                    window_ext[idx] if window_ext is not None else None,
                    return_levels=self.export_raw_candidates,
                    fpn_feats=(fpn if getattr(self, 'rank_head_mode', 'off')
                               != 'off' else None),
                    query_idx=idx,
                )
                if self.export_raw_candidates:
                    window_segs, window_scores, window_levels = collected
                else:
                    window_segs, window_scores = collected
                window_segs += (window_offset if self.adaptive_anchor else window_offset / self.vid_stride)
                window_segs_list += (window_segs.cpu(), )
                window_scores_list += (window_scores.cpu(), )
                if self.export_raw_candidates:
                    window_levels_list += (window_levels.cpu(), )

            segs_list += (window_segs_list, )
            scores_list += (window_scores_list, )
            if self.export_raw_candidates:
                levels_list += (window_levels_list, )

        segs_list = [torch.cat(x) for x in zip(*segs_list)]     # [bs x (n, 2)]
        scores_list = [torch.cat(x) for x in zip(*scores_list)] # [bs x (n,)]
        if self.export_raw_candidates:
            levels_list = [torch.cat(x) for x in zip(*levels_list)]

        results = tuple()
        for query_idx, (segs, scores) in enumerate(zip(segs_list, scores_list)):
            raw_candidates = None
            if self.export_raw_candidates:
                levels = levels_list[query_idx]
                raw_n_topk = min(len(segs), self.raw_topk)
                raw_idx = scores.argsort(descending=True)[:raw_n_topk]
                raw_candidates = self._format_raw_candidates(
                    segs[raw_idx],
                    scores[raw_idx],
                    levels[raw_idx],
                    data,
                )

            # only keep top-k scoring boxes
            n_topk = min(len(segs), self.pre_nms_topk)
            idx = scores.argsort(descending=True)[:n_topk]

            # NMS
            segs, scores = self.batched_nms(segs[idx], scores[idx])

            # convert segments to timestamps in seconds
            if len(segs) > 0:
                clip_stride = data['clip_stride']
                clip_size = data['clip_size']
                fps = data['fps']
                duration = data['duration']

                if not self.adaptive_anchor:
                    segs *= self.vid_stride
                segs = (segs * clip_stride + 0.5 * clip_size) / fps
                segs = torch.clamp(segs, min=0, max=duration)

            result = {'segments': segs, 'scores': scores}
            if self.export_raw_candidates:
                result['raw_candidates'] = raw_candidates
            results += (result, )

        return results
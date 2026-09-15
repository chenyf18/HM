import warnings

import torch
import torch.nn as nn

from .fusion import make_fusion
from .head import make_head
from .text_net import make_text_net
from .video_net import make_video_net
from .blocks import MaskedConv1D, masked_max_pool1d
from .query_mamba_adapter import QueryConditionedFPNMambaAdapter
from copy import deepcopy

models = dict()
def register_models_net(name):
    def decorator(module):
        models[name] = module
        return module
    return decorator

@register_models_net('pt_transformer')
class PtTransformer(nn.Module):
    """
    Transformer based model for single-stage sentence grounding
    """
    def __init__(self, opt):
        super().__init__()

        # backbones
        self.text_net = make_text_net(opt['text_net'])
        self.vid_net = make_video_net(opt['vid_net'])
        vid_net_param_trainable_count = sum(p.numel() for p in self.vid_net.parameters() if p.requires_grad)
        print(f"vid_net trainable parameter count: {vid_net_param_trainable_count}")
        vid_net_param_count = sum(p.numel() for p in self.vid_net.parameters())
        print(f"vid_net parameter count: {vid_net_param_count}")
        # fusion and prediction heads
        self.fusion = make_fusion(opt['fusion'])
        self.cls_head = make_head(opt['cls_head'])
        self.reg_head = make_head(opt['reg_head'])

    def encode_text(self, tokens, token_masks):
        text, text_masks = self.text_net(tokens, token_masks)
        return text, text_masks

    def encode_video(self, vid, vid_masks):
        fpn, fpn_masks = self.vid_net(vid, vid_masks)
        return fpn, fpn_masks

    def fuse_and_predict(self, fpn, fpn_masks, text, text_masks, text_size=None):
        fpn, fpn_masks = self.fusion(fpn, fpn_masks, text, text_masks, text_size)
        fpn_logits, _ = self.cls_head(fpn, fpn_masks)
        fpn_offsets, fpn_masks = self.reg_head(fpn, fpn_masks)
        return fpn_logits, fpn_offsets, fpn_masks
    
    def encode_text2(self, text, text_masks, text_size):
        # pack text features
        if text.ndim == 4:
            text = torch.cat([t[:k] for t, k in zip(text, text_size)])
        if text_masks.ndim == 3:
            text_masks = torch.cat(
                [t[:k] for t, k in zip(text_masks, text_size)]
            )
        text, text_masks = self.encode_text(text, text_masks)
        return text, text_masks
    
    def forward(self, vid, vid_masks, text, text_masks, text_size=None):
        # pack text features
        if text.ndim == 4:
            text = torch.cat([t[:k] for t, k in zip(text, text_size)])
        if text_masks.ndim == 3:
            text_masks = torch.cat(
                [t[:k] for t, k in zip(text_masks, text_size)]
            )
        
        text, text_masks = self.encode_text(text, text_masks)
        fpn, fpn_masks = self.encode_video(vid, vid_masks)
        fpn_logits, fpn_offsets, fpn_masks = \
            self.fuse_and_predict(fpn, fpn_masks, text, text_masks, text_size)

        return fpn_logits, fpn_offsets, fpn_masks


@register_models_net('hieramamba')
class HieraMamba(nn.Module):
    """
    HieraMamba: Hierarchical Mamba model for video temporal grounding
    """
    _QUERY_PARAMETER_MARKERS = (
        '.query_modulation_mlp.',
        '.gate1_query.',
        '.gate2_query.',
        '.importance_predictor.',
        '.coarse_to_fine_refiner.',
        '.qsm_modulator.',
        '.cdf_neck.',
        '.film.',
        'tefm.',
        'evidence_head.',
    )

    def __init__(self, opt):
        super().__init__()
        self.early_fusion = opt.get('early_fusion', True)
        self.opt = opt
        # backbones
        self.text_net = make_text_net(opt['text_net'])
        vid_net_opt = opt['vid_net']
        self.query_conditioned = any(
            bool(vid_net_opt.get(key, False))
            for key in (
                'query_modulation',
                'query_aware_gate',
                'query_boundary_importance',
                'adaptive_anchor',
                'coarse_to_fine_refine',
            )
        )
        if self.query_conditioned and not vid_net_opt.get('query_dim'):
            query_dim = opt['text_net'].get('embd_dim')
            if query_dim is None:
                query_dim = opt['text_net']['in_dim']
            vid_net_opt['query_dim'] = query_dim
        if self.early_fusion:
            vid_in_dim = opt['vid_net']['in_dim']
            vid_embd_dim = opt['vid_net']['embd_dim']
            opt['vid_net']['in_dim'] = vid_embd_dim
            self.vid_net = make_video_net(opt['vid_net'])
            self.vid_proj = MaskedConv1D(vid_in_dim, vid_embd_dim, 1)
        else:
            self.vid_net = make_video_net(opt['vid_net'])
        self.query_conditioned = bool(
            getattr(self.vid_net, 'query_conditioned', False)
        )
        self.importance_enabled = bool(
            getattr(self.vid_net, 'importance_enabled', False)
        )
        # fusion and prediction heads
        self.fusion = make_fusion(opt['fusion'])
        self.cls_head = make_head(opt['cls_head'])
        self.reg_head = make_head(opt['reg_head'])

        cdf_opt = opt.get('cdf', {})
        self.cdf_enable = bool(cdf_opt.get('enable', False))
        if self.cdf_enable:
            from .cdf import CDFNeck
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(20260910)
                self.cdf_neck = CDFNeck(
                    dim=opt['vid_net']['embd_dim'],
                    feedback_children=tuple(
                        cdf_opt.get('levels', (0, 1, 2))),
                    mode=cdf_opt.get('mode', 'conservative'),
                )
        else:
            self.cdf_neck = None

        # TEFM (HM-TEFM-038): temporal evidence formation between
        # fusion and heads; zero-init residual => baseline identity
        tefm_opt = opt.get('tefm', {})
        self.tefm_enable = bool(tefm_opt.get('enable', False))
        if self.tefm_enable:
            from .tefm import TEFM, EvidenceHead
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(20260913)
                self.tefm = TEFM(dim=opt['vid_net']['embd_dim'])
                self.evidence_head = EvidenceHead(
                    dim=opt['vid_net']['embd_dim'])
        else:
            self.tefm = None
            self.evidence_head = None
        self.last_evidence_logits = None

        adapter_opt = deepcopy(opt.get('query_mamba_adapter', {}))
        adapter_enabled = adapter_opt.pop('enable', False)
        if adapter_enabled:
            adapter_opt.setdefault('d_model', opt['vid_net']['embd_dim'])
            adapter_opt.setdefault('text_dim', opt['text_net']['embd_dim'])
            adapter_opt.setdefault('num_levels', opt['vid_net']['arch'][-1])
            self.query_mamba_adapter = QueryConditionedFPNMambaAdapter(
                **adapter_opt
            )
        else:
            self.query_mamba_adapter = None

    def load_compatible_state_dict(self, state_dict):
        """Load released weights while initializing new query branches."""
        _ch = getattr(self, 'cls_head', None)
        _rh = getattr(self, 'reg_head', None)
        head_film = getattr(_ch, 'query_film', False) or \
            getattr(_rh, 'query_film', False)
        tefm_on = getattr(self, 'tefm', None) is not None
        if not self.query_conditioned and not head_film and not tefm_on:
            return self.load_state_dict(state_dict, strict=True)

        incompatible = self.load_state_dict(state_dict, strict=False)
        invalid_missing = [
            key for key in incompatible.missing_keys
            if not any(
                marker in key for marker in self._QUERY_PARAMETER_MARKERS
            )
        ]
        if invalid_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Incompatible checkpoint: missing={}, unexpected={}".format(
                    invalid_missing, incompatible.unexpected_keys
                )
            )
        if incompatible.missing_keys:
            warnings.warn(
                "Loaded a baseline HieraMamba checkpoint; newly added "
                "query-aware parameters keep their initialization.",
                RuntimeWarning,
            )
        return incompatible

    def project_video(self, vid, vid_masks):
        vid, vid_masks = self.vid_proj(vid, vid_masks)
        return vid, vid_masks

    def encode_text(self, tokens, token_masks):
        text, text_masks = self.text_net(tokens, token_masks)
        return text, text_masks

    def encode_text2(self, text, text_masks, text_size):
        # pack text features
        if text.ndim == 4:
            text = torch.cat([t[:k] for t, k in zip(text, text_size)])
        if text_masks.ndim == 3:
            text_masks = torch.cat(
                [t[:k] for t, k in zip(text_masks, text_size)]
            )
        text, text_masks = self.encode_text(text, text_masks)
        return text, text_masks

    def _align_video_query_batches(
        self, vid, vid_masks, query_feat, text_size
    ):
        if not self.query_conditioned or query_feat is None:
            return vid, vid_masks

        query_batch = query_feat.size(0)
        if vid.size(0) == query_batch:
            return vid, vid_masks
        if text_size is None:
            raise ValueError(
                "text_size is required when query and video batches differ"
            )

        repeats = torch.as_tensor(
            text_size, dtype=torch.long, device=vid.device
        ).reshape(-1)
        if repeats.numel() != vid.size(0):
            raise ValueError(
                "text_size has {} entries for video batch {}".format(
                    repeats.numel(), vid.size(0)
                )
            )
        if torch.any(repeats <= 0):
            raise ValueError("text_size entries must be positive")
        if int(repeats.sum().item()) != query_batch:
            raise ValueError(
                "text_size sums to {}, but query batch is {}".format(
                    int(repeats.sum().item()), query_batch
                )
            )

        vid = vid.repeat_interleave(repeats, dim=0)
        vid_masks = vid_masks.repeat_interleave(repeats, dim=0)
        return vid, vid_masks

    @property
    def allocator_policy(self):
        return getattr(self.vid_net, 'allocator_policy', 'learned')

    def encode_video(
        self,
        vid,
        vid_masks,
        query_feat=None,
        query_mask=None,
        text_size=None,
        return_importance_debug=False,
        return_anchor_assignments=False,
        allocator_targets=None,
        forced_cut_offsets=None,
        forced_target_counts=None,
    ):
        vid, vid_masks = self._align_video_query_batches(
            vid, vid_masks, query_feat, text_size
        )
        if not self.query_conditioned:
            query_feat = None
            query_mask = None
        return self.vid_net(
            vid,
            vid_masks,
            query_feat=query_feat,
            query_mask=query_mask,
            return_importance_debug=return_importance_debug,
            return_anchor_assignments=return_anchor_assignments,
            allocator_targets=allocator_targets,
            forced_cut_offsets=forced_cut_offsets,
            forced_target_counts=forced_target_counts,
        )

    @staticmethod
    def _unpack_video_outputs(
        video_outputs,
        return_importance_debug,
        return_anchor_assignments,
    ):
        expected = (
            4
            + int(return_importance_debug)
            + int(return_anchor_assignments)
        )
        if len(video_outputs) != expected:
            raise RuntimeError(
                'expected {} video outputs, got {}'.format(
                    expected, len(video_outputs)
                )
            )
        fpn, fpn_masks, anchor_fpn, anchor_masks = video_outputs[:4]
        extra_index = 4
        importance_debug = None
        anchor_assignments = None
        if return_importance_debug:
            importance_debug = video_outputs[extra_index]
            extra_index += 1
        if return_anchor_assignments:
            anchor_assignments = video_outputs[extra_index]
        return (
            fpn,
            fpn_masks,
            anchor_fpn,
            anchor_masks,
            importance_debug,
            anchor_assignments,
        )

    def fuse_and_predict(self, fpn, fpn_masks, text, text_masks, text_size=None):
        fpn, fpn_masks_fusion = self.fusion(fpn, fpn_masks, text, text_masks, text_size)
        router = getattr(self, 'query_hierarchy_router', None)
        if router is not None:
            if text.ndim == 4:
                _t = torch.cat([tt[:k] for tt, k in zip(text, text_size)])
                _m = torch.cat([tm[:k] for tm, k in zip(text_masks, text_size)])
            else:
                _t, _m = text, text_masks
            if _t.size(1) != router.net[0].in_features:
                _t = _t.transpose(1, 2)          # (N, L, Cq) -> (N, Cq, L)
            if _m.ndim == 3:
                _m = _m.squeeze(1)               # (N, L)
            _tf = _t.float()
            wm = _m.float()[:, None, :]          # (N, 1, L)
            q_pool = (_tf * wm).sum(-1) / wm.sum(-1).clamp_min(1.0)
            w = router(q_pool.to(router.net[0].weight.dtype))
            self.last_router_weights = w.detach()
            L = len(fpn)
            fpn = tuple(fl * (L * w[:, l].view(-1, 1, 1)).to(fl.dtype)
                        for l, fl in enumerate(fpn))
        if self.query_mamba_adapter is not None:
            fpn = self.query_mamba_adapter(
                fpn, fpn_masks_fusion, text, text_masks
            )
        if self.cdf_neck is not None:
            mref = self.vid_net.module if hasattr(
                self.vid_net, 'module') else self.vid_net
            meta = mref.last_temporal_metadata
            if meta is None:
                raise RuntimeError(
                    'CDF requires adaptive temporal metadata')
            fpn = self.cdf_neck(
                fpn, meta,
                [m if m.ndim == 2 else m[:, 0]
                 for m in fpn_masks_fusion])
        # pooled query for optional head FiLM (HM-QCPH-036)
        _q = None
        if getattr(self.cls_head, 'query_film', False) or getattr(
                self.reg_head, 'query_film', False):
            if text.ndim == 4:
                _tf = torch.cat([tt[:k] for tt, k in zip(
                    text, text_size)])
                _tm = torch.cat([tm[:k] for tm, k in zip(
                    text_masks, text_size)])
            else:
                _tf, _tm = text, text_masks
            _tmf = _tm.float()[:, 0] if _tm.ndim == 3 else _tm.float()
            _tf = _tf.float() * _tmf.unsqueeze(1)
            _q = _tf.sum(-1) / _tmf.sum(-1).clamp_min(1.0).unsqueeze(1)
        if self.tefm is not None:
            fpn = self.tefm(fpn, fpn_masks_fusion)
            self.last_evidence_logits = self.evidence_head(
                fpn, fpn_masks_fusion)
        else:
            self.last_evidence_logits = None
        fpn_logits, _ = self.cls_head(fpn, fpn_masks_fusion, query=_q)
        fpn_offsets, fpn_masks = self.reg_head(
            fpn, fpn_masks_fusion, query=_q)
        return fpn_logits, fpn_logits, fpn_offsets, fpn_masks

    def forward(
        self,
        vid,
        vid_masks,
        text,
        text_masks,
        text_size=None,
        return_importance_debug=False,
        return_anchor_assignments=False,
        allocator_targets=None,
        forced_cut_offsets=None,
        forced_target_counts=None,
    ):
        if self.early_fusion:
            return self._forward_earlyfusion(
                vid,
                vid_masks,
                text,
                text_masks,
                text_size,
                return_importance_debug=return_importance_debug,
                return_anchor_assignments=return_anchor_assignments,
                allocator_targets=allocator_targets,
                forced_cut_offsets=forced_cut_offsets,
                forced_target_counts=forced_target_counts,
            )
        else:
            return self._forward_regular(
                vid,
                vid_masks,
                text,
                text_masks,
                text_size,
                return_importance_debug=return_importance_debug,
                return_anchor_assignments=return_anchor_assignments,
                allocator_targets=allocator_targets,
                forced_cut_offsets=forced_cut_offsets,
                forced_target_counts=forced_target_counts,
            )

    def _forward_regular(
        self,
        vid,
        vid_masks,
        text,
        text_masks,
        text_size=None,
        return_importance_debug=False,
        return_anchor_assignments=False,
        allocator_targets=None,
        forced_cut_offsets=None,
        forced_target_counts=None,
    ):
        # pack text features
        if text.ndim == 4:
            text = torch.cat([t[:k] for t, k in zip(text, text_size)])
        if text_masks.ndim == 3:
            text_masks = torch.cat(
                [t[:k] for t, k in zip(text_masks, text_size)]
            )
        
        text, text_masks = self.encode_text(text, text_masks)
        video_outputs = self.encode_video(
            vid,
            vid_masks,
            query_feat=text if self.query_conditioned else None,
            query_mask=text_masks if self.query_conditioned else None,
            text_size=text_size,
            return_importance_debug=return_importance_debug,
            return_anchor_assignments=return_anchor_assignments,
            allocator_targets=allocator_targets,
            forced_cut_offsets=forced_cut_offsets,
            forced_target_counts=forced_target_counts,
        )
        (
            fpn,
            sequence_fpn_masks,
            anchor_fpn,
            anchor_fpn_mask,
            importance_debug,
            anchor_assignments,
        ) = self._unpack_video_outputs(
            video_outputs,
            return_importance_debug,
            return_anchor_assignments,
        )
        fpn_logits, fpn_logits2, fpn_offsets, fpn_masks = \
            self.fuse_and_predict(fpn, sequence_fpn_masks, text, text_masks, text_size) # B, 1, 2304

        outputs = (
            fpn_logits,
            fpn_logits2,
            fpn_offsets,
            fpn_masks,
            fpn,
            sequence_fpn_masks,
            anchor_fpn,
            anchor_fpn_mask,
        )
        if return_importance_debug:
            outputs += (importance_debug, )
        if return_anchor_assignments:
            outputs += (anchor_assignments, )
        return outputs

    def _forward_earlyfusion(
        self,
        vid,
        vid_masks,
        text,
        text_masks,
        text_size=None,
        return_importance_debug=False,
        return_anchor_assignments=False,
        allocator_targets=None,
        forced_cut_offsets=None,
        forced_target_counts=None,
    ):
        # pack text features
        if text.ndim == 4:
            text = torch.cat([t[:k] for t, k in zip(text, text_size)])
        if text_masks.ndim == 3:
            text_masks = torch.cat(
                [t[:k] for t, k in zip(text_masks, text_size)]
            )
        
        text, text_masks = self.encode_text(text, text_masks)
        
        # Project raw video features to embedding space before fusion
        if vid_masks.ndim == 2:
            vid_masks = vid_masks.unsqueeze(1)
        vid, vid_masks = self.project_video(vid, vid_masks)
        
        # early fusion 
        vid_fused, vid_masks_fused = self.fusion(vid, vid_masks, text, text_masks, text_size) # vid_fused: (b_query, c, t), vid_masks_fused: (b_query, 1, t)
        
        # continue with video encoding using the fused features
        video_outputs = self.encode_video(
            vid_fused,
            vid_masks_fused,
            query_feat=text if self.query_conditioned else None,
            query_mask=text_masks if self.query_conditioned else None,
            text_size=text_size,
            return_importance_debug=return_importance_debug,
            return_anchor_assignments=return_anchor_assignments,
            allocator_targets=allocator_targets,
            forced_cut_offsets=forced_cut_offsets,
            forced_target_counts=forced_target_counts,
        )
        (
            fpn,
            sequence_fpn_masks,
            anchor_fpn,
            anchor_fpn_mask,
            importance_debug,
            anchor_assignments,
        ) = self._unpack_video_outputs(
            video_outputs,
            return_importance_debug,
            return_anchor_assignments,
        )
        fpn_logits, fpn_logits2, fpn_offsets, fpn_masks = \
            self.fuse_and_predict(fpn, sequence_fpn_masks, text, text_masks, text_size)

        outputs = (
            fpn_logits,
            fpn_logits2,
            fpn_offsets,
            fpn_masks,
            fpn,
            sequence_fpn_masks,
            anchor_fpn,
            anchor_fpn_mask,
        )
        if return_importance_debug:
            outputs += (importance_debug, )
        if return_anchor_assignments:
            outputs += (anchor_assignments, )
        return outputs


@register_models_net('adaptive_hieramamba')
@register_models_net('query_boundary_adaptive_hieramamba')
class AdaptiveHieraMamba(HieraMamba):
    """Unified configurable model built from the existing HieraMamba path."""

    pass


class BufferList(nn.Module):

    def __init__(self, buffers):
        super().__init__()

        for i, buf in enumerate(buffers):
            self.register_buffer(str(i), buf, persistent=False)

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


class PtGenerator(nn.Module):
    """
    A generator for candidate points from specified FPN levels.
    """
    def __init__(
        self,
        max_seq_len,        # max sequence length
        num_fpn_levels,     # number of feature pyramid levels
        regression_range=4, # normalized regression range
        sigma=1,            # controls overlap between adjacent levels
        use_offset=False,   # whether to align points at the middle of two tics
        allow_fixed_stride=True,
    ):
        super().__init__()

        self.num_fpn_levels = num_fpn_levels
        assert max_seq_len % 2 ** (self.num_fpn_levels - 1) == 0
        self.max_seq_len = max_seq_len

        # derive regression range for each pyramid level
        self.regression_range = ((0, regression_range), )
        assert sigma > 0 and sigma <= 1
        for l in range(1, self.num_fpn_levels):
            assert regression_range <= max_seq_len
            v_min = regression_range * sigma
            v_max = regression_range * 2
            if l == self.num_fpn_levels - 1:
                v_max = max(v_max, max_seq_len + 1)
            self.regression_range += ((v_min, v_max), )
            regression_range = v_max

        self.use_offset = use_offset
        self.allow_fixed_stride = bool(allow_fixed_stride)

        # generate and buffer all candidate points
        self.buffer_points = self._generate_points()

    def _generate_points(self):
        # tics on the input grid
        tics = torch.arange(0, self.max_seq_len, 1.0)

        points_list = tuple()
        for l in range(self.num_fpn_levels):
            stride = 2 ** l
            points = tics[::stride][:, None]                    # (t, 1)
            if self.use_offset:
                points += 0.5 * stride

            reg_range = torch.as_tensor(
                self.regression_range[l], dtype=torch.float32
            )[None].repeat(len(points), 1)                      # (t, 2)
            stride = torch.as_tensor(
                stride, dtype=torch.float32
            )[None].repeat(len(points), 1)                      # (t, 1)
            points = torch.cat((points, reg_range, stride), 1)  # (t, 4)
            points_list += (points, )

        return BufferList(points_list)

    def forward(self, fpn_n_points):
        """
        Args:
            fpn_n_points (int list [l]): number of points at specified levels.

        Returns:
            fpn_point (float tensor [l * (p, 4)]): candidate points from speficied levels.
        """
        if not self.allow_fixed_stride:
            raise RuntimeError(
                'fixed-stride PtGenerator was invoked while adaptive_anchor '
                'is enabled. Use AdaptiveTemporalPointGenerator with temporal '
                'metadata instead.'
            )
        assert len(fpn_n_points) == self.num_fpn_levels

        fpn_points = tuple()
        for n_pts, pts in zip(fpn_n_points, self.buffer_points):
            assert n_pts <= len(pts), (
                'number of requested points {:d} cannot exceed max number '
                'of buffered points {:d}'.format(n_pts, len(pts))
            )
            fpn_points += (pts[:n_pts], )

        return fpn_points
    
class QueryHierarchyRouter(nn.Module):
    """HQR: query -> per-level softmax weights (zero-init => uniform)."""

    def __init__(self, query_dim, num_levels, hidden_dim=256,
                 no_query=False):
        super().__init__()
        self.no_query = bool(no_query)
        self.mean_query = nn.Parameter(torch.zeros(query_dim))
        nn.init.normal_(self.mean_query, std=0.02)
        self.net = nn.Sequential(
            nn.Linear(query_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, num_levels),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, q):
        if self.no_query:
            q = self.mean_query.unsqueeze(0).expand(q.size(0), -1)
        return torch.softmax(self.net(q), dim=-1)


def make_models_net(opt):
    opt = deepcopy(opt)
    net = models[opt['model_net'].pop('name')](opt['model'])
    if opt.get('train', {}).get('loss_aux', {}).get(
            'listwise_ranking', {}
    ).get('enable', False):
        from .rank_head import MomentRankHead
        net.moment_rank_head = MomentRankHead(
            feat_dim=opt['model']['vid_net']['embd_dim'],
            num_levels=opt['model']['vid_net']['arch'][-1],
        )
    if opt.get('model', {}).get('hqr_enable', False):
        net.query_hierarchy_router = QueryHierarchyRouter(
            query_dim=opt['model']['text_net'].get(
                'embd_dim', opt['model']['text_net']['in_dim']),
            num_levels=opt['model']['vid_net']['arch'][-1],
            hidden_dim=int(opt['model'].get('hqr_hidden', 256)),
            no_query=bool(opt['model'].get('hqr_no_query', False)),
        )
    if opt.get('train', {}).get('loss_aux', {}).get(
            'span_denoising', {}
    ).get('enable', False):
        from .denoiser import MomentDenoiser
        net.moment_denoiser = MomentDenoiser(
            feat_dim=opt['model']['vid_net']['embd_dim'],
            query_dim=opt['model']['text_net'].get(
                'embd_dim', opt['model']['text_net']['in_dim']),
        )
    return net

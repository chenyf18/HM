from copy import deepcopy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    sinusoid_encoding, MaskedConv1D, LayerNorm, TransformerEncoder, MaskedMaxPool1D
)
from .coarse_to_fine_refinement import CoarseToFineTemporalRefiner
from .temporal_coordinates import AdaptiveTemporalPointGenerator

# Import all block types
from .anchor_mamba import *

backbones = dict()
def register_video_net(name):
    def decorator(module):
        backbones[name] = module
        return module
    return decorator

@register_video_net('hieramamba_backbone')
class HieraMambaBackbone(nn.Module):
    """
    A backbone that combines convolutions with transformer encoder layers 
    to build a feature pyramid.
    
    video clip features
    -> [embedding convs x L1]
    -> [stem transformer x L2]
    -> [branch transformer x L3]
    -> latent video feature pyramid
    """
    def __init__(
        self,
        in_dim=256,             # video feature dimension
        embd_dim=384,           # embedding dimension
        max_seq_len=2304,        # max sequence length
        n_heads=4,            # number of attention heads for MHA
        mha_win_size=0,       # local window size for MHA (0 for global attention)
        stride=1,           # conv stride applied to the input features
        arch=(2, 0, 8),     # (#convs, #stem transformers, #branch transformers)
        attn_pdrop=0.0,     # dropout rate for attention maps
        proj_pdrop=0.0,     # dropout rate for projection
        path_pdrop=0.0,     # dropout rate for residual paths
        use_abs_pe=False,   # whether to apply absolute position encoding
        local_window_size=5, # whether to encode local features
        pool_method='mean',
        return_anchor=False,
        block_type="AnchorMambaPoolingBlockGated",  # block type from anchor_mamba module
        local_encoder_type="transformer",  # Options: 'transformer' or 'mamba'
        ffn_ratio=2,
        local_encode=False,
        local_encode_num_layers=0,
        mamba_headdim=64, 
        mamba_dstate=64,
        mamba_expand=2,
        mamba_dconv=7,
        bidirectional=True,
        query_dim=0,
        query_modulation=False,
        query_aware_gate=False,
        query_mod_hidden_dim=0,
        query_boundary_importance=False,
        importance_debug=False,
        importance_hidden_dim=0,
        importance_alpha=0.4,
        importance_beta=0.4,
        importance_gamma=0.2,
        adaptive_anchor=False,
        target_keep_ratio=0.75,
        keep_ratio_per_level=None,
        allocator_policy="learned",
        allocator_random_seed=0,
        oracle_boundary_radius=1.0,
        reallocation_ratio=0.1,
        progressive_compression_debug=False,
        coarse_to_fine_refine=False,
        refine_levels=3,
        state_bridge=False,
        state_bridge_tokens=4,
        qstb_enable=False,
        qstb_levels=(1,),
        qstb_source="hydra",
        qstb_use_delta=True,
        qstb_fusion_gamma=1.0,
        qsm_enable=False,
        qsm_levels=(0, 1, 2),
        qsm_hidden=64,
    ):
        super().__init__()

        assert len(arch) == 3, '(embed convs, stem, branch)'
        assert stride & (stride - 1) == 0
        assert arch[0] >= int(math.log2(stride))
        self.max_seq_len = max_seq_len
        self.return_anchor = return_anchor
        self.adaptive_anchor = bool(adaptive_anchor)
        self.allocator_policy = str(allocator_policy).strip().lower()
        if self.allocator_policy not in (
            "uniform", "random", "learned", "oracle", "balanced",
        ):
            raise ValueError(
                "allocator_policy must be one of uniform, random, learned, oracle"
            )
        if self.allocator_policy != "learned" and not self.adaptive_anchor:
            raise ValueError(
                "allocator_policy '{}' requires adaptive_anchor=true".format(
                    self.allocator_policy
                )
            )
        self.allocator_random_seed = int(allocator_random_seed)
        self.oracle_boundary_radius = float(oracle_boundary_radius)
        self.reallocation_ratio = float(reallocation_ratio)
        self.importance_enabled = bool(
            query_boundary_importance or self.adaptive_anchor
        )
        self.amp_query_conditioned = bool(
            query_modulation or query_aware_gate or self.importance_enabled
        )
        self.qstb_enable = bool(qstb_enable)
        self.qstb_levels = tuple(int(x) for x in qstb_levels)
        self.state_bridge = bool(state_bridge)
        self.state_bridge_tokens = int(state_bridge_tokens)
        self.coarse_to_fine_refine = bool(coarse_to_fine_refine)
        self.qstb_branch_dim = query_dim
        self.query_conditioned = bool(
            self.amp_query_conditioned or self.coarse_to_fine_refine
        )
        self.importance_debug = bool(importance_debug)
        self.last_importance_debug = None
        self.progressive_compression_debug = bool(
            progressive_compression_debug
        )
        self.last_progressive_debug = None
        self.last_temporal_metadata = None
        # Convolutional embedding downsamples by the configured input stride.
        # Each resulting token therefore advances by `stride` input tokens.
        self.input_temporal_stride = float(stride)
        branch_depth = int(arch[2])
        self.keep_ratio_per_level = self._resolve_keep_ratios(
            branch_depth,
            target_keep_ratio,
            keep_ratio_per_level,
        )
        local_encode_num_layers = arch[2] if local_encode_num_layers == 0 else local_encode_num_layers
        # embedding projection
        self.embd_fc = MaskedConv1D(in_dim, embd_dim, 1)

        # embedding convs
        self.embd_convs = nn.ModuleList()
        self.embd_norms = nn.ModuleList()
        for _ in range(arch[0]):
            self.embd_convs.append(
                MaskedConv1D(
                    embd_dim, embd_dim,
                    kernel_size=5 if stride > 1 else 3,
                    stride=2 if stride > 1 else 1,
                    padding=2 if stride > 1 else 1,
                    bias=False
                )
            )
            self.embd_norms.append(LayerNorm(embd_dim))
            stride = max(stride // 2, 1)

        # position encoding (c, t)
        if use_abs_pe:
            pe = sinusoid_encoding(max_seq_len, embd_dim // 2)
            pe /= embd_dim ** 0.5
            self.register_buffer('pe', pe, persistent=False)
        else:
            self.pe = None

        # stem transformers
        self.stem = nn.ModuleList()
        for _ in range(arch[1]):
            self.stem.append(
                TransformerEncoder(
                    embd_dim,
                    stride=1,
                    n_heads=n_heads,
                    window_size=mha_win_size,
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop
                )
            )

        BlockClass = globals().get(block_type, AnchorMambaPoolingBlock)
        if self.amp_query_conditioned and not issubclass(
            BlockClass, AnchorMambaPoolingBlockGated
        ):
            raise ValueError(
                "query-conditioned AMP requires AnchorMambaPoolingBlockGated"
            )

        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(
                BlockClass(
                    d_model=embd_dim,
                    stride=2,
                    nhead=n_heads,
                    local_window_size=local_window_size if idx < 5 else 0,  # Use local window size for first 5 layers
                    local_encode=local_encode if idx < local_encode_num_layers else False,
                    pool_method=pool_method,
                    local_encoder_type=local_encoder_type,
                    ffn_ratio=ffn_ratio,
                    mamba_headdim=mamba_headdim,
                    mamba_dstate=mamba_dstate,
                    mamba_expand=mamba_expand,
                    mamba_dconv=mamba_dconv,
                    bidirectional=bidirectional,
                    **(
                        {
                            "query_dim": query_dim,
                            "query_modulation": query_modulation,
                            "query_aware_gate": query_aware_gate,
                            "query_mod_hidden_dim": query_mod_hidden_dim,
                            "query_boundary_importance": query_boundary_importance,
                            "importance_hidden_dim": importance_hidden_dim,
                            "importance_alpha": importance_alpha,
                            "importance_beta": importance_beta,
                            "importance_gamma": importance_gamma,
                            "adaptive_anchor": adaptive_anchor,
                            "target_keep_ratio": self.keep_ratio_per_level[idx],
                            "allocator_policy": self.allocator_policy,
                            "allocator_random_seed": self.allocator_random_seed,
                            "oracle_boundary_radius": self.oracle_boundary_radius,
                            "reallocation_ratio": self.reallocation_ratio,
                        }
                        if issubclass(BlockClass, AnchorMambaPoolingBlockGated)
                        else {}
                    )
                )
            )
        
        # print("LOCAL WINDOW SIZE: ", local_window_size)
        self.apply(self.__init_weights__)
        self.qstb_branch = None
        if self.qstb_enable:
            from .state_transition_boundary import (
                StateTransitionBoundaryBranch,
            )
            for lvl in self.qstb_levels:
                self.branch[lvl].stash_state_source = True
            self.qstb_branch = StateTransitionBoundaryBranch(
                d_model=embd_dim,
                query_dim=query_dim,
                levels=self.qstb_levels,
                source=qstb_source,
                use_delta=qstb_use_delta,
                fusion_gamma=qstb_fusion_gamma,
            )
        self.state_bridge_mlp = None
        if self.state_bridge:
            # Fine-to-coarse context prefix: maps the previous (finer)
            # level's pooled anchor summary to K prefix tokens.  Final
            # projection is zero-initialized so the enabled path starts
            # exactly equal to the baseline.
            self.state_bridge_mlp = nn.Sequential(
                nn.Linear(embd_dim, embd_dim),
                nn.GELU(),
                nn.Linear(embd_dim,
                          self.state_bridge_tokens * embd_dim),
            )
            nn.init.zeros_(self.state_bridge_mlp[-1].weight)
            nn.init.zeros_(self.state_bridge_mlp[-1].bias)
        self.coarse_to_fine_refiner = None
        if self.coarse_to_fine_refine:
            self.coarse_to_fine_refiner = CoarseToFineTemporalRefiner(
                d_model=embd_dim,
                query_dim=query_dim,
                num_levels=branch_depth,
                refine_levels=refine_levels,
            )

        # QSM-Delta (HM-QSM-D012): per-level query->dt-logit modulators.
        # Zero-initialised, so with qsm_enable=true the model still starts
        # exactly equal to the baseline; with qsm_enable=false (default)
        # nothing is constructed and the original path is untouched.
        self.qsm_enable = bool(qsm_enable)
        if self.qsm_enable:
            if query_dim <= 0:
                raise ValueError(
                    "qsm_enable requires a query-conditioned backbone "
                    "(query_dim > 0)"
                )
            from .anchor_mamba import QueryDeltaModulator
            self.qsm_levels = tuple(int(l) for l in qsm_levels)
            for lvl in self.qsm_levels:
                block = self.branch[lvl]
                n_heads = block.global_encoder.nheads
                # fork the RNG so modulator construction does not shift
                # the init of downstream modules (clean A-U comparison);
                # the modulator is zero-initialised anyway.
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(20260908 + lvl)
                    block.qsm_modulator = QueryDeltaModulator(
                        query_dim=query_dim,
                        n_heads=n_heads,
                        hidden_dim=int(qsm_hidden),
                    )
                block.qsm_enabled = True

    @staticmethod
    def _resolve_keep_ratios(
        branch_depth,
        target_keep_ratio,
        keep_ratio_per_level,
    ):
        if keep_ratio_per_level is None:
            raw_ratios = [target_keep_ratio] * branch_depth
        else:
            if not isinstance(keep_ratio_per_level, (list, tuple)):
                raise ValueError(
                    "keep_ratio_per_level must be a list or tuple"
                )
            if not keep_ratio_per_level:
                raise ValueError("keep_ratio_per_level must not be empty")
            if len(keep_ratio_per_level) != branch_depth:
                raise ValueError(
                    "keep_ratio_per_level length {} must match branch "
                    "depth {}".format(
                        len(keep_ratio_per_level), branch_depth
                    )
                )
            raw_ratios = keep_ratio_per_level

        resolved = []
        for level, ratio in enumerate(raw_ratios):
            try:
                ratio = float(ratio)
            except (TypeError, ValueError):
                raise ValueError(
                    "keep ratio at level {} must be numeric".format(level)
                )
            if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
                raise ValueError(
                    "keep ratio at level {} must be finite and in "
                    "(0, 1], got {}".format(level, ratio)
                )
            resolved.append(ratio)
        return tuple(resolved)

    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x,
        mask,
        query_feat=None,
        query_mask=None,
        return_importance_debug=False,
        return_anchor_assignments=False,
        allocator_targets=None,
        forced_cut_offsets=None,
        forced_target_counts=None,
    ):
        """
        Args:
            x (float tensor, (bs, c1, t1)): video features.
            mask (bool tensor, (bs, t1)): video mask.
            query_feat (optional tensor): Query features with shape
                (bs, cq, lq), (bs, lq, cq), or (bs, cq).
            query_mask (optional bool tensor): Query mask with shape
                (bs, lq) or (bs, 1, lq).
        """
        if mask.ndim == 2:
            mask = mask.unsqueeze(1)    # (bs, l) -> (bs, 1, l)
        if return_anchor_assignments and not self.adaptive_anchor:
            raise ValueError(
                'anchor assignments require adaptive_anchor=true'
            )

        # embedding projection
        x, _ = self.embd_fc(x, mask)

        # embedding convs
        for conv, norm in zip(self.embd_convs, self.embd_norms):
            x, mask = conv(x, mask)
            x = F.relu(norm(x), inplace=True)
            # x = F.relu(norm(x).clone(), inplace=False)

        # position encoding
        _, _, t = x.size()
        if self.pe is not None:
            pe = self.pe.to(x.dtype)
            if self.training:
                assert t <= self.max_seq_len
            else:
                if t > self.max_seq_len:
                    pe = F.interpolate(
                        pe[None], size=t, mode='linear', align_corners=True
                    )[0]
            x = x + pe[..., :t] * mask.to(x.dtype)
        else:
            x = x * mask.to(x.dtype)

        temporal_metadata = None
        if self.adaptive_anchor:
            temporal_metadata = AdaptiveTemporalPointGenerator.make_initial_metadata(
                mask, input_stride=self.input_temporal_stride
            )

        # stem layers
        for block in self.stem:
            x, mask = block(x, mask)

        # branch layers
        fpn, fpn_masks, anchor_fpn, anchor_fpn_masks = tuple(), tuple(), tuple(), tuple()
        temporal_metadata_fpn = []
        anchor_assignments = []
        collect_anchor_assignments = bool(
            return_anchor_assignments
            or (self.coarse_to_fine_refine and self.adaptive_anchor)
        )
        collect_importance = bool(
            return_importance_debug or self.importance_debug
        ) and self.importance_enabled
        importance_debug = []
        progressive_debug = []
        anchor_summary = None
        for level_idx, block in enumerate(self.branch):
            block_kwargs = {}
            if self.amp_query_conditioned:
                block_kwargs.update(
                    query_feat=query_feat,
                    query_mask=query_mask,
                )
            if collect_importance:
                block_kwargs['return_importance_debug'] = True
            if self.adaptive_anchor and self.allocator_policy == "oracle":
                # Diagnostic-only GT grouping resolution.  The current level's
                # input metadata is already available before the block runs.
                block_kwargs['temporal_metadata'] = temporal_metadata
                block_kwargs['allocator_targets'] = allocator_targets
            if (self.state_bridge and level_idx > 0
                    and anchor_summary is not None):
                block_kwargs['state_prefix'] = self.state_bridge_mlp(
                    anchor_summary).view(
                        anchor_summary.size(0),
                        self.state_bridge_tokens, -1)
            if self.adaptive_anchor and forced_cut_offsets is not None:
                # Offline counterfactual override (utility data collection);
                # policy-independent so any adaptive config can be audited.
                block_kwargs['forced_cut_offsets'] = forced_cut_offsets[level_idx]
                if forced_target_counts is not None:
                    block_kwargs['forced_target_counts'] = (
                        forced_target_counts[level_idx]
                    )
            if level_idx == 0:
                layer_input = x
                layer_input_mask = mask
            else:
                layer_input = anchor_out
                layer_input_mask = anchor_mask
            block_out = block(
                layer_input, layer_input_mask, **block_kwargs
            )
            if self.adaptive_anchor:
                if collect_importance:
                    (
                        anchor_out, x_out, anchor_mask, x_mask,
                        assignment_matrix, importance_score, block_debug,
                    ) = block_out
                    importance_debug.append(block_debug)
                else:
                    (
                        anchor_out, x_out, anchor_mask, x_mask,
                        assignment_matrix, importance_score,
                    ) = block_out
            elif collect_importance:
                (
                    anchor_out, x_out, anchor_mask, x_mask, block_debug
                ) = block_out
                importance_debug.append(block_debug)
            else:
                anchor_out, x_out, anchor_mask, x_mask = block_out
            if self.state_bridge:
                am = (anchor_mask if anchor_mask.ndim == 3
                      else anchor_mask[:, None]).to(torch.float32)
                af = anchor_out.float()
                msum = am.sum(-1).clamp_min(1.0)
                anchor_summary = (af * am).sum(-1) / msum
            if collect_anchor_assignments:
                anchor_assignments.append(assignment_matrix)
            if self.adaptive_anchor:
                temporal_metadata_fpn.append(temporal_metadata)
                temporal_metadata = AdaptiveTemporalPointGenerator.propagate(
                    temporal_metadata,
                    assignment_matrix,
                    layer_input_mask,
                    anchor_mask,
                )
                input_lengths = layer_input_mask.sum(dim=-1).flatten()
                anchor_lengths = anchor_mask.sum(dim=-1).flatten()
                layer_progress = {
                    "level": level_idx,
                    "input_len": input_lengths.detach(),
                    "anchor_len": anchor_lengths.detach(),
                    "keep_ratio": self.keep_ratio_per_level[level_idx],
                    "input_tensor_len": layer_input.size(-1),
                    "anchor_tensor_len": anchor_out.size(-1),
                }
                progressive_debug.append(layer_progress)
                if self.progressive_compression_debug:
                    print(
                        "[ProgressiveCompression] level={} input_len={} "
                        "-> anchor_len={} keep_ratio={:.6g}".format(
                            level_idx,
                            layer_progress["input_len"].cpu().tolist(),
                            layer_progress["anchor_len"].cpu().tolist(),
                            layer_progress["keep_ratio"],
                        )
                    )
            if (self.qstb_enable and level_idx in self.qstb_levels
                    and self.qstb_branch is not None):
                blk = self.branch[level_idx]
                h_src = (blk._stashed_state
                         if self.qstb_branch.source == "hydra"
                         else blk._stashed_pre)
                pos = blk._stashed_seq_pos            # (B, T)
                tmask = blk._stashed_seq_mask.squeeze(-2)  # (B, T) bool
                Bq, Tp, D = h_src.shape
                tok = h_src.gather(
                    1, pos.clamp_min(0).unsqueeze(-1).expand(-1, -1, D))
                q_pooled = blk._pool_query(
                    query_feat, query_mask, h_src.size(0),
                    tok.dtype, tok.device)
                logits_q, b_emb = self.qstb_branch(
                    tok.transpose(1, 2), q_pooled, tmask)
                x_out = self.qstb_branch.fused_fpn(x_out, b_emb)
                if not hasattr(self, 'last_qstb') or self.last_qstb is None:
                    self.last_qstb = {}
                self.last_qstb[level_idx] = {
                    'logits': logits_q, 'mask': tmask,
                }
            fpn += (x_out, )
            fpn_masks += (x_mask, )
            anchor_fpn += (anchor_out, )
            anchor_fpn_masks += (anchor_mask, )
        if self.coarse_to_fine_refine:
            refinement_assignments = (
                tuple(anchor_assignments) if self.adaptive_anchor else None
            )
            fpn = self.coarse_to_fine_refiner(
                fpn,
                fpn_masks,
                query_feat=query_feat,
                query_mask=query_mask,
                assignments=refinement_assignments,
                temporal_metadata=(
                    tuple(temporal_metadata_fpn)
                    if self.adaptive_anchor else None
                ),
            )
        if collect_importance:
            self.last_importance_debug = tuple(importance_debug)
        else:
            self.last_importance_debug = None
        self.last_progressive_debug = (
            tuple(progressive_debug) if self.adaptive_anchor else None
        )
        self.last_temporal_metadata = (
            tuple(temporal_metadata_fpn) if self.adaptive_anchor else None
        )
        outputs = (
            (fpn, fpn_masks, anchor_fpn, anchor_fpn_masks)
            if self.return_anchor else (fpn, fpn_masks)
        )
        if return_importance_debug:
            outputs += (self.last_importance_debug, )
        if return_anchor_assignments:
            outputs += (tuple(anchor_assignments), )
        return outputs





def make_video_net(opt):
    opt = deepcopy(opt)
    return backbones[opt.pop('name')](**opt)

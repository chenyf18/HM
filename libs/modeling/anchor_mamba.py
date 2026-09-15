from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from hydra.modules.hydra import Hydra
from mamba_ssm import Mamba2
from .blocks import TransformerEncoder, MaskedMaxPool1D, SwiGLUFFN
from .adaptive_anchor import QueryBoundaryAdaptiveAnchorAllocator
from .query_boundary_importance import QueryBoundaryImportancePredictor
import os

# Set environment variable for triton to use IEEE float32 precision for lower GPU capabilities
# Especially important for using Mamba2 to avoid issues on order GPUs
if torch.cuda.is_available():
    gpu_id = torch.cuda.current_device()
    gpu_capability = torch.cuda.get_device_capability(gpu_id)
    if gpu_capability[0] < 8:
        os.environ["TRITON_F32_DEFAULT"] = "ieee"

class BaseAnchorBlock(nn.Module):
    """
    Base class for all anchor blocks containing common anchor generation and interleaving operations.
    
    This base class provides:
    1. Efficient anchor position caching
    2. Robust anchor generation and interleaving
    3. Consistent mask creation and sequence extraction
    4. Configurable pooling methods for anchor generation
    
    Supported pooling methods:
    - "mean": Average pooling (default) - computes mean over sequence blocks
    - "max": Max pooling - takes maximum value over sequence blocks  
    - "attn": Attention pooling - uses learnable query vector similar to CLIP
    - "gated": Gated pooling - adaptive mean/max blending with learnable gates
    
    All derived anchor blocks should inherit from this class to avoid code duplication
    and ensure consistent anchor handling.
    """
    def __init__(self, stride: int, d_model: int, pool_method: str = "mean", dropout: float = 0.1):
        """
        Initialize BaseAnchorBlock with configurable pooling.
        
        Args:
            stride: Stride for downsampling (number of tokens per anchor block)
            d_model: Model embedding dimension
            pool_method: Pooling method for anchor generation. Options:
                        - "mean": Average pooling (default)
                        - "max": Max pooling 
                        - "attn": Attention pooling (CLIP-style with learnable query)
                        - "gated": Gated pooling (learnable mean/max combination)
            dropout: Dropout probability (used for attention pooling)
        """
        super().__init__()
        self.stride = stride
        self.d_model = d_model
        
        # Core anchor pooling component with configurable method
        self.anchor_pooling = AnchorPooling(
            stride=stride, 
            method=pool_method, 
            d_model=d_model, 
            nhead=1, 
            dropout=dropout
        )
        
        # Cache for anchor positions to avoid recomputation
        self._anchor_positions_cache = {}
    
    def _get_anchor_positions(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Cache and retrieve anchor positions to avoid recomputing them each time.
        Args:
            seq_len: Combined sequence length (after interleaving)
        Returns:
            Tensor of anchor positions
        """
        cache_key = (seq_len, str(device))
        if cache_key not in self._anchor_positions_cache:
            self._anchor_positions_cache[cache_key] = torch.arange(
                0, seq_len, self.stride + 1, device=device
            )
        return self._anchor_positions_cache[cache_key]

    def _generate_and_interleave_anchors(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        per_sample_mask: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Vectorized method to interleave anchors with sequence tokens.

        Args:
            x: Input tensor of shape (B, D, L)
            mask: Input mask of shape (B, 1, L)

        Returns:
            Tuple of:
            - x_combined: Interleaved anchors and sequences (B, combined_seq_len, D)
            - anchor_positions: Positions of anchors in combined sequence
            - expanded_mask: Mask for combined sequence (B, combined_seq_len, 1)
            - anchor_mask: Downsampled mask for anchors (B, 1, num_blocks)
        """
        B, D, L = x.shape
        s = self.stride
        device = x.device

        # 1) Correctly calculate number of blocks and required padding
        num_blocks = (L + s - 1) // s
        needed_len = num_blocks * s
        pad_len = needed_len - L

        if pad_len > 0:
            # Pad with zeros to make the sequence length a multiple of the stride
            x_padded = F.pad(x, (0, pad_len))
        else:
            x_padded = x

        # 2) Reshape into blocks of size <s> for pooling AND interleaving
        x_blocks = x_padded.reshape(B, D, num_blocks, s)

        # 3) Generate anchors using the configurable pooling method
        anchors = self.anchor_pooling(x_padded)
        
        # 4) Vectorized interleaving: Concatenate anchors and token blocks
        #    First, add a dimension to anchors to enable concatenation.
        anchors_expanded = anchors.unsqueeze(-1)  # (B, D, num_blocks, 1)
        mixed = torch.cat((anchors_expanded, x_blocks), dim=-1) # (B, D, num_blocks, s+1)

        # 5) Flatten back into a single sequence and permute to (B, L', D)
        x_combined = mixed.permute(0, 2, 3, 1).reshape(B, -1, D)

        # 6) Get anchor positions using the cached helper method
        anchor_positions = self._get_anchor_positions(x_combined.size(1), device)

        # 7) Create the correct masks for the new packed sequence
        anchor_mask = downsample_mask(mask, s)  # (B, 1, num_blocks)

        if per_sample_mask:
            seq_mask = mask.squeeze(1)
            if pad_len > 0:
                seq_mask = F.pad(seq_mask, (0, pad_len), value=False)
            seq_mask = seq_mask.reshape(B, num_blocks, s)
            anchor_mask_blocks = anchor_mask.transpose(1, 2)
            expanded_mask = torch.cat(
                (anchor_mask_blocks, seq_mask), dim=-1
            ).reshape(B, -1, 1)
        else:
            # Preserve released behavior when query-aware AMP is disabled.
            nonmasked_seq_len = int(mask.sum().item())
            nonmasked_anchor_len = int(anchor_mask.sum().item())
            nonmasked_len = nonmasked_seq_len + nonmasked_anchor_len
            expanded_mask = torch.zeros(
                (B, x_combined.size(1), 1),
                dtype=torch.bool,
                device=device,
            )
            expanded_mask[:, :nonmasked_len] = True

        return x_combined, anchor_positions, expanded_mask, anchor_mask

    def _generate_and_interleave_anchors_initial_v2(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Interleave an averaged anchor in front of every <stride> tokens:
            [a0, t0, t1,  a1, t2, t3,  ...]
        """
        B, D, L = x.shape
        s       = self.stride                    # e.g. 2
        device  = x.device

        # 1) Pad once so that L % s == 0  (keeps pooling semantics exact)
        pad_len = (-L) % s
        if pad_len:
            x = F.pad(x, (0, pad_len))           # (B, D, L+pad)

        # 2) Tokens grouped in blocks of length <s>
        num_blocks = x.size(-1) // s
        x_blocks   = x.view(B, D, num_blocks, s) # contiguous thanks to the pad

        # 3) Anchors via cuDNN avg_pool1d  (kernel = stride = s)
        #    shape comes out (B, D, num_blocks) → add a length-1 dim to match x_blocks
        anchors = F.avg_pool1d(x, kernel_size=s, stride=s)     # (B, D, num_blocks)
        anchors = anchors.unsqueeze(-1)                        # (B, D, num_blocks, 1)

        # 4) Concatenate anchor + tokens  →  (B, D, num_blocks, s+1)
        mixed = torch.cat((anchors, x_blocks), dim=-1)

        # 5) Flatten back to sequence, return (B, seq, D)
        x_combined = mixed.permute(0, 2, 3, 1).reshape(B, -1, D)

        # 6) Anchor positions are every (s+1) step
        anchor_positions = torch.arange(0, x_combined.size(1), s+1, device=device)

        # 7) Masks – stay compact
        anchor_mask = downsample_mask(mask, s)             # (B,1,num_blocks)

        seq_mask = mask.squeeze(1)                         # (B,L)
        if pad_len:
            seq_mask = F.pad(seq_mask, (0, pad_len), value=False)
        seq_mask = seq_mask.view(B, num_blocks, s)         # (B,blk,s)
           
        combined_seq_len = x_combined.size(1)
        # Create extended mask for combined sequence
        nonmasked_seq_len = int(mask.sum().item())
        nonmasked_anchor_len = int(anchor_mask.sum().item())
        nonmasked_len = nonmasked_seq_len + nonmasked_anchor_len
        # (B, T′)
        expanded_mask = torch.zeros((B, combined_seq_len, 1), dtype=torch.bool, device=device)
        expanded_mask[:, :nonmasked_len] = True
        return x_combined, anchor_positions, expanded_mask, anchor_mask
        
    def _generate_and_interleave_anchors__initial(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate anchors and interleave them with sequence tokens.

        Args:
            x: Input tensor of shape (B, D, L)
            mask: Input mask of shape (B, 1, L)
            
        Returns:
            Tuple of:
            - x_combined: Interleaved anchors and sequences (B, combined_seq_len, D)
            - anchor_positions: Positions of anchors in combined sequence
            - expanded_mask: Mask for combined sequence (B, combined_seq_len, 1)
            - anchor_mask: Downsampled mask for anchors (B, 1, num_blocks)
        """
        B, D, L = x.shape
        device = x.device
        
        # Downsample input mask for anchors
        anchor_mask = downsample_mask(mask, self.stride)
        
        # 1) Compute how many anchor-blocks we have
        num_blocks = (L + self.stride - 1) // self.stride
        
        # 2) Pad & block into shape (B, D, num_blocks, stride)
        needed_len = num_blocks * self.stride
        pad_len = needed_len - L
        
        # Avoid redundant copying when no padding is needed
        if pad_len > 0:
            x_padded = F.pad(x, (0, pad_len))
        else:
            x_padded = x
            
        # Use contiguous before reshape for better memory layout
        x_reshaped = x_padded.reshape(B, D, num_blocks, self.stride)
        
        # 3) Extract initial anchors
        anchors = self.anchor_pooling(x_reshaped)  # → (B, D, num_blocks)
        
        # Calculate combined sequence length once
        combined_seq_len = num_blocks * (self.stride + 1)
        
        # More efficient interleaving with fewer intermediate tensors
        # Pre-allocate the output tensor directly
        x_combined = torch.empty((B, combined_seq_len, D), dtype=x.dtype, device=device)
        
        # Place anchors at stride+1 intervals
        anchor_positions = self._get_anchor_positions(combined_seq_len, device)
        x_combined[:, anchor_positions] = anchors.permute(0, 2, 1)  # (B, num_blocks, D)
        
        # Place sequence tokens efficiently
        for i in range(self.stride):
            seq_positions = anchor_positions + i + 1
            # Handle boundary conditions
            valid_mask = seq_positions < combined_seq_len
            if not valid_mask.all():
                seq_positions = seq_positions[valid_mask]
                seq_blocks = x_reshaped[:, :, :seq_positions.size(0), i]
            else:
                seq_blocks = x_reshaped[:, :, :, i]
            x_combined[:, seq_positions] = seq_blocks.permute(0, 2, 1)  # (B, blocks, D)
        
        # Create extended mask for combined sequence
        nonmasked_seq_len = int(mask.sum().item())
        nonmasked_anchor_len = int(anchor_mask.sum().item())
        nonmasked_len = nonmasked_seq_len + nonmasked_anchor_len
        
        # Apply mask directly to avoid extra operations
        expanded_mask = torch.zeros((B, combined_seq_len, 1), dtype=torch.bool, device=device)
        expanded_mask[:, :nonmasked_len] = True
        
        return x_combined, anchor_positions, expanded_mask, anchor_mask

    def _extract_anchor_and_sequence_outputs(
            self,
            processed_features: torch.Tensor,      # (B, combined_seq_len, D)
            anchor_positions: torch.Tensor,
            original_seq_len: int                  # L (without padding)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Faster, allocation-free extraction of anchors and sequence tokens.
        """
        B, combined_len, D = processed_features.shape
        s        = self.stride                    # stride used during interleave
        step     = s + 1                          # anchor + <s> tokens
        device   = processed_features.device

        # 1) Reshape once so we can index by block and within-block position.
        num_blocks = combined_len // step
        mixed      = processed_features.view(B, num_blocks, step, D)  # no copy

        # 2) Anchors live at position 0 in every block.
        #    Shape wanted: (B, D, num_blocks)
        anchor_out = mixed[:, :, 0, :].transpose(1, 2)                # view

        # 3) Sequence tokens are positions 1..s in every block.
        #    Flatten them back to a 1-D temporal axis, trim to `L`.
        seq_tokens = mixed[:, :, 1:, :].reshape(B, num_blocks * s, D) # view
        seq_out    = seq_tokens.transpose(1, 2)[..., :original_seq_len]

        return anchor_out, seq_out
    
    def _extract_anchor_and_sequence_outputs_initial(self, processed_features: torch.Tensor, anchor_positions: torch.Tensor, original_seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract anchor and sequence outputs from processed combined features.
        
        Args:
            processed_features: Processed combined features (B, combined_seq_len, D)
            anchor_positions: Positions of anchors in combined sequence
            original_seq_len: Original sequence length L
            
        Returns:
            Tuple of:
            - anchor_out: Extracted anchor features (B, D, num_blocks)
            - seq_out: Extracted sequence features (B, D, L)
        """
        device = processed_features.device
        
        # Extract anchor tokens
        updated_anchors = processed_features.index_select(1, anchor_positions)  # (B, num_blocks, D)
        anchor_out = updated_anchors.permute(0, 2, 1)  # (B, D, num_blocks)
        
        # Extract sequence tokens efficiently
        seq_mask = torch.ones(processed_features.size(1), dtype=torch.bool, device=device)
        seq_mask.index_fill_(0, anchor_positions, False)
        seq_indices = seq_mask.nonzero(as_tuple=True)[0]
        
        updated_seq = processed_features.index_select(1, seq_indices)  # (B, seq_len, D)
        seq_out = updated_seq.permute(0, 2, 1)[:, :, :original_seq_len]  # (B, D, L)
        
        return anchor_out, seq_out

class QueryDeltaModulator(nn.Module):
    """QSM-Delta (HM-QSM-D012): query -> per-head additive dt-logit delta.

    Acts ONLY on the selective-scan timescale (dt/Δ) of Hydra, in the
    pre-bias/pre-softplus logit space; B/C/A untouched. The final layer is
    zero-initialised so that enabling QSM starts exactly equivalent to the
    baseline. A hard clamp on the logit delta keeps softplus inputs sane.
    """

    LOGIT_CLAMP = 3.0

    def __init__(self, query_dim, n_heads, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_heads),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, q):
        delta = self.net(q.float())
        return torch.clamp(delta, -self.LOGIT_CLAMP, self.LOGIT_CLAMP)


class AnchorMambaPoolingBlockGated(BaseAnchorBlock):
    """
    Anchor Mamba Pooling Block with Gated Fusion
    
    This block combines the refined gating mechanism 
    with the optional local encoder pattern from AnchorMambaPoolingBlock.
    
    Key Features:
    1. Hierarchical gated fusion (gate1 always, gate2 only when local encoder enabled)
    2. Optional local encoder (can be disabled)
    3. Final FFN always present (not optional)
    4. Strong residual connections throughout
    
    Architecture:
    When local_encode=True:
    input → global → gate1 → local → gate2 → ffn → output
                     ↓               ↓       ↓
                     └─── residual ──┴───────┘
    
    When local_encode=False:
    input → global → gate1 → ffn → output
                     ↓       ↓
                     └───────┘
    
    Flow:
    1. Global encoding with residual connection
    2. First gated fusion (input + global)
    3. Optional local encoding + second gated fusion (if local_encode=True)
    4. Final FFN (always applied)
    """
    def __init__(
        self, 
        stride: int = 2, 
        d_model: int = 384, 
        nhead: int = 4, 
        local_window_size: int = 5,
        dropout: float = 0.1, 
        ffn_ratio: int = 4,
        local_encode: bool = False,  # Optional local encoder
        pool_method: str = "mean",
        local_encoder_type: str = "transformer",
        mamba_headdim: int = 64, 
        mamba_dstate: int = 64,
        mamba_expand: int = 2,
        mamba_dconv: int = 7,
        bidirectional: bool = True,
        query_dim: int = 0,
        query_modulation: bool = False,
        query_aware_gate: bool = False,
        query_mod_hidden_dim: int = 0,
        query_boundary_importance: bool = False,
        importance_hidden_dim: int = 0,
        importance_alpha: float = 0.4,
        importance_beta: float = 0.4,
        importance_gamma: float = 0.2,
        adaptive_anchor: bool = False,
        target_keep_ratio: float = 0.75,
        allocator_policy: str = "learned",
        allocator_random_seed: int = 0,
        oracle_boundary_radius: float = 1.0,
        reallocation_ratio: float = 0.1,
    ):
        super().__init__(stride=stride, d_model=d_model, pool_method=pool_method, dropout=dropout)

        self.local_encode = local_encode
        self.query_dim = int(query_dim) if query_dim else 0
        self.query_modulation = bool(query_modulation)
        self.query_aware_gate = bool(query_aware_gate)
        self.adaptive_anchor = bool(adaptive_anchor)
        self.target_keep_ratio = float(target_keep_ratio)
        self.allocator_policy = str(allocator_policy).strip().lower()
        self.allocator_random_seed = int(allocator_random_seed)
        self.oracle_boundary_radius = float(oracle_boundary_radius)
        self.reallocation_ratio = float(reallocation_ratio)
        if (
            not math.isfinite(self.target_keep_ratio)
            or not 0.0 < self.target_keep_ratio <= 1.0
        ):
            raise ValueError("target_keep_ratio must be finite and in (0, 1]")
        self.importance_enabled = bool(
            query_boundary_importance or self.adaptive_anchor
        )
        self.stash_state_source = False
        self._stashed_state = None
        self.query_conditioned = (
            self.query_modulation
            or self.query_aware_gate
            or self.importance_enabled
        )
        if self.query_conditioned and self.query_dim <= 0:
            raise ValueError(
                "query_dim must be positive when query-aware AMP is enabled"
            )
        
        if bidirectional:
            # Global encoder block 
            self.global_encoder = Hydra(
                d_model=d_model, d_state=mamba_dstate, d_conv=mamba_dconv, expand=mamba_expand,
                use_mem_eff_path=True, headdim=mamba_headdim
            )
        else:
            self.global_encoder = Mamba2(
                d_model=d_model, d_state=mamba_dstate, d_conv=4, expand=mamba_expand,
                headdim=48
            )
        # self.global_encoder = RWKVEncoder(d_model=d_model, n_layers=2)
        self.norm_global = RMSNorm(d_model)
        self.drop_path_global = LayerScale2(d_model, dropout)

        # Hierarchical gating mechanisms
        self.gate1 = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid()
        )
        if self.query_modulation:
            hidden_dim = int(query_mod_hidden_dim) or d_model
            if hidden_dim <= 0:
                raise ValueError("query_mod_hidden_dim must be non-negative")
            self.query_modulation_mlp = nn.Sequential(
                nn.Linear(self.query_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2 * d_model),
            )
            nn.init.normal_(self.query_modulation_mlp[-1].weight, std=1e-3)
            nn.init.zeros_(self.query_modulation_mlp[-1].bias)
        else:
            self.query_modulation_mlp = None
        if self.query_aware_gate:
            self.gate1_query = nn.Linear(self.query_dim, d_model, bias=False)
            nn.init.normal_(self.gate1_query.weight, std=1e-3)
        else:
            self.gate1_query = None
        self.gate2_query = None
        
        # Optional local encoder components
        if local_encode:
            # Gate2 only needed when local encoder is enabled
            self.gate2 = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.Sigmoid()
            )
            if self.query_aware_gate:
                self.gate2_query = nn.Linear(
                    self.query_dim, d_model, bias=False
                )
                nn.init.normal_(self.gate2_query.weight, std=1e-3)
            self.norm_local = RMSNorm(d_model)
            self.drop_path_local = LayerScale2(d_model, dropout)
            
            # Choose local encoder based on type parameter
            if local_encoder_type != "transformer":
                raise ValueError(
                    f"Unsupported released local_encoder_type: {local_encoder_type}. "
                    "The code release only supports transformer local encoding."
                )
            self.local_encoder = TransformerEncoder(
                d_model,
                stride=1,
                window_size=local_window_size,
                n_heads=2
            )
            self.ffn = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
        else:
            self.ffn = SwiGLUFFN(d_model, dropout=dropout)

        self.norm_ffn = RMSNorm(d_model)
        self.drop_path_ffn = LayerScale2(d_model, dropout)
        if self.importance_enabled:
            self.importance_predictor = QueryBoundaryImportancePredictor(
                video_dim=d_model,
                query_dim=self.query_dim,
                hidden_dim=importance_hidden_dim,
                alpha=importance_alpha,
                beta=importance_beta,
                gamma=importance_gamma,
            )
        else:
            self.importance_predictor = None
        if self.adaptive_anchor:
            self.adaptive_anchor_allocator = (
                QueryBoundaryAdaptiveAnchorAllocator(
                    target_keep_ratio=self.target_keep_ratio,
                    importance_weighted_pooling=True,
                    allocator_policy=self.allocator_policy,
                    random_seed=self.allocator_random_seed,
                    oracle_boundary_radius=self.oracle_boundary_radius,
                    reallocation_ratio=self.reallocation_ratio,
                )
            )
        else:
            self.adaptive_anchor_allocator = None

    # Query helpers are defined below; the full forward starts after them.
    def _pool_query(
        self,
        query_feat,
        query_mask,
        batch_size,
        dtype,
        device,
    ):
        if not self.query_conditioned or query_feat is None:
            return None
        if query_feat.ndim == 2:
            if query_feat.shape != (batch_size, self.query_dim):
                raise ValueError(
                    "pooled query_feat must have shape ({}, {})".format(
                        batch_size, self.query_dim
                    )
                )
            return query_feat.to(device=device, dtype=dtype)
        if query_feat.ndim != 3:
            raise ValueError(
                "query_feat must have shape (B, Cq), (B, Cq, Lq), "
                "or (B, Lq, Cq)"
            )
        if query_feat.size(0) != batch_size:
            raise ValueError(
                "query batch size {} does not match video batch size {}".format(
                    query_feat.size(0), batch_size
                )
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
            elif query_mask.ndim != 2:
                raise ValueError(
                    "query_mask must have shape (B, 1, Lq) or (B, Lq)"
                )
            if query_mask.shape != (batch_size, query_len):
                raise ValueError(
                    "query_mask shape {} does not match query tokens {}".format(
                        tuple(query_mask.shape), (batch_size, query_len)
                    )
                )
            valid = query_mask.to(device=device, dtype=torch.bool)
        valid_float = valid.unsqueeze(1).to(dtype=dtype)
        denominator = valid_float.sum(dim=-1).clamp(min=1.0)
        return (query_tokens * valid_float).sum(dim=-1) / denominator

    @staticmethod
    def _apply_query_gate(gate, query_projection, features, query_repr):
        gate_input = torch.cat(features, dim=-1)
        if query_projection is None or query_repr is None:
            return gate(gate_input)
        gate_logits = gate[0](gate_input)
        gate_logits = gate_logits + query_projection(query_repr).unsqueeze(1)
        return torch.sigmoid(gate_logits)
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        query_feat: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
        return_importance: bool = False,
        return_importance_debug: bool = False,
        allocator_targets: Optional[torch.Tensor] = None,
        temporal_metadata: Optional[torch.Tensor] = None,
        forced_cut_offsets: Optional[torch.Tensor] = None,
        forced_target_counts: Optional[torch.Tensor] = None,
        state_prefix: Optional[torch.Tensor] = None,
    ):
        B, D, L = x.shape
        device = x.device

        if mask is None:
            mask = torch.ones((B, 1, L), dtype=torch.bool, device=device)
        elif mask.ndim == 2:
            mask = mask.unsqueeze(1)
        if mask.shape != (B, 1, L):
            raise ValueError(
                "mask must have shape ({}, 1, {})".format(B, L)
            )
        mask = mask.to(device=device, dtype=torch.bool)
        query_repr = self._pool_query(
            query_feat, query_mask, B, x.dtype, device
        )
        query_active = self.query_conditioned and query_repr is not None
        if query_active:
            x = x.masked_fill(~mask, 0.)
        else:
            x.masked_fill_(~mask, 0.)
        importance_debug = None
        if self.importance_predictor is not None and query_repr is not None:
            importance_debug = self.importance_predictor(x, query_repr, mask)
        importance_score = (
            importance_debug["importance"]
            if importance_debug is not None else None
        )

        # Generate and interleave anchors
        adaptive_state = None
        if self.adaptive_anchor:
            adaptive_state = self.adaptive_anchor_allocator(
                x,
                mask,
                importance_score,
                allocator_targets=allocator_targets,
                temporal_metadata=temporal_metadata,
                forced_cut_offsets=forced_cut_offsets,
                forced_target_counts=forced_target_counts,
            )
            x_combined = adaptive_state["combined"]
            anchor_positions = adaptive_state["anchor_positions"]
            sequence_positions = adaptive_state["sequence_positions"]
            expanded_mask = adaptive_state["expanded_mask"]
            anchor_mask = adaptive_state["anchor_mask"]
        else:
            x_combined, anchor_positions, expanded_mask, anchor_mask = (
                self._generate_and_interleave_anchors(
                    x, mask, per_sample_mask=query_active
                )
            )

        # Global encoding with residual
        global_norm = self.norm_global(x_combined)
        # StateBridge priming: context prefix tokens initialize the scan
        # state; their outputs are dropped so downstream layout is unchanged.
        if state_prefix is not None:
            global_norm = torch.cat(
                [state_prefix, global_norm], dim=1)
        if self.query_modulation_mlp is not None and query_repr is not None:
            gamma, beta = self.query_modulation_mlp(query_repr).chunk(2, dim=-1)
            global_norm = (
                (1.0 + gamma.unsqueeze(1)) * global_norm
                + beta.unsqueeze(1)
            )
        pre_hydra = (global_norm[:, state_prefix.size(1):, :]
                     if state_prefix is not None else global_norm)
        dt_delta = None
        if (getattr(self, 'qsm_enabled', False)
                and query_repr is not None):
            dt_delta = self.qsm_modulator(query_repr)
        global_out = (self.global_encoder(global_norm, dt_delta=dt_delta)
                      if dt_delta is not None
                      else self.global_encoder(global_norm))
        if state_prefix is not None:
            global_out = global_out[:, state_prefix.size(1):, :]
        global_out = self.drop_path_global(global_out) + x_combined
        global_out.mul_(expanded_mask)  # In-place masking

        # First gated fusion
        gate1_weights = self._apply_query_gate(
            self.gate1,
            self.gate1_query,
            (x_combined, global_out),
            query_repr,
        )
        fusion1_out = gate1_weights * global_out + (1 - gate1_weights) * x_combined

        # Optional local encoding and second fusion
        if self.local_encode:
            local_out = self.local_encoder(
                self.norm_local(fusion1_out).transpose(1, 2),
                expanded_mask.transpose(1, 2)
            )[0].transpose(1, 2)
            local_out = self.drop_path_local(local_out) + fusion1_out
            local_out.mul_(expanded_mask)

            gate2_weights = self._apply_query_gate(
                self.gate2,
                self.gate2_query,
                (fusion1_out, local_out),
                query_repr,
            )
            fused = gate2_weights * local_out + (1 - gate2_weights) * fusion1_out
        else:
            fused = fusion1_out  # skip local encoding

        # Final FFN with residual
        ffn_out = self.ffn(self.norm_ffn(fused))
        final_out = self.drop_path_ffn(ffn_out) + fused
        if self.adaptive_anchor:
            final_out = final_out * expanded_mask.to(final_out.dtype)

        # State-source stash for the QSTB boundary branch.
        if getattr(self, 'stash_state_source', False):
            self._stashed_state = (
                global_out[:, state_prefix.size(1):, :] if
                state_prefix is not None else global_out)
            self._stashed_pre = pre_hydra
            self._stashed_seq_pos = sequence_positions
            self._stashed_seq_mask = mask
        # Extract outputs
        if self.adaptive_anchor:
            importance_score = (
                final_out.new_zeros((B, L))
                if importance_score is None else importance_score
            )
            anchor_out, seq_out = (
                self.adaptive_anchor_allocator.extract_outputs(
                    final_out,
                    anchor_positions,
                    sequence_positions,
                    anchor_mask,
                    mask,
                )
            )
        else:
            anchor_out, seq_out = self._extract_anchor_and_sequence_outputs(
                final_out, anchor_positions, L
            )
        outputs = (anchor_out, seq_out, anchor_mask, mask)
        if self.adaptive_anchor:
            outputs += (
                adaptive_state["assignment_matrix"],
                importance_score,
            )
            if return_importance_debug:
                adaptive_debug = (
                    {} if importance_debug is None
                    else dict(importance_debug)
                )
                adaptive_debug["assignment_matrix"] = (
                    adaptive_state["assignment_matrix"]
                )
                adaptive_debug["anchor_mask"] = anchor_mask
                adaptive_debug["sequence_mask"] = mask
                return outputs + (adaptive_debug, )
            return outputs
        if return_importance_debug:
            return outputs + (importance_debug, )
        if return_importance:
            importance = (
                importance_debug["importance"]
                if importance_debug is not None else None
            )
            return outputs + (importance, )
        return outputs
        
    def forward_before(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, D, L = x.shape
        device = x.device

        if mask is None:
            mask = torch.ones((B, 1, L), dtype=torch.bool, device=device)
        
        x.masked_fill_(~mask, 0.)  # In-place masking

        # Generate and interleave anchors
        x_combined, anchor_positions, expanded_mask, anchor_mask = self._generate_and_interleave_anchors(x, mask)
        
        # Store original input for gating
        original_features = x_combined
        
        # Step 1: Global encoding with residual connection
        global_norm = self.norm_global(x_combined)
        global_out = self.global_encoder(global_norm)
        global_out = x_combined + self.drop_path_global(global_out)
        global_out = global_out * expanded_mask
        
        # Step 2: First gated fusion (input + global)
        gate1_input = torch.cat([original_features, global_out], dim=-1)
        gate1_weights = self.gate1(gate1_input)
        fusion1_out = gate1_weights * global_out + (1 - gate1_weights) * original_features
        
        # Step 3: Optional local encoding and gating
        if self.local_encode:
            local_norm = self.norm_local(fusion1_out)
            local_transposed = local_norm.transpose(1, 2).contiguous()
            # encoder_mask = expanded_mask.transpose(1, 2)
            local_out_transposed = self.local_encoder(local_transposed, expanded_mask.transpose(1, 2))[0]
            local_out = local_out_transposed.transpose(1, 2)
            local_out = fusion1_out + self.drop_path_local(local_out)
            local_out = local_out * expanded_mask
            
            # Step 4: Second gated fusion (fusion1 + local)
            gate2_input = torch.cat([fusion1_out, local_out], dim=-1)
            gate2_weights = self.gate2(gate2_input)
            gate2_out = gate2_weights * local_out + (1 - gate2_weights) * fusion1_out
        else:
            # No local encoder - skip gate2 entirely 
            gate2_out = fusion1_out
        
        # Step 5: Final FFN (always applied)
        ffn_norm = self.norm_ffn(gate2_out)
        ffn_out = self.ffn(ffn_norm)
        final_out = gate2_out + self.drop_path_ffn(ffn_out)
        
        # Extract anchor and sequence outputs
        anchor_out, seq_out = self._extract_anchor_and_sequence_outputs(
            final_out, anchor_positions, L
        )
        
        return anchor_out, seq_out, anchor_mask, mask

class AnchorMambaPoolingBlock(BaseAnchorBlock):
    """
    A two-stage global-local encoder block for sequence modeling that inherits from BaseAnchorBlock.
    
    This class use the BaseAnchorBlock's robust anchor generation and output extraction methods.

    Input:
        x:    Tensor of shape (B, D, L), where
              B = batch size,
              D = embedding dimension,
              L = sequence length (can be odd).
        mask: Tensor of shape (B, 1, L), optional boolean mask.

    Output:
        anchor_out: (B, D, ceil(L / stride)) - downsampled anchors
        seq_out:    (B, D, L)  -- same length as input (padding removed)
        anchor_mask: downsampled version of input mask
        mask: original input mask
    """
    def __init__(
        self, 
        stride: int, 
        d_model: int, 
        nhead: int = 4, 
        local_window_size: int = 5,
        dropout: float = 0.1, 
        ffn_ratio: int = 4,
        use_swiglu: bool = True,
        local_encode: bool = False, 
        pool_method: str = "mean",
        local_encoder_type: str = "transformer",  # Options: 'transformer' or 'mamba',
        mamba_headdim: int = 64, 
        mamba_dstate: int = 64,
        mamba_expand: int = 2,
        mamba_dconv: int = 7,
        bidirectional: bool = True
    ):
        super().__init__(stride=stride, d_model=d_model, pool_method=pool_method, dropout=dropout)
        
        self.local_encode = local_encode

        if bidirectional:
            # Global encoder block 
            self.global_encoder = Hydra(
                d_model=d_model, d_state=mamba_dstate, d_conv=mamba_dconv, expand=mamba_expand,
                use_mem_eff_path=True, headdim=mamba_headdim
            )
        else:
            self.global_encoder = Mamba2(
                d_model=d_model, d_state=mamba_dstate, d_conv=4, expand=mamba_expand,
                headdim=48
            )
        self.norm_global_in = RMSNorm(d_model)
        self.drop_path_global_1 = LayerScale2(d_model, dropout)

        self.norm_ffn_in = RMSNorm(d_model)
        self.drop_path_ffn = LayerScale2(d_model, dropout)

        if local_encode:
            self.norm_local_in = RMSNorm(d_model)
            self.drop_path_local = LayerScale2(d_model, dropout)
            # Choose local encoder based on type parameter
            if local_encoder_type != "transformer":
                raise ValueError(
                    f"Unsupported released local_encoder_type: {local_encoder_type}. "
                    "The code release only supports transformer local encoding."
                )
            self.local_encoder = TransformerEncoder(
                d_model,
                stride=1,
                window_size=local_window_size,
                n_heads=2
            )
            self.ffn = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
        else:
            self.ffn = SwiGLUFFN(d_model, dropout=dropout) if use_swiglu else nn.Sequential(
                nn.Linear(d_model, ffn_ratio * d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_ratio * d_model, d_model),
                nn.Dropout(dropout)
            )

    def forward(                     # ── (B , D , L)
        self,
        x:    torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        B, D, L  = x.shape
        device   = x.device
        s        = self.stride                     # cache once

        # ---------------------------------------------------------------------
        # 0) Ensure mask exists and zero-out padded / invalid tokens *in-place*
        # ---------------------------------------------------------------------
        if mask is None:
            mask = torch.ones((B, 1, L), dtype=torch.bool, device=device)

        x.masked_fill_(~mask, 0.)                  # no new allocation

        # ---------------------------------------------------------------------
        # 1) Interleave anchors  (compact expanded_mask is shape (B, T′) bool)
        # ---------------------------------------------------------------------
        x_combined, anchor_pos, expanded_mask, anchor_mask = \
            self._generate_and_interleave_anchors(x, mask)          # T′ = (s+1)·blocks
        # x_combined : (B, T′, D)     expanded_mask : (B, T′)  bool

        # ---------------------------------------------------------------------
        # 2) Global encoder (Hydra)  —  prenorm + residual
        # ---------------------------------------------------------------------
        g = self.norm_global_in(x_combined)
        g = self.global_encoder(g)                 # (B, T′, D)
        g = g + self.drop_path_global_1(x_combined)   # in-place add if LayerScale2 uses .add_

        # Zero out invalid slots again (they may have acquired values)
        g.masked_fill_(~expanded_mask, 0.)

        # ---------------------------------------------------------------------
        # 3) Optional local encoder  (single transpose pair, mask reused)
        # ---------------------------------------------------------------------
        if self.local_encode:
            l = self.norm_local_in(g)

            l_t = l.transpose(1, 2).contiguous()        # (B, D, T′)
            # local encoder expects key-padding mask shape  (B, 1, T′)
            encoder_mask = expanded_mask.transpose(1, 2)  # (B, 1, T′)
            l_t = self.local_encoder(l_t, encoder_mask)[0]    # (B, D, T′)

            l   = l_t.transpose(1, 2)                    # (B, T′, D)
            g   = g + self.drop_path_local(l)            # residual
        # else: g already holds the right value

        # ---------------------------------------------------------------------
        # 4) Feed-forward network (always applied)
        # ---------------------------------------------------------------------
        z  = self.norm_ffn_in(g)
        z  = self.ffn(z)                                 # (B, T′, D)
        z  = g + self.drop_path_ffn(z)                   # residual

        # ---------------------------------------------------------------------
        # 5) Split back into anchor / sequence streams (view-only extraction)
        # ---------------------------------------------------------------------
        anchor_out, seq_out = self._extract_anchor_and_sequence_outputs(
            z,                                            # (B, T′, D)
            anchor_pos,                                   # anchor positions
            L                                             # trims padding
        )

        return anchor_out, seq_out, anchor_mask, mask

class AnchorPooling(nn.Module):
    """
    Configurable pooling module for extracting anchor tokens from sequence blocks.
    
    Supports multiple pooling strategies for anchor generation:
    - 'mean': Average pooling over sequence blocks (default, fastest)
    - 'max': Max pooling over sequence blocks (good for sparse features)
    - 'attn': Attention pooling with learnable query (CLIP-style, most expressive)
    - 'gated': Gated pooling that adaptively combines mean and max pooling
    
    Args:
        stride: The stride used for downsampling (tokens per anchor block)
        method: Pooling method to use ('mean', 'max', 'attn', or 'gated')
        d_model: Embedding dimension (required for 'attn' and 'gated' methods)
        nhead: Number of attention heads (used for 'attn' method, default: 2)
        dropout: Dropout probability for attention (default: 0.0)
    
    Input:
        x_blk: Tensor of shape (B, D, num_blocks, stride) - sequence blocks
        
    Output:
        anchors: Tensor of shape (B, D, num_blocks) - pooled anchor tokens
    """
    def __init__(self,
                 stride: int,
                 method: str = "mean",
                 d_model: int = 0,
                 nhead: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.stride = stride
        self.method = method

        if method == "mean":
            self.pooler = MeanPooling(stride)
        elif method == "max":
            self.pooler = MaxPooling(stride)
        elif method == "attn":
            if d_model <= 0:
                raise ValueError("d_model and nhead must be provided for attention pooling")
            self.pooler = AttnPooling(stride, d_model, nhead, dropout)
        elif method == "gated":
            if d_model <= 0:
                raise ValueError("d_model must be provided for gated pooling")
            self.pooler = GatedPooling(stride, d_model)
        else:
            raise ValueError(f"Unknown pooling method {method!r}. Supported methods: 'mean', 'max', 'attn'")

    def forward(self, x_blk: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for anchor pooling.
        
        Args:
            x_blk: Input tensor of shape (B, D, num_blocks, stride)
            
        Returns:
            Pooled anchor tokens of shape (B, D, num_blocks)
        """
        return self.pooler(x_blk)

class MeanPooling(nn.Module):
    """Mean pooling along the last dimension"""
    def __init__(self, stride: int):
        super().__init__()
        self.stride = stride
    def forward(self, x_blk: torch.Tensor) -> torch.Tensor:
        # x_blk: (B, D, num_blocks, stride)
        # return x_blk.mean(dim=-1)  # → (B, D, num_blocks)
        return F.avg_pool1d(x_blk, kernel_size=self.stride, stride=self.stride)
        

class MaxPooling(nn.Module):
    """Max pooling along the last dimension"""
    def __init__(self, stride: int):
        super().__init__()
        self.stride = stride
    def forward(self, x_blk: torch.Tensor) -> torch.Tensor:
        # return x_blk.max(dim=-1).values
        return F.max_pool1d(x_blk, kernel_size=self.stride, stride=self.stride)
    
class GatedPooling(nn.Module):
    """
    Gated pooling that adaptively combines mean and max pooling operations.
    
    This module learns channel-wise gates to dynamically blend mean pooling
    and max pooling based on the input features. This allows the model to
    adaptively choose between averaging (for distributed features) and 
    max pooling (for sparse/salient features) on a per-channel basis.
    
    Args:
        d_model: Model embedding dimension
        
    Input:
        x: Tensor of shape (B, D, N, S) - N windows of length S
        
    Output:
        pooled: Tensor of shape (B, D, N) - gated pooled features
    """
    def __init__(self, stride: int, d_model: int):
        super().__init__()
        # Conv1d on the channel dimension: (2D) -> D
        self.gate_proj = nn.Conv1d(
            in_channels=2 * d_model,
            out_channels=d_model,
            kernel_size=1,
            groups=1,           # depth‑wise would be groups=d_model
            bias=True
        )
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        F.max_pool1d(x, kernel_size=self.stride, stride=self.stride)
        μ   = F.avg_pool1d(x, kernel_size=self.stride, stride=self.stride) # (B, D, N)
        m   = F.max_pool1d(x, kernel_size=self.stride, stride=self.stride) # (B, D, N)
        cat = torch.cat([μ, m], dim=1)  # (B, 2D, N)
        g   = torch.sigmoid(self.gate_proj(cat))  # (B, D, N)
        return g * m + (1.0 - g) * μ

class AttnPooling(nn.Module):
    """
    CLIP-style attention pooling using a learnable query vector.
    
    This module implements attention pooling similar to the approach used in CLIP,
    where a single learnable query vector attends to all tokens in a sequence block
    to produce a pooled representation. This provides more expressive pooling
    compared to simple mean or max pooling.
    
    Args:
        d_model: Model embedding dimension
        nhead: Number of attention heads
        dropout: Dropout probability for attention weights
        
    Input:
        x_blk: Tensor of shape (B, D, num_blocks, stride) - sequence blocks
        
    Output:
        pooled: Tensor of shape (B, D, num_blocks) - attention-pooled features
    """
    def __init__(self, stride: int, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.pool_q = nn.Parameter(torch.randn(1, 1, d_model) / (d_model ** 0.5))  # Initialize with proper scaling
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=False
        )
        self.stride = stride

    def forward(self, x_blk: torch.Tensor) -> torch.Tensor:
        B, D, L2 = x_blk.shape
        num_blocks = L2 // self.stride  # Adjust for stride
        x_blk = x_blk.reshape(B, D, num_blocks, self.stride)
        kv = (
            x_blk
            .permute(3, 0, 2, 1)             # (stride, B, num_blocks, D)
            .reshape(self.stride, B * num_blocks, D)
        )
        q = self.pool_q.expand(-1, B * num_blocks, -1)  # (1, B*num_blocks, D)
        out, _ = self.pool_attn(q, kv, kv)               # (1, B*num_blocks, D)
        return (
            out
            .squeeze(0)                # (B*num_blocks, D)
            .view(B, num_blocks, D)    # (B, num_blocks, D)
            .permute(0, 2, 1)          # (B, D, num_blocks)
        )

def anchor_pooling(x: torch.Tensor, stride: int, kernel_size: Optional[int] = None, method: str = "mean") -> torch.Tensor:
    """
    Legacy pooling function - consider using AnchorPooling class instead.
    
    Args:
        x: Input tensor of shape (B, D, L)
        stride: Stride for pooling
        kernel_size: Size of the pooling kernel (defaults to stride)
        method: Pooling method ('mean' or 'max')
        
    Returns:
        Pooled tensor of shape (B, D, ceil(L/stride))
    """
    if kernel_size is None:
        kernel_size = stride
    if method == "mean":
        return F.avg_pool1d(x, kernel_size=kernel_size, stride=stride, ceil_mode=True)
    elif method == "max":
        return F.max_pool1d(x, kernel_size=kernel_size, stride=stride, ceil_mode=True)
    else:
        raise ValueError(f"Unsupported pooling method: {method}. Use 'mean' or 'max'.")


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm and LlamaRMSNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class LayerScale2(nn.Module):
    """
    Multiple residual by a per-channel scaling factor (and zero init) before adding.
    https://arxiv.org/abs/2103.17239
    
    Args:
        n_channels: Number of channels/features
        pdrop: Dropout probability
        init_scale: Initial scale factor
    """
    def __init__(self, n_channels: int, pdrop: float = 0.1, init_scale: float = 1e-4):
        super().__init__()
        # Shape (1, 1, n_channels) for (batch, seq_len, channels) tensor format
        self.scale = nn.Parameter(init_scale * torch.ones((1, 1, n_channels)))
        self.pdrop = pdrop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer scaling and stochastic depth"""
        return drop_path(self.scale.to(x.dtype) * x, self.pdrop, self.training)
    
def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Stochastic Depth per sample.
    
    Args:
        x: Input tensor
        drop_prob: Probability of dropping a path
        training: Whether in training mode
        
    Returns:
        Output after applying stochastic depth
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    x = x.div(keep_prob) * mask.floor_()
    return x

def downsample_mask(mask: torch.Tensor, stride: int = 2) -> torch.Tensor:
    """
    Downsample a boolean mask using max pooling with ceiling rounding.
    
    Args:
        mask: Boolean mask of shape (B, 1, L)
        stride: Downsampling factor
        
    Returns:
        Downsampled mask of shape (B, 1, ceil(L/stride))
    """
    # Convert boolean mask to float (0.0 and 1.0)
    mask_float = mask.float()
    
    # Apply max pooling with ceil_mode=True to handle odd lengths
    downsampled_mask_float = F.max_pool1d(
        mask_float, 
        kernel_size=stride, 
        stride=stride, 
        ceil_mode=True  # This handles the ceiling rounding
    )
    
    # Convert back to boolean
    downsampled_mask = downsampled_mask_float.bool()
    
    return downsampled_mask

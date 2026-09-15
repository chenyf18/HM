import torch
import torch.nn as nn

from mamba_ssm import Mamba2


class QueryConditionedFPNMambaAdapter(nn.Module):
    """Query-conditioned temporal refinement for selected fused FPN levels."""

    def __init__(
        self,
        d_model,
        text_dim,
        num_levels,
        selected_levels=(2, 3, 4, 5, 6, 7),
        d_state=16,
        d_conv=3,
        expand=1,
        headdim=32,
        modulation_limit=0.1,
        bidirectional=True,
        use_mem_eff_path=True,
    ):
        super().__init__()

        selected_levels = tuple(sorted(set(int(i) for i in selected_levels)))
        if not selected_levels:
            raise ValueError("selected_levels must contain at least one FPN level")
        if selected_levels[0] < 0 or selected_levels[-1] >= num_levels:
            raise ValueError(
                "selected_levels {} are invalid for {} FPN levels".format(
                    selected_levels, num_levels
                )
            )
        if d_model % headdim != 0:
            raise ValueError("d_model must be divisible by headdim")

        self.d_model = d_model
        self.num_levels = num_levels
        self.selected_levels = selected_levels
        self.modulation_limit = float(modulation_limit)
        self.bidirectional = bool(bidirectional)

        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.query_proj = nn.Linear(text_dim, d_model)
        self.level_proj = nn.Linear(num_levels, d_model, bias=False)
        self.conditioner = nn.Sequential(
            nn.GELU(),
            nn.Linear(d_model, 3 * d_model),
        )
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            use_mem_eff_path=use_mem_eff_path,
        )
        self.out_projs = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(num_levels)]
        )
        self.register_buffer(
            "level_codes", torch.eye(num_levels), persistent=False
        )

        # The residual branch starts as an exact no-op while retaining normal
        # initialization in the conditioner and state-space mixer.
        for proj in self.out_projs:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    @staticmethod
    def _masked_text_mean(text, text_mask):
        mask = text_mask.to(dtype=text.dtype)
        denom = mask.sum(dim=-1).clamp(min=1.0)
        return (text * mask).sum(dim=-1) / denom

    @staticmethod
    def _reverse_valid_prefix(sequence, mask):
        """Reverse valid tokens per sample without moving right padding."""
        batch_size, seq_len, channels = sequence.shape
        positions = torch.arange(seq_len, device=sequence.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        lengths = mask.squeeze(1).long().sum(dim=-1, keepdim=True)
        reverse_positions = lengths - 1 - positions
        gather_positions = torch.where(
            positions < lengths, reverse_positions, positions
        ).clamp(min=0)
        return sequence.gather(
            1, gather_positions.unsqueeze(-1).expand(-1, -1, channels)
        )

    def _condition(self, pooled_text, level_idx):
        level_code = self.level_codes[level_idx].to(dtype=pooled_text.dtype)
        condition = self.query_proj(pooled_text) + self.level_proj(level_code)
        scale, shift, gate = self.conditioner(condition).chunk(3, dim=-1)
        scale = self.modulation_limit * torch.tanh(scale)
        shift = self.modulation_limit * torch.tanh(shift)
        return scale, shift, torch.sigmoid(gate)

    def forward(self, features, masks, text, text_mask):
        if len(features) != self.num_levels or len(masks) != self.num_levels:
            raise ValueError(
                "expected {} FPN levels, got {} features and {} masks".format(
                    self.num_levels, len(features), len(masks)
                )
            )

        pooled_text = self._masked_text_mean(text, text_mask)
        outputs = list(features)

        for level_idx in self.selected_levels:
            x = features[level_idx]
            mask = masks[level_idx].to(dtype=torch.bool)
            mask_sequence = mask.transpose(1, 2)
            mask_float = mask_sequence.to(dtype=x.dtype)
            sequence = x.transpose(1, 2) * mask_float

            scale, shift, gate = self._condition(pooled_text, level_idx)
            normalized = self.norm(sequence)
            modulated = (
                (1.0 + scale.unsqueeze(1)) * normalized
                + shift.unsqueeze(1)
            ) * mask_float

            forward_out = self.mamba(modulated.contiguous())
            if self.bidirectional:
                reverse_in = self._reverse_valid_prefix(modulated, mask)
                reverse_out = self.mamba(reverse_in.contiguous())
                reverse_out = self._reverse_valid_prefix(reverse_out, mask)
                mixed = 0.5 * (forward_out + reverse_out)
            else:
                mixed = forward_out

            mixed = mixed * mask_float
            delta = self.out_projs[level_idx](mixed).transpose(1, 2)
            refined = x + gate.unsqueeze(-1) * delta
            outputs[level_idx] = refined * mask.to(dtype=x.dtype)

        return tuple(outputs)

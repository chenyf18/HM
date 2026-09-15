import torch
import torch.nn.functional as F


def _normalize_temporal_mask(mask, batch_size, length, name, device):
    if mask.ndim == 3:
        if mask.size(1) != 1:
            raise ValueError(
                f"{name} must have shape (B, 1, L) or (B, L)"
            )
        mask = mask[:, 0]
    elif mask.ndim != 2:
        raise ValueError(
            f"{name} must have shape (B, 1, L) or (B, L)"
        )
    if mask.shape != (batch_size, length):
        raise ValueError(
            f"{name} shape {tuple(mask.shape)} does not match "
            f"({batch_size}, {length})"
        )
    return mask.to(device=device, dtype=torch.bool)


def assignment_aware_anchor_contrastive(
    anchors: torch.Tensor,
    seq_tokens: torch.Tensor,
    anchor_mask: torch.Tensor,
    seq_mask: torch.Tensor,
    assignment_matrix: torch.Tensor,
    projector: torch.nn.Module,
    temperature: float = 0.07,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Multi-positive InfoNCE defined by anchor-token assignments."""
    if anchors.ndim != 3 or seq_tokens.ndim != 3:
        raise ValueError(
            "anchors and seq_tokens must have shape (B, D, L)"
        )
    batch_size, channels, anchor_len = anchors.shape
    seq_batch, seq_channels, seq_len = seq_tokens.shape
    if (seq_batch, seq_channels) != (batch_size, channels):
        raise ValueError(
            "anchor and sequence batch/channel dimensions must match"
        )
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if eps <= 0:
        raise ValueError("eps must be positive")
    expected_shape = (batch_size, anchor_len, seq_len)
    if (
        assignment_matrix.ndim != 3
        or assignment_matrix.shape != expected_shape
    ):
        raise ValueError(
            "assignment_matrix must have shape {}, got {}".format(
                expected_shape, tuple(assignment_matrix.shape)
            )
        )
    if not torch.isfinite(assignment_matrix).all():
        raise ValueError(
            "assignment_matrix must contain only finite values"
        )
    if (
        torch.any(assignment_matrix < 0)
        or torch.any(assignment_matrix > 1)
    ):
        raise ValueError(
            "assignment_matrix values must be in [0, 1]"
        )

    anchor_valid = _normalize_temporal_mask(
        anchor_mask,
        batch_size,
        anchor_len,
        "anchor_mask",
        anchors.device,
    )
    sequence_valid = _normalize_temporal_mask(
        seq_mask,
        batch_size,
        seq_len,
        "seq_mask",
        anchors.device,
    )
    assignment = assignment_matrix.to(
        device=anchors.device, dtype=torch.float32
    )
    effective_assignment = assignment * sequence_valid[:, None].to(
        assignment.dtype
    )
    valid_anchor = anchor_valid & (
        effective_assignment > 0
    ).any(dim=-1)
    valid_indices = torch.nonzero(valid_anchor, as_tuple=False)
    if valid_indices.numel() == 0:
        return (
            anchors.float().sum() + seq_tokens.float().sum()
        ) * 0.0

    projected_anchors = projector(
        anchors.transpose(1, 2).reshape(
            batch_size * anchor_len, channels
        )
    )
    projected_tokens = projector(
        seq_tokens.transpose(1, 2).reshape(
            batch_size * seq_len, channels
        )
    )
    projected_anchors = F.normalize(
        projected_anchors.float(), p=2, dim=-1
    ).reshape(batch_size, anchor_len, -1)
    projected_tokens = F.normalize(
        projected_tokens.float(), p=2, dim=-1
    ).reshape(batch_size, seq_len, -1)

    # Compute each video's anchor-token similarities once. Expanding tokens
    # per valid anchor would require O(B * A * T * D) intermediate memory.
    logits = torch.bmm(
        projected_anchors,
        projected_tokens.transpose(1, 2),
    ) / float(temperature)

    batch_indices = valid_indices[:, 0]
    anchor_indices = valid_indices[:, 1]
    logits = logits[batch_indices, anchor_indices]

    selected_valid = sequence_valid.index_select(
        0, batch_indices
    )
    selected_assignment = effective_assignment[
        batch_indices, anchor_indices
    ]
    denominator_logits = logits.masked_fill(
        ~selected_valid, float("-inf")
    )
    positive_logits = logits + torch.log(
        selected_assignment.clamp_min(eps)
    )
    positive_logits = positive_logits.masked_fill(
        selected_assignment <= 0, float("-inf")
    )

    log_numerator = torch.logsumexp(
        positive_logits, dim=-1
    )
    log_denominator = torch.logsumexp(
        denominator_logits, dim=-1
    )
    return (log_denominator - log_numerator).mean()


def contrastive_subsample_negative_mp(
    anchors: torch.Tensor,
    seq_tokens: torch.Tensor,
    anchor_mask: torch.Tensor,
    seq_mask: torch.Tensor,
    projector: torch.nn.Module,
    radius: int = 0,
    temperature: float = 0.07,
    neg_ratio: float = 0.20,
    gap_ratio: float = 0.30,
    hard_neg: bool = False,
    cross_video_neg: bool = False,
) -> torch.Tensor:
    """
    Multi-positive InfoNCE with either hard-negative top-M or random-negative M.
    This is the only released ds_contrast helper used by the current code release.
    """
    assert radius >= 0
    batch_size, dim, anchor_len = anchors.shape
    seq_len = seq_tokens.shape[2]
    device = anchors.device

    num_anchors = batch_size * anchor_len
    flat_anchors = anchors.permute(0, 2, 1).reshape(num_anchors, dim)
    flat_seq = seq_tokens.permute(0, 2, 1).reshape(batch_size * seq_len, dim)
    flat_anchor_mask = anchor_mask.squeeze(1).reshape(num_anchors)
    flat_seq_mask = seq_mask.squeeze(1).reshape(batch_size * seq_len)

    vid_id = torch.arange(num_anchors, device=device) // anchor_len
    time_id = torch.arange(num_anchors, device=device) % anchor_len

    z_seq = F.normalize(projector(flat_seq), dim=-1)

    positive_features = []
    positive_masks = []
    for offset in range(-radius, radius + 1):
        shifted_t = time_id + offset
        in_range = (0 <= shifted_t) & (shifted_t < anchor_len)

        even_idx = (vid_id * seq_len + 2 * shifted_t).clamp(0, batch_size * seq_len - 1)
        odd_idx = (even_idx + 1).clamp(0, batch_size * seq_len - 1)

        even_mask = in_range & flat_seq_mask[even_idx]
        odd_mask = in_range & flat_seq_mask[odd_idx]

        positive_features.extend([z_seq[even_idx], z_seq[odd_idx]])
        positive_masks.extend([even_mask, odd_mask])

    z_pos = torch.stack(positive_features, dim=1)
    pos_mask = torch.stack(positive_masks, dim=1)

    valid = flat_anchor_mask & (pos_mask.sum(1) > 0)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)

    valid_idxs = valid.nonzero(as_tuple=False).view(-1)
    z_anchor = F.normalize(projector(flat_anchors[valid_idxs]), dim=-1)
    z_pos = z_pos[valid_idxs]
    pos_mask = pos_mask[valid_idxs]

    valid_vid = vid_id[valid_idxs]
    valid_t = time_id[valid_idxs]
    num_valid = z_anchor.size(0)

    sim_pos = (z_anchor.unsqueeze(1) * z_pos).sum(-1) / temperature
    exp_pos = torch.exp(sim_pos) * pos_mask.float()
    exp_pos = exp_pos.sum(1)

    sim_all = torch.matmul(z_anchor, z_anchor.T) / temperature
    eye = torch.eye(num_valid, device=device, dtype=torch.bool)
    same_vid = valid_vid.unsqueeze(0).eq(valid_vid.unsqueeze(1))

    gap = max(1, int(gap_ratio * anchor_len))
    near = same_vid & (torch.abs(valid_t.unsqueeze(0) - valid_t.unsqueeze(1)) <= gap)

    if cross_video_neg:
        candidates = ~(eye | near)
    else:
        candidates = same_vid & ~(eye | near)

    exp_sim = torch.exp(sim_all) * candidates.float()

    valid_neg_counts = candidates.sum(1)
    pos_counts = pos_mask.sum(1)
    samples_per_row = (neg_ratio * pos_counts.float()).ceil().long().clamp(min=1)
    samples_per_row = torch.minimum(samples_per_row, valid_neg_counts)
    max_samples = int(samples_per_row.max())

    if hard_neg:
        masked_sim = exp_sim.masked_fill(~candidates, -1)
        _, top_idx = torch.topk(masked_sim, k=max_samples, dim=1, largest=True)
    else:
        random_scores = torch.rand_like(exp_sim) * candidates.float() - (~candidates).float()
        _, top_idx = torch.topk(random_scores, k=max_samples, dim=1, largest=True)

    row_ids = torch.arange(num_valid, device=device).unsqueeze(1).expand_as(top_idx)
    keep_mask = torch.arange(max_samples, device=device).unsqueeze(0) < samples_per_row.unsqueeze(1)
    selected = torch.zeros_like(exp_sim, dtype=torch.bool)
    selected[row_ids, top_idx] = keep_mask

    neg_sum = (exp_sim * selected.float()).sum(1)
    loss = -torch.log(exp_pos / (exp_pos + neg_sum + 1e-9))
    return loss.mean()


def legacy_consistent_assignment_acc(
    anchors: torch.Tensor,
    seq_tokens: torch.Tensor,
    anchor_mask: torch.Tensor,
    seq_mask: torch.Tensor,
    assignment_matrix: torch.Tensor,
    projector: torch.nn.Module,
    temperature: float = 0.07,
    eps: float = 1e-9,
    neg_ratio: float = 0.20,
    gap_ratio: float = 0.30,
    radius: int = 0,
    hard_neg: bool = False,
    cross_video_neg: bool = False,
) -> torch.Tensor:
    """Assignment ACC that reduces exactly to ``contrastive_subsample_negative_mp``
    under strict stride-2 grouping.

    Only the positive-slot construction differs from the legacy function:
    instead of the two fixed positional slots {2j, 2j+1}, each valid anchor
    uses its assigned member tokens (ascending order, zero-padded to the
    batch-wide maximum group size). Numerator, negative pool, temporal gap
    exclusion, subsampling, temperature, normalization, and the final mean
    reduction are bit-compatible with the legacy implementation.
    """
    import torch.nn.functional as F

    if radius not in (0, None):
        raise ValueError(
            "legacy_consistent_assignment_acc supports radius=0 only"
        )
    batch_size, channels, anchor_len = anchors.shape
    seq_len = seq_tokens.shape[2]
    anchor_valid = _normalize_temporal_mask(
        anchor_mask, batch_size, anchor_len, "anchor_mask", anchors.device
    )
    sequence_valid = _normalize_temporal_mask(
        seq_mask, batch_size, seq_len, "seq_mask", anchors.device
    )
    assignment = assignment_matrix.to(
        device=anchors.device, dtype=torch.float32
    )
    effective = assignment * sequence_valid[:, None].to(assignment.dtype)
    valid_anchor = anchor_valid & (effective > 0).any(dim=-1)
    valid_indices = torch.nonzero(valid_anchor, as_tuple=False)
    if valid_indices.numel() == 0:
        return (anchors.float().sum() + seq_tokens.float().sum()) * 0.0

    z_anchor_flat = F.normalize(
        projector(anchors.transpose(1, 2).reshape(
            batch_size * anchor_len, channels)).float(), dim=-1
    )
    z_seq_flat = F.normalize(
        projector(seq_tokens.transpose(1, 2).reshape(
            batch_size * seq_len, channels)).float(), dim=-1
    )

    batch_indices = valid_indices[:, 0]
    anchor_indices = valid_indices[:, 1]
    member_rows = effective[batch_indices, anchor_indices]
    counts = member_rows.sum(dim=-1)
    slot_count = int(counts.max())
    keys = member_rows * (torch.arange(
        seq_len, device=anchors.device, dtype=torch.float32
    ).unsqueeze(0) + 1.0)
    topk = torch.topk(keys, k=slot_count, dim=1).values.sort(dim=1).values
    member_tokens = (topk.round().long() - 1).clamp_min(0)
    slot_mask = topk > 0

    flat_member = (
        batch_indices.unsqueeze(1) * seq_len + member_tokens
    ).clamp(0, batch_size * seq_len - 1)
    z_pos = z_seq_flat[flat_member]
    pos_mask = slot_mask

    vid_id = batch_indices
    time_id = anchor_indices
    num_valid = valid_indices.size(0)

    z_anchor = z_anchor_flat[
        batch_indices * anchor_len + anchor_indices
    ]
    sim_pos = (z_anchor.unsqueeze(1) * z_pos).sum(-1) / temperature
    exp_pos = (torch.exp(sim_pos) * pos_mask.float()).sum(1)

    sim_all = torch.matmul(z_anchor, z_anchor.T) / temperature
    eye = torch.eye(num_valid, device=anchors.device, dtype=torch.bool)
    same_vid = vid_id.unsqueeze(0).eq(vid_id.unsqueeze(1))
    anchor_len_max = int(anchor_indices.max()) + 1 if num_valid else 1
    gap = max(1, int(gap_ratio * anchor_len))
    near = same_vid & (
        torch.abs(time_id.unsqueeze(0) - time_id.unsqueeze(1)) <= gap
    )
    if cross_video_neg:
        candidates = ~(eye | near)
    else:
        candidates = same_vid & ~(eye | near)

    exp_sim = torch.exp(sim_all) * candidates.float()
    valid_neg_counts = candidates.sum(1)
    pos_counts = pos_mask.sum(1)
    samples_per_row = (neg_ratio * pos_counts.float()).ceil().long().clamp(min=1)
    samples_per_row = torch.minimum(samples_per_row, valid_neg_counts)
    if int(samples_per_row.max()) == 0:
        return (anchors.float().sum() + seq_tokens.float().sum()) * 0.0
    max_samples = int(samples_per_row.max())

    if hard_neg:
        masked_sim = exp_sim.masked_fill(~candidates, -1)
        _, top_idx = torch.topk(masked_sim, k=max_samples, dim=1, largest=True)
    else:
        random_scores = torch.rand_like(exp_sim) * candidates.float() - (
            ~candidates
        ).float()
        _, top_idx = torch.topk(random_scores, k=max_samples, dim=1, largest=True)

    row_ids = torch.arange(num_valid, device=anchors.device).unsqueeze(1)
    col_ids = torch.arange(max_samples, device=anchors.device).unsqueeze(0)
    keep_mask = col_ids < samples_per_row.unsqueeze(1)
    selected = torch.zeros_like(exp_sim, dtype=torch.bool)
    selected[row_ids.expand_as(top_idx), top_idx] = keep_mask

    neg_sum = (exp_sim * selected.float()).sum(1)
    loss = -torch.log(exp_pos / (exp_pos + neg_sum + eps))
    _ = anchor_len_max
    return loss.mean()

#!/usr/bin/env python3
"""Region statistics and GT information preservation for allocator variants.

Directive sections 5 and 6: for a trained adaptive variant, measure how each
allocator policy distributes resolution across boundary / foreground /
background regions and whether GT information survives grouping.

Usage:
  python tools/allocator_region_diagnostics.py \
    --config opts/research_allocator_AL_learned.yaml \
    --checkpoint experiments/allocator_diagnosis/seed_1/AL/models/last.pth \
    --output experiments/allocator_diagnosis/seed_1/AL/region_diagnostics.json \
    [--num-videos 200] [--split val]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "hydra"), str(ROOT / "tools")]

from libs import load_opt  # noqa: E402
from libs.modeling.model import make_models_net  # noqa: E402
from libs.data import make_dataset  # noqa: E402
from tools import research_validate_adaptive as rva  # noqa: E402


def load_state(model, checkpoint_path):
    payload = torch.load(checkpoint_path, map_location="cpu")
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    load_compatible = getattr(model, "load_compatible_state_dict", None)
    if load_compatible is not None:
        missing, unexpected = load_compatible(state)
        print("compatible load; missing={} unexpected={}".format(
            len(missing), len(unexpected)))
    else:
        model.load_state_dict(state)


def select_samples(opt, split, count, seed):
    data_opt = dict(opt["train"]["data"])
    data_opt.update(dict(opt["eval"]["data"]))
    data_opt["split"] = split
    data_opt["crop_ratio"] = None
    dataset = make_dataset(data_opt, num_epochs=1, is_training=False)
    order = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(seed)
    ).tolist()
    samples, indices = [], []
    for index in order:
        if len(samples) >= count:
            break
        samples.append(dataset[index])
        indices.append(index)
    return samples, indices


def region_masks(metadata, targets, radius=1.0):
    """Per-token boolean masks (foreground, boundary) from level metadata."""
    start = metadata[..., 0]
    end = metadata[..., 1]
    center = metadata[..., 2]
    span = metadata[..., 3].abs().clamp_min(1e-6)
    active = metadata[..., 3].abs() > 0
    gs = targets[:, 0].unsqueeze(1)
    ge = targets[:, 1].unsqueeze(1)
    foreground = (end >= gs) & (start <= ge) & active
    boundary_radius = span * radius
    boundary = foreground & (
        ((center - gs).abs() <= boundary_radius)
        | ((center - ge).abs() <= boundary_radius)
    )
    return foreground, boundary, active


def level_statistics(assignment, metadata, targets, radius):
    assignment = assignment.float()
    membership = assignment.argmax(dim=1)
    # Tokens per anchor, then the owning group's size at each token.
    tokens_per_anchor = assignment.sum(dim=2)
    group_size = tokens_per_anchor.gather(1, membership)
    foreground, boundary, active = region_masks(metadata, targets, radius)
    background = active & ~foreground

    anchor_count = assignment.size(1)
    anchor_start = torch.full(
        (metadata.size(0), anchor_count), float("inf")
    )
    anchor_end = torch.full(
        (metadata.size(0), anchor_count), float("-inf")
    )
    batch_index = torch.arange(metadata.size(0)).unsqueeze(1)
    anchor_start.scatter_reduce_(
        1, membership, metadata[..., 0], reduce="amin", include_self=True
    )
    anchor_end.scatter_reduce_(
        1, membership, metadata[..., 1], reduce="amax", include_self=True
    )
    anchor_span = (anchor_end - anchor_start).clamp_min(0)

    def region_stats(mask):
        if not bool(mask.any()):
            return None
        sizes = group_size[mask].float()
        spans = anchor_span.unsqueeze(-1).expand_as(assignment).gather(
            1, membership.unsqueeze(1)
        ).squeeze(1)[mask].float()
        owner = membership[mask].float()
        distinct = len(set(owner.tolist()))
        return {
            "tokens": int(mask.sum()),
            "mean_group_size": float(sizes.mean()),
            "mean_anchor_span": float(spans.mean()),
            "compression_ratio": float(mask.sum()) / max(distinct, 1),
        }

    coverage_rows = []
    for b in range(assignment.size(0)):
        if bool(foreground[b].any()):
            owners = membership[b][foreground[b]]
            coverage_rows.append(int(len(set(owners.tolist()))))
        else:
            coverage_rows.append(0)

    return {
        "boundary": region_stats(boundary),
        "foreground_only": region_stats(foreground & ~boundary),
        "background": region_stats(background),
        "gt_anchor_coverage_mean": float(
            sum(coverage_rows)
        )
        / max(len(coverage_rows), 1),
        "coverage_per_sample": coverage_rows,
        "assignment_finite": bool(torch.isfinite(assignment).all()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-videos", type=int, default=200)
    parser.add_argument("--split", default="val")
    parser.add_argument("--seed", type=int, default=1234567891)
    parser.add_argument("--radius", type=float, default=1.0)
    args = parser.parse_args()

    opt = load_opt(str(args.config), is_training=True)
    torch.manual_seed(args.seed)
    model = make_models_net(opt)
    load_state(model, args.checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    samples, indices = select_samples(opt, args.split, args.num_videos, args.seed)
    batch = rva.batchify(samples, opt)

    video = batch["video"].to(device)
    video_mask = batch["video_mask"].to(device)
    text = batch["text"].to(device)
    text_mask = batch["text_mask"].to(device)
    text_size = batch["text_size"].to(device)
    targets = rva.targets_for_model(batch, opt).to(device)

    oracle = opt["model"]["vid_net"].get("allocator_policy") == "oracle"
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"
    ):
        # Mirror HieraMamba._forward_earlyfusion preprocessing exactly.
        packed_text = text
        packed_text_mask = text_mask
        if packed_text.ndim == 4:
            packed_text = torch.cat(
                [t[:k] for t, k in zip(packed_text, text_size)]
            )
        if packed_text_mask.ndim == 3:
            packed_text_mask = torch.cat(
                [t[:k] for t, k in zip(packed_text_mask, text_size)]
            )
        packed_text, packed_text_mask = model.encode_text(
            packed_text, packed_text_mask
        )
        window_mask = video_mask.unsqueeze(1) if video_mask.ndim == 2 else video_mask
        window, window_mask = model.project_video(video, window_mask)
        window, window_mask = model.fusion(
            window, window_mask, packed_text, packed_text_mask, text_size
        )
        outputs = model.encode_video(
            window,
            window_mask,
            query_feat=(
                packed_text if model.query_conditioned else None
            ),
            query_mask=(
                packed_text_mask if model.query_conditioned else None
            ),
            text_size=text_size,
            return_anchor_assignments=True,
            allocator_targets=targets if oracle else None,
        )
    _, _, _, _, assignments = outputs
    assignments = list(assignments)
    metadata_levels = list(model.vid_net.last_temporal_metadata)

    counts = batch["text_size"].tolist()
    sources = []
    for sample_index, count in enumerate(counts):
        sources.extend([sample_index] * int(count))
    sources = torch.tensor(sources)
    expanded_targets = targets[torch.tensor(sources)]
    expanded_durations = (
        batch["raw_targets"][:, 1] - batch["raw_targets"][:, 0]
    )[sources]
    q1, q2 = (
        float(torch.quantile(expanded_durations.float(), 1.0 / 3.0)),
        float(torch.quantile(expanded_durations.float(), 2.0 / 3.0)),
    )

    per_level = []
    for level, (assignment, metadata) in enumerate(
        zip(assignments, metadata_levels)
    ):
        assignment = assignment.detach().float().cpu()
        metadata = metadata.detach().float().cpu()
        stats = level_statistics(
            assignment,
            metadata,
            expanded_targets.detach().float().cpu(),
            args.radius,
        )
        coverage = stats.pop("coverage_per_sample")
        buckets = {"short": [], "medium": [], "long": []}
        for row, value in enumerate(coverage):
            duration = float(expanded_durations[row])
            key = (
                "short"
                if duration <= float(q1)
                else "medium"
                if duration <= float(q2)
                else "long"
            )
            buckets[key].append(value)
        stats["level"] = level
        stats["shape"] = list(assignment.shape)
        stats["gt_survival_by_duration_bucket"] = {
            key: {
                "mean_anchors": sum(values) / len(values) if values else None,
                "count": len(values),
            }
            for key, values in buckets.items()
        }
        per_level.append(stats)

    report = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "num_samples": len(samples),
        "duration_buckets_seconds": {
            "short_lt": float(q1),
            "medium_lt": float(q2),
        },
        "per_level": per_level,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")

    pooled = {}
    for key in ("boundary", "foreground_only", "background"):
        values = [
            item[key]["mean_group_size"]
            for item in per_level
            if item.get(key)
        ]
        pooled[key + "_mean_group_size_pooled"] = (
            sum(values) / len(values) if values else None
        )
    print(json.dumps(pooled, indent=2))
    print("report:", output)


if __name__ == "__main__":
    main()

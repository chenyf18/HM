#!/usr/bin/env python3
"""Pre-training research validation for Adaptive HieraMamba.

This tool never launches train.py and never writes a checkpoint. Missing
Ego4D assets are a blocked result; another dataset is not substituted.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch


# The repository copy is the only formal validator. Keep the root explicit so
# imports always resolve to this checkout when invoked from another directory.
HIERAMAMBA_ROOT = os.path.abspath(
    os.environ.get("HIERAMAMBA_ROOT", "/data/nh/hieramamba-main")
)
if not os.path.isdir(HIERAMAMBA_ROOT):
    raise RuntimeError(
        "HieraMamba checkout was not found at {}. Set HIERAMAMBA_ROOT."
        .format(HIERAMAMBA_ROOT)
    )
ROOT = HIERAMAMBA_ROOT
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "hydra"))

from libs import load_opt
from libs.data import make_dataset
from libs.modeling import (
    MultiScaleMaskedContrastive,
    ctr_diou_loss,
    ctr_giou_loss,
    make_optimizer,
    sigmoid_focal_loss,
)
from libs.modeling.model import make_models_net
from libs.modeling.query_boundary_importance_loss import (
    QueryBoundaryImportanceLoss,
    make_boundary_supervision_loss,
)
from libs.modeling.temporal_coordinates import (
    AdaptiveTemporalPointGenerator,
    decode_offsets,
    encode_offsets,
)


METADATA_NAMES = (
    "start",
    "end",
    "center",
    "span",
    "regression_scale",
)

COORDINATE_UNIT = (
    "dataset feature-grid token after configured feature downsampling"
)


def _numeric_distribution(values):
    """Return stable numeric summaries while retaining nonfinite counts."""
    values = torch.as_tensor(values).detach().double().cpu().flatten()
    finite = values[torch.isfinite(values)]
    result = {
        "count": int(values.numel()),
        "nonfinite": int(values.numel() - finite.numel()),
    }
    keys = ("min", "max", "mean", "std", "p50", "p90", "p95", "p99")
    if finite.numel() == 0:
        result.update({key: None for key in keys})
    else:
        result.update({
            "min": float(finite.min()),
            "max": float(finite.max()),
            "mean": float(finite.mean()),
            "std": float(finite.std(unbiased=False)),
            "p50": float(torch.quantile(finite, 0.50)),
            "p90": float(torch.quantile(finite, 0.90)),
            "p95": float(torch.quantile(finite, 0.95)),
            "p99": float(torch.quantile(finite, 0.99)),
        })
    return result


def finite_stats(values):
    """Summarize values with the common validator statistics contract."""
    result = _numeric_distribution(values)
    tensor = torch.as_tensor(values).detach().flatten()
    result["finite_count"] = int(torch.isfinite(tensor).sum())
    return result


def distribution(values):
    return _numeric_distribution(values)

def tensor_tree_finite(value):
    if torch.is_tensor(value):
        if value.is_floating_point() or value.is_complex():
            return bool(torch.isfinite(value).all())
        return True
    if isinstance(value, dict):
        return all(tensor_tree_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(tensor_tree_finite(item) for item in value)
    return True


def finite_number(value):
    return value is not None and math.isfinite(float(value))


def optional_less(last, first):
    if not finite_number(first) or not finite_number(last):
        return None
    return bool(last < first)


def optional_greater(last, first):
    if not finite_number(first) or not finite_number(last):
        return None
    return bool(last > first)


def linear_trend(values):
    values = [float(value) for value in values if finite_number(value)]
    if len(values) < 2:
        return None
    x = torch.arange(len(values), dtype=torch.float64)
    y = torch.as_tensor(values, dtype=torch.float64)
    x = x - x.mean()
    denominator = x.square().sum()
    if denominator <= 0:
        return 0.0
    return float((x * (y - y.mean())).sum() / denominator)


def prefix_mask(mask, name="mask"):
    if not torch.is_tensor(mask):
        raise ValueError("{} must be a tensor".format(name))
    if mask.ndim == 3:
        if mask.size(1) != 1:
            raise ValueError(
                "{} must have shape (B, 1, T) or (B, T)".format(name)
            )
        mask = mask[:, 0]
    elif mask.ndim != 2:
        raise ValueError(
            "{} must have shape (B, 1, T) or (B, T)".format(name)
        )
    mask = mask.bool()
    valid_counts = mask.sum(dim=-1)
    expected = torch.arange(
        mask.size(-1), device=mask.device
    ).unsqueeze(0) < valid_counts.unsqueeze(1)
    if not torch.equal(mask, expected):
        raise ValueError(
            "{} contains a hole; temporal masks must be left-aligned "
            "prefixes".format(name)
        )
    return mask


def required_data_paths(opt):
    data_opt = opt["train"]["data"]
    video_dirs = data_opt.get("vid_feat_dir", [])
    if not isinstance(video_dirs, (list, tuple)):
        video_dirs = [video_dirs]
    return [data_opt.get("anno_file"), data_opt.get("text_feat_dir")] + list(
        video_dirs
    )


def missing_data_paths(opt):
    return [
        path
        for path in required_data_paths(opt)
        if not path or not os.path.exists(path)
    ]


def select_real_samples(opt, count, seed):
    """Select deterministic samples, preferring a multi-query video."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data_opt = dict(opt["train"]["data"])
    data_opt["crop_ratio"] = None
    dataset = make_dataset(data_opt, num_epochs=1, is_training=True)
    if not len(dataset):
        raise RuntimeError("training dataset is empty")
    order = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(seed)
    ).tolist()
    selected = []
    selected_indices = []
    for index in order:
        sample = dataset[index]
        if len(sample["text"]) >= 2:
            selected.append(sample)
            selected_indices.append(index)
            break
    for index in order:
        if len(selected) >= count:
            break
        if index in selected_indices:
            continue
        selected.append(dataset[index])
        selected_indices.append(index)
    return selected, selected_indices


def batchify(samples, opt):
    """Match TrainerOriginal padding and query flattening semantics."""
    batch_size = len(samples)
    video_dim = samples[0]["vid"].size(0)
    video_len = int(opt["model"]["max_vid_len"]) * int(
        opt["model"].get("vid_stride", 1)
    )
    text_len = int(opt["model"]["max_text_len"])
    text_dim = samples[0]["text"][0].size(0)
    text_size = torch.as_tensor(
        [len(sample["text"]) for sample in samples], dtype=torch.long
    )
    max_queries = int(text_size.max())
    video = samples[0]["vid"].new_zeros(
        batch_size, video_dim, video_len
    )
    video_mask = torch.zeros(batch_size, video_len, dtype=torch.bool)
    text = samples[0]["text"][0].new_zeros(
        batch_size, max_queries, text_dim, text_len
    )
    text_mask = torch.zeros(
        batch_size, max_queries, text_len, dtype=torch.bool
    )
    targets = []
    query_records = []
    output_index = 0
    for sample_index, sample in enumerate(samples):
        valid_len = sample["vid"].size(-1)
        if valid_len > video_len:
            raise ValueError("real sample exceeds configured training length")
        video[sample_index, :, :valid_len].copy_(sample["vid"])
        video_mask[sample_index, :valid_len] = True
        for query_index, query in enumerate(sample["text"]):
            valid_text_len = min(query.size(-1), text_len)
            text[sample_index, query_index, :, :valid_text_len].copy_(
                query[:, :valid_text_len]
            )
            text_mask[sample_index, query_index, :valid_text_len] = True
            targets.append(sample["target"][query_index])
            query_records.append({
                "output_index": output_index,
                "sample_index": sample_index,
                "query_index": query_index,
                "vid_id": sample.get("vid_id"),
                "sentence_id": (
                    sample.get("text_ids", ())[query_index]
                    if query_index < len(sample.get("text_ids", ()))
                    else None
                ),
                "text": sample.get("sentences", ())[query_index]
                if query_index < len(sample.get("sentences", ())) else None,
            })
            output_index += 1
    return {
        "video": video,
        "video_mask": video_mask,
        "text": text,
        "text_mask": text_mask,
        "text_size": text_size,
        "raw_targets": torch.stack(targets),
        "queries": query_records,
        "video_ids": [sample.get("vid_id") for sample in samples],
        "sentence_ids": [query.get("sentence_id") for query in query_records],
    }


def targets_for_model(batch, opt):
    adaptive = opt["model"]["vid_net"].get("adaptive_anchor", False)
    divisor = 1.0 if adaptive else float(
        opt["model"].get("vid_stride", 1)
    )
    return batch["raw_targets"] / divisor


def model_forward(opt, model, batch, importance=False, assignments=False):
    device = next(model.parameters()).device
    raw = model(
        batch["video"].to(device),
        batch["video_mask"].to(device),
        batch["text"].to(device),
        batch["text_mask"].to(device),
        batch["text_size"].to(device),
        return_importance_debug=importance,
        return_anchor_assignments=assignments,
    )
    base = raw[:8]
    extras = raw[8:]
    extra_index = 0
    importance_debug = None
    assignment_matrices = None
    if importance:
        importance_debug = extras[extra_index]
        extra_index += 1
    if assignments:
        assignment_matrices = extras[extra_index]
        extra_index += 1
    if len(extras) != extra_index:
        raise RuntimeError(
            "unexpected model output count: expected {} extras, got {}".format(
                extra_index, len(extras)
            )
        )
    return {
        "logits": base[0],
        "logits2": base[1],
        "offsets": base[2],
        "masks": base[3],
        "fpn": base[4],
        "sequence_masks": base[5],
        "anchors": base[6],
        "anchor_masks": base[7],
        "importance": importance_debug,
        "assignments": assignment_matrices,
        "temporal_metadata": (
            tuple(item.detach().clone() for item in model.vid_net.last_temporal_metadata)
            if getattr(model.vid_net, "last_temporal_metadata", None) is not None
            else None
        ),
        "targets": targets_for_model(batch, opt).to(device),
    }



def _slice_batch_for_sample(batch, sample_index):
    query_indices = [
        index for index, query in enumerate(batch["queries"])
        if query["sample_index"] == sample_index
    ]
    query_count = len(query_indices)
    if query_count <= 0:
        raise ValueError("sample has no valid queries")
    queries = []
    for output_index, query_index in enumerate(query_indices):
        query = dict(batch["queries"][query_index])
        query["output_index"] = output_index
        query["sample_index"] = 0
        queries.append(query)
    return {
        "video": batch["video"][sample_index:sample_index + 1].clone(),
        "video_mask": batch["video_mask"][sample_index:sample_index + 1].clone(),
        "text": batch["text"][sample_index:sample_index + 1, :query_count].clone(),
        "text_mask": batch["text_mask"][sample_index:sample_index + 1, :query_count].clone(),
        "text_size": torch.tensor([query_count], dtype=batch["text_size"].dtype),
        "raw_targets": batch["raw_targets"].index_select(0, torch.as_tensor(query_indices)),
        "queries": queries,
        "video_ids": [batch["video_ids"][sample_index]],
        "sentence_ids": [query.get("sentence_id") for query in queries],
    }


def _query_select(value, query_indices):
    if torch.is_tensor(value):
        if value.ndim and value.size(0) >= max(query_indices) + 1:
            return value.index_select(0, torch.as_tensor(query_indices, device=value.device))
        return value
    if isinstance(value, dict):
        return {key: _query_select(item, query_indices) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_query_select(item, query_indices) for item in value)
    return value


def _pad_tensor_to_shape(value, shape):
    """Zero-pad a tensor so mixed-batch padding can be compared explicitly."""
    if tuple(value.shape) == tuple(shape):
        return value
    padded = torch.zeros(tuple(shape), dtype=value.dtype, device=value.device)
    slices = tuple(slice(0, size) for size in value.shape)
    padded[slices] = value
    return padded


def _compare_output_trees(expected, actual, path="output"):
    errors = []
    max_error = 0.0
    if torch.is_tensor(expected) and torch.is_tensor(actual):
        if expected.ndim != actual.ndim:
            return [path + ":shape_mismatch"], None
        comparison_shape = tuple(
            max(expected_size, actual_size)
            for expected_size, actual_size in zip(expected.shape, actual.shape)
        )
        expected = _pad_tensor_to_shape(expected, comparison_shape)
        actual = _pad_tensor_to_shape(actual, comparison_shape)
        if expected.is_floating_point() or expected.is_complex():
            delta = (expected.detach() - actual.detach()).abs().float()
            max_error = float(delta.max()) if delta.numel() else 0.0
            if not torch.allclose(expected, actual, atol=1e-3, rtol=1e-4):
                errors.append(path + ":value_mismatch")
        elif not torch.equal(expected, actual):
            errors.append(path + ":value_mismatch")
        return errors, max_error
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            if key not in expected or key not in actual:
                errors.append(path + "." + str(key) + ":missing")
            else:
                child_errors, child_max = _compare_output_trees(expected[key], actual[key], path + "." + str(key))
                errors.extend(child_errors)
                if child_max is not None:
                    max_error = max(max_error, child_max)
        return errors, max_error
    if isinstance(expected, (tuple, list)) and isinstance(actual, (tuple, list)):
        if len(expected) != len(actual):
            return [path + ":length_mismatch"], None
        for index, (left, right) in enumerate(zip(expected, actual)):
            child_errors, child_max = _compare_output_trees(left, right, path + "[{}]".format(index))
            errors.extend(child_errors)
            if child_max is not None:
                max_error = max(max_error, child_max)
        return errors, max_error
    if expected != actual:
        errors.append(path + ":value_mismatch")
    return errors, max_error


def _mask_isolation_tensor(value, sequence_mask=None, anchor_mask=None):
    """Mask invalid temporal/anchor tails before comparing mixed-batch outputs."""
    if not torch.is_tensor(value) or value.ndim < 2:
        return value
    valid = None
    if torch.is_tensor(sequence_mask):
        sequence = sequence_mask.bool()
        if sequence.ndim == 3:
            sequence = sequence[:, 0, :]
        if sequence.ndim == 2 and value.size(0) == sequence.size(0):
            if value.ndim == 2 and value.shape[-1] == sequence.shape[-1]:
                sequence_valid = sequence
            elif value.ndim >= 3 and value.shape[-1] == sequence.shape[-1]:
                sequence_valid = sequence.reshape(
                    sequence.size(0), *([1] * (value.ndim - 2)), sequence.size(-1)
                )
            elif value.ndim >= 3 and value.shape[-2] == sequence.shape[-1]:
                sequence_valid = sequence.unsqueeze(-1)
            else:
                sequence_valid = None
            if sequence_valid is not None:
                valid = sequence_valid if valid is None else (valid & sequence_valid)
    if torch.is_tensor(anchor_mask):
        anchor = anchor_mask.bool()
        if anchor.ndim == 3:
            anchor = anchor[:, 0, :]
        if anchor.ndim == 2 and value.size(0) == anchor.size(0):
            if value.ndim == 2 and value.shape[-1] == anchor.shape[-1]:
                anchor_valid = anchor
            elif value.ndim >= 3 and value.shape[-1] == anchor.shape[-1]:
                anchor_valid = anchor.reshape(
                    anchor.size(0), *([1] * (value.ndim - 2)), anchor.size(-1)
                )
            elif value.ndim >= 3 and value.shape[-2] == anchor.shape[-1]:
                anchor_valid = anchor.unsqueeze(-1)
            else:
                anchor_valid = None
            if anchor_valid is not None:
                valid = anchor_valid if valid is None else (valid & anchor_valid)
    if valid is None:
        return value
    return torch.where(valid, value, torch.zeros_like(value))


def _mask_isolation_tree(value, sequence_masks=None, anchor_masks=None):
    """Apply per-level validity masks to an output tree."""
    if torch.is_tensor(value):
        return _mask_isolation_tensor(value, sequence_masks, anchor_masks)
    if isinstance(value, dict):
        return {
            key: _mask_isolation_tree(item, sequence_masks, anchor_masks)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        if (
            isinstance(sequence_masks, (tuple, list))
            and len(sequence_masks) == len(value)
        ):
            return type(value)(
                _mask_isolation_tree(
                    item,
                    sequence_masks[index],
                    anchor_masks[index]
                    if isinstance(anchor_masks, (tuple, list))
                    and len(anchor_masks) == len(value) else anchor_masks,
                )
                for index, item in enumerate(value)
            )
        return type(value)(
            _mask_isolation_tree(item, sequence_masks, anchor_masks)
            for item in value
        )
    return value


def batch_isolation_diagnostic(opt, model, batch, batched_outputs):
    """Check mixed-batch and one-sample forwards for query-conditioned leakage."""
    records = []
    for sample_index in range(int(batch["video"].size(0))):
        single = _slice_batch_for_sample(batch, sample_index)
        with torch.no_grad():
            single_outputs = model_forward(
                opt, model, single,
                importance=batched_outputs.get("importance") is not None,
                assignments=batched_outputs.get("assignments") is not None,
            )
        query_indices = [
            query["output_index"] for query in batch["queries"]
            if query["sample_index"] == sample_index
        ]
        comparable_keys = (
            "logits", "logits2", "offsets", "masks", "fpn",
            "sequence_masks", "anchors", "anchor_masks", "importance",
            "assignments", "temporal_metadata",
        )
        mixed = {key: batched_outputs.get(key) for key in comparable_keys}
        single_named = {key: single_outputs.get(key) for key in comparable_keys}
        mixed_selected = {
            key: _query_select(value, query_indices)
            if value is not None else None
            for key, value in mixed.items()
        }
        mixed_sequence_masks = _query_select(
            batched_outputs.get("sequence_masks"), query_indices
        )
        mixed_anchor_masks = _query_select(
            batched_outputs.get("anchor_masks"), query_indices
        )
        single_sequence_masks = single_outputs.get("sequence_masks")
        single_anchor_masks = single_outputs.get("anchor_masks")
        # Invalid tails can legitimately have arbitrary logits/features; the
        # masks and all valid values still must agree exactly within tolerance.
        mixed_selected = {
            key: _mask_isolation_tree(
                value, mixed_sequence_masks, mixed_anchor_masks
            )
            if value is not None else None
            for key, value in mixed_selected.items()
        }
        single_named = {
            key: _mask_isolation_tree(
                value, single_sequence_masks, single_anchor_masks
            )
            if value is not None else None
            for key, value in single_named.items()
        }
        errors = []
        max_error = 0.0
        for key in comparable_keys:
            if mixed_selected[key] is None and single_named[key] is None:
                continue
            child_errors, child_max = _compare_output_trees(
                mixed_selected[key], single_named[key], key
            )
            errors.extend(child_errors)
            if child_max is not None:
                max_error = max(max_error, child_max)
        records.append({
            "sample_index": sample_index,
            "video_id": batch["video_ids"][sample_index],
            "query_indices": query_indices,
            "max_abs_error": max_error,
            "errors": sorted(set(errors)),
            "masked_invalid_tails": True,
            "passed": not errors,
        })
    return {
        "sample_count": len(records),
        "samples": records,
        "masked_invalid_tails": True,
        "passed": all(record["passed"] for record in records),
    }

def build_points(opt, model, masks):
    device = next(model.parameters()).device
    configured_adaptive = bool(
        opt["model"]["vid_net"].get("adaptive_anchor", False)
    )
    model_adaptive = bool(getattr(model.vid_net, "adaptive_anchor", False))
    if configured_adaptive != model_adaptive:
        raise RuntimeError(
            "adaptive_anchor differs between loaded config and model"
        )
    if configured_adaptive:
        metadata = getattr(model.vid_net, "last_temporal_metadata", None)
        if metadata is None:
            raise RuntimeError(
                "adaptive_anchor=true but temporal metadata is unavailable"
            )
        generator = AdaptiveTemporalPointGenerator(
            max_seq_len=opt["pt_gen"]["max_seq_len"],
            num_fpn_levels=opt["pt_gen"]["num_fpn_levels"],
            regression_range=opt["pt_gen"].get("regression_range", 4),
            sigma=opt["pt_gen"].get("sigma", 1),
            input_stride=opt["model"]["vid_net"].get("stride", 1),
        ).to(device)
        return generator(metadata, masks)
    from libs.modeling import PtGenerator
    generator = PtGenerator(
        **opt["pt_gen"], allow_fixed_stride=True
    ).to(device)
    return generator([mask.size(-1) for mask in masks])


def sample_points(level_points, batch_index):
    return level_points[batch_index] if level_points.ndim == 3 else level_points


def annotate_points(points, target, center_sampling="radius", radius=1.5):
    center = points[:, 0]
    start = torch.minimum(target[0], target[1])
    end = torch.maximum(target[0], target[1])
    left = center - start
    right = end - center
    offsets = encode_offsets(points, target.unsqueeze(0))[0]
    if center_sampling == "radius":
        middle = 0.5 * (start + end)
        sample_radius = points[:, 3] * radius
        lower = (middle - sample_radius).clamp(min=start)
        upper = (middle + sample_radius).clamp(max=end)
        inside_window = (center - lower > 0) & (upper - center > 0)
    else:
        lower, upper = start, end
        inside_window = (left > 0) & (right > 0)
    distance = torch.maximum(left, right)
    inside_range = (distance >= points[:, 1]) & (distance < points[:, 2])
    return inside_window & inside_range, offsets, lower, upper, inside_range


def assignment_errors(matrix, input_mask, anchor_mask):
    input_valid = prefix_mask(input_mask, "input_mask")
    anchor_valid = prefix_mask(anchor_mask, "anchor_mask")
    if matrix.ndim != 3:
        return ["assignment_matrix_wrong_rank"]
    expected_shape = (
        input_valid.size(0), anchor_valid.size(1), input_valid.size(1)
    )
    if tuple(matrix.shape) != expected_shape:
        return ["assignment_matrix_shape_mismatch"]
    errors = []
    if not bool(torch.isfinite(matrix).all()):
        errors.append("assignment_matrix_nonfinite")
    if matrix.numel() and (
        bool((matrix < 0).any()) or bool((matrix > 1).any())
    ):
        errors.append("assignment_matrix_out_of_range")
    finite_values = matrix[torch.isfinite(matrix)]
    if finite_values.numel():
        binary = torch.isclose(
            finite_values, torch.zeros_like(finite_values)
        ) | torch.isclose(
            finite_values, torch.ones_like(finite_values)
        )
        if not bool(binary.all()):
            errors.append("assignment_matrix_not_binary")
    membership = matrix > 0
    if torch.any(membership & ~input_valid.unsqueeze(1)):
        errors.append("padding_assigned")
    group_sizes = membership.sum(-1)
    if torch.any(anchor_valid & (group_sizes == 0)):
        errors.append("empty_valid_group")
    if torch.any(~anchor_valid & (group_sizes != 0)):
        errors.append("padded_anchor_has_members")
    coverage = membership.sum(1)
    if torch.any(input_valid & (coverage != 1)):
        errors.append("valid_token_not_covered_exactly_once")
    if torch.any(~input_valid & (coverage != 0)):
        errors.append("padding_token_covered")
    for batch_index in range(matrix.size(0)):
        valid_anchor_count = int(anchor_valid[batch_index].sum())
        previous_end = -1
        for anchor_index in range(valid_anchor_count):
            indices = torch.nonzero(
                membership[batch_index, anchor_index], as_tuple=False
            ).flatten()
            if indices.numel() == 0:
                continue
            expected = torch.arange(
                indices[0], indices[-1] + 1, device=indices.device
            )
            if not torch.equal(indices, expected):
                errors.append("group_has_internal_hole")
            if int(indices[0]) != previous_end + 1:
                errors.append("gap_or_overlap_between_groups")
            previous_end = int(indices[-1])
        valid_length = int(input_valid[batch_index].sum())
        if valid_length and valid_anchor_count == 0:
            errors.append("valid_tokens_without_valid_anchor")
        if valid_length and previous_end != valid_length - 1:
            errors.append("groups_do_not_reach_last_valid_token")
    return sorted(set(errors))


def metadata_stats(metadata, mask):
    rows = metadata[prefix_mask(mask)]
    return {
        name: finite_stats(rows[:, index])
        for index, name in enumerate(METADATA_NAMES)
    }


def metadata_errors(name, metadata, mask):
    valid = prefix_mask(mask, name + "_mask")
    errors = []
    if metadata.ndim != 3 or metadata.size(-1) != len(METADATA_NAMES):
        return [name + "_metadata_shape_mismatch"]
    if metadata.shape[:2] != valid.shape:
        return [name + "_metadata_mask_shape_mismatch"]
    if not bool(torch.isfinite(metadata).all()):
        errors.append(name + "_metadata_nonfinite")
    padded = metadata[~valid]
    if padded.numel() and not torch.allclose(
        padded, torch.zeros_like(padded), atol=1e-6, rtol=0.0
    ):
        errors.append(name + "_padding_metadata_nonzero")
    for batch_index in range(metadata.size(0)):
        rows = metadata[batch_index, valid[batch_index]]
        if not rows.numel():
            continue
        if not bool(torch.isfinite(rows).all()):
            errors.append(name + "_valid_metadata_nonfinite")
            continue
        if rows.size(0) > 1:
            if not bool(torch.all(rows[1:, 2] > rows[:-1, 2])):
                errors.append(name + "_center_not_strictly_monotonic")
            support_delta = rows[1:, 0] - rows[:-1, 1]
            if not torch.allclose(
                support_delta, torch.zeros_like(support_delta),
                atol=1e-5, rtol=1e-5,
            ):
                errors.append(name + "_support_gap_or_overlap")
        if not bool(torch.all(rows[:, 1] > rows[:, 0])):
            errors.append(name + "_nonpositive_support")
        if not bool(torch.all((rows[:, 2] >= rows[:, 0]) & (rows[:, 2] <= rows[:, 1]))):
            errors.append(name + "_center_outside_support")
        if not bool(torch.all(rows[:, 3] > 0)):
            errors.append(name + "_nonpositive_span")
        if not torch.allclose(
            rows[:, 3], rows[:, 1] - rows[:, 0],
            atol=1e-5, rtol=1e-5,
        ):
            errors.append(name + "_span_inconsistent")
        if not torch.allclose(
            rows[:, 2], 0.5 * (rows[:, 0] + rows[:, 1]),
            atol=1e-5, rtol=1e-5,
        ):
            errors.append(name + "_center_inconsistent")
        if not bool(torch.all(rows[:, 4] > 0)):
            errors.append(name + "_nonpositive_regression_scale")
    return sorted(set(errors))


def _level_raw_values(level_metadata, anchor_metadata, input_valid,
                      anchor_valid, matrix, importance_debug):
    membership = matrix > 0
    valid_lengths = input_valid.sum(-1)
    anchor_numbers = anchor_valid.sum(-1)
    group_sizes = membership.sum(-1)[anchor_valid]
    values = {
        "input_valid_length": valid_lengths.detach().cpu().tolist(),
        "output_anchor_number": anchor_numbers.detach().cpu().tolist(),
        "effective_keep_ratio": (
            anchor_numbers.float() / valid_lengths.clamp(min=1).float()
        ).detach().cpu().tolist(),
        "group_size": group_sizes.detach().cpu().tolist(),
    }
    for prefix, metadata, mask in (
        ("representation", level_metadata, input_valid),
        ("allocated_anchor", anchor_metadata, anchor_valid),
    ):
        rows = metadata[mask]
        for index, name in enumerate(METADATA_NAMES):
            values[prefix + "_" + name] = rows[:, index].detach().cpu().tolist()
    if importance_debug is not None:
        for key, value in (
            ("relevance", importance_debug.get("relevance")),
            ("boundary", importance_debug.get("boundary")),
            ("temporal_change", importance_debug.get("temporal_change")),
            ("final_importance", importance_debug.get("importance")),
        ):
            if value is not None:
                values[key] = value[input_valid].detach().cpu().tolist()
    return values


def _stats_from_raw(raw, key):
    return finite_stats([item for row in raw for item in row.get(key, [])])


def temporal_diagnostics(model, outputs):
    metadata = outputs.get("temporal_metadata")
    if metadata is None:
        metadata = model.vid_net.last_temporal_metadata
    allocated = []
    report = {
        "coordinate_unit": COORDINATE_UNIT,
        "levels": [],
        "errors": [],
        "batch_count": int(outputs["sequence_masks"][0].size(0))
        if outputs.get("sequence_masks") else 0,
    }
    counts = (
        len(metadata) if metadata is not None else 0,
        len(outputs["sequence_masks"]),
        len(outputs["anchor_masks"]),
        len(outputs["assignments"]),
    )
    if len(set(counts)) != 1:
        return {
            "levels": [],
            "errors": ["temporal_level_count_mismatch:{}".format(counts)],
            "passed": False,
        }, tuple()
    for level, values in enumerate(zip(
        metadata,
        outputs["sequence_masks"],
        outputs["anchor_masks"],
        outputs["assignments"],
    )):
        level_metadata, input_mask, anchor_mask, matrix = values
        input_valid = prefix_mask(input_mask)
        anchor_valid = prefix_mask(anchor_mask)
        anchor_metadata = AdaptiveTemporalPointGenerator.propagate(
            level_metadata, matrix, input_valid, anchor_valid
        )
        allocated.append(anchor_metadata)
        errors = assignment_errors(matrix, input_valid, anchor_valid)
        for name, tensor, valid in (
            ("representation", level_metadata, input_valid),
            ("allocated_anchor", anchor_metadata, anchor_valid),
        ):
            errors.extend(metadata_errors(name, tensor, valid))
        importance_stats = None
        debug = None
        if outputs.get("importance") is not None:
            debug = outputs["importance"][level]
            importance_stats = {
                key: finite_stats(debug[key][input_valid])
                for key in ("relevance", "boundary", "temporal_change", "importance")
                if key in debug and debug[key] is not None
            }
            if not tensor_tree_finite(debug):
                errors.append("importance_debug_nonfinite")
        raw_values = _level_raw_values(
            level_metadata, anchor_metadata, input_valid, anchor_valid,
            matrix, debug,
        )
        report["levels"].append({
            "level": level,
            "coordinate_unit": COORDINATE_UNIT,
            "input_valid_length": finite_stats(raw_values["input_valid_length"]),
            "output_anchor_number": finite_stats(raw_values["output_anchor_number"]),
            "valid_length": [int(x) for x in input_valid.sum(-1).tolist()],
            "valid_length_stats": finite_stats(input_valid.sum(-1)),
            "anchor_number": [int(x) for x in anchor_valid.sum(-1).tolist()],
            "anchor_number_stats": finite_stats(anchor_valid.sum(-1)),
            "effective_keep_ratio": finite_stats(raw_values["effective_keep_ratio"]),
            "actual_keep_ratio": finite_stats(raw_values["effective_keep_ratio"]),
            "group_size": finite_stats(raw_values["group_size"]),
            "group_size_per_sample": [
                [int(x) for x in (matrix[batch_index, :int(anchor_valid[batch_index].sum()), :] > 0).sum(-1).tolist()]
                for batch_index in range(matrix.size(0))
            ],
            "representation_metadata": metadata_stats(level_metadata, input_valid),
            "allocated_anchor_metadata": metadata_stats(anchor_metadata, anchor_valid),
            "importance": importance_stats,
            "raw_values": raw_values,
            "errors": sorted(set(errors)),
            "passed": not errors,
        })
        report["errors"].extend(
            "level_{}:{}".format(level, error) for error in sorted(set(errors))
        )
    report["cross_level_propagation"] = []
    for level in range(max(0, len(allocated) - 1)):
        expected = allocated[level]
        actual = metadata[level + 1]
        expected_mask = prefix_mask(outputs["anchor_masks"][level])
        actual_mask = prefix_mask(outputs["sequence_masks"][level + 1])
        errors = []
        max_error = None
        if expected.shape != actual.shape:
            errors.append("metadata_shape_mismatch")
        if expected_mask.shape != actual_mask.shape or not torch.equal(expected_mask, actual_mask):
            errors.append("mask_mismatch")
        if not errors:
            delta = (expected - actual).abs()
            max_error = float(delta.max()) if delta.numel() else 0.0
            if not torch.allclose(expected, actual, atol=1e-5, rtol=1e-5):
                errors.append("metadata_value_mismatch")
        report["cross_level_propagation"].append({
            "from_level": level,
            "to_level": level + 1,
            "max_abs_error": max_error,
            "errors": errors,
            "passed": not errors,
        })
        report["errors"].extend(
            "level_{}_to_{}:{}".format(level, level + 1, error)
            for error in errors
        )
    report["errors"] = sorted(set(report["errors"]))
    report["passed"] = not report["errors"]
    return report, tuple(allocated)


def aggregate_temporal_diagnostics(reports):
    """Aggregate raw per-batch diagnostics without averaging summaries."""
    reports = [report for report in reports if report]
    if not reports:
        return {"batch_count": 0, "levels": [], "errors": ["no_batches"], "passed": False}
    levels = []
    max_levels = max((len(report.get("levels", [])) for report in reports), default=0)
    errors = []
    for level in range(max_levels):
        level_reports = [report["levels"][level] for report in reports if len(report.get("levels", [])) > level]
        raw = {}
        for item in level_reports:
            for key, values in item.get("raw_values", {}).items():
                raw.setdefault(key, []).extend(values)
        aggregate = {
            "level": level,
            "batch_count": len(level_reports),
            "input_valid_length": finite_stats(raw.get("input_valid_length", [])),
            "output_anchor_number": finite_stats(raw.get("output_anchor_number", [])),
            "effective_keep_ratio": finite_stats(raw.get("effective_keep_ratio", [])),
            "group_size": finite_stats(raw.get("group_size", [])),
            "metadata": {
                name: finite_stats(raw.get("allocated_anchor_" + name, []))
                for name in METADATA_NAMES
            },
            "importance": {
                key: finite_stats(raw.get(key, []))
                for key in ("relevance", "boundary", "temporal_change", "final_importance")
                if key in raw
            },
            "raw_count": {key: len(value) for key, value in raw.items()},
        }
        levels.append(aggregate)
    for report in reports:
        errors.extend(report.get("errors", []))
    return {
        "coordinate_unit": COORDINATE_UNIT,
        "batch_count": len(reports),
        "levels": levels,
        "errors": sorted(set(errors)),
        "passed": not errors,
    }

def toy_temporal_round_trip():
    """Validate adaptive geometry without requiring dataset assets or CUDA."""
    mask = torch.ones(1, 12, dtype=torch.bool)
    metadata = AdaptiveTemporalPointGenerator.make_initial_metadata(mask)
    groups = ((0, 4), (4, 5), (5, 6), (6, 8), (8, 11), (11, 12))
    assignment = torch.zeros(1, len(groups), 12)
    token_to_anchor = []
    for anchor_index, (start, end) in enumerate(groups):
        assignment[0, anchor_index, start:end] = 1.0
        token_to_anchor.extend([anchor_index] * (end - start))
    anchor_mask = torch.ones(1, 1, len(groups), dtype=torch.bool)
    propagated = AdaptiveTemporalPointGenerator.propagate(
        metadata, assignment, mask, anchor_mask
    )
    generator = AdaptiveTemporalPointGenerator(
        max_seq_len=12,
        num_fpn_levels=1,
        regression_range=4,
        sigma=0.5,
    )
    points = generator((propagated,), (anchor_mask,))[0]
    target = torch.tensor([[2.25, 10.75]], dtype=torch.float32)
    offsets = encode_offsets(points, target)
    decoded = decode_offsets(points, offsets)
    expected = target[:, None, :].expand_as(decoded)
    max_error = float((decoded - expected).abs().max())
    fixed_stride_fail_fast = False
    try:
        from libs.modeling import PtGenerator
        PtGenerator(
            max_seq_len=12,
            num_fpn_levels=1,
            allow_fixed_stride=False,
        )([12])
    except RuntimeError:
        fixed_stride_fail_fast = True
    errors = assignment_errors(assignment, mask, anchor_mask)
    if not bool(torch.isfinite(propagated).all()):
        errors.append("propagated_metadata_nonfinite")
    if not bool(torch.isfinite(points).all()):
        errors.append("adaptive_points_nonfinite")
    if max_error > 1e-5:
        errors.append("encode_decode_round_trip_failed")
    if not fixed_stride_fail_fast:
        errors.append("fixed_stride_generator_did_not_fail_fast")
    errors = sorted(set(errors))
    return {
        "coordinate_unit": COORDINATE_UNIT,
        "input_length": 12,
        "target_anchor_budget": 6,
        "groups_half_open": [list(group) for group in groups],
        "token_to_anchor": token_to_anchor,
        "assignment_matrix": assignment[0].int().tolist(),
        "anchor_metadata": [
            {name: float(row[index]) for index, name in enumerate(METADATA_NAMES)}
            for row in propagated[0]
        ],
        "regression_scale_is_not_span": bool(
            torch.any(
                ~torch.isclose(propagated[0, :, 3], propagated[0, :, 4])
            )
        ),
        "gt_segment": target[0].tolist(),
        "normalized_offsets": offsets[0].tolist(),
        "decoded_segments": decoded[0].tolist(),
        "max_round_trip_abs_error": max_error,
        "fixed_stride_fail_fast": fixed_stride_fail_fast,
        "errors": errors,
        "passed": not errors,
    }


def positive_point_diagnostic(opt, points, outputs, batch, seed, max_queries=None):
    """Inspect several real queries and verify GT encode/decode exactly."""
    query_count = len(outputs["targets"])
    if query_count == 0:
        return {
            "coordinate_unit": COORDINATE_UNIT,
            "queries": [], "query_count": 0,
            "errors": ["no_queries"], "passed": False,
        }
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(query_count, generator=generator).tolist()
    if max_queries is not None:
        order = order[:max(1, min(int(max_queries), query_count))]
    query_reports = []
    for selected in order:
        target = outputs["targets"][selected]
        center_sampling = opt["train"].get("center_sampling", "radius")
        radius = float(opt["train"].get("center_sampling_radius", 1.5))
        records = []
        errors = []
        decoded_errors = []
        for level, (level_points, mask) in enumerate(zip(points, outputs["masks"])):
            point = sample_points(level_points, selected)
            valid = prefix_mask(mask)[selected]
            labels, offsets, lower, upper, inside_range = annotate_points(
                point, target, center_sampling, radius
            )
            positive = valid & labels
            for point_index in torch.nonzero(positive, as_tuple=False).flatten().tolist():
                row = point[point_index]
                encoded = offsets[point_index]
                decoded = decode_offsets(
                    row.unsqueeze(0), encoded.unsqueeze(0)
                )[0]
                target_ordered = torch.stack((torch.minimum(target[0], target[1]), torch.maximum(target[0], target[1])))
                round_trip_error = float((decoded - target_ordered).abs().max())
                decoded_errors.append(round_trip_error)
                lower_value = lower if lower.ndim == 0 else lower[point_index]
                upper_value = upper if upper.ndim == 0 else upper[point_index]
                range_min, range_max = float(row[1]), float(row[2])
                records.append({
                    "level": level,
                    "point_index": point_index,
                    "center": float(row[0]),
                    "span": float(row[7]),
                    "regression_scale": float(row[3]),
                    "encoded_left": float(encoded[0]),
                    "encoded_right": float(encoded[1]),
                    "encoded_target": [float(x) for x in encoded],
                    "decoded_segment": [float(x) for x in decoded],
                    "round_trip_absolute_error": round_trip_error,
                    "center_sampling_window": [float(lower_value), float(upper_value)],
                    "inside_center_window": bool(row[0] > lower_value and row[0] < upper_value),
                    "regression_range": [range_min, range_max],
                    "inside_regression_range": bool(inside_range[point_index]),
                    "target_distance": float(torch.maximum(
                        (row[0] - target_ordered[0]).abs(),
                        (target_ordered[1] - row[0]).abs(),
                    )),
                })
                if round_trip_error > 1e-5:
                    errors.append("encode_decode_round_trip_failed")
                if not bool(inside_range[point_index]):
                    errors.append("positive_violates_regression_range")
                if not bool(torch.isfinite(row).all() and torch.isfinite(encoded).all()):
                    errors.append("positive_point_nonfinite")
            if positive.any() and not bool(inside_range[positive].all()):
                errors.append("positive_violates_assignment_rule")
        if not records:
            errors.append("selected_gt_has_no_positive_point")
        query = batch["queries"][selected] if selected < len(batch.get("queries", [])) else {}
        query_report = {
            "query_index": selected,
            "video_id": query.get("vid_id"),
            "sentence_id": query.get("sentence_id"),
            "query_text": query.get("text"),
            "query": query,
            "gt_segment": [float(x) for x in target],
            "positive_count": len(records),
            "positive_points": records,
            "max_round_trip_absolute_error": max(decoded_errors) if decoded_errors else None,
            "errors": sorted(set(errors)),
            "passed": not errors,
        }
        query_reports.append(query_report)
    errors = sorted({error for report in query_reports for error in report["errors"]})
    return {
        "coordinate_unit": COORDINATE_UNIT,
        "random_seed": seed,
        "query_count": len(query_reports),
        "queries": query_reports,
        "query": query_reports[0].get("query") if query_reports else None,
        "gt_round_trip_max_abs_error": max(
            (report["max_round_trip_absolute_error"] or 0.0 for report in query_reports),
            default=None,
        ),
        "errors": errors,
        "passed": not errors,
    }

def group_span_by_region(anchor_metadata, outputs, neighborhood):
    """Report span and group-size distributions by GT-relative region."""
    names = ("boundary", "foreground", "background")
    aggregate = {
        name: {"span": [], "group_size": []} for name in names
    }
    levels = []
    for level, (metadata, mask, assignments) in enumerate(zip(
        anchor_metadata, outputs["anchor_masks"], outputs["assignments"]
    )):
        level_values = {name: {"span": [], "group_size": []} for name in names}
        valid = prefix_mask(mask)
        for batch_index, target in enumerate(outputs["targets"]):
            rows = metadata[batch_index, valid[batch_index]]
            if not rows.numel():
                continue
            anchor_count = int(valid[batch_index].sum())
            membership = assignments[batch_index, :anchor_count] > 0
            group_sizes = membership.sum(-1).to(rows.dtype)
            start = torch.minimum(target[0], target[1])
            end = torch.maximum(target[0], target[1])
            center = rows[:, 2]
            boundary = ((center - start).abs() <= neighborhood) | ((center - end).abs() <= neighborhood)
            foreground = (center > start) & (center < end) & ~boundary
            background = ~(boundary | foreground)
            for name, selected in zip(names, (boundary, foreground, background)):
                if selected.any():
                    spans = rows[selected, 3]
                    sizes = group_sizes[selected]
                    level_values[name]["span"].extend(spans.detach().cpu().tolist())
                    level_values[name]["group_size"].extend(sizes.detach().cpu().tolist())
                    aggregate[name]["span"].extend(spans.detach().cpu().tolist())
                    aggregate[name]["group_size"].extend(sizes.detach().cpu().tolist())
        levels.append({
            "level": level,
            "regions": {
                name: {
                    "span": distribution(level_values[name]["span"]),
                    "group_size": distribution(level_values[name]["group_size"]),
                }
                for name in names
            },
        })
    aggregate_report = {
        name: {
            "span": distribution(aggregate[name]["span"]),
            "group_size": distribution(aggregate[name]["group_size"]),
        }
        for name in names
    }
    span_means = [aggregate_report[name]["span"]["mean"] for name in names]
    size_means = [aggregate_report[name]["group_size"]["mean"] for name in names]
    span_available = all(value is not None for value in span_means)
    size_available = all(value is not None for value in size_means)
    span_ordering = bool(span_means[0] < span_means[1] < span_means[2]) if span_available else None
    size_ordering = bool(size_means[0] < size_means[1] < size_means[2]) if size_available else None
    return {
        "coordinate_unit": COORDINATE_UNIT,
        "boundary_neighborhood_tokens": neighborhood,
        "region_precedence": list(names),
        "aggregate": aggregate_report,
        "per_level": levels,
        "boundary_span": aggregate_report["boundary"]["span"],
        "foreground_span": aggregate_report["foreground"]["span"],
        "background_span": aggregate_report["background"]["span"],
        "boundary_group_size": aggregate_report["boundary"]["group_size"],
        "foreground_group_size": aggregate_report["foreground"]["group_size"],
        "background_group_size": aggregate_report["background"]["group_size"],
        "expectation_available": span_available and size_available,
        "span_ordering_boundary_lt_foreground_lt_background": span_ordering,
        "group_size_ordering_boundary_lt_foreground_lt_background": size_ordering,
        "expected_boundary_lt_foreground_lt_background": span_ordering,
        "ordering_is_observational_only": True,
        "raw_values": aggregate,
    }


def aggregate_region_diagnostics(reports, neighborhood):
    names = ("boundary", "foreground", "background")
    raw = {name: {"span": [], "group_size": []} for name in names}
    for report in reports:
        for name in names:
            for metric in ("span", "group_size"):
                raw[name][metric].extend(report.get("raw_values", {}).get(name, {}).get(metric, []))
    aggregate = {
        name: {metric: distribution(raw[name][metric]) for metric in ("span", "group_size")}
        for name in names
    }
    span_means = [aggregate[name]["span"]["mean"] for name in names]
    size_means = [aggregate[name]["group_size"]["mean"] for name in names]
    span_available = all(value is not None for value in span_means)
    size_available = all(value is not None for value in size_means)
    return {
        "coordinate_unit": COORDINATE_UNIT,
        "batch_count": len(reports),
        "boundary_neighborhood_tokens": neighborhood,
        "per_batch": reports,
        "boundary_span": aggregate["boundary"]["span"],
        "foreground_span": aggregate["foreground"]["span"],
        "background_span": aggregate["background"]["span"],
        "boundary_group_size": aggregate["boundary"]["group_size"],
        "foreground_group_size": aggregate["foreground"]["group_size"],
        "background_group_size": aggregate["background"]["group_size"],
        "span_ordering_boundary_lt_foreground_lt_background": bool(span_means[0] < span_means[1] < span_means[2]) if span_available else None,
        "group_size_ordering_boundary_lt_foreground_lt_background": bool(size_means[0] < size_means[1] < size_means[2]) if size_available else None,
        "ordering_is_observational_only": True,
    }


def cut_positions(matrix, valid_length, anchor_count):
    membership = matrix[:anchor_count, :valid_length] > 0
    if valid_length <= 1:
        return set()
    group_ids = membership.float().argmax(0)
    return set((
        torch.nonzero(group_ids[1:] != group_ids[:-1], as_tuple=False)
        .flatten() + 1
    ).tolist())


def token_group_sizes(matrix, valid_length, anchor_count):
    membership = (matrix[:anchor_count, :valid_length] > 0).float()
    sizes = membership.sum(-1)
    return membership.transpose(0, 1).matmul(sizes)


def _contiguous_regions(values, threshold):
    values = torch.as_tensor(values).flatten()
    selected = torch.isfinite(values) & (values >= threshold)
    regions = []
    begin = None
    for index, flag in enumerate(selected.tolist() + [False]):
        if flag and begin is None:
            begin = index
        elif not flag and begin is not None:
            regions.append([begin, index])
            begin = None
    return regions


def query_specific_grouping(batch, outputs, anchor_metadata):
    """Compare at least two queries from one video at every adaptive level."""
    pair = None
    output_start = 0
    for query_count in batch["text_size"].tolist():
        if query_count >= 2:
            pair = (output_start, output_start + 1)
            break
        output_start += query_count
    if pair is None:
        return {
            "available": False,
            "reason": "selected samples contain no video with two queries",
            "risk": None,
        }
    first, second = pair
    first_query = batch["queries"][first]
    second_query = batch["queries"][second]
    levels = []
    for level, values in enumerate(zip(
        outputs["assignments"], outputs["sequence_masks"],
        outputs["anchor_masks"], anchor_metadata,
    )):
        matrix, input_mask, anchor_mask, metadata = values
        input_valid = prefix_mask(input_mask)
        anchor_valid = prefix_mask(anchor_mask)
        valid_length = min(int(input_valid[first].sum()), int(input_valid[second].sum()))
        first_count = int(anchor_valid[first].sum())
        second_count = int(anchor_valid[second].sum())
        first_cuts = sorted(cut_positions(matrix[first], valid_length, first_count))
        second_cuts = sorted(cut_positions(matrix[second], valid_length, second_count))
        first_set, second_set = set(first_cuts), set(second_cuts)
        union = first_set | second_set
        intersection = first_set & second_set
        first_sizes = token_group_sizes(matrix[first], valid_length, first_count)
        second_sizes = token_group_sizes(matrix[second], valid_length, second_count)
        identical = bool(
            first_count == second_count and torch.equal(
                matrix[first, :first_count, :valid_length],
                matrix[second, :second_count, :valid_length],
            )
        )
        importance_curves = {}
        high_resolution = {}
        if outputs.get("importance") is not None:
            debug = outputs["importance"][level]
            for query_index, label in ((first, "query_0"), (second, "query_1")):
                curve = debug.get("importance")
                if curve is not None:
                    curve = curve[query_index, :valid_length].detach().float().cpu()
                    importance_curves[label] = [float(x) for x in curve]
                    finite = curve[torch.isfinite(curve)]
                    threshold = float(torch.quantile(finite, 0.75)) if finite.numel() else None
                    high_resolution[label] = _contiguous_regions(curve, threshold) if threshold is not None else []
        levels.append({
            "level": level,
            "valid_length": valid_length,
            "assignment_identical": identical,
            "assignment_identity": "identical" if identical else "different",
            "cut_positions_query_0": first_cuts,
            "cut_positions_query_1": second_cuts,
            "cut_jaccard_similarity": float(len(intersection)) / len(union) if union else 1.0,
            "cut_position_jaccard_distance": 1.0 - (float(len(intersection)) / len(union) if union else 1.0),
            "different_cut_count": len(first_set ^ second_set),
            "group_spans_query_0": [float(x) for x in metadata[first, :first_count, 3].detach().cpu()],
            "group_spans_query_1": [float(x) for x in metadata[second, :second_count, 3].detach().cpu()],
            "query_0_group_span": distribution(metadata[first, :first_count, 3]),
            "query_1_group_span": distribution(metadata[second, :second_count, 3]),
            "token_group_size_difference_fraction": float((first_sizes != second_sizes).float().mean()) if valid_length else 0.0,
            "mean_absolute_token_group_size_difference": float((first_sizes - second_sizes).abs().mean()) if valid_length else 0.0,
            "importance_curves": importance_curves,
            "high_resolution_regions": high_resolution,
        })
    all_identical = bool(levels) and all(level["assignment_identical"] for level in levels)
    risk = (
        "query-specific grouping is effectively identical across all inspected levels; "
        "adaptive allocation may not be query-specific"
        if all_identical else None
    )
    return {
        "available": True,
        "same_video": first_query.get("vid_id"),
        "query_output_indices": [first, second],
        "queries": [first_query, second_query],
        "query_identifiers": [
            {"video_id": first_query.get("vid_id"), "sentence_id": first_query.get("sentence_id"), "text": first_query.get("text")},
            {"video_id": second_query.get("vid_id"), "sentence_id": second_query.get("sentence_id"), "text": second_query.get("text")},
        ],
        "levels": levels,
        "any_grouping_difference": any(not level["assignment_identical"] for level in levels),
        "assignment_basic_consistency": all(level["assignment_identical"] for level in levels),
        "risk": risk,
    }

def baseline_points_in_raw_coordinates(points, opt):
    stride = float(opt["model"].get("vid_stride", 1))
    converted = []
    for level_points in points:
        raw = level_points.clone()
        raw[:, :4] *= stride
        converted.append(raw)
    return tuple(converted)


def regression_target_distribution(opt, points, masks, raw_targets,
                                   points_are_raw, near_zero=1e-4,
                                   extreme_offset=100.0):
    if not points_are_raw:
        points = baseline_points_in_raw_coordinates(points, opt)
    center_sampling = opt["train"].get("center_sampling", "radius")
    radius = float(opt["train"].get("center_sampling_radius", 1.5))
    all_scales, all_offsets = [], []
    levels = []
    for level, (level_points, mask) in enumerate(zip(points, masks)):
        scales, offsets = [], []
        valid = prefix_mask(mask)
        for batch_index, target in enumerate(raw_targets):
            point = sample_points(level_points, batch_index)
            labels, encoded, _, _, _ = annotate_points(
                point, target, center_sampling, radius
            )
            positive = valid[batch_index] & labels
            if positive.any():
                scales.append(point[positive, 3])
                offsets.append(encoded[positive])
        device = level_points.device
        packed_scales = torch.cat(scales) if scales else torch.empty(0, device=device)
        packed_offsets = torch.cat(offsets) if offsets else torch.empty(0, 2, device=device)
        if packed_scales.numel():
            all_scales.append(packed_scales)
        if packed_offsets.numel():
            all_offsets.append(packed_offsets)
        levels.append({
            "level": level,
            "positive_count": int(packed_offsets.size(0)),
            "regression_scale": distribution(packed_scales),
            "regression_scale_tokens": distribution(packed_scales),
            "normalized_left_target": distribution(packed_offsets[:, 0]),
            "normalized_right_target": distribution(packed_offsets[:, 1]),
            "normalized_target_signed": distribution(packed_offsets.flatten()),
            "normalized_target_absolute": distribution(packed_offsets.abs().flatten()),
            "near_zero_scale_count": int((packed_scales <= near_zero).sum()),
            "extreme_normalized_offset_count": int((packed_offsets.abs() >= extreme_offset).sum()),
        })
    reference_device = points[0].device if points else raw_targets.device
    scales = torch.cat(all_scales) if all_scales else torch.empty(0, device=reference_device)
    offsets = torch.cat(all_offsets) if all_offsets else torch.empty(0, 2, device=reference_device)
    anomalies = []
    if offsets.size(0) == 0:
        anomalies.append("no_positive_points")
    if not bool(torch.isfinite(scales).all()) or not bool(torch.isfinite(offsets).all()):
        anomalies.append("nonfinite_value")
    if scales.numel() and bool((scales <= 0).any()):
        anomalies.append("nonpositive_regression_scale")
    if scales.numel() and bool((scales <= near_zero).any()):
        anomalies.append("near_zero_regression_scale")
    if offsets.numel() and bool((offsets.abs() >= extreme_offset).any()):
        anomalies.append("extreme_normalized_offset")
    if offsets.numel() and bool((offsets < 0).any()):
        anomalies.append("negative_positive_point_regression_target")
    return {
        "coordinate_unit": COORDINATE_UNIT,
        "positive_count": int(offsets.size(0)),
        "regression_scale": distribution(scales),
        "regression_scale_tokens": distribution(scales),
        "normalized_left_target": distribution(offsets[:, 0]),
        "normalized_right_target": distribution(offsets[:, 1]),
        "normalized_target_signed": distribution(offsets.flatten()),
        "normalized_target_absolute": distribution(offsets.abs().flatten()),
        "near_zero_scale_threshold": near_zero,
        "near_zero_scale_count": int((scales <= near_zero).sum()),
        "extreme_normalized_offset_threshold": extreme_offset,
        "extreme_normalized_offset_count": int((offsets.abs() >= extreme_offset).sum()),
        "per_level": levels,
        "anomalies": sorted(set(anomalies)),
        "raw_values": {
            "regression_scale": scales.detach().float().cpu().tolist(),
            "normalized_offsets": offsets.detach().float().cpu().tolist(),
        },
    }


def aggregate_regression_diagnostics(reports):
    scales = []
    offsets = []
    anomalies = []
    for report in reports:
        scales.extend(report.get("raw_values", {}).get("regression_scale", []))
        offsets.extend(report.get("raw_values", {}).get("normalized_offsets", []))
        anomalies.extend(report.get("anomalies", []))
    scales = torch.as_tensor(scales, dtype=torch.float64)
    offsets = torch.as_tensor(offsets, dtype=torch.float64)
    if offsets.numel() == 0:
        offsets = offsets.reshape(0, 2)
    return {
        "coordinate_unit": COORDINATE_UNIT,
        "batch_count": len(reports),
        "positive_count": int(offsets.size(0)),
        "regression_scale": distribution(scales),
        "regression_scale_tokens": distribution(scales),
        "normalized_left_target": distribution(offsets[:, 0]),
        "normalized_right_target": distribution(offsets[:, 1]),
        "normalized_target_signed": distribution(offsets.flatten()),
        "normalized_target_absolute": distribution(offsets.abs().flatten()),
        "per_batch": reports,
        "anomalies": sorted(set(anomalies)),
    }


def compare_regression_distributions(baseline, adaptive):
    baseline_p99 = baseline["normalized_target_absolute"]["p99"]
    adaptive_p99 = adaptive["normalized_target_absolute"]["p99"]
    ratio = None
    worse = None
    if finite_number(baseline_p99) and finite_number(adaptive_p99):
        ratio = float(adaptive_p99 / max(float(baseline_p99), 1e-12))
        worse = bool(ratio >= 2.0)
    return {
        "baseline_fixed_stride": baseline,
        "adaptive_dynamic_scale": adaptive,
        "adaptive_p99_to_baseline_p99_ratio": ratio,
        "adaptive_long_tail_clearly_worse": worse,
        "checks_only_formulation_unchanged": True,
    }

def grounding_losses(opt, points, outputs):
    center_sampling = opt["train"].get("center_sampling", "radius")
    radius = float(opt["train"].get("center_sampling_radius", 1.5))
    loss_norm = float(opt["train"].get("loss_norm", 1.0))
    cls_sum = outputs["logits"][0].float().sum() * 0.0
    reg_sum = cls_sum
    positive_count = 0
    for level_points, logits, offsets, mask in zip(
        points, outputs["logits"], outputs["offsets"], outputs["masks"]
    ):
        valid = prefix_mask(mask)
        labels = []
        encoded_offsets = []
        for batch_index, target in enumerate(outputs["targets"]):
            point = sample_points(level_points, batch_index)
            level_labels, level_offsets, _, _, _ = annotate_points(
                point, target, center_sampling, radius
            )
            labels.append(level_labels)
            encoded_offsets.append(level_offsets)
        labels = torch.stack(labels)
        encoded_offsets = torch.stack(encoded_offsets)
        positive = valid & labels
        smoothed = labels[valid].to(logits.dtype) * 0.8 + 0.1
        cls_sum = cls_sum + sigmoid_focal_loss(
            logits[valid], smoothed, alpha=0.5, reduction="sum"
        )
        if positive.any():
            loss_fn = (
                ctr_diou_loss
                if opt["train"].get("reg_loss", "diou") == "diou"
                else ctr_giou_loss
            )
            reg_sum = reg_sum + loss_fn(
                offsets[positive], encoded_offsets[positive], reduction="sum"
            )
            positive_count += int(positive.sum())
    return cls_sum / loss_norm, reg_sum / loss_norm, positive_count


def auxiliary_losses(opt, outputs, points):
    importance_opt = opt["train"].get("loss_aux", {}).get(
        "query_boundary_importance", {}
    )
    zero = outputs["logits"][0].float().sum() * 0.0
    importance_loss = None
    importance_weight = 0.0
    importance_components = {
        key: zero
        for key in ("relevance", "boundary", "temporal_change", "importance")
    }
    if importance_opt.get("enable", False):
        module = QueryBoundaryImportanceLoss(importance_opt).to(
            outputs["targets"].device
        )
        importance_loss, importance_components = module(
            outputs["importance"],
            points,
            outputs["sequence_masks"],
            outputs["targets"],
            return_components=True,
        )
        importance_weight = float(importance_opt.get("weight", 0.1))
    boundary_weight, boundary_module = make_boundary_supervision_loss(
        importance_opt
    )
    boundary_loss = None
    boundary_components = {"start": zero, "end": zero, "boundary": zero}
    if boundary_module is not None:
        boundary_loss, boundary_components = boundary_module.to(
            outputs["targets"].device
        )(
            outputs["importance"],
            points,
            outputs["sequence_masks"],
            outputs["targets"],
            return_components=True,
        )
    return {
        "importance": importance_loss,
        "importance_weight": importance_weight,
        "importance_components": importance_components,
        "boundary": boundary_loss,
        "boundary_weight": float(boundary_weight),
        "boundary_components": boundary_components,
        "zero": zero,
    }


def top1_decode_metrics(opt, points, outputs):
    raw_scale = (
        1.0
        if opt["model"]["vid_net"].get("adaptive_anchor", False)
        else float(opt["model"].get("vid_stride", 1))
    )
    values = []
    l1_values = []
    for batch_index, target in enumerate(outputs["targets"]):
        best_score = None
        best_segment = None
        for level_points, logits, offsets, mask in zip(
            points, outputs["logits"], outputs["offsets"], outputs["masks"]
        ):
            valid = prefix_mask(mask)[batch_index]
            if not valid.any():
                continue
            scores = logits[batch_index].masked_fill(~valid, -float("inf"))
            point_index = int(scores.argmax())
            point = sample_points(level_points, batch_index)
            segment = decode_offsets(point, offsets[batch_index])[point_index]
            if best_score is None or bool(scores[point_index] > best_score):
                best_score = scores[point_index]
                best_segment = segment
        if best_segment is None:
            continue
        segment = best_segment * raw_scale
        raw_target = target * raw_scale
        start = torch.minimum(raw_target[0], raw_target[1])
        end = torch.maximum(raw_target[0], raw_target[1])
        intersection = (
            torch.minimum(segment[1], end)
            - torch.maximum(segment[0], start)
        ).clamp(min=0)
        union = (
            (segment[1] - segment[0]).clamp(min=0)
            + end - start - intersection
        ).clamp(min=1e-6)
        values.append(intersection / union)
        l1_values.append(0.5 * (
            (segment[0] - start).abs() + (segment[1] - end).abs()
        ))
    return {
        "mean_top1_iou": float(torch.stack(values).mean()) if values else None,
        "mean_boundary_l1_raw_tokens": (
            float(torch.stack(l1_values).mean()) if l1_values else None
        ),
    }


def named_gradient_norm(model, substring):
    check = named_gradient_check(model, substring)
    return check["norm"]


def named_gradient_check(model, substring):
    if model is None:
        return {"present": False, "finite": None, "nonzero": None, "norm": None}
    gradients = [
        parameter.grad.detach().float()
        for name, parameter in model.named_parameters()
        if substring in name and parameter.grad is not None
    ]
    if not gradients:
        return {"present": False, "finite": None, "nonzero": None, "norm": None}
    finite = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    norm = float(torch.stack([gradient.norm() for gradient in gradients]).norm())
    return {
        "present": True,
        "finite": finite,
        "nonzero": bool(finite and norm > 0.0),
        "norm": norm,
    }


def optimizer_state_finite(optimizer):
    return all(
        not torch.is_tensor(value) or bool(torch.isfinite(value).all())
        for state in optimizer.state.values()
        for value in state.values()
    )


def register_runtime_monitors(model):
    state = {"global_encoder": [], "importance_weighted_pooling": []}
    handles = []

    def global_hook(name):
        def hook(module, inputs, output):
            tensor = inputs[0] if inputs and torch.is_tensor(inputs[0]) else None
            state["global_encoder"].append({
                "module": name,
                "batch": int(tensor.size(0)) if tensor is not None else None,
                "sequence_length": int(tensor.size(1)) if tensor is not None and tensor.ndim >= 2 else None,
                "output_finite": tensor_tree_finite(output),
            })
        return hook

    def pooling_hook(name):
        def hook(module, inputs, output):
            anchors = output.get("anchors") if isinstance(output, dict) else output
            state["importance_weighted_pooling"].append({
                "module": name,
                "output_finite": tensor_tree_finite(anchors),
            })
        return hook

    for name, module in model.named_modules():
        if name.endswith("global_encoder"):
            handles.append(module.register_forward_hook(global_hook(name)))
        if name.endswith("adaptive_anchor_allocator"):
            handles.append(module.register_forward_hook(pooling_hook(name)))
    return state, handles


def clear_runtime_monitors(state):
    for values in state.values():
        values.clear()


def remove_hooks(handles):
    for handle in handles:
        handle.remove()


def optimization_learning_evidence(result):
    records = result.get("records", [])
    if len(records) < 2:
        return {"learned": False, "reason": "fewer than two optimization records"}
    first, last = records[0], records[-1]
    total_decreased = optional_less(last.get("total"), first.get("total"))
    reg_decreased = optional_less(last.get("reg"), first.get("reg"))
    iou_improved = optional_greater(last.get("mean_top1_iou"), first.get("mean_top1_iou"))
    boundary_improved = optional_less(
        last.get("mean_boundary_l1_raw_tokens"),
        first.get("mean_boundary_l1_raw_tokens"),
    )
    positive = all(record.get("positive_count", 0) > 0 for record in records)
    finite = all(record.get("all_finite", False) for record in records)
    loss_and_regression_improved = (
        total_decreased is True and reg_decreased is True
    )
    decoded_quality_improved = iou_improved is True
    return {
        "learned": bool(
            finite and positive and loss_and_regression_improved
            and decoded_quality_improved
        ),
        "all_iterations_finite": finite,
        "positive_points_every_step": positive,
        "total_loss_decreased": total_decreased,
        "regression_loss_decreased": reg_decreased,
        "top1_iou_improved": iou_improved,
        "boundary_l1_improved": boundary_improved,
        "loss_and_regression_improved": loss_and_regression_improved,
        "decoded_quality_improved": decoded_quality_improved,
    }


def resolve_research_amp_dtype():
    """Prefer native BF16 on supported CUDA hardware; FP16 is fallback only."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def short_optimization(opt, batch, steps, seed, use_autocast,
                       include_acc=False, amp_dtype=None):
    """Run bounded real-data optimization with explicit AMP/gradient checks."""
    if not torch.cuda.is_available():
        raise RuntimeError("short optimization requires CUDA")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = make_models_net(opt).cuda().train()
    adaptive = bool(opt["model"]["vid_net"].get("adaptive_anchor", False))
    importance = bool(opt["model"]["vid_net"].get("query_boundary_importance", False) or adaptive)
    acc_module = None
    acc_weight = 0.0
    if include_acc:
        acc_opt = opt["train"].get("loss_aux", {}).get("ds_contrast", {})
        expected_mode = "assignment_acc" if adaptive else "legacy_acc"
        if acc_opt.get("acc_mode", "legacy_acc") != expected_mode:
            raise ValueError("ACC mode does not match baseline/adaptive path")
        acc_module = MultiScaleMaskedContrastive(
            acc_opt, opt["model"]["vid_net"]["embd_dim"]
        ).cuda().train()
        acc_weight = float(acc_opt.get("weight", 1.0))
    optimizer = make_optimizer(model, opt["train"]["optimizer"])
    optimizer_owner = model
    if acc_module is not None:
        optimizer_owner = torch.nn.Module()
        optimizer_owner.add_module("model", model)
        optimizer_owner.add_module("acc_module", acc_module)
        # The repository optimizer classifier intentionally knows model modules,
        # but the validator's external ACC projector is a diagnostic module.
        # Add its parameters with the same basic decay convention locally.
        acc_decay = []
        acc_no_decay = []
        for name, parameter in acc_module.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.endswith("bias") or parameter.ndim <= 1:
                acc_no_decay.append(parameter)
            else:
                acc_decay.append(parameter)
        optimizer_opt = opt["train"]["optimizer"]
        if acc_decay:
            optimizer.add_param_group({
                "params": acc_decay,
                "weight_decay": optimizer_opt["weight_decay"],
                "lr": optimizer_opt["lr"],
            })
        if acc_no_decay:
            optimizer.add_param_group({
                "params": acc_no_decay,
                "weight_decay": 0.0,
                "lr": optimizer_opt["lr"],
            })
    if use_autocast:
        amp_dtype = resolve_research_amp_dtype() if amp_dtype is None else amp_dtype
        if amp_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("amp_dtype must be torch.float16 or torch.bfloat16")
    else:
        amp_dtype = None
    use_grad_scaler = bool(use_autocast and amp_dtype == torch.float16)
    scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)
    monitor, handles = register_runtime_monitors(model)
    records = []
    try:
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            clear_runtime_monitors(monitor)
            with torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=use_autocast
            ):
                outputs = model_forward(
                    opt, model, batch, importance=importance,
                    assignments=adaptive,
                )
                forward_finite = tensor_tree_finite(outputs)
                points = build_points(opt, model, outputs["masks"])
                cls_loss, reg_loss, positive_count = grounding_losses(opt, points, outputs)
                zero = outputs["logits"][0].float().sum() * 0.0
                acc_loss = zero
                if acc_module is not None:
                    acc_loss = acc_module(
                        outputs["fpn"], outputs["sequence_masks"],
                        outputs["anchors"], outputs["anchor_masks"],
                        assignment_matrices=(outputs["assignments"] if adaptive else None),
                    ) / float(opt["train"].get("loss_norm", 1.0))
                total = cls_loss + float(opt["train"].get("loss_weight", 1.0)) * reg_loss
                total = total + acc_weight * acc_loss
            # The existing auxiliary loss formulation uses sigmoid probabilities
            # with F.binary_cross_entropy. Keep that formulation unchanged while
            # avoiding PyTorch's autocast prohibition for this operation.
            if use_autocast:
                with torch.cuda.amp.autocast(enabled=False):
                    auxiliary = auxiliary_losses(opt, outputs, points)
            else:
                auxiliary = auxiliary_losses(opt, outputs, points)
            if auxiliary["importance"] is not None:
                total = total + auxiliary["importance_weight"] * auxiliary["importance"]
            if auxiliary["boundary"] is not None:
                total = total + auxiliary["boundary_weight"] * auxiliary["boundary"]
            losses_finite = all(tensor_tree_finite(value) for value in (
                cls_loss, reg_loss, auxiliary["importance"],
                auxiliary["boundary"], acc_loss, total,
            ))
            if not forward_finite or not losses_finite:
                raise RuntimeError("non-finite forward/loss at step {}".format(step))
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            named_parameters = list(optimizer_owner.named_parameters())
            gradients = [
                parameter.grad.detach().float()
                for _, parameter in named_parameters
                if parameter.grad is not None
            ]
            missing_gradients = [
                name for name, parameter in named_parameters
                if parameter.requires_grad and parameter.grad is None
            ]
            nonfinite_gradients = [
                name for name, parameter in named_parameters
                if parameter.grad is not None
                and not bool(torch.isfinite(parameter.grad).all())
            ]
            backward_finite = bool(gradients) and not nonfinite_gradients
            if not backward_finite:
                raise RuntimeError(
                    "no usable or non-finite gradient at step {} "
                    "(unused={}, nonfinite={})"
                    .format(
                        step,
                        missing_gradients[:12],
                        nonfinite_gradients[:12],
                    )
                )
            grad_checks = {
                "importance_predictor": named_gradient_check(model, "importance_predictor"),
                "query_modulation": named_gradient_check(model, "query_modulation_mlp"),
                "boundary_predictor": named_gradient_check(model, "boundary_predictor"),
                "regression_head": named_gradient_check(model, "reg_head"),
                "assignment_acc_projector": named_gradient_check(acc_module, "projector"),
            }
            clip_norm = opt["train"].get("clip_grad_norm")
            if clip_norm:
                torch.nn.utils.clip_grad_norm_(optimizer_owner.parameters(), clip_norm)
            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            params_finite = all(bool(torch.isfinite(parameter).all()) for parameter in optimizer_owner.parameters())
            state_finite = optimizer_state_finite(optimizer)
            if not params_finite or not state_finite:
                raise RuntimeError("non-finite optimizer step/state")
            with torch.no_grad():
                outputs_after = model_forward(
                    opt, model, batch, importance=importance,
                    assignments=adaptive,
                )
                points_after = build_points(opt, model, outputs_after["masks"])
                metrics = top1_decode_metrics(opt, points_after, outputs_after)
            if not tensor_tree_finite(outputs_after):
                raise RuntimeError("non-finite post-step model output")
            importance_components = auxiliary["importance_components"]
            boundary_components = auxiliary["boundary_components"]
            boundary_signal = (
                auxiliary["boundary"] if auxiliary["boundary"] is not None
                else importance_components["boundary"]
                if auxiliary["importance"] is not None else None
            )
            hydra_finite = bool(monitor["global_encoder"]) and all(item["output_finite"] for item in monitor["global_encoder"])
            pooling_finite = (
                all(item["output_finite"] for item in monitor["importance_weighted_pooling"])
                if adaptive else None
            )
            records.append({
                "step": step,
                "cls_loss": float(cls_loss.detach()),
                "reg_loss": float(reg_loss.detach()),
                "importance_boundary_loss": float(auxiliary["importance"].detach()) if auxiliary["importance"] is not None else None,
                "boundary_supervision_loss": float(auxiliary["boundary"].detach()) if auxiliary["boundary"] is not None else None,
                "acc_loss": float(acc_loss.detach()) if acc_module is not None else None,
                "total_loss": float(total.detach()),
                "cls": float(cls_loss.detach()),
                "reg": float(reg_loss.detach()),
                "importance": float(auxiliary["importance"].detach()) if auxiliary["importance"] is not None else None,
                "boundary": float(auxiliary["boundary"].detach()) if auxiliary["boundary"] is not None else None,
                "boundary_signal": float(boundary_signal.detach()) if boundary_signal is not None else None,
                "acc": float(acc_loss.detach()) if acc_module is not None else None,
                "total": float(total.detach()),
                "positive_count": positive_count,
                "mean_top1_iou": metrics["mean_top1_iou"],
                "mean_boundary_l1_raw_tokens": metrics["mean_boundary_l1_raw_tokens"],
                "gradient_checks": grad_checks,
                "missing_gradients": missing_gradients,
                "query_modulator_grad_norm": grad_checks["query_modulation"]["norm"],
                "importance_predictor_grad_norm": grad_checks["importance_predictor"]["norm"],
                "boundary_predictor_grad_norm": grad_checks["boundary_predictor"]["norm"],
                "regression_head_grad_norm": grad_checks["regression_head"]["norm"],
                "acc_projector_grad_norm": grad_checks["assignment_acc_projector"]["norm"],
                "scaler_before": scale_before,
                "scaler_after": float(scaler.get_scale()),
                "forward_finite": forward_finite,
                "loss_finite": losses_finite,
                "backward_finite": backward_finite,
                "optimizer_step_finite": params_finite,
                "optimizer_state_finite": state_finite,
                "hydra_mamba_output_finite": hydra_finite,
                "importance_weighted_pooling_finite": pooling_finite,
                "all_finite": bool(forward_finite and losses_finite and backward_finite and params_finite and state_finite and hydra_finite and (pooling_finite is not False)),
            })
    finally:
        remove_hooks(handles)
    first, last = records[0], records[-1]
    result = {
        "steps": steps,
        "autocast": bool(use_autocast),
        "autocast_dtype": str(amp_dtype) if use_autocast else None,
        "grad_scaler": use_grad_scaler,
        "acc_included": bool(acc_module is not None),
        "acc_mode": getattr(acc_module, "acc_mode", None),
        "records": records,
        "trends": {
            "cls_slope": linear_trend([record["cls"] for record in records]),
            "reg_slope": linear_trend([record["reg"] for record in records]),
            "total_slope": linear_trend([record["total"] for record in records]),
            "boundary_signal_slope": linear_trend([record["boundary_signal"] for record in records]),
            "iou_slope": linear_trend([record["mean_top1_iou"] for record in records]),
            "boundary_l1_slope": linear_trend([record["mean_boundary_l1_raw_tokens"] for record in records]),
        },
        "checks": {
            "cls_decreased": optional_less(last["cls"], first["cls"]),
            "reg_decreased": optional_less(last["reg"], first["reg"]),
            "total_decreased": optional_less(last["total"], first["total"]),
            "top1_iou_improved": optional_greater(last["mean_top1_iou"], first["mean_top1_iou"]),
            "boundary_l1_improved": optional_less(last["mean_boundary_l1_raw_tokens"], first["mean_boundary_l1_raw_tokens"]),
            "importance_predictor_gradient_finite_nonzero": any(record["gradient_checks"]["importance_predictor"]["nonzero"] is True for record in records) if importance else None,
            "query_modulation_gradient_finite_nonzero": any(record["gradient_checks"]["query_modulation"]["nonzero"] is True for record in records) if opt["model"]["vid_net"].get("query_modulation", False) else None,
            "boundary_predictor_gradient_finite": all(record["gradient_checks"]["boundary_predictor"]["finite"] is True for record in records) if importance else None,
            "regression_head_gradient_finite": all(record["gradient_checks"]["regression_head"]["finite"] is True for record in records),
            "assignment_acc_projector_gradient_finite_nonzero": any(record["gradient_checks"]["assignment_acc_projector"]["nonzero"] is True for record in records) if acc_module is not None and adaptive else None,
            "hydra_mamba_outputs_finite": all(record["hydra_mamba_output_finite"] for record in records),
            "importance_weighted_pooling_finite": all(record["importance_weighted_pooling_finite"] is True for record in records) if adaptive else None,
            "all_iterations_finite": all(record["all_finite"] for record in records),
        },
    }
    result["learning_evidence"] = optimization_learning_evidence(result)
    return result


def failed_optimization_report(steps, use_autocast, include_acc, exc):
    """Represent a failed bounded optimization stage without hiding the failure."""
    checks = {
        "cls_decreased": False,
        "reg_decreased": False,
        "total_decreased": False,
        "top1_iou_improved": False,
        "boundary_l1_improved": False,
        "importance_predictor_gradient_finite_nonzero": False,
        "query_modulation_gradient_finite_nonzero": False,
        "boundary_predictor_gradient_finite": False,
        "regression_head_gradient_finite": False,
        "assignment_acc_projector_gradient_finite_nonzero": False,
        "hydra_mamba_outputs_finite": False,
        "importance_weighted_pooling_finite": False,
        "all_iterations_finite": False,
    }
    return {
        "status": "failed",
        "steps": steps,
        "autocast": bool(use_autocast),
        "grad_scaler": bool(use_autocast),
        "acc_included": bool(include_acc),
        "records": [],
        "checks": checks,
        "learning_evidence": {
            "learned": False,
            "reason": repr(exc),
        },
        "failure": repr(exc),
    }


def profile_model_compute(opt, batch, warmup, iterations):
    if not torch.cuda.is_available():
        raise RuntimeError("compute profiling requires CUDA")
    model = make_models_net(opt).cuda().eval()
    adaptive = bool(opt["model"]["vid_net"].get("adaptive_anchor", False))
    importance = bool(opt["model"]["vid_net"].get("query_boundary_importance", False) or adaptive)
    monitor, handles = register_runtime_monitors(model)
    latencies = []
    representative = None
    try:
        with torch.no_grad():
            for _ in range(warmup):
                clear_runtime_monitors(monitor)
                model_forward(opt, model, batch, importance=importance, assignments=adaptive)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            for iteration in range(iterations):
                clear_runtime_monitors(monitor)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                outputs = model_forward(opt, model, batch, importance=importance, assignments=adaptive)
                end.record()
                torch.cuda.synchronize()
                latencies.append(float(start.elapsed_time(end)))
                if representative is None:
                    representative = {
                        "outputs": outputs,
                        "global_events": [dict(item) for item in monitor["global_encoder"]],
                    }
            peak_allocated = int(torch.cuda.max_memory_allocated())
            peak_reserved = int(torch.cuda.max_memory_reserved())
    finally:
        remove_hooks(handles)
    outputs = representative["outputs"]
    per_level = []
    cumulative_valid = 0
    cumulative_tensor = 0
    for level, (sequence_mask, anchor_mask) in enumerate(zip(outputs["sequence_masks"], outputs["anchor_masks"])):
        input_counts = prefix_mask(sequence_mask).sum(-1)
        anchor_counts = prefix_mask(anchor_mask).sum(-1)
        event = representative["global_events"][level] if level < len(representative["global_events"]) else {}
        valid_global = input_counts + anchor_counts
        tensor_length = event.get("sequence_length")
        batch_size = event.get("batch")
        cumulative_valid += int(valid_global.sum())
        if tensor_length is not None and batch_size is not None:
            cumulative_tensor += int(tensor_length * batch_size)
        per_level.append({
            "level": level,
            "input_token_count": [int(x) for x in input_counts.tolist()],
            "anchor_count": [int(x) for x in anchor_counts.tolist()],
            "global_hydra_mamba_actual_sequence_length": tensor_length,
            "global_valid_sequence_length": [int(x) for x in valid_global.tolist()],
        })
    return {
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "per_level": per_level,
        "cumulative_valid_token_processing_count": cumulative_valid * iterations,
        "cumulative_tensor_token_processing_count": cumulative_tensor * iterations,
        "single_forward_valid_token_processing_count": cumulative_valid,
        "single_forward_tensor_token_processing_count": cumulative_tensor,
        "peak_allocated_gpu_memory_bytes": peak_allocated,
        "peak_reserved_gpu_memory_bytes": peak_reserved,
        "latency_ms": finite_stats(latencies),
    }


def compare_compute_profiles(baseline, adaptive, tolerance):
    ratios = {}
    for key in (
        "cumulative_valid_token_processing_count",
        "cumulative_tensor_token_processing_count",
        "peak_allocated_gpu_memory_bytes",
        "peak_reserved_gpu_memory_bytes",
    ):
        base = baseline.get(key)
        value = adaptive.get(key)
        ratios[key] = float(value / base) if base else None
    base_latency = baseline["latency_ms"]["mean"]
    adaptive_latency = adaptive["latency_ms"]["mean"]
    ratios["latency_mean"] = float(adaptive_latency / base_latency) if base_latency else None
    token_ratio = ratios["cumulative_valid_token_processing_count"]
    close = bool(token_ratio is not None and abs(token_ratio - 1.0) <= tolerance)
    return {
        "same_real_batch": True,
        "budget_tolerance_fraction": tolerance,
        "baseline": baseline,
        "adaptive": adaptive,
        "adaptive_to_baseline_ratios": ratios,
        "token_budget_sufficiently_close": close,
    }

def flatten_config(value, prefix=""):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            child = key if not prefix else prefix + "." + key
            result.update(flatten_config(item, child))
        return result
    return {prefix: value}


def config_summary(path, opt):
    video = opt["model"]["vid_net"]
    importance = opt["train"].get("loss_aux", {}).get("query_boundary_importance", {})
    contrastive = opt["train"].get("loss_aux", {}).get("ds_contrast", {})
    prohibited = {
        key: bool(video.get(key, False))
        for key in (
            "progressive_compression", "progressive_compression_debug",
            "coarse_to_fine_refine", "soft_assignment",
            "differentiable_grouping", "straight_through_estimator",
            "use_ste", "ste",
        )
    }
    return {
        "path": path,
        "query_modulation": bool(video.get("query_modulation", False)),
        "query_aware_gate": bool(video.get("query_aware_gate", False)),
        "query_boundary_importance": bool(video.get("query_boundary_importance", False)),
        "importance_loss": bool(importance.get("enable", False)),
        "boundary_loss_weight": float(importance.get("boundary_loss_weight", 0.0)),
        "adaptive_anchor": bool(video.get("adaptive_anchor", False)),
        "target_keep_ratio": float(video.get("target_keep_ratio", 0.75)),
        "keep_ratio_per_level": video.get("keep_ratio_per_level"),
        "branch_depth": int(video.get("arch", opt["model"].get("num_fpn_levels", 0))[-1]
                         if isinstance(video.get("arch", opt["model"].get("num_fpn_levels", 0)), (list, tuple))
                         else opt["model"].get("num_fpn_levels", 0)),
        "coarse_to_fine_refine": bool(video.get("coarse_to_fine_refine", False)),
        "progressive_compression_debug": bool(video.get("progressive_compression_debug", False)),
        "prohibited_features": prohibited,
        "acc_mode": contrastive.get("acc_mode", "legacy_acc"),
    }


def validate_ablation_configs(paths):
    loaded = [(path, load_opt(path, is_training=True)) for path in paths]
    summaries = [config_summary(path, opt) for path, opt in loaded]
    baseline_flat = flatten_config(loaded[0][1])
    deltas = []
    for path, opt in loaded:
        flat = flatten_config(opt)
        changed = sorted(key for key in set(baseline_flat) | set(flat)
                         if baseline_flat.get(key) != flat.get(key))
        deltas.append({"path": path, "changed_from_A": changed})
    errors = []
    if len(summaries) != 4:
        errors.append("exactly four A-D configs are required")
    else:
        a, b, c, d = summaries
        if any((a["query_modulation"], a["query_boundary_importance"], a["adaptive_anchor"], a["coarse_to_fine_refine"])):
            errors.append("A is not baseline")
        if not b["query_modulation"] or b["query_boundary_importance"] or b["adaptive_anchor"]:
            errors.append("B must add only query modulation")
        if not c["query_modulation"] or not c["query_boundary_importance"] or c["adaptive_anchor"]:
            errors.append("C must add boundary importance on B")
        if not d["query_modulation"] or not d["query_boundary_importance"] or not d["adaptive_anchor"]:
            errors.append("D must add adaptive anchor on C")
        if c["adaptive_anchor"]:
            errors.append("C must keep adaptive_anchor=false")
        if d["target_keep_ratio"] != 0.5:
            errors.append("D target_keep_ratio must be 0.5")
        ratios = d["keep_ratio_per_level"]
        if not isinstance(ratios, (list, tuple)):
            errors.append("D keep_ratio_per_level must be an explicit list")
        else:
            if len(ratios) != d["branch_depth"]:
                errors.append("D keep_ratio_per_level length must equal branch depth {}".format(d["branch_depth"]))
            if any(float(value) != 0.5 for value in ratios):
                errors.append("all D keep_ratio_per_level values must be 0.5")
        if d["prohibited_features"]["progressive_compression"] or d["progressive_compression_debug"]:
            errors.append("D progressive compression must be disabled")
        for key in ("coarse_to_fine_refine", "soft_assignment", "differentiable_grouping", "straight_through_estimator", "use_ste", "ste"):
            if d["prohibited_features"].get(key, False):
                errors.append("D {} must be disabled".format(key))
        if d["acc_mode"] != "assignment_acc":
            errors.append("D must use assignment-aware ACC")
    return {
        "configs": summaries,
        "deltas": deltas,
        "allocator": "QueryBoundaryAdaptiveAnchorAllocator with detached hard cut ranking; no differentiable grouping",
        "errors": sorted(set(errors)),
        "passed": not errors,
    }, dict(loaded)

def _gate(status, detail=None, passed=None):
    result = {"status": status}
    if passed is not None:
        result["passed"] = bool(passed)
    if detail is not None:
        result["detail"] = detail
    return result


def _section(status="pending", **values):
    result = {"status": status}
    result.update(values)
    return result


def aggregate_gt_diagnostics(reports):
    queries = [query for report in reports for query in report.get("queries", [])]
    errors = sorted({error for report in reports for error in report.get("errors", [])})
    round_trip = [
        query["max_round_trip_absolute_error"]
        for query in queries
        if query.get("max_round_trip_absolute_error") is not None
    ]
    return {
        "status": "completed",
        "coordinate_unit": COORDINATE_UNIT,
        "batch_count": len(reports),
        "query_count": len(queries),
        "queries": queries,
        "max_round_trip_absolute_error": max(round_trip) if round_trip else None,
        "errors": errors,
        "passed": bool(queries) and not errors,
    }


def aggregate_batch_isolation(reports):
    samples = [sample for report in reports for sample in report.get("samples", [])]
    return {
        "status": "completed",
        "batch_count": len(reports),
        "samples": samples,
        "passed": bool(reports) and all(report.get("passed", False) for report in reports),
    }


def _blocked_real_sections(report, status, reason, missing_paths=None):
    payload = {
        "reason": reason,
        "missing_paths": list(missing_paths or []),
        "passed": False,
    }
    section_names = (
        "temporal_diagnostics", "batch_isolation", "real_gt_round_trip",
        "region_statistics", "query_specific_grouping",
        "regression_distribution", "small_set_overfit",
        "mixed_precision", "ad_compute_comparison",
    )
    for name in section_names:
        report[name] = _section(status, **payload)
    report["random_positive_gt"] = report["real_gt_round_trip"]
    report["group_span_regions"] = report["region_statistics"]
    report["cuda_amp_smoke"] = report["mixed_precision"]
    for key in (
        "real_batch_temporal_diagnostics", "batch_isolation",
        "real_gt_round_trip", "region_span_group_size",
        "query_specific_grouping", "regression_distribution",
        "small_set_overfit", "mixed_precision", "ad_compute_comparison",
    ):
        report["validation_gate"][key] = _gate(status, reason, passed=False)


def run(args):
    toy = toy_temporal_round_trip()
    report = {
        "status": "initializing",
        "formal_training_gate": "closed",
        "long_training_started": False,
        "substitute_dataset_used": False,
        "environment": {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
        },
        "offline_validation": {"toy_temporal_round_trip": toy},
        "temporal_diagnostics": _section("pending"),
        "batch_isolation": _section("pending"),
        "real_gt_round_trip": _section("pending"),
        "region_statistics": _section("pending"),
        "query_specific_grouping": _section("pending"),
        "regression_distribution": _section("pending"),
        "small_set_overfit": _section("pending"),
        "mixed_precision": _section("pending"),
        "ad_compute_comparison": _section("pending"),
        "risks": [],
        "ablation_readiness": {
            "ready_to_start_A_D_formal_ablation": False,
            "formal_training_gate": "closed",
            "automatic_long_training_start": False,
        },
        "validation_gate": {
            "toy_temporal_round_trip": _gate(
                "passed" if toy["passed"] else "failed", passed=toy["passed"]
            ),
            "formal_training": _gate(
                "closed",
                "This validator reports readiness only and never starts long training.",
                passed=False,
            ),
        },
    }
    try:
        ablations, loaded = validate_ablation_configs(args.configs)
    except Exception as exc:
        report["status"] = "blocked_invalid_ablation_configs"
        report["ablation_configs"] = {"passed": False, "errors": [repr(exc)]}
        report["blocking_reason"] = "A-D configuration loading failed."
        report["risks"].append(report["blocking_reason"])
        return report
    report["ablation_configs"] = ablations
    report["validation_gate"]["ablation_configs"] = _gate(
        "passed" if ablations["passed"] else "failed", passed=ablations["passed"]
    )
    if not ablations["passed"]:
        report["status"] = "blocked_invalid_ablation_configs"
        report["blocking_reason"] = "A-D configuration invariants failed."
        report["risks"].extend(ablations["errors"])
        return report

    baseline_opt = loaded[args.configs[0]]
    adaptive_opt = loaded[args.configs[3]]
    required = [os.path.normpath(path) for path in required_data_paths(adaptive_opt)]
    missing = sorted(set(
        os.path.normpath(path)
        for path in missing_data_paths(baseline_opt) + missing_data_paths(adaptive_opt)
    ))
    report["data"] = {
        "required_paths": required,
        "missing_paths": missing,
        "substitute_dataset_used": False,
    }
    if missing:
        reason = (
            "Real Ego4D annotations/features are absent. Real-batch diagnostics, "
            "GT round-trip, small-set overfit, mixed precision, and A/D compute "
            "comparison were not run; no substitute dataset was used."
        )
        _blocked_real_sections(report, "blocked_missing_ego4d", reason, missing)
        report["status"] = "blocked_missing_ego4d"
        report["blocking_reason"] = reason
        report["risks"].append("missing Ego4D assets: " + ", ".join(missing))
        return report
    if not torch.cuda.is_available():
        reason = "CUDA is required for the requested real-model validation stages."
        _blocked_real_sections(report, "blocked_no_cuda", reason)
        report["status"] = "blocked_no_cuda"
        report["blocking_reason"] = reason
        report["risks"].append(reason)
        return report

    try:
        requested_samples = max(
            args.small_set_size,
            args.diagnostic_batches * args.diagnostic_batch_size,
        )
        samples, sample_indices = select_real_samples(
            baseline_opt, requested_samples, args.seed
        )
        diagnostic_batches = []
        diagnostic_index_batches = []
        for offset in range(0, len(samples), args.diagnostic_batch_size):
            if len(diagnostic_batches) >= args.diagnostic_batches:
                break
            chunk = samples[offset:offset + args.diagnostic_batch_size]
            if not chunk:
                break
            diagnostic_batches.append(batchify(chunk, baseline_opt))
            diagnostic_index_batches.append(sample_indices[offset:offset + len(chunk)])
        if len(diagnostic_batches) < 2:
            raise RuntimeError("multi-batch diagnostics require at least two real batches")
        overfit_samples = samples[:args.small_set_size]
        overfit_indices = sample_indices[:args.small_set_size]
        overfit_batch = batchify(overfit_samples, baseline_opt)
        report["data"].update({
            "dataset_indices": sample_indices,
            "diagnostic_index_batches": diagnostic_index_batches,
            "diagnostic_batch_count": len(diagnostic_batches),
            "diagnostic_batch_size": args.diagnostic_batch_size,
            "overfit_dataset_indices": overfit_indices,
            "random_crop_disabled": True,
            "target_coordinate_unit": COORDINATE_UNIT,
        })
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        temporal_reports = []
        isolation_reports = []
        gt_reports = []
        region_reports = []
        grouping_reports = []
        adaptive_regression_reports = []
        adaptive_model = make_models_net(adaptive_opt).cuda().eval()
        for batch_index, batch in enumerate(diagnostic_batches):
            with torch.no_grad():
                outputs = model_forward(
                    adaptive_opt, adaptive_model, batch,
                    importance=True, assignments=True,
                )
                points = build_points(adaptive_opt, adaptive_model, outputs["masks"])
            diagnostics, anchor_metadata = temporal_diagnostics(adaptive_model, outputs)
            diagnostics["batch_index"] = batch_index
            temporal_reports.append(diagnostics)
            if batch["video"].size(0) > 1:
                isolation = batch_isolation_diagnostic(
                    adaptive_opt, adaptive_model, batch, outputs
                )
                isolation["batch_index"] = batch_index
                isolation_reports.append(isolation)
            gt_reports.append(positive_point_diagnostic(
                adaptive_opt, points, outputs, batch,
                args.seed + 100 + batch_index,
                max_queries=args.gt_queries_per_batch,
            ))
            region_reports.append(group_span_by_region(
                anchor_metadata, outputs, args.boundary_neighborhood
            ))
            grouping = query_specific_grouping(batch, outputs, anchor_metadata)
            grouping["batch_index"] = batch_index
            grouping_reports.append(grouping)
            adaptive_regression_reports.append(regression_target_distribution(
                adaptive_opt, points, outputs["masks"],
                batch["raw_targets"].cuda(), points_are_raw=True,
            ))
        del adaptive_model
        torch.cuda.empty_cache()

        baseline_regression_reports = []
        baseline_model = make_models_net(baseline_opt).cuda().eval()
        for batch in diagnostic_batches:
            with torch.no_grad():
                outputs = model_forward(baseline_opt, baseline_model, batch)
                points = build_points(baseline_opt, baseline_model, outputs["masks"])
            baseline_regression_reports.append(regression_target_distribution(
                baseline_opt, points, outputs["masks"],
                batch["raw_targets"].cuda(), points_are_raw=False,
            ))
        del baseline_model
        torch.cuda.empty_cache()

        temporal_aggregate = aggregate_temporal_diagnostics(temporal_reports)
        temporal_aggregate["status"] = "completed"
        temporal_aggregate["per_batch"] = temporal_reports
        report["temporal_diagnostics"] = temporal_aggregate
        isolation_aggregate = aggregate_batch_isolation(isolation_reports)
        report["batch_isolation"] = isolation_aggregate
        gt_aggregate = aggregate_gt_diagnostics(gt_reports)
        report["real_gt_round_trip"] = gt_aggregate
        report["random_positive_gt"] = gt_aggregate
        region_aggregate = aggregate_region_diagnostics(
            region_reports, args.boundary_neighborhood
        )
        region_aggregate["status"] = "completed"
        report["region_statistics"] = region_aggregate
        report["group_span_regions"] = region_aggregate
        available_grouping = [item for item in grouping_reports if item.get("available")]
        grouping_risks = [item["risk"] for item in available_grouping if item.get("risk")]
        report["query_specific_grouping"] = {
            "status": "completed",
            "batch_count": len(grouping_reports),
            "available_pair_count": len(available_grouping),
            "per_batch": grouping_reports,
            "risk": grouping_risks or None,
            "passed": bool(available_grouping),
        }
        baseline_regression = aggregate_regression_diagnostics(baseline_regression_reports)
        adaptive_regression = aggregate_regression_diagnostics(adaptive_regression_reports)
        regression_compare = compare_regression_distributions(
            baseline_regression, adaptive_regression
        )
        regression_compare["status"] = "completed"
        regression_compare["formulation_modified"] = False
        report["regression_distribution"] = regression_compare

        report["validation_gate"]["real_batch_temporal_diagnostics"] = _gate(
            "passed" if temporal_aggregate["passed"] else "failed",
            passed=temporal_aggregate["passed"],
        )
        report["validation_gate"]["batch_isolation"] = _gate(
            "passed" if isolation_aggregate["passed"] else "failed",
            passed=isolation_aggregate["passed"],
        )
        report["validation_gate"]["real_gt_round_trip"] = _gate(
            "passed" if gt_aggregate["passed"] else "failed",
            passed=gt_aggregate["passed"],
        )
        report["validation_gate"]["region_span_group_size"] = _gate(
            "reported", {
                "span_ordering": region_aggregate["span_ordering_boundary_lt_foreground_lt_background"],
                "group_size_ordering": region_aggregate["group_size_ordering_boundary_lt_foreground_lt_background"],
            }, passed=True,
        )
        grouping_ok = report["query_specific_grouping"]["passed"]
        report["validation_gate"]["query_specific_grouping"] = _gate(
            "reported" if grouping_ok else "unavailable", grouping_risks or None,
            passed=grouping_ok,
        )
        severe_regression = sorted(set(
            baseline_regression["anomalies"] + adaptive_regression["anomalies"]
        ))
        regression_ok = not any(anomaly in severe_regression for anomaly in (
            "nonfinite_value", "nonpositive_regression_scale",
            "near_zero_regression_scale", "no_positive_points",
        ))
        report["validation_gate"]["regression_distribution"] = _gate(
            "passed" if regression_ok else "failed",
            {"anomalies": severe_regression}, passed=regression_ok,
        )

        baseline_overfit = short_optimization(
            baseline_opt, overfit_batch, args.overfit_steps, args.seed,
            False,
            include_acc=bool(baseline_opt["train"].get("loss_aux", {}).get("ds_contrast", {}).get("enable", False)),
        )
        torch.cuda.empty_cache()
        adaptive_overfit = short_optimization(
            adaptive_opt, overfit_batch, args.overfit_steps, args.seed,
            False,
            include_acc=bool(adaptive_opt["train"].get("loss_aux", {}).get("ds_contrast", {}).get("enable", False)),
        )
        torch.cuda.empty_cache()
        both_learned = bool(
            baseline_overfit["learning_evidence"]["learned"]
            and adaptive_overfit["learning_evidence"]["learned"]
        )
        report["small_set_overfit"] = {
            "status": "completed",
            "same_dataset_indices": overfit_indices,
            "same_batch": True,
            "baseline": baseline_overfit,
            "adaptive": adaptive_overfit,
            "baseline_learned": baseline_overfit["learning_evidence"]["learned"],
            "adaptive_learned": adaptive_overfit["learning_evidence"]["learned"],
            "both_learned": both_learned,
            "passed": both_learned,
        }
        report["validation_gate"]["small_set_overfit"] = _gate(
            "passed" if both_learned else "failed",
            passed=both_learned,
        )
        if not baseline_overfit["learning_evidence"].get("decoded_quality_improved"):
            report["risks"].append(
                "baseline small-set decoded top-1 IoU did not improve"
            )
        if not adaptive_overfit["learning_evidence"].get("decoded_quality_improved"):
            report["risks"].append(
                "adaptive small-set decoded top-1 IoU did not improve"
            )

        amp_failures = []
        try:
            baseline_amp = short_optimization(
                baseline_opt, overfit_batch, args.smoke_steps, args.seed + 2,
                True, include_acc=False,
            )
        except Exception as exc:
            baseline_amp = failed_optimization_report(
                args.smoke_steps, True, False, exc
            )
            amp_failures.append({"variant": "baseline", "error": repr(exc)})
        finally:
            torch.cuda.empty_cache()
        try:
            adaptive_amp = short_optimization(
                adaptive_opt, overfit_batch, args.smoke_steps, args.seed + 2,
                True, include_acc=True,
            )
        except Exception as exc:
            adaptive_amp = failed_optimization_report(
                args.smoke_steps, True, True, exc
            )
            amp_failures.append({"variant": "adaptive", "error": repr(exc)})
        finally:
            torch.cuda.empty_cache()
        baseline_amp_checks = baseline_amp.get("checks", {})
        adaptive_checks = adaptive_amp.get("checks", {})
        amp_passed = bool(
            baseline_amp_checks.get("all_iterations_finite") is True
            and adaptive_checks.get("all_iterations_finite") is True
            and adaptive_checks.get("importance_predictor_gradient_finite_nonzero") is True
            and adaptive_checks.get("query_modulation_gradient_finite_nonzero") is True
            and adaptive_checks.get("boundary_predictor_gradient_finite") is True
            and adaptive_checks.get("regression_head_gradient_finite") is True
            and adaptive_checks.get("assignment_acc_projector_gradient_finite_nonzero") is True
            and adaptive_checks.get("hydra_mamba_outputs_finite") is True
            and adaptive_checks.get("importance_weighted_pooling_finite") is True
        )
        report["mixed_precision"] = {
            "status": "completed",
            "iterations": args.smoke_steps,
            "baseline": baseline_amp,
            "adaptive": adaptive_amp,
            "failures": amp_failures or None,
            "passed": amp_passed,
        }
        report["cuda_amp_smoke"] = report["mixed_precision"]
        report["validation_gate"]["mixed_precision"] = _gate(
            "passed" if amp_passed else "failed", passed=amp_passed
        )

        profile_batch = diagnostic_batches[0]
        baseline_profile = None
        adaptive_profile = None
        compute_failures = []
        try:
            baseline_profile = profile_model_compute(
                baseline_opt, profile_batch, args.compute_warmup,
                args.compute_iterations,
            )
        except Exception as exc:
            compute_failures.append({"variant": "baseline", "error": repr(exc)})
        finally:
            torch.cuda.empty_cache()
        try:
            adaptive_profile = profile_model_compute(
                adaptive_opt, profile_batch, args.compute_warmup,
                args.compute_iterations,
            )
        except Exception as exc:
            compute_failures.append({"variant": "adaptive", "error": repr(exc)})
        finally:
            torch.cuda.empty_cache()
        if baseline_profile is not None and adaptive_profile is not None:
            compute = compare_compute_profiles(
                baseline_profile, adaptive_profile,
                args.compute_budget_tolerance,
            )
            compute["status"] = "completed"
            compute["passed"] = compute["token_budget_sufficiently_close"]
        else:
            compute = {
                "status": "failed",
                "baseline": baseline_profile,
                "adaptive": adaptive_profile,
                "failures": compute_failures,
                "token_budget_sufficiently_close": False,
                "passed": False,
            }
        report["ad_compute_comparison"] = compute
        compute_gate_status = (
            "passed" if compute["passed"]
            else "failed" if compute["status"] == "failed"
            else "budget_mismatch"
        )
        report["validation_gate"]["ad_compute_comparison"] = _gate(
            compute_gate_status,
            passed=compute["passed"],
        )

        if grouping_risks:
            report["risks"].extend(grouping_risks)
        if regression_compare.get("adaptive_long_tail_clearly_worse"):
            report["risks"].append("adaptive normalized regression targets have a materially worse p99 tail")
        if region_aggregate["span_ordering_boundary_lt_foreground_lt_background"] is False:
            report["risks"].append("observed boundary span ordering does not satisfy boundary < foreground < background")
        if region_aggregate["group_size_ordering_boundary_lt_foreground_lt_background"] is False:
            report["risks"].append("observed boundary group-size ordering does not satisfy boundary < foreground < background")
        if not compute["token_budget_sufficiently_close"]:
            report["risks"].append("A/D measured token-processing budgets are not sufficiently close")
        readiness_checks = {
            "toy_round_trip": toy["passed"],
            "temporal_invariants": temporal_aggregate["passed"],
            "batch_isolation": isolation_aggregate["passed"],
            "real_gt_round_trip": gt_aggregate["passed"],
            "query_pair_available": grouping_ok,
            "regression_safe": regression_ok,
            "baseline_and_adaptive_overfit": both_learned,
            "mixed_precision": amp_passed,
            "compute_budget_close": compute["passed"],
        }
        ready = all(readiness_checks.values())
        report["ablation_readiness"] = {
            "ready_to_start_A_D_formal_ablation": ready,
            "checks": readiness_checks,
            "formal_training_gate": "closed",
            "automatic_long_training_start": False,
        }
        report["status"] = "completed" if ready else "validation_failed"
        if not ready:
            report["blocking_reason"] = "One or more research validation gates failed."
    except Exception as exc:
        report["status"] = "validation_failed"
        report["validation_error"] = repr(exc)
        report["blocking_reason"] = (
            "A real-data validation step raised an exception; formal training remains closed."
        )
        report["risks"].append(report["blocking_reason"] + " " + repr(exc))
    report["formal_training_gate"] = "closed"
    report["long_training_started"] = False
    return report

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs=4,
        default=[
            "opts/research_ablation_A_baseline.yaml",
            "opts/research_ablation_B_query_modulation.yaml",
            "opts/research_ablation_C_boundary_importance.yaml",
            "opts/research_ablation_D_adaptive_anchor.yaml",
        ],
        metavar=("A", "B", "C", "D"),
    )
    parser.add_argument(
        "--report", default="experiments/research_validation/report.json"
    )
    parser.add_argument("--small-set-size", type=int, default=1)
    parser.add_argument("--diagnostic-batches", type=int, default=3)
    parser.add_argument("--diagnostic-batch-size", type=int, default=2)
    parser.add_argument("--gt-queries-per-batch", type=int, default=3)
    parser.add_argument("--overfit-steps", type=int, default=20)
    parser.add_argument("--smoke-steps", type=int, default=5)
    parser.add_argument("--compute-warmup", type=int, default=2)
    parser.add_argument("--compute-iterations", type=int, default=10)
    parser.add_argument("--compute-budget-tolerance", type=float, default=0.10)
    parser.add_argument("--boundary-neighborhood", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    if args.small_set_size <= 0:
        parser.error("--small-set-size must be positive")
    if args.diagnostic_batches < 2:
        parser.error("--diagnostic-batches must be at least 2")
    if args.diagnostic_batch_size < 2:
        parser.error("--diagnostic-batch-size must be at least 2 for isolation")
    if args.gt_queries_per_batch < 2:
        parser.error("--gt-queries-per-batch must be at least 2")
    if args.overfit_steps < 2:
        parser.error("--overfit-steps must be at least 2")
    if args.smoke_steps < 5:
        parser.error("--smoke-steps must be at least 5")
    if args.compute_warmup < 1:
        parser.error("--compute-warmup must be positive")
    if args.compute_iterations < 2:
        parser.error("--compute-iterations must be at least 2")
    if not 0.0 <= args.compute_budget_tolerance <= 1.0:
        parser.error("--compute-budget-tolerance must be in [0, 1]")
    if (
        not math.isfinite(args.boundary_neighborhood)
        or args.boundary_neighborhood <= 0
    ):
        parser.error("--boundary-neighborhood must be finite and positive")
    return args


def main():
    os.chdir(ROOT)
    args = parse_args()
    report = run(args)
    report_dir = os.path.dirname(args.report)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(args.report, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

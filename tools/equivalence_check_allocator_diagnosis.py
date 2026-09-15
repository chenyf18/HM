#!/usr/bin/env python3
"""Pipeline equivalence diagnostics for the allocator-policy study (Step 1).

Checks, on real Ego4D batches:
1. Allocator level: uniform policy at keep_ratio 0.5 reproduces stride-2
   grouping exactly for many lengths (CPU part).
2. Model level: A (fixed pipeline) vs AU (uniform adaptive pipeline) with
   identical shared weights. The two pipelines are structurally different by
   design -- the adaptive path interleaves anchors into the Mamba scan -- so
   bitwise output equality is NOT expected. This script quantifies every
   difference source the directive lists: FPN shapes/masks, logits/offsets
   deltas, per-level anchor counts, assignment coverage, temporal metadata,
   and regression scales.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hydra"))
sys.path.insert(0, str(ROOT / "tools"))

from libs import load_opt  # noqa: E402
from libs.modeling.model import make_models_net  # noqa: E402
from tools import research_validate_adaptive as rva  # noqa: E402

CONFIG_A = ROOT / "opts/research_ablation_A_baseline.yaml"
CONFIG_AU = ROOT / "opts/research_allocator_AU_uniform.yaml"
OUTPUT = ROOT / "experiments/allocator_diagnosis/pipeline_equivalence_check.json"


def tensor_stats(value):
    flat = value.detach().float().flatten()
    if flat.numel() == 0:
        return {"numel": 0}
    return {
        "numel": int(flat.numel()),
        "abs_max": float(flat.abs().max()),
        "mean": float(flat.mean()),
        "std": float(flat.std()) if flat.numel() > 1 else 0.0,
    }


def allocator_level_stride2_check():
    from libs.modeling.adaptive_anchor import (
        QueryBoundaryAdaptiveAnchorAllocator,
    )

    allocator = QueryBoundaryAdaptiveAnchorAllocator(
        target_keep_ratio=0.5,
        importance_weighted_pooling=True,
        allocator_policy="uniform",
    )
    results = []
    for seq_len in range(2, 65):
        tokens = torch.randn(1, 3, seq_len)
        mask = torch.ones(1, seq_len, dtype=torch.bool)
        outputs = allocator(tokens, mask)
        assignment = outputs["assignment_matrix"][0]
        target = int(outputs["target_counts"][0])
        members = assignment.argmax(dim=0)
        exact = True
        for group in range(target):
            expected = [
                idx for idx in (2 * group, 2 * group + 1) if idx < seq_len
            ]
            got = torch.nonzero(
                members == group, as_tuple=False
            ).flatten().tolist()
            if got != expected:
                exact = False
                break
        results.append({
            "seq_len": seq_len,
            "anchors": target,
            "equals_stride2": bool(exact),
        })
    passed = all(item["equals_stride2"] for item in results)
    return {"passed": bool(passed), "lengths": results}


def build_model(config_path, seed):
    opt = load_opt(str(config_path), is_training=True)
    torch.manual_seed(seed)
    model = make_models_net(opt)
    return opt, model


def shared_weight_transfer(source, target):
    source_state = source.state_dict()
    missing_before = set(target.state_dict().keys())
    loaded, unexpected = [], []
    filtered = {}
    for key, value in source_state.items():
        if key in missing_before:
            target_state_shape = target.state_dict()[key].shape
            if tuple(target_state_shape) == tuple(value.shape):
                filtered[key] = value
                loaded.append(key)
            else:
                unexpected.append(key)
    target.load_state_dict(filtered, strict=False)
    remaining_missing = sorted(set(target.state_dict().keys()) - set(loaded))
    return loaded, remaining_missing


def main():
    seed = 1234567891
    report = {}

    print("Part 1: allocator-level uniform==stride2 check", flush=True)
    report["allocator_stride2_equivalence"] = allocator_level_stride2_check()

    print("Part 2: real-batch A vs AU model comparison", flush=True)
    opt_a, model_a = build_model(CONFIG_A, seed)
    opt_au, model_au = build_model(CONFIG_AU, seed)

    samples, sample_indices = rva.select_real_samples(opt_a, 4, seed)
    batch = rva.batchify(samples, opt_a)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_a = model_a.to(device).eval()
    model_au = model_au.to(device).eval()
    loaded_keys, au_only_keys = shared_weight_transfer(model_a, model_au)
    report["weight_transfer"] = {
        "loaded_shared_keys": len(loaded_keys),
        "au_only_keys": au_only_keys,
    }

    video = batch["video"].to(device)
    video_mask = batch["video_mask"].to(device)
    text = batch["text"].to(device)
    text_mask = batch["text_mask"].to(device)
    text_size = batch["text_size"].to(device)

    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"
    ):
        outputs_a = model_a(
            video, video_mask, text, text_mask, text_size,
            return_importance_debug=False,
        )
        outputs_au = model_au(
            video, video_mask, text, text_mask, text_size,
            return_importance_debug=True,
        )

    names = (
        "fpn_logits", "fpn_logits2", "fpn_offsets", "fpn_masks",
        "fpn", "sequence_fpn_masks", "anchor_fpn", "anchor_fpn_masks",
    )
    valid_lengths = [int(video_mask[b].sum()) for b in range(video.size(0))]
    # Adaptive chain: level input length T_0 = valid length, then
    # T_{l+1} = ceil(T_l * keep_ratio). Legacy widths instead derive from the
    # padded input length through fixed 2**level strides.
    def adaptive_chain(length):
        chain = [int(length)]
        while len(chain) < 8:
            chain.append(int(math.ceil(chain[-1] * 0.5)))
        return chain

    chains = [adaptive_chain(length) for length in valid_lengths]
    comparison = {}
    for index, name in enumerate(names):
        value_a = outputs_a[index]
        value_au = outputs_au[index]
        entry = {"shapes_a": [list(t.shape) for t in value_a],
                 "shapes_au": [list(t.shape) for t in value_au]}
        entry["shapes_equal"] = entry["shapes_a"] == entry["shapes_au"]
        if name in ("fpn_logits", "fpn_offsets"):
            finite = all(
                bool(torch.isfinite(t.float()).all()) for t in value_au
            )
            entry["au_outputs_finite"] = finite
            if entry["shapes_equal"]:
                diffs = []
                for tensor_a, tensor_au in zip(value_a, value_au):
                    diffs.append(
                        tensor_stats(tensor_a.float() - tensor_au.float())
                    )
                entry["per_level_abs_diff"] = diffs
            else:
                entry["structural_difference"] = (
                    "legacy widths follow padded-length fixed grids "
                    "(e.g. {} at level 0); adaptive widths follow the "
                    "per-sample valid-length chain {}".format(
                        entry["shapes_a"][0][-1] if entry["shapes_a"] else None,
                        chains[0][:3],
                    )
                )
        comparison[name] = entry
    report["output_comparison"] = {
        "per_sample_valid_lengths": valid_lengths,
        "expected_adaptive_length_chain_sample0": chains[0],
        **comparison,
    }

    metadata = getattr(model_au.vid_net, "last_temporal_metadata", None)
    # Early fusion expands the video batch by the number of queries per
    # sample; map every expanded row back to its source sample.
    expanded_sources = []
    for sample_index, query_count in enumerate(
        batch["text_size"].tolist()
    ):
        expanded_sources.extend([sample_index] * int(query_count))
    if metadata:
        # Extend each per-sample chain to the recorded metadata depth.
        depth = len(metadata)

        def _extend(chain):
            while len(chain) < depth:
                chain.append(int(math.ceil(chain[-1] * 0.5)))
            return chain

        chains = [
            _extend(list(chains[expanded_sources[row]]))
            for row in range(len(expanded_sources))
        ]
    metadata_report = []
    if metadata:
        for level, meta in enumerate(metadata):
            meta = meta.float()
            # Rows correspond to this level's input tokens in temporal
            # order; padding rows are zeroed by make_initial_metadata /
            # propagate, so restrict statistics to nonzero-span rows.
            active = meta[..., 3].abs() > 0
            counts = active.sum(dim=-1)
            spans = meta[..., 3].abs().clamp_min(1e-12)
            centers = meta[..., 2]
            monotonic = []
            chain_match = []
            for b in range(meta.size(0)):
                count = int(counts[b])
                if count >= 2:
                    monotonic.append(
                        bool(torch.all(centers[b, 1:count].diff() >= -1e-6))
                    )
                chain_match.append(count == chains[b][level])
            metadata_report.append({
                "level": level,
                "metadata_shape": list(meta.shape),
                "active_rows": [int(c) for c in counts],
                "expected_chain_rows": [chain[level] for chain in chains],
                "active_rows_match_chain": all(chain_match),
                "span_abs_max": float(spans.max()),
                "regression_scale_abs_max": float(meta[..., 4].abs().max()),
                "centers_nondecreasing_active_prefix_all_samples": (
                    all(monotonic) and len(monotonic) == meta.size(0)
                ),
                "finite": bool(torch.isfinite(meta).all()),
            })
    report["adaptive_metadata"] = {
        "per_level": metadata_report,
    }

    importance_debug = outputs_au[-1]
    report["importance_debug_levels"] = len(importance_debug)

    metadata_ok = bool(metadata_report) and all(
        item["finite"] and item["active_rows_match_chain"]
        and item["centers_nondecreasing_active_prefix_all_samples"]
        and item["regression_scale_abs_max"] > 0
        for item in metadata_report
    )
    au_finite = all(
        comparison[name].get("au_outputs_finite", False)
        for name in ("fpn_logits", "fpn_offsets")
    )
    overall_pass = (
        report["allocator_stride2_equivalence"]["passed"]
        and metadata_ok
        and au_finite
    )
    report["overall"] = {
        "passed": bool(overall_pass),
        "allocator_stride2_equivalence": report[
            "allocator_stride2_equivalence"
        ]["passed"],
        "adaptive_metadata_ok": metadata_ok,
        "au_outputs_finite": au_finite,
        "shape_equality_expected_false": {
            name: comparison[name]["shapes_equal"]
            for name in ("fpn_logits", "fpn_masks")
        },
        "interpretation": (
            "Uniform adaptive grouping equals stride-2 exactly at the "
            "allocator level; the adaptive geometry chain (per-sample "
            "ceil-half lengths, ordered centers, positive spans/scales) is "
            "consistent; AU outputs are finite. A-vs-AU output widths "
            "differ BY DESIGN: legacy FPN uses padded fixed grids while the "
            "adaptive pipeline compacts per-sample valid lengths, and the "
            "interleaved anchors change the Mamba scan length. These are the "
            "documented structural differences of the adaptive pipeline."
            if overall_pass else "See components; investigate before launch."
        ),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["overall"], indent=2))
    print("report:", OUTPUT)


if __name__ == "__main__":
    main()

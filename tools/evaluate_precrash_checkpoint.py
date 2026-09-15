#!/usr/bin/env python3
"""Evaluate an existing formal-ablation checkpoint without mutating its run."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hydra"))

from libs import load_opt  # noqa: E402
from tools.run_formal_ablation import FormalEvaluator, _write_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", default="D_precrash_diagnostic")
    args = parser.parse_args()

    source = Path(args.source_run).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir(exist_ok=True)

    source_model = source / "models" / "last.pth"
    source_state = source / "states" / "last.pth"
    output_model = output / "models" / "precrash.pth"
    if not output_model.exists():
        os.symlink(source_model, output_model)

    state = torch.load(source_state, map_location="cpu", weights_only=False)
    opt = load_opt(str(source / "opt.yaml"), is_training=False)
    opt["_root"] = str(output)
    opt["_ckpt"] = "precrash"

    evaluator = FormalEvaluator(opt)
    evaluator.run()

    prediction_path = output / "predictions_precrash.json"
    payload = json.loads(prediction_path.read_text())
    metrics = payload.get("summary", {}).get("overall_recall_at_iou", {})
    _write_json(output / "evaluation_summary.json", {
        "label": args.label,
        "diagnostic_only": True,
        "source_checkpoint": str(source_model),
        "source_state": str(source_state),
        "checkpoint_epoch": int(state["epoch"]),
        "checkpoint_iteration": int(state["itr"]),
        "precision": "bf16 with existing global_encoder FP32 boundary",
        "metrics": metrics,
    })


if __name__ == "__main__":
    main()

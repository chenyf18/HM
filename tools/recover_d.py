#!/usr/bin/env python3
"""Resume the failed D run from its last normal checkpoint.

The source run is never used as an output directory.  Checkpoints are copied
to a separate recovery directory, then the epoch-6 sampler order is rebuilt
and the remaining batches are run with the verified P3 precision policy.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hydra"))

from libs import load_opt  # noqa: E402
from tools.replay_d_forensics import install_replay_loader, replay_epoch_order  # noqa: E402
from tools.run_formal_ablation import FormalTrainer, _write_json  # noqa: E402


def _copy_source(source: Path, output: Path) -> None:
    """Copy immutable source snapshots; never hard-link the original files."""
    (output / "models").mkdir(parents=True, exist_ok=True)
    (output / "states").mkdir(parents=True, exist_ok=True)
    for name in ("config_source.yaml", "opt.yaml", "run_manifest.json"):
        source_file = source / name
        if source_file.exists():
            shutil.copy2(source_file, output / name)
    shutil.copy2(
        source / "models" / "last.pth",
        output / "models" / "resume_source_model_last.pth",
    )
    shutil.copy2(
        source / "states" / "last.pth",
        output / "states" / "resume_source_state_last.pth",
    )


def _load_checkpoint(trainer: FormalTrainer, output: Path):
    model_ckpt = torch.load(
        output / "models" / "resume_source_model_last.pth",
        map_location="cpu",
        weights_only=False,
    )
    state_ckpt = torch.load(
        output / "states" / "resume_source_state_last.pth",
        map_location="cpu",
        weights_only=False,
    )
    trainer._load_model_state(trainer.model, model_ckpt["model"])
    trainer._load_model_state(trainer.model_ema, model_ckpt["model_ema"])
    trainer.optimizer.load_state_dict(state_ckpt["optimizer"])
    trainer.scheduler.load_state_dict(state_ckpt["scheduler"])
    trainer.epoch = int(state_ckpt["epoch"])
    trainer.itr = int(state_ckpt["itr"])
    return state_ckpt


def _assert_finite(trainer: FormalTrainer) -> None:
    for name, parameter in trainer.model.named_parameters():
        if not bool(torch.isfinite(parameter).all()):
            raise FloatingPointError("nonfinite model parameter at resume: {}".format(name))
    for index, state in enumerate(trainer.optimizer.state.values()):
        for key, value in state.items():
            if torch.is_tensor(value) and value.is_floating_point():
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError(
                        "nonfinite optimizer state at resume: {} {}".format(index, key)
                    )


def run(source: Path, output: Path, policy: str) -> None:
    if output.exists() and any(output.iterdir()):
        summary = output / "training_summary.json"
        if summary.exists():
            payload = json.loads(summary.read_text())
            if payload.get("status") == "completed":
                print("recovery already completed: {}".format(output), flush=True)
                return
        raise RuntimeError(
            "recovery output is non-empty; refusing to mix runs: {}".format(output)
        )
    output.mkdir(parents=True, exist_ok=True)
    _copy_source(source, output)

    opt = load_opt(str(source / "opt.yaml"), is_training=True)
    opt["_root"] = str(output)
    opt["_resume"] = False
    opt["_distributed"] = False
    opt["_world_size"] = 1
    (output / "opt.yaml").write_text(__import__("yaml").dump(opt, sort_keys=False))

    trainer = FormalTrainer(
        opt,
        label="D_recovery",
        run_root=output,
        max_steps=None,
        precision_policy=policy,
    )
    try:
        state_ckpt = _load_checkpoint(trainer, output)
        _assert_finite(trainer)

        itrs_per_epoch = len(trainer.dataloader)
        target_steps = int(trainer.num_itrs)
        resume_epoch = int(trainer.epoch)
        resume_itr = int(trainer.itr)
        epoch_offset = resume_itr - resume_epoch * itrs_per_epoch
        if epoch_offset != 0:
            raise RuntimeError(
                "checkpoint is not at an epoch boundary: epoch={} itr={} steps/epoch={} offset={}"
                .format(resume_epoch, resume_itr, itrs_per_epoch, epoch_offset)
            )
        if resume_epoch >= trainer.num_epochs or resume_itr >= target_steps:
            raise RuntimeError("checkpoint already reaches the configured training horizon")

        # Reproduce the seeded RandomSampler state through the saved epoch and
        # install the complete remaining epoch.  The crash offset (3720) is
        # intentionally included because no checkpoint exists at that point.
        order = replay_epoch_order(trainer, resume_epoch)
        if len(order) != itrs_per_epoch:
            raise RuntimeError(
                "sampler length {} does not match steps/epoch {}".format(
                    len(order), itrs_per_epoch
                )
            )
        install_replay_loader(trainer, order)
        trainer.dataset.set_epoch(resume_epoch)
        remaining_steps = target_steps - resume_itr
        metadata = {
            "label": "D_recovery",
            "source_root": str(source),
            "output_root": str(output),
            "resume_epoch": resume_epoch,
            "resume_step": resume_itr,
            "steps_per_epoch": itrs_per_epoch,
            "crash_epoch_offset": 35712 - resume_epoch * itrs_per_epoch,
            "target_steps": target_steps,
            "remaining_steps": remaining_steps,
            "precision_policy": policy,
            "optimizer_scheduler_restored": True,
            "adaptive_anchor_unchanged": True,
            "source_state_fields": sorted(state_ckpt.keys()),
        }
        _write_json(output / "recovery_manifest.json", metadata)
        trainer._diagnostic_summary.update(metadata)
        print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
        trainer.max_steps = target_steps
        trainer.run()
    finally:
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="experiments/formal_ablation/seed_1/D",
        help="failed D run containing the last normal checkpoint",
    )
    parser.add_argument(
        "--output",
        default="experiments/formal_ablation_recovery/seed_1/D_p3",
    )
    parser.add_argument("--policy", choices=("p3",), default="p3")
    args = parser.parse_args()
    run(ROOT / args.source, ROOT / args.output, args.policy)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""HM-HCMS-001 four-arm training (H0/H1/H2/H3).

Same protocol as qact_arms.py: R1 model_ema start, seed=1, constant
lr=1e-4, 8000 steps, eval at 0/2000/4000/8000. The ONLY difference is
opt['train']['data']['hcms_mode'] (off/random/hard/bg_control at
prob=0.25, strict). Monitors per-eval activation rate and (H2)
used_hard rate via a focal-loss-time counter.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "hydra"), str(ROOT / "tools")]

import tools.run_formal_ablation as rfa  # noqa: E402
from tools.qact_arms import run_diag_eval, EVAL_STEPS, LOCKED_LR  # noqa: E402

R1_OPT = ROOT / "experiments/allocator_diagnosis/seed_1/AUC/opt.yaml"
R1_CKPT = ROOT / "experiments/allocator_diagnosis/seed_1/AUC/models/last.pth"
AROOT = ROOT / "experiments/hcms_arms"
MAX_STEPS = 8000
ARM_MODES = {"H0": None, "H1": "random", "H2": "hard",
             "H3": "bg_control"}
HCMS_PROB = 0.25


def build_arm_trainer(arm):
    from libs import load_opt
    root = AROOT / arm
    root.mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(exist_ok=True)
    (root / "states").mkdir(exist_ok=True)
    opt = load_opt(str(R1_OPT), is_training=True)
    if ARM_MODES[arm]:
        opt["train"]["data"]["hcms_mode"] = ARM_MODES[arm]
        opt["train"]["data"]["hcms_len_mode"] = "strict"
        opt["train"]["data"]["hcms_prob"] = HCMS_PROB
    opt["seed"] = 1
    opt["_root"] = str(root)
    opt["_resume"] = False
    opt["_distributed"] = False
    opt["_world_size"] = 1

    monitor = {"samples": 0, "activated": 0, "used_hard": 0,
               "skipped": 0, "log": []}

    class HcmsTrainer(rfa.FormalTrainer):
        def __init__(self, opt, label, root, max_steps):
            self._monitor = monitor
            super().__init__(opt, label, root, max_steps=max_steps,
                             precision_policy="p3")

        def _install(self):
            ckpt = torch.load(R1_CKPT, map_location="cpu",
                              weights_only=False)
            self.model.load_compatible_state_dict(ckpt["model_ema"])
            self._ema_init()
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lambda e: 1.0)
            for g in self.optimizer.param_groups:
                g["lr"] = LOCKED_LR
                g["initial_lr"] = LOCKED_LR
            trainer = self
            # count HCMS stats by sampling dataset entries alongside
            # training steps (dataset object is accessible via
            # self.dataloader -> dataset)
            orig_jsonl = self._write_jsonl

            def jsonl_rec(record):
                orig_jsonl(record)
                if trainer.itr % 500 == 0 and trainer.itr > 0:
                    trainer._sample_hcms_stats()
                if trainer.itr in EVAL_STEPS and trainer.itr > 0:
                    trainer.eval_at(trainer.itr)

            self._write_jsonl = jsonl_rec

        def _sample_hcms_stats(self, n=20):
            import random as _r
            ds = self.dataloader.dataset
            n_act = n_hard = n_skip = 0
            for k in range(n):
                idx = _r.randrange(len(ds))
                ds[idx]                     # triggers _hcms_info
                info = getattr(ds, "_hcms_info", None)
                if info is None:
                    continue
                if info.get("skipped_contamination"):
                    n_skip += 1
                if info["activated"]:
                    n_act += 1
                    if info.get("used_hard"):
                        n_hard += 1
            self._monitor["log"].append({
                "itr": self.itr, "sampled": n,
                "activated": n_act, "used_hard": n_hard,
                "contamination_skipped": n_skip})
            print(f"[{self.formal_label}] itr={self.itr} "
                  f"hcms act={n_act}/{n} hard={n_hard}/{max(n_act,1)}"
                  f" skip={n_skip}/{n}", flush=True)

        def eval_at(self, step):
            mroot = Path(self.opt["_root"])
            ckpt_path = mroot / "models" / f"step{step}.pth"
            torch.save({"model": self._unwrap(self.model).state_dict(),
                        "model_ema": self.model_ema.state_dict()},
                       ckpt_path)
            eroot = mroot / "evals" / f"step{step}"
            (eroot / "models").mkdir(parents=True, exist_ok=True)
            dst = eroot / "models" / f"step{step}.pth"
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(ckpt_path.resolve())
            opt2 = load_opt(str(R1_OPT), is_training=False)
            opt2["_root"] = str(eroot)
            opt2["_ckpt"] = f"step{step}"
            metrics = run_diag_eval(opt2, eroot)
            (eroot / "metrics.json").write_text(
                json.dumps(metrics, indent=2) + "\n")
            print(f"[{self.formal_label}] step {step}: "
                  + " ".join(f"{k}={100*v:.2f}" for k, v in
                             metrics["official"].items()), flush=True)

    t = HcmsTrainer(opt, f"HCMS_{arm}", root, max_steps=MAX_STEPS)
    t._install()
    return t, monitor


def run_train(arm):
    t, monitor = build_arm_trainer(arm)
    t.model.train()
    t.eval_at(0)
    t.run()
    (Path(t.opt["_root"]) / "hcms_monitor.json").write_text(
        json.dumps(monitor, indent=2) + "\n")
    print(f"arm {arm} done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARM_MODES), required=True)
    run_train(ap.parse_args().arm)

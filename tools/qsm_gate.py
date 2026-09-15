#!/usr/bin/env python3
"""QSM-D012 small-set gate (HM-QSM-D012, section 16/17).

Short monitored training (default 800 steps) of the QSM-D012 config on
the formal trainer, recording per-interval:
  - grounding loss (vs the A-U-clean training curve)
  - QSM parameter grad norms per level
  - delta_q magnitude / query specificity (mean pairwise cosine between
    different queries' delta_q inside a batch)
  - finite checks on dt delta / params
Stops are manual (section 17 criteria evaluated from the report).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "hydra"), str(ROOT / "tools")]

import tools.run_formal_ablation as rfa  # noqa: E402

GATE_ROOT = ROOT / "experiments/qsm_gate"
STATS_JSON = GATE_ROOT / "qsm_gate_stats.json"


def run_gate(max_steps=800, interval=50):
    from libs import load_opt

    GATE_ROOT.mkdir(parents=True, exist_ok=True)
    models_dir = GATE_ROOT / "models"
    if models_dir.is_symlink():
        # a symlink here once let the 800-step gate checkpoint overwrite
        # the formal A-U-clean weights (HM-MAINT-029 incident); never
        # allow that again
        raise RuntimeError(
            "qsm_gate models must be a real directory, not a symlink")
    models_dir.mkdir(exist_ok=True)
    opt = load_opt(str(ROOT / "opts/research_QSM_D012.yaml"),
                   is_training=True)
    opt["seed"] = 1
    opt["_root"] = str(GATE_ROOT)
    opt["_resume"] = False
    opt["_distributed"] = False
    opt["_world_size"] = 1

    delta_captures = {}      # level -> list of (delta tensor,)
    grad_log = []
    stats_log = []

    class GateTrainer(rfa.FormalTrainer):
        def __init__(self, opt, label, root, max_steps,
                     precision_policy="p3"):
            super().__init__(opt, label, root, max_steps=max_steps,
                             precision_policy=precision_policy)
            self.qsm_params = {
                n: p for n, p in self.model.named_parameters()
                if "qsm_modulator" in n
            }
            print(f"QSM params tracked: {len(self.qsm_params)}")
            for lvl in (0, 1, 2):
                mod = self.model.vid_net.branch[lvl].qsm_modulator
                mod.register_forward_hook(self._make_hook(lvl))
            orig_step = self.optimizer.step

            def step_with_log(*a, **k):
                out = orig_step(*a, **k)
                if self.itr % interval == 0:
                    entry = {"itr": self.itr}
                    for n, p in self.qsm_params.items():
                        g = p.grad
                        entry[n] = (
                            float(g.norm()) if g is not None else None)
                        if not bool(torch.isfinite(p).all()):
                            raise FloatingPointError(f"nonfinite {n}")
                    grad_log.append(entry)
                return out

            self.optimizer.step = step_with_log

        @staticmethod
        def _make_hook(level):
            def hook(module, inputs, output):
                delta_captures.setdefault(level, []).append(
                    output.detach().float().cpu())
            return hook

        def _diagnostics(self, data_list, loss_dict, gradient, elapsed,
                         memory):
            record = super()._diagnostics(
                data_list, loss_dict, gradient, elapsed, memory)
            if self.itr % interval == 0:
                for lvl, caps in list(delta_captures.items()):
                    if not caps:
                        continue
                    d = torch.cat([c for c in caps if c.numel()], dim=0)
                    if d.size(0) >= 2:
                        dn = torch.nn.functional.normalize(d, dim=-1)
                        cos = (dn @ dn.T)
                        off = cos[~torch.eye(
                            cos.size(0), dtype=torch.bool)]
                        pair_cos = float(off.abs().mean())
                    else:
                        pair_cos = float("nan")
                    stats_log.append({
                        "itr": self.itr, "level": lvl,
                        "n_rows": int(d.size(0)),
                        "delta_mean": float(d.mean()),
                        "delta_absmean": float(d.abs().mean()),
                        "delta_absmax": float(d.abs().max()),
                        "delta_std": float(d.std()),
                        "pairwise_cos_absmean": pair_cos,
                        "finite": bool(torch.isfinite(d).all()),
                    })
                    delta_captures[lvl] = []
                record["qsm_delta_stats"] = [
                    s for s in stats_log[-6:]]
            return record

    trainer = GateTrainer(opt, "QSM_GATE", GATE_ROOT,
                          max_steps=max_steps, precision_policy="p3")
    try:
        trainer.run()
    finally:
        STATS_JSON.write_text(json.dumps({
            "grad_log": grad_log,
            "delta_stats": stats_log,
            "loss_meters": {
                k: float(v.sum / max(v.count, 1)) for k, v in trainer.loss_meters.items()
            },
            "max_steps": max_steps,
        }, indent=2) + "\n")
        print("stats ->", STATS_JSON)
        print("\n=== QSM gate summary (last entries) ===")
        for e in grad_log[-3:]:
            print({k: (round(v, 5) if isinstance(v, float) else v)
                   for k, v in e.items()})
        tail = [s for s in stats_log if s["itr"] >= max(0, max_steps - interval)]
        for s in tail:
            print(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=800)
    ap.add_argument("--interval", type=int, default=50)
    args = ap.parse_args()
    run_gate(args.max_steps, args.interval)


if __name__ == "__main__":
    main()

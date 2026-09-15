#!/usr/bin/env python3
"""HM-QCPH-036: query-conditioned proposal head (FiLM) two arms."""
import argparse, json, shutil, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"hydra"), str(ROOT/"tools")]
import tools.run_formal_ablation as rfa
from tools.qact_arms import run_diag_eval, EVAL_STEPS, LOCKED_LR

R1_OPT = ROOT/"experiments/allocator_diagnosis/seed_1/AUC/opt.yaml"
R1_CKPT = ROOT/"experiments/allocator_diagnosis/seed_1/AUC/models/last.pth"
AROOT = ROOT/"experiments/qcph_arms"
MAX_STEPS = 8000

def build(arm):
    from libs import load_opt
    root = AROOT/arm; root.mkdir(parents=True, exist_ok=True)
    (root/"models").mkdir(exist_ok=True); (root/"states").mkdir(exist_ok=True)
    cfg = R1_OPT if arm == "G0" else ROOT/"opts/research_QCPH_G1.yaml"
    opt = load_opt(str(cfg), is_training=True)
    opt["seed"]=1; opt["_root"]=str(root); opt["_resume"]=False
    opt["_distributed"]=False; opt["_world_size"]=1
    class T(rfa.FormalTrainer):
        def __init__(self, o, l, r, m):
            super().__init__(o, l, r, max_steps=m, precision_policy="p3")
        def _install(self):
            c = torch.load(R1_CKPT, map_location="cpu", weights_only=False)
            self.model.load_compatible_state_dict(c["model_ema"])
            self._ema_init()
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lambda e: 1.0)
            for g in self.optimizer.param_groups:
                g["lr"]=LOCKED_LR; g["initial_lr"]=LOCKED_LR
            orig = self._write_jsonl
            trainer = self
            def rec(r):
                orig(r)
                if trainer.itr in EVAL_STEPS and trainer.itr > 0:
                    trainer.eval_at(trainer.itr)
            self._write_jsonl = rec
        def eval_at(self, step):
            mr = Path(self.opt["_root"])
            cp = mr/"models"/f"step{step}.pth"
            torch.save({"model":self._unwrap(self.model).state_dict(),
                        "model_ema":self.model_ema.state_dict()}, cp)
            er = mr/"evals"/f"step{step}"
            (er/"models").mkdir(parents=True, exist_ok=True)
            d = er/"models"/f"step{step}.pth"
            if d.exists() or d.is_symlink(): d.unlink()
            d.symlink_to(cp.resolve())
            cfg2 = R1_OPT if self.formal_label.endswith('G0') else ROOT/'opts/research_QCPH_G1.yaml'
            o2 = load_opt(str(cfg2), is_training=False)
            o2["_root"]=str(er); o2["_ckpt"]=f"step{step}"
            m = run_diag_eval(o2, er)
            (er/"metrics.json").write_text(json.dumps(m, indent=2)+"\n")
            print(f"[{self.formal_label}] step {step}: "+
                  " ".join(f"{k}={100*v:.2f}" for k,v in m["official"].items()),
                  flush=True)
    t = T(opt, f"QCPH_{arm}", root, MAX_STEPS)
    t._install()
    return t

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["G0","G1"], required=True)
    args = ap.parse_args()
    t = build(args.arm); t.model.train(); t.eval_at(0); t.run()
    print(f"arm {args.arm} done")

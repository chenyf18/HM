#!/usr/bin/env python3
"""Allocator/innovation experiment runner (rebuilt after overwrite).

Runs labels sequentially; reuses completed runs; writes per-seed results
json and summary markdown. The original corrupted content is backed up at
/tmp/corrupt_runner_backup.py (it was worker.py text pasted over this file).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hydra"))

import tools.run_formal_ablation as rfa  # noqa: E402

CONFIGS = {
    "AU": ROOT / "opts/research_allocator_AU_uniform.yaml",
    "AR0": ROOT / "opts/research_allocator_AR0_random.yaml",
    "AR1": ROOT / "opts/research_allocator_AR1_random.yaml",
    "AR2": ROOT / "opts/research_allocator_AR2_random.yaml",
    "AL": ROOT / "opts/research_allocator_AL_learned.yaml",
    "AO": ROOT / "opts/research_allocator_AO_oracle.yaml",
    "AUC": ROOT / "opts/research_allocator_AUC_clean.yaml",
    "ALC": ROOT / "opts/research_allocator_ALC_learned.yaml",
    "ARC": ROOT / "opts/research_allocator_ARC_s1.yaml",
    "R005": ROOT / "opts/research_ratio_R005.yaml",
    "R020": ROOT / "opts/research_ratio_R020.yaml",
    "R030": ROOT / "opts/research_ratio_R030.yaml",
    "R000": ROOT / "opts/research_ratio_R000.yaml",
    "AUCP1": ROOT / "opts/research_allocator_AUCP1.yaml",
    "ABR": ROOT / "opts/research_allocator_ABR_balanced.yaml",
    "ALMR": ROOT / "opts/research_allocator_ALMR_listwise.yaml",
    "ASC": ROOT / "opts/research_allocator_ASC_supervised.yaml",
    "ALDN": ROOT / "opts/research_allocator_ALDN_denoise.yaml",
    "AQR": ROOT / "opts/research_allocator_AQR_clean.yaml",
    "ASB": ROOT / "opts/research_allocator_ASB_statebridge.yaml",
    "T1": ROOT / "opts/research_T1_qstb.yaml",
    "T1S": ROOT / "opts/research_T1S_state.yaml",
    "T1F": ROOT / "opts/research_T1F_feature.yaml",
    "HQR": ROOT / "opts/research_HQR.yaml",
    "QSM": ROOT / "opts/research_QSM_D012.yaml",
    "CDFB": ROOT / "opts/research_CDF_B.yaml",
    "CDFC": ROOT / "opts/research_CDF_C.yaml",
    "QACTG1": ROOT / "opts/research_QACT_G1.yaml",
    "QACTG2": ROOT / "opts/research_QACT_G2.yaml",
    "HCMSH1": ROOT / "opts/research_HCMS_H1.yaml",
    "HCMSH2": ROOT / "opts/research_HCMS_H2.yaml",
    "HCMSH3": ROOT / "opts/research_HCMS_H3.yaml",
    "QCPH": ROOT / "opts/research_QCPH_G1.yaml",
    "TEFMG1": ROOT / "opts/research_TEFM_G1.yaml",
    "TEFMG2": ROOT / "opts/research_TEFM_G2.yaml",
    "HQRN": ROOT / "opts/research_HQRN.yaml",
    "TAUC": ROOT / "opts/tacos_au_clean.yaml",
    "TACTRL": ROOT / "opts/tacos_control.yaml",
    "TACORIG": ROOT / "opts/tacos_hieramamba.yaml",
}

ORDER = ["AU", "AR0", "AR1", "AR2", "AL", "AO", "AUC", "ALC", "ARC",
         "R005", "R020", "R030", "AUCP1", "ABR", "ALMR", "ASC", "ALDN",
         "AQR", "ASB", "T1", "T1S", "T1F", "TAUC", "TACTRL"]

GATED_LABELS = {"AR0", "AR1", "AR2", "AL", "AO"}
BASELINE_A = (ROOT /
              "experiments/formal_ablation/seed_1/A/evaluation_summary.json")
PRECISION_POLICY = "p3"
GATE_THRESHOLD_PP = 1.0
METRIC_KEYS = ("Rank@1_IoU@0.3", "Rank@1_IoU@0.5",
               "Rank@5_IoU@0.3", "Rank@5_IoU@0.5")

GROUPING_LABELS = {
    "A": "Original fixed stride-2", "AU": "Uniform adaptive pipeline",
    "AR0": "Random seed 0", "AR1": "Random seed 1", "AR2": "Random seed 2",
    "AL": "Learned", "AO": "Oracle diagnostic (GT leakage)",
    "AUC": "Uniform clean", "ALC": "Learned + lc-ACC",
    "ARC": "Random s1 + lc-ACC", "R005": "BTRR ratio .05",
    "R000": "BTRR ratio 0 (=AUC)", "R020": "BTRR ratio .20",
    "R030": "BTRR ratio .30", "AUCP1": "AUC max_num_text=1",
    "ABR": "BTRR r0.1 + ranking", "ALMR": "CLC-LMR listwise",
    "ASC": "GT-supervised allocation", "ALDN": "DN span denoising",
    "AQR": "QR-TG objectives", "ASB": "StateBridge priming",
    "T1": "QSTB state-transition", "T1S": "QSTB state-only ctrl",
    "T1F": "QSTB feature-diff ctrl",
    "HQR": "Query hierarchy routing",
    "QSM": "QSM-Delta dt mod L0-2",
    "CDFB": "CDF unconstrained full",
    "CDFC": "CDF conservative full", "HQRN": "HQR no-query ctrl", "TAUC": "TACoS A-U-clean port",
    "TACTRL": "TACoS non-adaptive control",
}


def evaluate_gate(au, base):
    au_v = [100 * float(au[k]) for k in METRIC_KEYS]
    b_v = [100 * float(base[k]) for k in METRIC_KEYS]
    d = sum(au_v) / 4 - sum(b_v) / 4
    return {"passed": bool(d >= -GATE_THRESHOLD_PP),
            "mean_delta_pp": d,
            "decision": ("pass" if d >= -GATE_THRESHOLD_PP else
                         "AU below A by {:.2f}pp; queue stopped".format(d))}


def write_summary(path, results, base):
    lines = ["# Allocator/innovation results (percent)", "",
             "| Exp | Grouping | R1@0.3 | R1@0.5 | R5@0.3 | R5@0.5 | Avg |",
             "| --- | --- | ---: | ---: | ---: | ---: | --: |"]
    rows = dict(results)
    if base:
        rows["A"] = base
    for label in ["A"] + [x for x in ORDER if x in rows]:
        m = rows.get(label)
        if not m:
            continue
        v = [100 * float(m[k]) for k in METRIC_KEYS]
        lines.append("| {} | {} | {} | {:.4f} |".format(
            label, GROUPING_LABELS.get(label, label),
            " | ".join("{:.4f}".format(x) for x in v), sum(v) / 4))
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--output-root", default="experiments/allocator_diagnosis")
    ap.add_argument("--only", default=",".join(ORDER))
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    selected = [x for x in args.only.split(",") if x]
    unknown = [x for x in selected if x not in CONFIGS]
    if unknown:
        raise SystemExit("unknown labels: {}".format(unknown))

    seed_root = ROOT / args.output_root / "seed_{}".format(args.seed)
    seed_root.mkdir(parents=True, exist_ok=True)
    rfa.CONFIGS = CONFIGS

    baseline = None
    if not args.max_steps:
        baseline = json.loads(BASELINE_A.read_text())["metrics"]

    results_path = seed_root / "allocator_seed_{}_results.json".format(
        args.seed)
    summary_path = seed_root / "allocator_seed_{}_summary.md".format(
        args.seed)
    gate_path = seed_root / "gate_decision.json"
    results = {}
    if results_path.exists():
        results = json.loads(results_path.read_text()).get("metrics", {})

    gate_passed = True
    if gate_path.exists():
        gate_passed = bool(
            json.loads(gate_path.read_text()).get("passed", False))

    for label in selected:
        if label != "AU" and label in GATED_LABELS and not gate_passed:
            print("Gate failed; stopping queue before", label)
            break
        print("Starting", label, "seed", args.seed, flush=True)
        metrics = rfa.run_one(label, args.seed, seed_root / label,
                              max_steps=args.max_steps,
                              evaluate=not args.no_eval,
                              precision_policy=PRECISION_POLICY)
        if metrics and not args.max_steps:
            results[label] = metrics
            results_path.write_text(json.dumps(
                {"seed": args.seed,
                 "precision_policy": PRECISION_POLICY,
                 "metrics": results}, indent=2) + "\n")
            write_summary(summary_path, results, baseline)
        if label == "AU" and not args.max_steps and baseline:
            gate = evaluate_gate(metrics, baseline)
            gate_path.write_text(json.dumps(gate, indent=2) + "\n")
            print("Gate:", gate["decision"], flush=True)
            gate_passed = gate["passed"]

    write_summary(summary_path, results, baseline)
    print("Done; results at", results_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Aggregate multi-seed AUC vs R020 results (HM-MSEED-009)."""
import json, statistics as st
from pathlib import Path
ROOT = Path("/data/nh/hieramamba-main/experiments/allocator_diagnosis")
KEYS = ("Rank@1_IoU@0.3","Rank@1_IoU@0.5","Rank@5_IoU@0.3","Rank@5_IoU@0.5")
def metrics(seed_root, label):
    p = seed_root / label / "evaluation_summary.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())["metrics"]
    return [100.0*float(m[k]) for k in KEYS]
rows = []
for seed in (1, 2, 3):
    root = ROOT / f"seed_{seed}"
    au, br = metrics(root, "AUC"), metrics(root, "R020")
    if au and br:
        rows.append((seed, au, br))
print("| Seed | A-U Mean | BR-r020 Mean | BR-AU |")
print("|---:|---:|---:|---:|")
for seed, au, br in rows:
    print(f"| {seed} | {sum(au)/4:.2f} | {sum(br)/4:.2f} | {sum(br)/4-sum(au)/4:+.2f} |")
if rows:
    for name, idx in (("Mean", 4), ("R1@0.3",0),("R1@0.5",1),("R5@0.3",2),("R5@0.5",3)):
        au_vals = [sum(au)/4 if idx==4 else au[idx] for _,au,_ in rows]
        br_vals = [sum(br)/4 if idx==4 else br[idx] for _,_,br in rows]
        deltas = [b-a for a,b in zip(au_vals,br_vals)]
        print(f"{name}: AU={st.mean(au_vals):.2f}±{st.stdev(au_vals) if len(au_vals)>1 else 0:.2f} "
              f"BR={st.mean(br_vals):.2f}±{st.stdev(br_vals) if len(br_vals)>1 else 0:.2f} "
              f"delta={st.mean(deltas):+.2f}±{st.stdev(deltas) if len(deltas)>1 else 0:.2f}")

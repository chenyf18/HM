#!/usr/bin/env python3
"""HM-CCC-037: Counterfactual Candidate Correction.

Learns whether the baseline winner (top-1) should be replaced by a
challenger. Frozen backbone; <2M params; NMS unchanged.

Stage 1: oracle pair generation + sanity stats
Stage 2: train CCC module (306 updates ≈ 1 pass over switch pairs)
Stage 3: evaluate (switch precision / corrected / regressed / net gain)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "hydra"), str(ROOT / "tools")]

from tools.audit_ranking_bottleneck import iou_1d  # noqa

CROOT = ROOT / "experiments/ccc"
TRAIN_NPZ = ROOT / "experiments/csrm/train_topk.npz"
VAL_NPZ = ROOT / "experiments/csrm/val_topk.npz"
CKPT = CROOT / "ccc.pt"
RESULTS = CROOT / "ccc_results.json"
FEAT = 384
QDIM = 384
SWITCH_DELTA = 0.1


def decode_iou(arr):
    n = len(arr["gt_s"])
    ious = np.zeros((n, 50), np.float32)
    starts = np.zeros((n, 50), np.float32)
    ends = np.zeros((n, 50), np.float32)
    for i in range(n):
        k = int(arr["n_valid"][i])
        if k == 0:
            continue
        c = arr["center"][i, :k]
        sc = arr["scale"][i, :k]
        s = c - arr["off"][i, :k, 0] * sc
        e = c + arr["off"][i, :k, 1] * sc
        cs_, cc_, fps_, dur_ = (arr["clip_stride"][i],
                                arr["clip_size"][i],
                                arr["fps"][i], arr["duration"][i])
        ss = np.clip((s * cs_ + 0.5 * cc_) / fps_, 0, dur_)
        ee = np.clip((e * cs_ + 0.5 * cc_) / fps_, 0, dur_)
        starts[i, :k] = s
        ends[i, :k] = e
        ious[i, :k] = iou_1d(ss, ee, arr["gt_s"][i], arr["gt_e"][i])
    return starts, ends, ious


# ---------------------------------------------------------------------------
# stage 1: oracle pairs
# ---------------------------------------------------------------------------

def gen_pairs():
    for split, npz_path in (("train", TRAIN_NPZ), ("val", VAL_NPZ)):
        arr = np.load(npz_path)
        _, _, ious = decode_iou(arr)
        n = len(arr["gt_s"])
        stats = {
            "n_queries": n,
            "winner_iou_ge.5": 0,       # keep-safe
            "winner_iou_lt.5": 0,       # winner wrong
            "winner_wrong_with_better": 0,  # challenger exists
            "winner_wrong_no_better": 0,
            "switch_pairs": 0,
            "keep_pairs": 0,
            "ignore_queries": 0,
        }
        pairs = []  # (qi, challenger_idx, label, delta_iou)
        for i in range(n):
            k = int(arr["n_valid"][i])
            if k < 2:
                stats["ignore_queries"] += 1
                continue
            w_iou = ious[i, 0]  # top-1 by cls score
            if w_iou >= 0.5:
                stats["winner_iou_ge.5"] += 1
                # keep pairs: winner is good, challengers that would hurt
                for j in range(1, k):
                    if ious[i, j] < 0.3:
                        pairs.append((i, j, 0, ious[i, j] - w_iou))
                        stats["keep_pairs"] += 1
            else:
                stats["winner_iou_lt.5"] += 1
                # find best challenger
                best_j = 1 + int(np.argmax(ious[i, 1:k]))
                best_iou = ious[i, best_j]
                if best_iou - w_iou >= SWITCH_DELTA:
                    stats["winner_wrong_with_better"] += 1
                    # switch pair: this challenger
                    pairs.append((i, best_j, 1, best_iou - w_iou))
                    stats["switch_pairs"] += 1
                    # also add some keep negatives from same query
                    for j in range(1, min(k, 6)):
                        if j != best_j and ious[i, j] < 0.3:
                            pairs.append((i, j, 0, ious[i, j] - w_iou))
                            stats["keep_pairs"] += 1
                else:
                    stats["winner_wrong_no_better"] += 1
        print(f"\n[{split}] oracle pairs:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        print(f"  total pairs: {len(pairs)}")
        # save
        CROOT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            CROOT / f"{split}_pairs.npz",
            qi=np.array([p[0] for p in pairs], np.int32),
            cj=np.array([p[1] for p in pairs], np.int32),
            label=np.array([p[2] for p in pairs], np.int32),
            delta=np.array([p[3] for p in pairs], np.float32),
        )
    print("\noracle pair generation done")


# ---------------------------------------------------------------------------
# stage 2: CCC module
# ---------------------------------------------------------------------------

class CCCModule(nn.Module):
    """<2M params. Input: winner feat + challenger feat + query."""

    def __init__(self, feat_dim=FEAT, q_dim=QDIM, hidden=256):
        super().__init__()
        in_dim = feat_dim * 3 + q_dim  # winner + challenger + diff + query
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, w_feat, c_feat, query):
        diff = w_feat - c_feat
        x = torch.cat([w_feat, c_feat, diff, query], dim=-1)
        return self.net(x).squeeze(-1)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


def run_train(n_updates=306):
    tr_pairs = np.load(CROOT / "train_pairs.npz")
    tr_npz = np.load(TRAIN_NPZ)
    va_pairs = np.load(CROOT / "val_pairs.npz")
    va_npz = np.load(VAL_NPZ)

    dev = "cuda:0"
    torch.manual_seed(0)
    module = CCCModule().to(dev)
    print(f"CCC params: {module.param_count():,} "
          f"({'OK' if module.param_count() < 2_000_000 else 'OVER'})")
    opt = torch.optim.AdamW(module.parameters(), lr=1e-3, weight_decay=0.01)

    # build training tensors
    qi_t = torch.tensor(tr_pairs["qi"], device=dev)
    cj_t = torch.tensor(tr_pairs["cj"], device=dev)
    y_t = torch.tensor(tr_pairs["label"], dtype=torch.float32, device=dev)
    feat_tr = torch.tensor(
        tr_npz["feat"].astype(np.float32), device=dev)
    q_tr = torch.tensor(tr_npz["q384"], device=dev)

    w_f = feat_tr[qi_t, 0]         # winner = top-1
    c_f = feat_tr[qi_t, cj_t]      # challenger
    q = q_tr[qi_t]

    n = len(y_t)
    bs = 64
    steps_per_epoch = max(1, n // bs)
    n_epochs = max(1, math.ceil(n_updates / steps_per_epoch))
    step = 0
    for ep in range(n_epochs):
        perm = torch.randperm(n, device=dev)
        for i0 in range(0, n, bs):
            if step >= n_updates:
                break
            idx = perm[i0:i0 + bs]
            logit = module(w_f[idx], c_f[idx], q[idx])
            loss = F.binary_cross_entropy_with_logits(
                logit, y_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
        if step >= n_updates:
            break
    torch.save(module.state_dict(), CKPT)
    print(f"trained {step} updates -> {CKPT}")


# ---------------------------------------------------------------------------
# stage 3: evaluation
# ---------------------------------------------------------------------------

def run_eval(threshold=0.5):
    va_npz = np.load(VAL_NPZ)
    _, _, ious = decode_iou(va_npz)
    n = len(va_npz["gt_s"])
    dev = "cuda:0"
    module = CCCModule().to(dev).eval()
    module.load_state_dict(torch.load(CKPT, weights_only=True))

    feat_va = torch.tensor(
        va_npz["feat"].astype(np.float32), device=dev)
    q_va = torch.tensor(va_npz["q384"], device=dev)

    # for each query: baseline winner IoU vs corrected IoU
    baseline_hits = np.zeros(4)   # R1@0.3/0.5, R5@0.3/0.5
    ccc_hits = np.zeros(4)
    n_switch = n_correct_switch = 0
    n_keep = n_correct_keep = 0
    n_corrected = n_regressed = 0
    # also track: among switched, how many improved vs worsened
    switched = 0
    improved = 0
    worsened = 0

    with torch.no_grad():
        for i in range(n):
            k = int(va_npz["n_valid"][i])
            if k < 2:
                baseline_hits += (ious[i, 0] >= .3 if k else 0,
                                  ious[i, 0] >= .5 if k else 0, 0, 0)
                ccc_hits += baseline_hits[-4:]
                continue
            w_iou = ious[i, 0]
            baseline_hits += (w_iou >= .3, w_iou >= .5, 0, 0)

            # find best challenger by CCC
            w_f = feat_va[i, 0].unsqueeze(0).expand(k - 1, -1)
            c_f = feat_va[i, 1:k]
            q = q_va[i].unsqueeze(0).expand(k - 1, -1)
            probs = torch.sigmoid(module(w_f, c_f, q)).cpu().numpy()

            best_j = 1 + int(np.argmax(probs))
            best_p = probs[best_j - 1]
            c_iou = ious[i, best_j]

            if best_p >= threshold and c_iou != w_iou:
                # CCC says switch
                switched += 1
                n_switch += 1
                if c_iou > w_iou:
                    improved += 1
                    n_correct_switch += 1
                else:
                    worsened += 1
                final_iou = c_iou
            else:
                n_keep += 1
                if w_iou >= 0.5:
                    n_correct_keep += 1
                final_iou = w_iou

            ccc_hits += (final_iou >= .3, final_iou >= .5, 0, 0)
            if final_iou > w_iou:
                n_corrected += 1
            elif final_iou < w_iou:
                n_regressed += 1

    out = {
        "n_queries": n,
        "threshold": threshold,
        "baseline_R1@0.3": float(baseline_hits[0] / n),
        "baseline_R1@0.5": float(baseline_hits[1] / n),
        "ccc_R1@0.3": float(ccc_hits[0] / n),
        "ccc_R1@0.5": float(ccc_hits[1] / n),
        "n_switched": switched,
        "switch_rate": switched / n,
        "switch_precision": n_correct_switch / max(switched, 1),
        "n_corrected": n_corrected,
        "n_regressed": n_regressed,
        "net_gain_R1@0.5": float(
            (ccc_hits[1] - baseline_hits[1]) / n),
        "net_gain_R1@0.3": float(
            (ccc_hits[0] - baseline_hits[0]) / n),
    }
    RESULTS.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print("->", RESULTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["pairs", "train", "eval"])
    ap.add_argument("--updates", type=int, default=306)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    if args.stage == "pairs":
        gen_pairs()
    elif args.stage == "train":
        run_train(args.updates)
    else:
        run_eval(args.threshold)


if __name__ == "__main__":
    main()

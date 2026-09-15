#!/usr/bin/env python3
"""Query Hard Negative Audit (HM-QHNA-028).

Zero-training audit of top-ranked false positives. Candidate table =
HM-AUDIT-RANK-021 pool (re-exported with per-candidate query cosines in
experiments/qc_audit/val_rank.npz; identical forward, fp32 decode).

Stages:
  types      : per-query top-50 by cls score; positives IoU>=0.5, hard
               negatives rank<=20 (and <=50) with IoU<0.3; feature
               (cos_F0/F1/F2), temporal-distance, duration and overlap
               statistics; A/B/C typing of every hard negative.
  dump-train / dump-val : light dumps capturing F0/F1/F2 for exactly the
               top-50-by-score rows plus up to 20 best-IoU positives
               (binary-probe training data).
  probe      : frozen binary probe [F1, q] -> P(positive), trained on the
               train dump; AUROC + pairwise accuracy on val top-50.
  analyze    : competition margins (cls vs probe vs IoU-regression head)
               and the final Case A/B verdict data.
"""
from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "hydra"), str(ROOT / "tools")]

from tools.audit_ranking_bottleneck import (  # noqa: E402
    AU_ROOT, iou_1d, rankdata_avg,
)
from tools.qc_audit import (  # noqa: E402
    AUC_P32_ROOT, build_evaluator, iou_seconds, center_seconds,
)

QROOT = ROOT / "experiments/qhna_audit"
VAL_RANK_NPZ = ROOT / "experiments/qc_audit/val_rank.npz"
TRAIN_NPZ = QROOT / "train_top.npz"
VAL_NPZ = QROOT / "val_top.npz"
PROBE_PT = QROOT / "probe.pt"
RESULTS_JSON = QROOT / "qhna_results.json"

TOPK = 50
HN_RANK = 20
POS_IOU = 0.5
HN_IOU = 0.3
MAX_POS = 20


# ---------------------------------------------------------------------------
# stage: types (from the full-pool table)
# ---------------------------------------------------------------------------

def run_types():
    rk = np.load(VAL_RANK_NPZ)
    pool = rk["pool"]
    offs = rk["pool_offsets"]
    ious = rk["ious"]
    cs, cc = rk["clip_stride"], rk["clip_size"]
    fps, dur = rk["fps"], rk["duration"]
    gt_s, gt_e = rk["gt_start"], rk["gt_end"]
    cosF0_all = rk["cos_F0"]
    cosF1_all = rk["cos_F1"]
    cosF2_all = rk["cos_F2"]
    n = len(gt_s)

    rows = []          # per hard negative: metrics
    pos_rows = []      # per positive: same metrics
    n_pos50 = 0
    n_hn20 = n_hn50 = 0
    queries_with_both = 0
    per_q_margin_cls = []

    for i in range(n):
        sl = slice(offs[i], offs[i + 1])
        p = pool[sl]
        if p.shape[0] == 0:
            continue
        order = np.argsort(-p[:, 1], kind="stable")
        p = p[order][:TOPK]
        io = ious[sl][order][:TOPK]
        cf0 = cosF0_all[sl][order][:TOPK]
        cf1 = cosF1_all[sl][order][:TOPK]
        cf2 = cosF2_all[sl][order][:TOPK]

        s = np.clip((p[:, 7] * cs[i] + 0.5 * cc[i]) / fps[i], 0, dur[i])
        e = np.clip((p[:, 8] * cs[i] + 0.5 * cc[i]) / fps[i], 0, dur[i])
        inter = np.maximum(
            0.0, np.minimum(e, gt_e[i]) - np.maximum(s, gt_s[i]))
        centers = 0.5 * (s + e)
        gt_c = 0.5 * (gt_s[i] + gt_e[i])
        gt_dur = max(gt_e[i] - gt_s[i], 1e-6)
        dist = np.abs(centers - gt_c)               # seconds
        dur_ratio = (e - s) / gt_dur

        is_pos = io >= POS_IOU
        is_hn = io < HN_IOU
        n_pos50 += int(is_pos.sum())

        def metrics(k):
            return {
                "iou": float(io[k]), "cls": float(p[k, 1]),
                "cos_F0": float(cf0[k]), "cos_F1": float(cf1[k]),
                "cos_F2": float(cf2[k]),
                "dist_s": float(dist[k]), "dist_norm": float(
                    dist[k] / gt_dur),
                "overlap": float(inter[k]), "dur_ratio": float(
                    dur_ratio[k]), "level": int(p[k, 2]),
            }

        pos_idx = np.where(is_pos)[0]
        for k in pos_idx:
            pos_rows.append(metrics(k))
        hn_idx20 = np.where(is_hn)[0]
        hn_idx20 = hn_idx20[hn_idx20 < HN_RANK]
        hn_idx50 = np.where(is_hn)[0]
        n_hn20 += len(hn_idx20)
        n_hn50 += len(hn_idx50)
        for k in hn_idx50:
            rows.append(metrics(k))
        if len(pos_idx) and len(hn_idx20):
            queries_with_both += 1
            per_q_margin_cls.append(
                float(p[pos_idx].max(axis=0)[1]
                      - p[hn_idx20].max(axis=0)[1]))

    pos_med_cosF1 = float(np.median([r["cos_F1"] for r in pos_rows])) \
        if pos_rows else float("nan")

    # A/B/C typing for rank<=20 hard negatives
    types = {"A_semantic_wrong_time": 0, "B_wrong_region": 0,
             "C_boundary_or_duration": 0}
    for idx, r in enumerate(rows):
        if idx >= n_hn20:
            break                     # rank<=20 subset only
        if r["overlap"] > 1e-6:
            types["C_boundary_or_duration"] += 1
        elif r["cos_F1"] >= pos_med_cosF1:
            types["A_semantic_wrong_time"] += 1
        else:
            types["B_wrong_region"] += 1

    def agg(rs, key):
        v = np.array([r[key] for r in rs], dtype=float)
        if len(v) == 0:
            return {"mean": float("nan"), "median": float("nan")}
        return {"mean": float(v.mean()), "median": float(np.median(v))}

    out = {
        "n_queries": n,
        "n_positives_top50": n_pos50,
        "n_hard_neg_rank20": n_hn20,
        "n_hard_neg_rank50": n_hn50,
        "queries_with_pos_and_hn20": queries_with_both,
        "positive_median_cosF1": pos_med_cosF1,
        "positive_stats": {k: agg(pos_rows, k) for k in
                           ("cos_F0", "cos_F1", "cos_F2", "dist_norm",
                            "dur_ratio", "cls")},
        "hardneg_stats_rank50": {k: agg(rows, k) for k in
                                 ("cos_F0", "cos_F1", "cos_F2", "dist_norm",
                                  "dur_ratio", "overlap", "cls")},
        "hardneg_types_rank20": types,
        "cls_margin_pos_minus_hn20": {
            "mean": float(np.mean(per_q_margin_cls))
            if per_q_margin_cls else float("nan"),
            "median": float(np.median(per_q_margin_cls))
            if per_q_margin_cls else float("nan")},
    }
    QROOT.mkdir(parents=True, exist_ok=True)
    (QROOT / "qhna_types.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")

    print(f"queries {n} | positives(top50) {n_pos50} | HN rank20 {n_hn20} "
          f"| rank50 {n_hn50} | queries w/ both {queries_with_both}")
    print(f"positive median cos_F1 = {pos_med_cosF1:.4f}")
    print("\nmetric                 positives   hard-negatives(r50)")
    for k in ("cos_F0", "cos_F1", "cos_F2", "dist_norm", "dur_ratio",
              "cls"):
        print(f"  {k:18s} {out['positive_stats'][k]['mean']:9.4f}   "
              f"{out['hardneg_stats_rank50'][k]['mean']:9.4f}")
    tot = sum(types.values())
    print("\nhard-negative types (rank20):")
    for k, v in types.items():
        print(f"  {k:26s} {v:7d}  ({100*v/max(tot,1):.1f}%)")


# ---------------------------------------------------------------------------
# stage: light dumps (top-50 by score + positives), reusing the QC evaluator
# ---------------------------------------------------------------------------

_STASH = {}


def _patched_iou_seconds(rec, pool):
    _STASH["pool"] = pool
    return iou_seconds(rec, pool)


def _patched_balanced_select(rng, ious):
    pool = _STASH.get("pool")
    if pool is None or pool.shape[0] == 0:
        return np.zeros(0, dtype=int)
    scores = pool[:, 1]
    order = np.argsort(-scores, kind="stable")
    top = order[:TOPK]
    pos_idx = np.where(ious >= POS_IOU)[0]
    if len(pos_idx) > MAX_POS:
        pos_idx = pos_idx[np.argsort(-ious[pos_idx])[:MAX_POS]]
    sel = np.unique(np.concatenate([top, pos_idx]))
    return np.sort(sel)


def run_dump(split, limit_videos=None):
    import tools.qc_audit as qc
    qc.QROOT = QROOT                      # isolated root + fresh AU link
    # Retrained A-U-clean weights (HM-MAINT-029 recovery; Mean 25.04,
    # within noise of the original 25.07). Formal baseline for recheck.
    ckpt_root = ROOT / "experiments/allocator_diagnosis/seed_1/AUC/models"
    QROOT.mkdir(parents=True, exist_ok=True)
    models_link = QROOT / "models"
    if models_link.exists() or models_link.is_symlink():
        models_link.unlink()
    models_link.symlink_to(ckpt_root, target_is_directory=True)
    opt_yaml = QROOT / f"opt_{split}.yaml"
    if opt_yaml.exists():
        opt_yaml.unlink()
    shutil.copyfile(AUC_P32_ROOT / "opt.yaml", opt_yaml)
    qc.iou_seconds = _patched_iou_seconds
    qc.balanced_select = _patched_balanced_select
    ev = build_evaluator(split, limit_videos=limit_videos)
    ev.run()
    recs = ev.records
    n = len(recs)
    lengths = [r["sample_pool"].shape[0] for r in recs]
    rows = np.concatenate([r["sample_pool"] for r in recs])
    ious = np.concatenate([r["sample_ious"] for r in recs])
    qi = np.concatenate([
        np.full(lengths[i], i, dtype=np.int64) for i in range(n)])
    out = TRAIN_NPZ if split == "train" else VAL_NPZ
    np.savez_compressed(
        out, rows=rows, ious=ious, qi=qi,
        F1=np.concatenate([r["sample_F1"] for r in recs]),
        F0=np.concatenate([r["sample_F0"] for r in recs]),
        F2=np.concatenate([r["sample_F2"] for r in recs]),
        q384=qc_pack(ev, recs, "q384"),
        gt=qc_pack(ev, recs, "gt"))
    print(f"[{split}] dumped {sum(lengths)} rows over {n} queries -> {out}")


def qc_pack(ev, recs, key):
    if key == "q384":
        return np.stack([r["q384"] for r in recs])
    return np.stack([[r["gt_start"], r["gt_end"]] for r in recs])


# ---------------------------------------------------------------------------
# stage: probe
# ---------------------------------------------------------------------------

def run_probe():
    tr = np.load(TRAIN_NPZ)
    va = np.load(VAL_NPZ)
    dev = "cuda:0"

    def prep(d):
        q = d["q384"][d["qi"]]
        x = np.concatenate([d["F1"].astype(np.float32), q], axis=1)
        y = (d["ious"] >= POS_IOU).astype(np.float32)
        return torch.tensor(x), torch.tensor(y), torch.tensor(d["qi"])

    Xtr, Ytr, Qtr = prep(tr)
    Xva, Yva, Qva = prep(va)
    print(f"train rows {len(Ytr)} (pos {int(Ytr.sum())}), "
          f"val rows {len(Yva)} (pos {int(Yva.sum())})")

    torch.manual_seed(5)
    probe = torch.nn.Sequential(
        torch.nn.Linear(Xtr.size(1), 128), torch.nn.GELU(),
        torch.nn.Linear(128, 1)).to(dev)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3,
                           weight_decay=1e-4)
    n = len(Ytr)
    for ep in range(10):
        perm = torch.randperm(n)
        for i in range(0, n, 8192):
            idx = perm[i:i + 8192]
            logit = probe(Xtr[idx].to(dev)).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logit, Ytr[idx].to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    with torch.no_grad():
        p_va = torch.cat([
            torch.sigmoid(probe(
                Xva[i:i + (1 << 17)].to(dev)).squeeze(-1)).cpu()
            for i in range(0, len(Yva), 1 << 17)]).numpy()

    pos = Yva.numpy() == 1
    neg = ~pos
    r = rankdata_avg(p_va)
    auroc = float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2)
                  / (pos.sum() * neg.sum()))

    # pairwise accuracy within query: positive vs hard negative pairs
    wins = tot = 0
    va_ious = va["ious"]
    for qid in np.unique(va["qi"]):
        m = va["qi"] == qid
        q_pos = np.where(m & pos)[0]
        q_hn = np.where(m & (va_ious < HN_IOU))[0]
        if len(q_pos) == 0 or len(q_hn) == 0:
            continue
        for a in q_pos:
            for b in q_hn:
                tot += 1
                wins += float(p_va[a] > p_va[b])
    pairwise = wins / max(tot, 1)

    torch.save({"state": probe.state_dict(), "auroc": auroc,
                "pairwise": pairwise, "pairs": tot}, PROBE_PT)
    print(f"Probe [F1,q]->pos vs HN: AUROC={auroc:.4f}  "
          f"pairwise={pairwise:.4f} ({tot} pairs)")
    return {"auroc": auroc, "pairwise": pairwise, "pairs": tot}


# ---------------------------------------------------------------------------
# stage: competition margins
# ---------------------------------------------------------------------------

def run_analyze():
    rk = np.load(VAL_RANK_NPZ)
    va = np.load(VAL_NPZ)
    probe = torch.load(PROBE_PT, map_location="cpu", weights_only=False)
    dev = "cuda:0"
    probe_net = torch.nn.Sequential(
        torch.nn.Linear(768, 128), torch.nn.GELU(),
        torch.nn.Linear(128, 1)).to(dev)
    probe_net.load_state_dict(probe["state"])
    probe_net.eval()

    pool = rk["pool"]
    offs = rk["pool_offsets"]
    ious = rk["ious"]
    sf1 = rk["score_F1"]
    n = len(rk["gt_start"])

    # probe score lookup for val rows (qhna val dump rows are a superset
    # of top-50 rows? they ARE the top-50 + positives = exactly top-50)
    q_index = {}
    for qid in np.unique(va["qi"]):
        q_index[int(qid)] = np.where(va["qi"] == qid)[0]

    margins = {"cls": [], "probe": [], "ioureg": []}
    win = {"cls": 0, "probe": 0, "ioureg": 0}
    pairs = 0
    unmatched = 0
    with torch.no_grad():
        for i in range(n):
            sl = slice(offs[i], offs[i + 1])
            p = pool[sl]
            if p.shape[0] == 0:
                continue
            order = np.argsort(-p[:, 1], kind="stable")
            top = order[:TOPK]
            io = ious[sl][order][:TOPK]
            pos_m = io >= POS_IOU
            hn_m = np.zeros(TOPK, dtype=bool)
            hn_rank = np.where(io < HN_IOU)[0]
            hn_rank = hn_rank[hn_rank < HN_RANK]
            hn_m[hn_rank] = True
            if not pos_m.any() or not hn_m.any():
                continue
            # locate these rows in the qhna val dump
            rows_idx = q_index.get(i)
            if rows_idx is None:
                unmatched += 1
                continue
            vp = va["rows"][rows_idx]          # (m, 9) same fp32 pool rows
            # match by (score, start_tok) keys
            key = lambda arr: np.round(arr[:, 1], 6) * 1e6 + np.round(
                arr[:, 7], 3)
            va_keys = {k: j for j, k in enumerate(key(vp))}
            pos_rows = [va_keys.get(round(float(p[t, 1]), 6) * 1e6
                                    + round(float(p[t, 7]), 3))
                        for t in np.where(pos_m)[0]]
            hn_rows = [va_keys.get(round(float(p[t, 1]), 6) * 1e6
                                   + round(float(p[t, 7]), 3))
                       for t in np.where(hn_m)[0]]
            if any(x is None for x in pos_rows) or any(
                    x is None for x in hn_rows):
                unmatched += 1
                continue
            q = torch.tensor(
                va["q384"][i], dtype=torch.float32).repeat(
                    len(pos_rows) + len(hn_rows), 1)
            allrows = pos_rows + hn_rows
            x = torch.cat([torch.tensor(
                va["F1"][allrows], dtype=torch.float32), q], dim=1).to(dev)
            pp = torch.sigmoid(probe_net(x)).squeeze(-1).cpu().numpy()
            p_pos, p_hn = pp[:len(pos_rows)], pp[len(pos_rows):]
            m_cls = float(p[pos_m][:, 1].max() - p[hn_m][:, 1].max())
            m_probe = float(p_pos.max() - p_hn.max())
            m_ioureg = float(
                sf1[sl][order][:TOPK][pos_m].max()
                - sf1[sl][order][:TOPK][hn_m].max())
            margins["cls"].append(m_cls)
            margins["probe"].append(m_probe)
            margins["ioureg"].append(m_ioureg)
            pairs += 1
            for k, m in (("cls", m_cls), ("probe", m_probe),
                         ("ioureg", m_ioureg)):
                win[k] += float(m > 0)

    out = {
        "n_queries_compared": pairs,
        "n_unmatched": unmatched,
        "margins": {k: {
            "mean": float(np.mean(v)), "median": float(np.median(v)),
            "positive_wins": win[k] / max(pairs, 1)}
            for k, v in margins.items()},
        "probe_separability": {"auroc": probe["auroc"],
                               "pairwise": probe["pairwise"],
                               "pairs": probe["pairs"]},
    }
    RESULTS_JSON.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"\n=== competition margins ({pairs} queries) ===")
    for k in ("cls", "probe", "ioureg"):
        m = out["margins"][k]
        print(f"{k:8s} margin mean {m['mean']:+.4f} median "
              f"{m['median']:+.4f}  P(pos ranked above HN) = "
              f"{100*m['positive_wins']:.1f}%")
    print("results ->", RESULTS_JSON)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=[
        "types", "dump-train", "dump-val", "probe", "analyze"])
    ap.add_argument("--limit-videos", type=int, default=None)
    args = ap.parse_args()
    if args.stage == "types":
        run_types()
    elif args.stage == "dump-train":
        run_dump("train", args.limit_videos)
    elif args.stage == "dump-val":
        run_dump("val", args.limit_videos)
    elif args.stage == "probe":
        run_probe()
    else:
        run_analyze()


if __name__ == "__main__":
    main()

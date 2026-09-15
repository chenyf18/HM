#!/usr/bin/env python3
"""Level-Aware Localization Quality Calibration (LQAC) - frozen probe
(HM-LQAC-022).

Stages (frozen A-U-clean backbone; no base-model training):

  dump-train : candidate generation over the TRAIN split; balanced
               IoU-bucket sample WITH fused point features.
  dump-val   : same over VAL (balanced sample + FULL pool scalars +
               official post-NMS results).
  train      : Q-Base / Q-Level quality heads (SmoothL1 vs BCE stability
               comparison) + LC-only per-level logit calibration.
  rescore    : val pass applying trained heads to every pool candidate
               on the fly; stores q_hat with pool scalars.
  analyze    : offline metrics (Gate 1, rescoring, level/duration).

Candidate feature f_i = head-fusion output (the tensor feeding cls/reg
heads) at the candidate's point position. All scalars use the FP32 decode
(A-U-clean-P32 baseline).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "hydra"), str(ROOT / "tools")]

from tools.audit_ranking_bottleneck import (  # noqa: E402
    AU_ROOT, iou_1d, rankdata_avg, corr, soft_nms,
)

AUC_P32_ROOT = ROOT / "experiments/allocator_diagnosis/seed_1/AUC_P32"
LQAC_ROOT = ROOT / "experiments/lqac_probe"
TRAIN_NPZ = LQAC_ROOT / "train_candidates.npz"
VAL_NPZ = LQAC_ROOT / "val_candidates.npz"
VAL_FULL_NPZ = LQAC_ROOT / "val_full_scalars.npz"
VAL_RESCORE_NPZ = LQAC_ROOT / "val_rescore.npz"
HEADS_PT = LQAC_ROOT / "lqac_heads.pt"
RESULTS_JSON = LQAC_ROOT / "lqac_results.json"

IOU_BUCKETS = ((0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01))
LOW_BUCKET_CAP = 40
FEAT_DIM = 384


# ---------------------------------------------------------------------------
# evaluator with candidate / feature capture
# ---------------------------------------------------------------------------

class _FusionRecorder:
    """Wrap model.fusion.forward; record outputs (head fusion = odd calls).

    Keeps the module object (and any installed fp32 precision boundary)
    intact by rebinding only its forward.
    """

    def __init__(self, module):
        self.module = module
        self.outputs = []
        self._original_forward = module.forward

    def install(self):
        rec = self

        def recording_forward(*args, **kwargs):
            out = rec._original_forward(*args, **kwargs)
            rec.outputs.append(out)
            return out

        self.module.forward = recording_forward

    def clear(self):
        self.outputs = []


def make_head(level_aware, feat_dim=FEAT_DIM, hidden=128, n_levels=8,
              emb=8):
    import torch.nn as nn

    class QualityHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.feat_proj = nn.Sequential(
                nn.Linear(feat_dim, hidden), nn.ReLU())
            self.emb = nn.Embedding(n_levels, emb) if level_aware else None
            geo_in = 3 + (emb if level_aware else 0)
            self.geo_proj = nn.Sequential(nn.Linear(geo_in, 32), nn.ReLU())
            self.out = nn.Linear(hidden + 32, 1)

        def forward(self, f, geo, level):
            h = self.feat_proj(f)
            if self.emb is not None:
                geo = torch.cat([geo, self.emb(level)], dim=-1)
            return self.out(
                torch.cat([h, self.geo_proj(geo)], dim=-1)).squeeze(-1)

    return QualityHead()


def geo_from_pool(pool):
    """[off_l, off_r, log1p(token duration)] - shared by train & rescore."""
    return np.stack([
        pool[:, 5], pool[:, 6],
        np.log1p(np.maximum(pool[:, 8] - pool[:, 7], 0.0)),
    ], axis=1).astype(np.float32)


def build_lqac_evaluator(split, limit_videos=None, sample=True, heads=None):
    import tools.run_formal_ablation as rfa
    from libs import load_opt

    LQAC_ROOT.mkdir(parents=True, exist_ok=True)
    opt_path = LQAC_ROOT / f"opt_{split}.yaml"
    if not opt_path.exists():
        shutil.copyfile(AUC_P32_ROOT / "opt.yaml", opt_path)
    opt = load_opt(str(opt_path), is_training=False)
    opt["eval"]["data"]["split"] = split
    opt["_root"] = str(LQAC_ROOT)
    opt["_ckpt"] = "last"
    models_link = LQAC_ROOT / "models"
    if not models_link.exists():
        models_link.symlink_to(AU_ROOT / "models", target_is_directory=True)

    head_base = head_lvl = None
    if heads is not None:
        head_base = make_head(False).cuda().eval()
        head_base.load_state_dict(heads["q_base"])
        head_lvl = make_head(True).cuda().eval()
        head_lvl.load_state_dict(heads["q_level"])

    class LqacEvaluator(rfa.FormalEvaluator):
        def __init__(self, opt, precision_policy="p2"):
            self.records = []
            self._collect_log = []
            self._fusion_rec = None
            self.rng = np.random.default_rng(1234)
            super().__init__(opt, precision_policy=precision_policy)
            original_collect = self._collect_segments
            self._fusion_rec = _FusionRecorder(self.model.fusion)
            self._fusion_rec.install()

            def collect_wrapper(fpn_points, fpn_logits, fpn_offsets,
                                fpn_masks, ext_scores=None,
                                return_levels=False, fpn_feats=None,
                                query_idx=0):
                result = original_collect(
                    fpn_points, fpn_logits, fpn_offsets, fpn_masks,
                    ext_scores, return_levels=return_levels,
                    fpn_feats=fpn_feats, query_idx=query_idx,
                )
                per_level = []
                for level in range(len(fpn_logits)):
                    points = fpn_points[level]
                    if points.ndim == 3:
                        points = points[0]
                    logits = fpn_logits[level]
                    while logits.ndim > 1:
                        logits = logits[0]
                    offsets = fpn_offsets[level]
                    while offsets.ndim > 2:
                        offsets = offsets[0]
                    mask = fpn_masks[level]
                    while mask.ndim > 1:
                        mask = mask[0]
                    scores = torch.sigmoid(logits) * mask.float()
                    keep = scores > self.pre_nms_thresh
                    per_level.append({
                        "level": level,
                        "pos": keep.nonzero(as_tuple=False).flatten(),
                        "logit": logits[keep].detach().float().cpu(),
                        "score": scores[keep].detach().float().cpu(),
                        "center": points[:, 0][keep].detach().float().cpu(),
                        "scale": points[:, 3][keep].detach().float().cpu(),
                        "offset": offsets[keep].detach().float().cpu(),
                    })
                self._collect_log.append(per_level)
                return result

            self._collect_segments = collect_wrapper

        def predict(self, data):
            tokens = data["text"]
            if not isinstance(tokens, tuple):
                tokens = (tokens, )
            vid_len = data["vid"].size(-1)
            window_size = min(self.window_size or vid_len, vid_len)
            window_stride = self.window_stride or window_size
            n_win = vid_len - window_size
            offsets_w = []
            idx = 0
            while idx <= n_win:
                offsets_w.append(idx)
                idx += window_stride
            if n_win > 0 and n_win % window_stride > 0:
                offsets_w.append(n_win)

            self._collect_log = []
            self._fusion_rec.clear()
            results = super().predict(data)
            n_q = len(results)
            n_w = len(offsets_w)
            if len(self._collect_log) != n_w * n_q:
                raise RuntimeError("collect calls != windows x queries")
            if len(self._fusion_rec.outputs) != 2 * n_w:
                raise RuntimeError("fusion calls != 2 x windows")
            segments = data["segment"]
            for q in range(n_q):
                merged = None
                feats_rows = []
                for w in range(n_w):
                    per_level = self._collect_log[w * n_q + q]
                    fused = self._fusion_rec.outputs[2 * w + 1][0]
                    for pl in per_level:
                        n_k = pl["score"].numel()
                        entry = np.empty((n_k, 9), dtype=np.float64)
                        entry[:, 0] = pl["logit"].numpy()
                        entry[:, 1] = pl["score"].numpy()
                        entry[:, 2] = pl["level"]
                        entry[:, 3] = pl["center"].numpy()
                        entry[:, 4] = pl["scale"].numpy()
                        entry[:, 5] = pl["offset"][:, 0].numpy()
                        entry[:, 6] = pl["offset"][:, 1].numpy()
                        centers = entry[:, 3] + offsets_w[w]
                        entry[:, 7] = centers - entry[:, 5] * entry[:, 4]
                        entry[:, 8] = centers + entry[:, 6] * entry[:, 4]
                        merged = (entry if merged is None
                                  else np.concatenate([merged, entry]))
                        lvl = int(pl["level"])
                        f_q = fused[lvl][q][:, pl["pos"].to(fused[lvl].device)]
                        feats_rows.append(
                            f_q.transpose(0, 1).detach().half().cpu().numpy())
                if merged is None:
                    merged = np.zeros((0, 9), dtype=np.float64)
                feats = (np.concatenate(feats_rows, axis=0) if feats_rows
                         else np.zeros((0, FEAT_DIM), np.float16))
                gt = np.asarray(segments[q], dtype=np.float64)
                rec = {
                    "vid_id": str(data["vid_id"]),
                    "query_idx": q,
                    "gt_start": float(gt[0]),
                    "gt_end": float(gt[1]),
                    "clip_stride": float(data["clip_stride"]),
                    "clip_size": float(data["clip_size"]),
                    "fps": float(data["fps"]),
                    "duration": float(data["duration"]),
                    "pool": merged,
                    "official_segments": results[q][
                        "segments"].detach().cpu().numpy().astype(
                            np.float64),
                    "official_scores": results[q][
                        "scores"].detach().cpu().numpy().astype(np.float64),
                }
                ious = _iou_seconds(rec, merged) if merged.shape[0] \
                    else np.zeros(0)
                if head_base is not None and merged.shape[0]:
                    with torch.no_grad():
                        f = torch.tensor(
                            feats, dtype=torch.float32).cuda()
                        g = torch.tensor(
                            geo_from_pool(merged)).cuda()
                        lv = torch.tensor(
                            merged[:, 2], dtype=torch.long).cuda()
                        rec["q_base"] = torch.sigmoid(
                            head_base(f, g, lv)).cpu().numpy()
                        rec["q_level"] = torch.sigmoid(
                            head_lvl(f, g, lv)).cpu().numpy()
                if sample:
                    sel = _balanced_select(self.rng, ious)
                    rec["sample_pool"] = merged[sel].astype(np.float32)
                    rec["sample_ious"] = ious[sel].astype(np.float32)
                    rec["sample_feats"] = feats[sel]
                self.records.append(rec)
            return results

    ev = LqacEvaluator(opt, precision_policy="p3")
    if limit_videos:
        ev.dataloader = list(itertools.islice(
            iter(ev.dataloader), limit_videos))
        ev.num_itrs = len(ev.dataloader)
    return ev


def _iou_seconds(rec, pool):
    cs, cc, fps, dur = (rec["clip_stride"], rec["clip_size"],
                        rec["fps"], rec["duration"])
    s = np.clip((pool[:, 7] * cs + 0.5 * cc) / fps, 0, dur)
    e = np.clip((pool[:, 8] * cs + 0.5 * cc) / fps, 0, dur)
    return iou_1d(s, e, rec["gt_start"], rec["gt_end"])


def _bucket_id(v):
    if v < 0.1:
        return 0
    if v < 0.3:
        return 1
    if v < 0.5:
        return 2
    if v < 0.7:
        return 3
    return 4


def _balanced_select(rng, ious):
    bucket_ids = np.array([_bucket_id(v) for v in ious])
    sel = []
    for b in range(len(IOU_BUCKETS)):
        idxs = np.where(bucket_ids == b)[0]
        if b == 0 and len(idxs) > LOW_BUCKET_CAP:
            idxs = rng.choice(idxs, LOW_BUCKET_CAP, replace=False)
        sel.append(idxs)
    sel = np.concatenate(sel)
    return np.sort(sel)


def run_dump(split, limit_videos=None):
    ev = build_lqac_evaluator(split, limit_videos=limit_videos)
    ev.run()
    rows = np.concatenate([r["sample_pool"] for r in ev.records])
    ious = np.concatenate([r["sample_ious"] for r in ev.records])
    feats = np.concatenate([r["sample_feats"] for r in ev.records])
    out = TRAIN_NPZ if split == "train" else VAL_NPZ
    np.savez_compressed(out, rows=rows, ious=ious, feats=feats)
    hist = np.histogram(ious, bins=[0, .1, .3, .5, .7, 1.01])[0]
    print(f"[{split}] sampled {len(rows)} candidates  hist={hist.tolist()}")
    print("saved ->", out)
    if split != "train":
        _save_full(ev.records, VAL_FULL_NPZ)


def _save_full(records, path, extra_keys=()):
    n = len(records)
    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum([r["pool"].shape[0] for r in records], out=offsets[1:])
    arrays = {
        "pool": np.concatenate([r["pool"] for r in records], axis=0),
        "pool_offsets": offsets,
        "ious": np.concatenate([
            (_iou_seconds(r, r["pool"]) if r["pool"].shape[0]
             else np.zeros(0)) for r in records]),
    }
    for k in ("gt_start", "gt_end", "clip_stride", "clip_size", "fps",
              "duration"):
        arrays[k] = np.array([r[k] for r in records], dtype=np.float64)
    max_off = max(len(r["official_segments"]) for r in records)
    seg = np.full((n, max_off, 2), np.nan)
    sc = np.full((n, max_off), np.nan)
    for i, r in enumerate(records):
        k = len(r["official_segments"])
        if k:
            seg[i, :k] = r["official_segments"]
            sc[i, :k] = r["official_scores"]
    arrays["official_segments"] = seg
    arrays["official_scores"] = sc
    arrays["vid_ids"] = np.array([r["vid_id"] for r in records])
    arrays["query_idx"] = np.array(
        [r["query_idx"] for r in records], dtype=np.int64)
    for key in extra_keys:
        arrays[key] = np.concatenate(
            [r[key] if key in r else np.zeros(r["pool"].shape[0])
             for r in records])
    np.savez_compressed(path, **arrays)
    print(f"full pool scalars ({int(offsets[-1])} candidates) -> {path}")


def run_rescore(limit_videos=None):
    heads = torch.load(HEADS_PT, map_location="cpu", weights_only=False)
    ev = build_lqac_evaluator(
        "val", limit_videos=limit_videos, sample=False, heads=heads)
    ev.run()
    _save_full(ev.records, VAL_RESCORE_NPZ,
               extra_keys=("q_base", "q_level"))


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def _spearman_chunks(q, y, chunk=50000):
    vals = []
    for s in np.array_split(np.arange(len(q)), max(1, len(q) // chunk)):
        if len(s) < 3:
            continue
        v = corr(rankdata_avg(q[s]), rankdata_avg(y[s]))
        if not math.isnan(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def train_heads():
    tr = np.load(TRAIN_NPZ, allow_pickle=False)
    va = np.load(VAL_NPZ, allow_pickle=False)
    dev = "cuda:0"

    def prep(d):
        return (
            torch.tensor(d["feats"], dtype=torch.float32),
            torch.tensor(geo_from_pool(d["rows"])),
            torch.tensor(d["rows"][:, 2], dtype=torch.long),
            torch.tensor(d["ious"], dtype=torch.float32),
        )

    ftr, gtr, ltr, ytr = prep(tr)
    fva, gva, lva, yva = prep(va)
    print(f"train candidates {len(ytr)}, val candidates {len(yva)}")

    def clone_state(head):
        return {k: v.detach().cpu().clone()
                for k, v in head.state_dict().items()}

    def val_spearman(head):
        head.eval()
        qs = []
        with torch.no_grad():
            for i in range(0, len(yva), 1 << 17):
                qs.append(torch.sigmoid(head(
                    fva[i:i + (1 << 17)].to(dev),
                    gva[i:i + (1 << 17)].to(dev),
                    lva[i:i + (1 << 17)].to(dev))).cpu())
        qv = torch.cat(qs).numpy()
        return _spearman_chunks(qv, yva.numpy()), float(
            np.abs(qv - yva.numpy()).mean())

    def train_one(level_aware, loss_name, epochs, lr=1e-3, bs=8192,
                  report=False):
        torch.manual_seed(7)
        head = make_head(level_aware).to(dev)
        opt = torch.optim.Adam(head.parameters(), lr=lr)
        n = len(ytr)
        best = {"sp": -2.0, "state": None, "epoch": -1, "mae": float("nan")}
        curve = []
        for ep in range(epochs):
            head.train()
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                q = torch.sigmoid(head(
                    ftr[idx].to(dev), gtr[idx].to(dev), ltr[idx].to(dev)))
                y = ytr[idx].to(dev)
                if loss_name == "smoothl1":
                    loss = torch.nn.functional.smooth_l1_loss(
                        q, y, beta=0.1)
                else:
                    loss = torch.nn.functional.binary_cross_entropy(
                        q, y.clamp(1e-4, 1 - 1e-4))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            sp, mae = val_spearman(head)
            curve.append({"epoch": ep, "val_spearman": sp, "val_mae": mae})
            if sp > best["sp"]:
                best = {"sp": sp, "epoch": ep,
                        "state": clone_state(head), "mae": mae}
            if report:
                print(f"  ep{ep}: val_spearman={sp:.4f} val_mae={mae:.4f}")
        return head, best, curve

    print("== loss comparison (4 epochs, Q-Base) ==")
    comparison = {}
    for loss_name in ("smoothl1", "bce"):
        _, best, curve = train_one(False, loss_name, 4, report=True)
        comparison[loss_name] = {
            "best_val_spearman": best["sp"],
            "best_val_mae": best["mae"], "curve": curve}
    chosen = max(comparison,
                 key=lambda k: comparison[k]["best_val_spearman"])
    print(f"chosen loss: {chosen}")

    print("== training Q-Base ==")
    head_base, best_base, curve_base = train_one(
        False, chosen, 20, report=True)
    print("== training Q-Level ==")
    head_lvl, best_lvl, curve_lvl = train_one(
        True, chosen, 20, report=True)
    head_base.load_state_dict(best_base["state"])
    head_lvl.load_state_dict(best_lvl["state"])

    print("== training LC-only (per-level affine, BCE vs IoU) ==")
    lt = torch.tensor(tr["rows"][:, 0], dtype=torch.float64)
    lv_t = torch.tensor(tr["rows"][:, 2].astype(int))
    yt = torch.tensor(tr["ious"], dtype=torch.float64)
    a = torch.ones(8, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(8, dtype=torch.float64, requires_grad=True)
    opt_lc = torch.optim.Adam([a, b], lr=0.05)
    for _ in range(1500):
        cal = torch.sigmoid((a[lv_t] * lt + b[lv_t]).clamp(-30, 30))
        loss = torch.nn.functional.binary_cross_entropy(
            cal.double(), yt.clamp(0.0, 1.0))
        opt_lc.zero_grad()
        loss.backward()
        opt_lc.step()
    lc = {"a": a.detach().numpy().tolist(), "b": b.detach().numpy().tolist()}
    print("LC a:", [round(x, 3) for x in lc["a"]],
          "b:", [round(x, 3) for x in lc["b"]])

    torch.save({
        "q_base": head_base.state_dict(),
        "q_level": head_lvl.state_dict(),
        "lc": lc,
        "chosen_loss": chosen,
        "loss_comparison": comparison,
        "best_base": {"val_spearman": best_base["sp"],
                      "val_mae": best_base["mae"],
                      "epoch": best_base["epoch"]},
        "best_lvl": {"val_spearman": best_lvl["sp"],
                     "val_mae": best_lvl["mae"],
                     "epoch": best_lvl["epoch"]},
        "curve_base": curve_base,
        "curve_lvl": curve_lvl,
    }, HEADS_PT)
    print("heads saved ->", HEADS_PT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=[
        "dump-train", "dump-val", "train", "rescore", "analyze"])
    ap.add_argument("--limit-videos", type=int, default=None)
    args = ap.parse_args()
    if args.stage == "dump-train":
        run_dump("train", args.limit_videos)
    elif args.stage == "dump-val":
        run_dump("val", args.limit_videos)
    elif args.stage == "train":
        train_heads()
    elif args.stage == "rescore":
        run_rescore(args.limit_videos)
    else:
        from tools.lqac_analyze import run_analyze
        run_analyze()


if __name__ == "__main__":
    main()

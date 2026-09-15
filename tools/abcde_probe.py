#!/usr/bin/env python3
"""HM-ABCDE-035: 5-level feature probe for query-conditioned localization
evidence.

Captures A(encoder)/B(fused)/C(cls hidden)/D(reg hidden)/E(C+D concat)
for the cls-score top-K candidates of each train/val query, then trains
identical probes (linear + MLP) on each feature type for three tasks:
  1) IoU regression (SmoothL1)
  2) pos vs hard-neg binary classification (pos: IoU>=.5, HN: IoU<.3)
  3) query-internal ranking (Spearman on val)

Frozen R1 backbone; no model/loss modification.
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "hydra"), str(ROOT / "tools")]

from tools.audit_ranking_bottleneck import iou_1d, rankdata_avg, corr  # noqa
from libs.modeling.temporal_coordinates import decode_offsets  # noqa

R1_OPT = ROOT / "experiments/allocator_diagnosis/seed_1/AUC/opt.yaml"
R1_MODELS = ROOT / "experiments/allocator_diagnosis/seed_1/AUC/models"
PROOT = ROOT / "experiments/abcde_probe"
TRAIN_NPZ = PROOT / "train_abcde.npz"
VAL_NPZ = PROOT / "val_abcde.npz"
RESULTS = PROOT / "abcde_results.json"
TOPK = 50
FEAT = 384

# ---------------------------------------------------------------------------
# dump evaluator
# ---------------------------------------------------------------------------

def build_dump(split, limit_videos=None):
    import tools.run_formal_ablation as rfa
    from libs import load_opt
    from tools.lqac_probe import _FusionRecorder
    from tools.anchor_audit import _EncodeRecorder

    work = PROOT / f"backbone_{split}"
    work.mkdir(parents=True, exist_ok=True)
    ml = work / "models"
    if ml.is_symlink() or ml.exists():
        ml.unlink()
    ml.symlink_to(R1_MODELS, target_is_directory=True)
    op = work / f"opt_{split}.yaml"
    if op.exists():
        op.unlink()
    shutil.copyfile(R1_OPT, op)
    opt = load_opt(str(op), is_training=False)
    opt["eval"]["data"]["split"] = split
    opt["_root"] = str(work)
    opt["_ckpt"] = "last"

    class ABCDEDump(rfa.FormalEvaluator):
        def __init__(self, opt):
            self.recs = []
            self._collect_log = []
            self._fusion_rec = None
            self._enc_rec = None
            self._cls_hidden = {}     # level -> (B, C, T)
            self._reg_hidden = {}
            super().__init__(opt, precision_policy="p3")
            m = self.model
            self._enc_rec = _EncodeRecorder(m)
            self._enc_rec.install()
            self._fusion_rec = _FusionRecorder(m.fusion)
            self._fusion_rec.install()
            # hook cls/reg head final conv inputs
            for lvl in range(8):
                cls_final = m.cls_head.cls_head
                reg_final = m.reg_head.reg_head
                orig_cls = cls_final.forward
                orig_reg = reg_final.forward

                def mk_cls(orig, level):
                    def rec(x, mask):
                        self._cls_hidden[level] = x.detach()
                        return orig(x, mask)
                    return rec

                def mk_reg(orig, level):
                    def rec(x, mask):
                        self._reg_hidden[level] = x.detach()
                        return orig(x, mask)
                    return rec

                # heads are shared across levels; hook once (overwrites
                # per level; store by last call = current level)
                cls_final.forward = mk_cls(orig_cls, lvl)
                reg_final.forward = mk_reg(orig_reg, lvl)
            # simpler: single hook on the final convs (called per level
            # inside head forward loop, so we store a list)
            self._cls_hidden_list = []
            self._reg_hidden_list = []
            cf = m.cls_head.cls_head
            rf = m.reg_head.reg_head
            orig_cf = cf.forward
            orig_rf = rf.forward

            def cf_rec(x, mask):
                self._cls_hidden_list.append(x.detach())
                return orig_cf(x, mask)

            def rf_rec(x, mask):
                self._reg_hidden_list.append(x.detach())
                return orig_rf(x, mask)

            cf.forward = cf_rec
            rf.forward = rf_rec
            # also need original collect wrapper
            orig_collect = self._collect_segments

            def wrap(fpn_points, fpn_logits, fpn_offsets, fpn_masks,
                     ext_scores=None, return_levels=False,
                     fpn_feats=None, query_idx=0):
                res = orig_collect(
                    fpn_points, fpn_logits, fpn_offsets, fpn_masks,
                    ext_scores, return_levels=return_levels,
                    fpn_feats=fpn_feats, query_idx=query_idx)
                per_level = []
                for level in range(len(fpn_logits)):
                    pts = fpn_points[level]
                    if pts.ndim == 3:
                        pts = pts[0]
                    lg = fpn_logits[level]
                    while lg.ndim > 1:
                        lg = lg[0]
                    of = fpn_offsets[level]
                    while of.ndim > 2:
                        of = of[0]
                    mk = fpn_masks[level]
                    while mk.ndim > 1:
                        mk = mk[0]
                    sc = torch.sigmoid(lg) * mk.float()
                    keep = sc > self.pre_nms_thresh
                    per_level.append({
                        "level": level,
                        "pos": keep.nonzero(as_tuple=False).flatten(),
                        "logit": lg[keep].detach().float().cpu(),
                        "score": sc[keep].detach().float().cpu(),
                        "points": pts[keep].detach().float().cpu(),
                        "offset": of[keep].detach().float().cpu(),
                    })
                self._collect_log.append(per_level)
                return res

            self._collect_segments = wrap
            orig_t2 = self.model.encode_text2

            def t2_rec(t, tm, ts):
                out = orig_t2(t, tm, ts)
                self._text_pool = (out[0].detach(), out[1].detach())
                return out

            self.model.encode_text2 = t2_rec

        def _q384(self):
            t, m = self._text_pool
            with torch.no_grad():
                l0 = self.model.fusion.layers[0]
                v = l0.xattn.xattn.value(l0.ln_xattn_kv(t.float()))
                mf = m.float()
                q = (v * mf).sum(-1) / mf.sum(-1).clamp_min(1.0)
                return F.normalize(q, dim=-1)

        def predict(self, data):
            tokens = data["text"]
            if not isinstance(tokens, tuple):
                tokens = (tokens, )
            self._collect_log = []
            self._fusion_rec.clear()
            self._enc_rec.clear()
            self._cls_hidden_list = []
            self._reg_hidden_list = []
            results = super().predict(data)
            n_q = len(results)
            f0 = self._enc_rec.outputs[0][0]    # (Q,C,T_l) per level
            f1 = self._fusion_rec.outputs[1][0]  # fused
            cls_h = self._cls_hidden_list        # [level][ (B,C,T) ]
            reg_h = self._reg_hidden_list
            q384 = self._q384().cpu().numpy()
            for q in range(n_q):
                per_level = self._collect_log[q]
                _parts = {k: [] for k in ("A", "B", "C", "D")}
                all_score, all_logit, all_level = [], [], []
                all_center, all_off, all_scale = [], [], []
                for li, pl in enumerate(per_level):
                    n_k = pl["score"].numel()
                    if n_k == 0:
                        continue
                    all_score.append(pl["score"])
                    all_logit.append(pl["logit"])
                    all_level.append(torch.full((n_k,), pl["level"],
                                                dtype=torch.long))
                    all_center.append(pl["points"][:, 0])
                    all_scale.append(pl["points"][:, 3])
                    all_off.append(torch.stack(
                        [pl["offset"][:, 0], pl["offset"][:, 1]], 1))
                    lvl = pl["level"]
                    pos = pl["pos"].to(f0[lvl].device)
                    _parts["A"].append(
                        f0[lvl][q][:, pos].T.detach().float().cpu())
                    _parts["B"].append(
                        f1[lvl][q][:, pos].T.detach().float().cpu())
                    _parts["C"].append(
                        cls_h[lvl][q][:, pos].T.detach().float().cpu())
                    _parts["D"].append(
                        reg_h[lvl][q][:, pos].T.detach().float().cpu())
                if not all_score:
                    continue
                scores = torch.cat(all_score)
                order = torch.argsort(-scores, stable=True)[:TOPK]
                k = min(TOPK, len(order))
                gt = np.asarray(data["segment"][q], dtype=np.float64)
                rec = {
                    "gt_s": float(gt[0]), "gt_e": float(gt[1]),
                    "q384": q384[q].astype(np.float32),
                    "score": scores[order[:k]].numpy(),
                    "logit": torch.cat(all_logit)[order[:k]].numpy(),
                    "level": torch.cat(all_level)[order[:k]].numpy(),
                    "center": torch.cat(all_center)[order[:k]].numpy(),
                    "scale": torch.cat(all_scale)[order[:k]].numpy(),
                    "off": torch.cat(all_off)[order[:k]].numpy(),
                    "clip_stride": float(data["clip_stride"]),
                    "clip_size": float(data["clip_size"]),
                    "fps": float(data["fps"]),
                    "duration": float(data["duration"]),
                }
                for key in ("A", "B", "C", "D"):
                    feat = torch.cat(_parts[key], 0)[order[:k]]
                    rec[key] = feat.numpy().astype(np.float16)
                self.recs.append(rec)
            return results

    ev = ABCDEDump(opt)
    if limit_videos:
        ev.dataloader = list(itertools.islice(
            iter(ev.dataloader), limit_videos))
        ev.num_itrs = len(ev.dataloader)
    return ev


def run_dump(split, limit_videos=None):
    ev = build_dump(split, limit_videos)
    ev.run()
    recs = ev.recs
    n = len(recs)
    arrays = {}
    # per-feature arrays
    for key in ("A", "B", "C", "D"):
        mat = np.zeros((n, TOPK, FEAT), np.float16)
        for i, r in enumerate(recs):
            k = r[key].shape[0]
            if k:
                mat[i, :k] = r[key][:TOPK]
        arrays[key] = mat
    for key in ("score", "logit", "center", "scale"):
        mat = np.zeros((n, TOPK), np.float32)
        for i, r in enumerate(recs):
            k = len(r[key])
            if k:
                mat[i, :k] = r[key][:TOPK]
        arrays[key] = mat
    arrays["level"] = np.zeros((n, TOPK), np.int64)
    for i, r in enumerate(recs):
        k = len(r["level"])
        if k:
            arrays["level"][i, :k] = r["level"][:TOPK]
    arrays["off"] = np.zeros((n, TOPK, 2), np.float32)
    for i, r in enumerate(recs):
        k = r["off"].shape[0]
        if k:
            arrays["off"][i, :k] = r["off"][:TOPK]
    arrays["q384"] = np.stack([r["q384"] for r in recs])
    for key in ("gt_s", "gt_e", "clip_stride", "clip_size", "fps",
                "duration"):
        arrays[key] = np.array([r[key] for r in recs], np.float64)
    arrays["n_valid"] = np.array(
        [min(len(r["score"]), TOPK) for r in recs], np.int64)
    out = TRAIN_NPZ if split == "train" else VAL_NPZ
    PROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    print(f"[{split}] {n} queries -> {out}")


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------

def decode_iou(arrays):
    n = len(arrays["gt_s"])
    ious = np.zeros((n, TOPK), np.float32)
    for i in range(n):
        k = int(arrays["n_valid"][i])
        if k == 0:
            continue
        c = arrays["center"][i, :k]
        sc = arrays["scale"][i, :k]
        s_tok = c - arrays["off"][i, :k, 0] * sc
        e_tok = c + arrays["off"][i, :k, 1] * sc
        cs, cc = arrays["clip_stride"][i], arrays["clip_size"][i]
        fps, dur = arrays["fps"][i], arrays["duration"][i]
        ss = np.clip((s_tok * cs + 0.5 * cc) / fps, 0, dur)
        ee = np.clip((e_tok * cs + 0.5 * cc) / fps, 0, dur)
        ious[i, :k] = iou_1d(ss, ee, arrays["gt_s"][i], arrays["gt_e"][i])
    return ious


def run_probes():
    tr = np.load(TRAIN_NPZ)
    va = np.load(VAL_NPZ)
    dev = "cuda:0"
    tr_iou = decode_iou(tr)
    va_iou = decode_iou(va)

    # build labels: pos(HN for task 2), all for task 1 & 3
    feat_keys = ("A", "B", "C", "D", "E")
    results = {}

    for fk in feat_keys:
        if fk == "E":
            tr_X = np.concatenate(
                [tr["C"].astype(np.float32),
                 tr["D"].astype(np.float32)], axis=-1)
            va_X = np.concatenate(
                [va["C"].astype(np.float32),
                 va["D"].astype(np.float32)], axis=-1)
            dim = FEAT * 2
        else:
            tr_X = tr[fk].astype(np.float32)
            va_X = va[fk].astype(np.float32)
            dim = FEAT

        # flatten valid rows
        tr_rows, tr_iou_f, tr_qi, tr_nv = [], [], [], []
        for i in range(len(tr["gt_s"])):
            k = int(tr["n_valid"][i])
            tr_rows.append(tr_X[i, :k])
            tr_iou_f.append(tr_iou[i, :k])
            tr_qi.append(np.full(k, i))
            tr_nv.append(k)
        tr_rows = np.concatenate(tr_rows)
        tr_iou_f = np.concatenate(tr_iou_f)
        tr_qi = np.concatenate(tr_qi)

        va_rows, va_iou_f, va_qi, va_nv = [], [], [], []
        for i in range(len(va["gt_s"])):
            k = int(va["n_valid"][i])
            va_rows.append(va_X[i, :k])
            va_iou_f.append(va_iou[i, :k])
            va_qi.append(np.full(k, i))
            va_nv.append(k)
        va_rows = np.concatenate(va_rows)
        va_iou_f = np.concatenate(va_iou_f)
        va_qi = np.concatenate(va_qi)

        Xtr = torch.tensor(tr_rows, device=dev)
        Ytr_iou = torch.tensor(tr_iou_f, device=dev)
        Xva = torch.tensor(va_rows, device=dev)
        Yva_iou = torch.tensor(va_iou_f, device=dev)

        # === task 1: IoU regression (MLP) ===
        torch.manual_seed(7)
        mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, 128), torch.nn.GELU(),
            torch.nn.Linear(128, 1)).to(dev)
        opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
        n1 = len(Ytr_iou)
        for ep in range(8):
            perm = torch.randperm(n1, device=dev)
            for i0 in range(0, n1, 8192):
                idx = perm[i0:i0 + 8192]
                pred = mlp(Xtr[idx]).squeeze(-1)
                loss = F.smooth_l1_loss(pred, Ytr_iou[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        with torch.no_grad():
            va_pred = torch.cat([
                mlp(Xva[i:i + 65536]).squeeze(-1)
                for i in range(0, len(Yva_iou), 65536)]).cpu().numpy()

        # Spearman per query (task 3)
        sp_qs = []
        offset = 0
        for i, k in enumerate(va_nv):
            if k < 3:
                offset += k
                continue
            p = va_pred[offset:offset + k]
            io = va_iou[i, :k]
            v = corr(rankdata_avg(p), rankdata_avg(io))
            if not math.isnan(v):
                sp_qs.append(v)
            offset += k
        spearman = float(np.mean(sp_qs)) if sp_qs else float("nan")

        # === task 2: pos vs HN binary ===
        pos_m = va_iou_f >= 0.5
        hn_m = va_iou_f < 0.3
        X_pos, X_hn = Xva[pos_m], Xva[hn_m]
        y_pos = torch.ones(len(X_pos), device=dev)
        y_hn = torch.zeros(len(X_hn), device=dev)
        # train binary
        tr_pos_m = tr_iou_f >= 0.5
        tr_hn_m = tr_iou_f < 0.3
        Xtr_b = torch.cat([Xtr[tr_pos_m], Xtr[tr_hn_m]])
        ytr_b = torch.cat([
            torch.ones(int(tr_pos_m.sum()), device=dev),
            torch.zeros(int(tr_hn_m.sum()), device=dev)])
        torch.manual_seed(7)
        clf = torch.nn.Sequential(
            torch.nn.Linear(dim, 128), torch.nn.GELU(),
            torch.nn.Linear(128, 1)).to(dev)
        opt2 = torch.optim.Adam(clf.parameters(), lr=1e-3,
                                weight_decay=1e-4)
        n2 = len(ytr_b)
        for ep in range(8):
            perm = torch.randperm(n2, device=dev)
            for i0 in range(0, n2, 8192):
                idx = perm[i0:i0 + 8192]
                loss = F.binary_cross_entropy_with_logits(
                    clf(Xtr_b[idx]).squeeze(-1), ytr_b[idx])
                opt2.zero_grad(set_to_none=True)
                loss.backward()
                opt2.step()
        with torch.no_grad():
            p_pos = torch.sigmoid(clf(X_pos)).squeeze(-1).cpu().numpy()
            p_hn = torch.sigmoid(clf(X_hn)).squeeze(-1).cpu().numpy()
        from tools.audit_ranking_bottleneck import rankdata_avg as _ra
        r_all = _ra(np.concatenate([p_pos, p_hn]))
        r_pos = r_all[:len(p_pos)]
        npos = len(p_pos)
        auroc = float((r_pos.sum() - npos * (npos + 1) / 2)
                      / (npos * len(p_hn))) if len(p_hn) > 0 else float("nan")

        results[fk] = {
            "task1_spearman": spearman,
            "task1_mae": float(np.abs(va_pred - va_iou_f).mean()),
            "task2_auroc_pos_vs_hn": auroc,
            "n_val_rows": len(Yva_iou),
        }
        print(f"[{fk}] spearman={spearman:.4f} mae={results[fk]['task1_mae']:.4f} "
              f"auroc={auroc:.4f}")

    RESULTS.write_text(json.dumps(results, indent=2) + "\n")
    print("->", RESULTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["dump-train", "dump-val", "probes"])
    ap.add_argument("--limit-videos", type=int, default=None)
    args = ap.parse_args()
    if args.stage == "dump-train":
        run_dump("train", args.limit_videos)
    elif args.stage == "dump-val":
        run_dump("val", args.limit_videos)
    else:
        run_probes()


if __name__ == "__main__":
    main()

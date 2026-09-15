#!/usr/bin/env python3
"""HM-CSRM-034: two-stage candidate-set ranking pipeline.

Stage 2: freeze the R1 backbone, train ONLY the CandidateSetRankingModule
on top-K candidates. At inference, CSRM scores replace cls scores for the
top-K subset (rest keep original scores); NMS parameters unchanged.

Modes:
  dump-train : dump top-K candidates + F1 features + query per train query
  dump-val   : same over val
  train      : train CSRM on the train dump (pairwise margin loss with
               structured hard negatives)
  eval       : full val eval with CSRM re-scoring (official pipeline)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "hydra"), str(ROOT / "tools")]

from tools.audit_ranking_bottleneck import iou_1d, soft_nms  # noqa
from libs.modeling.csrm import CandidateSetRankingModule  # noqa
from libs.modeling.temporal_coordinates import decode_offsets  # noqa

R1_OPT = ROOT / "experiments/allocator_diagnosis/seed_1/AUC/opt.yaml"
R1_MODELS = ROOT / "experiments/allocator_diagnosis/seed_1/AUC/models"
CROOT = ROOT / "experiments/csrm"
TRAIN_NPZ = CROOT / "train_topk.npz"
VAL_NPZ = CROOT / "val_topk.npz"
CKPT = CROOT / "csrm.pt"
RESULTS = CROOT / "csrm_results.json"
TOPK = 50
FEAT_DIM = 384
MARGIN = 0.5
MAX_PAIRS_PER_QUERY = 32
# locked protocol
LR = 1e-4
N_EPOCHS = 10
SEED = 0

# ---------------------------------------------------------------------------
# candidate dump evaluator (frozen R1, top-K by cls, F1 features + query)
# ---------------------------------------------------------------------------

def build_dump_eval(split, limit_videos=None):
    import shutil
    import tools.run_formal_ablation as rfa
    from libs import load_opt
    from tools.lqac_probe import _FusionRecorder

    work = CROOT / f"backbone_{split}"
    work.mkdir(parents=True, exist_ok=True)
    models_link = work / "models"
    if models_link.is_symlink() or models_link.exists():
        models_link.unlink()
    models_link.symlink_to(R1_MODELS, target_is_directory=True)
    opt_path = work / f"opt_{split}.yaml"
    if opt_path.exists():
        opt_path.unlink()
    shutil.copyfile(R1_OPT, opt_path)
    opt = load_opt(str(opt_path), is_training=False)
    opt["eval"]["data"]["split"] = split
    opt["_root"] = str(work)
    opt["_ckpt"] = "last"

    class DumpEval(rfa.FormalEvaluator):
        def __init__(self, opt):
            self.recs = []
            self._collect_log = []
            self._fusion_rec = None
            self._text_pool = None
            super().__init__(opt, precision_policy="p3")
            self._fusion_rec = _FusionRecorder(self.model.fusion)
            self._fusion_rec.install()
            l0 = self.model.fusion.layers[0]
            self.proj_ln = l0.ln_xattn_kv
            self.proj_val = l0.xattn.xattn.value
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

            def t2_rec(text, text_masks, text_size):
                out = orig_t2(text, text_masks, text_size)
                self._text_pool = (out[0].detach(), out[1].detach())
                return out

            self.model.encode_text2 = t2_rec

        def _q384(self):
            t, m = self._text_pool
            with torch.no_grad():
                v = self.proj_val(self.proj_ln(t.float()))
                mf = m.float()
                q = (v * mf).sum(-1) / mf.sum(-1).clamp_min(1.0)
                return torch.nn.functional.normalize(q, dim=-1)

        def predict(self, data):
            tokens = data["text"]
            if not isinstance(tokens, tuple):
                tokens = (tokens, )
            self._collect_log = []
            self._fusion_rec.clear()
            results = super().predict(data)
            n_q = len(results)
            if len(self._collect_log) != n_q:
                raise RuntimeError("collect/query mismatch")
            if len(self._fusion_rec.outputs) != 2:
                raise RuntimeError("fusion calls != 2")
            f1, _ = self._fusion_rec.outputs[1]
            q384 = self._q384().cpu().numpy()
            for q in range(n_q):
                per_level = self._collect_log[q]
                _feat_parts = []
                all_score, all_logit, all_level = [], [], []
                all_center, all_off_l, all_off_r, all_scale = \
                    [], [], [], []
                for pl in per_level:
                    n_k = pl["score"].numel()
                    if n_k == 0:
                        continue
                    all_score.append(pl["score"])
                    all_logit.append(pl["logit"])
                    all_level.append(
                        torch.full((n_k,), pl["level"], dtype=torch.long))
                    all_center.append(pl["points"][:, 0])
                    all_scale.append(pl["points"][:, 3])
                    all_off_l.append(pl["offset"][:, 0])
                    all_off_r.append(pl["offset"][:, 1])
                    # F1 gather (accumulate per level in order)
                    lvl = pl["level"]
                    pos = pl["pos"].to(f1[lvl].device)
                    feat_q = f1[lvl][q][:, pos].transpose(0, 1) \
                        .detach().float().cpu()
                    _feat_parts.append(feat_q)
                if not all_score or not _feat_parts:
                    self.recs.append(self._empty_rec(data, q, q384))
                    continue
                feats = torch.cat(_feat_parts, dim=0)
                scores = torch.cat(all_score)
                logits = torch.cat(all_logit)
                levels = torch.cat(all_level)
                centers = torch.cat(all_center)
                scales = torch.cat(all_scale)
                offs = torch.stack([torch.cat(all_off_l),
                                    torch.cat(all_off_r)], dim=1)
                order = torch.argsort(-scores, stable=True)[:TOPK]
                k = min(TOPK, len(order))
                gt = np.asarray(data["segment"][q], dtype=np.float64)
                rec = {
                    "vid_id": str(data["vid_id"]),
                    "gt_s": float(gt[0]), "gt_e": float(gt[1]),
                    "q384": q384[q].astype(np.float32),
                    "score": scores[order[:k]].numpy(),
                    "logit": logits[order[:k]].numpy(),
                    "level": levels[order[:k]].numpy(),
                    "center": centers[order[:k]].numpy(),
                    "scale": scales[order[:k]].numpy(),
                    "off": offs[order[:k]].numpy(),
                    "feat": feats[order[:k]].numpy().astype(np.float16),
                    "clip_stride": float(data["clip_stride"]),
                    "clip_size": float(data["clip_size"]),
                    "fps": float(data["fps"]),
                    "duration": float(data["duration"]),
                }
                self.recs.append(rec)
            return results

    @staticmethod
    def _empty_rec(data, q, q384):
        return {
            "vid_id": str(data["vid_id"]),
            "gt_s": float(np.asarray(
                data["segment"][q])[0]),
            "gt_e": float(np.asarray(
                data["segment"][q])[1]),
            "q384": q384[q].astype(np.float32),
            "score": np.zeros(0, np.float32),
            "logit": np.zeros(0, np.float32),
            "level": np.zeros(0, np.int64),
            "center": np.zeros(0, np.float32),
            "scale": np.zeros(0, np.float32),
            "off": np.zeros((0, 2), np.float32),
            "feat": np.zeros((0, FEAT_DIM), np.float16),
            "clip_stride": float(data["clip_stride"]),
            "clip_size": float(data["clip_size"]),
            "fps": float(data["fps"]),
            "duration": float(data["duration"]),
        }

    ev = DumpEval(opt)
    if limit_videos:
        import itertools
        ev.dataloader = list(itertools.islice(
            iter(ev.dataloader), limit_videos))
        ev.num_itrs = len(ev.dataloader)
    return ev


def run_dump(split, limit_videos=None):
    ev = build_dump_eval(split, limit_videos)
    ev.run()
    recs = ev.recs
    n = len(recs)
    arrays = {}
    for key in ("score", "logit", "level", "center", "scale", "off",
                "feat", "q384"):
        if key == "q384":
            arrays[key] = np.stack([r[key] for r in recs])
        elif key in ("off",):
            lens = [r[key].shape[0] for r in recs]
            mat = np.zeros((n, max(TOPK, 1), 2), np.float32)
            for i, r in enumerate(recs):
                if r[key].shape[0]:
                    mat[i, :r[key].shape[0]] = r[key]
            arrays[key] = mat
        elif key == "feat":
            lens = [r[key].shape[0] for r in recs]
            mat = np.zeros((n, TOPK, FEAT_DIM), np.float16)
            for i, r in enumerate(recs):
                k = r[key].shape[0]
                if k:
                    mat[i, :k] = r[key]
            arrays[key] = mat
        else:
            dt = np.int64 if key == "level" else np.float32
            mat = np.zeros((n, TOPK), dt)
            for i, r in enumerate(recs):
                k = len(r[key]) if np.ndim(r[key]) > 0 else 0
                if k:
                    mat[i, :k] = r[key][:TOPK]
            arrays[key] = mat
    for key in ("gt_s", "gt_e", "clip_stride", "clip_size", "fps",
                "duration"):
        arrays[key] = np.array([r[key] for r in recs], np.float64)
    arrays["n_valid"] = np.array(
        [min(len(r["score"]), TOPK) for r in recs], np.int64)
    out = TRAIN_NPZ if split == "train" else VAL_NPZ
    CROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    print(f"[{split}] {n} queries -> {out}")


# ---------------------------------------------------------------------------
# decode + IoU for the dumped candidates (FP32, seconds)
# ---------------------------------------------------------------------------

def decode_and_iou(arrays):
    """Returns per-query ious (n, TOPK) and decoded start/end (n, TOPK)."""
    n = len(arrays["gt_s"])
    starts = np.zeros((n, TOPK), np.float32)
    ends = np.zeros((n, TOPK), np.float32)
    ious = np.zeros((n, TOPK), np.float32)
    for i in range(n):
        k = int(arrays["n_valid"][i])
        if k == 0:
            continue
        c = arrays["center"][i, :k]
        s_scale = arrays["scale"][i, :k]
        ol = arrays["off"][i, :k, 0]
        orr = arrays["off"][i, :k, 1]
        s_tok = c - ol * s_scale
        e_tok = c + orr * s_scale
        cs = arrays["clip_stride"][i]
        cc = arrays["clip_size"][i]
        fps = arrays["fps"][i]
        dur = arrays["duration"][i]
        s_sec = np.clip((s_tok * cs + 0.5 * cc) / fps, 0, dur)
        e_sec = np.clip((e_tok * cs + 0.5 * cc) / fps, 0, dur)
        starts[i, :k] = s_tok
        ends[i, :k] = e_tok
        ious[i, :k] = iou_1d(s_sec, e_sec, arrays["gt_s"][i],
                             arrays["gt_e"][i])
    return starts, ends, ious


# ---------------------------------------------------------------------------
# CSRM training (pairwise margin with structured negatives)
# ---------------------------------------------------------------------------

def run_train():
    tr = np.load(TRAIN_NPZ)
    va = np.load(VAL_NPZ)
    dev = "cuda:0"
    torch.manual_seed(SEED)

    # decode train ious for supervision
    _, _, tr_iou = decode_and_iou(tr)
    _, _, va_iou = decode_and_iou(va)

    module = CandidateSetRankingModule(topk=TOPK).to(dev)
    print(f"CSRM params: {module.param_count():,}")
    opt = torch.optim.AdamW(module.parameters(), lr=LR, weight_decay=0.01)

    def to_tensor(arr, key, dt=torch.float32):
        return torch.tensor(arr[key], dtype=dt, device=dev)

    tr_q = to_tensor(tr, "q384")
    tr_f = torch.tensor(tr["feat"], dtype=torch.float32, device=dev)
    tr_logit = to_tensor(tr, "logit")
    tr_level = to_tensor(tr, "level", torch.long)
    tr_iou_t = torch.tensor(tr_iou, device=dev)
    tr_nv = torch.tensor(tr["n_valid"], device=dev)

    va_q = to_tensor(va, "q384")
    va_f = torch.tensor(va["feat"], dtype=torch.float32, device=dev)
    va_logit = to_tensor(va, "logit")
    va_level = to_tensor(va, "level", torch.long)
    va_iou_t = torch.tensor(va_iou, device=dev)
    va_nv = torch.tensor(va["n_valid"], device=dev)
    tr_s, tr_e, _ = decode_and_iou(tr)
    va_s, va_e, _ = decode_and_iou(va)
    tr_s_t = torch.tensor(tr_s, device=dev)
    tr_e_t = torch.tensor(tr_e, device=dev)
    va_s_t = torch.tensor(va_s, device=dev)
    va_e_t = torch.tensor(va_e, device=dev)

    n = len(tr["gt_s"])
    best = {"sp": -2.0, "state": None}
    rng = np.random.default_rng(SEED)

    for ep in range(N_EPOCHS):
        module.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        n_batches = 0
        for i0 in range(0, n, 64):
            idx = perm[i0:i0 + 64]
            b = len(idx)
            scores = module(
                tr_q[idx], tr_f[idx], tr_s_t[idx], tr_e_t[idx],
                tr_logit[idx], tr_level[idx])
            loss = torch.zeros(1, device=dev)
            n_pairs_total = 0
            for bi in range(b):
                qi = int(idx[bi])
                k = int(tr_nv[qi])
                if k < 2:
                    continue
                sc = scores[bi, :k]
                io = tr_iou_t[qi, :k]
                pos = torch.where(io >= 0.5)[0]
                hn = torch.where(io < 0.3)[0]
                if len(pos) == 0 or len(hn) == 0:
                    continue
                np_pairs = min(len(pos) * len(hn), MAX_PAIRS_PER_QUERY)
                sel_p = pos[torch.randint(
                    len(pos), (np_pairs,), device=dev)]
                sel_n = hn[torch.randint(
                    len(hn), (np_pairs,), device=dev)]
                margin_loss = F.relu(
                    MARGIN - sc[sel_p] + sc[sel_n]).mean()
                loss = loss + margin_loss
                n_pairs_total += np_pairs
            if n_pairs_total > 0:
                loss = loss / b
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
                opt.step()
                total_loss += float(loss)
                n_batches += 1
        # val spearman
        module.eval()
        with torch.no_grad():
            qs = []
            for i0 in range(0, len(va["gt_s"]), 256):
                sl = slice(i0, min(i0 + 256, len(va["gt_s"])))
                sc = module(
                    va_q[sl], va_f[sl], va_s_t[sl], va_e_t[sl],
                    va_logit[sl], va_level[sl])
                for bi in range(sc.size(0)):
                    qi = i0 + bi
                    k = int(va_nv[qi])
                    if k < 3:
                        continue
                    v = sc[bi, :k].cpu().numpy()
                    io = va_iou_t[qi, :k].cpu().numpy()
                    from tools.audit_ranking_bottleneck import (
                        rankdata_avg, corr)
                    sp = corr(rankdata_avg(v), rankdata_avg(io))
                    if not math.isnan(sp):
                        qs.append(sp)
        sp = float(np.mean(qs)) if qs else float("nan")
        print(f"ep{ep}: train_loss={total_loss/max(n_batches,1):.4f} "
              f"val_spearman={sp:.4f}")
        if sp > best["sp"]:
            best = {"sp": sp, "state": {
                k: v.cpu().clone() for k, v in
                module.state_dict().items()}}
    module.load_state_dict(best["state"])
    torch.save({"state": module.state_dict(), "best_sp": best["sp"]},
               CKPT)
    print(f"saved -> {CKPT} (best val spearman {best['sp']:.4f})")


# ---------------------------------------------------------------------------
# CSRM inference: replace scores for top-K, run official NMS pipeline
# ---------------------------------------------------------------------------

def run_eval():
    va = np.load(VAL_NPZ)
    dev = "cuda:0"
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    module = CandidateSetRankingModule(topk=TOPK).to(dev).eval()
    module.load_state_dict(ck["state"])
    _, _, ious = decode_and_iou(va)
    n = len(va["gt_s"])

    def to_t(key, dt=torch.float32):
        return torch.tensor(va[key], dtype=dt, device=dev)

    q = to_t("q384")
    f = torch.tensor(va["feat"], dtype=torch.float32, device=dev)
    lg = to_t("logit")
    lv = to_t("level", torch.long)
    s_tok = torch.tensor(
        np.zeros((n, TOPK), np.float32), device=dev)
    e_tok = torch.tensor(
        np.zeros((n, TOPK), np.float32), device=dev)
    # reconstruct start/end from center-scale-offset
    for i in range(n):
        k = int(va["n_valid"][i])
        if k == 0:
            continue
        s_tok[i, :k] = torch.tensor(
            va["center"][i, :k] - va["off"][i, :k, 0]
            * va["scale"][i, :k], device=dev)
        e_tok[i, :k] = torch.tensor(
            va["center"][i, :k] + va["off"][i, :k, 1]
            * va["scale"][i, :k], device=dev)

    counts_base = np.zeros(4)
    counts_csrm = np.zeros(4)
    oracle_counts = np.zeros(4)
    spear_qs = []
    pairwise_win = pairwise_tot = 0
    with torch.no_grad():
        for i0 in range(0, n, 256):
            sl = slice(i0, min(i0 + 256, n))
            csrm_scores = module(
                q[sl], f[sl], s_tok[sl], e_tok[sl], lg[sl], lv[sl]
            ).cpu().numpy()
            for bi in range(csrm_scores.shape[0]):
                qi = i0 + bi
                k = int(va["n_valid"][qi])
                if k == 0:
                    continue
                cs = csrm_scores[bi, :k]
                orig = va["score"][qi, :k]
                io = ious[qi, :k]
                # official NMS on both
                def pipeline(scores):
                    order = np.argsort(-scores, kind="stable")
                    segs = np.stack([
                        (s_tok[qi, :k].cpu().numpy()),
                        (e_tok[qi, :k].cpu().numpy())], axis=1)
                    ks, _ = soft_nms(segs[order], torch.tensor(
                        scores[order]))
                    cs_ = va["clip_stride"][qi]
                    cc_ = va["clip_size"][qi]
                    fps_ = va["fps"][qi]
                    dur_ = va["duration"][qi]
                    ss = np.clip((ks[:, 0] * cs_ + 0.5 * cc_) / fps_,
                                 0, dur_)
                    ee = np.clip((ks[:, 1] * cs_ + 0.5 * cc_) / fps_,
                                 0, dur_)
                    return iou_1d(ss, ee, va["gt_s"][qi], va["gt_e"][qi])
                qi_base = pipeline(orig)
                qi_csrm = pipeline(cs)
                counts_base += (qi_base[0] >= .3, qi_base[0] >= .5,
                                qi_base.max() >= .3, qi_base.max() >= .5)
                counts_csrm += (qi_csrm[0] >= .3, qi_csrm[0] >= .5,
                                qi_csrm.max() >= .3, qi_csrm.max() >= .5)
                # oracle
                best_iou_idx = np.argmax(io)
                oracle_counts += (io[best_iou_idx] >= .3,
                                  io[best_iou_idx] >= .5,
                                  io.max() >= .3, io.max() >= .5)
                # spearman
                from tools.audit_ranking_bottleneck import (
                    rankdata_avg, corr)
                sp = corr(rankdata_avg(cs), rankdata_avg(io))
                if not math.isnan(sp):
                    spear_qs.append(sp)
                # pairwise
                pos_i = np.where(io >= 0.5)[0]
                hn_i = np.where(io < 0.3)[0]
                for a in pos_i:
                    for b_ in hn_i:
                        pairwise_tot += 1
                        pairwise_win += float(cs[a] > cs[b_])

    base_m = {k: float(v / n) for k, v in zip(
        ["R1@0.3", "R1@0.5", "R5@0.3", "R5@0.5"], counts_base)}
    csrm_m = {k: float(v / n) for k, v in zip(
        ["R1@0.3", "R1@0.5", "R5@0.3", "R5@0.5"], counts_csrm)}
    oracle_m = {k: float(v / n) for k, v in zip(
        ["R1@0.3", "R1@0.5", "R5@0.3", "R5@0.5"], oracle_counts)}
    for m in (base_m, csrm_m, oracle_m):
        m["Mean"] = sum(m.values()) / 4
    out = {
        "baseline": base_m,
        "csrm": csrm_m,
        "oracle": oracle_m,
        "csrm_spearman_query_mean": float(np.mean(spear_qs)),
        "csrm_pairwise_acc": pairwise_win / max(pairwise_tot, 1),
        "n_queries": n,
    }
    RESULTS.write_text(json.dumps(out, indent=2) + "\n")
    print(f"baseline Mean: {100*base_m['Mean']:.2f}")
    print(f"CSRM     Mean: {100*csrm_m['Mean']:.2f} "
          f"(delta {100*(csrm_m['Mean']-base_m['Mean']):+.2f})")
    print(f"Oracle   Mean: {100*oracle_m['Mean']:.2f}")
    print(f"CSRM Spearman: {out['csrm_spearman_query_mean']:.4f} "
          f"pairwise: {out['csrm_pairwise_acc']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=[
        "dump-train", "dump-val", "train", "eval"])
    ap.add_argument("--limit-videos", type=int, default=None)
    args = ap.parse_args()
    if args.stage == "dump-train":
        run_dump("train", args.limit_videos)
    elif args.stage == "dump-val":
        run_dump("val", args.limit_videos)
    elif args.stage == "train":
        run_train()
    else:
        run_eval()


if __name__ == "__main__":
    main()

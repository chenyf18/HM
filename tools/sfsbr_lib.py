#!/usr/bin/env python3
"""SFSBR candidate-stream trainer + evaluator hook (HM-SFSBR-031).

The backbone stays frozen in eval mode; only the SFSBRRefiner trains.
Candidates come from the exact inference pipeline (sigmoid>0.001 ->
level concat -> top-2000 -> FP32 decode -> seg-len filter), so training
and inference see identical feature/coordination conventions. GT is
used only for supervision targets/masks during training, never for
candidate selection or window placement.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "hydra"), str(ROOT / "tools")]

import tools.run_formal_ablation as rfa  # noqa: E402
from libs.modeling.sfsbr import (  # noqa: E402
    SFSBRRefiner, gather_window, apply_delta, supervision_targets,
)

R1_OPT = ROOT / "experiments/allocator_diagnosis/seed_1/AUC/opt.yaml"
R1_MODELS = ROOT / "experiments/allocator_diagnosis/seed_1/AUC/models"
SROOT = ROOT / "experiments/sfsbr"
TOPK_REFINE = 50
FEAT_DIM = 384


def build_frozen_evaluator(split, mode, limit_videos=None,
                           recorder_slots=None):
    """FormalEvaluator with recorders for F1 features, pools and q384.

    mode: "interp" (arm B) or "fine" (arm C) - only controls which local
    feature source gather_window uses at refine time.
    """
    import shutil
    from tools.lqac_probe import _FusionRecorder
    from libs import load_opt

    work = SROOT / f"backbone_{split}"
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

    class CandEvaluator(rfa.FormalEvaluator):
        def __init__(self, opt, precision_policy="p2"):
            self.records = []
            self._collect_log = []
            self._fusion_rec = None
            self._text_pool = None
            self.rng = np.random.default_rng(7)
            super().__init__(opt, precision_policy=precision_policy)
            self._fusion_rec = _FusionRecorder(self.model.fusion)
            self._fusion_rec.install()
            layer0 = self.model.fusion.layers[0]
            self.proj_ln = layer0.ln_xattn_kv
            self.proj_val = layer0.xattn.xattn.value
            original_collect = self._collect_segments

            def collect_wrapper(fpn_points, fpn_logits, fpn_offsets,
                                fpn_masks, ext_scores=None,
                                return_levels=False, fpn_feats=None,
                                query_idx=0):
                result = original_collect(
                    fpn_points, fpn_logits, fpn_offsets, fpn_masks,
                    ext_scores, return_levels=return_levels,
                    fpn_feats=fpn_feats, query_idx=query_idx)
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
                        "pos": keep.nonzero(as_tuple=False)
                        .flatten().cpu(),
                        "score": scores[keep].detach().float().cpu(),
                        "center": points[:, 0][keep].detach()
                        .float().cpu(),
                        "scale": points[:, 3][keep].detach()
                        .float().cpu(),
                        "offset": offsets[keep].detach().float().cpu(),
                    })
                self._collect_log.append(per_level)
                return result

            self._collect_segments = collect_wrapper
            orig_text2 = self.model.encode_text2

            def text2_rec(text, text_masks, text_size):
                out = orig_text2(text, text_masks, text_size)
                self._text_pool = (out[0].detach(), out[1].detach())
                return out

            self.model.encode_text2 = text2_rec

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
                raise RuntimeError("collect calls != queries")
            if len(self._fusion_rec.outputs) != 2:
                raise RuntimeError("fusion calls != 2 (single window)")
            f1, f1_masks = self._fusion_rec.outputs[1]
            q384 = self._q384().cpu().numpy()
            raw_vid = data["vid"]                # (1, C, T) coarse
            for q in range(n_q):
                per_level = self._collect_log[q]
                rows = []
                feats = []
                for pl in per_level:
                    n_k = pl["score"].numel()
                    entry = np.empty((n_k, 6), dtype=np.float32)
                    entry[:, 0] = pl["score"].numpy()
                    entry[:, 1] = pl["level"]
                    entry[:, 2] = pl["center"].numpy()
                    entry[:, 3] = pl["scale"].numpy()
                    entry[:, 4] = pl["offset"][:, 0].numpy()
                    entry[:, 5] = pl["offset"][:, 1].numpy()
                    rows.append(entry)
                    lvl = int(pl["level"])
                    pos = pl["pos"].to(f1[lvl].device)
                    feats.append(f1[lvl][q][:, pos].transpose(0, 1)
                                 .detach().float().cpu())
                pool = (np.concatenate(rows) if rows
                        else np.zeros((0, 6), np.float32))
                F1 = (torch.cat(feats) if feats
                      else torch.zeros(0, FEAT_DIM))
                gt = np.asarray(data["segment"][q], dtype=np.float64)
                self.records.append({
                    "pool": pool, "F1": F1,
                    "q384": q384[q].astype(np.float32),
                    "gt": gt.astype(np.float32),
                    "raw_vid": (raw_vid if raw_vid.dim() == 3 else raw_vid[None]),          # shared ref, read-only
                    "vid_id": str(data["vid_id"]),
                    "clip_stride": float(data["clip_stride"]),
                    "clip_size": float(data["clip_size"]),
                    "fps": float(data["fps"]),
                    "duration": float(data["duration"]),
                })
            return results

    ev = CandEvaluator(opt, precision_policy="p3")
    ev.sfsbr_mode = mode
    if limit_videos:
        import itertools
        ev.dataloader = list(itertools.islice(
            iter(ev.dataloader), limit_videos))
        ev.num_itrs = len(ev.dataloader)
    return ev


_fine_cache = {}


def fine_matrix(vid_id):
    if vid_id in _fine_cache:
        return _fine_cache[vid_id]
    import glob
    f = glob.glob(f"data/ego4d/egovlp_features/video/{vid_id}.*")
    mat = np.load(f[0], mmap_mode="r")
    _fine_cache[vid_id] = mat
    if len(_fine_cache) > 8:
        _fine_cache.pop(next(iter(_fine_cache)))
    return mat


def refine_pool(refiner, rec, mode, topk=TOPK_REFINE, device="cuda"):
    """Return corrected (start,end) token arrays for the top-k rows.

    All other pool rows keep their decoded coordinates unchanged.
    """
    pool = rec["pool"]
    n = pool.shape[0]
    if n == 0:
        return (np.zeros(0, np.float32), np.zeros(0, np.float32))
    order = np.argsort(-pool[:, 0], kind="stable")[:topk]
    centers = pool[order, 2]
    scales = pool[order, 3]
    start_tok = centers - pool[order, 4] * scales
    end_tok = centers + pool[order, 5] * scales
    st = torch.tensor(start_tok, dtype=torch.float32)
    en = torch.tensor(end_tok, dtype=torch.float32)
    fine_mat = None
    if mode == "fine":
        fine_mat = fine_matrix(rec["vid_id"])
    with torch.no_grad():
        # window centres in FINE-row coords (token*2); video offset 0
        sw = gather_window(st * 2.0, rec["raw_vid"], fine_mat)
        ew = gather_window(en * 2.0, rec["raw_vid"], fine_mat)
        delta = refiner(
            rec["F1"][order].to(device),
            torch.from_numpy(np.broadcast_to(
                rec["q384"], (len(order), FEAT_DIM)).copy()).to(device),
            sw.to(device), ew.to(device)).cpu()
    t_valid = float(rec["raw_vid"].size(-1))
    s2, e2 = apply_delta(st, en, delta, t_valid)
    return s2.numpy().astype(np.float32), e2.numpy().astype(np.float32), \
        order

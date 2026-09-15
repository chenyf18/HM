#!/usr/bin/env python3
"""Mamba State Sensitivity Audit (HM-MAMBA-STATE-026).

Frozen A-U-clean. Question: does the Hydra (bidirectional Mamba2)
temporal encoder carry ANY query-conditioned modulation space, and is
the modulation localized at GT moments?

Per code trace (HM-MAMBA-STATE-026 section 1), in the A-U-clean forward
the only query path into the scan is the upstream input-level early
fusion (per-query x); inside the block, importance weighting is applied
only under allocator_policy == "learned" (AUC uses uniform -> weights
ones), and query modulation / query gates are disabled.

Experiment: for each sampled val video, forward the same video under
  Q1 = a real query of the video
  Q2 = a random query from another video
  Q3 = another real query of the same video (when available)
and record, per FPN level:
  - Hydra output (state-propagated sequence) at sequence-token slots
  - block output features (seq_out)
Statistics: cross-query cosine, relative difference norm, and the
inside-GT / outside-GT ratio of |h(Q1) - h(Q2)|.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "hydra"), str(ROOT / "tools")]

from tools.audit_ranking_bottleneck import AU_ROOT  # noqa: E402
from tools.qc_audit import AUC_P32_ROOT  # noqa: E402

MROOT = ROOT / "experiments/mamba_state_audit"
RESULTS_JSON = MROOT / "mamba_state_results.json"
N_LEVELS = 8


def build_evaluator(limit_videos=None):
    import shutil
    import itertools
    import tools.run_formal_ablation as rfa
    from libs import load_opt

    MROOT.mkdir(parents=True, exist_ok=True)
    opt_path = MROOT / "opt_val.yaml"
    if not opt_path.exists():
        shutil.copyfile(AUC_P32_ROOT / "opt.yaml", opt_path)
    opt = load_opt(str(opt_path), is_training=False)
    opt["_root"] = str(MROOT)
    opt["_ckpt"] = "last"
    models_link = MROOT / "models"
    if not models_link.exists():
        models_link.symlink_to(AU_ROOT / "models",
                               target_is_directory=True)

    class StateEvaluator(rfa.FormalEvaluator):
        def __init__(self, opt, precision_policy="p2"):
            self.hydra_log = []       # per level-call: (out, seq_pos)
            self.seqout_log = []      # per level-call: seq_out
            super().__init__(opt, precision_policy=precision_policy)
            vid_net = self.model.vid_net
            for lvl, block in enumerate(vid_net.branch):
                hydra = block.global_encoder
                alloc = block.adaptive_anchor_allocator
                orig_hydra = hydra.forward
                orig_alloc = alloc.forward

                def make_hydra_rec(orig, level):
                    def rec(x, **kwargs):
                        out = orig(x, **kwargs)
                        self.hydra_log.append((out.detach(), level))
                        return out
                    return rec

                def make_alloc_rec(orig, level):
                    def rec(x, mask, importance, **kwargs):
                        state = orig(x, mask, importance, **kwargs)
                        self.hydra_log.append(
                            (state["sequence_positions"].detach(), level,
                             "layout"))
                        return state
                    return rec

                hydra.forward = make_hydra_rec(orig_hydra, lvl)
                alloc.forward = make_alloc_rec(orig_alloc, lvl)

            orig_collect = self._collect_segments

            def collect_noop(*args, **kwargs):
                return orig_collect(*args, **kwargs)

            self._collect_segments = collect_noop

        def forward_video_queries(self, vid, window_size, input_vid_len,
                                  text_feats, text_masks):
            """Batched forward of one window under the given query batch.

            hydra_log receives, per level in order: the allocator layout
            (sequence_positions) then the Hydra output.
            """
            m = self.model
            self.hydra_log = []
            with torch.no_grad(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16):
                window = vid[..., :window_size]
                window = torch.nn.functional.pad(
                    window, (0, input_vid_len - window_size))[None].cuda()
                window_mask = (
                    torch.arange(input_vid_len).view(1, 1, -1).cuda()
                    < window_size)
                wp, wp_mask = m.vid_proj(window, window_mask)
                text = text_feats
                text_masks = text_masks
                # single video repeated per packed query (official
                # kv_size semantics: repeat_interleave([n_q]))
                text_size = torch.tensor(
                    [text.size(0)], device="cuda", dtype=torch.long)
                fused, fused_mask = m.fusion(
                    wp, wp_mask, text, text_masks, text_size)
                m.encode_video(
                    fused, fused_mask,
                    query_feat=text if self.query_conditioned else None,
                    query_mask=text_masks if self.query_conditioned
                    else None,
                    text_size=text_size,
                )
            layouts, hydras = [], []
            pending_layout = {}
            for entry in self.hydra_log:
                if len(entry) == 3 and entry[2] == "layout":
                    pending_layout[entry[1]] = entry[0]
                else:
                    lvl = entry[1]
                    layouts.append(pending_layout.pop(lvl))
                    hydras.append(entry[0])
            return text_feats.size(0), layouts, hydras

        def forward_and_capture(self, vid, window_size, input_vid_len,
                                text_feats, text_masks):
            """Full capture incl. seq_out via branch-level wrappers."""
            m = self.model
            self.hydra_log = []
            self.seqout_log = []
            vid_net = m.vid_net
            for lvl, block in enumerate(vid_net.branch):
                if not hasattr(block, "_seqout_wrapped"):
                    orig_fwd = type(block).forward

                    def make_block_rec(orig, level):
                        def rec(bself, x, mask, **kwargs):
                            out = orig(bself, x, mask, **kwargs)
                            # anchor_out, seq_out (B,D,T) -> record
                            self.seqout_log.append(
                                (out[1].detach(), level))
                            return out
                        return rec

                    import types
                    block.forward = types.MethodType(
                        make_block_rec(orig_fwd, lvl), block)
                    block._seqout_wrapped = True
            n_q, layouts, hydras = self.forward_video_queries(
                vid, window_size, input_vid_len, text_feats, text_masks)
            return n_q, layouts, hydras, self.seqout_log

    ev = StateEvaluator(opt, precision_policy="p3")
    if limit_videos:
        ev.dataloader = list(itertools.islice(
            iter(ev.dataloader), limit_videos))
    return ev


def to_seconds(tok, clip_stride, clip_size, fps, duration):
    sec = (tok * clip_stride + 0.5 * clip_size) / fps
    return float(np.clip(sec, 0, duration))


def run(n_videos=120):
    ev = build_evaluator(limit_videos=n_videos)
    rng = np.random.default_rng(99)
    videos = []
    # pass 1: collect per-video first two queries' text features + GT
    for data_list in ev.dataloader:
        data = data_list[0]
        tokens = data["text"]
        if not isinstance(tokens, tuple):
            tokens = (tokens, )
        if len(tokens) < 1:
            continue
        vid_len = data["vid"].size(-1)
        videos.append({
            "vid": data["vid"],
            "vid_len": vid_len,
            "tokens": list(tokens[:2]),
            "gts": [np.asarray(s, dtype=np.float64)
                    for s in data["segment"][:2]],
            "clip_stride": float(data["clip_stride"]),
            "clip_size": float(data["clip_size"]),
            "fps": float(data["fps"]),
            "duration": float(data["duration"]),
        })
        if len(videos) >= n_videos:
            break

    stride_mul = ev.min_chunk_size * ev.vid_stride
    stats = {f"L{l}": {k: [] for k in (
        "cos_h_q1q2", "cos_h_q1q3", "rel_diff_h_q1q2", "rel_diff_h_q1q3",
        "cos_out_q1q2", "cos_out_q1q3",
        "gt_ratio_h", "gt_ratio_out",
        "gt_in_diff", "gt_out_diff")} for l in range(N_LEVELS)}
    n_pairs = 0

    for i, v in enumerate(videos):
        q1, q2t, q3t = 0, None, None
        if i == 0:
            continue
        rand_src = videos[rng.integers(1, len(videos))]
        if rand_src is v:
            continue
        tokens_sel = [v["tokens"][0], rand_src["tokens"][0]]  # Q1, Q2
        if len(v["tokens"]) > 1:
            tokens_sel.append(v["tokens"][1])                 # Q3
        with torch.no_grad():
            text_b, text_m, text_s = ev._batchify_text2(
                text_list=[tuple(tokens_sel)])
            text_b = text_b.cuda()
            text_m = text_m.cuda()
            text_s = text_s.cuda()
            feats, masks = ev.model.encode_text2(text_b, text_m, text_s)
        feats = feats.detach()
        masks = masks.detach()
        window_size = v["vid_len"]
        input_vid_len = (
            (window_size + stride_mul - 1) // stride_mul) * stride_mul
        gt = v["gts"][0]
        try:
            n_q, layouts, hydras, seqouts = ev.forward_and_capture(
                v["vid"], window_size, input_vid_len, feats, masks)
        except RuntimeError as exc:
            print("skip video", i, repr(exc))
            continue
        n_q = len(tokens_sel)
        for lvl in range(N_LEVELS):
            h = hydras[lvl].float()               # (B, T', D)
            layout = layouts[lvl].float()          # (B, T') slot->seq idx
            so = seqouts[lvl][0].float()           # (B, D, T_l) seq_out
            # h at sequence slots for each query condition
            h_seq = []
            for b in range(n_q):
                pos = layout[b].long()
                h_seq.append(h[b][pos])            # (T_l, D)
            span = 2 ** lvl
            centers_tok = (torch.arange(
                h_seq[0].size(0), dtype=torch.float64) + 0.5) * span
            in_gt = torch.tensor([
                gt[0] <= to_seconds(
                    c, v["clip_stride"], v["clip_size"], v["fps"],
                    v["duration"]) <= gt[1]
                for c in centers_tok.tolist()
            ], dtype=torch.bool)
            if in_gt.sum() == 0 or (~in_gt).sum() == 0:
                continue
            h1, h2 = h_seq[0], h_seq[1]
            cos12 = torch.nn.functional.cosine_similarity(
                h1, h2, dim=-1)
            diff12 = (h1 - h2).norm(dim=-1)
            base = h1.norm(dim=-1).clamp_min(1e-6)
            st = stats[f"L{lvl}"]
            st["cos_h_q1q2"].append(float(cos12.mean()))
            st["rel_diff_h_q1q2"].append(float((diff12 / base).mean()))
            st["gt_in_diff"].append(float(diff12[in_gt].mean()))
            st["gt_out_diff"].append(float(diff12[~in_gt].mean()))
            st["gt_ratio_h"].append(
                float(diff12[in_gt].mean() / diff12[~in_gt].mean()))
            o1 = so[0].transpose(0, 1)
            o2 = so[1].transpose(0, 1)
            cos_o = torch.nn.functional.cosine_similarity(o1, o2, dim=-1)
            do = (o1 - o2).norm(dim=-1)
            ob = o1.norm(dim=-1).clamp_min(1e-6)
            st["cos_out_q1q2"].append(float(cos_o.mean()))
            st["gt_ratio_out"].append(
                float(do[in_gt].mean() / do[~in_gt].mean()))
            if n_q == 3:
                h3 = h_seq[2]
                cos13 = torch.nn.functional.cosine_similarity(
                    h1, h3, dim=-1)
                d13 = (h1 - h3).norm(dim=-1)
                st["cos_h_q1q3"].append(float(cos13.mean()))
                st["rel_diff_h_q1q3"].append(float((d13 / base).mean()))
                o3 = so[2].transpose(0, 1)
                st["cos_out_q1q3"].append(float(
                    torch.nn.functional.cosine_similarity(
                        o1, o3, dim=-1).mean()))
        n_pairs += 1

    out = {"n_videos": n_pairs, "levels": {}}
    print(f"videos compared: {n_pairs}")
    import numpy as _np
    print(f"{'Lvl':4s} {'cos_h12':>8s} {'cos_h13':>8s} "
          f"{'reldiff':>9s} {'GTin':>7s} {'GTout':>7s} {'ratio':>6s} "
          f"{'cos_o12':>8s} {'ratio_o':>7s}")
    for lvl in range(N_LEVELS):
        st = {}
        for k, v in stats[f"L{lvl}"].items():
            if not len(v):
                st[k] = float("nan")
                continue
            arr = np.asarray(v, dtype=float)
            st[k] = float(arr.mean())
            if k == "gt_ratio_h":
                st[k + "_std"] = float(arr.std())
                st[k + "_frac_gt1"] = float((arr > 1.0).mean())
                se = arr.std() / max(math.sqrt(len(arr)), 1e-9)
                st[k + "_tstat"] = float(
                    (arr.mean() - 1.0) / se) if se > 0 else float("nan")
            if k == "gt_ratio_out":
                st[k + "_std"] = float(arr.std())
                st[k + "_frac_gt1"] = float((arr > 1.0).mean())
        out["levels"][f"L{lvl}"] = st
        print(f"L{lvl:<3d} {st['cos_h_q1q2']:8.4f} "
              f"{st['cos_h_q1q3']:8.4f} {st['rel_diff_h_q1q2']:10.4f} "
              f"{st['gt_in_diff']:7.4f} {st['gt_out_diff']:7.4f} "
              f"{st['gt_ratio_h']:6.3f} {st['cos_out_q1q2']:8.4f} "
              f"{st['gt_ratio_out']:7.3f}")
    RESULTS_JSON.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print("results ->", RESULTS_JSON)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-videos", type=int, default=120)
    args = ap.parse_args()
    run(args.n_videos)


if __name__ == "__main__":
    main()

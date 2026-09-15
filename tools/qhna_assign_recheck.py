#!/usr/bin/env python3
"""QHNA recheck part 2: true assignment conflicts + matched-distribution
probe (HM-QHNA-028-R2). No training, no model modification.

A. Replays N training batches through the REAL training path pieces
   (trainer dataloader, _batchify, model forward, adaptive points,
   trainer._annotate_points with the actual radius=1.5 rule) and reports
   P(gt_labels=0 | decoded IoU>=0.5) and P(gt_labels=1 | decoded IoU<0.3)
   per FPN level with numerators/denominators. No in-GT proxy.

B. One matched-distribution probe: train and evaluate strictly on the
   TRUE top-50-by-cls candidates (no GT-assisted positive injection),
   same MLP architecture, single run, no hyperparameter search.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "hydra"), str(ROOT / "tools")]

RECHECK = ROOT / "experiments/qhna_recheck"
ASSIGN_JSON = RECHECK / "assignment_conflicts.json"
PROBE_JSON = RECHECK / "matched_probe.json"
N_BATCHES = 40


def iou_1d_np(s1, e1, s2, e2):
    inter = np.maximum(0, np.minimum(e1, e2) - np.maximum(s1, s2))
    union = (e1 - s1) + (e2 - s2) - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def run_assignment():
    import tools.run_formal_ablation as rfa
    from libs import load_opt
    from libs.modeling.temporal_coordinates import decode_offsets

    src = ROOT / "experiments/allocator_diagnosis/seed_1/AUC"
    work = RECHECK / "assign_replay"
    if work.exists():
        shutil.rmtree(work)
    (work / "models").mkdir(parents=True)
    (work / "states").mkdir(parents=True)
    shutil.copyfile(src / "models/last.pth", work / "models/last.pth")
    st = src / "states/last.pth"
    if st.exists():
        shutil.copyfile(st, work / "states/last.pth")
    opt = load_opt(str(src / "opt.yaml"), is_training=True)
    opt["seed"] = 1
    opt["_root"] = str(work)
    opt["_resume"] = True
    opt["_distributed"] = False
    opt["_world_size"] = 1
    trainer = rfa.FormalTrainer(opt, "ASSIGN_REPLAY", work,
                                max_steps=None, precision_policy="p3")
    trainer.model.eval()

    counts = {lv: {"pos_den": 0, "pos_lbl0": 0, "hn_den": 0, "hn_lbl1": 0,
                   "topk_hn_den": 0, "topk_hn_lbl1": 0, "n_points": 0}
              for lv in range(8)}
    batches = 0
    with torch.no_grad():
        for data_list in trainer.dataloader:
            vid, vid_masks, text, text_masks, text_size = trainer._batchify(
                vid_list=[d['vid'] for d in data_list],
                text_list=[d['text'] for d in data_list])
            vid = vid.cuda()
            vid_masks = vid_masks.cuda()
            text = text.cuda()
            text_masks = text_masks.cuda()
            text_size = text_size.cuda()
            targets = torch.cat(
                [d['target'] for d in data_list]).cuda().float()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                fpn_logits, _, fpn_offsets, fpn_masks, fpn, fpn_masks_lv, \
                    *_ = trainer.model(
                        vid, vid_masks, text, text_masks, text_size)
            fpn_logits = torch.cat(fpn_logits, dim=1).float()
            fpn_offsets = torch.cat(fpn_offsets, dim=1).float()
            fpn_masks = torch.cat(fpn_masks, dim=1)
            mref = trainer.model.module if hasattr(
                trainer.model, 'module') else trainer.model
            fpn_points = trainer.adaptive_pt_gen(
                mref.vid_net.last_temporal_metadata, fpn_masks_lv)
            n_points = [m.size(-1) for m in fpn_masks_lv]
            points = torch.cat(fpn_points, dim=1).float()   # (bs, p, 8)
            gt_labels, _ = trainer._annotate_points(points, targets)
            segs = decode_offsets(points, fpn_offsets)      # (bs, p, 2)
            ious = iou_1d_np(
                segs[..., 0].cpu().numpy(), segs[..., 1].cpu().numpy(),
                targets[:, 0][:, None].cpu().numpy(),
                targets[:, 1][:, None].cpu().numpy())
            labels = gt_labels.cpu().numpy()
            valid = fpn_masks.cpu().numpy()
            cls_sc = torch.sigmoid(fpn_logits).cpu().numpy()
            # per-level slices along the concatenated point axis
            start = 0
            for lv, np_l in enumerate(n_points):
                sl = slice(start, start + np_l)
                start += np_l
                lab = labels[:, sl][valid[:, sl]]
                io = ious[:, sl][valid[:, sl]]
                c = counts[lv]
                c["n_points"] += int(lab.shape[0])
                m_pos = io >= 0.5
                c["pos_den"] += int(m_pos.sum())
                c["pos_lbl0"] += int((m_pos & (lab == 0)).sum())
                m_hn = io < 0.3
                c["hn_den"] += int(m_hn.sum())
                c["hn_lbl1"] += int((m_hn & (lab == 1)).sum())
            # top-50-by-cls HN (any level)
            for b in range(labels.shape[0]):
                vb = valid[b]
                order = np.argsort(-cls_sc[b][vb], kind="stable")[:50]
                io50 = ious[b][vb][order]
                lab50 = labels[b][vb][order]
                m = io50 < 0.3
                # attribute to level of each point
                lv_of = np.concatenate(
                    [np.full(n_points[l], l) for l in range(8)])[vb][order]
                for lv in range(8):
                    mm = m & (lv_of == lv)
                    counts[lv]["topk_hn_den"] += int(mm.sum())
                    counts[lv]["topk_hn_lbl1"] += int(
                        (mm & (lab50 == 1)).sum())
            batches += 1
            if batches >= N_BATCHES:
                break

    out = {"n_batches": batches,
           "rule": "labels = inside_center_window(radius=1.5*stride) AND "
                   "inside_regression_range; decoded IoU from FP32 decode "
                   "of the same training forward"}
    print(f"batches replayed: {batches}")
    print(f"{'L':>2} {'P(lbl0|IoU>=.5)':>16} {'num/den':>16} "
          f"{'P(lbl1|IoU<.3)':>15} {'num/den':>16} {'P(lbl1|top50HN)':>16}")
    for lv in range(8):
        c = counts[lv]
        p1 = c["pos_lbl0"] / max(c["pos_den"], 1)
        p2 = c["hn_lbl1"] / max(c["hn_den"], 1)
        p3 = c["topk_hn_lbl1"] / max(c["topk_hn_den"], 1)
        print(f"L{lv} {p1:16.4f} {c['pos_lbl0']:>7}/{c['pos_den']:<8} "
              f"{p2:15.4f} {c['hn_lbl1']:>7}/{c['hn_den']:<8} {p3:16.4f}")
        out[f"L{lv}"] = {k: int(v) for k, v in c.items()}
    ASSIGN_JSON.write_text(json.dumps(out, indent=2) + "\n")
    print("->", ASSIGN_JSON)


def run_matched_probe():
    va = np.load(RECHECK / "val_top.npz")
    tr = np.load(RECHECK / "train_top.npz")
    dev = "cuda:0"

    def true_top50(d):
        keep = []
        for qid in np.unique(d["qi"]):
            m = d["qi"] == qid
            idxs = np.where(m)[0]
            order = np.argsort(-d["rows"][idxs, 1], kind="stable")[:50]
            keep.append(idxs[order])
        keep = np.concatenate(keep)
        return keep

    kv, kt = true_top50(va), true_top50(tr)
    def build(d, keep):
        q = d["q384"][d["qi"][keep]]
        x = np.concatenate([d["F1"][keep].astype(np.float32), q], 1)
        y = (d["ious"][keep] >= 0.5).astype(np.float32)
        return (torch.tensor(x), torch.tensor(y),
                torch.tensor(d["qi"][keep]))

    Xtr, Ytr, Qtr = build(tr, kt)
    Xva, Yva, Qva = build(va, kv)
    print(f"matched-distribution rows: train {len(Ytr)} "
          f"(pos {int(Ytr.sum())}), val {len(Yva)} "
          f"(pos {int(Yva.sum())})")

    torch.manual_seed(5)
    probe = torch.nn.Sequential(
        torch.nn.Linear(768, 128), torch.nn.GELU(),
        torch.nn.Linear(128, 1)).to(dev)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3,
                           weight_decay=1e-4)
    n = len(Ytr)
    for ep in range(10):
        perm = torch.randperm(n)
        for i in range(0, n, 8192):
            idx = perm[i:i + 8192]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                probe(Xtr[idx].to(dev)).squeeze(-1), Ytr[idx].to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    with torch.no_grad():
        P = torch.cat([
            torch.sigmoid(probe(
                Xva[i:i + (1 << 17)].to(dev)).squeeze(-1)).cpu()
            for i in range(0, len(Yva), 1 << 17)]).numpy()
    io_v = va["ious"][kv]
    sc_v = va["rows"][kv][:, 1]
    qp = Qva.numpy()

    rank_full = np.full(len(io_v), 10**9)
    for qid in np.unique(qp):
        idxs = np.where(qp == qid)[0]
        order = np.argsort(-sc_v[idxs], kind="stable")
        rank_full[idxs[order]] = np.arange(len(idxs))

    def collect(hn_rank_lim):
        gp, gn, gq = [], [], []
        for qid in np.unique(qp):
            m = qp == qid
            ps = np.where(m & (io_v >= 0.5))[0]
            hsel = m & (io_v < 0.3)
            if hn_rank_lim is not None:
                hsel = hsel & (rank_full < hn_rank_lim)
            hs = np.where(hsel)[0]
            for a in ps:
                for b in hs:
                    gp.append(a); gn.append(b); gq.append(qid)
        return np.array(gp), np.array(gn), np.array(gq)

    def wr(gpos, gneg):
        sp, sn = P[gpos], P[gneg]
        return float(((sp > sn).sum() + 0.5 * (sp == sn).sum()) / len(sp))

    from tools.audit_ranking_bottleneck import rankdata_avg as _ra
    res = {}
    for tag, lim in (("pairwise_allHN_top50", None), ("pairwise_HN_rank20", 20)):
        gp, gn, gq = collect(lim)
        w = wr(gp, gn)
        rng = np.random.default_rng(0)
        qs = np.unique(gq); vals = []
        for _ in range(500):
            sel = rng.choice(qs, qs.size, replace=True)
            mk = np.isin(gq, sel)
            vals.append(wr(gp[mk], gn[mk]))
        res[tag] = {"win": w, "ci95": [float(np.percentile(vals, 2.5)),
                                       float(np.percentile(vals, 97.5))],
                    "n_pairs": int(len(gp))}
    np.savez_compressed(RECHECK / "matched_probe_scores.npz",
                        kv=kv, P=P, qi=qp)
    wins = res["pairwise_allHN_top50"]["win"]
    tot = res["pairwise_allHN_top50"]["n_pairs"]
    import math
    from tools.audit_ranking_bottleneck import rankdata_avg
    pos = Yva.numpy() == 1
    r = rankdata_avg(P)
    auroc = float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2)
                  / (pos.sum() * (~pos).sum()))
    res["auroc"] = auroc
    PROBE_JSON.write_text(json.dumps(res, indent=2) + "\n")
    print(f"matched-distribution probe: AUROC={auroc:.4f}")
    for tag in ("pairwise_allHN_top50", "pairwise_HN_rank20"):
        r = res[tag]
        print(f"  {tag}: win={r['win']:.4f} CI95=[{r['ci95'][0]:.4f},"
              f"{r['ci95'][1]:.4f}] n={r['n_pairs']}")


def main():
    run_assignment()
    run_matched_probe()


if __name__ == "__main__":
    main()

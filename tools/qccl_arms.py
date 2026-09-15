#!/usr/bin/env python3
"""QCCL fixed-budget three-arm verification (HM-QCCL-030).

Arms (shared mining, frozen spec):
  G0: base loss only
  G1: base + mined pointwise BCE  (0.5*posBCE + 0.5*hnBCE, query-mean)
  G2: base + mined pairwise softplus(m=0.5 - z_p + z_n), query-mean
Mining (identical for G1/G2, FP32 detached decode):
  pos = any valid candidate with decoded IoU>=0.5, top-4 by IoU
  HN  = cls top-50 valid candidates with decoded IoU<0.3, top-8 by cls
  pairs per query = |pos| x |HN| <= 32; query skipped iff either is empty

Common start: retrained A-U-clean-R1 checkpoint, model_ema branch, for
both the train model and EMA; optimizer/scheduler re-initialised; LR
locked to constant 1e-4 (no warmup/decay, no search). Budget: 8000 steps,
evals at 0/2000/4000/8000 on the official val pipeline.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "hydra"), str(ROOT / "tools")]

import tools.run_formal_ablation as rfa  # noqa: E402
from libs.modeling.temporal_coordinates import decode_offsets  # noqa: E402

R1_CKPT = ROOT / "experiments/allocator_diagnosis/seed_1/AUC/models/last.pth"
R1_OPT = ROOT / "experiments/allocator_diagnosis/seed_1/AUC/opt.yaml"
ARMS_ROOT = ROOT / "experiments/qccl_arms"
CALIB_JSON = ARMS_ROOT / "calibration.json"
SPEC = dict(POS_IOU=0.5, HN_IOU=0.3, TOPK=50, MAX_POS=4, MAX_HN=8,
            MARGIN=0.5)
EVAL_STEPS = (0, 2000, 4000, 8000)
MAX_STEPS = 8000
LOCKED_LR = 1e-4
N_CALIB_BATCHES = 8


def iou_t(segs, targets):
    s1, e1 = segs[..., 0], segs[..., 1]
    s2, e2 = targets[:, 0][:, None], targets[:, 1][:, None]
    inter = (torch.minimum(e1, e2) - torch.maximum(s1, s2)).clamp_min(0)
    union = (e1 - s1) + (e2 - s2) - inter
    return torch.where(union > 0, inter / union.clamp_min(1e-12),
                       torch.zeros_like(union))


def compute_aux(arm, logits, offsets, masks, points, targets):
    """Return (loss, diag). logits/offsets keep grad; mining detached."""
    z = logits.float()
    bs = z.size(0)
    with torch.no_grad():
        segs = decode_offsets(points.float(), offsets.float())
        iou = iou_t(segs, targets)                       # (bs, p)
        valid = masks.bool()
        order = torch.argsort(
            torch.argsort(z, dim=1, descending=True, stable=True),
            dim=1, stable=True)
        rank = order.argsort(dim=1, stable=True)         # 0 = highest cls
        pos_m = valid & (iou >= SPEC["POS_IOU"])
        hn_m = valid & (iou < SPEC["HN_IOU"]) & (rank < SPEC["TOPK"])
    losses = []
    n_pairs = 0
    n_valid_q = 0
    for b in range(bs):
        pi = torch.where(pos_m[b])[0]
        ni = torch.where(hn_m[b])[0]
        if pi.numel() == 0 or ni.numel() == 0:
            continue
        with torch.no_grad():
            pi = pi[torch.argsort(iou[b][pi], descending=True)
                     [:SPEC["MAX_POS"]]]
            ni = ni[torch.argsort(z[b][ni], descending=True)
                     [:SPEC["MAX_HN"]]]
        n_valid_q += 1
        n_pairs += pi.numel() * ni.numel()
        zp, zn = z[b][pi], z[b][ni]
        if arm == "G2":
            marg = SPEC["MARGIN"] - zp[:, None] + zn[None, :]
            losses.append(F.softplus(marg).mean())
        else:
            losses.append(
                0.5 * F.binary_cross_entropy_with_logits(
                    zp, torch.ones_like(zp))
                + 0.5 * F.binary_cross_entropy_with_logits(
                    zn, torch.zeros_like(zn)))
    if not losses:
        return 0.0 * z.sum(), {"n_valid_q": 0, "n_pairs": 0,
                               "coverage": 0.0}
    loss = torch.stack(losses).mean()
    return loss, {"n_valid_q": n_valid_q, "n_pairs": n_pairs,
                  "coverage": n_valid_q / bs}


class ArmTrainer(rfa.FormalTrainer):
    def __init__(self, opt, label, root, arm, lam, max_steps=MAX_STEPS):
        self.arm = arm
        self.lam = float(lam)
        self._stash_model = None
        self._stash_points = None
        self._aux_diag = []
        super().__init__(opt, label, root, max_steps=max_steps,
                         precision_policy="p3")

    def _install_common_init(self):
        """Unified start: R1 model_ema -> model & EMA; fresh optimizer
        state already guaranteed by construction; lock constant LR."""
        ckpt = torch.load(R1_CKPT, map_location="cpu", weights_only=False)
        self.model.load_compatible_state_dict(ckpt["model_ema"])
        self._ema_init()
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda epoch: 1.0)
        for g in self.optimizer.param_groups:
            g["lr"] = LOCKED_LR
            g["initial_lr"] = LOCKED_LR

    def _install_hooks(self):
        m = self.model
        orig_forward = m.forward
        trainer = self

        def forward_recorder(*args, **kwargs):
            out = orig_forward(*args, **kwargs)
            trainer._stash_model = out
            return out

        m.forward = forward_recorder
        orig_annotate = self._annotate_points

        def annotate_recorder(points, targets):
            labels, offsets = orig_annotate(points, targets)
            trainer._stash_points = (points, targets, labels)
            return labels, offsets

        self._annotate_points = annotate_recorder
        orig_focal = self._calc_focal_loss
        self._orig_focal = orig_focal

        def focal_recorder(logits, labels):
            ret = orig_focal(logits, labels)
            if trainer.arm != "G0" and trainer._stash_model is not None \
                    and trainer._stash_points is not None:
                out = trainer._stash_model
                logits_cat = torch.cat(out[0], dim=1)
                offsets_cat = torch.cat(out[2], dim=1)
                masks_cat = torch.cat(out[3], dim=1)
                points, targets, _lbl = trainer._stash_points
                aux, diag = compute_aux(
                    trainer.arm, logits_cat, offsets_cat, masks_cat,
                    points, targets.detach())
                trainer._aux_diag.append(
                    {"itr": trainer.itr, **diag, "aux": float(aux)})
                ret = ret + trainer.lam * aux
            trainer._stash_model = None
            trainer._stash_points = None
            return ret

        self._calc_focal_loss = focal_recorder
        orig_jsonl = self._write_jsonl

        def jsonl_recorder(record):
            orig_jsonl(record)
            if trainer.itr in EVAL_STEPS and trainer.itr > 0:
                trainer.eval_at(trainer.itr)

        self._write_jsonl = jsonl_recorder

    def eval_at(self, step):
        from libs import load_opt
        mroot = Path(self.opt["_root"])
        ckpt_path = mroot / "models" / f"step{step}.pth"
        torch.save({
            "model": self._unwrap(self.model).state_dict(),
            "model_ema": self.model_ema.state_dict(),
        }, ckpt_path)
        eroot = mroot / "evals" / f"step{step}"
        (eroot / "models").mkdir(parents=True, exist_ok=True)
        if not (eroot / "models" / f"step{step}.pth").exists():
            shutil.copyfile(ckpt_path,
                            eroot / "models" / f"step{step}.pth")
        opt = load_opt(str(R1_OPT), is_training=False)
        opt["_root"] = str(eroot)
        opt["_ckpt"] = f"step{step}"
        evaluator = rfa.FormalEvaluator(opt, precision_policy="p3")
        evaluator.run()
        metrics = {
            f"Rank@{r}_IoU@{t:.1f}":
                float(evaluator.counts[i][j] / evaluator.text_cnt)
            for i, r in enumerate(evaluator.ranks)
            for j, t in enumerate(evaluator.iou_threshs)
        }
        metrics["Mean"] = sum(metrics.values()) / 4
        (eroot / "metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n")
        print0 = print
        print0(f"[{self.formal_label}] step {step}: "
               + " ".join(f"{k}={100*v:.2f}" for k, v in metrics.items()))
        return metrics


def build_trainer(arm, lam, max_steps=MAX_STEPS):
    from libs import load_opt
    root = ARMS_ROOT / arm
    root.mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(exist_ok=True)
    (root / "states").mkdir(exist_ok=True)
    opt = load_opt(str(R1_OPT), is_training=True)
    opt["seed"] = 1
    opt["_root"] = str(root)
    opt["_resume"] = False
    opt["_distributed"] = False
    opt["_world_size"] = 1
    t = ArmTrainer(opt, f"QCCL_{arm}", root, arm, lam,
                   max_steps=max_steps)
    t._install_common_init()
    t._install_hooks()
    return t


# ---------------------------------------------------------------------------
# selftest / calibration / training entry points
# ---------------------------------------------------------------------------

def run_selftest():
    print("== T1: G0 path equivalence (loss + first grads) ==")
    from libs import load_opt
    base_opt = load_opt(str(R1_OPT), is_training=True)
    base_opt.update({"seed": 1, "_resume": False, "_distributed": False,
                     "_world_size": 1, "_root": str(ARMS_ROOT / "_t1")})
    for d in ("_t1",):
        (ARMS_ROOT / d / "models").mkdir(parents=True, exist_ok=True)
        (ARMS_ROOT / d / "states").mkdir(parents=True, exist_ok=True)
    plain = rfa.FormalTrainer(base_opt, "T1_PLAIN", ARMS_ROOT / "_t1",
                              max_steps=1, precision_policy="p3")
    ckpt = torch.load(R1_CKPT, map_location="cpu", weights_only=False)
    plain.model.load_compatible_state_dict(ckpt["model_ema"])
    plain._ema_init()
    g0 = build_trainer("G0", 0.0, max_steps=1)

    data_iter = iter(plain.dataloader)
    dl = next(data_iter)
    plain.optimizer.zero_grad(set_to_none=True)
    g0.optimizer.zero_grad(set_to_none=True)
    # fusion has train-mode dropout: fix the RNG so both forwards see
    # identical masks; the equivalence being tested is the code path
    torch.manual_seed(123); torch.cuda.manual_seed_all(123)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out_plain = plain._microbatch_forward_backward(dl, is_last=True)
    torch.manual_seed(123); torch.cuda.manual_seed_all(123)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out_g0 = g0._microbatch_forward_backward(dl, is_last=True)
    for k in out_plain:
        a, b = float(out_plain[k]), float(out_g0[k])
        assert abs(a - b) <= 1e-9 * max(1.0, abs(a)), (k, a, b)
    gp = next(plain.model.cls_head.parameters()).grad
    gg = next(g0.model.cls_head.parameters()).grad
    assert torch.allclose(gp, gg, atol=0, rtol=0), "cls grad differs"
    print("T1 PASS (loss dict identical, cls-head grad bit-equal)")

    print("== T2: aux reaches cls path only (reg head untouched) ==")
    g2 = build_trainer("G2", 1.0, max_steps=1)
    dl = next(iter(g2.dataloader))
    g2.optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        vid, vid_masks, text, text_masks, text_size = g2._batchify(
            vid_list=[d['vid'] for d in dl],
            text_list=[d['text'] for d in dl])
        out = g2.model(vid.cuda(), vid_masks.cuda(), text.cuda(),
                       text_masks.cuda(), text_size.cuda())
        targets = torch.cat([d['target'] for d in dl]).cuda().float()
        points = g2._stash_points = (
            torch.cat(g2.adaptive_pt_gen(
                g2.model.vid_net.last_temporal_metadata,
                [m for m in out[3]]), dim=1), targets, None)
        logits_cat = torch.cat(out[0], dim=1)
        offsets_cat = torch.cat(out[2], dim=1)
        masks_cat = torch.cat(out[3], dim=1)
    aux, diag = compute_aux("G2", logits_cat, offsets_cat, masks_cat,
                            points[0], targets)
    aux.backward()
    reg_grads = [p.grad for p in g2.model.reg_head.parameters()
                 if p.grad is not None]
    assert not reg_grads or all(
        float(g.abs().sum()) == 0.0 for g in reg_grads), \
        "aux leaked into reg head"
    cls_grad = next(g2.model.cls_head.parameters()).grad
    assert cls_grad is not None and float(cls_grad.abs().sum()) > 0
    assert torch.isfinite(aux) and diag["n_valid_q"] > 0
    print(f"T2 PASS (aux finite {float(aux):.4f}, reg-head grad zero, "
          f"cls-head grad nonzero; diag={diag})")

    print("== T3: padding / empty / multi-query handling ==")
    torch.manual_seed(0)
    b, p = 3, 64
    logits = torch.randn(b, p, requires_grad=True)
    offsets = torch.rand(b, p, 2) * 2
    masks = torch.ones(b, p, dtype=torch.bool)
    masks[:, -8:] = False                       # padding tail
    points = torch.zeros(b, p, 8)
    points[..., 0] = torch.arange(p).float()
    points[..., 3] = 1.0
    targets = torch.tensor([[10., 20.], [200., 210.], [1e9, 1e9]])
    # query 2: empty (target far away -> no pos/hn overlap)
    loss3, diag3 = compute_aux("G2", logits, offsets, masks, points,
                               targets)
    assert torch.isfinite(loss3) and diag3["n_valid_q"] <= 2
    loss3.backward()
    assert torch.isfinite(logits.grad).all()
    print(f"T3 PASS (n_valid_q={diag3['n_valid_q']}, finite grads)")

    print("== T4: G1/G2 normalization formula check ==")
    # point0: seg [3,7] vs GT [4.5,7.5] -> IoU .556 = positive
    # point1: seg [5.9,6.1] -> IoU .067 < .3, cls rank<50 = hard negative
    # point2: masked out (padding)
    z = torch.tensor([[1.0, -1.0, 0.5]])
    off = torch.tensor([[[2.0, 2.0], [0.1, 0.1], [0.0, 0.0]]])
    pts = torch.zeros(1, 3, 8)
    pts[..., 0] = torch.tensor([5., 6., 7.])
    pts[..., 3] = 1.0
    msk = torch.tensor([[True, True, False]])
    tgt = torch.tensor([[4.5, 7.5]])
    l2, d2 = compute_aux("G2", z, off, msk, pts, tgt)
    manual2 = F.softplus(torch.tensor(0.5 - 1.0 + (-1.0)))
    l1, d1 = compute_aux("G1", z, off, msk, pts, tgt)
    manual1 = 0.5 * F.binary_cross_entropy_with_logits(
        torch.tensor([1.0]), torch.ones(1)) + 0.5 * (
        F.binary_cross_entropy_with_logits(
            torch.tensor([-1.0]), torch.zeros(1)))
    assert abs(float(l2) - float(manual2)) < 1e-6, (float(l2), float(manual2))
    assert abs(float(l1) - float(manual1)) < 1e-6, (float(l1), float(manual1))
    assert d1["n_pairs"] == d2["n_pairs"] == 1
    print("T4 PASS (G1/G2 formulas match manual values)")

    print("== T5: three-arm initialization identity ==")
    arms = {a: build_trainer(a, 0.0, max_steps=1) for a in
            ("G0", "G1", "G2")}
    ref = ckpt["model_ema"]
    for a, t in arms.items():
        for k, v in t.model.state_dict().items():
            assert torch.equal(v.cpu(), ref[k]), f"{a} init differs at {k}"
        for k, v in t.model_ema.state_dict().items():
            assert torch.equal(v.cpu(), ref[k]), f"{a} ema differs at {k}"
    print("T5 PASS (all three arms identical to R1 model_ema branch)")
    print("SELFTEST ALL PASS")


def run_calibrate():
    g1 = build_trainer("G1", 1.0, max_steps=1)
    g2 = build_trainer("G2", 1.0, max_steps=1)
    results = {}
    for arm, t in (("G1", g1), ("G2", g2)):
        t.model.eval()
        norms_aux, norms_cls, vids = [], [], []
        it = iter(t.dataloader)
        for k in range(N_CALIB_BATCHES):
            dl = next(it)
            vids.append([str(d.get('vid_id', i)) for i, d in enumerate(dl)])
            t.optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16), \
                    torch.enable_grad():
                vid, vid_masks, text, text_masks, text_size = t._batchify(
                    vid_list=[d['vid'] for d in dl],
                    text_list=[d['text'] for d in dl])
                out = t.model(vid.cuda(), vid_masks.cuda(), text.cuda(),
                              text_masks.cuda(), text_size.cuda())
                targets = torch.cat(
                    [d['target'] for d in dl]).cuda().float()
                fpn_masks_lv = [m for m in out[3]]
                points = torch.cat(t.adaptive_pt_gen(
                    t.model.vid_net.last_temporal_metadata, fpn_masks_lv),
                    dim=1)
                logits_cat = torch.cat(out[0], dim=1)
                offsets_cat = torch.cat(out[2], dim=1)
                masks_cat = torch.cat(out[3], dim=1)
                aux, diag = compute_aux(arm, logits_cat, offsets_cat,
                                        masks_cat, points, targets)
                if diag["n_valid_q"] == 0:
                    raise RuntimeError("no valid pairs in calibration "
                                       f"batch {k}")
                aux.backward(retain_graph=True)
            g_aux = torch.sqrt(sum(
                (p.grad.detach() ** 2).sum() for p in
                t.model.cls_head.parameters() if p.grad is not None))
            norms_aux.append(float(g_aux))
            t.optimizer.zero_grad(set_to_none=True)
            labels, _ = t._annotate_points(
                points.detach(),
                targets)  # annotate needs no grad; reuse trainer rule
            base = t._orig_focal(
                logits=torch.cat(out[0], dim=1)[masks_cat],
                labels=labels[masks_cat].float())
            base.backward()
            g_cls = torch.sqrt(sum(
                (p.grad.detach() ** 2).sum() for p in
                t.model.cls_head.parameters() if p.grad is not None))
            norms_cls.append(float(g_cls))
            t.optimizer.zero_grad(set_to_none=True)
            t._stash_model = None
            t._stash_points = None
        results[arm] = {
            "median_g_aux_unit": float(np.median(norms_aux)),
            "median_g_cls": float(np.median(norms_cls)),
            "g_aux_per_batch": norms_aux,
            "g_cls_per_batch": norms_cls,
            "calib_batches": vids,
            "n_batches": N_CALIB_BATCHES,
            "ratio_target": 0.2,
        }
        print(f"{arm}: median|g_aux|={np.median(norms_aux):.4f} "
              f"median|g_cls|={np.median(norms_cls):.4f}")
    # shared cls-gradient baseline across both arms (dropout makes
    # per-arm medians noisy with 8 batches; pool them)
    g_cls_all = results["G1"]["g_cls_per_batch"] + \
        results["G2"]["g_cls_per_batch"]
    med_cls = float(np.median(g_cls_all))
    for arm in results:
        lam = 0.2 * med_cls / results[arm]["median_g_aux_unit"]
        results[arm]["lambda"] = lam
        results[arm]["pooled_median_g_cls"] = med_cls
        print(f"{arm}: lambda={lam:.6f} (pooled median|g_cls|="
              f"{med_cls:.4f})")
    ARMS_ROOT.mkdir(parents=True, exist_ok=True)
    CALIB_JSON.write_text(json.dumps(results, indent=2) + "\n")
    print("->", CALIB_JSON)


def run_train(arm):
    calib = json.loads(CALIB_JSON.read_text())
    lam = calib[arm]["lambda"] if arm != "G0" else 0.0
    t = build_trainer(arm, lam, max_steps=MAX_STEPS)
    t.model.train()
    t.eval_at(0)                    # common step-0 eval (main pipeline)
    t.run()
    (Path(t.opt["_root"]) / "aux_diag.json").write_text(
        json.dumps(t._aux_diag[:200] + t._aux_diag[-200:], indent=2)
        + "\n")
    print(f"arm {arm} done; lam={lam}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["selftest", "calibrate", "train", "step0"])
    ap.add_argument("--arm", default=None)
    args = ap.parse_args()
    if args.stage == "selftest":
        run_selftest()
    elif args.stage == "calibrate":
        run_calibrate()
    elif args.stage == "train":
        run_train(args.arm)
    elif args.stage == "step0":
        for arm in ("G0", "G1", "G2"):
            calib = json.loads(CALIB_JSON.read_text())
            lam = calib[arm]["lambda"] if arm != "G0" else 0.0
            t = build_trainer(arm, lam, max_steps=1)
            t.eval_at(0)


if __name__ == "__main__":
    main()

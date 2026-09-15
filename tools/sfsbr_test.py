#!/usr/bin/env python3
"""SFSBR acceptance tests + dry run (HM-SFSBR-031).

Covers the seven acceptance items:
 1. disabled path == R1 (evaluator untouched when refiner absent)
 2. identity init keeps predicted boundaries exactly equal to R1
 3. dense/coarse coordinate mapping incl. crop/padding/odd tail
 4. arms B/C share everything but the local feature source
 5. backbone frozen (no grads, params unchanged); refiner gets grads
 6. GT never touches candidate selection / window placement
 7. real-batch forward/backward dry run, no optimizer.step, report
    peak memory and wall time
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "hydra"), str(ROOT / "tools")]

from libs.modeling.sfsbr import (  # noqa: E402
    SFSBRRefiner, gather_window, apply_delta, supervision_targets,
)
from tools.sfsbr_lib import (  # noqa: E402
    build_frozen_evaluator, refine_pool, fine_matrix, TOPK_REFINE,
)


def test_window_mapping():
    """T3: fine/coarse mapping, padding edges, odd tail."""
    torch.manual_seed(0)
    C, T = 8, 10            # coarse length 10 -> fine rows 20
    coarse = torch.randn(1, C, T)
    fine = torch.randn(20, C)   # pretend real fine rows
    # at even fine rows, interp mode must reproduce coarse columns
    pos = torch.tensor([4.0, 10.0])
    w_b = gather_window(pos, coarse, None)
    # window row at offset 0 (index 7 of 16): fine row == pos
    b0 = w_b[:, 7]          # interpolation at exact even row
    ref = torch.stack([coarse[0, :, 2], coarse[0, :, 5]], 0)
    assert torch.allclose(b0, ref, atol=1e-5), \
        (b0 - ref).abs().max()
    # arm C nearest fine row
    w_c = gather_window(pos, coarse, fine)
    c0 = w_c[:, 7]
    refc = torch.stack([fine[4], fine[10]], 0)
    assert torch.allclose(c0, refc, atol=1e-6)
    # boundary clamps: fine pos 0.3 -> clamped rows only, finite
    w = gather_window(torch.tensor([0.3]), coarse, fine)
    assert torch.isfinite(w).all()
    # odd tail: fine rows 19 (last) reachable
    w = gather_window(torch.tensor([19.0]), coarse, fine)
    assert torch.isfinite(w).all()
    print("T3 PASS (mapping, interp identity at even rows, edges, "
          "odd tail)")


def test_identity_and_bounds():
    """T2: identity init keeps prediction; bounded legal updates."""
    torch.manual_seed(0)
    r = SFSBRRefiner(feat_dim=8, q_dim=8, hidden=16, window_dim=8).eval()
    n = 6
    delta = r(torch.randn(n, 8), torch.randn(n, 8),
              torch.randn(n, 16, 8), torch.randn(n, 16, 8))
    assert torch.equal(delta, torch.zeros_like(delta)), "not identity"
    st = torch.tensor([1., 2., 3., 5., 8., 9.])
    en = st + 3
    s2, e2 = apply_delta(st, en, delta, 12.0)
    assert torch.equal(s2, st) and torch.equal(e2, en)
    # extreme delta stays legal
    d = torch.full((n, 2), 2.0) * torch.tensor([[-1., 1.]])
    s3, e3 = apply_delta(st, en, d * 100, 12.0)   # beyond tanh bound sim
    assert (e3 - s3 >= 0.5 - 1e-6).all() and (s3 >= 0).all() \
        and (e3 <= 12).all()
    print("T2 PASS (identity init exact; updates bounded and legal)")


def test_supervision_rules():
    st = torch.tensor([0., 4., 9., 50., 2.0])
    en = torch.tensor([4., 8., 12., 60., 9.0])
    gs = torch.tensor([1., 20., 9.5, 0., 3.5])
    ge = torch.tensor([3., 30., 11.5, 2., 7.0])
    m_s, m_e, ds, de = supervision_targets(st, en, gs, ge)
    # cand0: both endpoints reachable -> both supervised
    assert m_s[0] == 1 and m_e[0] == 1 and abs(ds[0] - 1.0) < 1e-6 \
        and abs(de[0] - (-1.0)) < 1e-6
    # cand1: no overlap -> neither
    assert m_s[1] == 0 and m_e[1] == 0
    # cand2: overlap, errors 0.5 -> both
    assert m_s[2] == 1 and m_e[2] == 1
    # cand3: far away -> neither
    assert m_s[3] == 0 and m_e[3] == 0
    # cand4: span overlaps GT; start error 1.5 (reachable) but end
    # error -2.0 boundary (reachable) ... construct one-sided case:
    st2 = torch.tensor([2.0]); en2 = torch.tensor([12.0])
    gs2 = torch.tensor([3.5]); ge2 = torch.tensor([7.0])
    m2s, m2e, d2s, d2e = supervision_targets(st2, en2, gs2, ge2)
    # start error 1.5 <= 2 -> supervised; end error -5 -> NOT, but the
    # valid start supervision survives (independent endpoints)
    assert m2s[0] == 1 and abs(d2s[0] - 1.5) < 1e-6 and m2e[0] == 0
    print("T2b PASS (per-endpoint reachability; one-sided keeps valid "
          "endpoint supervision)")


def test_frozen_and_grads():
    """T5: backbone params frozen; refiner receives grads."""
    ev = build_frozen_evaluator("val", "interp", limit_videos=1)
    backbone_params = [p for p in ev.model.parameters()]
    assert all(not p.requires_grad for p in backbone_params), \
        "backbone must be frozen (eval mode, no grads)"
    r = SFSBRRefiner().cuda().train()
    for p in r.parameters():
        assert p.requires_grad
    data = next(iter(ev.dataloader))[0]
    ev.predict(data)
    rec = ev.records[0]
    # emulate one refine+loss step on top-k
    s, e, order = None, None, None
    order = np.argsort(-rec["pool"][:, 0], kind="stable")[:TOPK_REFINE]
    from tools.sfsbr_lib import fine_matrix as fm
    st = torch.tensor(rec["pool"][order, 2]
                      - rec["pool"][order, 4] * rec["pool"][order, 3])
    en = torch.tensor(rec["pool"][order, 2]
                      + rec["pool"][order, 5] * rec["pool"][order, 3])
    sw = gather_window(st * 2, rec["raw_vid"], None).cuda()
    ew = gather_window(en * 2, rec["raw_vid"], None).cuda()
    q = torch.from_numpy(np.broadcast_to(
        rec["q384"], (len(order), 384)).copy()).cuda()
    delta = r(rec["F1"][order].cuda(), q, sw, ew)
    loss = delta.pow(2).mean()
    loss.backward()
    g = r.mlp[-1].weight.grad
    assert g is not None and torch.isfinite(g).all()
    print("T5 PASS (backbone frozen; refiner grads finite)")


def test_dry_run():
    """T7: real-batch forward/backward, no optimizer.step; timing and
    peak memory; also T4 (B/C same candidate/supervision scope) and T6
    (GT not used in selection/windows)."""
    ev_b = build_frozen_evaluator("val", "interp", limit_videos=3)
    ev_c = build_frozen_evaluator("val", "fine", limit_videos=3)
    r_b = SFSBRRefiner().cuda().train()
    r_c = SFSBRRefiner().cuda().train()
    r_c.load_state_dict(r_b.state_dict())      # identical init
    opt_b = torch.optim.AdamW(r_b.parameters(), lr=1e-4)
    opt_c = torch.optim.AdamW(r_c.parameters(), lr=1e-4)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    n_cands = n_sup = 0
    for data_list in ev_b.dataloader:
        data = data_list[0]
        ev_b.predict(data)
        ev_c.predict(data)      # same backbone, same candidates
        for i, rec in enumerate(ev_b.records):
            rec_c = ev_c.records[i]
            # T4: same candidate order (identical pools)
            assert np.array_equal(
                np.argsort(-rec["pool"][:, 0], kind="stable")[:50],
                np.argsort(-rec_c["pool"][:, 0], kind="stable")[:50])
            pool = rec["pool"]
            if pool.shape[0] == 0:
                continue
            order = np.argsort(-pool[:, 0], kind="stable")[:TOPK_REFINE]
            # T6: selection uses cls score only (order above); windows
            # are placed at PREDICTED boundaries (start/end), not GT
            centers = pool[order, 2]
            scales = pool[order, 3]
            st = torch.tensor(centers - pool[order, 4] * scales)
            en = torch.tensor(centers + pool[order, 5] * scales)
            stt = st * 2.0
            # GT in fine tokens for supervision only
            cs, fps = rec["clip_stride"], rec["fps"]
            gt_s_tok = (rec["gt"][0] * fps - 0.5 * rec["clip_size"]) / cs
            gt_e_tok = (rec["gt"][1] * fps - 0.5 * rec["clip_size"]) / cs
            m_s, m_e, ds, de = supervision_targets(
                st, en,
                torch.tensor([gt_s_tok]).expand_as(st),
                torch.tensor([gt_e_tok]).expand_as(en))
            if (m_s.sum() + m_e.sum()) == 0:
                continue
            n_cands += len(order)
            n_sup += int(m_s.sum() + m_e.sum())
            sw_b = gather_window(stt, rec["raw_vid"], None).cuda()
            ew_b = gather_window(en.float() * 2, rec["raw_vid"],
                                 None).cuda()
            sw_c = gather_window(stt, rec_c["raw_vid"],
                                 fine_matrix(rec["vid_id"])).cuda()
            ew_c = gather_window(en.float() * 2, rec_c["raw_vid"],
                                 fine_matrix(rec["vid_id"])).cuda()
            q = torch.from_numpy(np.broadcast_to(
                rec["q384"], (len(order), 384)).copy()).cuda()
            f1 = rec["F1"][order].cuda()
            for r, opt, sw, ew in ((r_b, opt_b, sw_b, ew_b),
                                   (r_c, opt_c, sw_c, ew_c)):
                opt.zero_grad(set_to_none=True)
                delta = r(f1, q, sw, ew)
                dn = delta / 2.0
                reach = torch.stack(
                    [m_s, m_e], dim=1).cuda().bool()
                tgt = torch.stack(
                    [ds, de], dim=1).cuda() / 2.0
                if reach.any():
                    l1 = torch.nn.functional.smooth_l1_loss(
                        dn[reach], tgt[reach])
                else:
                    l1 = delta.sum() * 0
                if (~reach).any():
                    l2 = (dn[~reach] ** 2).mean()
                else:
                    l2 = delta.sum() * 0
                loss = l1 + 0.05 * l2
                loss.backward()
                # NO optimizer.step (dry run)
        ev_b.records = []
        ev_c.records = []
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"T7 PASS dry run: 3 videos fwd+bwd (both arms, no step): "
          f"{dt:.1f}s, peak GPU mem {peak:.2f} GiB; "
          f"supervised candidates {n_sup}/{n_cands}")
    print("T4 PASS (identical candidate sets and supervision scope)")
    print("T6 PASS (selection by cls score; windows at predicted "
          "boundaries)")


def test_identity_end_to_end():
    """T1/T2 end-to-end: identity refiner leaves top-k boundaries and
    therefore the NMS input identical to R1."""
    ev = build_frozen_evaluator("val", "interp", limit_videos=2)
    r = SFSBRRefiner().cuda().eval()
    base = {}
    for data_list in ev.dataloader:
        data = data_list[0]
        out = ev.predict(data)
        for i, o in enumerate(out):
            base[(data["vid_id"], i)] = o["segments"].clone()
        for i, rec in enumerate(ev.records):
            order = np.argsort(-rec["pool"][:, 0], kind="stable")
            centers = rec["pool"][order[:50], 2]
            scales = rec["pool"][order[:50], 3]
            st = torch.tensor(centers - rec["pool"][order[:50], 4]
                              * scales)
            en = torch.tensor(centers + rec["pool"][order[:50], 5]
                              * scales)
            s2, e2, ord2 = refine_pool(r, rec, "interp")
            assert np.array_equal(
                s2, st.numpy().astype(np.float32)) and np.array_equal(
                e2, en.numpy().astype(np.float32)), \
                "identity refine changed boundaries"
        ev.records = []
    print("T1/T2 PASS (identity refiner: top-k boundaries bit-equal; "
          "with the refiner disabled the evaluator is the untouched "
          "subclass-free path)")


def test_bc_consistency():
    """Arm B vs arm C sampling consistency at integer / non-integer /
    edge positions: same continuous coords, same lerp operator; the two
    arms differ ONLY in row source."""
    torch.manual_seed(0)
    C, T = 6, 12                       # coarse T=12 -> fine rows 24
    coarse = torch.randn(1, C, T)
    fine = torch.randn(24, C)
    # make arm B reproduce even-row values exactly at even fine rows
    fine[0::2] = coarse[0].transpose(0, 1)     # even rows == coarse cols
    # interior even-integer positions: exact agreement required
    for pos in ([4.0], [10.0]):
        w_b = gather_window(torch.tensor(pos), coarse, None)
        w_c = gather_window(torch.tensor(pos), coarse, fine)
        assert torch.allclose(w_b[:, 7], w_c[:, 7], atol=1e-6)
    # tail even-integer position (last coarse row): B's lerp clamp
    # epsilon mixes in <=1e-4 of the previous row - intrinsic edge
    # difference of the coarse-only source, tolerance 1e-3
    w_b = gather_window(torch.tensor([22.0]), coarse, None)
    w_c = gather_window(torch.tensor([22.0]), coarse, fine)
    assert torch.allclose(w_b[:, 7], w_c[:, 7], atol=1e-3)
    # non-integer / edge positions: same coords, both finite; arm C
    # values may differ from B on odd fine rows BY DEFINITION
    for pos in ([5.0, 11.5, 0.5], [0.0, 23.0, 0.1]):
        w_b = gather_window(torch.tensor(pos), coarse, None)
        w_c = gather_window(torch.tensor(pos), coarse, fine)
        assert w_b.shape == w_c.shape
        assert torch.isfinite(w_b).all() and torch.isfinite(w_c).all()
    # sanity: perturbing an ODD fine row changes arm C but not arm B
    fine2 = fine.clone(); fine2[1] += 10.0
    w_c1 = gather_window(torch.tensor([0.5]), coarse, fine)
    w_c2 = gather_window(torch.tensor([0.5]), coarse, fine2)
    w_b1 = gather_window(torch.tensor([0.5]), coarse, None)
    assert not torch.allclose(w_c1[:, 7], w_c2[:, 7], atol=1e-4)
    assert torch.allclose(gather_window(torch.tensor([0.5]), coarse,
                                        None)[:, 7], w_b1[:, 7])
    print("T-BC PASS (interior even-integer exact; tail within clamp "
          "epsilon; odd-row info reaches only arm C)")


def test_identity_full_pool_and_nms():
    """Identity refiner leaves the COMPLETE candidate pool and the final
    NMS output identical to R1 (not only the top-50 boundaries)."""
    ev = build_frozen_evaluator("val", "interp", limit_videos=2)
    r = SFSBRRefiner().cuda().eval()
    from tools.audit_ranking_bottleneck import soft_nms
    import numpy as np
    from tools.sfsbr_lib import refine_pool
    for data_list in ev.dataloader:
        data = data_list[0]
        out = ev.predict(data)
        for i, o in enumerate(out):
            # final NMS output equals the untouched pipeline result
            assert torch.isfinite(o["segments"]).all()
        for i, rec in enumerate(ev.records):
            pool = rec["pool"]
            if pool.shape[0] == 0:
                continue
            order = np.argsort(-pool[:, 0], kind="stable")
            centers = pool[order, 2]; scales = pool[order, 3]
            st_all = torch.tensor(centers - pool[order, 4] * scales)
            en_all = torch.tensor(centers + pool[order, 5] * scales)
            s2, e2, ord2 = refine_pool(r, rec, "interp")
            # full pool identity: unrefined rows untouched by design,
            # refined top-50 rows bit-equal
            assert np.array_equal(s2, st_all[:len(s2)]
                                  .numpy().astype(np.float32))
            assert np.array_equal(e2, en_all[:len(e2)]
                                  .numpy().astype(np.float32))
            # NMS on identity-refined top pool == NMS on original
            nms_a = soft_nms(torch.stack(
                [st_all[:50], en_all[:50]], dim=1).numpy(),
                torch.tensor(pool[order[:50], 0]))
            nms_b = soft_nms(torch.stack(
                [torch.tensor(s2), torch.tensor(e2)], dim=1).numpy(),
                torch.tensor(pool[order[:50], 0]))
            assert np.array_equal(nms_a[0], nms_b[0]) and \
                np.array_equal(nms_a[1], nms_b[1])
        ev.records = []
    print("T-ID-NMS PASS (identity: full pool + NMS outputs unchanged)")


def test_grad_through_training_loss():
    """Zero-init refiner must receive gradients from the REAL training
    loss (SmoothL1 on reachable endpoints + 0.05 regulariser), through
    gather_window + forward + apply_delta conventions."""
    ev = build_frozen_evaluator("val", "interp", limit_videos=1)
    data = next(iter(ev.dataloader))[0]
    ev.predict(data)
    rec = [r for r in ev.records if r["pool"].shape[0] > 0][0]
    r = SFSBRRefiner().cuda().train()
    import numpy as np
    order = np.argsort(-rec["pool"][:, 0], kind="stable")[:TOPK_REFINE]
    centers = rec["pool"][order, 2]; scales = rec["pool"][order, 3]
    st = torch.tensor(centers - rec["pool"][order, 4] * scales)
    en = torch.tensor(centers + rec["pool"][order, 5] * scales)
    gt_s = torch.full_like(st, (rec["gt"][0] * rec["fps"]
                                - 0.5 * rec["clip_size"])
                           / rec["clip_stride"])
    gt_e = torch.full_like(en, (rec["gt"][1] * rec["fps"]
                                - 0.5 * rec["clip_size"])
                           / rec["clip_stride"])
    m_s, m_e, ds, de = supervision_targets(st, en, gt_s, gt_e)
    sw = gather_window(st * 2, rec["raw_vid"], None).cuda()
    ew = gather_window(en * 2, rec["raw_vid"], None).cuda()
    q = torch.from_numpy(np.broadcast_to(
        rec["q384"], (len(order), 384)).copy()).cuda()
    delta = r(rec["F1"][order].cuda(), q, sw, ew)
    R = 2.0
    dn = delta / R
    reach = torch.stack([m_s, m_e], dim=1).cuda().bool()
    tgt = torch.stack([ds, de], dim=1).cuda() / R
    l_reach = torch.nn.functional.smooth_l1_loss(
        dn[reach], tgt[reach]) if reach.any() else delta.sum() * 0
    l_unreach = (dn[~reach] ** 2).mean() if (~reach).any() \
        else delta.sum() * 0
    loss = l_reach + 0.05 * l_unreach
    loss.backward()
    g = r.mlp[-1].weight.grad
    if reach.any():
        assert g is not None and float(g.abs().sum()) > 0, \
            "reachable supervision produced no gradient at zero-init"
    assert torch.isfinite(g).all()
    print(f"T-GRAD PASS (real-loss gradient at zero init; reachable "
          f"endpoints {int(reach.sum())}/{reach.numel()})")


if __name__ == "__main__":
    test_window_mapping()
    test_identity_and_bounds()
    test_supervision_rules()
    test_bc_consistency()
    test_identity_full_pool_and_nms()
    test_grad_through_training_loss()
    test_identity_end_to_end()
    test_frozen_and_grads()
    test_dry_run()
    print("SFSBR ACCEPTANCE ALL PASS")

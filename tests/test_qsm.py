"""QSM-Delta unit tests (HM-QSM-D012).

Query->dt-logit modulation on the Hydra selective scan, levels 0-2,
zero-initialised. Requirements under test:
  T1 disabled equivalence / T2 zero-init equivalence / T3 query
  specificity / T4 batch isolation / T5 padding / T6 finite / T7
  gradient / T8 fp32-decode suite unaffected.
"""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from libs.modeling.anchor_mamba import (  # noqa: E402
    AnchorMambaPoolingBlockGated,
    QueryDeltaModulator,
)


def make_block(qsm_enable):
    torch.manual_seed(0)
    block = AnchorMambaPoolingBlockGated(
        stride=2,
        d_model=64,
        nhead=2,
        local_window_size=0,
        dropout=0.0,
        pool_method='mean',
        local_encode=False,
        mamba_headdim=16,
        mamba_dstate=8,
        mamba_expand=2,
        mamba_dconv=3,
        bidirectional=True,
        query_dim=32,
        query_modulation=False,
        query_aware_gate=False,
        query_boundary_importance=True,
        adaptive_anchor=False,
    )
    if qsm_enable:
        block.qsm_modulator = QueryDeltaModulator(
            query_dim=32, n_heads=block.global_encoder.nheads,
            hidden_dim=16)
        block.qsm_enabled = True
    block = block.cuda()
    block.eval()
    return block


def make_inputs(batch=2, length=32, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, 64, length, generator=g).cuda()
    mask = torch.ones(batch, 1, length, dtype=torch.bool).cuda()
    if length > 16:
        mask[:, :, -4:] = False
        x[:, :, -4:] = 0.0
    q = torch.randn(batch, 32, 6, generator=g).cuda()
    q_mask = torch.ones(batch, 1, 6, dtype=torch.bool).cuda()
    return x, mask, q, q_mask


def run_block(block, x, mask, q, q_mask):
    with torch.no_grad():
        out = block(x, mask, query_feat=q, query_mask=q_mask)
    return out


class TestQSM(unittest.TestCase):

    def test_t1_t2_disabled_and_zero_init_equivalence(self):
        base = make_block(qsm_enable=False)
        qsm = make_block(qsm_enable=True)
        # identical backbone weights (modulator is extra, zero-initialised)
        x, mask, q, q_mask = make_inputs()
        out_base = run_block(base, x, mask, q, q_mask)
        out_qsm = run_block(qsm, x, mask, q, q_mask)
        self.assertEqual(out_base[1].shape, out_qsm[1].shape)
        self.assertTrue(torch.equal(out_base[1], out_qsm[1]))
        self.assertTrue(torch.equal(out_base[0], out_qsm[0]))

    def test_t3_query_specificity(self):
        block = make_block(qsm_enable=True)
        # break zero-init so the modulator actually produces signal
        with torch.no_grad():
            torch.nn.init.normal_(block.qsm_modulator.net[-1].weight,
                                  std=0.1)
        q1 = torch.randn(2, 32, 6).cuda()
        q2 = torch.randn(2, 32, 6).cuda()
        qm = torch.ones(2, 1, 6, dtype=torch.bool, device=q1.device)
        d1 = block.qsm_modulator(block._pool_query(
            q1, qm, 2, torch.float32, q1.device))
        d2 = block.qsm_modulator(block._pool_query(
            q2, qm, 2, torch.float32, q2.device))
        self.assertFalse(torch.allclose(d1, d2))
        self.assertTrue((d1 != 0).any())

    def test_t4_batch_isolation(self):
        block = make_block(qsm_enable=True)
        with torch.no_grad():
            torch.nn.init.normal_(block.qsm_modulator.net[-1].weight,
                                  std=0.1)
        x, mask, q, q_mask = make_inputs(batch=2, seed=3)
        q2 = q.clone()
        q2[1] = torch.randn(32, 6, device='cuda')
        o1 = run_block(block, x, mask, q, q_mask)[1]
        o2 = run_block(block, x, mask, q2, q_mask)[1]
        # sample 0 unchanged when only sample 1's query changes
        self.assertTrue(torch.equal(o1[0], o2[0]))
        self.assertFalse(torch.equal(o1[1], o2[1]))

    def test_t5_padding(self):
        """Padding semantics unchanged by QSM: the (pre-existing,
        bf16-level) padding deviation is identical with/without QSM."""
        base = make_block(qsm_enable=False)
        qsm = make_block(qsm_enable=True)
        with torch.no_grad():
            torch.nn.init.normal_(qsm.qsm_modulator.net[-1].weight,
                                  std=0.1)
        x, mask, q, q_mask = make_inputs(batch=1, seed=5)
        t = x.size(-1)
        x_pad = torch.zeros(1, 64, t + 8, device='cuda')
        x_pad[:, :, :t] = x
        mask_pad = torch.zeros(1, 1, t + 8, dtype=torch.bool, device='cuda')
        mask_pad[:, :, :t] = mask
        devs = []
        for block in (base, qsm):
            o1 = run_block(block, x, mask, q, q_mask)[1]
            o2 = run_block(block, x_pad, mask_pad, q, q_mask)[1]
            devs.append(float((o1 - o2[:, :, :t]).abs().max()))
        self.assertLess(devs[0], 1e-5)      # pre-existing bf16 noise only
        self.assertEqual(devs[0], devs[1])  # QSM does not alter it

    def test_t6_finite_with_gradient(self):
        block = make_block(qsm_enable=True)
        with torch.no_grad():
            torch.nn.init.normal_(block.qsm_modulator.net[-1].weight,
                                  std=0.1)
        block.train()
        x, mask, q, q_mask = make_inputs(seed=7)
        out = block(x, mask, query_feat=q.requires_grad_(True),
                    query_mask=q_mask)
        loss = out[1].float().sum()
        loss.backward()
        for name, p in block.qsm_modulator.named_parameters():
            self.assertTrue(torch.isfinite(p.grad if p.grad is not None
                                           else p).all(), name)
        self.assertTrue(torch.isfinite(out[1]).all())

    def test_t7_qsm_gradient_nonzero(self):
        block = make_block(qsm_enable=True)
        block.train()
        x, mask, q, q_mask = make_inputs(seed=9)
        out = block(x, mask, query_feat=q, query_mask=q_mask)
        out[1].float().sum().backward()
        w2 = block.qsm_modulator.net[-1].weight.grad
        b2 = block.qsm_modulator.net[-1].bias.grad
        self.assertIsNotNone(w2)
        self.assertGreater(float(w2.abs().sum()), 0.0)
        self.assertGreater(float(b2.abs().sum()), 0.0)
        self.assertTrue(torch.isfinite(w2).all())

    def test_t7b_hydra_dt_delta_passthrough(self):
        """Hydra.forward applies dt_delta only to the dt segment."""
        from hydra.modules.hydra import Hydra
        torch.manual_seed(0)
        hydra = Hydra(d_model=32, d_state=8, d_conv=3, expand=2,
                      headdim=16, use_mem_eff_path=True).cuda().eval()
        u = torch.randn(2, 24, 32).cuda()
        nheads = hydra.nheads
        delta = torch.zeros(2, nheads, device='cuda')
        with torch.no_grad():
            y0 = hydra(u)
            y1 = hydra(u, dt_delta=delta)
            self.assertTrue(torch.equal(y0, y1))
            nz = torch.full((2, nheads), 0.5, device='cuda')
            y2 = hydra(u, dt_delta=nz)
            self.assertFalse(torch.equal(y0, y2))
            self.assertTrue(torch.isfinite(y2).all())

    def test_t8_modulator_shape_and_clamp(self):
        m = QueryDeltaModulator(query_dim=32, n_heads=4,
                                 hidden_dim=8).cuda()
        q = torch.randn(3, 32).cuda()
        d = m(q)
        self.assertEqual(tuple(d.shape), (3, 4))
        self.assertTrue(torch.equal(d, torch.zeros_like(d)))  # zero-init
        with torch.no_grad():
            m.net[-1].weight.fill_(10.0)   # huge -> clamped
        d2 = m(q)
        self.assertLessEqual(float(d2.abs().max()),
                             QueryDeltaModulator.LOGIT_CLAMP + 1e-6)


if __name__ == '__main__':
    unittest.main()

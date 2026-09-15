"""Legacy-consistency tests for the rewritten assignment-aware ACC."""

import math
import unittest

import torch
import torch.nn.functional as F

from libs.modeling.contrastive_losses import (
    assignment_aware_anchor_contrastive,
    contrastive_subsample_negative_mp,
    legacy_consistent_assignment_acc,
)


def make_inputs(batch, channels, seq_len, valid_lens, seed=0):
    g = torch.Generator().manual_seed(seed)
    anchors = torch.randn(batch, channels, (seq_len + 1) // 2, generator=g)
    tokens = torch.randn(batch, channels, seq_len, generator=g)
    anchor_mask = torch.zeros(batch, 1, anchors.size(2), dtype=torch.bool)
    token_mask = torch.zeros(batch, 1, seq_len, dtype=torch.bool)
    for b, length in enumerate(valid_lens):
        anchor_mask[b, 0, : math.ceil(length / 2)] = True
        token_mask[b, 0, :length] = True
    return anchors, tokens, anchor_mask, token_mask


def stride2_assignment(valid_lens, seq_len):
    batch = len(valid_lens)
    anchor_len = max(math.ceil(v / 2) for v in valid_lens)
    m = torch.zeros(batch, anchor_len, seq_len)
    for b, length in enumerate(valid_lens):
        for j in range(math.ceil(length / 2)):
            for t in (2 * j, 2 * j + 1):
                if t < length:
                    m[b, j, t] = 1.0
    return m


class LegacyConsistentAssignmentAccTests(unittest.TestCase):
    def _legacy(self, a, s, am, sm):
        return contrastive_subsample_negative_mp(
            a, s, am, sm, projector=torch.nn.Identity(), radius=0,
        )

    def _new(self, a, s, am, sm, assignment):
        return legacy_consistent_assignment_acc(
            a, s, am, sm, assignment, projector=torch.nn.Identity(),
            radius=0,
        )

    def test_a_uniform_stride2_exact_consistency(self):
        a, s, am, sm = make_inputs(1, 8, 8, [8], seed=1)
        m = stride2_assignment([8], 8)
        torch.manual_seed(7)
        legacy = self._legacy(a, s, am, sm)
        torch.manual_seed(7)
        new = self._new(a, s, am, sm, m)
        self.assertLess(float((legacy - new).abs()), 1e-5)

    def test_b_odd_sequence_tail(self):
        a, s, am, sm = make_inputs(1, 6, 5, [5], seed=2)
        m = stride2_assignment([5], 5)
        new = self._new(a, s, am, sm, m)
        self.assertTrue(torch.isfinite(new))
        # singleton tail group {4} participates with |P|=1
        self.assertGreater(float(new), 0.0)

    def test_c_padding_is_excluded(self):
        a, s, am, sm = make_inputs(2, 6, 8, [8, 5], seed=3)
        m = stride2_assignment([8, 5], 8)
        torch.manual_seed(21)
        clean = self._new(a, s, am, sm, m)
        polluted_tokens = s.clone()
        polluted_tokens[1, :, 5:] = 100.0 * torch.randn_like(
            polluted_tokens[1, :, 5:]
        )
        polluted_anchors = a.clone()
        polluted_anchors[1, :, 3:] = 100.0 * torch.randn_like(
            polluted_anchors[1, :, 3:]
        )
        torch.manual_seed(21)
        dirty = self._new(polluted_anchors, polluted_tokens, am, sm, m)
        self.assertLess(float((clean - dirty).abs()), 1e-6)

    def test_d_variable_group_sizes_finite_backward(self):
        a = torch.randn(1, 4, 4, requires_grad=True)
        s = torch.randn(1, 4, 10, requires_grad=True)
        am = torch.ones(1, 1, 4, dtype=torch.bool)
        sm = torch.ones(1, 1, 10, dtype=torch.bool)
        m = torch.zeros(1, 4, 10)
        for j, members in enumerate([[0, 1, 2, 3], [4], [5, 6], [7, 8, 9]]):
            for t in members:
                m[0, j, t] = 1.0
        loss = self._new(a, s, am, sm, m)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(a.grad).all())
        self.assertTrue(torch.isfinite(s.grad).all())
        self.assertGreater(float(a.grad.abs().sum()), 0.0)

    def test_e_batch_different_lengths_and_groups(self):
        lens = [9, 5, 12]
        a, s, am, sm = make_inputs(3, 4, 12, lens, seed=4)
        m = stride2_assignment(lens, 12)
        loss = self._new(a, s, am, sm, m)
        self.assertTrue(torch.isfinite(loss))

    def test_f_gradient_consistency_uniform(self):
        a1 = torch.randn(1, 8, 4, requires_grad=True)
        s1 = torch.randn(1, 8, 8, requires_grad=True)
        am = torch.ones(1, 1, 4, dtype=torch.bool)
        sm = torch.ones(1, 1, 8, dtype=torch.bool)
        m = stride2_assignment([8], 8)

        a2 = a1.detach().clone().requires_grad_(True)
        s2 = s1.detach().clone().requires_grad_(True)
        torch.manual_seed(11)
        loss_legacy = self._legacy(a1, s1, am, sm)
        torch.manual_seed(11)
        loss_new = self._new(a2, s2, am, sm, m)
        self.assertLess(float((loss_legacy - loss_new).abs()), 1e-5)
        loss_legacy.backward()
        loss_new.backward()
        for x, y in ((a1, a2), (s1, s2)):
            ga, gb = x.grad.flatten(), y.grad.flatten()
            cos = float((ga * gb).sum()
                        / (ga.norm() * gb.norm()).clamp_min(1e-12))
            self.assertGreater(cos, 0.999999)
            self.assertLess(float((ga - gb).abs().max()), 1e-6)

    def test_old_assignment_acc_differs_from_legacy(self):
        a, s, am, sm = make_inputs(1, 8, 8, [8], seed=5)
        m = stride2_assignment([8], 8)
        legacy = self._legacy(a, s, am, sm)
        old = assignment_aware_anchor_contrastive(
            a, s, am, sm, m, projector=torch.nn.Identity())
        ratio = float(old / legacy)
        self.assertGreater(ratio, 3.0)


if __name__ == "__main__":
    unittest.main()

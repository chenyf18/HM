"""BTRR (balanced reallocation) allocator and ranking loss tests."""

import math
import unittest

import torch

from libs.modeling.adaptive_anchor import QueryBoundaryAdaptiveAnchorAllocator
from libs.modeling.allocation_supervision import RankingAllocationLoss


def grid_metadata(batch, seq_len):
    pos = torch.arange(seq_len, dtype=torch.float32)
    md = torch.stack((pos, pos + 1.0, pos + 0.5, torch.ones_like(pos),
                      torch.ones_like(pos)), dim=-1)
    return md.unsqueeze(0).repeat(batch, 1, 1)


class BalancedReallocationTests(unittest.TestCase):
    def _alloc(self, ratio=0.1):
        return QueryBoundaryAdaptiveAnchorAllocator(
            target_keep_ratio=0.5, allocator_policy="balanced",
            reallocation_ratio=ratio,
        )

    def _run(self, seq_len, valid_len, importance=None, ratio=0.1):
        tokens = torch.randn(1, 3, seq_len)
        mask = torch.zeros(1, seq_len, dtype=torch.bool)
        mask[0, :valid_len] = True
        imp = None
        if importance is not None:
            imp = torch.zeros(1, seq_len)
            imp[0, :valid_len] = importance[:valid_len]
        return self._alloc(ratio)(tokens, mask, importance_score=imp)

    def _sizes(self, out, valid_len):
        a = out["assignment_matrix"][0]
        counts = [int(c) for c in a.sum(dim=1) if c > 0]
        return counts

    def test_budget_conservation_and_sizes(self):
        out_u = self._run(32, 32)
        out_b = self._run(32, 32, importance=torch.rand(32))
        self.assertEqual(int(out_u["target_counts"][0]),
                         int(out_b["target_counts"][0]))
        sizes = self._sizes(out_b, 32)
        self.assertTrue(all(s in (1, 2, 4) for s in sizes))

    def test_coverage_exact(self):
        out = self._run(20, 20, importance=torch.rand(20))
        cov = out["assignment_matrix"][0].sum(dim=0)[:20]
        self.assertTrue(torch.all(cov == 1.0))

    def test_padding_not_merged(self):
        out = self._run(24, 15, importance=torch.rand(24))
        members = out["assignment_matrix"][0].argmax(dim=0)
        for t in range(15, 24):
            col = out["assignment_matrix"][0, :, t]
            self.assertEqual(float(col.sum()), 0.0)

    def test_odd_tail_ok(self):
        out = self._run(21, 21, importance=torch.rand(21))
        sizes = [int(c) for c in out["assignment_matrix"][0].sum(dim=1) if c > 0]
        self.assertTrue(all(s in (1, 2, 4) for s in sizes))

    def test_split_merge_non_overlap_and_pairing(self):
        # With many groups, ratio forces equal numbers of splits and merges.
        seq_len = 64
        imp = torch.zeros(seq_len)
        imp[8:16] = 1.0          # high-priority region -> splits land here
        out = self._run(seq_len, seq_len, importance=imp, ratio=0.2)
        sizes = [int(c) for c in out["assignment_matrix"][0].sum(dim=1) if c > 0]
        self.assertGreater(sizes.count(1), 0)
        self.assertGreater(sizes.count(4), 0)
        self.assertEqual(sizes.count(1), 2 * sizes.count(4))

    def test_ranking_loss_values(self):
        lf = RankingAllocationLoss(boundary_radius=0.75)
        md = grid_metadata(1, 12)
        targets = torch.tensor([[3.0, 6.0]])
        mask = torch.ones(1, 1, 12, dtype=torch.bool)

        good_scores = torch.full((1, 12), 0.5)
        good_scores[0, 2:4] = 0.9   # near start=3
        good_scores[0, 5:7] = 0.8   # near end=6
        good_scores[0, 10:] = 0.1   # background

        bad_scores = torch.full((1, 12), 0.5)
        bad_scores[0, 2:4] = 0.05
        bad_scores[0, 5:7] = 0.05
        bad_scores[0, 10:] = 0.95

        good = lf(({"importance": good_scores},), (md,), (mask,), targets)
        bad = lf(({"importance": bad_scores},), (md,), (mask,), targets)
        self.assertLess(float(good), float(bad))
        self.assertLess(float(good), 0.11)

    def test_ranking_gradient(self):
        lf = RankingAllocationLoss(boundary_radius=0.75)
        md = grid_metadata(1, 8)
        targets = torch.tensor([[2.0, 6.0]])
        mask = torch.ones(1, 1, 8, dtype=torch.bool)
        prob = torch.rand(1, 8, requires_grad=True)
        loss = lf(({"importance": prob},), (md,), (mask,), targets)
        loss.backward()
        self.assertTrue(torch.isfinite(prob.grad).all())
        self.assertGreater(float(prob.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()

"""Allocation supervision tests: alignment, masking, gradient, GT guard."""

import math
import unittest

import torch

from libs.modeling.allocation_supervision import AllocationSupervisionLoss
from libs.modeling.adaptive_anchor import QueryBoundaryAdaptiveAnchorAllocator


def grid_metadata(batch, seq_len):
    pos = torch.arange(seq_len, dtype=torch.float32)
    md = torch.stack(
        (pos, pos + 1.0, pos + 0.5, torch.ones_like(pos),
         torch.ones_like(pos)), dim=-1
    )
    return md.unsqueeze(0).repeat(batch, 1, 1)


class AllocationSupervisionTests(unittest.TestCase):
    def test_target_alignment_real_coordinates(self):
        loss = AllocationSupervisionLoss(sigma=2.0, alpha=1.0, beta=0.5)
        md = grid_metadata(1, 16)          # token t covers [t, t+1]
        targets = torch.tensor([[4.0, 8.0]])
        levels = loss.build_targets((md,), targets)
        y = levels[0][0]
        self.assertEqual(float(y[5]), 1.0)   # inside GT -> alpha*1+beta*1 clamped
        self.assertLess(float(y[12]), 0.15)  # gaussian tail, near zero
        self.assertGreater(float(y[4]), float(y[2]))  # boundary peak decays

    def test_padding_rows_are_zero(self):
        loss = AllocationSupervisionLoss()
        md = grid_metadata(1, 10)
        md[0, 7:, :] = 0.0                  # zero span = padding
        targets = torch.tensor([[1.0, 9.0]])
        y = loss.build_targets((md,), targets)[0][0]
        self.assertEqual(float(y[7:].abs().sum()), 0.0)

    def test_gradient_flows_to_predicted_scores(self):
        loss = AllocationSupervisionLoss()
        md = grid_metadata(1, 8)
        targets = torch.tensor([[2.0, 6.0]])
        mask = torch.ones(1, 1, 8, dtype=torch.bool)
        prob = torch.rand(1, 8, requires_grad=True)
        debug = ({"importance": prob},)
        out = loss(debug, (md,), (mask,), targets)
        out.backward()
        self.assertTrue(torch.isfinite(prob.grad).all())
        self.assertGreater(float(prob.grad.abs().sum()), 0.0)

    def test_non_oracle_allocator_rejects_targets(self):
        allocator = QueryBoundaryAdaptiveAnchorAllocator(
            target_keep_ratio=0.5, allocator_policy="learned")
        tokens = torch.randn(1, 3, 8)
        mask = torch.ones(1, 8, dtype=torch.bool)
        with self.assertRaises(ValueError):
            allocator(tokens, mask, allocator_targets=torch.tensor([[1.0, 4.0]]))

    def test_cut_count_matches_uniform_budget(self):
        mask = torch.ones(1, 12, dtype=torch.bool)
        outs = []
        for policy in ("uniform", "learned"):
            allocator = QueryBoundaryAdaptiveAnchorAllocator(
                target_keep_ratio=0.5, allocator_policy=policy)
            tokens = torch.randn(1, 3, 12)
            importance = torch.rand(1, 12) if policy == "learned" else None
            outs.append(int(allocator(
                tokens, mask, importance_score=importance
            )["target_counts"][0]))
        mask = torch.ones(1, 12, dtype=torch.bool)
        self.assertEqual(outs[0], outs[1])


if __name__ == "__main__":
    unittest.main()

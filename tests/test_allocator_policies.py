"""Allocator policy tests for the Uniform / Random / Learned / Oracle study.

CPU-only allocator-level checks. The single research variable between
A-U/A-R/A-L/A-O is allocator_policy, so these tests pin down the behavior of
each policy before any formal training is started.
"""

import math
import unittest

import torch

from libs.modeling.adaptive_anchor import (
    QueryBoundaryAdaptiveAnchorAllocator,
)


def make_inputs(batch_size, seq_len, channels=6, seed=0):
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randn(batch_size, channels, seq_len, generator=generator)
    lengths = torch.randint(
        max(1, seq_len // 2), seq_len + 1, (batch_size,), generator=generator
    )
    lengths[-1] = seq_len
    mask = torch.arange(seq_len).unsqueeze(0) < lengths.unsqueeze(1)
    return tokens, mask


def make_metadata(seq_len, batch_size=1):
    positions = torch.arange(seq_len, dtype=torch.float32)
    token_start = positions
    token_end = token_start + 1.0
    centers = token_start + 0.5
    spans = torch.ones_like(token_start)
    metadata = torch.stack(
        (token_start, token_end, centers, spans), dim=-1
    )
    return metadata.unsqueeze(0).repeat(batch_size, 1, 1)


def group_sizes_from_assignment(assignment, valid_mask):
    """Mean per-token group size per sample."""
    membership = assignment.argmax(dim=1)
    sizes = []
    for b in range(assignment.size(0)):
        count = int(valid_mask[b].sum())
        if count == 0:
            sizes.append([])
            continue
        per_group = assignment[b].sum(dim=1)
        sizes.append(
            per_group[membership[b][:count]].tolist()
        )
    return sizes


def assert_partition_invariants(test, outputs, valid_mask, target_counts):
    assignment = outputs["assignment_matrix"]
    batch_size, num_anchors, seq_len = assignment.shape
    for b in range(batch_size):
        count = int(valid_mask[b].sum())
        target = int(target_counts[b])
        test.assertEqual(target, outputs["target_counts"][b].item())
        column_sum = assignment[b, :target, :count].sum(dim=0)
        test.assertTrue(
            torch.all(column_sum == 1.0),
            "every valid token must belong to exactly one anchor",
        )
        if count == 0:
            continue
        members = assignment[b, :target, :count].argmax(dim=0)
        test.assertTrue(torch.all(members.diff() >= 0), "temporal order")
        first_tokens = assignment[b, :target, :count].argmax(dim=1)
        for group in range(target):
            indices = (
                torch.nonzero(members == group, as_tuple=False).flatten()
            )
            test.assertTrue(indices.numel() > 0, "no empty group")
            test.assertTrue(
                bool(torch.all(indices == indices[0] + torch.arange(indices.numel()))),
                "groups must be contiguous",
            )
            test.assertEqual(int(first_tokens[group]), int(indices[0]))
    # Padded tail positions belong to no anchor.
    for b in range(batch_size):
        count = int(valid_mask[b].sum())
        if count < seq_len:
            test.assertTrue(
                torch.all(assignment[b, :, count:] == 0.0),
                "padding tokens must not be assigned",
            )


class AllocatorPolicyTests(unittest.TestCase):
    def _allocator(self, policy, ratio=0.5, seed=0, radius=1.0):
        return QueryBoundaryAdaptiveAnchorAllocator(
            target_keep_ratio=ratio,
            importance_weighted_pooling=True,
            allocator_policy=policy,
            random_seed=seed,
            oracle_boundary_radius=radius,
        )

    def test_invalid_policy_rejected(self):
        with self.assertRaises(ValueError):
            self._allocator("adaptive")

    def test_oracle_requires_targets_and_metadata(self):
        allocator = self._allocator("oracle")
        tokens, mask = make_inputs(2, 10)
        with self.assertRaises(ValueError):
            allocator(tokens, mask)

    def test_uniform_ratio_half_matches_stride_two(self):
        for seq_len in (2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 31, 32):
            allocator = self._allocator("uniform")
            tokens, mask = make_inputs(1, seq_len)
            outputs = allocator(tokens, mask)
            assignment = outputs["assignment_matrix"]
            target = int(outputs["target_counts"][0])
            expected_target = min(
                seq_len, max(1, int(math.ceil(seq_len * 0.5)))
            )
            self.assertEqual(target, expected_target)
            members = assignment[0, :target].argmax(dim=0)
            for group in range(target):
                start = 2 * group
                expected = [
                    idx
                    for idx in (start, start + 1)
                    if idx < seq_len
                ]
                got = torch.nonzero(
                    members == group, as_tuple=False
                ).flatten().tolist()
                self.assertEqual(
                    got,
                    expected,
                    "ratio 0.5 uniform grouping must equal stride-2 "
                    f"pairing at length {seq_len}",
                )

    def test_uniform_custom_ratio_budget_order(self):
        allocator = self._allocator("uniform", ratio=0.75)
        tokens, mask = make_inputs(1, 12)
        outputs = allocator(tokens, mask)
        self.assertEqual(int(outputs["target_counts"][0]), 9)
        assert_partition_invariants(
            self, outputs, mask, outputs["target_counts"]
        )

    def test_random_is_deterministic_and_seed_sensitive(self):
        tokens, mask = make_inputs(2, 24, seed=3)
        first = self._allocator("random", seed=0)(tokens, mask)
        second = self._allocator("random", seed=0)(tokens, mask)
        other = self._allocator("random", seed=999)(tokens, mask)
        for key in ("anchor_positions", "assignment_matrix"):
            self.assertTrue(torch.equal(first[key], second[key]))
        self.assertFalse(
            torch.equal(first["assignment_matrix"], other["assignment_matrix"]),
            "different random seeds should usually change cuts",
        )
        assert_partition_invariants(
            self, first, mask, first["target_counts"]
        )
        assert_partition_invariants(
            self, other, mask, other["target_counts"]
        )

    def test_oracle_prioritizes_boundary_foreground_background(self):
        seq_len = 64
        gt_start, gt_end = 16.0, 32.0
        allocator = self._allocator("oracle")
        uniform_allocator = self._allocator("uniform")
        tokens, mask = make_inputs(1, seq_len)
        metadata = make_metadata(seq_len)
        targets = torch.tensor([[gt_start, gt_end]])
        outputs = allocator(tokens, mask, allocator_targets=targets,
                            temporal_metadata=metadata)
        reference = allocator.reference_forward(
            tokens, mask, allocator_targets=targets,
            temporal_metadata=metadata,
        )
        uniform_outputs = uniform_allocator(tokens, mask)

        assert_partition_invariants(
            self, outputs, mask, outputs["target_counts"]
        )
        self.assertTrue(
            torch.equal(
                outputs["target_counts"], uniform_outputs["target_counts"]
            ),
            "oracle must keep the same anchor budget as uniform",
        )
        for key in ("assignment_matrix", "anchors"):
            diff = (outputs[key] - reference[key]).abs().max().item()
            self.assertLess(diff, 1e-5)

        assignment = outputs["assignment_matrix"][0]
        members = assignment.argmax(dim=0)
        sizes = assignment.sum(dim=0)
        centers = torch.arange(seq_len, dtype=torch.float32) + 0.5

        def region_mean(region):
            return sizes[region].mean().item()

        foreground = (centers + 0.5 > gt_start) & (centers - 0.5 < gt_end)
        boundary = foreground & (
            ((centers - gt_start).abs() <= 1.0)
            | ((centers - gt_end).abs() <= 1.0)
        )
        inner_fg = foreground & ~boundary
        background = ~foreground
        boundary_mean = region_mean(boundary)
        fg_mean = region_mean(inner_fg)
        bg_mean = region_mean(background)
        self.assertLessEqual(
            boundary_mean, fg_mean, "boundary groups must be smallest"
        )
        self.assertLessEqual(
            fg_mean, bg_mean, "foreground groups must be smaller than background"
        )
        self.assertLess(
            bg_mean, float(seq_len),
            "sanity: background group size should stay bounded",
        )

    def test_learned_cut_selection_unchanged(self):
        allocator = self._allocator("learned")
        tokens = torch.randn(1, 4, 8)
        mask = torch.ones(1, 1, 8, dtype=torch.bool)
        importance = torch.tensor(
            [[0.1, 0.9, 0.1, 0.8, 0.1, 0.7, 0.1, 0.2]]
        )
        outputs = allocator(tokens, mask, importance_score=importance)
        reference = allocator.reference_forward(tokens, mask, importance)
        assignment = outputs["assignment_matrix"][0]
        members = assignment.argmax(dim=0).tolist()
        self.assertEqual(
            members,
            [0, 1, 2, 3, 3, 3, 3, 3],
            "adjacent-max ranking must pick gaps after tokens 0,1,2",
        )
        for key in ("anchors", "assignment_matrix"):
            diff = (outputs[key] - reference[key]).abs().max().item()
            self.assertLess(diff, 1e-5)
        # Importance-weighted pooling inside each group.
        weights = importance[0].clamp_min(0.0) + 1e-6
        expected_last = (
            tokens[0].transpose(0, 1)[3:]
            * weights[3:].unsqueeze(-1)
        ).sum(dim=0) / weights[3:].sum()
        diff = (outputs["anchors"][0, :, -1] - expected_last).abs().max()
        self.assertLess(float(diff), 1e-5)

    def test_non_learned_policies_ignore_importance(self):
        tokens, mask = make_inputs(2, 20, seed=5)
        importance = torch.rand(2, 20)
        metadata = make_metadata(20, batch_size=2)
        targets = torch.tensor([[4.0, 12.0], [2.0, 18.0]])
        for policy in ("uniform", "random"):
            allocator = self._allocator(policy, seed=1)
            with self.assertRaises(ValueError):
                allocator(tokens, mask, allocator_targets=targets)
        allocator = self._allocator("oracle", seed=1)
        with_importance = allocator(
            tokens, mask, importance_score=importance,
            allocator_targets=targets, temporal_metadata=metadata,
        )
        plain = allocator(tokens, mask, allocator_targets=targets,
                          temporal_metadata=metadata)
        for key in ("anchors", "assignment_matrix"):
            diff = (plain[key] - with_importance[key]).abs().max()
            self.assertLess(float(diff), 1e-6,
                            "oracle must ignore importance scores")

    def test_reference_tensorized_equivalence_all_policies(self):
        # Exact cross-path equality is only guaranteed per sample: the
        # reference path reseeds the random generator for every batch item,
        # while the production tensorized path draws independent rows for the
        # whole batch. Single-sample inputs make both paths coincide.
        tokens, mask = make_inputs(1, 26, seed=11)
        importance = torch.rand(1, 26)
        metadata = make_metadata(26, batch_size=1)
        targets = torch.tensor([[3.0, 15.0]])
        cases = {
            "uniform": {},
            "random": {},
            "learned": {"importance_score": importance},
            "oracle": {
                "allocator_targets": targets,
                "temporal_metadata": metadata,
            },
        }
        for policy, kwargs in cases.items():
            allocator = self._allocator(policy, seed=2)
            fast = allocator(tokens, mask, **kwargs)
            ref = allocator.reference_forward(tokens, mask, **kwargs)
            self.assertTrue(
                torch.equal(fast["anchor_mask"], ref["anchor_mask"])
            )
            self.assertTrue(
                torch.equal(fast["sequence_mask"], ref["sequence_mask"])
            )
            self.assertTrue(
                torch.equal(fast["target_counts"], ref["target_counts"])
            )
            diff_assignment = (
                fast["assignment_matrix"] - ref["assignment_matrix"]
            ).abs().max().item()
            self.assertLess(diff_assignment, 1e-5, policy)
            diff_anchors = (
                fast["anchors"] - ref["anchors"]
            ).abs().max().item()
            self.assertLess(diff_anchors, 1e-4, policy)
            diff_positions = (
                fast["anchor_positions"] != ref["anchor_positions"]
            )
            self.assertFalse(
                bool(diff_positions.any()),
                f"interleave order must match between paths ({policy})",
            )

    def test_random_batch_rows_are_independent_in_production_path(self):
        tokens, mask = make_inputs(3, 24, seed=7)
        allocator = self._allocator("random", seed=0)
        outputs = allocator(tokens, mask)
        assignment = outputs["assignment_matrix"]
        valid2d = mask[:, 0] if mask.ndim == 3 else mask
        lengths = valid2d.sum(dim=-1)
        patterns = []
        for b in range(assignment.size(0)):
            count = int(lengths[b])
            target = int(outputs["target_counts"][b])
            starts = (
                assignment[b, :target, :count]
                .argmax(dim=1)
                .tolist()
            )
            patterns.append(tuple(starts))
        self.assertEqual(
            len(set(patterns)), len(patterns),
            "samples in a batch must receive independent random cuts",
        )


if __name__ == "__main__":
    unittest.main()

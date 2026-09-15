import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.modeling.contrastive_losses import (
    assignment_aware_anchor_contrastive,
)
from libs.modeling.losses import MultiScaleMaskedContrastive


class IdentityProjector(nn.Module):

    def forward(self, features):
        return features


def reference_assignment_loss(
    anchors,
    tokens,
    anchor_mask,
    token_mask,
    assignment,
    temperature,
):
    anchors = F.normalize(anchors.transpose(1, 2).float(), dim=-1)
    tokens = F.normalize(tokens.transpose(1, 2).float(), dim=-1)
    anchor_mask = anchor_mask[:, 0].bool()
    token_mask = token_mask[:, 0].bool()
    losses = []
    for batch_index in range(anchors.size(0)):
        valid_tokens = token_mask[batch_index]
        for anchor_index in range(anchors.size(1)):
            weights = assignment[batch_index, anchor_index]
            weights = weights * valid_tokens.to(weights.dtype)
            if not anchor_mask[batch_index, anchor_index]:
                continue
            positive = weights > 0
            if not positive.any():
                continue
            logits = (
                tokens[batch_index]
                @ anchors[batch_index, anchor_index]
            ) / temperature
            log_numerator = torch.logsumexp(
                logits[positive] + weights[positive].log(),
                dim=0,
            )
            log_denominator = torch.logsumexp(
                logits[valid_tokens],
                dim=0,
            )
            losses.append(log_denominator - log_numerator)
    if not losses:
        return (anchors.sum() + tokens.sum()) * 0.0
    return torch.stack(losses).mean()


class AssignmentAwareAnchorContrastiveTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(41)
        self.projector = IdentityProjector()

    def test_binary_groups_of_size_one_to_four_match_reference(self):
        tokens = torch.randn(1, 5, 10, requires_grad=True)
        anchors = torch.randn(1, 5, 4, requires_grad=True)
        anchor_mask = torch.ones(1, 1, 4, dtype=torch.bool)
        token_mask = torch.ones(1, 1, 10, dtype=torch.bool)
        assignment = torch.zeros(1, 4, 10)
        group_sizes = (1, 2, 3, 4)
        cursor = 0
        for anchor_index, group_size in enumerate(group_sizes):
            assignment[
                0, anchor_index, cursor:cursor + group_size
            ] = 1.0
            cursor += group_size

        actual = assignment_aware_anchor_contrastive(
            anchors,
            tokens,
            anchor_mask,
            token_mask,
            assignment,
            self.projector,
            temperature=0.2,
        )
        expected = reference_assignment_loss(
            anchors,
            tokens,
            anchor_mask,
            token_mask,
            assignment,
            temperature=0.2,
        )

        torch.testing.assert_close(actual, expected)
        self.assertEqual(
            assignment.sum(dim=-1).long().tolist(),
            [[1, 2, 3, 4]],
        )
        actual.backward()
        self.assertTrue(torch.isfinite(anchors.grad).all())
        self.assertTrue(torch.isfinite(tokens.grad).all())
        self.assertGreater(float(anchors.grad.abs().sum()), 0.0)
        self.assertGreater(float(tokens.grad.abs().sum()), 0.0)

    def test_batch_variable_anchors_and_padding_are_isolated(self):
        anchors = torch.randn(2, 4, 4)
        tokens = torch.randn(2, 4, 8)
        anchor_mask = torch.tensor(
            [
                [[True, True, True, True]],
                [[True, True, False, False]],
            ]
        )
        token_mask = torch.tensor(
            [
                [[True, True, True, True, True, True, True, True]],
                [[True, True, True, True, True, False, False, False]],
            ]
        )
        assignment = torch.zeros(2, 4, 8)
        assignment[0, 0, 0] = 1
        assignment[0, 1, 1:3] = 1
        assignment[0, 2, 3:6] = 1
        assignment[0, 3, 6:8] = 1
        assignment[1, 0, :2] = 1
        assignment[1, 1, 2:5] = 1

        baseline = assignment_aware_anchor_contrastive(
            anchors,
            tokens,
            anchor_mask,
            token_mask,
            assignment,
            self.projector,
        )

        perturbed_anchors = anchors.clone()
        perturbed_tokens = tokens.clone()
        perturbed_assignment = assignment.clone()
        perturbed_anchors[1, :, 2:] = 1e6
        perturbed_tokens[1, :, 5:] = -1e6
        perturbed_assignment[1, 2:, :] = 1.0
        perturbed_assignment[1, :2, 5:] = 1.0
        perturbed = assignment_aware_anchor_contrastive(
            perturbed_anchors,
            perturbed_tokens,
            anchor_mask,
            token_mask,
            perturbed_assignment,
            self.projector,
        )

        torch.testing.assert_close(baseline, perturbed)

    def test_soft_assignment_uses_log_domain_weights(self):
        anchors = torch.randn(1, 3, 2)
        tokens = torch.randn(1, 3, 4)
        anchor_mask = torch.ones(1, 1, 2, dtype=torch.bool)
        token_mask = torch.ones(1, 1, 4, dtype=torch.bool)
        assignment = torch.tensor(
            [[
                [0.75, 0.25, 0.0, 0.0],
                [0.0, 0.10, 0.30, 0.60],
            ]],
            requires_grad=True,
        )

        actual = assignment_aware_anchor_contrastive(
            anchors,
            tokens,
            anchor_mask,
            token_mask,
            assignment,
            self.projector,
            temperature=0.3,
        )
        expected = reference_assignment_loss(
            anchors,
            tokens,
            anchor_mask,
            token_mask,
            assignment,
            temperature=0.3,
        )

        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertIsNotNone(assignment.grad)
        self.assertTrue(torch.isfinite(assignment.grad).all())

    def test_logsumexp_is_finite_for_extreme_logits(self):
        anchors = torch.randn(1, 8, 2, requires_grad=True)
        tokens = torch.randn(1, 8, 4, requires_grad=True)
        anchor_mask = torch.ones(1, 1, 2, dtype=torch.bool)
        token_mask = torch.ones(1, 1, 4, dtype=torch.bool)
        assignment = torch.tensor(
            [[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]]
        )

        loss = assignment_aware_anchor_contrastive(
            anchors,
            tokens,
            anchor_mask,
            token_mask,
            assignment,
            self.projector,
            temperature=1e-5,
        )

        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(anchors.grad).all())
        self.assertTrue(torch.isfinite(tokens.grad).all())

    def test_no_valid_positive_returns_differentiable_zero(self):
        anchors = torch.randn(2, 3, 2, requires_grad=True)
        tokens = torch.randn(2, 3, 4, requires_grad=True)
        anchor_mask = torch.zeros(2, 1, 2, dtype=torch.bool)
        token_mask = torch.ones(2, 1, 4, dtype=torch.bool)
        assignment = torch.zeros(2, 2, 4)

        loss = assignment_aware_anchor_contrastive(
            anchors,
            tokens,
            anchor_mask,
            token_mask,
            assignment,
            self.projector,
        )

        self.assertEqual(float(loss), 0.0)
        loss.backward()
        self.assertTrue(torch.equal(anchors.grad, torch.zeros_like(anchors)))
        self.assertTrue(torch.equal(tokens.grad, torch.zeros_like(tokens)))

    def test_invalid_assignment_and_temperature_are_rejected(self):
        anchors = torch.randn(1, 3, 1)
        tokens = torch.randn(1, 3, 2)
        anchor_mask = torch.ones(1, 1, 1, dtype=torch.bool)
        token_mask = torch.ones(1, 1, 2, dtype=torch.bool)

        with self.assertRaises(ValueError):
            assignment_aware_anchor_contrastive(
                anchors,
                tokens,
                anchor_mask,
                token_mask,
                torch.tensor([[[1.1, 0.0]]]),
                self.projector,
            )
        with self.assertRaises(ValueError):
            assignment_aware_anchor_contrastive(
                anchors,
                tokens,
                anchor_mask,
                token_mask,
                torch.tensor([[[1.0, 0.0]]]),
                self.projector,
                temperature=0.0,
            )


class MultiScaleAssignmentACCTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(43)

    @staticmethod
    def make_options(acc_mode):
        return {
            "acc_mode": acc_mode,
            "contr_type": "contr_mp",
            "temperature": 0.1,
            "proj_outdim": 4,
            "proj_expand": 1.0,
            "proj_num_layers": 1,
            "radius": 0,
            "neg_ratio": 1.0,
            "gap_ratio": 0.2,
            "hard_neg": False,
            "cross_video_neg": False,
        }

    def test_assignment_mode_requires_and_consumes_each_level(self):
        criterion = MultiScaleMaskedContrastive(
            self.make_options("assignment_acc"), vid_embd_dim=4
        )
        sequences = (
            torch.randn(2, 4, 6, requires_grad=True),
            torch.randn(2, 4, 3, requires_grad=True),
        )
        anchors = (
            torch.randn(2, 4, 3, requires_grad=True),
            torch.randn(2, 4, 2, requires_grad=True),
        )
        sequence_masks = (
            torch.tensor([
                [[True, True, True, True, True, True]],
                [[True, True, True, True, False, False]],
            ]),
            torch.tensor([
                [[True, True, True]],
                [[True, True, False]],
            ]),
        )
        anchor_masks = (
            torch.tensor([
                [[True, True, True]],
                [[True, True, False]],
            ]),
            torch.tensor([
                [[True, True]],
                [[True, False]],
            ]),
        )
        assignments = (
            torch.tensor([
                [
                    [1, 1, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 1, 1],
                ],
                [
                    [1, 1, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
            ], dtype=torch.float32),
            torch.tensor([
                [
                    [1, 1, 0],
                    [0, 0, 1],
                ],
                [
                    [1, 1, 0],
                    [0, 0, 0],
                ],
            ], dtype=torch.float32),
        )

        loss = criterion(
            sequences,
            sequence_masks,
            anchors,
            anchor_masks,
            assignment_matrices=assignments,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for tensor in sequences + anchors:
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())

        with self.assertRaises(ValueError):
            criterion(
                sequences,
                sequence_masks,
                anchors,
                anchor_masks,
            )
        with self.assertRaises(ValueError):
            criterion(
                sequences,
                sequence_masks,
                anchors,
                anchor_masks,
                assignment_matrices=assignments[:1],
            )

    def test_legacy_mode_remains_default(self):
        default_options = self.make_options("legacy_acc")
        default_options.pop("acc_mode")
        default_criterion = MultiScaleMaskedContrastive(
            default_options, vid_embd_dim=4
        )
        explicit_criterion = MultiScaleMaskedContrastive(
            self.make_options("legacy_acc"), vid_embd_dim=4
        )
        explicit_criterion.load_state_dict(default_criterion.state_dict())
        self.assertEqual(default_criterion.acc_mode, "legacy_acc")
        self.assertEqual(explicit_criterion.acc_mode, "legacy_acc")

        sequence = (torch.randn(1, 4, 8),)
        anchors = (torch.randn(1, 4, 4),)
        sequence_mask = (
            torch.ones(1, 1, 8, dtype=torch.bool),
        )
        anchor_mask = (
            torch.ones(1, 1, 4, dtype=torch.bool),
        )
        torch.manual_seed(47)
        default_loss = default_criterion(
            sequence,
            sequence_mask,
            anchors,
            anchor_mask,
        )
        torch.manual_seed(47)
        explicit_loss = explicit_criterion(
            sequence,
            sequence_mask,
            anchors,
            anchor_mask,
        )
        self.assertTrue(torch.isfinite(default_loss))
        torch.testing.assert_close(
            default_loss, explicit_loss, rtol=0.0, atol=0.0
        )

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            MultiScaleMaskedContrastive(
                self.make_options("unknown"), vid_embd_dim=4
            )


if __name__ == "__main__":
    unittest.main()

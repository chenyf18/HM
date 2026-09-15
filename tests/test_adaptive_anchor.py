import math
import unittest
from unittest import mock

import torch
import torch.nn as nn

from libs.modeling import anchor_mamba
from libs.modeling.adaptive_anchor import (
    QueryBoundaryAdaptiveAnchorAllocator,
)
from libs.modeling.anchor_mamba import AnchorMambaPoolingBlockGated
from libs.modeling.losses import MultiScaleMaskedContrastive
from libs.modeling.video_net import HieraMambaBackbone


class DummyGlobalEncoder(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


def make_block(**kwargs):
    options = {
        "stride": 2,
        "d_model": 8,
        "nhead": 2,
        "dropout": 0.0,
        "ffn_ratio": 2,
        "local_encode": False,
        "pool_method": "mean",
        "bidirectional": True,
    }
    options.update(kwargs)
    with mock.patch.object(anchor_mamba, "Hydra", DummyGlobalEncoder):
        block = AnchorMambaPoolingBlockGated(**options)
    return block.eval()


class AdaptiveAnchorAllocatorTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(23)

    def test_toy_assignment_preserves_order_and_budget(self):
        allocator = QueryBoundaryAdaptiveAnchorAllocator(
            target_keep_ratio=0.75,
            importance_weighted_pooling=False,
        )
        tokens = torch.arange(12, dtype=torch.float32).reshape(1, 1, 12)
        mask = torch.ones(1, 1, 12, dtype=torch.bool)
        importance = torch.tensor(
            [[
                0.01, 0.02, 0.90, 0.95, 0.92, 0.88,
                0.03, 0.04, 0.85, 0.82, 0.015, 0.025,
            ]]
        )

        output = allocator(tokens, mask, importance)
        assignment = output["assignment_matrix"]
        token_to_anchor = assignment.argmax(dim=1)[0].tolist()
        print("toy token -> anchor assignment:", token_to_anchor)

        expected = [0, 0, 1, 2, 3, 4, 5, 5, 6, 7, 8, 8]
        self.assertEqual(token_to_anchor, expected)
        self.assertEqual(output["target_counts"].tolist(), [9])
        self.assertEqual(int(output["anchor_mask"].sum().item()), 9)
        self.assertEqual(assignment.shape, (1, 9, 12))
        torch.testing.assert_close(
            assignment.sum(dim=1),
            torch.ones(1, 12),
        )

        previous_end = -1
        for anchor_index in range(assignment.size(1)):
            assigned = torch.nonzero(
                assignment[0, anchor_index], as_tuple=False
            ).flatten()
            self.assertGreater(assigned.numel(), 0)
            expected_range = torch.arange(
                assigned[0], assigned[-1] + 1
            )
            self.assertTrue(torch.equal(assigned.cpu(), expected_range))
            self.assertGreater(int(assigned[0]), previous_end)
            previous_end = int(assigned[-1])
            first_token = int(assigned[0])
            self.assertLess(
                int(output["anchor_positions"][0, anchor_index]),
                int(output["sequence_positions"][0, first_token]),
            )

    def test_variable_lengths_budget_mask_and_identity_extraction(self):
        allocator = QueryBoundaryAdaptiveAnchorAllocator(
            target_keep_ratio=0.6
        )
        tokens = torch.randn(3, 4, 7)
        mask = torch.tensor(
            [
                [[True, True, True, True, True, True, True]],
                [[True, True, True, True, False, False, False]],
                [[True, False, False, False, False, False, False]],
            ]
        )
        importance = torch.rand(3, 7)

        output = allocator(tokens, mask, importance)

        self.assertEqual(output["target_counts"].tolist(), [5, 3, 1])
        self.assertEqual(output["anchor_mask"].shape, (3, 1, 5))
        self.assertEqual(
            output["anchor_mask"].sum(dim=(1, 2)).tolist(),
            [5, 3, 1],
        )
        self.assertEqual(output["assignment_matrix"].shape, (3, 5, 7))
        torch.testing.assert_close(
            output["assignment_matrix"].sum(dim=1),
            mask[:, 0].to(tokens.dtype),
        )
        self.assertTrue(
            torch.equal(output["sequence_mask"], mask)
        )

        anchor_out, sequence_out = allocator.extract_outputs(
            output["combined"],
            output["anchor_positions"],
            output["sequence_positions"],
            output["anchor_mask"],
            output["sequence_mask"],
        )
        torch.testing.assert_close(anchor_out, output["anchors"])
        torch.testing.assert_close(
            sequence_out,
            tokens * mask.to(tokens.dtype),
        )
        self.assertTrue(torch.equal(anchor_out[1, :, 3:], torch.zeros(4, 2)))
        self.assertTrue(torch.equal(anchor_out[2, :, 1:], torch.zeros(4, 4)))
        self.assertTrue(
            torch.equal(
                output["assignment_matrix"][1, 3:],
                torch.zeros(2, 7),
            )
        )

    def test_tensorized_allocator_matches_reference(self):
        for weighted in (False, True):
            torch.manual_seed(97)
            allocator = QueryBoundaryAdaptiveAnchorAllocator(0.6, weighted)
            token_seed = torch.randn(3, 4, 11)
            importance_seed = torch.randn(3, 11)
            actual_tokens = token_seed.detach().clone().requires_grad_()
            reference_tokens = token_seed.detach().clone().requires_grad_()
            actual_importance = importance_seed.detach().clone().requires_grad_()
            reference_importance = importance_seed.detach().clone().requires_grad_()
            mask = torch.tensor(
                [
                    [[True] * 11],
                    [[True] * 7 + [False] * 4],
                    [[True] * 2 + [False] * 9],
                ]
            )
            actual = allocator(actual_tokens, mask, actual_importance)
            reference = allocator.reference_forward(
                reference_tokens, mask, reference_importance
            )
            for key in (
                "combined", "anchors", "anchor_positions",
                "sequence_positions", "expanded_mask", "anchor_mask",
                "sequence_mask", "assignment_matrix", "target_counts",
            ):
                torch.testing.assert_close(
                    actual[key], reference[key], atol=1e-6, rtol=1e-6,
                    check_dtype=False,
                )
            actual["combined"].square().mean().backward()
            reference["combined"].square().mean().backward()
            torch.testing.assert_close(
                actual_tokens.grad, reference_tokens.grad,
                atol=1e-6, rtol=1e-5,
            )
            if weighted:
                torch.testing.assert_close(
                    actual_importance.grad, reference_importance.grad,
                    atol=1e-6, rtol=1e-5,
                )

    def test_invalid_keep_ratio_is_rejected(self):
        for ratio in (0.0, -0.1, 1.1):
            with self.assertRaises(ValueError):
                QueryBoundaryAdaptiveAnchorAllocator(ratio)


class AdaptiveAnchorAMPTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(29)

    def test_disabled_path_keeps_stride_two_contract(self):
        block = make_block(adaptive_anchor=False)
        output = block(
            torch.randn(2, 8, 7),
            torch.ones(2, 1, 7, dtype=torch.bool),
        )
        self.assertEqual(len(output), 4)
        self.assertEqual(output[0].shape, (2, 8, 4))
        self.assertEqual(output[1].shape, (2, 8, 7))

    def test_adaptive_outputs_mask_and_backward(self):
        block = make_block(
            query_dim=6,
            adaptive_anchor=True,
            target_keep_ratio=0.75,
            importance_hidden_dim=8,
        )
        tokens = torch.randn(2, 8, 8, requires_grad=True)
        mask = torch.tensor(
            [
                [[True, True, True, True, True, True, True, True]],
                [[True, True, True, True, True, False, False, False]],
            ]
        )
        query = torch.randn(2, 6, 4, requires_grad=True)
        query_mask = torch.tensor(
            [
                [[True, True, True, True]],
                [[True, True, False, False]],
            ]
        )

        output = block(
            tokens,
            mask,
            query,
            query_mask,
            return_importance_debug=True,
        )
        (
            anchor_out,
            sequence_out,
            anchor_mask,
            sequence_mask,
            assignment,
            importance,
            debug,
        ) = output

        self.assertEqual(anchor_out.shape, (2, 8, 6))
        self.assertEqual(sequence_out.shape, (2, 8, 8))
        self.assertEqual(anchor_mask.sum(dim=(1, 2)).tolist(), [6, 4])
        self.assertTrue(torch.equal(sequence_mask, mask))
        self.assertEqual(assignment.shape, (2, 6, 8))
        self.assertEqual(importance.shape, (2, 8))
        torch.testing.assert_close(
            assignment.sum(dim=1),
            mask[:, 0].to(assignment.dtype),
        )
        self.assertTrue(torch.equal(importance[1, 5:], torch.zeros(3)))
        self.assertTrue(torch.equal(debug["assignment_matrix"], assignment))

        for tensor in (
            anchor_out,
            sequence_out,
            assignment,
            importance,
        ):
            self.assertTrue(torch.isfinite(tensor).all())
        loss = (
            anchor_out.square().mean()
            + sequence_out.square().mean()
            + importance.mean()
        )
        loss.backward()

        self.assertIsNotNone(tokens.grad)
        self.assertIsNotNone(query.grad)
        self.assertTrue(torch.isfinite(tokens.grad).all())
        self.assertTrue(torch.isfinite(query.grad).all())
        predictor_grads = [
            parameter.grad
            for name, parameter in block.named_parameters()
            if "importance_predictor" in name and parameter.grad is not None
        ]
        self.assertTrue(predictor_grads)
        for grad in predictor_grads:
            self.assertTrue(torch.isfinite(grad).all())

    def test_adaptive_block_feeds_assignment_acc_backward(self):
        block = make_block(
            query_dim=6,
            adaptive_anchor=True,
            target_keep_ratio=0.75,
            importance_hidden_dim=8,
        )
        criterion = MultiScaleMaskedContrastive(
            {
                "acc_mode": "assignment_acc",
                "temperature": 0.07,
                "proj_outdim": 8,
                "proj_expand": 1.0,
                "proj_num_layers": 1,
            },
            vid_embd_dim=8,
        )
        tokens = torch.randn(2, 8, 8, requires_grad=True)
        mask = torch.tensor(
            [
                [[True, True, True, True, True, True, True, True]],
                [[True, True, True, True, True, False, False, False]],
            ]
        )
        query = torch.randn(2, 6, 4, requires_grad=True)
        query_mask = torch.tensor(
            [
                [[True, True, True, True]],
                [[True, True, False, False]],
            ]
        )

        output = block(tokens, mask, query, query_mask)
        loss = criterion(
            (output[1],),
            (output[3],),
            (output[0],),
            (output[2],),
            assignment_matrices=(output[4],),
        )

        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(tokens.grad)
        self.assertIsNotNone(query.grad)
        self.assertTrue(torch.isfinite(tokens.grad).all())
        self.assertTrue(torch.isfinite(query.grad).all())
        projector_grads = [
            parameter.grad
            for parameter in criterion.projector.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(projector_grads)
        for grad in projector_grads:
            self.assertTrue(torch.isfinite(grad).all())

    def test_missing_query_uses_uniform_budgeted_fallback(self):
        block = make_block(
            query_dim=6,
            adaptive_anchor=True,
            target_keep_ratio=0.5,
        )
        output = block(
            torch.randn(2, 8, 7),
            torch.tensor(
                [
                    [[True, True, True, True, True, True, True]],
                    [[True, True, True, True, False, False, False]],
                ]
            ),
        )
        self.assertEqual(len(output), 6)
        self.assertEqual(output[2].sum(dim=(1, 2)).tolist(), [4, 2])
        self.assertTrue(torch.equal(output[5], torch.zeros(2, 7)))

    def test_backbone_propagates_progressive_adaptive_budgets(self):
        with mock.patch.object(anchor_mamba, "Hydra", DummyGlobalEncoder):
            backbone = HieraMambaBackbone(
                in_dim=8,
                embd_dim=8,
                max_seq_len=8,
                n_heads=2,
                stride=1,
                arch=(1, 0, 2),
                use_abs_pe=False,
                local_encode=False,
                return_anchor=True,
                block_type="AnchorMambaPoolingBlockGated",
                query_dim=6,
                adaptive_anchor=True,
                target_keep_ratio=0.75,
                importance_debug=True,
                importance_hidden_dim=8,
            ).eval()

        tokens = torch.randn(2, 8, 8)
        mask = torch.tensor(
            [
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, False, False, False],
            ]
        )
        query = torch.randn(2, 6, 4)
        query_mask = torch.tensor(
            [
                [True, True, True, True],
                [True, True, False, False],
            ]
        )

        (
            fpn,
            fpn_masks,
            anchor_fpn,
            anchor_masks,
            assignments,
        ) = backbone(
            tokens,
            mask,
            query,
            query_mask,
            return_anchor_assignments=True,
        )

        self.assertEqual([value.shape[-1] for value in fpn], [8, 6])
        self.assertEqual(
            [value.shape[-1] for value in anchor_fpn], [6, 5]
        )
        self.assertEqual(
            [value.sum(dim=(1, 2)).tolist() for value in anchor_masks],
            [[6, 4], [5, 3]],
        )
        self.assertEqual(len(assignments), 2)
        self.assertEqual(len(backbone.last_importance_debug), 2)
        for assignment, layer_debug in zip(
            assignments, backbone.last_importance_debug
        ):
            self.assertTrue(
                torch.equal(
                    assignment,
                    layer_debug["assignment_matrix"],
                )
            )
        self.assertEqual(
            backbone.last_importance_debug[0]["assignment_matrix"].shape,
            (2, 6, 8),
        )
        self.assertEqual(
            backbone.last_importance_debug[1]["assignment_matrix"].shape,
            (2, 5, 6),
        )
        for layer_debug, layer_mask in zip(
            backbone.last_importance_debug, fpn_masks
        ):
            torch.testing.assert_close(
                layer_debug["assignment_matrix"].sum(dim=1),
                layer_mask[:, 0].to(tokens.dtype),
            )


if __name__ == "__main__":
    unittest.main()

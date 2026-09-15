import io
import math
import unittest
from contextlib import redirect_stdout
from unittest import mock

import torch
import torch.nn as nn

from libs.modeling import anchor_mamba
from libs.modeling.video_net import HieraMambaBackbone


class DummyGlobalEncoder(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class ProgressiveAdaptiveCompressionTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(31)

    @staticmethod
    def make_backbone(
        depth=3,
        target_keep_ratio=0.75,
        keep_ratio_per_level=None,
        adaptive_anchor=True,
        progressive_compression_debug=False,
    ):
        options = {
            "in_dim": 8,
            "embd_dim": 8,
            "max_seq_len": 16,
            "n_heads": 2,
            "stride": 1,
            "arch": (0, 0, depth),
            "use_abs_pe": False,
            "local_encode": False,
            "return_anchor": True,
            "block_type": "AnchorMambaPoolingBlockGated",
            "query_dim": 6,
            "adaptive_anchor": adaptive_anchor,
            "target_keep_ratio": target_keep_ratio,
            "keep_ratio_per_level": keep_ratio_per_level,
            "progressive_compression_debug": (
                progressive_compression_debug
            ),
            "importance_hidden_dim": 8,
        }
        with mock.patch.object(
            anchor_mamba, "Hydra", DummyGlobalEncoder
        ):
            backbone = HieraMambaBackbone(**options)
        return backbone.eval()

    def test_per_level_budget_order_debug_and_backward(self):
        ratios = (0.9, 0.6, 0.4)
        backbone = self.make_backbone(
            keep_ratio_per_level=ratios,
            progressive_compression_debug=True,
        )
        self.assertEqual(backbone.keep_ratio_per_level, ratios)
        self.assertEqual(
            [block.target_keep_ratio for block in backbone.branch],
            list(ratios),
        )
        self.assertEqual(
            [
                block.adaptive_anchor_allocator.target_keep_ratio
                for block in backbone.branch
            ],
            list(ratios),
        )

        tokens = torch.randn(2, 8, 10, requires_grad=True)
        mask = torch.tensor(
            [
                [True, True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, False, False, False],
            ]
        )
        captured = io.StringIO()
        with redirect_stdout(captured):
            (
                fpn,
                fpn_masks,
                anchor_fpn,
                anchor_masks,
                assignments,
            ) = backbone(
                tokens,
                mask,
                return_anchor_assignments=True,
            )

        expected_inputs = [[10, 7], [9, 7], [6, 5]]
        expected_anchors = [[9, 7], [6, 5], [3, 2]]
        self.assertEqual(
            [value.size(-1) for value in fpn],
            [10, 9, 6],
        )
        self.assertEqual(
            [value.size(-1) for value in anchor_fpn],
            [9, 6, 3],
        )
        self.assertEqual(
            [
                level_mask.sum(dim=(1, 2)).tolist()
                for level_mask in anchor_masks
            ],
            expected_anchors,
        )

        debug = backbone.last_progressive_debug
        self.assertEqual(len(debug), len(ratios))
        for level, state in enumerate(debug):
            self.assertEqual(state["level"], level)
            self.assertEqual(
                state["input_len"].tolist(),
                expected_inputs[level],
            )
            self.assertEqual(
                state["anchor_len"].tolist(),
                expected_anchors[level],
            )
            self.assertEqual(state["keep_ratio"], ratios[level])
            self.assertEqual(
                state["input_tensor_len"],
                [10, 9, 6][level],
            )
            self.assertEqual(
                state["anchor_tensor_len"],
                [9, 6, 3][level],
            )
            if level:
                self.assertEqual(
                    state["input_len"].tolist(),
                    debug[level - 1]["anchor_len"].tolist(),
                )

        printed = captured.getvalue()
        for level, (input_len, anchor_len, ratio) in enumerate(
            zip(expected_inputs, expected_anchors, ratios)
        ):
            self.assertIn(
                "level={} input_len={} -> anchor_len={} "
                "keep_ratio={}".format(
                    level, input_len, anchor_len, ratio
                ),
                printed,
            )

        input_masks = (mask.unsqueeze(1),) + anchor_masks[:-1]
        for assignment, input_mask, anchor_mask in zip(
            assignments, input_masks, anchor_masks
        ):
            for batch_index in range(tokens.size(0)):
                valid_tokens = torch.nonzero(
                    input_mask[batch_index, 0], as_tuple=False
                ).flatten()
                anchor_count = int(
                    anchor_mask[batch_index, 0].sum().item()
                )
                valid_assignment = assignment[
                    batch_index, :anchor_count
                ].index_select(1, valid_tokens)
                torch.testing.assert_close(
                    valid_assignment.sum(dim=0),
                    torch.ones(
                        valid_tokens.numel(),
                        dtype=assignment.dtype,
                    ),
                )
                token_to_anchor = valid_assignment.argmax(dim=0)
                self.assertTrue(
                    torch.all(
                        token_to_anchor[1:] >= token_to_anchor[:-1]
                    )
                )

        loss = sum(value.square().mean() for value in fpn)
        loss = loss + sum(
            value.square().mean() for value in anchor_fpn
        )
        loss.backward()
        self.assertIsNotNone(tokens.grad)
        self.assertTrue(torch.isfinite(tokens.grad).all())
        self.assertGreater(float(tokens.grad.abs().sum()), 0.0)

    def test_scalar_fallback_uses_actual_branch_depth(self):
        backbone = self.make_backbone(
            depth=4,
            target_keep_ratio=0.7,
            keep_ratio_per_level=None,
        )
        self.assertEqual(
            backbone.keep_ratio_per_level,
            (0.7, 0.7, 0.7, 0.7),
        )
        self.assertEqual(len(backbone.branch), 4)

    def test_invalid_ratio_configuration_is_rejected(self):
        invalid_lists = (
            [],
            [0.9, 0.8],
            [0.9, 0.8, 0.7, 0.6],
            [0.0, 0.8, 0.7],
            [-0.1, 0.8, 0.7],
            [1.01, 0.8, 0.7],
            [math.nan, 0.8, 0.7],
            [math.inf, 0.8, 0.7],
            ["invalid", 0.8, 0.7],
            "0.9,0.8,0.7",
        )
        for ratios in invalid_lists:
            with self.subTest(ratios=ratios):
                with self.assertRaises(ValueError):
                    self.make_backbone(
                        depth=3,
                        keep_ratio_per_level=ratios,
                    )

        for ratio in (0.0, -0.1, 1.01, math.nan, math.inf):
            with self.subTest(target_keep_ratio=ratio):
                with self.assertRaises(ValueError):
                    self.make_backbone(
                        depth=3,
                        target_keep_ratio=ratio,
                    )

    def test_extremely_short_sequences_never_drop_to_zero(self):
        backbone = self.make_backbone(
            depth=4,
            keep_ratio_per_level=(0.1, 0.1, 0.1, 0.1),
        )
        tokens = torch.randn(3, 8, 3)
        mask = torch.tensor(
            [
                [True, False, False],
                [True, True, False],
                [True, True, True],
            ]
        )

        fpn, fpn_masks, anchor_fpn, anchor_masks = backbone(
            tokens, mask
        )
        self.assertEqual(len(fpn), 4)
        self.assertEqual(len(anchor_fpn), 4)
        for level_mask in anchor_masks:
            self.assertEqual(
                level_mask.sum(dim=(1, 2)).tolist(),
                [1, 1, 1],
            )
            self.assertGreater(level_mask.size(-1), 0)

        self.assertEqual(
            backbone.last_progressive_debug[0]["input_len"].tolist(),
            [1, 2, 3],
        )
        for state in backbone.last_progressive_debug:
            self.assertEqual(
                state["anchor_len"].tolist(),
                [1, 1, 1],
            )
        for previous, current in zip(
            backbone.last_progressive_debug,
            backbone.last_progressive_debug[1:],
        ):
            self.assertEqual(
                current["input_len"].tolist(),
                previous["anchor_len"].tolist(),
            )

    def test_legacy_mode_does_not_emit_progressive_state(self):
        backbone = self.make_backbone(
            depth=3,
            adaptive_anchor=False,
        )
        tokens = torch.randn(2, 8, 7)
        mask = torch.ones(2, 7, dtype=torch.bool)
        outputs = backbone(tokens, mask)
        self.assertEqual(len(outputs), 4)
        self.assertIsNone(backbone.last_progressive_debug)


if __name__ == "__main__":
    unittest.main()

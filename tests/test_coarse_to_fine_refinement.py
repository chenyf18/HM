import unittest
import warnings
from unittest import mock

import torch
import torch.nn as nn

from libs.modeling import anchor_mamba
from libs.modeling.coarse_to_fine_refinement import (
    CoarseToFineTemporalRefiner,
)
from libs.modeling.model import HieraMamba
from libs.modeling.video_net import HieraMambaBackbone


class DummyGlobalEncoder(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class CoarseToFineTemporalRefinerTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(41)

    def test_assignment_alignment_is_exact_and_supports_soft_weights(self):
        refiner = CoarseToFineTemporalRefiner(
            d_model=1,
            query_dim=1,
            num_levels=2,
            refine_levels=1,
        )
        coarse = torch.tensor([[[10.0, 20.0]]])
        coarse_mask = torch.tensor([[[True, True]]])
        fine_mask = torch.tensor([[[True, True, True, True]]])
        binary_assignment = torch.tensor(
            [[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]]
        )

        aligned = refiner._align_with_assignment(
            coarse,
            coarse_mask,
            fine_mask,
            binary_assignment,
        )
        torch.testing.assert_close(
            aligned,
            torch.tensor([[[10.0, 10.0, 20.0, 20.0]]]),
        )

        soft_assignment = torch.tensor(
            [[[1.0, 0.25], [0.0, 0.75]]]
        )
        soft_aligned = refiner._align_with_assignment(
            coarse,
            coarse_mask,
            torch.ones(1, 1, 2, dtype=torch.bool),
            soft_assignment,
        )
        torch.testing.assert_close(
            soft_aligned,
            torch.tensor([[[10.0, 17.5]]]),
        )

    def test_mask_aware_interpolation_uses_each_sample_valid_length(self):
        coarse = torch.tensor(
            [
                [[0.0, 10.0, 20.0]],
                [[4.0, 8.0, 999.0]],
            ]
        )
        coarse_mask = torch.tensor(
            [
                [[True, True, True]],
                [[True, True, False]],
            ]
        )
        fine_mask = torch.tensor(
            [
                [[True, True, True, True, True]],
                [[True, True, True, False, False]],
            ]
        )

        aligned = CoarseToFineTemporalRefiner._align_with_interpolation(
            coarse,
            coarse_mask,
            fine_mask,
        )
        expected = torch.tensor(
            [
                [[0.0, 5.0, 10.0, 15.0, 20.0]],
                [[4.0, 6.0, 8.0, 0.0, 0.0]],
            ]
        )
        torch.testing.assert_close(aligned, expected)

    def test_only_requested_levels_are_refined_and_masks_are_applied(self):
        refiner = CoarseToFineTemporalRefiner(
            d_model=4,
            query_dim=3,
            num_levels=4,
            refine_levels=2,
        )
        features = tuple(
            torch.randn(2, 4, length)
            for length in (8, 4, 2, 1)
        )
        masks = (
            torch.tensor(
                [
                    [[True, True, True, True, True, True, True, True]],
                    [[True, True, True, True, True, True, False, False]],
                ]
            ),
            torch.tensor(
                [
                    [[True, True, True, True]],
                    [[True, True, True, False]],
                ]
            ),
            torch.ones(2, 1, 2, dtype=torch.bool),
            torch.ones(2, 1, 1, dtype=torch.bool),
        )
        query = torch.randn(2, 3, 3)
        query_mask = torch.tensor(
            [
                [[True, True, True]],
                [[True, True, False]],
            ]
        )

        outputs = refiner(
            features,
            masks,
            query_feat=query,
            query_mask=query_mask,
        )

        self.assertEqual(
            [output.shape for output in outputs],
            [feature.shape for feature in features],
        )
        self.assertIs(outputs[2], features[2])
        self.assertIs(outputs[3], features[3])
        self.assertTrue(torch.equal(outputs[0][1, :, 6:], torch.zeros(4, 2)))
        self.assertTrue(torch.equal(outputs[1][1, :, 3:], torch.zeros(4, 1)))
        self.assertEqual(
            refiner.last_alignment_modes,
            ("interpolation", "interpolation"),
        )

    def test_query_changes_output_and_backward_is_finite(self):
        refiner = CoarseToFineTemporalRefiner(
            d_model=4,
            query_dim=3,
            num_levels=3,
            refine_levels=2,
        )
        with torch.no_grad():
            refiner.query_projection.weight.fill_(0.2)
            for gate_mlp in refiner.gate_mlps:
                gate_mlp[0].weight.zero_()
                gate_mlp[0].weight[:, 8:, :].fill_(0.5)
                gate_mlp[0].bias.zero_()
                gate_mlp[-1].weight.fill_(0.5)
                gate_mlp[-1].bias.fill_(-1.0)

        base_features = tuple(
            torch.randn(1, 4, length)
            for length in (6, 3, 2)
        )
        masks = tuple(
            torch.ones(1, 1, length, dtype=torch.bool)
            for length in (6, 3, 2)
        )
        query_mask = torch.tensor([[[True, True, False]]])
        query_a = torch.zeros(1, 3, 3)
        query_b = torch.ones(1, 3, 3)
        query_c = torch.ones(1, 3, 3)
        query_b[..., -1] = 1000.0
        query_c[..., -1] = -1000.0

        output_a = refiner(
            base_features,
            masks,
            query_feat=query_a,
            query_mask=query_mask,
        )
        output_b = refiner(
            base_features,
            masks,
            query_feat=query_b,
            query_mask=query_mask,
        )
        output_c = refiner(
            base_features,
            masks,
            query_feat=query_c,
            query_mask=query_mask,
        )
        self.assertFalse(torch.allclose(output_a[0], output_b[0]))
        for expected, actual in zip(output_b, output_c):
            torch.testing.assert_close(expected, actual)
        for output in output_b:
            self.assertTrue(torch.isfinite(output).all())

        features = tuple(
            feature.detach().clone().requires_grad_(True)
            for feature in base_features
        )
        query = torch.randn(1, 3, 3, requires_grad=True)
        outputs = refiner(
            features,
            masks,
            query_feat=query,
            query_mask=query_mask,
        )
        loss = sum(output.square().mean() for output in outputs)
        loss.backward()

        for feature in features:
            self.assertIsNotNone(feature.grad)
            self.assertTrue(torch.isfinite(feature.grad).all())
            self.assertGreater(float(feature.grad.abs().sum()), 0.0)
        self.assertIsNotNone(query.grad)
        self.assertTrue(torch.isfinite(query.grad).all())
        self.assertGreater(float(query.grad.abs().sum()), 0.0)
        parameter_grads = [
            parameter.grad for parameter in refiner.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(parameter_grads)
        for gradient in parameter_grads:
            self.assertTrue(torch.isfinite(gradient).all())

    def test_invalid_refinement_configuration_is_rejected(self):
        invalid_options = (
            {"d_model": 0, "query_dim": 3, "num_levels": 3, "refine_levels": 1},
            {"d_model": 4, "query_dim": 0, "num_levels": 3, "refine_levels": 1},
            {"d_model": 4, "query_dim": 3, "num_levels": 1, "refine_levels": 1},
            {"d_model": 4, "query_dim": 3, "num_levels": 3, "refine_levels": 0},
            {"d_model": 4, "query_dim": 3, "num_levels": 3, "refine_levels": 3},
            {"d_model": 4, "query_dim": 3, "num_levels": 3, "refine_levels": 1.5},
        )
        for options in invalid_options:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    CoarseToFineTemporalRefiner(**options)


class CoarseToFineBackboneIntegrationTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(43)

    @staticmethod
    def make_backbone(
        adaptive_anchor,
        coarse_to_fine_refine=True,
        refine_levels=2,
        block_type="AnchorMambaPoolingBlockGated",
    ):
        options = {
            "in_dim": 8,
            "embd_dim": 8,
            "max_seq_len": 16,
            "n_heads": 2,
            "stride": 1,
            "arch": (0, 0, 4),
            "use_abs_pe": False,
            "local_encode": False,
            "return_anchor": True,
            "block_type": block_type,
            "query_dim": 6,
            "adaptive_anchor": adaptive_anchor,
            "target_keep_ratio": 0.75,
            "importance_hidden_dim": 8,
            "coarse_to_fine_refine": coarse_to_fine_refine,
            "refine_levels": refine_levels,
        }
        with mock.patch.object(
            anchor_mamba, "Hydra", DummyGlobalEncoder
        ):
            backbone = HieraMambaBackbone(**options)
        return backbone.eval()

    @staticmethod
    def inputs():
        video = torch.randn(2, 8, 10)
        video_mask = torch.tensor(
            [
                [True, True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, False, False, False],
            ]
        )
        query = torch.randn(2, 6, 4)
        query_mask = torch.tensor(
            [
                [True, True, True, True],
                [True, True, False, False],
            ]
        )
        return video, video_mask, query, query_mask

    def test_adaptive_path_collects_assignments_internally(self):
        backbone = self.make_backbone(adaptive_anchor=True)
        video, video_mask, query, query_mask = self.inputs()
        video.requires_grad_(True)
        query.requires_grad_(True)

        outputs = backbone(video, video_mask, query, query_mask)

        self.assertEqual(len(outputs), 4)
        loss = sum(feature.square().mean() for feature in outputs[0])
        loss.backward()
        self.assertIsNotNone(video.grad)
        self.assertIsNotNone(query.grad)
        self.assertTrue(torch.isfinite(video.grad).all())
        self.assertTrue(torch.isfinite(query.grad).all())
        self.assertGreater(float(video.grad.abs().sum()), 0.0)
        self.assertGreater(float(query.grad.abs().sum()), 0.0)
        refiner_grads = [
            parameter.grad
            for parameter in backbone.coarse_to_fine_refiner.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(refiner_grads)
        self.assertTrue(all(torch.isfinite(grad).all() for grad in refiner_grads))
        self.assertEqual(
            backbone.coarse_to_fine_refiner.last_alignment_modes,
            ("assignment", "assignment"),
        )
        self.assertTrue(backbone.amp_query_conditioned)
        self.assertTrue(backbone.query_conditioned)
        explicit_outputs = backbone(
            video,
            video_mask,
            query,
            query_mask,
            return_anchor_assignments=True,
        )
        self.assertEqual(len(explicit_outputs), 5)
        self.assertEqual(len(explicit_outputs[-1]), 4)

    def test_legacy_path_uses_mask_aware_interpolation(self):
        backbone = self.make_backbone(adaptive_anchor=False)
        video, video_mask, query, query_mask = self.inputs()

        fpn, fpn_masks, anchor_fpn, anchor_masks = backbone(
            video,
            video_mask,
            query,
            query_mask,
        )

        self.assertEqual(len(fpn), 4)
        self.assertEqual(len(anchor_fpn), 4)
        self.assertEqual(
            backbone.coarse_to_fine_refiner.last_alignment_modes,
            ("interpolation", "interpolation"),
        )
        self.assertFalse(backbone.amp_query_conditioned)
        self.assertTrue(backbone.query_conditioned)
        for feature, mask in zip(fpn[:2], fpn_masks[:2]):
            self.assertTrue(
                torch.equal(
                    feature.masked_select(~mask.expand_as(feature)),
                    torch.zeros_like(
                        feature.masked_select(~mask.expand_as(feature))
                    ),
                )
            )

    def test_refinement_can_wrap_non_query_amp_block(self):
        backbone = self.make_backbone(
            adaptive_anchor=False,
            block_type="AnchorMambaPoolingBlock",
        )
        self.assertFalse(backbone.amp_query_conditioned)
        self.assertTrue(backbone.query_conditioned)
        outputs = backbone(*self.inputs())
        self.assertEqual(len(outputs), 4)

    def test_disabled_path_adds_no_parameters_and_keeps_interface(self):
        backbone = self.make_backbone(
            adaptive_anchor=False,
            coarse_to_fine_refine=False,
        )
        self.assertIsNone(backbone.coarse_to_fine_refiner)
        self.assertFalse(backbone.query_conditioned)
        self.assertFalse(
            any(
                "coarse_to_fine_refiner" in key
                for key in backbone.state_dict()
            )
        )
        outputs = backbone(*self.inputs())
        self.assertEqual(len(outputs), 4)

    def test_baseline_checkpoint_loads_with_refiner_enabled(self):
        baseline = self.make_backbone(
            adaptive_anchor=False,
            coarse_to_fine_refine=False,
        )
        enabled = self.make_backbone(adaptive_anchor=False)

        class TinyModel(HieraMamba):
            pass

        baseline_model = TinyModel.__new__(TinyModel)
        nn.Module.__init__(baseline_model)
        baseline_model.query_conditioned = False
        baseline_model.backbone = baseline

        enabled_model = TinyModel.__new__(TinyModel)
        nn.Module.__init__(enabled_model)
        enabled_model.query_conditioned = True
        enabled_model.backbone = enabled

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            incompatible = enabled_model.load_compatible_state_dict(
                baseline_model.state_dict()
            )
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(
            all(
                ".coarse_to_fine_refiner." in key
                for key in incompatible.missing_keys
            )
        )
        self.assertFalse(incompatible.unexpected_keys)


if __name__ == "__main__":
    unittest.main()

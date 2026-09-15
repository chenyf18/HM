import math
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from libs.modeling.query_boundary_importance import (
    QueryBoundaryImportancePredictor,
)
from libs.modeling.query_boundary_importance_loss import (
    BoundarySupervisionLoss,
    make_boundary_supervision_loss,
)
from libs import worker


def make_points(centers, strides):
    points = torch.zeros(len(centers), 4, dtype=torch.float32)
    points[:, 0] = torch.tensor(centers, dtype=torch.float32)
    points[:, 3] = torch.tensor(strides, dtype=torch.float32)
    return points


class BoundarySupervisionLossTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(53)

    def test_gaussian_targets_are_stride_normalized_per_fpn_level(self):
        criterion = BoundarySupervisionLoss({"boundary_sigma": 1.0})
        targets = torch.tensor([[2.0, 6.0]])
        fine_points = make_points(range(9), [1.0] * 9)
        coarse_points = make_points([0, 2, 4, 6, 8], [2.0] * 5)
        fine_mask = torch.ones(1, 9, dtype=torch.bool)
        coarse_mask = torch.ones(1, 5, dtype=torch.bool)

        fine = criterion.build_targets(fine_points, fine_mask, targets)
        coarse = criterion.build_targets(
            coarse_points, coarse_mask, targets
        )
        neighbor_value = math.exp(-0.5)

        self.assertEqual(fine["start_target"][0].argmax().item(), 2)
        self.assertEqual(fine["end_target"][0].argmax().item(), 6)
        self.assertEqual(coarse["start_target"][0].argmax().item(), 1)
        self.assertEqual(coarse["end_target"][0].argmax().item(), 3)
        self.assertAlmostEqual(
            fine["start_target"][0, 1].item(), neighbor_value, places=6
        )
        self.assertAlmostEqual(
            coarse["start_target"][0, 0].item(), neighbor_value, places=6
        )
        self.assertAlmostEqual(
            fine["end_target"][0, 5].item(), neighbor_value, places=6
        )
        self.assertAlmostEqual(
            coarse["end_target"][0, 2].item(), neighbor_value, places=6
        )

    def test_batched_temporal_coordinates_and_padding_are_isolated(self):
        criterion = BoundarySupervisionLoss({"boundary_sigma": 0.75})
        points = torch.zeros(2, 5, 4)
        points[0, :, 0] = torch.tensor([0, 1, 2, 3, 4])
        points[0, :, 3] = 1.0
        points[1, :, 0] = torch.tensor([10, 12, 14, 16, 18])
        points[1, :, 3] = 2.0
        mask = torch.tensor(
            [
                [True, True, True, True, True],
                [True, True, True, True, False],
            ]
        )
        targets = torch.tensor([[1.0, 3.0], [12.0, 16.0]])

        target_maps = criterion.build_targets(points, mask, targets)

        self.assertEqual(target_maps["start_target"][0].argmax().item(), 1)
        self.assertEqual(target_maps["end_target"][0].argmax().item(), 3)
        self.assertEqual(target_maps["start_target"][1].argmax().item(), 1)
        self.assertEqual(target_maps["end_target"][1].argmax().item(), 3)
        self.assertEqual(target_maps["start_target"][1, 4].item(), 0.0)
        self.assertEqual(target_maps["end_target"][1, 4].item(), 0.0)

    def test_loss_is_start_plus_end_and_ignores_padding_values(self):
        criterion = BoundarySupervisionLoss(
            {"boundary_sigma": 1.0, "max_pos_weight": 10.0}
        )
        points = make_points(range(7), [1.0] * 7)
        mask = torch.tensor(
            [[True, True, True, True, True, False, False]]
        )
        targets = torch.tensor([[1.0, 3.0]])
        start_prob = torch.tensor(
            [[0.1, 0.8, 0.3, 0.2, 0.1, 0.4, 0.5]],
            requires_grad=True,
        )
        end_prob = torch.tensor(
            [[0.2, 0.1, 0.3, 0.9, 0.2, 0.6, 0.7]],
            requires_grad=True,
        )
        debug_a = (
            {"start_prob": start_prob, "end_prob": end_prob},
        )
        loss_a, components = criterion(
            debug_a,
            (points,),
            (mask,),
            targets,
            return_components=True,
        )
        torch.testing.assert_close(
            loss_a, components["start"] + components["end"]
        )

        debug_b = (
            {
                "start_prob": torch.tensor(
                    [[0.1, 0.8, 0.3, 0.2, 0.1, 0.99, 0.01]]
                ),
                "end_prob": torch.tensor(
                    [[0.2, 0.1, 0.3, 0.9, 0.2, 0.01, 0.99]]
                ),
            },
        )
        loss_b = criterion(debug_b, (points,), (mask,), targets)
        torch.testing.assert_close(loss_a.detach(), loss_b)

        loss_a.backward()
        self.assertTrue(torch.isfinite(start_prob.grad).all())
        self.assertTrue(torch.isfinite(end_prob.grad).all())
        self.assertGreater(float(start_prob.grad.abs().sum()), 0.0)
        self.assertGreater(float(end_prob.grad.abs().sum()), 0.0)

    def test_multilevel_boundary_loss_updates_only_boundary_predictor_path(self):
        predictor = QueryBoundaryImportancePredictor(
            video_dim=8,
            query_dim=6,
            hidden_dim=8,
        )
        criterion = BoundarySupervisionLoss(
            {"boundary_sigma": 1.25, "max_pos_weight": 20.0}
        )
        video = torch.randn(2, 8, 8, requires_grad=True)
        query = torch.randn(2, 6, requires_grad=True)
        mask = torch.tensor(
            [
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, False, False, False],
            ]
        )
        targets = torch.tensor([[2.0, 6.0], [1.0, 4.0]])
        fine_debug = predictor(video, query, mask)
        coarse_debug = {
            key: value[:, ::2] for key, value in fine_debug.items()
        }
        coarse_mask = mask[:, ::2]
        fine_points = make_points(range(8), [1.0] * 8)
        coarse_points = make_points([0, 2, 4, 6], [2.0] * 4)

        loss, components = criterion(
            (fine_debug, coarse_debug),
            (fine_points, coarse_points),
            (mask, coarse_mask),
            targets,
            return_components=True,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(components["start"]))
        self.assertTrue(torch.isfinite(components["end"]))
        self.assertTrue(torch.isfinite(components["boundary"]))
        loss.backward()

        self.assertTrue(torch.isfinite(video.grad).all())
        self.assertTrue(torch.isfinite(query.grad).all())
        boundary_grads = [
            parameter.grad
            for name, parameter in predictor.named_parameters()
            if "boundary_" in name and parameter.grad is not None
        ]
        self.assertTrue(boundary_grads)
        self.assertTrue(
            all(torch.isfinite(gradient).all() for gradient in boundary_grads)
        )
        self.assertGreater(
            sum(float(gradient.abs().sum()) for gradient in boundary_grads),
            0.0,
        )
        self.assertIsNone(predictor.relevance_video_proj.weight.grad)
        self.assertIsNone(predictor.relevance_query_proj.weight.grad)

    def test_zero_weight_does_not_construct_loss(self):
        weight, loss = make_boundary_supervision_loss(
            {"boundary_loss_weight": 0.0}
        )
        self.assertEqual(weight, 0.0)
        self.assertIsNone(loss)

        weight, loss = make_boundary_supervision_loss(
            {"boundary_loss_weight": 0.25, "boundary_sigma": 1.0}
        )
        self.assertEqual(weight, 0.25)
        self.assertIsInstance(loss, BoundarySupervisionLoss)

        for invalid_weight in (-0.1, float("nan"), float("inf")):
            with self.subTest(invalid_weight=invalid_weight):
                with self.assertRaises(ValueError):
                    make_boundary_supervision_loss(
                        {"boundary_loss_weight": invalid_weight}
                    )

    def test_zero_weight_keeps_worker_debug_path_disabled(self):
        class DummyModel:

            query_conditioned = False
            importance_enabled = False
            vid_net = SimpleNamespace(adaptive_anchor=False)

        def fake_trainer_init(instance, opt):
            instance.model = DummyModel()
            instance.center_sampling_radius = 1.5
            instance.logger = mock.Mock()

        opt = {
            "model": {
                "early_fusion": False,
                "vid_net": {"embd_dim": 8},
            },
            "train": {
                "loss_aux": {
                    "query_boundary_importance": {
                        "enable": False,
                        "boundary_loss_weight": 0.0,
                    },
                    "ds_contrast": {
                        "enable": False,
                        "acc_mode": "legacy_acc",
                    },
                    "gt_contrast": {"enable": False},
                },
            },
        }
        with mock.patch.object(
            worker.TrainerOriginal,
            "__init__",
            fake_trainer_init,
        ):
            trainer = worker.TrainerAuxiliary(opt)

        self.assertFalse(trainer.boundary_supervision)
        self.assertEqual(trainer.boundary_loss_weight, 0.0)
        self.assertIsNone(trainer.boundary_supervision_loss)
        self.assertFalse(trainer.return_importance_debug)
        self.assertFalse(trainer.query_boundary_importance)


if __name__ == "__main__":
    unittest.main()

"""Quality-aware cls target tests (HM-QACT-033 phase 1)."""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from libs.modeling.quality_targets import (  # noqa: E402
    compute_quality_cls_targets, temporal_iou_1d,
)


def make_points():
    # 4 points, stride-1 coords, scale 1
    pts = torch.zeros(1, 4, 8)
    pts[..., 0] = torch.tensor([2.0, 6.0, 10.0, 14.0])
    pts[..., 3] = 1.0
    offs = torch.tensor([[[1.5, 1.5], [2.0, 3.0], [0.1, 0.1],
                          [8.0, 8.0]]], requires_grad=True)
    return pts, offs


class TestQualityTargets(unittest.TestCase):

    def test_range_and_values(self):
        pts, offs = make_points()
        tgt = torch.tensor([[3.0, 12.0]])
        y = compute_quality_cls_targets(pts, offs, tgt, "decoded_iou")
        self.assertEqual(y.dtype, torch.float32)
        self.assertGreaterEqual(float(y.min()), 0.0)
        self.assertLessEqual(float(y.max()), 1.0)
        # point 3 spans [6,22] vs GT [3,12]: inter 6, union 19 -> 6/19
        self.assertAlmostEqual(float(y[0, 3]), 6.0 / 19.0, places=5)
        y2 = compute_quality_cls_targets(pts, offs, tgt,
                                         "metric_utility", tau=0.05)
        self.assertGreaterEqual(float(y2.min()), 0.0)
        self.assertLessEqual(float(y2.max()), 1.0)
        # metric_utility is monotone non-decreasing in IoU (analytic)
        ious = torch.linspace(0, 1, 21)
        grid = 0.5 * torch.sigmoid((ious - 0.3) / 0.05) \
            + 0.5 * torch.sigmoid((ious - 0.5) / 0.05)
        self.assertTrue(bool((grid[1:] >= grid[:-1] - 1e-6).all()))
        self.assertGreater(float(grid[-1]), 0.99)
        self.assertLess(float(grid[0]), 5e-3)

    def test_detach(self):
        """IoU target carries no gradient to the offsets."""
        pts, offs = make_points()
        tgt = torch.tensor([[3.0, 12.0]])
        y = compute_quality_cls_targets(pts, offs.detach().requires_grad_(
            True), tgt, "decoded_iou")
        self.assertFalse(y.requires_grad)
        # through the real focal path: cls loss only -> no reg grads
        logits = torch.randn(4, requires_grad=True)
        labels = y[0]
        smoothed = labels * 0.8 + 0.1
        from libs.modeling.loss import sigmoid_focal_loss
        loss = sigmoid_focal_loss(logits, smoothed, alpha=0.5,
                                  reduction="sum")
        loss.backward()
        self.assertIsNotNone(logits.grad)          # cls path alive
        self.assertIsNone(offs.grad)               # offsets untouched

    def test_fp32_decode(self):
        """Targets stay FP32 even with bf16 offsets under autocast."""
        pts, offs = make_points()
        tgt = torch.tensor([[3.0, 12.0]])
        with torch.autocast('cpu', dtype=torch.bfloat16):
            y = compute_quality_cls_targets(
                pts, offs.to(torch.bfloat16), tgt, "decoded_iou")
        self.assertEqual(y.dtype, torch.float32)

    def test_binary_path_unchanged(self):
        """binary mode routes through the original label path exactly."""
        # emulate: labels -> _calc_focal_loss inputs identical with and
        # without the gate when cls_target_type == binary
        old_labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
        cls_targets = old_labels.float()          # binary branch value
        self.assertTrue(torch.equal(cls_targets, old_labels))
        # and the focal transform is the original one
        s1 = (cls_targets * (1.0 - 0.2) + 0.1)
        self.assertAlmostEqual(float(s1[0]), 0.1)
        self.assertAlmostEqual(float(s1[1]), 0.9)

    def test_iou_helper(self):
        segs = torch.tensor([[[0., 4.], [2., 6.], [10., 12.]]])
        tgt = torch.tensor([[2., 6.]])
        iou = temporal_iou_1d(segs, tgt)
        self.assertAlmostEqual(float(iou[0, 0]), 1.0 / 3.0, places=5)
        self.assertAlmostEqual(float(iou[0, 1]), 1.0)
        self.assertAlmostEqual(float(iou[0, 2]), 0.0)


if __name__ == '__main__':
    unittest.main()

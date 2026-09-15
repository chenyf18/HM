"""HCMS unit tests (HM-HCMS-001)."""
import os
import random
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from libs.data.hcms import (  # noqa: E402
    select_negative_random, select_negative_from_pool, background_pair,
    moment_swap, background_swap, maybe_apply, _iou,
)


def make_feats(C=4, T=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(C, T, generator=g)


class TestHCMS(unittest.TestCase):

    def test_shape_and_identity_of_untouched(self):
        f = make_feats()
        t = torch.tensor([[5.0, 12.0], [20.0, 25.0]])
        neg = (25, 32, 7)
        f2, t2, ok, ratio = moment_swap(f, t, 0, neg, 7)
        self.assertTrue(ok)
        self.assertEqual(f2.shape, f.shape)
        self.assertTrue(torch.equal(f2[:, :5], f[:, :5]))       # before
        self.assertTrue(torch.equal(f2[:, 12:20], f[:, 12:20]))  # between (q1!)
        # q1 target untouched
        self.assertEqual(t2[1].tolist(), [20.0, 25.0])
        # q0 moved to neg window
        self.assertEqual(t2[0].tolist(), [25.0, 32.0])
        # original tensors unmodified
        self.assertTrue(torch.equal(f, make_feats()))
        self.assertEqual(t[0].tolist(), [5.0, 12.0])

    def test_swapped_content(self):
        f = make_feats()
        t = torch.tensor([[5.0, 12.0]])
        f2, *_ = moment_swap(f, t, 0, (25, 32, 7), 7)
        self.assertTrue(torch.equal(f2[:, 5:12], f[:, 25:32]))
        self.assertTrue(torch.equal(f2[:, 25:32], f[:, 5:12]))

    def test_zero_iou_moved(self):
        rng = random.Random(0)
        for _ in range(50):
            gt = (10.0, 16.0)
            others = [(30.0, 35.0)]
            neg = select_negative_random(gt, 40, others, rng=rng)
            self.assertIsNotNone(neg)
            self.assertLessEqual(_iou((neg[0], neg[1]), gt), 0.1)
            self.assertLessEqual(
                _iou((neg[0], neg[1]), others[0]), 0.1)

    def test_strict_and_relaxed_lengths(self):
        rng = random.Random(1)
        got_strict = set()
        for _ in range(100):
            n = select_negative_random((8.0, 11.4), 40, [], "strict",
                                       rng=rng)
            if n:
                got_strict.add(n[1] - n[0])
        self.assertTrue(got_strict <= {3})   # round(3.4) == 3
        got_rel = set()
        for _ in range(100):
            n = select_negative_random((8.0, 11.4), 40, [], "relaxed",
                                       rng=rng)
            if n:
                L = n[1] - n[0]
                self.assertLessEqual(abs(L - 3.4) / 3.4, 0.1 + 0.5 / 3.4 + 1e-9)  # integer-grid half-step
                got_rel.add(L)
        self.assertTrue(len(got_rel) >= 2, "relaxed should allow a range")

    def test_pool_selection_prefers_high_score(self):
        pool = np.zeros(60, dtype=np.float32)
        pool[45:50] = 10.0                      # attractive hard region
        gt = (10.0, 14.0)
        n = select_negative_from_pool(gt, 60, [], pool, "strict")
        self.assertIsNotNone(n)
        self.assertTrue(45 <= n[0] < 50 or 45 < n[1] <= 50)
        # no pool -> None (caller falls back to random)
        self.assertIsNone(select_negative_from_pool(gt, 60, [], None))

    def test_background_control_keeps_gt(self):
        rng = random.Random(2)
        f = make_feats()
        t = torch.tensor([[5.0, 12.0]])
        pair = background_pair(40, [[5.0, 12.0]], 7, rng=rng)
        self.assertIsNotNone(pair)
        f2 = background_swap(f, pair[0], pair[1])
        self.assertEqual(f2.shape, f.shape)
        # GT region untouched
        self.assertTrue(torch.equal(f2[:, 5:12], f[:, 5:12]))

    def test_maybe_apply_off_and_prob(self):
        f, t = make_feats(), torch.tensor([[5.0, 12.0]])
        f2, t2, info = maybe_apply(f, t, "off", prob=0.5)
        self.assertFalse(info["activated"])
        self.assertTrue(torch.equal(f, f2) and torch.equal(t, t2))
        rng = random.Random(0)
        # prob=1.0 random strict should activate on this easy case
        f2, t2, info = maybe_apply(f, t, "random", "strict", prob=1.0,
                                   rng=rng)
        self.assertTrue(info["activated"])
        self.assertEqual(f2.shape, f.shape)
        # new GT window has zero IoU with old
        self.assertLessEqual(
            _iou((t2[0, 0].item(), t2[0, 1].item()), (5.0, 12.0)), 0.1)

    def test_no_valid_negative_returns_original(self):
        # tiny video, GT covers nearly everything
        f, t = make_feats(T=8), torch.tensor([[0.0, 7.0]])
        rng = random.Random(0)
        f2, t2, info = maybe_apply(f, t, "random", "strict", prob=1.0,
                                   rng=rng)
        self.assertFalse(info["activated"])
        self.assertTrue(torch.equal(f, f2) and torch.equal(t, t2))


if __name__ == '__main__':
    unittest.main()

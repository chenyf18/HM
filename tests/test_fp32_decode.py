"""Correctness tests for the FP32 temporal-coordinate decode fix
(HM-LQAC-022 Stage 0). decode_offsets must always perform coordinate
arithmetic in float32, regardless of the incoming offset dtype, so that
bf16 autocast evaluation no longer quantizes decoded segment coordinates
onto the bf16 grid (HM-AUDIT-RANK-021).
"""
import math
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from libs.modeling.temporal_coordinates import decode_offsets  # noqa: E402


def _bf16(x):
    return x.to(torch.bfloat16)


def _legacy_bf16_decode(points, offsets):
    """Reproduce the pre-fix behaviour: cast centers/scales to the (bf16)
    offset dtype and do the arithmetic in bf16, one rounding per op."""
    centers = _bf16(points[:, 0].float())
    scales = _bf16(points[:, 3].float())
    off = _bf16(offsets.float())
    left = _bf16(off[:, 0] * scales)
    right = _bf16(off[:, 1] * scales)
    start = _bf16(centers - left)
    end = _bf16(centers + right)
    return torch.stack((start.float(), end.float()), dim=-1)


class TestFP32Decode(unittest.TestCase):

    def _points(self, centers, scales):
        pts = torch.zeros(len(centers), 4, dtype=torch.float32)
        pts[:, 0] = torch.as_tensor(centers, dtype=torch.float32)
        pts[:, 3] = torch.as_tensor(scales, dtype=torch.float32)
        return pts

    def test_t1_bf16_offsets_yield_fp32_output(self):
        points = self._points([10.0, 100.0], [2.0, 4.0])
        offsets = torch.tensor([[0.5, 1.0], [0.25, 0.75]],
                               dtype=torch.bfloat16)
        decoded = decode_offsets(points, offsets)
        self.assertEqual(decoded.dtype, torch.float32)

    def test_t2_matches_fp32_reference(self):
        centers = [3.0, 127.5, 1024.7, 2049.3]
        scales = [1.0, 2.0, 8.0, 16.0]
        offsets = torch.tensor(
            [[0.31, 1.72], [0.93, 0.11], [1.37, 0.58], [0.42, 0.67]],
            dtype=torch.float32,
        )
        decoded = decode_offsets(self._points(centers, scales), offsets)
        ref = torch.stack(
            (torch.tensor(centers) - offsets[:, 0] * torch.tensor(scales),
             torch.tensor(centers) + offsets[:, 1] * torch.tensor(scales)),
            dim=-1,
        )
        self.assertEqual(decoded.dtype, ref.dtype)
        self.assertTrue(torch.equal(decoded, ref))

    def test_t3_legacy_bf16_decode_quantizes_large_coordinates(self):
        # Large temporal indices: bf16 mantissa (8 bits) gives a spacing of
        # 4 tokens above 512 and 8 tokens above 1024. The legacy decode
        # must show that quantization; the fixed decode must not.
        centers = [520.3, 1030.7, 2050.1]
        scales = [8.0, 8.0, 16.0]
        offsets = torch.tensor(
            [[0.37, 1.13], [0.61, 0.89], [0.53, 0.47]], dtype=torch.float32
        )
        points = self._points(centers, scales)
        legacy = _legacy_bf16_decode(points, _bf16(offsets))
        fixed = decode_offsets(points, _bf16(offsets))
        # reference: same (bf16-quantized) offsets, but exact fp32 arithmetic
        ref = decode_offsets(points, _bf16(offsets).float())
        legacy_err = (legacy - ref).abs().max()
        fixed_err = (fixed - ref).abs().max()
        self.assertGreater(float(legacy_err), 2.0)
        self.assertLess(float(fixed_err), 1e-6)

    def test_t4_candidate_count_and_length_semantics(self):
        # Shape/count logic is unchanged: N points in, N segments out, and
        # segment lengths follow the same positivity semantics in fp32.
        n = 257
        centers = [1.7 * i for i in range(n)]
        scales = [2.0 ** (i % 8) for i in range(n)]
        offsets = torch.rand(n, 2, dtype=torch.float32) * 2
        points = self._points(centers, scales)
        decoded = decode_offsets(points, offsets)
        self.assertEqual(decoded.shape, (n, 2))
        lengths = decoded[:, 1] - decoded[:, 0]
        ref = (offsets[:, 0] + offsets[:, 1]) * torch.tensor(scales)
        self.assertTrue(torch.allclose(lengths, ref, atol=1e-4))
        self.assertTrue((lengths > 0).all())

    def test_batched_shapes_preserved(self):
        b, t = 3, 64
        points = self._points(
            [float(i) for i in range(t)], [2.0] * t
        )
        offsets = torch.rand(b, t, 2, dtype=torch.bfloat16)
        decoded = decode_offsets(points, offsets)
        self.assertEqual(tuple(decoded.shape), (b, t, 2))
        self.assertEqual(decoded.dtype, torch.float32)

    def test_3d_points(self):
        b, t = 2, 32
        pts = torch.zeros(b, t, 8, dtype=torch.float32)
        pts[..., 0] = torch.arange(t).float()
        pts[..., 3] = 4.0
        offsets = torch.rand(b, t, 2, dtype=torch.float32)
        decoded = decode_offsets(pts, offsets)
        self.assertEqual(decoded.dtype, torch.float32)
        expected = torch.stack(
            (pts[..., 0] - offsets[..., 0] * 4.0,
             pts[..., 0] + offsets[..., 1] * 4.0), dim=-1
        )
        self.assertTrue(torch.allclose(decoded, expected, atol=1e-6))


if __name__ == '__main__':
    unittest.main()

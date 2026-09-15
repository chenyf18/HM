import unittest

import torch

from libs.modeling.model import PtGenerator
from libs.modeling.temporal_coordinates import (
    AdaptiveTemporalPointGenerator,
    decode_offsets,
    encode_offsets,
)


class TemporalCoordinateTest(unittest.TestCase):
    def test_grouping_metadata_and_round_trip(self):
        mask = torch.ones(1, 8, dtype=torch.bool)
        metadata = AdaptiveTemporalPointGenerator.make_initial_metadata(mask)
        assignment = torch.zeros(1, 4, 8)
        assignment[0, 0, :3] = 1
        assignment[0, 1, 3] = 1
        assignment[0, 2, 4:6] = 1
        assignment[0, 3, 6:8] = 1
        anchor_mask = torch.ones(1, 1, 4, dtype=torch.bool)
        propagated = AdaptiveTemporalPointGenerator.propagate(
            metadata, assignment, mask, anchor_mask
        )
        expected = torch.tensor([
            [-0.5, 2.5, 1.0, 3.0],
            [2.5, 3.5, 3.0, 1.0],
            [3.5, 5.5, 4.5, 2.0],
            [5.5, 7.5, 6.5, 2.0],
        ])
        torch.testing.assert_close(propagated[0, :, :4], expected)
        generator = AdaptiveTemporalPointGenerator(
            max_seq_len=8, num_fpn_levels=1, regression_range=4, sigma=0.5
        )
        points = generator((propagated,), (anchor_mask,))[0]
        self.assertFalse(torch.allclose(points[0, :, 3], points[0, :, 7]))
        targets = torch.tensor([[0.75, 6.25]])
        offsets = encode_offsets(points, targets)
        decoded = decode_offsets(points, offsets)
        torch.testing.assert_close(decoded[0], targets.expand(4, -1))

    def test_batch_padding_and_fail_fast_invariants(self):
        mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
        metadata = AdaptiveTemporalPointGenerator.make_initial_metadata(mask)
        assignment = torch.zeros(2, 3, 4)
        assignment[0, 0, :2] = 1
        assignment[0, 1, 2] = 1
        assignment[0, 2, 3] = 1
        assignment[1, 0, :2] = 1
        anchor_mask = torch.tensor([[[True, True, True]], [[True, False, False]]])
        assignment[1, 1, 2] = 1
        with self.assertRaises(ValueError):
            AdaptiveTemporalPointGenerator.propagate(
                metadata, assignment, mask, anchor_mask
            )
        assignment[1, 1, 2] = 0
        assignment[1, 2, 3:] = 0
        propagated = AdaptiveTemporalPointGenerator.propagate(
            metadata, assignment, mask, anchor_mask
        )
        self.assertTrue(torch.isfinite(propagated).all())
        with self.assertRaises(ValueError):
            AdaptiveTemporalPointGenerator.propagate(
                metadata, torch.zeros_like(assignment), mask, anchor_mask
            )

    def test_tensorized_propagation_matches_reference(self):
        torch.manual_seed(103)
        mask = torch.tensor(
            [
                [True] * 12,
                [True] * 7 + [False] * 5,
                [True] + [False] * 11,
                [False] * 12,
            ]
        )
        metadata = AdaptiveTemporalPointGenerator.make_initial_metadata(mask)
        assignment = torch.zeros(4, 6, 12)
        assignment[0, 0, :2] = 1
        assignment[0, 1, 2:4] = 1
        assignment[0, 2, 4:6] = 1
        assignment[0, 3, 6:8] = 1
        assignment[0, 4, 8:10] = 1
        assignment[0, 5, 10:12] = 1
        assignment[1, 0, :2] = 1
        assignment[1, 1, 2:4] = 1
        assignment[1, 2, 4:6] = 1
        assignment[1, 3, 6:7] = 1
        assignment[2, 0, :1] = 1
        anchor_mask = torch.tensor(
            [
                [[True] * 6],
                [[True] * 4 + [False] * 2],
                [[True] + [False] * 5],
                [[False] * 6],
            ]
        )
        actual = AdaptiveTemporalPointGenerator.propagate(
            metadata, assignment, mask, anchor_mask
        )
        reference = AdaptiveTemporalPointGenerator.reference_propagate(
            metadata, assignment, mask, anchor_mask
        )
        torch.testing.assert_close(actual, reference, atol=0.0, rtol=0.0)
        self.assertTrue(torch.isfinite(actual).all())

    def test_fixed_generator_fail_fast_switch(self):
        generator = PtGenerator(
            max_seq_len=8, num_fpn_levels=2, allow_fixed_stride=False
        )
        with self.assertRaises(RuntimeError):
            generator([8, 4])


if __name__ == "__main__":
    unittest.main()

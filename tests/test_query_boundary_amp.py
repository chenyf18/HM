import unittest
from unittest import mock

import torch
import torch.nn as nn

from libs.modeling import anchor_mamba
from libs.modeling.anchor_mamba import AnchorMambaPoolingBlockGated
from libs.modeling.model import HieraMamba
from libs.modeling.query_boundary_importance import (
    QueryBoundaryImportancePredictor,
)
from libs.modeling.video_net import HieraMambaBackbone
from libs.modeling.query_boundary_importance_loss import (
    QueryBoundaryImportanceLoss,
)


class DummyGlobalEncoder(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


def make_block(**kwargs):
    options = {
        'stride': 2,
        'd_model': 8,
        'nhead': 2,
        'dropout': 0.0,
        'ffn_ratio': 2,
        'local_encode': False,
        'pool_method': 'mean',
        'bidirectional': True,
    }
    options.update(kwargs)
    with mock.patch.object(anchor_mamba, 'Hydra', DummyGlobalEncoder):
        block = AnchorMambaPoolingBlockGated(**options)
    return block.eval()


class QueryModulatedAMPTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(7)

    def test_disabled_path_is_query_agnostic_and_checkpoint_compatible(self):
        block = make_block()
        x = torch.randn(2, 8, 7)
        mask = torch.tensor(
            [
                [[True, True, True, True, True, True, True]],
                [[True, True, True, True, False, False, False]],
            ]
        )
        query = torch.randn(2, 6, 3)
        query_mask = torch.ones(2, 1, 3, dtype=torch.bool)

        baseline = block(x.clone(), mask.clone())
        with_query = block(
            x.clone(),
            mask.clone(),
            query_feat=query,
            query_mask=query_mask,
        )

        self.assertEqual(len(baseline), 4)
        self.assertEqual(baseline[0].shape, (2, 8, 4))
        self.assertEqual(baseline[1].shape, (2, 8, 7))
        for expected, actual in zip(baseline, with_query):
            self.assertTrue(torch.equal(expected, actual))

        state = block.state_dict()
        self.assertFalse(any('query_' in key for key in state))
        self.assertFalse(any('importance_predictor' in key for key in state))
        reloaded = make_block()
        reloaded.load_state_dict(state, strict=True)

    def test_baseline_state_loads_into_query_conditioned_model(self):
        baseline = make_block()
        enabled = make_block(
            query_dim=6,
            query_modulation=True,
            query_aware_gate=True,
            query_boundary_importance=True,
            importance_hidden_dim=8,
        )

        class TinyModel(HieraMamba):
            pass

        baseline_model = TinyModel.__new__(TinyModel)
        nn.Module.__init__(baseline_model)
        baseline_model.query_conditioned = False
        baseline_model.block = baseline

        enabled_model = TinyModel.__new__(TinyModel)
        nn.Module.__init__(enabled_model)
        enabled_model.query_conditioned = True
        enabled_model.block = enabled

        result = enabled_model.load_compatible_state_dict(
            baseline_model.state_dict()
        )

        self.assertTrue(result.missing_keys)
        self.assertFalse(result.unexpected_keys)
        for key, value in baseline_model.state_dict().items():
            self.assertTrue(torch.equal(value, enabled_model.state_dict()[key]))

    def test_query_modulation_changes_output_and_accepts_none(self):
        block = make_block(
            query_dim=6,
            query_modulation=True,
            query_aware_gate=True,
            query_mod_hidden_dim=8,
        )
        with torch.no_grad():
            block.query_modulation_mlp[0].weight.fill_(0.05)
            block.query_modulation_mlp[0].bias.zero_()
            block.query_modulation_mlp[-1].weight.fill_(0.02)
            block.gate1_query.weight.fill_(0.03)

        x = torch.randn(1, 8, 8)
        mask = torch.ones(1, 1, 8, dtype=torch.bool)
        query_mask = torch.tensor([[[True, True, False]]])
        query_a = torch.zeros(1, 6, 3)
        query_b = torch.ones(1, 6, 3)
        query_b[..., -1] = 100.0

        no_query = block(x.clone(), mask.clone(), query_feat=None)
        output_a = block(
            x.clone(), mask.clone(), query_a, query_mask
        )
        output_b = block(
            x.clone(), mask.clone(), query_b, query_mask
        )

        self.assertEqual(no_query[1].shape, output_a[1].shape)
        self.assertFalse(torch.allclose(output_a[1], output_b[1]))
        for tensor in output_b:
            self.assertTrue(torch.isfinite(tensor).all())

    def test_importance_mask_edges_and_backward(self):
        block = make_block(
            query_dim=6,
            query_modulation=True,
            query_aware_gate=True,
            query_mod_hidden_dim=8,
            query_boundary_importance=True,
            importance_hidden_dim=8,
        )
        x = torch.randn(2, 8, 7, requires_grad=True)
        mask = torch.tensor(
            [
                [[True, True, True, True, True, True, True]],
                [[True, True, True, True, False, False, False]],
            ]
        )
        query = torch.randn(2, 6, 5, requires_grad=True)
        query_mask = torch.tensor(
            [
                [[True, True, True, True, True]],
                [[True, True, True, False, False]],
            ]
        )

        output = block(
            x,
            mask,
            query,
            query_mask,
            return_importance_debug=True,
        )
        anchor_out, sequence_out, anchor_mask, sequence_mask, debug = output

        self.assertEqual(anchor_out.shape, (2, 8, 4))
        self.assertEqual(sequence_out.shape, (2, 8, 7))
        self.assertTrue(torch.equal(sequence_mask, mask))
        self.assertTrue(
            torch.equal(
                anchor_mask,
                torch.tensor(
                    [
                        [[True, True, True, True]],
                        [[True, True, False, False]],
                    ]
                ),
            )
        )

        expected_keys = {
            'relevance',
            'start',
            'end',
            'start_prob',
            'end_prob',
            'boundary',
            'temporal_change',
            'importance',
        }
        self.assertEqual(set(debug), expected_keys)
        self.assertTrue(torch.equal(debug['start'], debug['start_prob']))
        self.assertTrue(torch.equal(debug['end'], debug['end_prob']))
        for score in debug.values():
            self.assertEqual(score.shape, (2, 7))
            self.assertTrue(torch.isfinite(score).all())
            self.assertTrue(torch.equal(score[1, 4:], torch.zeros(3)))
        self.assertTrue(torch.equal(debug['temporal_change'][:, 0], torch.zeros(2)))

        loss = sequence_out.square().mean() + debug['importance'].mean()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(query.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(query.grad).all())
        query_grads = [
            parameter.grad
            for name, parameter in block.named_parameters()
            if (
                'query_modulation_mlp' in name
                or 'gate1_query' in name
                or 'importance_predictor' in name
            )
        ]
        self.assertTrue(any(grad is not None for grad in query_grads))
        for grad in query_grads:
            if grad is not None:
                self.assertTrue(torch.isfinite(grad).all())

    def test_single_token_importance_has_valid_edges(self):
        predictor = QueryBoundaryImportancePredictor(
            video_dim=8,
            query_dim=6,
            hidden_dim=8,
        )
        output = predictor(
            torch.randn(2, 8, 1),
            torch.randn(2, 6),
            torch.tensor([[True], [False]]),
        )
        for score in output.values():
            self.assertEqual(score.shape, (2, 1))
            self.assertTrue(torch.isfinite(score).all())
            self.assertEqual(score[1, 0].item(), 0.0)
        self.assertEqual(output['temporal_change'][0, 0].item(), 0.0)

    def test_backbone_propagates_query_to_every_amp_layer(self):
        with mock.patch.object(anchor_mamba, 'Hydra', DummyGlobalEncoder):
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
                block_type='AnchorMambaPoolingBlockGated',
                query_dim=6,
                query_modulation=True,
                query_aware_gate=True,
                query_mod_hidden_dim=8,
                query_boundary_importance=True,
                importance_debug=True,
                importance_hidden_dim=8,
            ).eval()

        x = torch.randn(2, 8, 8)
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

        output = backbone(x, mask, query, query_mask)
        self.assertEqual(len(output), 4)
        fpn, fpn_masks, anchor_fpn, anchor_masks = output
        self.assertEqual([tensor.shape[-1] for tensor in fpn], [8, 4])
        self.assertEqual([tensor.shape[-1] for tensor in anchor_fpn], [4, 2])
        self.assertEqual([tensor.shape[-1] for tensor in fpn_masks], [8, 4])
        self.assertEqual([tensor.shape[-1] for tensor in anchor_masks], [4, 2])
        self.assertEqual(len(backbone.last_importance_debug), 2)
        self.assertEqual(
            backbone.last_importance_debug[0]['importance'].shape,
            (2, 8),
        )
        self.assertEqual(
            backbone.last_importance_debug[1]['importance'].shape,
            (2, 4),
        )

        output_debug = backbone(
            x,
            mask,
            query,
            query_mask,
            return_importance_debug=True,
        )
        self.assertEqual(len(output_debug), 5)
        self.assertEqual(len(output_debug[-1]), 2)

    def test_regular_fusion_batch_alignment_supports_multiple_videos(self):
        model = HieraMamba.__new__(HieraMamba)
        nn.Module.__init__(model)
        model.query_conditioned = True

        video = torch.tensor(
            [
                [[1.0, 2.0, 0.0, 0.0]],
                [[3.0, 4.0, 5.0, 0.0]],
            ]
        )
        video_mask = torch.tensor(
            [
                [True, True, False, False],
                [True, True, True, False],
            ]
        )
        query = torch.randn(3, 6, 2)
        expanded_video, expanded_mask = model._align_video_query_batches(
            video,
            video_mask,
            query,
            torch.tensor([2, 1]),
        )

        self.assertEqual(expanded_video.shape[0], 3)
        self.assertTrue(torch.equal(expanded_video[0], video[0]))
        self.assertTrue(torch.equal(expanded_video[1], video[0]))
        self.assertTrue(torch.equal(expanded_video[2], video[1]))
        self.assertTrue(torch.equal(expanded_mask[0], video_mask[0]))
        self.assertTrue(torch.equal(expanded_mask[1], video_mask[0]))
        self.assertTrue(torch.equal(expanded_mask[2], video_mask[1]))


class QueryBoundaryImportanceLossTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(13)

    @staticmethod
    def make_points(length, stride=1.0):
        points = torch.zeros(length, 4)
        points[:, 0] = torch.arange(length, dtype=torch.float32) * stride
        points[:, 3] = stride
        return points

    @staticmethod
    def make_debug(batch_size, seq_len):
        keys = (
            'relevance',
            'start',
            'end',
            'boundary',
            'temporal_change',
            'importance',
        )
        return {
            key: 0.1 + 0.8 * torch.rand(batch_size, seq_len)
            for key in keys
        }

    def test_targets_peak_at_boundaries_and_mask_padding(self):
        criterion = QueryBoundaryImportanceLoss({})
        points = self.make_points(8)
        mask = torch.tensor(
            [
                [[True, True, True, True, True, True, True, True]],
                [[True, True, True, True, True, False, False, False]],
            ]
        )
        targets = torch.tensor([[2.0, 5.0], [1.0, 3.0]])

        target_maps = criterion.build_targets(points, mask, targets)

        self.assertEqual(target_maps['start'][0].argmax().item(), 2)
        self.assertEqual(target_maps['end'][0].argmax().item(), 5)
        self.assertEqual(target_maps['temporal_change'][0, 0].item(), 0.0)
        for key in (
            'relevance',
            'start',
            'end',
            'boundary',
            'temporal_change',
            'importance',
        ):
            self.assertTrue(
                torch.equal(target_maps[key][1, 5:], torch.zeros(3))
            )

    def test_padding_does_not_change_target_normalization(self):
        criterion = QueryBoundaryImportanceLoss({})
        targets = torch.tensor([[4.4, 7.5]])
        short_mask = torch.ones(1, 5, dtype=torch.bool)
        padded_mask = torch.tensor(
            [[True, True, True, True, True, False, False, False]]
        )

        short_targets = criterion.build_targets(
            self.make_points(5), short_mask, targets
        )
        padded_targets = criterion.build_targets(
            self.make_points(8), padded_mask, targets
        )

        for key in (
            'relevance',
            'start',
            'end',
            'boundary',
            'temporal_change',
            'importance',
        ):
            torch.testing.assert_close(
                short_targets[key],
                padded_targets[key][:, :5],
                rtol=0.0,
                atol=1e-7,
            )

    def test_padding_values_do_not_change_loss(self):
        criterion = QueryBoundaryImportanceLoss({})
        points = self.make_points(8)
        mask = torch.tensor(
            [[True, True, True, True, True, False, False, False]]
        )
        targets = torch.tensor([[1.5, 3.5]])
        debug_a = self.make_debug(1, 8)
        debug_b = {
            key: value.clone() for key, value in debug_a.items()
        }
        for value in debug_b.values():
            value[:, 5:] = torch.tensor([0.01, 0.5, 0.99])

        loss_a = criterion(
            (debug_a,), (points,), (mask,), targets
        )
        loss_b = criterion(
            (debug_b,), (points,), (mask,), targets
        )

        torch.testing.assert_close(loss_a, loss_b, rtol=0.0, atol=1e-7)

    def test_auxiliary_loss_backpropagates_to_predictor(self):
        predictor = QueryBoundaryImportancePredictor(
            video_dim=8,
            query_dim=6,
            hidden_dim=8,
        )
        criterion = QueryBoundaryImportanceLoss({})
        video = torch.randn(2, 8, 8, requires_grad=True)
        query = torch.randn(2, 6, requires_grad=True)
        mask = torch.tensor(
            [
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, False, False],
            ]
        )
        targets = torch.tensor([[2.0, 6.0], [1.0, 4.0]])
        debug = predictor(video, query, mask)

        loss, components = criterion(
            (debug,),
            (self.make_points(8),),
            (mask,),
            targets,
            return_components=True,
        )
        self.assertTrue(torch.isfinite(loss))
        for component in components.values():
            self.assertTrue(torch.isfinite(component))

        loss.backward()

        self.assertTrue(torch.isfinite(video.grad).all())
        self.assertTrue(torch.isfinite(query.grad).all())
        predictor_grads = [
            parameter.grad
            for parameter in predictor.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(predictor_grads)
        for grad in predictor_grads:
            self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(
            sum(grad.abs().sum().item() for grad in predictor_grads),
            0.0,
        )

    def test_fixed_example_optimization_reduces_loss(self):
        predictor = QueryBoundaryImportancePredictor(
            video_dim=8,
            query_dim=6,
            hidden_dim=8,
        )
        criterion = QueryBoundaryImportanceLoss(
            {
                'change_weight': 0.0,
                'final_weight': 0.0,
                'boundary_sigma': 1.0,
            }
        )
        optimizer = torch.optim.Adam(predictor.parameters(), lr=0.03)
        video = torch.randn(1, 8, 8)
        query = torch.randn(1, 6)
        mask = torch.ones(1, 8, dtype=torch.bool)
        targets = torch.tensor([[2.0, 6.0]])
        points = self.make_points(8)

        with torch.no_grad():
            initial_loss = criterion(
                (predictor(video, query, mask),),
                (points,),
                (mask,),
                targets,
            ).item()

        for _ in range(40):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(
                (predictor(video, query, mask),),
                (points,),
                (mask,),
                targets,
            )
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            final_loss = criterion(
                (predictor(video, query, mask),),
                (points,),
                (mask,),
                targets,
            ).item()

        self.assertLess(final_loss, initial_loss)

    def test_model_default_returns_eight_and_debug_returns_nine(self):
        class TinyHieraMamba(HieraMamba):

            def __init__(self):
                nn.Module.__init__(self)
                self.early_fusion = False
                self.query_conditioned = True
                self.encode_calls = 0

            def encode_text(self, text, text_masks):
                return text, text_masks

            def encode_video(
                self,
                vid,
                vid_masks,
                query_feat=None,
                query_mask=None,
                text_size=None,
                return_importance_debug=False,
                return_anchor_assignments=False,
                allocator_targets=None,
                forced_cut_offsets=None,
                forced_target_counts=None,
            ):
                self.encode_calls += 1
                if vid_masks.ndim == 2:
                    vid_masks = vid_masks.unsqueeze(1)
                outputs = (
                    (vid,),
                    (vid_masks,),
                    (vid[..., ::2],),
                    (vid_masks[..., ::2],),
                )
                if return_importance_debug:
                    debug = (
                        {
                            'importance': vid.new_zeros(
                                vid.size(0), vid.size(-1)
                            )
                        },
                    )
                    outputs += (debug,)
                if return_anchor_assignments:
                    assignment = vid.new_zeros(
                        vid.size(0),
                        vid[..., ::2].size(-1),
                        vid.size(-1),
                    )
                    assignment[:, :, ::2] = 1.0
                    assignment[:, :, 1::2] = 1.0
                    outputs += ((assignment,),)
                return outputs

            def fuse_and_predict(
                self,
                fpn,
                fpn_masks,
                text,
                text_masks,
                text_size=None,
            ):
                logits = tuple(
                    feature.new_zeros(feature.size(0), feature.size(-1))
                    for feature in fpn
                )
                offsets = tuple(
                    feature.new_zeros(
                        feature.size(0), feature.size(-1), 2
                    )
                    for feature in fpn
                )
                masks = tuple(
                    mask[:, 0] if mask.ndim == 3 else mask
                    for mask in fpn_masks
                )
                return logits, logits, offsets, masks

        model = TinyHieraMamba()
        vid = torch.randn(2, 8, 8)
        vid_masks = torch.ones(2, 8, dtype=torch.bool)
        text = torch.randn(2, 6, 4)
        text_masks = torch.ones(2, 4, dtype=torch.bool)
        text_size = torch.ones(2, dtype=torch.long)

        default_outputs = model(
            vid, vid_masks, text, text_masks, text_size
        )
        debug_outputs = model(
            vid,
            vid_masks,
            text,
            text_masks,
            text_size,
            return_importance_debug=True,
        )

        assignment_outputs = model(
            vid,
            vid_masks,
            text,
            text_masks,
            text_size,
            return_anchor_assignments=True,
        )
        combined_outputs = model(
            vid,
            vid_masks,
            text,
            text_masks,
            text_size,
            return_importance_debug=True,
            return_anchor_assignments=True,
        )

        self.assertEqual(len(default_outputs), 8)
        self.assertEqual(len(debug_outputs), 9)
        self.assertEqual(len(assignment_outputs), 9)
        self.assertEqual(len(combined_outputs), 10)
        self.assertEqual(model.encode_calls, 4)
        self.assertEqual(len(debug_outputs[-1]), 1)
        self.assertEqual(
            assignment_outputs[-1][0].shape,
            (2, 4, 8),
        )
        self.assertEqual(len(combined_outputs[-2]), 1)
        self.assertEqual(len(combined_outputs[-1]), 1)


if __name__ == '__main__':
    unittest.main()

import os
import tempfile
import unittest
from unittest import mock

import torch
import torch.nn as nn
import yaml

from libs.core.adaptive_hieramamba import (
    normalize_adaptive_hieramamba_config,
)
from libs.core.opt import load_opt
from libs.modeling import anchor_mamba
from libs.modeling.losses import MultiScaleMaskedContrastive
from libs.modeling.model import (
    AdaptiveHieraMamba,
    HieraMamba,
    make_models_net,
    models,
)
from libs.modeling.query_boundary_importance_loss import (
    make_boundary_supervision_loss,
)


class DummyGlobalEncoder(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


def make_spec(enabled):
    return {
        'query_modulation': enabled,
        'query_aware_gate': enabled,
        'query_mod_hidden_dim': 8,
        'importance': {
            'enable': enabled,
            'use_query_relevance': True,
            'use_boundary': True,
            'use_temporal_change': True,
            'alpha': 0.4,
            'beta': 0.4,
            'gamma': 0.2,
            'hidden_dim': 8,
            'debug': False,
        },
        'adaptive_anchor': enabled,
        'target_keep_ratio': 0.75,
        'progressive_compression': enabled,
        'keep_ratio_per_level': [0.8, 0.6, 0.5],
        'progressive_compression_debug': False,
        'assignment_acc': enabled,
        'coarse_to_fine_refine': enabled,
        'refine_levels': 2,
        'boundary_loss_weight': 0.5 if enabled else 0.0,
    }


def make_tiny_config(model_name, enabled):
    model = {
        'early_fusion': False,
        'text_net': {
            'name': 'identity',
            'in_dim': 6,
            'embd_dim': 6,
            'max_seq_len': 4,
            'n_heads': 2,
            'use_abs_pe': False,
            'use_bkgd_token': False,
        },
        'vid_net': {
            'name': 'hieramamba_backbone',
            'in_dim': 8,
            'embd_dim': 8,
            'n_heads': 2,
            'max_seq_len': 16,
            'stride': 1,
            'arch': [0, 0, 3],
            'mha_win_size': 0,
            'use_abs_pe': False,
            'local_encode': False,
            'return_anchor': True,
            'block_type': 'AnchorMambaPoolingBlockGated',
        },
        'fusion': {
            'name': 'xattn',
            'n_layers': 1,
            'n_heads': 2,
            'attn_pdrop': 0.0,
            'proj_pdrop': 0.0,
            'path_pdrop': 0.0,
            'xattn_mode': 'adaln',
        },
        'cls_head': {
            'name': 'cls',
            'n_layers': 1,
            'prior_prob': 0.0,
        },
        'reg_head': {
            'name': 'reg',
            'n_layers': 1,
        },
    }
    if model_name != 'hieramamba':
        model['adaptive_hieramamba'] = make_spec(enabled)
    return {
        'model_net': {'name': model_name},
        'model': model,
        'train': {
            'epochs': 1,
            'warmup_epochs': 0,
            'loss_aux': {
                'query_boundary_importance': {
                    'enable': False,
                    'boundary_loss_weight': 0.0,
                    'boundary_sigma': 1.0,
                    'max_pos_weight': 10.0,
                },
                'ds_contrast': {
                    'enable': False,
                    'type': 'ds_contrastive',
                    'acc_mode': 'legacy_acc',
                    'temperature': 0.1,
                    'proj_outdim': 8,
                    'proj_expand': 1.0,
                    'proj_num_layers': 1,
                },
                'gt_contrast': {'enable': False},
            },
        },
    }


def load_config_dict(config):
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.yaml', delete=False
    ) as config_file:
        yaml.safe_dump(config, config_file)
        path = config_file.name
    try:
        return load_opt(path)
    finally:
        os.remove(path)


def assert_nested_close(test_case, expected, actual):
    test_case.assertEqual(len(expected), len(actual))
    for expected_item, actual_item in zip(expected, actual):
        if isinstance(expected_item, tuple):
            assert_nested_close(test_case, expected_item, actual_item)
        else:
            torch.testing.assert_close(expected_item, actual_item)


class AdaptiveHieraMambaConfigTest(unittest.TestCase):

    def test_registry_aliases_share_inherited_implementation(self):
        self.assertIs(models['adaptive_hieramamba'], AdaptiveHieraMamba)
        self.assertIs(
            models['query_boundary_adaptive_hieramamba'],
            AdaptiveHieraMamba,
        )
        self.assertTrue(issubclass(AdaptiveHieraMamba, HieraMamba))
        self.assertIs(AdaptiveHieraMamba.forward, HieraMamba.forward)

    def test_nested_config_maps_model_and_training_features(self):
        config = make_tiny_config('adaptive_hieramamba', True)
        normalized = load_config_dict(config)
        vid = normalized['model']['vid_net']
        loss_aux = normalized['train']['loss_aux']

        self.assertTrue(vid['query_modulation'])
        self.assertTrue(vid['query_aware_gate'])
        self.assertTrue(vid['query_boundary_importance'])
        self.assertTrue(vid['adaptive_anchor'])
        self.assertTrue(vid['coarse_to_fine_refine'])
        self.assertEqual(vid['query_mod_hidden_dim'], 8)
        self.assertEqual(vid['importance_hidden_dim'], 8)
        self.assertEqual(vid['importance_alpha'], 0.4)
        self.assertEqual(vid['importance_beta'], 0.4)
        self.assertEqual(vid['importance_gamma'], 0.2)
        self.assertEqual(vid['keep_ratio_per_level'], [0.8, 0.6, 0.5])
        self.assertEqual(vid['refine_levels'], 2)
        self.assertTrue(loss_aux['ds_contrast']['enable'])
        self.assertEqual(
            loss_aux['ds_contrast']['acc_mode'], 'assignment_acc'
        )
        self.assertEqual(
            loss_aux['query_boundary_importance']['boundary_loss_weight'],
            0.5,
        )

    def test_importance_components_can_be_switched_independently(self):
        config = make_tiny_config('adaptive_hieramamba', True)
        importance = config['model']['adaptive_hieramamba']['importance']
        importance.update({
            'use_query_relevance': True,
            'use_boundary': False,
            'use_temporal_change': True,
            'alpha': 0.7,
            'beta': 0.2,
            'gamma': 0.3,
        })

        normalized = normalize_adaptive_hieramamba_config(config)
        vid = normalized['model']['vid_net']
        self.assertEqual(vid['importance_alpha'], 0.7)
        self.assertEqual(vid['importance_beta'], 0.0)
        self.assertEqual(vid['importance_gamma'], 0.3)

    def test_disabled_features_restore_legacy_modes_and_boundary_noop(self):
        config = make_tiny_config('adaptive_hieramamba', False)
        normalized = load_config_dict(config)
        vid = normalized['model']['vid_net']
        loss_aux = normalized['train']['loss_aux']

        self.assertFalse(vid['query_modulation'])
        self.assertFalse(vid['query_aware_gate'])
        self.assertFalse(vid['query_boundary_importance'])
        self.assertFalse(vid['adaptive_anchor'])
        self.assertFalse(vid['coarse_to_fine_refine'])
        self.assertIsNone(vid['keep_ratio_per_level'])
        self.assertEqual(loss_aux['ds_contrast']['acc_mode'], 'legacy_acc')
        weight, criterion = make_boundary_supervision_loss(
            loss_aux['query_boundary_importance']
        )
        self.assertEqual(weight, 0.0)
        self.assertIsNone(criterion)

    def test_original_hieramamba_configuration_bypasses_normalization(self):
        legacy = make_tiny_config('hieramamba', False)
        legacy['model']['vid_net']['keep_ratio_per_level'] = [0.9, 0.8, 0.7]
        legacy['train']['loss_aux']['ds_contrast']['acc_mode'] = (
            'assignment_acc'
        )

        normalized = normalize_adaptive_hieramamba_config(legacy)

        self.assertIs(normalized, legacy)
        self.assertEqual(
            normalized['model']['vid_net']['keep_ratio_per_level'],
            [0.9, 0.8, 0.7],
        )
        self.assertEqual(
            normalized['train']['loss_aux']['ds_contrast']['acc_mode'],
            'assignment_acc',
        )

    def test_invalid_cross_feature_combinations_are_rejected(self):
        invalid_specs = []

        no_importance = make_spec(True)
        for key in (
            'use_query_relevance', 'use_boundary', 'use_temporal_change'
        ):
            no_importance['importance'][key] = False
        invalid_specs.append(no_importance)

        assignment_without_adaptive = make_spec(False)
        assignment_without_adaptive['assignment_acc'] = True
        invalid_specs.append(assignment_without_adaptive)

        progressive_without_adaptive = make_spec(False)
        progressive_without_adaptive['progressive_compression'] = True
        invalid_specs.append(progressive_without_adaptive)

        missing_ratios = make_spec(True)
        del missing_ratios['keep_ratio_per_level']
        invalid_specs.append(missing_ratios)

        boundary_without_predictor = make_spec(False)
        boundary_without_predictor['boundary_loss_weight'] = 1.0
        invalid_specs.append(boundary_without_predictor)

        for spec in invalid_specs:
            with self.subTest(spec=spec):
                config = make_tiny_config('adaptive_hieramamba', False)
                config['model']['adaptive_hieramamba'] = spec
                with self.assertRaises(ValueError):
                    normalize_adaptive_hieramamba_config(config)

    def test_repository_configs_keep_legacy_and_map_unified_model(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        legacy = load_opt(os.path.join(root, 'opts', 'ego4d_hieramamba.yaml'))
        adaptive = load_opt(
            os.path.join(root, 'opts', 'ego4d_adaptive_hieramamba.yaml')
        )

        self.assertEqual(legacy['model_net']['name'], 'hieramamba')
        self.assertNotIn('adaptive_hieramamba', legacy['model'])
        self.assertFalse(legacy['model']['vid_net']['adaptive_anchor'])
        self.assertEqual(
            legacy['train']['loss_aux']['ds_contrast']['acc_mode'],
            'legacy_acc',
        )

        self.assertEqual(
            adaptive['model_net']['name'], 'adaptive_hieramamba'
        )
        self.assertTrue(adaptive['model']['vid_net']['adaptive_anchor'])
        self.assertEqual(
            len(adaptive['model']['vid_net']['keep_ratio_per_level']),
            adaptive['model']['vid_net']['arch'][-1],
        )
        self.assertEqual(
            adaptive['train']['loss_aux']['ds_contrast']['acc_mode'],
            'assignment_acc',
        )


class AdaptiveHieraMambaIntegrationTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(67)

    def test_all_disabled_matches_original_state_and_forward(self):
        legacy_opt = load_config_dict(make_tiny_config('hieramamba', False))
        adaptive_opt = load_config_dict(
            make_tiny_config('adaptive_hieramamba', False)
        )
        with mock.patch.object(
            anchor_mamba, 'Hydra', DummyGlobalEncoder
        ):
            legacy = make_models_net(legacy_opt).eval()
            adaptive = make_models_net(adaptive_opt).eval()

        self.assertIs(type(legacy), HieraMamba)
        self.assertIs(type(adaptive), AdaptiveHieraMamba)
        self.assertEqual(
            tuple(legacy.state_dict()), tuple(adaptive.state_dict())
        )
        adaptive.load_compatible_state_dict(legacy.state_dict())
        self.assertFalse(adaptive.query_conditioned)
        self.assertFalse(adaptive.vid_net.adaptive_anchor)

        video = torch.randn(2, 8, 8)
        video_mask = torch.tensor(
            [[True] * 8, [True, True, True, True, True, True, False, False]]
        )
        text = torch.randn(2, 6, 3)
        text_mask = torch.tensor(
            [[True, True, True], [True, True, False]]
        )
        text_size = torch.ones(2, dtype=torch.long)
        with torch.no_grad():
            legacy_output = legacy(
                video, video_mask, text, text_mask, text_size
            )
            adaptive_output = adaptive(
                video, video_mask, text, text_mask, text_size
            )
        assert_nested_close(self, legacy_output, adaptive_output)

    def test_unified_model_forward_losses_and_backward(self):
        opt = load_config_dict(
            make_tiny_config('query_boundary_adaptive_hieramamba', True)
        )
        with mock.patch.object(
            anchor_mamba, 'Hydra', DummyGlobalEncoder
        ):
            model = make_models_net(opt).train()

        video = torch.randn(2, 8, 10, requires_grad=True)
        video_mask = torch.tensor(
            [
                [True] * 10,
                [True, True, True, True, True, True, True, False, False, False],
            ]
        )
        text = torch.randn(2, 6, 3, requires_grad=True)
        text_mask = torch.tensor(
            [[True, True, True], [True, True, False]]
        )
        text_size = torch.ones(2, dtype=torch.long)

        outputs = model(
            video,
            video_mask,
            text,
            text_mask,
            text_size,
            return_importance_debug=True,
            return_anchor_assignments=True,
        )
        self.assertEqual(len(outputs), 10)
        (
            logits,
            _,
            offsets,
            masks,
            fpn,
            sequence_masks,
            anchors,
            anchor_masks,
            importance_debug,
            assignments,
        ) = outputs
        self.assertEqual(len(fpn), 3)
        self.assertEqual(len(importance_debug), 3)
        self.assertEqual(len(assignments), 3)
        self.assertEqual(model.vid_net.keep_ratio_per_level, (0.8, 0.6, 0.5))
        self.assertEqual(
            model.vid_net.coarse_to_fine_refiner.last_alignment_modes,
            ('assignment', 'assignment'),
        )
        for level in range(3):
            self.assertEqual(fpn[level].shape[-1], sequence_masks[level].shape[-1])
            self.assertEqual(anchors[level].shape[-1], anchor_masks[level].shape[-1])
            self.assertEqual(
                assignments[level].shape,
                (
                    video.size(0),
                    anchors[level].size(-1),
                    fpn[level].size(-1),
                ),
            )
            for value in importance_debug[level].values():
                self.assertTrue(torch.isfinite(value).all())

        acc = MultiScaleMaskedContrastive(
            opt['train']['loss_aux']['ds_contrast'],
            vid_embd_dim=8,
        )
        acc_loss = acc(
            fpn,
            sequence_masks,
            anchors,
            anchor_masks,
            assignment_matrices=assignments,
        )
        points = []
        for mask in sequence_masks:
            level_points = torch.zeros(mask.size(-1), 4)
            level_points[:, 0] = torch.arange(mask.size(-1)).float()
            level_points[:, 3] = 1.0
            points.append(level_points)
        boundary_weight, boundary_criterion = make_boundary_supervision_loss(
            opt['train']['loss_aux']['query_boundary_importance']
        )
        boundary_loss = boundary_criterion(
            importance_debug,
            tuple(points),
            sequence_masks,
            torch.tensor([[2.0, 7.0], [1.0, 5.0]]),
        )
        model_loss = sum(value.square().mean() for value in logits)
        model_loss = model_loss + sum(
            value.square().mean() for value in offsets
        )
        total_loss = model_loss + acc_loss + boundary_weight * boundary_loss

        self.assertTrue(torch.isfinite(total_loss))
        total_loss.backward()
        self.assertIsNotNone(video.grad)
        self.assertIsNotNone(text.grad)
        self.assertTrue(torch.isfinite(video.grad).all())
        self.assertTrue(torch.isfinite(text.grad).all())
        self.assertGreater(float(video.grad.abs().sum()), 0.0)
        self.assertGreater(float(text.grad.abs().sum()), 0.0)
        trainable_grads = [
            parameter.grad for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(trainable_grads)
        self.assertTrue(
            all(torch.isfinite(gradient).all() for gradient in trainable_grads)
        )


if __name__ == '__main__':
    unittest.main()

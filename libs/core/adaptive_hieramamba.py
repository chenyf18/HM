"""Normalize the public configuration of Adaptive HieraMamba."""

from copy import deepcopy
import math


ADAPTIVE_MODEL_NAMES = (
    'adaptive_hieramamba',
    'query_boundary_adaptive_hieramamba',
)

_VID_KEYS = (
    'query_modulation',
    'query_aware_gate',
    'query_mod_hidden_dim',
    'query_boundary_importance',
    'importance_debug',
    'importance_hidden_dim',
    'adaptive_anchor',
    'target_keep_ratio',
    'progressive_compression_debug',
    'coarse_to_fine_refine',
    'refine_levels',
)

_IMPORTANCE_COMPONENTS = (
    ('use_query_relevance', 'alpha', 'importance_alpha', 0.4),
    ('use_boundary', 'beta', 'importance_beta', 0.4),
    ('use_temporal_change', 'gamma', 'importance_gamma', 0.2),
)


def _as_bool(value, key):
    if not isinstance(value, bool):
        raise ValueError('{} must be boolean'.format(key))
    return value


def _as_non_negative_float(value, key):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError('{} must be numeric'.format(key))
    if not math.isfinite(value) or value < 0:
        raise ValueError('{} must be finite and non-negative'.format(key))
    return value


def normalize_adaptive_hieramamba_config(opt):
    """Map nested Adaptive HieraMamba options to existing flat options.

    The function returns a deep copy. It intentionally does nothing for the
    original hieramamba registry name, preserving legacy configuration.
    """
    model_name = opt.get('model_net', {}).get('name')
    if model_name not in ADAPTIVE_MODEL_NAMES:
        return opt

    normalized = deepcopy(opt)
    model = normalized.setdefault('model', {})
    spec = model.get('adaptive_hieramamba', {})
    if not isinstance(spec, dict):
        raise ValueError('model.adaptive_hieramamba must be a mapping')
    vid = model.setdefault('vid_net', {})

    for key in _VID_KEYS:
        if key in spec:
            vid[key] = spec[key]

    importance = spec.get('importance', {})
    if not isinstance(importance, dict):
        raise ValueError(
            'model.adaptive_hieramamba.importance must be a mapping'
        )
    if 'enable' in importance:
        vid['query_boundary_importance'] = _as_bool(
            importance['enable'], 'importance.enable'
        )
    if 'hidden_dim' in importance:
        vid['importance_hidden_dim'] = importance['hidden_dim']
    if 'debug' in importance:
        vid['importance_debug'] = _as_bool(
            importance['debug'], 'importance.debug'
        )
    component_total = 0.0
    for flag_key, short_weight, vid_weight, default in _IMPORTANCE_COMPONENTS:
        enabled = _as_bool(importance.get(flag_key, True), flag_key)
        raw_weight = importance.get(
            short_weight,
            importance.get(vid_weight, vid.get(vid_weight, default)),
        )
        weight = _as_non_negative_float(raw_weight, vid_weight)
        vid[vid_weight] = weight if enabled else 0.0
        component_total += vid[vid_weight]

    importance_required = bool(
        vid.get('query_boundary_importance', False)
        or vid.get('adaptive_anchor', False)
    )
    if importance_required and component_total <= 0:
        raise ValueError(
            'at least one importance component must be enabled when the '
            'importance predictor is active'
        )

    progressive = _as_bool(
        spec.get('progressive_compression', False),
        'progressive_compression',
    )
    if progressive:
        if 'keep_ratio_per_level' not in spec:
            raise ValueError(
                'progressive_compression=true requires keep_ratio_per_level'
            )
        if not bool(vid.get('adaptive_anchor', False)):
            raise ValueError(
                'progressive_compression requires adaptive_anchor=true'
            )
        vid['keep_ratio_per_level'] = spec['keep_ratio_per_level']
    else:
        vid['keep_ratio_per_level'] = None

    loss_aux = normalized.setdefault('train', {}).setdefault('loss_aux', {})
    ds_opt = loss_aux.setdefault('ds_contrast', {})
    assignment_acc = _as_bool(
        spec.get('assignment_acc', False), 'assignment_acc'
    )
    if assignment_acc:
        if not bool(vid.get('adaptive_anchor', False)):
            raise ValueError('assignment_acc requires adaptive_anchor=true')
        ds_opt['enable'] = True
        ds_opt['acc_mode'] = 'assignment_acc'
    else:
        ds_opt['acc_mode'] = 'legacy_acc'

    boundary_opt = loss_aux.setdefault('query_boundary_importance', {})
    if 'boundary_loss_weight' in spec:
        boundary_opt['boundary_loss_weight'] = _as_non_negative_float(
            spec['boundary_loss_weight'], 'boundary_loss_weight'
        )
    boundary_weight = float(boundary_opt.get('boundary_loss_weight', 0.0))
    if boundary_weight > 0 and not importance_required:
        raise ValueError(
            'boundary_loss_weight > 0 requires query_boundary_importance=true '
            'or adaptive_anchor=true'
        )

    return normalized

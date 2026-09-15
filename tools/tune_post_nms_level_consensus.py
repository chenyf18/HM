#!/usr/bin/env python3
import argparse
import json
import math
from collections import OrderedDict
from pathlib import Path

import torch


DEFAULT_ALPHAS = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0)
DEFAULT_MODES = ('all', 'cross_level', 'cross_level_balanced')
DEFAULT_GATES = ('none', 'margin_linear')


def parse_csv_floats(value):
    return tuple(float(item.strip()) for item in value.split(',') if item.strip())


def parse_csv_strings(value):
    return tuple(item.strip() for item in value.split(',') if item.strip())


def temporal_iou_matrix(anchors, candidates):
    left = torch.maximum(anchors[:, None, 0], candidates[None, :, 0])
    right = torch.minimum(anchors[:, None, 1], candidates[None, :, 1])
    intersection = (right - left).clamp(min=0)
    anchor_lengths = (anchors[:, 1] - anchors[:, 0]).clamp(min=0)
    candidate_lengths = (candidates[:, 1] - candidates[:, 0]).clamp(min=0)
    union = (
        anchor_lengths[:, None]
        + candidate_lengths[None, :]
        - intersection
    )
    return intersection / union.clamp(min=1e-12)


def temporal_iou(segment, target):
    intersection = max(
        0.0,
        min(segment[1], target[1]) - max(segment[0], target[0]),
    )
    union = max(segment[1], target[1]) - min(segment[0], target[0])
    return intersection / union if union > 0 else 0.0


def weighted_mean_support(overlaps, scores, mask=None):
    weights = scores[None, :].expand(len(overlaps), -1)
    if mask is not None:
        weights = weights * mask.float()
    return (
        (overlaps * weights).sum(dim=1)
        / weights.sum(dim=1).clamp(min=1e-12)
    )


def infer_origin_levels(overlaps, raw_scores, raw_levels):
    normalized_scores = raw_scores / raw_scores.max().clamp(min=1e-12)
    matching_score = overlaps * (0.5 + 0.5 * normalized_scores[None, :])
    return raw_levels[matching_score.argmax(dim=1)]


def compute_supports(
    final_segments,
    raw_segments,
    raw_scores,
    raw_levels,
    modes,
):
    overlaps = temporal_iou_matrix(final_segments, raw_segments)
    origin_levels = infer_origin_levels(
        overlaps, raw_scores, raw_levels
    )
    supports = {}

    if 'all' in modes:
        supports['all'] = weighted_mean_support(
            overlaps, raw_scores
        )

    if 'cross_level' in modes:
        cross_level_mask = (
            raw_levels[None, :] != origin_levels[:, None]
        )
        supports['cross_level'] = weighted_mean_support(
            overlaps,
            raw_scores,
            cross_level_mask,
        )

    if 'cross_level_balanced' in modes:
        support_sum = torch.zeros(len(final_segments), dtype=torch.float32)
        support_count = torch.zeros(len(final_segments), dtype=torch.float32)
        for level in torch.unique(raw_levels):
            level_mask = raw_levels == level
            level_support = weighted_mean_support(
                overlaps[:, level_mask],
                raw_scores[level_mask],
            )
            valid = origin_levels != level
            support_sum[valid] += level_support[valid]
            support_count[valid] += 1
        supports['cross_level_balanced'] = (
            support_sum / support_count.clamp(min=1)
        )

    return supports, origin_levels


def confidence_margin(scores):
    if len(scores) < 2:
        return 1.0
    return float(
        ((scores[0] - scores[1]) / scores[0].clamp(min=1e-12)).item()
    )


def gate_value(gate, margin):
    if gate == 'none':
        return 1.0
    if gate == 'margin_linear':
        return max(0.0, 1.0 - margin)
    raise ValueError(f'Unknown gate: {gate}')


def load_records(predictions_path, modes, max_queries=None):
    with predictions_path.open() as file_obj:
        predictions = json.load(file_obj)

    records = []
    for video_id, video in predictions['videos'].items():
        for query in video['queries']:
            raw_candidates = query.get('raw_candidates')
            if not raw_candidates:
                raise ValueError(
                    f'Missing raw candidates for {video_id}:{query["query_id"]}'
                )

            final_segments = torch.tensor(
                [item['segment'] for item in query['predictions']],
                dtype=torch.float32,
            )
            final_scores = torch.tensor(
                [item['score'] for item in query['predictions']],
                dtype=torch.float32,
            )
            raw_segments = torch.tensor(
                [item['segment'] for item in raw_candidates],
                dtype=torch.float32,
            )
            raw_scores = torch.tensor(
                [item['score'] for item in raw_candidates],
                dtype=torch.float32,
            )
            raw_levels = torch.tensor(
                [item['level'] for item in raw_candidates],
                dtype=torch.long,
            )
            supports, origin_levels = compute_supports(
                final_segments,
                raw_segments,
                raw_scores,
                raw_levels,
                modes,
            )
            records.append({
                'video_id': video_id,
                'query_id': query['query_id'],
                'ground_truth': query['ground_truth'],
                'segments': final_segments,
                'scores': final_scores,
                'supports': supports,
                'origin_levels': origin_levels,
                'margin': confidence_margin(final_scores),
            })
            if max_queries is not None and len(records) >= max_queries:
                return records, predictions.get('summary', {})

    return records, predictions.get('summary', {})


def consensus_scores(scores, support, alpha, gate):
    strength = alpha * gate
    logits = scores.clamp(min=1e-12).log() + strength * support
    adjusted = torch.exp(logits - logits.max()) * scores.max()
    return logits, adjusted


def evaluate(records, mode, gate, alpha, collect_predictions=False):
    counts = {
        'r1_iou_0.3': 0,
        'r1_iou_0.5': 0,
        'r5_iou_0.3': 0,
        'r5_iou_0.5': 0,
    }
    top1_iou_sum = 0.0
    baseline_iou_sum = 0.0
    ranking_changed = 0
    improved_iou_0_5 = 0
    worsened_iou_0_5 = 0
    output_videos = OrderedDict()

    for record in records:
        query_gate = gate_value(gate, record['margin'])
        logits, adjusted_scores = consensus_scores(
            record['scores'],
            record['supports'][mode],
            alpha,
            query_gate,
        )
        ranking = logits.argsort(descending=True)
        selected_idx = int(ranking[0].item())
        ranking_changed += selected_idx != 0

        segments = record['segments']
        target = record['ground_truth']
        top1_iou = temporal_iou(
            segments[selected_idx].tolist(), target
        )
        baseline_iou = temporal_iou(segments[0].tolist(), target)
        top5_iou = max(
            temporal_iou(segment.tolist(), target)
            for segment in segments
        )

        top1_iou_sum += top1_iou
        baseline_iou_sum += baseline_iou
        counts['r1_iou_0.3'] += top1_iou >= 0.3
        counts['r1_iou_0.5'] += top1_iou >= 0.5
        counts['r5_iou_0.3'] += top5_iou >= 0.3
        counts['r5_iou_0.5'] += top5_iou >= 0.5
        improved_iou_0_5 += baseline_iou < 0.5 <= top1_iou
        worsened_iou_0_5 += top1_iou < 0.5 <= baseline_iou

        if collect_predictions:
            video = output_videos.setdefault(
                record['video_id'],
                {'queries': [], 'recall_at_iou': {}},
            )
            video['queries'].append({
                'query_id': record['query_id'],
                'ground_truth': target,
                'confidence_margin': record['margin'],
                'predictions': [
                    {
                        'segment': segments[idx].tolist(),
                        'score': float(adjusted_scores[idx].item()),
                        'original_score': float(record['scores'][idx].item()),
                        'original_rank': int(idx),
                        'origin_level': int(
                            record['origin_levels'][idx].item()
                        ),
                        'consensus': float(
                            record['supports'][mode][idx].item()
                        ),
                        'consensus_logit': float(logits[idx].item()),
                    }
                    for idx in ranking.tolist()
                ],
            })

    num_queries = len(records)
    result = {
        'mode': mode,
        'gate': gate,
        'alpha': alpha,
        **{
            key: value / num_queries
            for key, value in counts.items()
        },
        'mean_top1_iou': top1_iou_sum / num_queries,
        'mean_baseline_iou': baseline_iou_sum / num_queries,
        'ranking_changed_fraction': ranking_changed / num_queries,
        'rescued_at_iou_0.5_fraction': improved_iou_0_5 / num_queries,
        'lost_at_iou_0.5_fraction': worsened_iou_0_5 / num_queries,
    }
    result['r1_average'] = 0.5 * (
        result['r1_iou_0.3'] + result['r1_iou_0.5']
    )
    result['four_metric_average'] = 0.25 * (
        result['r1_iou_0.3']
        + result['r1_iou_0.5']
        + result['r5_iou_0.3']
        + result['r5_iou_0.5']
    )

    if collect_predictions:
        output = {
            'summary': {
                'overall_recall_at_iou': {
                    'Rank@1_IoU@0.3': result['r1_iou_0.3'],
                    'Rank@1_IoU@0.5': result['r1_iou_0.5'],
                    'Rank@5_IoU@0.3': result['r5_iou_0.3'],
                    'Rank@5_IoU@0.5': result['r5_iou_0.5'],
                },
                'total_queries': num_queries,
                'total_videos': len(output_videos),
            },
            'post_nms_level_consensus': result,
            'videos': output_videos,
        }
        return result, output

    return result


def add_deltas(result, baseline):
    output = dict(result)
    for key in (
        'r1_iou_0.3',
        'r1_iou_0.5',
        'r5_iou_0.3',
        'r5_iou_0.5',
        'r1_average',
        'four_metric_average',
        'mean_top1_iou',
    ):
        output[f'delta_{key}'] = output[key] - baseline[key]
    return output


def result_key(result):
    return (
        result['r1_average'],
        result['four_metric_average'],
        result['mean_top1_iou'],
        -result['ranking_changed_fraction'],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--output-predictions', type=Path, required=True)
    parser.add_argument(
        '--alphas',
        type=parse_csv_floats,
        default=DEFAULT_ALPHAS,
    )
    parser.add_argument(
        '--modes',
        type=parse_csv_strings,
        default=DEFAULT_MODES,
    )
    parser.add_argument(
        '--gates',
        type=parse_csv_strings,
        default=DEFAULT_GATES,
    )
    parser.add_argument('--max-queries', type=int)
    args = parser.parse_args()

    unknown_modes = set(args.modes) - set(DEFAULT_MODES)
    if unknown_modes:
        raise ValueError(f'Unknown modes: {sorted(unknown_modes)}')
    unknown_gates = set(args.gates) - set(DEFAULT_GATES)
    if unknown_gates:
        raise ValueError(f'Unknown gates: {sorted(unknown_gates)}')

    torch.set_num_threads(1)
    print('Loading predictions and computing level support...', flush=True)
    records, source_summary = load_records(
        args.predictions,
        args.modes,
        max_queries=args.max_queries,
    )
    print(f'Loaded {len(records)} queries', flush=True)

    baseline_mode = args.modes[0]
    baseline_gate = args.gates[0]
    baseline = evaluate(
        records, baseline_mode, baseline_gate, 0.0
    )
    print(
        f'baseline R1@0.3={baseline["r1_iou_0.3"] * 100:.2f} '
        f'R1@0.5={baseline["r1_iou_0.5"] * 100:.2f}',
        flush=True,
    )

    results = []
    for mode in args.modes:
        for gate in args.gates:
            for alpha in args.alphas:
                if alpha == 0:
                    result = dict(baseline)
                    result.update({
                        'mode': mode,
                        'gate': gate,
                        'alpha': alpha,
                    })
                else:
                    result = evaluate(
                        records, mode, gate, alpha
                    )
                result = add_deltas(result, baseline)
                results.append(result)
                print(
                    f'{mode} gate={gate} alpha={alpha:g} '
                    f'R1@0.3={result["r1_iou_0.3"] * 100:.2f} '
                    f'R1@0.5={result["r1_iou_0.5"] * 100:.2f} '
                    f'changed={result["ranking_changed_fraction"] * 100:.2f}%',
                    flush=True,
                )

    results.sort(key=result_key, reverse=True)
    best = results[0]
    best_by_mode = {
        mode: max(
            (result for result in results if result['mode'] == mode),
            key=result_key,
        )
        for mode in args.modes
    }
    best_by_gate = {
        gate: max(
            (result for result in results if result['gate'] == gate),
            key=result_key,
        )
        for gate in args.gates
    }

    best_metrics, best_predictions = evaluate(
        records,
        best['mode'],
        best['gate'],
        best['alpha'],
        collect_predictions=True,
    )
    best_metrics = add_deltas(best_metrics, baseline)
    best_predictions['post_nms_level_consensus'] = best_metrics

    report = {
        'predictions': str(args.predictions),
        'source_summary': source_summary,
        'num_queries': len(records),
        'support': (
            'confidence-weighted temporal IoU support from raw candidates; '
            'cross-level modes exclude the matched source level'
        ),
        'score_adjustment': (
            'log(nms_score) + alpha * gate * level_support'
        ),
        'selection_metric': (
            'R1 average, then four-metric average, then mean top1 IoU'
        ),
        'baseline': baseline,
        'best': best_metrics,
        'best_by_mode': best_by_mode,
        'best_by_gate': best_by_gate,
        'results': results,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + '\n')
    args.output_predictions.write_text(
        json.dumps(best_predictions, indent=2) + '\n'
    )

    print(json.dumps({'baseline': baseline, 'best': best_metrics}, indent=2))
    print(f'Wrote {args.output_json}')
    print(f'Wrote {args.output_predictions}')


if __name__ == '__main__':
    main()

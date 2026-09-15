#!/usr/bin/env python3
import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.nms import batched_nms


DEFAULT_ALPHAS = (0.0, 1.0, 2.0, 2.2, 3.0)
DEFAULT_MODES = ('all', 'cross_level', 'cross_level_balanced')


def parse_csv_floats(value):
    return tuple(float(item.strip()) for item in value.split(',') if item.strip())


def parse_csv_strings(value):
    return tuple(item.strip() for item in value.split(',') if item.strip())


def pairwise_temporal_iou(segments):
    left = torch.maximum(segments[:, None, 0], segments[None, :, 0])
    right = torch.minimum(segments[:, None, 1], segments[None, :, 1])
    intersection = (right - left).clamp(min=0)
    lengths = (segments[:, 1] - segments[:, 0]).clamp(min=0)
    union = lengths[:, None] + lengths[None, :] - intersection
    return intersection / union.clamp(min=1e-12)


def temporal_iou(segments, target):
    if len(segments) == 0:
        return torch.zeros(0, dtype=torch.float32)
    left = torch.maximum(segments[:, 0], target[0])
    right = torch.minimum(segments[:, 1], target[1])
    intersection = (right - left).clamp(min=0)
    segment_lengths = (segments[:, 1] - segments[:, 0]).clamp(min=0)
    target_length = (target[1] - target[0]).clamp(min=0)
    union = segment_lengths + target_length - intersection
    return intersection / union.clamp(min=1e-12)


def weighted_support(pairwise_iou, scores, mask):
    weights = scores[None, :] * mask.float()
    denominator = weights.sum(dim=1)
    numerator = (pairwise_iou * weights).sum(dim=1)
    return numerator / denominator.clamp(min=1e-12)


def compute_agreements(segments, scores, levels, modes):
    pairwise_iou = pairwise_temporal_iou(segments)
    agreements = {}
    num_candidates = len(scores)

    if 'all' in modes:
        mask = ~torch.eye(num_candidates, dtype=torch.bool)
        agreements['all'] = weighted_support(pairwise_iou, scores, mask)

    if 'cross_level' in modes:
        mask = levels[:, None] != levels[None, :]
        agreements['cross_level'] = weighted_support(
            pairwise_iou, scores, mask
        )

    if 'cross_level_balanced' in modes:
        support_sum = torch.zeros_like(scores)
        support_count = torch.zeros_like(scores)
        for level in torch.unique(levels):
            level_mask = levels == level
            level_scores = scores[level_mask]
            level_support = (
                pairwise_iou[:, level_mask] * level_scores[None, :]
            ).sum(dim=1) / level_scores.sum().clamp(min=1e-12)
            valid = levels != level
            support_sum[valid] += level_support[valid]
            support_count[valid] += 1
        agreements['cross_level_balanced'] = (
            support_sum / support_count.clamp(min=1)
        )

    return agreements


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
            segments = torch.tensor(
                [item['segment'] for item in raw_candidates],
                dtype=torch.float32,
            )
            scores = torch.tensor(
                [item['score'] for item in raw_candidates],
                dtype=torch.float32,
            )
            levels = torch.tensor(
                [item['level'] for item in raw_candidates],
                dtype=torch.long,
            )
            records.append({
                'video_id': video_id,
                'query_id': query['query_id'],
                'ground_truth': torch.tensor(
                    query['ground_truth'], dtype=torch.float32
                ),
                'segments': segments,
                'scores': scores,
                'levels': levels,
                'agreements': compute_agreements(
                    segments, scores, levels, modes
                ),
            })
            if max_queries is not None and len(records) >= max_queries:
                return records, predictions.get('summary', {})

    return records, predictions.get('summary', {})


def adjust_scores(scores, agreement, alpha):
    adjusted = scores * torch.exp(alpha * agreement)
    max_adjusted = adjusted.max().clamp(min=1e-12)
    return adjusted * (scores.max() / max_adjusted)


def run_nms(segments, scores, nms_config):
    return batched_nms(
        segments,
        scores,
        mode=nms_config['mode'],
        iou_thresh=nms_config['iou_thresh'],
        min_score=nms_config['min_score'],
        max_num_segs=nms_config['max_num_segs'],
        sigma=nms_config['sigma'],
        voting_thresh=nms_config['voting_thresh'],
    )


def evaluate(records, mode, alpha, nms_config, collect_predictions=False):
    metric_counts = torch.zeros((2, 2), dtype=torch.long)
    top1_iou_sum = 0.0
    ranking_changed = 0
    selected_agreement_sum = 0.0
    output_videos = OrderedDict()

    for record in records:
        scores = record['scores']
        agreement = record['agreements'][mode]
        adjusted_scores = adjust_scores(scores, agreement, alpha)
        selected_idx = int(adjusted_scores.argmax().item())
        ranking_changed += selected_idx != int(scores.argmax().item())
        selected_agreement_sum += float(agreement[selected_idx].item())

        nms_segments, nms_scores = run_nms(
            record['segments'], adjusted_scores, nms_config
        )
        candidate_ious = temporal_iou(
            nms_segments, record['ground_truth']
        )
        top1_iou = float(candidate_ious[0].item()) if len(candidate_ious) else 0.0
        top1_iou_sum += top1_iou

        for rank_idx, rank in enumerate((1, 5)):
            best_iou = (
                float(candidate_ious[:rank].max().item())
                if len(candidate_ious) else 0.0
            )
            metric_counts[rank_idx, 0] += best_iou >= 0.3
            metric_counts[rank_idx, 1] += best_iou >= 0.5

        if collect_predictions:
            video = output_videos.setdefault(
                record['video_id'],
                {'queries': [], 'recall_at_iou': {}},
            )
            video['queries'].append({
                'query_id': record['query_id'],
                'ground_truth': record['ground_truth'].tolist(),
                'predictions': [
                    {
                        'segment': segment.tolist(),
                        'score': float(score.item()),
                    }
                    for segment, score in zip(nms_segments, nms_scores)
                ],
            })

    num_queries = len(records)
    metrics = metric_counts.float() / num_queries
    result = {
        'mode': mode,
        'alpha': alpha,
        'r1_iou_0.3': float(metrics[0, 0].item()),
        'r1_iou_0.5': float(metrics[0, 1].item()),
        'r5_iou_0.3': float(metrics[1, 0].item()),
        'r5_iou_0.5': float(metrics[1, 1].item()),
        'r1_average': float(metrics[0].mean().item()),
        'four_metric_average': float(metrics.mean().item()),
        'mean_top1_iou': top1_iou_sum / num_queries,
        'ranking_changed_fraction': ranking_changed / num_queries,
        'mean_selected_agreement': selected_agreement_sum / num_queries,
    }

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
            'level_consensus': result,
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
    parser.add_argument('--max-queries', type=int)
    parser.add_argument('--iou-thresh', type=float, default=0.1)
    parser.add_argument('--min-score', type=float, default=0.001)
    parser.add_argument('--max-num-segs', type=int, default=5)
    parser.add_argument('--sigma', type=float, default=0.9)
    parser.add_argument('--voting-thresh', type=float, default=0.95)
    args = parser.parse_args()

    unknown_modes = set(args.modes) - set(DEFAULT_MODES)
    if unknown_modes:
        raise ValueError(f'Unknown modes: {sorted(unknown_modes)}')
    if not args.alphas:
        raise ValueError('At least one alpha is required')

    torch.set_num_threads(1)
    nms_config = {
        'mode': 'soft_nms',
        'iou_thresh': args.iou_thresh,
        'min_score': args.min_score,
        'max_num_segs': args.max_num_segs,
        'sigma': args.sigma,
        'voting_thresh': args.voting_thresh,
    }

    print('Loading raw candidates and computing agreements...', flush=True)
    records, source_summary = load_records(
        args.predictions,
        args.modes,
        max_queries=args.max_queries,
    )
    print(f'Loaded {len(records)} queries', flush=True)

    baseline_mode = args.modes[0]
    baseline = evaluate(
        records, baseline_mode, 0.0, nms_config
    )
    print(
        f'baseline R1@0.3={baseline["r1_iou_0.3"] * 100:.2f} '
        f'R1@0.5={baseline["r1_iou_0.5"] * 100:.2f}',
        flush=True,
    )

    results = []
    for mode in args.modes:
        for alpha in args.alphas:
            if alpha == 0:
                result = dict(baseline)
                result['mode'] = mode
            else:
                result = evaluate(
                    records, mode, alpha, nms_config
                )
            result = add_deltas(result, baseline)
            results.append(result)
            print(
                f'{mode} alpha={alpha:g} '
                f'R1@0.3={result["r1_iou_0.3"] * 100:.2f} '
                f'R1@0.5={result["r1_iou_0.5"] * 100:.2f} '
                f'R5@0.3={result["r5_iou_0.3"] * 100:.2f} '
                f'R5@0.5={result["r5_iou_0.5"] * 100:.2f}',
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

    best_metrics, best_predictions = evaluate(
        records,
        best['mode'],
        best['alpha'],
        nms_config,
        collect_predictions=True,
    )
    best_metrics = add_deltas(best_metrics, baseline)
    best_predictions['level_consensus'] = best_metrics

    report = {
        'predictions': str(args.predictions),
        'source_summary': source_summary,
        'num_queries': len(records),
        'score_adjustment': (
            'score * exp(alpha * agreement), normalized per query '
            'to preserve the original maximum score'
        ),
        'selection_metric': (
            'R1 average, then four-metric average, then mean top1 IoU'
        ),
        'nms': nms_config,
        'baseline': baseline,
        'best': best_metrics,
        'best_by_mode': best_by_mode,
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

#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


def temporal_iou(segment, target):
    inter = max(0.0, min(segment[1], target[1]) - max(segment[0], target[0]))
    union = max(segment[1], target[1]) - min(segment[0], target[0])
    return inter / union if union > 0 else 0.0


def consensus_scores(candidates, alpha):
    adjusted = []
    for idx, candidate in enumerate(candidates):
        weighted_overlap = 0.0
        weight_sum = 0.0
        for other_idx, other in enumerate(candidates):
            if idx == other_idx:
                continue
            weight = max(other['score'], 0.0)
            weighted_overlap += weight * temporal_iou(
                candidate['segment'], other['segment']
            )
            weight_sum += weight
        agreement = weighted_overlap / weight_sum if weight_sum > 0 else 0.0
        adjusted_score = math.log(max(candidate['score'], 1e-12)) + alpha * agreement
        adjusted.append((adjusted_score, agreement, idx))
    return adjusted


def fuse_segment(candidate_idx, candidates, threshold):
    anchor = candidates[candidate_idx]['segment']
    if threshold is None:
        return list(anchor)

    weighted_start = 0.0
    weighted_end = 0.0
    weight_sum = 0.0
    for candidate in candidates:
        if temporal_iou(anchor, candidate['segment']) < threshold:
            continue
        weight = max(candidate['score'], 1e-12)
        weighted_start += weight * candidate['segment'][0]
        weighted_end += weight * candidate['segment'][1]
        weight_sum += weight
    if weight_sum == 0:
        return list(anchor)
    return [weighted_start / weight_sum, weighted_end / weight_sum]


def evaluate(predictions, alpha, fusion_threshold):
    top1_ious = []
    top5_ious = []
    changed = 0
    for video in predictions['videos'].values():
        for query in video['queries']:
            candidates = query['predictions']
            if not candidates:
                top1_ious.append(0.0)
                top5_ious.append(0.0)
                continue

            ranking = sorted(
                consensus_scores(candidates, alpha),
                key=lambda item: item[0],
                reverse=True,
            )
            top_idx = ranking[0][2]
            changed += top_idx != 0
            top_segment = fuse_segment(top_idx, candidates, fusion_threshold)
            top1_ious.append(temporal_iou(top_segment, query['ground_truth']))
            top5_ious.append(max(
                temporal_iou(candidate['segment'], query['ground_truth'])
                for candidate in candidates
            ))

    count = len(top1_ious)
    r1_03 = sum(iou >= 0.3 for iou in top1_ious) / count
    r1_05 = sum(iou >= 0.5 for iou in top1_ious) / count
    r5_03 = sum(iou >= 0.3 for iou in top5_ious) / count
    r5_05 = sum(iou >= 0.5 for iou in top5_ious) / count
    return {
        'alpha': alpha,
        'fusion_threshold': fusion_threshold,
        'r1_iou_0.3': r1_03,
        'r1_iou_0.5': r1_05,
        'r5_iou_0.3': r5_03,
        'r5_iou_0.5': r5_05,
        'r1_average': 0.5 * (r1_03 + r1_05),
        'four_metric_average': 0.25 * (r1_03 + r1_05 + r5_03 + r5_05),
        'mean_top1_iou': sum(top1_ious) / count,
        'ranking_changed_fraction': changed / count,
    }


def rerank_predictions(predictions, alpha, fusion_threshold):
    output = json.loads(json.dumps(predictions))
    for video in output['videos'].values():
        for query in video['queries']:
            candidates = query['predictions']
            if not candidates:
                continue
            ranking = sorted(
                consensus_scores(candidates, alpha),
                key=lambda item: item[0],
                reverse=True,
            )
            reranked = []
            for adjusted_score, agreement, candidate_idx in ranking:
                candidate = dict(candidates[candidate_idx])
                candidate['original_score'] = candidate['score']
                candidate['consensus'] = agreement
                candidate['consensus_score'] = adjusted_score
                candidate['segment'] = fuse_segment(
                    candidate_idx, candidates, fusion_threshold
                )
                reranked.append(candidate)
            query['predictions'] = reranked
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--output-predictions', type=Path, required=True)
    parser.add_argument('--alpha-max', type=float, default=4.0)
    parser.add_argument('--alpha-step', type=float, default=0.1)
    args = parser.parse_args()

    with args.predictions.open() as file_obj:
        predictions = json.load(file_obj)

    alphas = []
    alpha = 0.0
    while alpha <= args.alpha_max + 1e-9:
        alphas.append(round(alpha, 10))
        alpha += args.alpha_step
    fusion_thresholds = [None, 0.3, 0.5, 0.7, 0.9]

    results = [
        evaluate(predictions, alpha, fusion_threshold)
        for alpha in alphas
        for fusion_threshold in fusion_thresholds
    ]
    results.sort(
        key=lambda result: (
            result['r1_average'],
            result['mean_top1_iou'],
            -result['ranking_changed_fraction'],
        ),
        reverse=True,
    )
    best = results[0]
    baseline = evaluate(predictions, 0.0, None)
    report = {
        'predictions': str(args.predictions),
        'selection_metric': 'mean of R1@0.3 and R1@0.5',
        'baseline': baseline,
        'best': best,
        'top_results': results[:20],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + '\n')
    reranked = rerank_predictions(
        predictions,
        best['alpha'],
        best['fusion_threshold'],
    )
    reranked['consensus_rerank'] = best
    args.output_predictions.write_text(json.dumps(reranked, indent=2) + '\n')

    print(json.dumps(report, indent=2))
    print(f'Wrote {args.output_json}')
    print(f'Wrote {args.output_predictions}')


if __name__ == '__main__':
    main()

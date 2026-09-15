#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


def temporal_iou(segment, target):
    inter = max(0.0, min(segment[1], target[1]) - max(segment[0], target[0]))
    union = max(segment[1], target[1]) - min(segment[0], target[0])
    return inter / union if union > 0 else 0.0


def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def mean(values):
    return sum(values) / len(values) if values else 0.0


def summarize(records):
    if not records:
        return {'count': 0}

    top1_ious = [record['top1_iou'] for record in records]
    best5_ious = [record['best5_iou'] for record in records]
    return {
        'count': len(records),
        'mean_duration': mean([record['duration'] for record in records]),
        'mean_top1_iou': mean(top1_ious),
        'median_top1_iou': percentile(top1_ious, 0.5),
        'mean_best5_iou': mean(best5_ious),
        'r1_iou_0.3': mean([iou >= 0.3 for iou in top1_ious]),
        'r1_iou_0.5': mean([iou >= 0.5 for iou in top1_ious]),
        'r5_iou_0.3': mean([iou >= 0.3 for iou in best5_ious]),
        'r5_iou_0.5': mean([iou >= 0.5 for iou in best5_ious]),
        'mean_start_error_sec': mean([record['start_error'] for record in records]),
        'mean_end_error_sec': mean([record['end_error'] for record in records]),
        'mean_center_error_sec': mean([record['center_error'] for record in records]),
        'mean_normalized_boundary_error': mean(
            [record['normalized_boundary_error'] for record in records]
        ),
        'mean_top1_score': mean([record['top1_score'] for record in records]),
        'mean_top1_consensus': mean([record['top1_consensus'] for record in records]),
        'top1_fail_r5_rescue_iou_0.5': mean(
            [record['top1_iou'] < 0.5 <= record['best5_iou'] for record in records]
        ),
        'mean_duplicate_pair_rate': mean(
            [record['duplicate_pair_rate'] for record in records]
        ),
    }


def group_by_ranges(records, ranges):
    groups = []
    for label, lower, upper in ranges:
        selected = [
            record for record in records
            if record['duration'] > lower and record['duration'] <= upper
        ]
        groups.append({'label': label, **summarize(selected)})
    return groups


def score_calibration(records, num_bins=10):
    ordered = sorted(records, key=lambda record: record['top1_score'])
    bins = []
    for bin_idx in range(num_bins):
        start = round(bin_idx * len(ordered) / num_bins)
        end = round((bin_idx + 1) * len(ordered) / num_bins)
        selected = ordered[start:end]
        if not selected:
            continue
        bins.append({
            'count': len(selected),
            'score_min': selected[0]['top1_score'],
            'score_max': selected[-1]['top1_score'],
            'mean_score': mean([record['top1_score'] for record in selected]),
            'mean_iou': mean([record['top1_iou'] for record in selected]),
            'accuracy_iou_0.5': mean(
                [record['top1_iou'] >= 0.5 for record in selected]
            ),
            'mean_consensus': mean(
                [record['top1_consensus'] for record in selected]
            ),
        })
    return bins


def load_sentences(annotation_path, split):
    if annotation_path is None:
        return {}
    with annotation_path.open() as file_obj:
        annotations = json.load(file_obj)[split]
    sentences = {}
    for video_id, video in annotations.items():
        for query_idx, annotation in enumerate(video['annotations']):
            sentences[(video_id, query_idx)] = annotation.get('sentence', '')
    return sentences


def build_records(predictions, sentences):
    records = []
    for video_id, video in predictions['videos'].items():
        for query in video['queries']:
            target = query['ground_truth']
            candidates = query['predictions']
            candidate_ious = [
                temporal_iou(candidate['segment'], target)
                for candidate in candidates
            ]
            if candidates:
                top1 = candidates[0]
                top1_iou = candidate_ious[0]
                top1_score = top1['score']
                top1_segment = top1['segment']
            else:
                top1_iou = 0.0
                top1_score = 0.0
                top1_segment = [0.0, 0.0]

            other_ious = [
                temporal_iou(top1_segment, candidate['segment'])
                for candidate in candidates[1:]
            ]
            other_scores = [candidate['score'] for candidate in candidates[1:]]
            score_total = sum(other_scores)
            top1_consensus = (
                sum(score * overlap for score, overlap in zip(other_scores, other_ious))
                / score_total
                if score_total > 0 else 0.0
            )

            pair_count = 0
            duplicate_count = 0
            for left_idx in range(len(candidates)):
                for right_idx in range(left_idx + 1, len(candidates)):
                    pair_count += 1
                    overlap = temporal_iou(
                        candidates[left_idx]['segment'],
                        candidates[right_idx]['segment'],
                    )
                    duplicate_count += overlap >= 0.95

            duration = max(target[1] - target[0], 1e-6)
            top1_center = 0.5 * (top1_segment[0] + top1_segment[1])
            target_center = 0.5 * (target[0] + target[1])
            sentence = sentences.get((video_id, query['query_id']), '')
            records.append({
                'video_id': video_id,
                'query_id': query['query_id'],
                'sentence': sentence,
                'query_words': len(sentence.split()),
                'target': target,
                'top1_segment': top1_segment,
                'duration': duration,
                'top1_score': top1_score,
                'top1_iou': top1_iou,
                'best5_iou': max(candidate_ious, default=0.0),
                'start_error': abs(top1_segment[0] - target[0]),
                'end_error': abs(top1_segment[1] - target[1]),
                'center_error': abs(top1_center - target_center),
                'normalized_boundary_error': (
                    abs(top1_segment[0] - target[0])
                    + abs(top1_segment[1] - target[1])
                ) / (2.0 * duration),
                'top1_consensus': top1_consensus,
                'duplicate_pair_rate': (
                    duplicate_count / pair_count if pair_count else 0.0
                ),
            })
    return records


def markdown_table(rows, columns):
    header = '| ' + ' | '.join(label for _, label in columns) + ' |'
    separator = '| ' + ' | '.join('---' for _ in columns) + ' |'
    lines = [header, separator]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key, 0)
            if isinstance(value, float):
                values.append(f'{value:.4f}')
            else:
                values.append(str(value))
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def write_markdown(report, output_path):
    overall = report['overall']
    lines = [
        '# HieraMamba TACoS Prediction Diagnostics',
        '',
        f"Queries: {overall['count']}",
        '',
        '## Overall',
        '',
        markdown_table([overall], [
            ('r1_iou_0.3', 'R1@0.3'),
            ('r1_iou_0.5', 'R1@0.5'),
            ('r5_iou_0.3', 'R5@0.3'),
            ('r5_iou_0.5', 'R5@0.5'),
            ('mean_top1_iou', 'Mean top1 IoU'),
            ('mean_best5_iou', 'Mean best5 IoU'),
        ]),
        '',
        '## Duration Buckets',
        '',
        markdown_table(report['duration_buckets_fixed'], [
            ('label', 'Bucket'),
            ('count', 'Count'),
            ('mean_duration', 'Mean duration'),
            ('r1_iou_0.3', 'R1@0.3'),
            ('r1_iou_0.5', 'R1@0.5'),
            ('r5_iou_0.5', 'R5@0.5'),
            ('mean_normalized_boundary_error', 'Norm boundary error'),
        ]),
        '',
        '## Ranking And Boundary Headroom',
        '',
        markdown_table([overall], [
            ('top1_fail_r5_rescue_iou_0.5', 'Top5 rescues top1@0.5'),
            ('mean_start_error_sec', 'Start error (s)'),
            ('mean_end_error_sec', 'End error (s)'),
            ('mean_center_error_sec', 'Center error (s)'),
            ('mean_top1_consensus', 'Top1 consensus'),
            ('mean_duplicate_pair_rate', 'Duplicate pair rate'),
        ]),
        '',
        '## Score Calibration',
        '',
        markdown_table(report['score_calibration'], [
            ('count', 'Count'),
            ('score_min', 'Score min'),
            ('score_max', 'Score max'),
            ('mean_score', 'Mean score'),
            ('mean_iou', 'Mean IoU'),
            ('accuracy_iou_0.5', 'Accuracy@0.5'),
            ('mean_consensus', 'Consensus'),
        ]),
        '',
        '## Worst High-Confidence Predictions',
        '',
    ]
    for record in report['worst_high_confidence']:
        lines.append(
            f"- {record['video_id']}:{record['query_id']} score={record['top1_score']:.4f} "
            f"IoU={record['top1_iou']:.4f} query={record['sentence']}"
        )
    output_path.write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--annotations', type=Path)
    parser.add_argument('--split', default='val')
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--output-md', type=Path, required=True)
    args = parser.parse_args()

    with args.predictions.open() as file_obj:
        predictions = json.load(file_obj)
    sentences = load_sentences(args.annotations, args.split)
    records = build_records(predictions, sentences)

    durations = [record['duration'] for record in records]
    q1 = percentile(durations, 1.0 / 3.0)
    q2 = percentile(durations, 2.0 / 3.0)
    report = {
        'predictions': str(args.predictions),
        'split': args.split,
        'overall': summarize(records),
        'duration_buckets_fixed': group_by_ranges(records, [
            ('short (0, 10s]', 0.0, 10.0),
            ('medium (10s, 30s]', 10.0, 30.0),
            ('long (30s, inf)', 30.0, float('inf')),
        ]),
        'duration_buckets_quantile': group_by_ranges(records, [
            (f'short (0, {q1:.2f}s]', 0.0, q1),
            (f'medium ({q1:.2f}s, {q2:.2f}s]', q1, q2),
            (f'long ({q2:.2f}s, inf)', q2, float('inf')),
        ]),
        'score_calibration': score_calibration(records),
        'worst_high_confidence': sorted(
            [record for record in records if record['top1_iou'] < 0.3],
            key=lambda record: record['top1_score'],
            reverse=True,
        )[:20],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + '\n')
    write_markdown(report, args.output_md)

    print(json.dumps(report['overall'], indent=2))
    print(f'Wrote {args.output_json}')
    print(f'Wrote {args.output_md}')


if __name__ == '__main__':
    main()

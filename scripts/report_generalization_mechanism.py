#!/usr/bin/env python3
"""Descriptive accounting after the frozen analysis; no new acceptance tests."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np


def geometric(values):
    return math.exp(statistics.mean(math.log(x) for x in values))


def features(row):
    metrics = row['assignment_metrics']
    service = np.subtract(row['coordinator_finish_ns'], row['coordinator_start_ns'])
    boundary = np.array([63, 127, 191, 255])
    ordinary = np.delete(service, boundary)
    gaps = np.diff(row['arrival_offsets_ns'])
    return {
        'mean_active_physical_copies': statistics.mean(sum(x['per_rank_active_copies']) for x in metrics),
        'mean_max_rank_active_copies': statistics.mean(max(x['per_rank_active_copies']) for x in metrics),
        'mean_max_rank_padded_rows': statistics.mean(max(x['per_rank_padded_rows']) for x in metrics),
        'mean_max_rank_m_tiles': statistics.mean(max(x['per_rank_m_tiles']) for x in metrics),
        'mean_rank_token_imbalance': statistics.mean(2*max(x['per_rank_tokens'])/sum(x['per_rank_tokens']) for x in metrics),
        'ordinary_service_mean_ns': float(ordinary.mean()),
        'boundary_service_mean_ns': float(service[boundary].mean()),
        'boundary_excess_ns_per_trial': float(service[boundary].sum()-4*ordinary.mean()),
        'interarrival_gap_cv': float(gaps.std()/gaps.mean()),
        'zero_interarrival_gap_fraction': float((gaps == 0).mean()),
        'largest_gap_over_mean': float(gaps.max()/gaps.mean()),
    }


def ranks(values):
    """Average tied ranks, implemented here to avoid an extra dependency."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = np.empty(len(values), dtype=float)
    at = 0
    while at < len(order):
        end = at+1
        while end < len(order) and values[order[end]] == values[order[at]]:
            end += 1
        out[order[at:end]] = (at+end-1)/2
        at = end
    return out


def correlation(x, y):
    a, b = ranks(x), ranks(y)
    if not a.std() or not b.std():
        return None
    return float(np.corrcoef(a, b)[0, 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('records', 'summary', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data = args.records.read_bytes()
    payload = gzip.decompress(data) if args.records.suffix == '.gz' else data
    rows = [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
    summary = json.loads(args.summary.read_text())
    if not summary['valid'] or hashlib.sha256(payload).hexdigest() != summary['records_jsonl_payload_sha256']:
        raise ValueError('Complete verified raw records and matching summary required')
    if summary['phase'] not in ('pilot', 'final'):
        raise ValueError('Treatment records required')
    index = {(r['content_group_id'], r['arrival_type'], r['rate_kind'], r['arm']): r for r in rows}
    feature_index = {k: features(r) for k, r in index.items()}
    paired = []
    by_workload = defaultdict(list)
    by_content = defaultdict(list)
    for pair in summary['paired_cases']:
        if pair['contrast'] != 'combined_over_shortcut':
            continue
        key = tuple(pair[k] for k in ('content_group_id', 'arrival_type', 'rate_kind'))
        control, treatment = feature_index[(*key, 'shortcut')], feature_index[(*key, 'combined')]
        item = {k: pair[k] for k in ('content_group_id', 'dataset', 'arrival_type', 'rate_kind', 'arrival_date')}
        item.update(control_features=control, combined_features=treatment,
                    timing_ratios=pair['ratios'], timing_differences=pair['differences'],
                    feature_ratios={k: treatment[k]/control[k] if control[k] else None for k in control},
                    feature_differences={k: treatment[k]-control[k] for k in control})
        # Response = queue + service on the same completed jobs. This identity
        # is an accounting decomposition, not a causal mediation estimate.
        delta = pair['differences']
        residual = delta['response_mean_ns']-delta['service_mean_ns']-delta['queue_wait_mean_ns']
        if abs(residual) > 1e-6:
            raise ValueError('Response decomposition does not match raw timing')
        item['response_decomposition_residual_ns'] = residual
        paired.append(item)
        by_workload[pair['dataset'], pair['rate_kind']].append(pair)
        if pair['rate_kind'] == 'calibrated':
            by_content[pair['content_group_id']].append(item)
    workloads = []
    for (dataset, rate), pairs in sorted(by_workload.items()):
        item = dict(dataset=dataset, rate_kind=rate, paired_cases=len(pairs))
        item['geometric_timing_ratios'] = {
            m: geometric([p['ratios'][m] for p in pairs])
            for m in ('response_mean_ns', 'response_p95_ns', 'service_mean_ns',
                      'observed_elapsed_seconds', 'completion_elapsed_seconds',
                      'completed_per_evaluation_second')}
        item['mean_timing_differences'] = {m: statistics.mean(p['differences'][m] for p in pairs)
            for m in ('response_mean_ns', 'queue_wait_mean_ns', 'service_mean_ns', 'drain_ns')}
        arm_means = {}
        for arm in ('native', 'shortcut', 'combined'):
            trials = [r for r in summary['trial_summaries'] if r['dataset'] == dataset and r['rate_kind'] == rate and r['arm'] == arm]
            arm_means[arm] = {m: statistics.mean(r['metrics'][m] for r in trials)
                for m in ('response_mean_ns', 'response_p95_ns', 'service_mean_ns', 'queue_wait_mean_ns',
                          'observed_elapsed_seconds', 'completion_elapsed_seconds', 'drain_ns',
                          'completed_per_evaluation_second')}
        item['arithmetic_mean_trial_metrics_by_arm'] = arm_means
        workloads.append(item)
    content = []
    for gid, pairs in sorted(by_content.items()):
        content.append(dict(content_group_id=gid, dataset=pairs[0]['dataset'],
            log_mean_response_ratio=statistics.mean(math.log(x['timing_ratios']['response_mean_ns']) for x in pairs),
            max_rank_active_reduction=statistics.mean(1-x['feature_ratios']['mean_max_rank_active_copies'] for x in pairs),
            padded_row_reduction=statistics.mean(1-x['feature_ratios']['mean_max_rank_padded_rows'] for x in pairs),
            token_imbalance_change=statistics.mean(x['feature_differences']['mean_rank_token_imbalance'] for x in pairs),
            mean_gap_cv=statistics.mean(x['control_features']['interarrival_gap_cv'] for x in pairs)))
    associations = {feature: correlation([x[feature] for x in content], [x['log_mean_response_ratio'] for x in content])
        for feature in ('max_rank_active_reduction', 'padded_row_reduction', 'token_imbalance_change', 'mean_gap_cv')}
    tile_equals_active = all(m['per_rank_m_tiles'] == m['per_rank_active_copies'] for r in rows for m in r['assignment_metrics'])
    result = dict(schema='generalization_descriptive_mechanism_v1', phase=summary['phase'],
        every_rank_job_has_one_tile_per_active_copy=tile_equals_active,
        frozen_acceptance_unchanged=True, paired_cases=paired, workloads=workloads,
        content_bundle_descriptions=content, exploratory_spearman_with_log_response_ratio=associations,
        records_jsonl_payload_sha256=hashlib.sha256(payload).hexdigest(),
        summary_sha256=hashlib.sha256(args.summary.read_bytes()).hexdigest(),
        reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=[
            'Descriptive post-hoc accounting, not new confirmatory tests or a fitted dispatch rule.',
            'Active copies and padded tile counts are structural proxies, not measured FLOPs, memory traffic, utilization or isolated kernel time.',
            'Maximum-count rank is not necessarily the measured critical-path rank. When every active copy uses one tile, active-count and padding reductions are the same signal, not independent evidence.',
            'Boundary excess includes ordinary job variation and all charged coordination/control work; it does not isolate a specific function.',
            'Across-content correlations mix datasets and assigned arrivals; they do not identify causal effects or out-of-sample prediction.',
            'Separate offline assignment diagnostics are needed to distinguish execution savings from selection/adapter overhead.',
            'Arithmetic means of trial metrics and geometric means of paired ratios have different estimands.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'phase': result['phase'], 'paired_cases': len(paired), 'workloads': len(workloads)}))


if __name__ == '__main__':
    main()

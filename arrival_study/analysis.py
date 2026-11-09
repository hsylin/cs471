"""Descriptive, all-job analysis of the supplementary layer-arrival study.

This module does not estimate population confidence intervals or make a new
performance acceptance decision. Raw clocks are the source of reported metrics.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import statistics

from arrival_study.schedules import calibration_capacity, fcfs_metrics, validate_schedule


IDENTITY = ('block_id', 'regime', 'batch_tokens', 'schedule_name', 'repetition', 'arm')
MATCH = ('capture_sha256', 'replay_sha256', 'source_prompt_ids', 'source_weights_sha256',
         'source_weight_rank_sha256', 'source_map', 'final_map', 'resolved_kernel_config',
         'logical_routing_sha256', 'decision_trajectory_sha256', 'arrival_clock_contract',
         'warmup_raw_indices', 'conditioning_raw_indices')
SCALARS = ('backlog_at_horizon', 'waiting_jobs_at_horizon', 'drain_ns',
           'trailing_scheduled_idle_ns', 'offered_jobs_per_second',
           'completed_by_horizon_per_second', 'completed_per_evaluation_second',
           'required_observation_extent_ns', 'observation_overshoot_ns')


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _state(group):
    return f'{group[0]}/{group[1]}/B{group[2]}'


def _schedule_name(schedule):
    return f"{schedule['pattern']}-rho{schedule['rho']}"


def _flat_metrics(metrics):
    result = {name: metrics[name] for name in SCALARS}
    exposure = metrics['required_observation_extent_ns'] + metrics['observation_overshoot_ns']
    result['coordinator_server_busy_fraction'] = sum(metrics['vectors_ns']['service']) / exposure
    for category, values in metrics['summary_ns'].items():
        result.update({f'{category}_{stat}_ns': value for stat, value in values.items()})
    for category in ('queue_occupancy', 'system_occupancy'):
        result.update({f'{category}_{key}': value for key, value in metrics[category].items()})
    return result


def _contrast(native, online):
    return {key: {'native': value, 'online': online[key], 'online_minus_native': online[key] - value,
                  'online_over_native': online[key] / value if value else None}
            for key, value in native.items()}


def _spans(row):
    result = []
    for end in (63, 127, 191, 255):
        begin = end - 15
        result.append(dict(end_step=end, jobs=16,
            scheduled_arrival_span_ns=row['arrival_offsets_ns'][end] - row['arrival_offsets_ns'][begin],
            service_coverage_span_ns=row['coordinator_finish_ns'][end] - row['coordinator_start_ns'][begin],
            completion_span_ns=row['coordinator_finish_ns'][end] - row['coordinator_finish_ns'][begin]))
    return result


def analyze(plan, records):
    if ('analysis_file_sha256' in plan
            and plan['analysis_file_sha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
        raise ValueError('analysis implementation differs from the pinned plan')
    phase = plan['phase']
    matrix = plan['matrix']
    if phase not in ('calibration', 'evaluation'):
        raise ValueError('unsupported arrival-study phase')
    if (len(matrix['block_ids']) != 2 or len(set(matrix['block_ids'])) != 2
            or set(matrix['regimes']) != {'domain0', 'domain1'} or len(matrix['regimes']) != 2
            or set(matrix['batch_tokens']) != {128, 1024} or len(matrix['batch_tokens']) != 2
            or matrix['arms'] != ['native', 'online'] or matrix['repetitions'] != 2):
        raise ValueError('the fixed design requires two development blocks, two domains, two batches and two arms/repeats')
    plan_hash = _hash({key: value for key, value in plan.items() if key != 'freeze'})
    if plan.get('freeze', {}).get('sha256') != plan_hash:
        raise ValueError('unfrozen or altered plan')
    states = list(itertools.product(matrix['block_ids'], matrix['regimes'], matrix['batch_tokens']))
    expected = {}
    for state in states:
        if phase == 'calibration':
            expected[(*state, 'calibration_saturated', 0, 'native')] = None
        else:
            schedules = plan['schedules'][_state(state)]
            if len(schedules) != 4 or {(s['pattern'], s['rho']) for s in schedules} != {
                    ('constant', .5), ('constant', .85), ('burst8', .5), ('burst8', .85)}:
                raise ValueError('exactly four prespecified calendars are required per state')
            for schedule in schedules:
                validate_schedule(schedule)
                if schedule['n_jobs'] != 256:
                    raise ValueError('every trial must retain all 256 scheduled jobs')
                for rep, arm in itertools.product((0, 1), ('native', 'online')):
                    expected[(*state, _schedule_name(schedule), rep, arm)] = schedule
    errors, index, metrics_by_key, trials = [], {}, {}, []
    for row in records:
        key = tuple(row.get(field) for field in IDENTITY)
        if key not in expected:
            errors.append(f'unexpected record: {key}'); continue
        if key in index:
            errors.append(f'duplicate record: {key}'); continue
        index[key] = row
        issues = []
        for field, required in (('status', 'complete'), ('phase', phase), ('config_hash', plan_hash),
                ('arrival_source_sha256', plan['arrival_source_sha256']), ('evidence_eligible', False),
                ('correctness_pass', True), ('normal_completion_verified', True),
                ('resident_state_verified', True), ('rank_assignment_agreement', True),
                ('migration_bytes', 0), ('migration_sent_messages', 0),
                ('terminal_proposals_suppressed', 1), ('policy', 'SKIP')):
            if row.get(field) != required:
                issues.append(field)
        if row.get('preflight_correctness', {}).get('correctness_pass') is not True:
            issues.append('preflight_correctness')
        if row.get('source_map') is None or row.get('source_map') != row.get('final_map'):
            issues.append('fixed_map')
        if not row.get('allocation_id'):
            issues.append('allocation_id')
        clock = row.get('arrival_clock_contract', {})
        if (clock.get('monotonic') is not True or clock.get('adjustable') is not False
                or 'CLOCK_MONOTONIC' not in clock.get('implementation', '')):
            issues.append('shared_clock_contract')
        schedule = expected[key]
        if phase == 'evaluation':
            calibration = plan.get('calibration', {}).get(_state(key[:3]), {})
            if str(row.get('allocation_id')) != str(calibration.get('allocation_id')):
                issues.append('calibration_evaluation_allocation_mismatch')
        if row.get('arrival_schedule') != schedule:
            issues.append('arrival_schedule')
        if schedule is not None and (row['arrival_offsets_ns'] != schedule['arrival_offsets_ns']
                or row['horizon_ns'] != schedule['horizon_ns']):
            issues.append('raw_calendar')
        if schedule is None and (row.get('arrival_offsets_ns') != [0]*256 or row.get('horizon_ns') != 1):
            issues.append('saturated_calibration_calendar')
        try:
            metrics = fcfs_metrics(row)
            if not metrics['complete'] or metrics['scheduled_jobs'] != 256:
                issues.append('incomplete_job_accounting')
            if row.get('queue_metrics') != metrics:
                issues.append('stored_queue_metrics_differ_from_raw_timestamps')
            if metrics['complete']:
                spans = _spans(row)
                if row.get('observed_window_spans') != spans:
                    issues.append('observed_window_spans')
                if phase == 'calibration':
                    capacity = calibration_capacity(row['coordinator_start_ns'], row['coordinator_finish_ns'])
                    if row.get('calibration_capacity') != capacity:
                        issues.append('calibration_capacity')
            else:
                spans = []
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            issues.append(f'raw_timing: {exc}')
        if issues:
            errors.append(dict(key=list(key), issues=issues)); continue
        metrics_by_key[key] = metrics
        trials.append({**dict(zip(IDENTITY, key)), 'allocation_id': row['allocation_id'],
            'record_id': row.get('record_id'), 'metrics': _flat_metrics(metrics), 'observed_window_spans': spans,
            **({'calibration_capacity': capacity} if phase == 'calibration' else {})})
    missing = sorted(set(expected) - set(index))
    allocations = sorted({str(row.get('allocation_id')) for row in index.values()})
    if len(allocations) != 1:
        errors.append('the complete supplementary matrix must share one allocation')
    pairs, averaged = [], []
    if phase == 'evaluation':
        for state in states:
            state_rows = [row for key, row in index.items() if key[:3] == state]
            for field in MATCH:
                if any(field not in row for row in state_rows) or len({_hash(row.get(field)) for row in state_rows}) != 1:
                    errors.append(f'within-state {field} changed: {_state(state)}')
            for schedule in plan['schedules'][_state(state)]:
                name = _schedule_name(schedule)
                repeat_pairs = []
                for rep in (0, 1):
                    nk, ok = (*state, name, rep, 'native'), (*state, name, rep, 'online')
                    if nk not in metrics_by_key or ok not in metrics_by_key:
                        continue
                    nr, other = index[nk], index[ok]
                    if nr.get('arm_order') != other.get('arm_order') or set(nr.get('arm_order', [])) != {'native', 'online'}:
                        errors.append(f'paired order differs: {nk[:-1]}')
                    pair = {**dict(zip(IDENTITY[:-1], nk[:-1])), 'allocation_id': nr['allocation_id'],
                        'pattern': schedule['pattern'], 'rho': schedule['rho'],
                        'contrasts': _contrast(_flat_metrics(metrics_by_key[nk]), _flat_metrics(metrics_by_key[ok]))}
                    pairs.append(pair); repeat_pairs.append(pair)
                if len(repeat_pairs) == 2:
                    orders = [index[(*state, name, rep, 'native')].get('arm_order') for rep in (0, 1)]
                    if orders[0] != list(reversed(orders[1])):
                        errors.append(f'repeat arm orders not reversed: {(*state, name)}')
                    means = {arm: {metric: statistics.mean(p['contrasts'][metric][arm] for p in repeat_pairs)
                        for metric in repeat_pairs[0]['contrasts']} for arm in ('native', 'online')}
                    averaged.append({**dict(zip(IDENTITY[:4], (*state, name))), 'pattern': schedule['pattern'],
                        'rho': schedule['rho'], 'technical_repetitions': 2,
                        'contrasts': _contrast(means['native'], means['online'])})
    return dict(analysis_schema_version=1, phase=phase, valid=not errors and not missing,
        expected_records=len(expected), observed_records=len(records), valid_records=len(metrics_by_key),
        missing_keys=[list(key) for key in missing], errors=errors, allocation_ids=allocations,
        plan_sha256=plan_hash, arrival_source_sha256=plan['arrival_source_sha256'],
        trial_summaries=trials, paired_cases=pairs, repeat_averaged_cases=averaged,
        interpretation='Supplementary descriptive comparison of fixed-batch layer jobs; no confidence interval, significance test or performance acceptance decision.',
        repeat_aggregation='Arithmetic mean of each arm\'s two run-level summaries; contrast is online minus native and ratio of these means. Quantiles are not pooled across repeats.',
        units='All *_ns fields are nanoseconds; throughput is layer jobs/second, not user requests/second.',
        coordinator_busy_definition='Sum of coordinator command-to-ACK intervals divided by the observed exposure; host-server occupancy, not GPU utilization.',
        limitations=['Two previously observed development source blocks and one allocation.',
            'Jobs reuse captured payloads and are temporally dependent; 256 jobs are not 256 independent workloads.',
            'Latency includes CPU command/ACK and per-job GPU synchronization in the new wrapper.',
            'At fixed offered rate, both arms completing by the horizon have equal horizon throughput by construction.',
            'Numerical and routing provenance requires the separate independent raw-input audit.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('plan', 'records', 'output'):
        parser.add_argument('--' + field, type=Path, required=True)
    args = parser.parse_args()
    opener = gzip.open if args.records.suffix == '.gz' else open
    with opener(args.records, 'rt') as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    result = analyze(json.loads(args.plan.read_text()), rows)
    result['analysis_file_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['records_file_sha256'] = hashlib.sha256(args.records.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in ('valid', 'expected_records', 'observed_records', 'valid_records')}))
    if not result['valid']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

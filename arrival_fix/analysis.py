"""Prespecified arrival-calendar inference conditional on fixed payload strata.

Jobs and reverse-order repeats are nested observations, never bootstrap units.
This module reads recorded clocks; it performs no GPU work or policy tuning.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

import numpy as np

from arrival_study.schedules import fcfs_metrics


KEY_FIELDS = ('arrival_block_id', 'batch_tokens', 'pattern', 'rho', 'repetition', 'arm')
MATCH_FIELDS = ('block_id', 'regime', 'arrival_seed', 'capture_sha256', 'replay_sha256',
    'source_prompt_ids', 'source_weights_sha256', 'source_weight_rank_sha256', 'source_map',
    'final_map', 'resolved_kernel_config', 'logical_routing_sha256', 'decision_trajectory_sha256',
    'arrival_clock_contract', 'warmup_raw_indices', 'conditioning_raw_indices',
    'measured_raw_indices', 'assignment_sha256')
ANALYSIS_DEFAULTS = dict(bootstrap_samples=100000, bootstrap_seed=471420,
    bootstrap='stratified_arrival_blocks', primary_ratio_limit=.95, guard_ratio_limit=1.02)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def passes_upper(upper, limit, *, strict=False):
    """A failed upper-bound requirement does not itself prove regression."""
    if upper is None or not math.isfinite(upper):
        return False
    return upper < limit if strict else upper <= limit


def stratified_indices(block_strata, *, samples=100000, seed=471420):
    """Draw two blocks with replacement within each of four fixed strata."""
    blocks = sorted(block_strata)
    strata = defaultdict(list)
    for i, block in enumerate(blocks):
        strata[tuple(block_strata[block])].append(i)
    if len(blocks) != 8 or len(strata) != 4 or any(len(indices) != 2 for indices in strata.values()):
        raise ValueError('eight distinct arrival blocks, two in each of four fixed payload strata, required')
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise ValueError('positive integer bootstrap sample count required')
    rng = np.random.default_rng(seed)
    pieces = []
    for stratum in sorted(strata):
        indices = np.asarray(strata[stratum], dtype=np.int64)
        pieces.append(indices[rng.integers(0, 2, size=(samples, 2))])
    return blocks, np.concatenate(pieces, axis=1)


def endpoint(block_logs, blocks, draws, *, name, limit, primary=False):
    if set(block_logs) != set(blocks):
        raise ValueError('every arrival block must contribute to every confirmatory endpoint')
    values = np.asarray([block_logs[block] for block in blocks], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('nonfinite block log ratio')
    boot = np.exp(values[draws].mean(axis=1))
    ratio = float(np.exp(values.mean()))
    upper = float(np.quantile(boot, .975, method='linear'))
    result = dict(name=name, ratio=ratio, arrival_blocks=8,
        block_log_ratios={block: float(value) for block, value in zip(blocks, values)},
        upper_confidence_bound=upper, ratio_limit=limit,
        passes=passes_upper(upper, limit, strict=primary),
        criterion='upper < limit' if primary else 'upper <= limit')
    if primary:
        result.update(ci95=[float(np.quantile(boot, .025, method='linear')), upper],
                      confidence='two-sided percentile 95%')
    else:
        result.update(one_sided_confidence_level=.975, guard_family_size=2,
                      confidence='one-sided percentile 97.5%; Bonferroni for two guards')
    result['status'] = 'pass' if result['passes'] else 'not_established'
    return result


def infer_endpoints(pairs, block_strata, **options):
    """Average repeat log ratios, then four B/pattern cells within each block."""
    config = {**ANALYSIS_DEFAULTS, **options}
    if config != ANALYSIS_DEFAULTS:
        raise ValueError('analysis settings differ from the prespecified final contract')
    blocks, draws = stratified_indices(block_strata, samples=config['bootstrap_samples'], seed=config['bootstrap_seed'])
    expected = set(itertools.product(blocks, (128, 1024), ('spread', 'cluster'), (.5, .85), (0, 1)))
    index = {}
    for pair in pairs:
        key = tuple(pair[field] for field in KEY_FIELDS[:-1])
        if key not in expected or key in index:
            raise ValueError('duplicate or unexpected paired statistical unit')
        index[key] = pair
    if set(index) != expected:
        raise ValueError('complete 128-pair matrix required for confirmatory analysis')
    endpoints = {}
    for name, rho, metric in (('high_load_mean_response', .85, 'response_mean_ns'),
                              ('low_load_mean_response', .5, 'response_mean_ns'),
                              ('high_load_p95_response', .85, 'response_p95_ns')):
        logs = {}
        for block in blocks:
            cell_logs = []
            for batch, pattern in itertools.product((128, 1024), ('spread', 'cluster')):
                replicate_logs = [index[block, batch, pattern, rho, rep]['log_ratios'][metric] for rep in (0, 1)]
                cell_logs.append(statistics.mean(replicate_logs))
            logs[block] = statistics.mean(cell_logs)
        primary = name == 'high_load_mean_response'
        endpoints[name] = endpoint(logs, blocks, draws, name=name,
            limit=config['primary_ratio_limit'] if primary else config['guard_ratio_limit'], primary=primary)
    return dict(primary=endpoints['high_load_mean_response'],
        guards=[endpoints['low_load_mean_response'], endpoints['high_load_p95_response']],
        acceptance='pass' if all(item['passes'] for item in endpoints.values()) else 'not_established',
        bootstrap={**config, 'rng': 'numpy.random.default_rng/PCG64', 'quantile_method': 'linear',
            'strata': {block: list(block_strata[block]) for block in blocks},
            'resampling': 'draw two arrival blocks with replacement within each fixed source/domain stratum, retain all nested cells and repeats',
            'draw_indices_sha256': hashlib.sha256(draws.tobytes()).hexdigest()})


def _flat_metrics(row, metrics):
    result = {f'{category}_{name}_ns': value for category, summary in metrics['summary_ns'].items()
              for name, value in summary.items()}
    for name in ('backlog_at_horizon', 'waiting_jobs_at_horizon', 'drain_ns',
                 'completed_by_horizon_per_second', 'completed_per_evaluation_second',
                 'required_observation_extent_ns', 'observation_overshoot_ns'):
        result[name] = metrics[name]
    elapsed_ns = metrics['evaluation_end_ns'] - row['epoch_monotonic_ns']
    result['observed_elapsed_seconds'] = elapsed_ns / 1e9
    result['completion_elapsed_seconds'] = (row['coordinator_finish_ns'][-1] - row['epoch_monotonic_ns']) / 1e9
    result['coordinator_server_busy_fraction'] = sum(metrics['vectors_ns']['service']) / elapsed_ns
    return result


def _window_spans(row):
    return [dict(end_step=end, jobs=16,
        scheduled_arrival_span_ns=row['arrival_offsets_ns'][end] - row['arrival_offsets_ns'][end-15],
        service_coverage_span_ns=row['coordinator_finish_ns'][end] - row['coordinator_start_ns'][end-15],
        completion_span_ns=row['coordinator_finish_ns'][end] - row['coordinator_finish_ns'][end-15])
        for end in (63, 127, 191, 255)]


def _guard_errors(row):
    guards = row.get('noop_guard_ranks')
    if row['arm'] == 'native':
        return [] if guards == [] else ['native_baseline_was_wrapped']
    if not isinstance(guards, list) or len(guards) != 2:
        return ['missing_two_rank_fast_path_counters']
    required = dict(mode='fast', diagnostic_timing=False, pending_pair=False,
        movement_calls=4, commit_calls=4, agreement_calls=4, local_unchanged_calls=4,
        unanimously_unchanged_calls=4, movement_skips=4, commit_skips=4,
        original_movement_calls=0, original_commit_calls=0, fallback_calls=0, diagnostics=[])
    return [f'fast_path_rank{rank}_{key}' for rank, guard in enumerate(guards)
            for key, value in required.items() if guard.get(key) != value]


def analyze(plan, records):
    if plan.get('phase') != 'final':
        raise ValueError('this analyzer is final-only; calibration and A/A pilot are excluded')
    if ('analysis_file_sha256' in plan and plan['analysis_file_sha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
        raise ValueError('analysis source differs from its frozen pin')
    plan_hash = canonical_hash({key: value for key, value in plan.items() if key != 'freeze'})
    if plan.get('freeze', {}).get('sha256') != plan_hash:
        raise ValueError('unfrozen or altered final plan')
    matrix = plan['matrix']
    if (matrix['batches'] != [128, 1024] or matrix['patterns'] != ['spread', 'cluster']
            or matrix['rhos'] != [.5, .85] or matrix['reps'] != 2 or matrix['arms'] != ['native', 'fast']):
        raise ValueError('final matrix differs from the prespecified contrast')
    block_info = {block['id']: block for block in matrix['arrival_blocks']}
    if len(block_info) != 8 or len(matrix['arrival_blocks']) != 8 or len({block['seed'] for block in block_info.values()}) != 8:
        raise ValueError('eight unique fresh arrival block IDs and seeds required')
    block_strata = {key: (block['block_id'], block['regime']) for key, block in block_info.items()}
    if Counter(block_strata.values()) != Counter({stratum: 2 for stratum in set(block_strata.values())}) or len(set(block_strata.values())) != 4:
        raise ValueError('arrival seeds must be balanced two per source/domain stratum')
    options = plan.get('analysis', ANALYSIS_DEFAULTS)
    if {**ANALYSIS_DEFAULTS, **options} != ANALYSIS_DEFAULTS:
        raise ValueError('final inference settings changed')
    expected = set(itertools.product(sorted(block_info), (128, 1024), ('spread', 'cluster'), (.5, .85), (0, 1), ('native', 'fast')))
    schedules = {}
    for block, batch in itertools.product(block_info, (128, 1024)):
        entries = plan['schedules'][f'{block}/B{batch}']
        if len(entries) != 4 or {(x['pattern'], x['rho']) for x in entries} != set(itertools.product(('spread', 'cluster'), (.5, .85))):
            raise ValueError('four frozen calendars required for each block/batch')
        for entry in entries:
            if entry.get('schedule_sha256') != canonical_hash({key: value for key, value in entry.items() if key != 'schedule_sha256'}):
                raise ValueError('calendar checksum changed')
            schedules[block, batch, entry['pattern'], entry['rho']] = entry
    errors, index, flat, trial_summaries = [], {}, {}, []
    actual_orders = defaultdict(list)
    for row in records:
        key = tuple(row.get(field) for field in KEY_FIELDS)
        if key not in expected or key in index:
            errors.append(dict(key=list(key), issues=['duplicate_or_unexpected_record'])); continue
        index[key] = row
        actual_orders[key[:-1]].append(row['arm'])
        issues = []
        for field, value in (('status', 'complete'), ('phase', 'final'), ('config_hash', plan_hash),
            ('arrival_fix_source_sha256', plan['arrival_fix_source_sha256']), ('correctness_pass', True),
            ('evidence_eligible', True), ('diagnostic_only', False), ('guard_symbols_restored', True),
            ('normal_completion_verified', True), ('resident_state_verified', True), ('rank_assignment_agreement', True),
            ('migration_bytes', 0), ('migration_sent_messages', 0), ('terminal_proposals_suppressed', 1), ('policy', 'SKIP')):
            if row.get(field) != value:
                issues.append(field)
        info = block_info[key[0]]
        for field, value in (('arrival_seed', info['seed']), ('block_id', info['block_id']), ('regime', info['regime'])):
            if row.get(field) != value:
                issues.append(field)
        if row.get('preflight_correctness', {}).get('correctness_pass') is not True:
            issues.append('preflight_correctness')
        if not row.get('allocation_id') or row.get('source_map') is None or row.get('source_map') != row.get('final_map'):
            issues.append('allocation_or_fixed_map')
        calibration_key = f"{info['block_id']}/{info['regime']}/B{key[1]}"
        calibration = plan.get('calibration', {}).get(calibration_key, {})
        if str(row.get('allocation_id')) != str(calibration.get('allocation_id')):
            issues.append('calibration_allocation_mismatch')
        hardware = plan['hardware']
        if (sorted(row.get('physical_gpu_uuids', [])) != sorted(hardware['gpu_uuids'])
                or str(row.get('hostname', '')).split('.')[0] != hardware['hostname']):
            issues.append('physical_hardware')
        clock = row.get('arrival_clock_contract', {})
        if (clock.get('monotonic') is not True or clock.get('adjustable') is not False
                or 'CLOCK_MONOTONIC' not in clock.get('implementation', '')):
            issues.append('clock_contract')
        issues.extend(_guard_errors(row))
        schedule = schedules[key[:4]]
        if (row.get('arrival_schedule') != schedule or row.get('arrival_offsets_ns') != schedule['arrival_offsets_ns']
                or row.get('horizon_ns') != schedule['horizon_ns']):
            issues.append('frozen_calendar')
        try:
            metrics = fcfs_metrics(row)
            if not metrics['complete'] or metrics['scheduled_jobs'] != 256:
                issues.append('incomplete_trial')
            if row.get('queue_metrics') != metrics:
                issues.append('stored_metrics_differ_from_raw_clocks')
            if metrics['complete'] and row.get('observed_window_spans') != _window_spans(row):
                issues.append('window_spans')
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            issues.append('raw_timing:' + str(exc))
        if issues:
            errors.append(dict(key=list(key), issues=issues)); continue
        values = _flat_metrics(row, metrics)
        flat[key] = values
        trial_summaries.append({**dict(zip(KEY_FIELDS, key)), 'block_id': info['block_id'], 'regime': info['regime'],
            'allocation_id': row['allocation_id'], 'metrics': values, 'observed_window_spans': row['observed_window_spans']})
    missing = sorted(expected - set(index))
    allocations = sorted({str(row.get('allocation_id')) for row in index.values()})
    if len(allocations) != 1:
        errors.append('one shared allocation required for the conditional inference')
    pairs = []
    for key in sorted({item[:-1] for item in expected}):
        nk, fk = (*key, 'native'), (*key, 'fast')
        if nk not in flat or fk not in flat:
            continue
        native, fast = index[nk], index[fk]
        for field in MATCH_FIELDS:
            if field not in native or field not in fast or native[field] != fast[field]:
                errors.append(dict(key=list(key), issues=['paired_' + field]))
        if (native.get('arm_order') != fast.get('arm_order') or native.get('arm_order') != actual_orders[key]
                or set(actual_orders[key]) != {'native', 'fast'}):
            errors.append(dict(key=list(key), issues=['actual_arm_order']))
        logs = {}
        for name in ('response_mean_ns', 'response_p95_ns', 'service_mean_ns'):
            if flat[nk][name] <= 0 or flat[fk][name] <= 0:
                errors.append(dict(key=list(key), issues=['nonpositive_endpoint'])); break
            logs[name] = math.log(flat[fk][name]) - math.log(flat[nk][name])
        pairs.append({**dict(zip(KEY_FIELDS[:-1], key)), 'block_id': native['block_id'], 'regime': native['regime'],
            'allocation_id': native['allocation_id'], 'native': flat[nk], 'fast': flat[fk], 'log_ratios': logs})
    for group in sorted({item[:4] for item in expected}):
        if actual_orders.get((*group, 0)) != list(reversed(actual_orders.get((*group, 1), []))):
            errors.append(dict(key=list(group), issues=['technical_repetitions_not_reverse_order']))
    valid = not errors and not missing
    result = dict(analysis_schema_version=1, phase='final', valid=valid,
        expected_records=256, observed_records=len(records), valid_records=len(flat), matched_pairs=len(pairs),
        missing_keys=[list(key) for key in missing], errors=errors, plan_sha256=plan_hash,
        arrival_fix_source_sha256=plan['arrival_fix_source_sha256'], allocation_ids=allocations,
        trial_summaries=trial_summaries, paired_cases=pairs, confirmatory=None, descriptive_cells=[])
    if valid:
        result['confirmatory'] = infer_endpoints(pairs, block_strata, **options)
        for batch, pattern, rho in itertools.product((128, 1024), ('spread', 'cluster'), (.5, .85)):
            selected = [pair for pair in pairs if (pair['batch_tokens'], pair['pattern'], pair['rho']) == (batch, pattern, rho)]
            result['descriptive_cells'].append(dict(batch_tokens=batch, pattern=pattern, rho=rho,
                paired_repetitions=len(selected), arrival_blocks=8,
                ratios={metric: math.exp(statistics.mean(pair['log_ratios'][metric] for pair in selected))
                        for metric in ('response_mean_ns', 'response_p95_ns', 'service_mean_ns')},
                native={metric: statistics.mean(pair['native'][metric] for pair in selected) for metric in selected[0]['native']},
                fast={metric: statistics.mean(pair['fast'][metric] for pair in selected) for metric in selected[0]['fast']},
                inference='descriptive only; no cell-level confidence interval or acceptance test'))
    result['scope'] = ('Arrival-distribution uncertainty conditional on four fixed, previously used payload strata, '
        'native calibration and one allocation. Two fresh arrival seeds per stratum; jobs and technical repeats are not independent blocks.')
    result['limitations'] = ['The percentile bootstrap has only two independent calendar blocks within each stratum.',
        'There are only 81 distinct stratum-count combinations; 100000 Monte Carlo draws do not add independent arrival evidence.',
        'No inference to unseen documents, production arrivals, full-model serving or a hardware population.',
        'All response, service, quantile and observed-elapsed metrics include their declared CPU/GPU coordination; observation duration includes horizon idle and drain.',
        'A nonpassing upper-bound guard means the margin is not established; it does not automatically prove a regression.',
        'Raw route/map/weight invariants and exact GPU UUIDs require the independent execution audit.']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'records', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    opener = gzip.open if args.records.suffix == '.gz' else open
    with opener(args.records, 'rt') as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    result = analyze(json.loads(args.plan.read_text()), records)
    result['analysis_file_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['records_file_sha256'] = hashlib.sha256(args.records.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in ('valid', 'expected_records', 'observed_records', 'matched_pairs')}))
    if not result['valid']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

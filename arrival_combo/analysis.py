"""Frozen three-arm inference: dispatcher value beyond an unchanged-map shortcut.

Reuses immutable raw-clock/statistical helpers; never edits prior experiments.
Calendar blocks are the uncertainty units, conditional on fixed payload strata.
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

from arrival_fix.analysis import (canonical_hash, stratified_indices, passes_upper,
    _flat_metrics, _window_spans, _guard_errors, MATCH_FIELDS as PREVIOUS_MATCH_FIELDS)
from arrival_study.schedules import fcfs_metrics

ARMS = ['native', 'shortcut', 'combined']
KEY_FIELDS = ('arrival_block_id', 'batch_tokens', 'pattern', 'rho', 'repetition', 'arm')
MATCH_FIELDS = tuple(field for field in PREVIOUS_MATCH_FIELDS if field != 'assignment_sha256')
CONTRASTS = {'combined_over_shortcut': ('shortcut', 'combined'),
             'combined_over_native': ('native', 'combined'),
             'shortcut_over_native': ('native', 'shortcut')}
ANALYSIS_DEFAULTS = dict(bootstrap_samples=100000, bootstrap_seed=471432,
    bootstrap='stratified_arrival_blocks', primary_ratio_limit=.95,
    guard_ratio_limit=1.02, guard_family_size=3, secondary_ratio_limit=.95)


def _order_key(block, batch, pattern, rho):
    return f'{block}/B{batch}/{pattern}/rho{rho}'


def _endpoint(logs, blocks, draws, *, name, contrast, limit, guard=False):
    values = np.asarray([logs[block] for block in blocks], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('nonfinite block log ratio')
    boot = np.exp(values[draws].mean(axis=1))
    quantile = 1-.05/3 if guard else .975
    upper = float(np.quantile(boot, quantile, method='linear'))
    result = dict(name=name, contrast=contrast, ratio=float(np.exp(values.mean())),
        block_log_ratios={block: float(value) for block, value in zip(blocks, values)},
        upper_confidence_bound=upper, ratio_limit=limit, arrival_blocks=8,
        passes=passes_upper(upper, limit, strict=not guard), criterion='upper <= limit' if guard else 'upper < limit')
    if guard:
        result.update(one_sided_confidence_level=quantile, guard_family_size=3,
            confidence='one-sided percentile 98.333333%; Bonferroni for three shortcut-relative guards')
    else:
        result.update(ci95=[float(np.quantile(boot,.025,method='linear')),upper], confidence='two-sided percentile 95%')
    result['status'] = 'pass' if result['passes'] else 'not_established'
    return result


def infer_endpoints(pairs, block_strata, **options):
    config = {**ANALYSIS_DEFAULTS, **options}
    if config != ANALYSIS_DEFAULTS:
        raise ValueError('analysis settings differ from the prespecified contract')
    blocks, draws = stratified_indices(block_strata, samples=config['bootstrap_samples'], seed=config['bootstrap_seed'])
    expected = set(itertools.product(blocks,(128,1024),('spread','cluster'),(.5,.85),(0,1),CONTRASTS))
    index = {}
    for pair in pairs:
        key = tuple(pair[field] for field in KEY_FIELDS[:-1]) + (pair['contrast'],)
        if key not in expected or key in index:
            raise ValueError('duplicate or unexpected paired statistical unit')
        index[key] = pair
    if set(index) != expected:
        raise ValueError('128 matched comparisons per contrast are required')
    def block_logs(contrast, rho, metric, batches=(128,1024)):
        return {block: statistics.mean(statistics.mean(
            index[block,batch,pattern,rho,rep,contrast]['log_ratios'][metric] for rep in (0,1))
            for batch,pattern in itertools.product(batches,('spread','cluster'))) for block in blocks}
    increment = 'combined_over_shortcut'
    primary = _endpoint(block_logs(increment,.85,'response_mean_ns'),blocks,draws,
        name='high_load_mean_response',contrast=increment,limit=.95)
    guards = [_endpoint(block_logs(increment,rho,metric,batches),blocks,draws,
        name=name,contrast=increment,limit=1.02,guard=True) for name,rho,metric,batches in (
            ('low_load_mean_response',.5,'response_mean_ns',(128,1024)),
            ('high_load_p95_response',.85,'response_p95_ns',(128,1024)),
            ('b128_high_load_mean_response',.85,'response_mean_ns',(128,)))]
    gate = primary['passes'] and all(guard['passes'] for guard in guards)
    baseline_logs = block_logs('combined_over_native',.85,'response_mean_ns')
    if gate:
        secondary = _endpoint(baseline_logs,blocks,draws,name='high_load_mean_response_baseline',
            contrast='combined_over_native',limit=.95)
        secondary.update(gate_passed=True,confirmatory=True)
    else:
        secondary = dict(name='high_load_mean_response_baseline',contrast='combined_over_native',
            ratio=math.exp(statistics.mean(baseline_logs.values())),arrival_blocks=8,
            block_log_ratios=baseline_logs,ci95=None,upper_confidence_bound=None,ratio_limit=.95,
            gate_passed=False,confirmatory=False,passes=None,status='not_tested_gate_closed',
            confidence='point estimate only; hierarchical confirmation gate did not pass')
    return dict(primary=primary,guards=guards,secondary_baseline=secondary,
        incremental_acceptance='pass' if gate else 'not_established',
        acceptance='pass' if gate and secondary['passes'] else 'not_established',
        bootstrap={**config,'rng':'numpy.random.default_rng/PCG64','quantile_method':'linear',
            'strata':{block:list(block_strata[block]) for block in blocks},
            'resampling':'two arrival blocks within each fixed source/domain stratum; all nested cells/repeats retained',
            'draw_indices_sha256':hashlib.sha256(draws.tobytes()).hexdigest()},
        hierarchy='Test combined/native high-load mean only after incremental primary and all three guards pass. Baseline gains cannot rescue failure to establish incremental dispatcher value.')


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
            or matrix['rhos'] != [.5, .85] or matrix['reps'] != 2 or matrix['arms'] != ARMS):
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
    expected = set(itertools.product(sorted(block_info), (128, 1024), ('spread', 'cluster'), (.5, .85), (0, 1), tuple(ARMS)))
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
            ('arrival_combo_schema_version', 1),
            ('arrival_combo_source_sha256', plan['arrival_combo_source_sha256']), ('correctness_pass', True),
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
        combined = row['arm'] == 'combined'
        for field, value in (('dispatch_mode', 'online' if combined else 'native'),
                ('noop_enabled', row['arm'] != 'native'), ('assignment_selection_cost_included', True),
                ('online_residency_metadata_verified', combined), ('mapper_calls_measured', 256 if combined else 0)):
            if row.get(field) != value:
                issues.append(field)
        if sorted(row.get('rank_gpu_uuids', [])) != sorted(hardware['gpu_uuids']):
            issues.append('actual_rank_gpu_identity')
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
    pairs, order_counts = [], {arm: [0,0,0] for arm in ARMS}
    for key in sorted({item[:-1] for item in expected}):
        keys = {arm: (*key,arm) for arm in ARMS}
        if any(value not in flat for value in keys.values()):
            continue
        rows = {arm:index[value] for arm,value in keys.items()}
        for field in MATCH_FIELDS:
            if any(field not in row for row in rows.values()) or len({canonical_hash(row.get(field)) for row in rows.values()}) != 1:
                errors.append(dict(key=list(key),issues=['triplet_'+field]))
        if (rows['native'].get('assignment_sha256') is None
                or rows['native']['assignment_sha256'] != rows['shortcut'].get('assignment_sha256')):
            errors.append(dict(key=list(key),issues=['native_shortcut_assignment']))
        declared = plan['arm_orders'].get(_order_key(*key[:4]), [None,None])[key[4]]
        if (declared != actual_orders[key] or sorted(declared or []) != sorted(ARMS)
                or any(row.get('arm_order') != declared for row in rows.values())):
            errors.append(dict(key=list(key),issues=['frozen_actual_arm_order']))
        else:
            for position,arm in enumerate(declared): order_counts[arm][position] += 1
        for contrast,(control,treatment) in CONTRASTS.items():
            control_metrics,treatment_metrics=flat[keys[control]],flat[keys[treatment]]
            logs={}
            for name in ('response_mean_ns','response_p95_ns','service_mean_ns'):
                if control_metrics[name] <= 0 or treatment_metrics[name] <= 0:
                    errors.append(dict(key=list(key),issues=['nonpositive_endpoint']));break
                logs[name]=math.log(treatment_metrics[name])-math.log(control_metrics[name])
            pairs.append({**dict(zip(KEY_FIELDS[:-1],key)), 'contrast':contrast,
                'control_arm':control,'treatment_arm':treatment,
                'block_id':rows[control]['block_id'],'regime':rows[control]['regime'],
                'allocation_id':rows[control]['allocation_id'], 'control':control_metrics,
                'treatment':treatment_metrics,'log_ratios':logs})
    for group in sorted({item[:4] for item in expected}):
        if actual_orders.get((*group,0)) != list(reversed(actual_orders.get((*group,1),[]))):
            errors.append(dict(key=list(group),issues=['technical_repeats_not_reverse_order']))
    valid=not errors and not missing
    result=dict(analysis_schema_version=1,phase='final',valid=valid,expected_records=384,
        observed_records=len(records),valid_records=len(flat),matched_pairs=len(pairs),
        pairs_per_contrast=dict(Counter(pair['contrast'] for pair in pairs)),
        matched_triplets=len(pairs)//3,missing_keys=[list(key) for key in missing],errors=errors,
        plan_sha256=plan_hash,arrival_combo_source_sha256=plan['arrival_combo_source_sha256'],
        allocation_ids=allocations,trial_summaries=trial_summaries,paired_cases=pairs,
        arm_position_counts=order_counts,confirmatory=None,descriptive_cells=[])
    if valid:
        result['confirmatory']=infer_endpoints(pairs,block_strata,**options)
        for batch,pattern,rho in itertools.product((128,1024),('spread','cluster'),(.5,.85)):
            selected=[pair for pair in pairs if (pair['batch_tokens'],pair['pattern'],pair['rho'])==(batch,pattern,rho)]
            arms={arm:{metric:statistics.mean(flat[key][metric] for key in flat if
                key[1:4]==(batch,pattern,rho) and key[-1]==arm) for metric in next(iter(flat.values()))} for arm in ARMS}
            contrasts={}
            for contrast in CONTRASTS:
                members=[pair for pair in selected if pair['contrast']==contrast]
                contrasts[contrast]=dict(paired_repetitions=len(members),
                    ratios={metric:math.exp(statistics.mean(pair['log_ratios'][metric] for pair in members))
                        for metric in ('response_mean_ns','response_p95_ns','service_mean_ns')})
            result['descriptive_cells'].append(dict(batch_tokens=batch,pattern=pattern,rho=rho,
                arrival_blocks=8,arms=arms,contrasts=contrasts,
                inference='descriptive only; no cell-level confidence interval or acceptance test'))
    result['scope']='Arrival-distribution uncertainty conditional on four reused payload strata, native calibration and one allocation; two calendar blocks per stratum.'
    result['limitations']=[
        'Only 81 stratum-count combinations exist; 100000 bootstrap draws do not add independent evidence.',
        'Reversed triplets balance pairwise precedence; the middle arm stays middle within a case. Position counts are only near-balanced and are reported exactly.',
        'The 2% guards are relative to shortcut on their declared mixtures; they do not establish non-regression against native in every cell.',
        'Jobs and technical repeats are nested observations, not independent arrival blocks. No hardware-population, unseen-document or full-model-serving inference.',
        'Physical assignments may differ for combined; native and shortcut assignments must agree. Independent per-arm routing/weight audits remain required.',
        'Elapsed and service metrics include their declared coordination; observation elapsed includes horizon idle and drain.',
        'A nonpassing bound means the requirement is not established, not automatically proven regression.']
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','records','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();opener=gzip.open if args.records.suffix=='.gz' else open
    with opener(args.records,'rt') as handle:records=[json.loads(line) for line in handle if line.strip()]
    result=analyze(json.loads(args.plan.read_text()),records)
    result['analysis_file_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['records_file_sha256']=hashlib.sha256(args.records.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:result[key] for key in ('valid','expected_records','observed_records','matched_pairs')}))
    if not result['valid']:raise SystemExit(1)


if __name__=='__main__':main()

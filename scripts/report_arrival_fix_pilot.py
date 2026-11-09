#!/usr/bin/env python3
"""Independent descriptive pilot accounting; no timing eligibility criterion."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

import numpy as np

CASE = ('batch_tokens', 'pattern', 'rho', 'repetition')
ARMS = ('native', 'native_alias', 'fast')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_jsonl(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def raw_times(row, n):
    """Direct timestamp arithmetic, independent of the frozen analyzer."""
    origin = row['epoch_monotonic_ns']
    names = ('arrival_offsets_ns', 'coordinator_admit_ns', 'coordinator_start_ns', 'coordinator_finish_ns')
    arrays = []
    for name in names:
        values = row[name]
        if len(values) != n or any(type(value) is not int for value in values):
            raise ValueError(f'incomplete integer clock array: {name}')
        arrays.append(np.asarray(values, dtype=np.int64))
    offsets, admits, starts, finishes = arrays
    arrivals = offsets + origin
    if (offsets[0] != 0 or np.any(np.diff(offsets) < 0) or offsets[-1] >= row['horizon_ns']
            or np.any(admits < arrivals) or np.any(starts < admits) or np.any(finishes <= starts)
            or np.any(admits[1:] < finishes[:-1])):
        raise ValueError('invalid immutable-calendar FCFS clock ordering')
    end = row['observation_end_ns']
    if end < max(origin + row['horizon_ns'], int(finishes[-1])):
        raise ValueError('observation does not cover horizon and full drain')
    service = finishes - starts
    response = finishes - arrivals
    boundary = np.arange(63, n, 64)
    ordinary = np.asarray([step for step in range(n) if step not in boundary])
    return dict(mean_response_ms=float(response.mean())/1e6,
        p95_response_ms=float(np.quantile(response, .95, method='linear'))/1e6,
        mean_queue_ms=float((starts-arrivals).mean())/1e6,
        mean_service_ms=float(service.mean())/1e6,
        total_charged_service_ms=int(service.sum())/1e6,
        ordinary_mean_service_ms=float(service[ordinary].mean())/1e6,
        boundary_mean_service_ms=float(service[boundary].mean())/1e6,
        boundary_service_ms={str(i): int(service[i])/1e6 for i in boundary},
        terminal_service_ms=int(service[-1])/1e6,
        observed_elapsed_seconds=(end-origin)/1e9,
        last_completion_elapsed_seconds=int(finishes[-1]-origin)/1e9,
        drain_ms=max(0, int(finishes[-1]-origin-row['horizon_ns']))/1e6,
        per_step_service_ms=(service/1e6).tolist())


def analyze(plan, records, diagnostic_records, proof):
    plan_hash = digest({key: value for key, value in plan.items() if key != 'freeze'})
    if plan.get('phase') != 'pilot' or plan.get('freeze', {}).get('sha256') != plan_hash:
        raise ValueError('a frozen pilot plan is required')
    matrix = plan['matrix']
    if (matrix['arms'] != list(ARMS) or matrix['batches'] != [128, 1024]
            or matrix['patterns'] != ['constant', 'burst8'] or matrix['rhos'] != [.85] or matrix['reps'] != 2):
        raise ValueError('helper supports the declared four-case, three-arm pilot only')
    expected = set(itertools.product((128, 1024), ('constant', 'burst8'), (.85,), (0, 1), ARMS))
    index, trials, issues = {}, [], []
    allocations = set()
    for row in records:
        key = tuple(row.get(name) for name in (*CASE, 'arm'))
        if key not in expected or key in index:
            raise ValueError('unexpected or duplicate pilot record')
        index[key] = row
        allocations.add(str(row.get('allocation_id')))
        for field, value in (('phase', 'pilot'), ('status', 'complete'), ('evidence_eligible', False),
                ('diagnostic_only', False), ('correctness_pass', True), ('guard_symbols_restored', True),
                ('config_hash', plan_hash), ('arrival_fix_source_sha256', plan['arrival_fix_source_sha256'])):
            if row.get(field) != value:
                issues.append(dict(key=list(key), field=field))
        clocks = raw_times(row, 256)
        trials.append({**dict(zip((*CASE, 'arm'), key)), 'allocation_id': row.get('allocation_id'),
                       'arm_order': row.get('arm_order'), **clocks})
    if set(index) != expected or len(records) != 24:
        raise ValueError('all 24 pilot trials are required; no survivor-only timing report')
    if len(allocations) != 1:
        issues.append('pilot records cross allocations')
    timed = {tuple(row[name] for name in (*CASE, 'arm')): row for row in trials}
    pairs, cases = [], []
    for batch, pattern, rho, rep in sorted({key[:-1] for key in expected}):
        selected = {arm: timed[batch, pattern, rho, rep, arm] for arm in ARMS}
        raw = [index[batch, pattern, rho, rep, arm] for arm in ARMS]
        for field in ('arrival_schedule', 'capture_sha256', 'replay_sha256', 'source_weights_sha256',
                      'assignment_sha256', 'decision_trajectory_sha256', 'source_map', 'final_map'):
            if len({digest(row.get(field)) for row in raw}) != 1 or any(field not in row for row in raw):
                issues.append(dict(case=[batch, pattern, rho, rep], field='paired_'+field))
        result = dict(batch_tokens=batch, pattern=pattern, rho=rho, repetition=rep,
            arms={arm: {key: value for key, value in row.items() if key != 'per_step_service_ms'} for arm, row in selected.items()})
        for metric in ('mean_response_ms', 'p95_response_ms', 'mean_service_ms', 'total_charged_service_ms', 'observed_elapsed_seconds'):
            result['aa_'+metric+'_ratio'] = selected['native_alias'][metric]/selected['native'][metric]
            result['fast_'+metric+'_ratio'] = selected['fast'][metric]/selected['native'][metric]
        pairs.append(result)
    for batch, pattern in itertools.product((128, 1024), ('constant', 'burst8')):
        selected = [pair for pair in pairs if (pair['batch_tokens'], pair['pattern']) == (batch, pattern)]
        case = dict(batch_tokens=batch, pattern=pattern, rho=.85, technical_repetitions=2)
        for key in selected[0]:
            if key.endswith('_ratio'):
                case[key] = math.exp(statistics.mean(math.log(pair[key]) for pair in selected))
        cases.append(case)
    expected_diag = set(itertools.product((128, 1024), ('native_observe', 'fast')))
    seen_diag, diagnostics, diagnostic_trials = set(), [], []
    for row in diagnostic_records:
        key = (row['batch_tokens'], row['arm'])
        if key not in expected_diag or key in seen_diag:
            raise ValueError('unexpected or duplicate stage-diagnostic record')
        seen_diag.add(key)
        for field, value in (('phase', 'diagnostic'), ('evidence_eligible', False), ('diagnostic_only', True),
                ('correctness_pass', True), ('guard_symbols_restored', True), ('config_hash', plan_hash)):
            if row.get(field) != value:
                issues.append(dict(diagnostic_key=list(key), field=field))
        clocks = raw_times(row, 128)
        diagnostic_trials.append(dict(batch_tokens=key[0], arm=key[1], **clocks))
        ranks = row['noop_guard_ranks']
        if len(ranks) != 2:
            raise ValueError('both rank diagnostic logs required')
        for rank, guard in enumerate(ranks):
            stage_index = {(stage['pair_index'], stage['stage']): stage for stage in guard['diagnostics']}
            if set(stage_index) != {(pair, stage) for pair in (1, 2) for stage in ('movement', 'commit')} or len(guard['diagnostics']) != 4:
                raise ValueError('exactly two movement/commit stage pairs required per diagnostic rank')
            for pair in (1, 2):
                movement, commit = stage_index[pair, 'movement'], stage_index[pair, 'commit']
                report = dict(batch_tokens=key[0], arm=key[1], rank=rank, pair_index=pair, step=64*pair-1,
                    movement_skipped=movement['skipped'], commit_skipped=commit['skipped'],
                    coordinator_service_ms=clocks['per_step_service_ms'][64*pair-1])
                for output, stage, source in (
                    ('equality_wall_ms', movement, 'equality_host_wall_seconds'),
                    ('agreement_wall_ms', movement, 'agreement_host_wall_seconds'),
                    ('original_movement_wall_ms', movement, 'original_function_host_wall_seconds'),
                    ('original_commit_wall_ms', commit, 'original_function_host_wall_seconds'),
                    ('movement_wrapper_wall_ms', movement, 'wrapper_host_wall_seconds'),
                    ('commit_wrapper_wall_ms', commit, 'wrapper_host_wall_seconds')):
                    value = stage[source]
                    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                        raise ValueError('invalid diagnostic function-wall timing')
                    report[output] = value*1000
                diagnostics.append(report)
    if seen_diag != expected_diag or len(diagnostic_records) != 4:
        raise ValueError('all four stage-diagnostic trials are required')
    proof_issues = []
    for name in ('passed', 'source_restore_verified', 'native_symbols_restored', 'no_pending_async_work'):
        if proof.get(name) is not True: proof_issues.append(name)
    if (proof.get('evidence_eligible') is not False or proof.get('diagnostic_only') is not True
            or proof.get('source_weights_sha256') != proof.get('restored_weights_sha256')):
        proof_issues.append('proof_scope_or_weight_restoration')
    if proof.get('proof_source_sha256') != plan['arrival_fix_source_files']['arrival_fix/proof.py']:
        proof_issues.append('proof_source_pin')
    expected_steps = [('unchanged_before', True), ('changed_swap_0_80', False), ('restore_original', False), ('unchanged_after_restore', True)]
    if [(step.get('name'), step.get('expected_skip')) for step in proof.get('steps', [])] != expected_steps:
        proof_issues.append('proof_steps')
    for step in proof.get('steps', []):
        if not step.get('correctness_pass') or not step.get('exact_checkpoint_weights_verified') or not step.get('inverse_map_and_replica_count_verified'):
            proof_issues.append(step.get('name', 'unknown')+'_correctness')
        if (step['expected_skip'] and (step['actual_transfer_bytes'] or step['actual_transfer_messages'])) or (
                not step['expected_skip'] and (step['actual_transfer_bytes'] <= 0 or step['actual_transfer_messages'] <= 0)):
            proof_issues.append(step['name']+'_transport')
    if sum(step['actual_transfer_bytes'] for step in proof.get('steps', [])) != proof.get('total_transfer_bytes'):
        proof_issues.append('total_transfer_bytes')
    return dict(pilot_reporting_schema_version=1, complete_matrix=True,
        observed_pilot_records=24, observed_diagnostic_records=4, allocation_ids=sorted(allocations),
        metadata_issues=issues, proof_issues=proof_issues, metadata_checks_pass=not issues and not proof_issues,
        plan_sha256=plan_hash, arrival_fix_source_sha256=plan['arrival_fix_source_sha256'],
        timing_eligibility_gate=None, performance_selection=False,
        trials=trials, paired_repetitions=pairs, descriptive_cases=cases,
        diagnostic_trials=diagnostic_trials, diagnostic_stage_pairs=diagnostics, gpu_proof=proof,
        limits=['A/A timing ratios are descriptive; no percent threshold determines eligibility or selection.',
            'All 24 pilot and four stage-diagnostic outcomes are retained. No timing outliers are removed.',
            'Stage measurements are CPU-function wall intervals including natural blocking, not exclusive CPU or GPU time.',
            'Do not sum rank timings as serial work; ranks execute concurrently. native_observe includes diagnostic checks absent from the unwrapped baseline.',
            'The untimed proof tests two unchanged skips and two actual changed-map transports; it is not performance evidence.'])


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer=csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def render(result, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/'pilot-description.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    write_csv(output_dir/'pilot-four-cases.csv', result['descriptive_cases'])
    write_csv(output_dir/'pilot-stage-diagnostics.csv', result['diagnostic_stage_pairs'])
    lines=['# Descriptive unchanged-map pilot', '',
        'All 24 pilot trials and four diagnostic trials are retained. A/A ratios are descriptive: there is no timing eligibility threshold, confidence interval, or performance-based selection. N/A/F means native/native alias/fast.', '',
        '| B | Arrivals | Repeat | Mean response N/A/F (ms) | A/A response ratio | Fast/native response ratio | Mean service N/A/F (ms) | Total charged service N/A/F (ms) |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for pair in result['paired_repetitions']:
        fmt=lambda metric:' / '.join(f"{pair['arms'][arm][metric]:.3f}" for arm in ARMS)
        lines.append(f"| {pair['batch_tokens']} | {pair['pattern']} | {pair['repetition']} | {fmt('mean_response_ms')} | {pair['aa_mean_response_ms_ratio']:.4f} | {pair['fast_mean_response_ms_ratio']:.4f} | {fmt('mean_service_ms')} | {fmt('total_charged_service_ms')} |")
    lines += ['', '| B | Arm | Ordinary service (ms) | Boundary service (ms) | Terminal service (ms) |', '| --- | --- | --- | --- | --- |']
    for batch, arm in itertools.product((128, 1024), ARMS):
        rows=[row for row in result['trials'] if row['batch_tokens']==batch and row['arm']==arm]
        values=[statistics.mean(row[key] for row in rows) for key in ('ordinary_mean_service_ms','boundary_mean_service_ms','terminal_service_ms')]
        lines.append(f'| {batch} | {arm} | '+' | '.join(f'{x:.3f}' for x in values)+' |')
    lines += ['', 'Ordinary service excludes jobs 63/127/191/255. Boundary service averages those four jobs. These are charged command-to-ACK intervals, not isolated GPU kernels.', '',
        '| B | Diagnostic arm | Rank | Equality (ms) | Agreement (ms) | Original movement (ms) | Original commit (ms) | Movement wrapper (ms) | Commit wrapper (ms) |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    fields=('equality_wall_ms','agreement_wall_ms','original_movement_wall_ms','original_commit_wall_ms','movement_wrapper_wall_ms','commit_wrapper_wall_ms')
    for batch, arm, rank in itertools.product((128, 1024), ('native_observe','fast'), (0,1)):
        rows=[row for row in result['diagnostic_stage_pairs'] if (row['batch_tokens'],row['arm'],row['rank'])==(batch,arm,rank)]
        lines.append(f'| {batch} | {arm} | {rank} | '+' | '.join(f'{statistics.mean(row[key] for row in rows):.3f}' for key in fields)+' |')
    lines += ['', 'Diagnostic entries average the two measured boundaries in each separate 128-job diagnostic. The CSV retains both stage pairs. Function-wall intervals include natural blocking; rank intervals overlap and must not be summed as serial CPU/GPU time. The observed-native wrapper is diagnostic and is not the unwrapped timing baseline.', '',
        '| Untimed proof step | Expected skip | Actual sender bytes | Actual messages | Numerical check |', '| --- | --- | --- | --- | --- |']
    for step in result['gpu_proof'].get('steps',[]):
        lines.append(f"| {step['name']} | {step['expected_skip']} | {step['actual_transfer_bytes']} | {step['actual_transfer_messages']} | {step['correctness_pass']} |")
    lines += ['', 'Metadata/proof checks: '+str(result['metadata_checks_pass'])+'. This is a structural check, not a timing gate. Raw records and source/proof checksums remain the evidence; this report does not replace the complete independent route/weight audit.']
    (output_dir/'pilot-description.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','records','diagnostics','proof','output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    args=parser.parse_args()
    result=analyze(json.loads(args.plan.read_text()),read_jsonl(args.records),read_jsonl(args.diagnostics),json.loads(args.proof.read_text()))
    result['file_sha256']={name:hashlib.sha256(getattr(args,name).read_bytes()).hexdigest() for name in ('plan','records','diagnostics','proof')}
    result['helper_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    render(result,args.output_dir)
    print(json.dumps({key:result[key] for key in ('observed_pilot_records','observed_diagnostic_records','metadata_checks_pass','timing_eligibility_gate')}))
    if not result['metadata_checks_pass']: raise SystemExit(1)


if __name__=='__main__': main()

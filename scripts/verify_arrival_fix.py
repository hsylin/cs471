#!/usr/bin/env python3
"""Independent CPU execution audit of the unchanged-map arrival experiment.

Imports no arrival_fix, arrival_study, or other production experiment module.
Earlier independent auditors supply timestamp arithmetic and the raw-route oracle.
The new calendar, order, source, state, guard, and proof contracts are checked here.
This checks recorded evidence; it does not reproduce GPU arithmetic or significance.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import random

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STATE = ('block_id', 'regime', 'batch_tokens')
KEY = ('arrival_block_id', 'batch_tokens', 'pattern', 'rho', 'repetition', 'arm')
UUIDS = ['GPU-204281f9-9c87-3c25-adfb-9edd6c972ec3',
         'GPU-7d5c9009-305b-9341-957f-1deb96611ed5']
SOURCE_NAMES = ['arrival_fix/'+x for x in
                ('__init__.py', 'common.py', 'calendars.py', 'runner.py', 'noop.py', 'proof.py')]
SOURCE_NAMES += ['arrival_study/'+x for x in ('__init__.py', 'common.py', 'schedules.py', 'runner.py')]
SOURCE_NAMES += ['scripts/freeze_arrival_fix.py', 'scripts/run-arrival-fix-kuma.sh']


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


old = load_module(ROOT/'scripts/verify_instructor_arrivals.py', 'independent_arrival_timing')
require, compare, digest = old.require, old.compare, old.digest
file_hash, array_hash, canonical = old.file_hash, old.array_hash, old.canonical


def expected_blocks():
    states = list(itertools.product(('opt-pilot-01', 'opt-pilot-02'), ('domain0', 'domain1')))
    return [dict(id=f'arrival-fresh-{i+1:02d}', seed=2026110901+i,
                 block_id=state[0], regime=state[1]) for i, state in enumerate(states*2)]


def verify_weights(weights, seed, pattern):
    """Regenerate frozen RNG values as well as checking their archived bytes."""
    require(pattern in ('spread', 'cluster'), 'Gamma pattern')
    shape = {'spread': 4., 'cluster': .25}[pattern]
    rng_seed = seed*2 + int(pattern == 'cluster')
    rng = random.Random(rng_seed)
    values = [rng.gammavariate(shape, 1.).hex() for _ in range(256)]
    result = dict(seed=seed, rng_seed=rng_seed, pattern=pattern, shape=shape,
        generator='Python Random.gammavariate; exact binary64 draws archived as hex',
        gamma_draws_hex=values, n_jobs=256)
    result['weights_sha256'] = digest(result)
    compare(weights, result, 'Prespecified gamma draw')
    require(set(weights) == set(result), 'Gamma weight schema')
    return weights


def rational_round(value):
    return old.nearest_even(value.numerator, value.denominator)


def rebuild_calendar(weights, rho, elapsed):
    """Use exact integer/rational arithmetic, independently of calendar code."""
    require(rho in (.5, .85) and type(elapsed) is int and elapsed > 0, 'Calendar calibration/load')
    require(weights['weights_sha256'] == digest({k:v for k,v in weights.items() if k != 'weights_sha256'}),
            'Gamma archive checksum')
    require(len(weights['gamma_draws_hex']) == weights['n_jobs'] == 256, 'Gamma job count')
    values = []
    for encoded in weights['gamma_draws_hex']:
        value = float.fromhex(encoded)
        require(math.isfinite(value) and value > 0 and value.hex() == encoded, 'Canonical finite positive gamma weight')
        values.append(Fraction(*value.as_integer_ratio()))
    extent = Fraction(elapsed)*Fraction(1)/Fraction(str(rho))
    horizon = rational_round(extent)
    require(horizon > 1, 'Representable horizon')
    total, running, offsets = sum(values), Fraction(0), []
    for value in values:
        offsets.append(min(rational_round(extent*running/total), horizon-1))
        running += value
    result = dict(schedule_schema_version=2, pattern=weights['pattern'], rho=float(rho),
        seed=weights['seed'], shape=weights['shape'], weights_sha256=weights['weights_sha256'],
        calibration_jobs=256, calibration_elapsed_ns=elapsed, n_jobs=256,
        horizon_ns=horizon, arrival_offsets_ns=offsets,
        arrival_rate_jobs_per_second=256*1e9/horizon,
        rounding='exact archived binary64 weights; nearest nanosecond ties-to-even; releases capped at H-1ns',
        arrival_unit='synthetic preassembled B-token layer job',
        process='fixed-count normalized gamma-gap calendar, not an unconditioned renewal process')
    result['schedule_sha256'] = digest(result)
    return result


def plan_identity(plan, repo):
    require(plan['phase'] in ('pilot', 'calibration', 'final'), 'Supported phase')
    sha = digest({k:v for k,v in plan.items() if k != 'freeze'})
    compare(plan['freeze'], dict(algorithm='sha256', sha256=sha), 'Plan freeze')
    require(plan['arrival_fix_schema_version'] == 1, 'Plan schema')
    files = {name:file_hash(repo/name) for name in SOURCE_NAMES}
    compare(plan['arrival_fix_source_files'], files, 'Pinned execution sources')
    require(set(plan['arrival_fix_source_files']) == set(files), 'Exact execution source inventory')
    require(plan['arrival_fix_source_sha256'] == digest(files), 'Execution fingerprint')
    for field, name in (('protocol_file_sha256', 'docs/arrival-response-fix-protocol.md'),
                        ('analysis_file_sha256', 'arrival_fix/analysis.py')):
        require(plan[field] == file_hash(repo/name), 'Pinned '+name)
    compare(plan['hardware'], dict(hostname='kh014', gpu_uuids=UUIDS, world_size=2, interconnect='NV6'), 'Hardware plan')
    compare(plan['jobs_per_trial'], 256, 'Job budget')
    compare(plan['order_seed'], 471421, 'Order seed')
    compare(plan['analysis'], dict(bootstrap_samples=100000, bootstrap_seed=471420,
        bootstrap='stratified_arrival_blocks', primary_ratio_limit=.95, guard_ratio_limit=1.02), 'Analysis settings')
    base = json.loads((repo/plan['base_manifest']['path']).read_text())
    base_sha = digest({k:v for k,v in base.items() if k != 'freeze'})
    require(base_sha == base['freeze']['sha256'] == plan['base_manifest']['sha256'], 'Frozen underlying resident manifest')
    require(base['phase'] == 'validation' and base['split'] == 'pilot', 'Reused development payload identity')
    for name, sha0 in base['execution_source_sha256'].items():
        require(file_hash(repo/name) == sha0, 'Pinned resident source '+name)
    require(digest(base['execution_source_sha256']) == base['environment']['resident_dispatch_sha256'], 'Resident source fingerprint')
    matrix = dict(arrival_blocks=expected_blocks(), batches=[128, 1024], patterns=['spread', 'cluster'],
                  rhos=[.5, .85], reps=2, arms=['native', 'fast'])
    if plan['phase'] == 'pilot':
        matrix.update(arrival_blocks=[dict(id='pilot-old-calendar', seed=0,
                                          block_id='opt-pilot-01', regime='domain1')],
                      patterns=['constant', 'burst8'], rhos=[.85], arms=['native', 'native_alias', 'fast'])
        path = repo/'results/instructor/arrivals-v1/evaluation-plan.json'
        require(plan['pilot_calendar_source_sha256'] == file_hash(path), 'Old pilot calendar provenance')
        previous = json.loads(path.read_text())
        require(previous['freeze']['sha256'] == digest({k:v for k,v in previous.items() if k != 'freeze'}), 'Archived old calendar plan')
        expected = {f'pilot-old-calendar/B{B}': [s for s in previous['schedules'][f'opt-pilot-01/domain1/B{B}'] if s['rho'] == .85]
                    for B in (128, 1024)}
        compare(plan['schedules'], expected, 'Exact previous pilot calendars')
        for schedules in expected.values():
            for schedule in schedules:
                old.calendar(schedule)
        require('gamma_weights' not in plan, 'Pilot must not consume fresh gamma calendars')
    else:
        require(set(plan['gamma_weights']) == {b['id'] for b in matrix['arrival_blocks']}, 'Fresh draw inventory')
        for block in matrix['arrival_blocks']:
            require(set(plan['gamma_weights'][block['id']]) == {'spread', 'cluster'}, 'Fresh patterns')
            for pattern in ('spread', 'cluster'):
                verify_weights(plan['gamma_weights'][block['id']][pattern], block['seed'], pattern)
    compare(plan['matrix'], matrix, 'Exact bounded matrix and seed assignment')
    require(set(plan['matrix']) == set(matrix), 'Unknown matrix dimension')
    return base, sha, base_sha


def execution_inventory(plan, diagnostic=False):
    """Reconstruct the full ordered stream, not just a set of final identities."""
    blocks = plan['matrix']['arrival_blocks']
    states = sorted({(b['block_id'], b['regime'], B) for b in blocks for B in (128, 1024)})
    random.Random(plan['order_seed']).shuffle(states)
    result = []
    for si, state in enumerate(states):
        if diagnostic or plan['phase'] == 'calibration':
            arms = ['native_observe', 'fast'] if diagnostic else ['native']
            for arm in arms:
                result.append((state, ('native-calibration', state[2], 'saturated', None, 0, arm), arms, None, None))
            continue
        subset = [b for b in blocks if (b['block_id'], b['regime']) == state[:2]]
        random.Random(plan['order_seed']+si+1).shuffle(subset)
        for bi, block in enumerate(subset):
            schedules = plan['schedules'][f"{block['id']}/B{state[2]}"]
            require([(s['rho'], s['pattern']) for s in schedules] ==
                    list(itertools.product(plan['matrix']['rhos'], plan['matrix']['patterns'])), 'Declared calendar order')
            for ci, schedule in enumerate(schedules):
                for rep in (0, 1):
                    arms = list(plan['matrix']['arms'])
                    if (si+bi+ci+rep) % 2:
                        arms = arms[::-1]
                    for arm in arms:
                        key = (block['id'], state[2], schedule['pattern'], schedule['rho'], rep, arm)
                        result.append((state, key, arms, schedule, block))
    return result


def hardware(row):
    require(row['hostname'].split('.')[0] == 'kh014', 'Recorded physical host')
    require(sorted(row['physical_gpu_uuids']) == sorted(UUIDS), 'Recorded physical GPU pair')
    require(len(row['rank_gpu_uuids']) == 2 and sorted(row['rank_gpu_uuids']) == sorted(UUIDS), 'Actual CUDA rank UUIDs')
    compare(row, dict(world_size=2, dtype='bfloat16'), 'Rank/dtype contract')
    clock = row['arrival_clock_contract']
    require(clock['monotonic'] is True and clock['adjustable'] is False
            and 'CLOCK_MONOTONIC' in clock['implementation'], 'Monotonic clock contract')


def guard_counts(row, diagnostic=False):
    guards = row['noop_guard_ranks']
    if row['arm'] in ('native', 'native_alias'):
        require(guards == [], 'Unwrapped native/alias baseline')
        return
    require(row['arm'] in ('fast', 'native_observe') and len(guards) == 2, 'Two rank guards')
    n = row['n_jobs']//64
    observe = row['arm'] == 'native_observe'
    require(not observe or diagnostic, 'Observe wrapper excluded from primary timing')
    expected = dict(mode='observe' if observe else 'fast', diagnostic_timing=diagnostic,
        pending_pair=False, movement_calls=n, commit_calls=n, agreement_calls=n,
        local_unchanged_calls=n, unanimously_unchanged_calls=n,
        movement_skips=0 if observe else n, commit_skips=0 if observe else n,
        original_movement_calls=n if observe else 0, original_commit_calls=n if observe else 0,
        fallback_calls=0)
    for guard in guards:
        compare(guard, expected, 'Fast-path paired operation counters')
        if not diagnostic:
            require(guard['diagnostics'] == [], 'Primary measurements must not collect stage timing')
        else:
            require([(d['pair_index'], d['stage']) for d in guard['diagnostics']] ==
                    list(itertools.product(range(1, n+1), ('movement', 'commit'))), 'Diagnostic stage inventory')
            for d in guard['diagnostics']:
                compare(d, dict(local_unchanged=True, unanimous_unchanged=True, skipped=not observe), 'Diagnostic branch')
                for field in [k for k in d if k.endswith('_seconds')]:
                    require(type(d[field]) in (float, int) and math.isfinite(d[field]) and d[field] >= 0, 'Finite diagnostic residence')
    compare({k:v for k,v in guards[0].items() if k != 'diagnostics'},
            {k:v for k,v in guards[1].items() if k != 'diagnostics'}, 'Rank guard agreement')


def spans(row):
    return [dict(end_step=end, jobs=16,
        scheduled_arrival_span_ns=row['arrival_offsets_ns'][end]-row['arrival_offsets_ns'][end-15],
        service_coverage_span_ns=row['coordinator_finish_ns'][end]-row['coordinator_start_ns'][end-15],
        completion_span_ns=row['coordinator_finish_ns'][end]-row['coordinator_finish_ns'][end-15])
        for end in range(63, row['n_jobs'], 64)]


def numerical(result, count):
    compare(result, dict(correctness_pass=True, correctness_atol=.02, correctness_rtol=.03,
                         full_batch_inputs_checked=count), 'Native numerical attestation')
    for field in ('correctness_max_abs_error', 'full_batch_max_abs_difference'):
        require(type(result[field]) in (float, int) and math.isfinite(result[field]) and result[field] >= 0,
                'Finite nonnegative numerical error')


def routing(row, base, c, fixed_map):
    """Both arms use native assignment; no production mapper is imported."""
    normalized = dict(row, arm='native')
    sequence = c['sequence'][:row['n_jobs']]
    compare(normalized, dict(**c['provenance'], logical_routing_sha256=array_hash(c['logical']),
        assignment_sha256=c['hashes']['native'], source_map=fixed_map.tolist(), final_map=fixed_map.tolist(),
        resolved_kernel_config=base['default_configs'][str(row['batch_tokens'])],
        warmup_raw_indices=list(range(16)), conditioning_raw_indices=[16+i%32 for i in range(64)],
        measured_raw_indices=sequence, full_unique_preflight_gpu_routes_checked=48,
        assignment_metrics=[c['diagnostics']['native'][i] for i in sequence]), 'Native route/input contract')
    decisions = row['decision_log']
    compare([d['step'] for d in decisions], list(range(63, row['n_jobs']-1, 64)), 'Native decisions excluding terminal proposal')
    for d in decisions:
        raw = sequence[d['step']-15:d['step']+1]
        counts = c['counts']['native'][raw]
        require(np.array_equal(d['physical_slot_assignments_window'], counts), 'Actual timed physical observation')
        require(d['physical_slot_assignments_sha256'] == array_hash(counts, np.int32), 'Physical window hash')
        require(np.array_equal(d['current_counts'], np.bincount(c['logical'][raw].ravel(), minlength=128)*2), 'Native logical observation')
        require(np.array_equal(d['physical_owner_assignments_per_step'], counts.reshape(16, 2, 80).sum(axis=2)), 'Rank observation ownership')
        compare(d, dict(current_map=fixed_map.tolist(), committed_map=fixed_map.tolist(),
            committed_map_sha256=array_hash(fixed_map), allowed=False, reason='always_skip',
            commit_completed=True, transfer_sent_bytes=0, transfer_sent_bytes_by_rank=[0, 0],
            physical_window_next_write=0, physical_counter_input_multiplicity=2,
            physical_window_measured_step_range=[d['step']-15, d['step']]), 'Native unchanged-map decision')
        proposed = np.asarray(d['proposed_map'])
        require(proposed.shape == (160,) and set(proposed) == set(range(128))
                and d['candidate_changed_slots'] == int((proposed != fixed_map).sum()), 'Native proposed map')
    require(np.array_equal(row['terminal_physical_window'], c['counts']['native'][sequence[-16:]]), 'Terminal actual physical window')
    require(row['decision_trajectory_sha256'] == digest([{k:d[k] for k in
            ('step', 'current_counts', 'current_map', 'proposed_map', 'committed_map')} for d in decisions]), 'Native planner trajectory checksum')
    hashes = row['source_weight_rank_sha256']
    require(len(hashes) == 2 and all(isinstance(x, str) and len(x) == 64 for x in hashes), 'Resident rank tensor hashes')
    require(hashlib.sha256(json.dumps(hashes).encode()).hexdigest() == row['source_weights_sha256'], 'Resident aggregate hash')
    numerical(row['preflight_correctness'], 48)
    numerical(row, 2)


def verify_proof(proof, row, repo):
    fixed = np.r_[np.arange(128), np.arange(32)].astype(np.int64)
    changed = fixed.copy(); changed[0], changed[80] = changed[80], changed[0]
    compare(proof, dict(gpu_noop_proof_schema_version=1, passed=True, diagnostic_only=True,
        evidence_eligible=False, source_restore_verified=True, native_symbols_restored=True,
        no_pending_async_work=True, numerical_raw_indices=[16, 47],
        proof_source_sha256=file_hash(repo/'arrival_fix/proof.py'), source_map_sha256=array_hash(fixed),
        changed_map_sha256=array_hash(changed)), 'Untimed GPU proof identity')
    for key in ('batch_tokens', 'native_vllm_commit', 'capture_sha256', 'replay_sha256',
                'source_prompt_ids', 'assignment_artifact', 'logical_routing_sha256',
                'resolved_kernel_config', 'source_weights_sha256'):
        compare(proof[key], row[key], 'Proof source '+key)
    compare(proof['source_weight_hashes_by_rank'], row['source_weight_rank_sha256'], 'Proof rank weights')
    compare(proof['restored_weight_hashes_by_rank'], row['source_weight_rank_sha256'], 'Proof restored rank weights')
    require(proof['restored_weights_sha256'] == row['source_weights_sha256'], 'Proof exact restored source')
    guards = proof['guard_counters_by_rank']
    require(len(guards) == 2 and guards[0] == guards[1], 'Both rank GPU proof agreement')
    compare(guards[0], dict(mode='fast', diagnostic_timing=False, pending_pair=False,
        movement_calls=4, commit_calls=4, agreement_calls=4, local_unchanged_calls=2,
        unanimously_unchanged_calls=2, movement_skips=2, commit_skips=2,
        original_movement_calls=2, original_commit_calls=2, fallback_calls=2, diagnostics=[]), 'Two skips and two real native fallbacks')
    steps = [('unchanged_before', fixed, True), ('changed_swap_0_80', changed, False),
             ('restore_original', fixed, False), ('unchanged_after_restore', fixed, True)]
    require(len(proof['steps']) == 4, 'Complete GPU proof operations')
    total = 0
    for evidence, (name, mapping, skipped) in zip(proof['steps'], steps):
        compare(evidence, dict(name=name, expected_skip=skipped, map_sha256=array_hash(mapping),
            exact_checkpoint_weights_verified=True, inverse_map_and_replica_count_verified=True), 'Proof step identity')
        numerical(evidence, 2)
        ranks = evidence['transfer_payload_by_rank']
        require(len(ranks) == 2 and [r['rank'] for r in ranks] == [0, 1], 'Proof transport ranks')
        for rank in ranks:
            old.integer(rank['bytes'], 'Proof actual bytes'); old.integer(rank['messages'], 'Proof actual messages')
        count, messages = sum(r['bytes'] for r in ranks), sum(r['messages'] for r in ranks)
        require(evidence['actual_transfer_bytes'] == count and evidence['actual_transfer_messages'] == messages,
                'Proof actual payload conservation')
        require((count == messages == 0) if skipped else (count > 0 and messages > 0), 'Proof skipped versus real movement')
        total += count
    require(proof['total_transfer_bytes'] == total, 'Proof total payload')
    return dict(verified=True, unchanged_skips=2, real_native_movements=2, actual_transfer_bytes=total,
                mixed_rank_gpu_corruption_test=False)


def flat(row, metrics):
    result = {f'{category}_{name}_ns':value for category, summary in metrics['summary_ns'].items()
              for name, value in summary.items()}
    for key in ('backlog_at_horizon', 'waiting_jobs_at_horizon', 'drain_ns',
                'completed_by_horizon_per_second', 'completed_per_evaluation_second',
                'required_observation_extent_ns', 'observation_overshoot_ns'):
        result[key] = metrics[key]
    elapsed = row['observation_end_ns']-row['epoch_monotonic_ns']
    result.update(observed_elapsed_seconds=elapsed/1e9,
        completion_elapsed_seconds=(row['coordinator_finish_ns'][-1]-row['epoch_monotonic_ns'])/1e9,
        coordinator_server_busy_fraction=sum(metrics['vectors_ns']['service'])/elapsed)
    return result


def check_summary(summary, plan, rows, computed):
    compare(summary, dict(valid=True, phase='final', expected_records=256, observed_records=256,
        valid_records=256, matched_pairs=128, errors=[], missing_keys=[],
        plan_sha256=plan['freeze']['sha256'], arrival_fix_source_sha256=plan['arrival_fix_source_sha256']), 'Final report identity')
    values = {tuple(r[k] for k in KEY):flat(r, computed[tuple(r[k] for k in KEY)]) for r in rows}
    trials = {tuple(r[k] for k in KEY):r for r in summary['trial_summaries']}
    require(len(trials) == len(summary['trial_summaries']) == 256 and set(trials) == set(values), 'Report trial completeness')
    for row in rows:
        key = tuple(row[k] for k in KEY)
        compare(trials[key], dict(metrics=values[key], observed_window_spans=spans(row),
            allocation_id=row['allocation_id'], block_id=row['block_id'], regime=row['regime']), 'Recomputed trial report')
    pairs = {tuple(p[k] for k in KEY[:-1]):p for p in summary['paired_cases']}
    require(len(pairs) == len(summary['paired_cases']) == 128 and set(pairs) == {k[:-1] for k in values}, 'Report pair completeness')
    for key, pair in pairs.items():
        native, fast = values[(*key, 'native')], values[(*key, 'fast')]
        logs = {m:math.log(fast[m])-math.log(native[m]) for m in ('response_mean_ns', 'response_p95_ns', 'service_mean_ns')}
        compare(pair, dict(native=native, fast=fast, log_ratios=logs), 'Recomputed paired report')


def audit(plan, rows, repo, capture_root, input_roots, *, calibration_plan=None,
          calibration_rows=None, calibration_raw_sha=None, diagnostics=None, proof=None,
          summary=None, cache=None):
    base, plan_sha, base_sha = plan_identity(plan, repo)
    cache = {} if cache is None else cache
    oracle = old.load_route_oracle(repo)
    cal_index, calibration_audit = {}, None
    if plan['phase'] == 'final':
        require(calibration_plan is not None and calibration_rows is not None, 'Final requires raw native calibration evidence')
        require(plan['calibration_manifest_sha256'] == calibration_plan['freeze']['sha256']
                and plan['calibration_records_sha256'] == calibration_raw_sha, 'Frozen calibration provenance')
        require(calibration_plan['phase'] == 'calibration', 'Correct calibration phase')
        calibration_audit = audit(calibration_plan, calibration_rows, repo, capture_root, input_roots, cache=cache)
        for key in ('arrival_fix_source_sha256', 'base_manifest', 'matrix', 'gamma_weights', 'hardware', 'analysis'):
            compare(plan[key], calibration_plan[key], 'Calibration-to-final unchanged contract '+key)
        cal_index = {tuple(r[k] for k in STATE):r for r in calibration_rows}
        require(set(plan['calibration']) == {old.state_name(s) for s in cal_index}, 'Final calibration state inventory')
        for state, row in cal_index.items():
            compare(plan['calibration'][old.state_name(state)], dict(**old.capacity(row), record_id=row['record_id'],
                allocation_id=row['allocation_id'], capture_sha256=row['capture_sha256'], replay_sha256=row['replay_sha256']), 'Final capacity from raw native clocks')
        require(set(plan['schedules']) == {f"{b['id']}/B{B}" for b in expected_blocks() for B in (128, 1024)}, 'Final calendar inventory')
        for block in expected_blocks():
            for B in (128, 1024):
                elapsed = old.capacity(cal_index[block['block_id'], block['regime'], B])['calibration_elapsed_ns']
                expected = [rebuild_calendar(plan['gamma_weights'][block['id']][pattern], rho, elapsed)
                            for rho, pattern in itertools.product((.5, .85), ('spread', 'cluster'))]
                compare(plan['schedules'][f"{block['id']}/B{B}"], expected, 'Exact normalized gamma calendar')
    inventory = execution_inventory(plan)
    require(len(rows) == len(inventory) == {'pilot':24, 'calibration':8, 'final':256}[plan['phase']], 'Complete primary trial count')
    require([tuple(r[k] for k in KEY) for r in rows] == [item[1] for item in inventory], 'Exact primary execution order/identity')
    if plan['phase'] == 'pilot':
        require(diagnostics is not None and proof is not None, 'Pilot requires separate stage diagnostics and GPU branch proof')
        diagnostic_inventory = execution_inventory(plan, diagnostic=True)
        require(len(diagnostics) == 4 and [tuple(r[k] for k in KEY) for r in diagnostics] ==
                [item[1] for item in diagnostic_inventory], 'Complete separate diagnostic stream')
    else:
        require(diagnostics is None and proof is None, 'Diagnostics/proof belong to pilot only')
        diagnostic_inventory, diagnostics = [], []
    all_rows = rows + diagnostics
    require(len({str(r['allocation_id']) for r in all_rows}) == 1 and all(r['allocation_id'] for r in all_rows), 'One nonempty allocation')
    require(len({tuple(r['rank_gpu_uuids']) for r in all_rows}) == 1, 'Stable rank-to-GPU mapping')
    artifacts, computed, state_rows = {}, {}, {}
    entries = [(r, item, False) for r, item in zip(rows, inventory)]
    entries += [(r, item, True) for r, item in zip(diagnostics, diagnostic_inventory)]
    for row, (state, key, order, schedule, block), diagnostic in entries:
        n, phase = (128, 'diagnostic') if diagnostic else (256, plan['phase'])
        compare({k:row[k] for k in STATE}, dict(zip(STATE, state)), 'Payload state assignment')
        hardware(row)
        expected_meta = dict(arrival_fix_schema_version=1, phase=phase, config_hash=plan_sha,
            base_config_hash=base_sha, arrival_fix_source_sha256=plan['arrival_fix_source_sha256'],
            status='complete', policy='SKIP', evidence_eligible=phase == 'final', diagnostic_only=diagnostic,
            guard_symbols_restored=True, normal_completion_verified=True, resident_state_verified=True,
            rank_assignment_agreement=True, migration_bytes=0, migration_sent_messages=0,
            terminal_proposals_suppressed=1, measurement_setup_revision=2,
            gc_enabled_during_measurement=True, n_jobs=n,
            native_vllm_commit=base['environment']['vllm_commit'],
            resident_dispatch_file_sha256=base['execution_source_sha256'], arm_order=order,
            arrival_seed=block['seed'] if block else None)
        for field in ('legacy_harness_sha256', 'followup_sha256', 'optimization_sha256',
                      'resident_dispatch_sha256', 'dispatch_backend', 'eplb_mode'):
            expected_meta[field] = base['environment'][field]
        compare(row, expected_meta, 'Record source/execution contract')
        expected_id = digest({k:row[k] for k in ('config_hash', 'phase', 'arrival_block_id',
            'block_id', 'regime', 'batch_tokens', 'arm', 'repetition', 'schedule_name')})
        require(row['record_id'] == expected_id, 'Record identity checksum')
        compare(row, dict(arrival_schedule=schedule,
            arrival_offsets_ns=schedule['arrival_offsets_ns'] if schedule else [0]*n,
            horizon_ns=schedule['horizon_ns'] if schedule else 1,
            schedule_name=f"{schedule['pattern']}-rho{schedule['rho']}" if schedule else 'saturated'), 'Immutable assigned calendar')
        metrics = old.timing_metrics(row)
        compare(row['queue_metrics'], metrics, 'Independent timestamp and queue arithmetic')
        compare(row['observed_window_spans'], spans(row), 'Native window time spans')
        guard_counts(row, diagnostic)
        if phase == 'calibration':
            compare(row['calibration_capacity'], old.capacity(row), 'Saturated baseline capacity')
        if state not in cache:
            cache[state] = old.route_state(oracle, base, state, capture_root)
        c = cache[state]
        sha = row['assignment_artifact']['sha256']
        if sha not in artifacts:
            artifact = oracle.verify_cell_artifact(dict(row, arm='native'), input_roots, c['arrays'], c['artifact_meta'])
            artifacts[sha] = dict(state=list(state), artifact=artifact)
        require(artifacts[sha]['state'] == list(state), 'Route artifact must belong to this state')
        routing(row, base, c, oracle.FIXED_MAP)
        if not diagnostic:
            computed[key] = metrics
            state_rows.setdefault(state, []).append(row)
    invariant_fields = ('source_map', 'final_map', 'source_weights_sha256', 'source_weight_rank_sha256',
        'capture_sha256', 'replay_sha256', 'source_prompt_ids', 'logical_routing_sha256', 'assignment_sha256',
        'decision_trajectory_sha256', 'resolved_kernel_config', 'arrival_clock_contract', 'rank_gpu_uuids')
    for state, subset in state_rows.items():
        for field in invariant_fields:
            require(len({canonical(r[field]) for r in subset}) == 1, 'Within-state invariant '+field)
        if plan['phase'] == 'final':
            calibration = cal_index[state]
            require(all(str(r['allocation_id']) == str(calibration['allocation_id']) for r in subset), 'Final/calibration allocation identity')
            for field in invariant_fields:
                compare(subset[0][field], calibration[field], 'Calibration/final source state '+field)
    require(len({r['source_weights_sha256'] for r in all_rows}) == 1, 'Same resident weights across all states')
    proof_report = verify_proof(proof, rows[0], repo) if proof is not None else None
    if summary is not None:
        require(plan['phase'] == 'final', 'Production report comparison is final-only')
        check_summary(summary, plan, rows, computed)
    return dict(**{'pass':True}, phase=plan['phase'], plan_sha256=plan_sha, records=len(rows),
        diagnostic_records=len(diagnostics), jobs_checked=sum(r['n_jobs'] for r in all_rows),
        timed_native_windows_checked=sum(r['n_jobs']//64 for r in all_rows),
        route_artifacts=len(artifacts), artifacts=list(artifacts.values()),
        allocation_id=str(rows[0]['allocation_id']), hostname=rows[0]['hostname'],
        rank_gpu_uuids=rows[0]['rank_gpu_uuids'], calibration_audit=calibration_audit,
        gpu_proof=proof_report, production_trials_and_pairs_recomputed=summary is not None,
        statistical_bootstrap_independently_recomputed=False,
        independent_timing_auditor_sha256=file_hash(repo/'scripts/verify_instructor_arrivals.py'),
        independent_route_auditor_sha256=file_hash(repo/'scripts/verify_resident_online_records.py'),
        limitations=['Same-team independent arithmetic, not an external laboratory replication.',
            'CPU audit reconstructs saved raw routes and actual observed count windows, not live GPU numerical output.',
            'GPU numerical/state assertions are checked as recorded evidence; weights are not re-executed on GPU.',
            'Fresh randomized arrival calendars reuse previously observed captured layer inputs.',
            'Final endpoint confidence intervals require the separate independent statistical check.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--records', type=Path, nargs='+', required=True)
    parser.add_argument('--repo-root', type=Path, default=ROOT)
    parser.add_argument('--capture-root', type=Path, required=True)
    parser.add_argument('--inputs-root', type=Path, nargs='*', default=[])
    parser.add_argument('--calibration-plan', type=Path)
    parser.add_argument('--calibration-records', type=Path)
    parser.add_argument('--diagnostics', type=Path)
    parser.add_argument('--proof', type=Path)
    parser.add_argument('--summary', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        roots = [*args.inputs_root, *(p.parent/'inputs' for p in args.records)]
        if args.calibration_records:
            roots.append(args.calibration_records.parent/'inputs')
        report = audit(json.loads(args.plan.read_text()), old.read_records(args.records),
            args.repo_root, args.capture_root, roots,
            calibration_plan=json.loads(args.calibration_plan.read_text()) if args.calibration_plan else None,
            calibration_rows=old.read_records([args.calibration_records]) if args.calibration_records else None,
            calibration_raw_sha=old.original_raw_hash(args.calibration_records) if args.calibration_records else None,
            diagnostics=old.read_records([args.diagnostics]) if args.diagnostics else None,
            proof=json.loads(args.proof.read_text()) if args.proof else None,
            summary=json.loads(args.summary.read_text()) if args.summary else None)
    except (AssertionError, KeyError, ValueError, TypeError, OSError, IndexError, OverflowError) as exc:
        report = {'pass':False, 'error':f'{type(exc).__name__}: {exc}'}
    report['verifier_sha256'] = file_hash(Path(__file__))
    report['raw_decompressed_sha256'] = {str(p):old.original_raw_hash(p) for p in args.records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('pass', 'phase', 'records', 'jobs_checked', 'error') if k in report}))
    if not report['pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

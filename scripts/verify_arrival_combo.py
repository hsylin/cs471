#!/usr/bin/env python3
"""Independent raw-input, clock and state audit for the three-arm composition.

No production experiment modules are imported. Earlier independent auditors
provide raw-route reconstruction and integer-clock accounting. Native/shortcut
physical assignments must agree; combined assignments must match the separate
CPU online oracle. Logical demand and placement must agree across all three.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import itertools
import json
import math
from pathlib import Path
import random

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('independent_arrival_fix_auditor', ROOT/'scripts/verify_arrival_fix.py')
fix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fix)
old = fix.old
require, compare, digest = fix.require, fix.compare, fix.digest
canonical, file_hash, array_hash = fix.canonical, fix.file_hash, fix.array_hash
KEY, STATE = fix.KEY, fix.STATE
ARMS = ('native', 'shortcut', 'combined')
NEW_SOURCES = ['arrival_combo/'+n for n in ('__init__.py', 'common.py', 'calendars.py', 'runner.py', 'proof.py')]
NEW_SOURCES += ['scripts/freeze_arrival_combo.py', 'scripts/run-arrival-combo-kuma.sh']


def expected_blocks():
    states = list(itertools.product(('opt-pilot-01', 'opt-pilot-02'), ('domain0', 'domain1')))
    return [dict(id=f'combo-fresh-{i+1:02d}', seed=2026110911+i, block_id=s[0], regime=s[1])
            for i, s in enumerate(states*2)]


def order_key(block, batch, pattern, rho):
    return f'{block}/B{batch}/{pattern}/rho{rho}'


def reconstruct_orders(matrix, seed):
    result = {}
    for load_index, rho in enumerate(matrix['rhos']):
        cases = [(b['id'], B, p) for b in matrix['arrival_blocks']
                 for B in matrix['batches'] for p in matrix['patterns']]
        random.Random(seed+100+load_index).shuffle(cases)
        for position, case in enumerate(cases):
            rotation = (position+load_index) % 3
            forward = [ARMS[(rotation+i) % 3] for i in range(3)]
            result[order_key(*case, rho)] = [forward, forward[::-1]]
    return result


def plan_identity(plan, repo):
    require(plan['phase'] in ('pilot', 'calibration', 'final'), 'Supported composition phase')
    sha = digest({k:v for k,v in plan.items() if k != 'freeze'})
    compare(plan['freeze'], dict(algorithm='sha256', sha256=sha), 'Composition plan freeze')
    require(plan['arrival_combo_schema_version'] == 1, 'Composition schema')
    previous = json.loads((repo/'configs/arrival-response-calibration-v4.json').read_text())
    fix.plan_identity(previous, repo)
    files = dict(previous['arrival_fix_source_files'])
    files['arrival_fix/analysis.py'] = previous['analysis_file_sha256']
    files.update({name:file_hash(repo/name) for name in NEW_SOURCES})
    compare(plan['arrival_combo_source_files'], files, 'Full old/new execution source inventory')
    require(set(plan['arrival_combo_source_files']) == set(files), 'Exact composition source set')
    require(plan['arrival_combo_source_sha256'] == digest(files), 'Composition source fingerprint')
    for field, name in (('protocol_file_sha256', 'docs/arrival-combined-protocol.md'),
                        ('analysis_file_sha256', 'arrival_combo/analysis.py')):
        require(plan[field] == file_hash(repo/name), 'Frozen '+name)
    compare(plan['hardware'], dict(hostname='kh014', gpu_uuids=fix.UUIDS, world_size=2, interconnect='NV6'), 'Reserved hardware contract')
    compare(plan['jobs_per_trial'], 256, 'Composition job budget')
    compare(plan['order_seed'], 471431, 'Composition order seed')
    compare(plan['analysis'], dict(bootstrap_samples=100000, bootstrap_seed=471432,
        bootstrap='stratified_arrival_blocks', primary_ratio_limit=.95, guard_ratio_limit=1.02,
        guard_family_size=3, secondary_ratio_limit=.95), 'Prespecified composition inference settings')
    base = json.loads((repo/plan['base_manifest']['path']).read_text())
    base_sha = digest({k:v for k,v in base.items() if k != 'freeze'})
    require(base_sha == base['freeze']['sha256'] == plan['base_manifest']['sha256'], 'Frozen resident input/implementation manifest')
    require(base == json.loads((repo/previous['base_manifest']['path']).read_text()), 'Same previously reviewed resident implementation')
    matrix = dict(arrival_blocks=expected_blocks(), batches=[128, 1024], patterns=['spread', 'cluster'],
                  rhos=[.5, .85], reps=2, arms=list(ARMS))
    if plan['phase'] == 'pilot':
        matrix.update(arrival_blocks=[dict(id='pilot-old-calendar', seed=0,
                                          block_id='opt-pilot-01', regime='domain1')],
                      patterns=['constant', 'burst8'], rhos=[.85])
        source = repo/'results/instructor/arrivals-v1/evaluation-plan.json'
        require(plan['pilot_calendar_source_sha256'] == file_hash(source), 'Pilot old-calendar provenance')
        prior = json.loads(source.read_text())
        require(prior['freeze']['sha256'] == digest({k:v for k,v in prior.items() if k != 'freeze'}), 'Archived pilot calendar freeze')
        schedules = {f'pilot-old-calendar/B{B}':[s for s in prior['schedules'][f'opt-pilot-01/domain1/B{B}'] if s['rho'] == .85]
                     for B in (128, 1024)}
        compare(plan['schedules'], schedules, 'Exact previously used pilot schedules')
        for values in schedules.values():
            for schedule in values:
                old.calendar(schedule)
        require('gamma_weights' not in plan, 'Pilot cannot consume fresh final draws')
    else:
        require(set(plan['gamma_weights']) == {b['id'] for b in expected_blocks()}, 'Fresh combo weight inventory')
        for b in expected_blocks():
            require(set(plan['gamma_weights'][b['id']]) == {'spread', 'cluster'}, 'Fresh pattern inventory')
            for pattern in ('spread', 'cluster'):
                fix.verify_weights(plan['gamma_weights'][b['id']][pattern], b['seed'], pattern)
    compare(plan['matrix'], matrix, 'Fixed three-arm matrix and fresh seeds')
    require(set(plan['matrix']) == set(matrix), 'Unknown matrix dimension')
    compare(plan['arm_orders'], reconstruct_orders(matrix, plan['order_seed']), 'Independently reconstructed three-arm orders')
    require(set(plan['arm_orders']) == set(reconstruct_orders(matrix, plan['order_seed'])), 'Complete order cases')
    return base, sha, base_sha


def execution_inventory(plan):
    blocks = plan['matrix']['arrival_blocks']
    states = sorted({(b['block_id'], b['regime'], B) for b in blocks for B in (128, 1024)})
    random.Random(plan['order_seed']).shuffle(states)
    stream = []
    orders = reconstruct_orders(plan['matrix'], plan['order_seed'])
    for si, state in enumerate(states):
        if plan['phase'] == 'calibration':
            stream.append((state, ('native-calibration', state[2], 'saturated', None, 0, 'native'), ['native'], None, None))
            continue
        subset = [b for b in blocks if (b['block_id'], b['regime']) == state[:2]]
        random.Random(plan['order_seed']+si+1).shuffle(subset)
        for block in subset:
            schedules = plan['schedules'][f"{block['id']}/B{state[2]}"]
            require([(s['rho'], s['pattern']) for s in schedules] ==
                    list(itertools.product(plan['matrix']['rhos'], plan['matrix']['patterns'])), 'Frozen calendar sequence')
            for schedule in schedules:
                for rep in (0, 1):
                    order = orders[order_key(block['id'], state[2], schedule['pattern'], schedule['rho'])][rep]
                    for arm in order:
                        stream.append((state, (block['id'], state[2], schedule['pattern'], schedule['rho'], rep, arm),
                                       order, schedule, block))
    return stream


def dispatch_for(arm):
    require(arm in ARMS, 'Known composition arm')
    return 'online' if arm == 'combined' else 'native'


def routing(row, base, c, fixed):
    dispatch = dispatch_for(row['arm'])
    compare(row, dict(dispatch_mode=dispatch, noop_enabled=row['arm'] != 'native',
        assignment_selection_cost_included=True, online_residency_metadata_verified=dispatch == 'online',
        mapper_calls_measured=row['n_jobs'] if dispatch == 'online' else 0,
        measured_raw_indices=c['sequence'][:row['n_jobs']]), 'Dispatcher execution contract')
    # Only the route-oracle selector changes; recorded arrays are never relabeled.
    normalized = dict(row, arm=dispatch)
    old.verify_routing_row(normalized, base, c, fixed)
    fix.guard_counts(dict(row, arm='native' if row['arm'] == 'native' else 'fast'))


def compare_invariants(rows):
    common = ('source_map', 'final_map', 'source_weights_sha256', 'source_weight_rank_sha256',
        'capture_sha256', 'replay_sha256', 'source_prompt_ids', 'logical_routing_sha256',
        'decision_trajectory_sha256', 'resolved_kernel_config', 'arrival_clock_contract', 'rank_gpu_uuids')
    for field in common:
        require(len({canonical(row[field]) for row in rows}) == 1, 'Common logical/resident invariant '+field)
    native = [r for r in rows if r['arm'] in ('native', 'shortcut')]
    online = [r for r in rows if r['arm'] == 'combined']
    for subset in (native, online):
        if not subset:
            continue
        for field in ('assignment_sha256', 'assignment_metrics', 'terminal_physical_window'):
            require(len({canonical(r[field]) for r in subset}) == 1, 'Within-dispatch invariant '+field)
        windows = [[d['physical_slot_assignments_sha256'] for d in r['decision_log']] for r in subset]
        require(len({canonical(w) for w in windows}) == 1, 'Within-dispatch timed window invariant')
    return common


def check_summary(summary, plan, rows, computed):
    """Raw trial metrics are checked here; inference is checked separately."""
    compare(summary, dict(valid=True, phase='final', expected_records=384, observed_records=384,
        valid_records=384, matched_pairs=384, matched_triplets=128, errors=[], missing_keys=[], plan_sha256=plan['freeze']['sha256'],
        arrival_combo_source_sha256=plan['arrival_combo_source_sha256']), 'Final composition report identity')
    trials = {tuple(r[k] for k in KEY):r for r in summary['trial_summaries']}
    require(len(trials) == len(summary['trial_summaries']) == 384, 'Complete report trial inventory')
    require(set(trials) == {tuple(r[k] for k in KEY) for r in rows}, 'Report trials match raw identities')
    for row in rows:
        key = tuple(row[k] for k in KEY)
        compare(trials[key], dict(metrics=fix.flat(row, computed[key]), observed_window_spans=fix.spans(row),
            allocation_id=row['allocation_id'], block_id=row['block_id'], regime=row['regime']), 'Independent report trial arithmetic')
    contrasts = {'combined_over_shortcut':('shortcut','combined'),
                 'combined_over_native':('native','combined'), 'shortcut_over_native':('native','shortcut')}
    pairs = {(tuple(p[k] for k in KEY[:-1]), p['contrast']):p for p in summary['paired_cases']}
    groups = {tuple(r[k] for k in KEY[:-1]) for r in rows}
    require(len(pairs) == len(summary['paired_cases']) == 384
            and set(pairs) == set(itertools.product(groups, contrasts)), 'Complete three-contrast report inventory')
    values = {tuple(r[k] for k in KEY):fix.flat(r, computed[tuple(r[k] for k in KEY)]) for r in rows}
    for (key, name), pair in pairs.items():
        control_arm, treatment_arm = contrasts[name]
        control, treatment = values[(*key,control_arm)], values[(*key,treatment_arm)]
        logs = {metric:math.log(treatment[metric])-math.log(control[metric])
                for metric in ('response_mean_ns','response_p95_ns','service_mean_ns')}
        compare(pair, dict(control_arm=control_arm, treatment_arm=treatment_arm,
            control=control, treatment=treatment, log_ratios=logs), 'Independent paired report arithmetic')


def audit(plan, rows, repo, capture_root, input_roots, *, calibration_plan=None,
          calibration_rows=None, calibration_raw_sha=None, proofs=None, summary=None, cache=None):
    base, plan_sha, base_sha = plan_identity(plan, repo)
    cache = {} if cache is None else cache
    oracle = old.load_route_oracle(repo)
    cal_index, calibration_audit = {}, None
    if plan['phase'] == 'final':
        require(calibration_plan is not None and calibration_rows is not None, 'Final needs complete raw calibration')
        require(calibration_plan['phase'] == 'calibration'
                and plan['calibration_manifest_sha256'] == calibration_plan['freeze']['sha256']
                and plan['calibration_records_sha256'] == calibration_raw_sha, 'Frozen raw calibration provenance')
        calibration_audit = audit(calibration_plan, calibration_rows, repo, capture_root, input_roots, cache=cache)
        for key in ('arrival_combo_source_sha256', 'base_manifest', 'matrix', 'gamma_weights', 'hardware', 'analysis', 'arm_orders'):
            compare(plan[key], calibration_plan[key], 'Unchanged calibration/final contract '+key)
        cal_index = {tuple(r[k] for k in STATE):r for r in calibration_rows}
        require(set(plan['calibration']) == {old.state_name(s) for s in cal_index}, 'Complete capacity inventory')
        for state, row in cal_index.items():
            compare(plan['calibration'][old.state_name(state)], dict(**old.capacity(row), record_id=row['record_id'],
                allocation_id=row['allocation_id'], capture_sha256=row['capture_sha256'], replay_sha256=row['replay_sha256']), 'Raw native capacity')
        require(set(plan['schedules']) == {f"{b['id']}/B{B}" for b in expected_blocks() for B in (128, 1024)}, 'Complete final calendar inventory')
        for block in expected_blocks():
            for B in (128, 1024):
                elapsed = old.capacity(cal_index[block['block_id'], block['regime'], B])['calibration_elapsed_ns']
                schedules = [fix.rebuild_calendar(plan['gamma_weights'][block['id']][p], rho, elapsed)
                             for rho, p in itertools.product((.5, .85), ('spread', 'cluster'))]
                compare(plan['schedules'][f"{block['id']}/B{B}"], schedules, 'Exact independent fresh calendar reconstruction')
    inventory = execution_inventory(plan)
    require(len(rows) == len(inventory) == {'pilot':24, 'calibration':8, 'final':384}[plan['phase']], 'Complete bounded trial matrix')
    require([tuple(r[k] for k in KEY) for r in rows] == [x[1] for x in inventory], 'Complete deterministic raw execution order')
    require(len({str(r['allocation_id']) for r in rows}) == 1 and all(r['allocation_id'] for r in rows), 'One shared allocation')
    require(len({tuple(r['rank_gpu_uuids']) for r in rows}) == 1, 'Stable rank-device assignment')
    artifacts, computed, state_rows = {}, {}, {}
    for row, (state, key, order, schedule, block) in zip(rows, inventory):
        compare({k:row[k] for k in STATE}, dict(zip(STATE, state)), 'Assigned reused payload state')
        fix.hardware(row)
        expected = dict(arrival_combo_schema_version=1, phase=plan['phase'], config_hash=plan_sha,
            base_config_hash=base_sha, arrival_combo_source_sha256=plan['arrival_combo_source_sha256'],
            status='complete', policy='SKIP', evidence_eligible=plan['phase'] == 'final', diagnostic_only=False,
            guard_symbols_restored=True, normal_completion_verified=True, resident_state_verified=True,
            rank_assignment_agreement=True, migration_bytes=0, migration_sent_messages=0,
            terminal_proposals_suppressed=1, measurement_setup_revision=2,
            gc_enabled_during_measurement=True, n_jobs=256, arm_order=order,
            arrival_seed=block['seed'] if block else None,
            native_vllm_commit=base['environment']['vllm_commit'],
            resident_dispatch_file_sha256=base['execution_source_sha256'])
        for field in ('legacy_harness_sha256', 'followup_sha256', 'optimization_sha256',
                      'resident_dispatch_sha256', 'dispatch_backend', 'eplb_mode'):
            expected[field] = base['environment'][field]
        compare(row, expected, 'Composition record source and state contract')
        require(row['record_id'] == digest({k:row[k] for k in ('config_hash', 'phase', 'arrival_block_id',
            'block_id', 'regime', 'batch_tokens', 'arm', 'repetition', 'schedule_name')}), 'Record identity checksum')
        compare(row, dict(arrival_schedule=schedule, arrival_offsets_ns=schedule['arrival_offsets_ns'] if schedule else [0]*256,
            horizon_ns=schedule['horizon_ns'] if schedule else 1,
            schedule_name=f"{schedule['pattern']}-rho{schedule['rho']}" if schedule else 'saturated'), 'Immutable paired arrival calendar')
        metrics = old.timing_metrics(row)
        compare(row['queue_metrics'], metrics, 'Independent FCFS clock arithmetic')
        compare(row['observed_window_spans'], fix.spans(row), 'Actual native window time spans')
        if plan['phase'] == 'calibration':
            compare(row['calibration_capacity'], old.capacity(row), 'Native saturated capacity')
        if state not in cache:
            cache[state] = old.route_state(oracle, base, state, capture_root)
        c = cache[state]
        sha = row['assignment_artifact']['sha256']
        if sha not in artifacts:
            artifact = oracle.verify_cell_artifact(row, input_roots, c['arrays'], c['artifact_meta'])
            artifacts[sha] = dict(state=list(state), artifact=artifact)
        require(artifacts[sha]['state'] == list(state), 'Raw assignment artifact ownership')
        routing(row, base, c, oracle.FIXED_MAP)
        computed[key] = metrics
        state_rows.setdefault(state, []).append(row)
    for state, subset in state_rows.items():
        common = compare_invariants(subset)
        if plan['phase'] == 'final':
            calibration = cal_index[state]
            require(all(str(r['allocation_id']) == str(calibration['allocation_id']) for r in subset), 'Calibration and treatment share allocation')
            for field in common:
                compare(subset[0][field], calibration[field], 'Calibration/final common state '+field)
            native = next(r for r in subset if r['arm'] == 'native')
            for field in ('assignment_sha256', 'assignment_metrics', 'terminal_physical_window'):
                compare(native[field], calibration[field], 'Native calibration/final route '+field)
    require(len({r['source_weights_sha256'] for r in rows}) == 1, 'Same resident tensors across states')
    proof_reports = []
    if plan['phase'] == 'pilot':
        require(proofs is not None and len(proofs) == 2, 'Both batch sizes need independent composition proof evidence')
        require({p['batch_tokens'] for p in proofs} == {128, 1024}, 'Composition proof batch inventory')
        for proof in proofs:
            row = next(r for r in rows if r['batch_tokens'] == proof['batch_tokens'])
            proof_reports.append(verify_proof(proof, row, base, cache[tuple(row[k] for k in STATE)], repo))
    else:
        require(not proofs, 'Composition proof belongs to pilot only')
    if summary is not None:
        require(plan['phase'] == 'final', 'Only final reports contain planned comparisons')
        check_summary(summary, plan, rows, computed)
    positions = {arm:[0, 0, 0] for arm in ARMS}
    for row in rows:
        positions[row['arm']][row['arm_order'].index(row['arm'])] += 1
    return dict(**{'pass':True}, phase=plan['phase'], plan_sha256=plan_sha, records=len(rows),
        jobs_checked=len(rows)*256, timed_native_windows_checked=len(rows)*4,
        route_artifacts=len(artifacts), artifacts=list(artifacts.values()),
        allocation_id=str(rows[0]['allocation_id']), hostname=rows[0]['hostname'],
        rank_gpu_uuids=rows[0]['rank_gpu_uuids'], arm_position_counts=positions,
        calibration_audit=calibration_audit, composition_proofs=proof_reports,
        production_trial_and_pair_metrics_recomputed=summary is not None,
        statistical_bootstrap_independently_recomputed=False,
        helper_sha256={name:file_hash(repo/'scripts'/name) for name in
                      ('verify_arrival_fix.py', 'verify_instructor_arrivals.py', 'verify_resident_online_records.py')},
        limitations=['Independent same-team reanalysis, not another laboratory or hardware allocation.',
            'Raw saved preflight routes and actual native observation windows are reconstructed; live GPU numerical arithmetic is not rerun.',
            'Native proposed maps are checked for validity and invariance, not independently optimized.',
            'Fresh calendars reuse fixed captured payloads; physical assignment equality is required only within each dispatcher.',
            'Three arms estimate component contrasts, not a factorial interaction or synergy.',
            'Statistical confidence intervals require the separate independent inference audit.'])


def verify_proof(proof, row, base, c, repo):
    fixed = np.r_[np.arange(128), np.arange(32)].astype(np.int64)
    compare(proof, dict(gpu_combo_proof_schema_version=1, passed=True, correctness_pass=True,
        diagnostic_only=True, evidence_eligible=False, common_logical_candidate_trajectory_verified=True,
        arm_specific_physical_windows_verified=True, native_symbols_restored=True,
        reference_raw_indices=list(range(48)), fp32_raw_indices=[16, 47], proof_steps_per_arm=128,
        expected_decision_steps=[63], expected_noop_pairs_per_shortcut_arm=2,
        source_map_sha256=array_hash(fixed), proof_source_sha256=file_hash(repo/'arrival_combo/proof.py')),
        'Composition proof identity')
    for field in ('batch_tokens', 'native_vllm_commit', 'capture_sha256', 'replay_sha256',
                  'source_prompt_ids', 'assignment_artifact', 'logical_routing_sha256',
                  'resolved_kernel_config', 'source_weights_sha256'):
        compare(proof[field], row[field], 'Composition proof payload '+field)
    require([a['arm'] for a in proof['arms']] == list(ARMS), 'All three proof arms in declared order')
    sequence, trajectories = c['sequence'][:128], []
    for arm in proof['arms']:
        dispatch = dispatch_for(arm['arm'])
        compare(arm, dict(routing_arm=dispatch, passed=True, correctness_pass=True,
            normal_completion_verified=True, exact_checkpoint_weights_verified=True,
            inverse_map_and_replica_count_verified=True, native_symbols_restored=True,
            no_pending_async_work=True, online_residency_metadata_verified=dispatch == 'online',
            source_map=fixed.tolist(), final_map=fixed.tolist(), migration_bytes=0,
            migration_sent_messages=0, terminal_proposals_suppressed=1,
            source_weights_sha256=row['source_weights_sha256'], final_weights_sha256=row['source_weights_sha256'],
            source_weight_rank_sha256=row['source_weight_rank_sha256'], final_weight_rank_sha256=row['source_weight_rank_sha256'],
            assignment_sha256=c['hashes'][dispatch], mapper_calls_measured=128 if dispatch == 'online' else 0,
            measured_raw_indices=sequence), 'Composition proof arm contract')
        guards = arm['guard_counters_by_rank']
        fix.guard_counts(dict(arm='native' if arm['arm'] == 'native' else 'fast', n_jobs=128, noop_guard_ranks=guards))
        completions = arm['completion_by_rank']
        require(len(completions) == 2, 'Both ranks completed proof')
        for rank, evidence in enumerate(completions):
            compare(evidence, dict(guard=None if not guards else guards[rank],
                mapper_calls_measured=128 if dispatch == 'online' else 0, sent_bytes=0, sent_messages=0),
                'Proof rank completion')
        fix.numerical(arm['preflight_correctness'], 48)
        fix.numerical(arm['postflight_correctness'], 48)
        adversarial = arm['adversarial_checks']
        if dispatch == 'native':
            require(adversarial is None, 'Native proof must not claim an online invocation')
        else:
            compare(adversarial, dict(passed=True, rank_agreement=True, native_recording_state_unchanged=True,
                cases_checked=9, recording_modes_per_case=2, gpu_invocations_checked=18,
                batch_tokens=row['batch_tokens']), 'Composition online adversarial checks')
            require(adversarial['evidence_sha256'] == digest(adversarial['cases']), 'Composition adversarial evidence checksum')
        decisions = arm['decision_log']
        require([d['step'] for d in decisions] == [63], 'Composition proof nonterminal proposal count')
        for decision in decisions:
            raw = sequence[48:64]
            counts = c['counts'][dispatch][raw]
            require(np.array_equal(decision['physical_slot_assignments_window'], counts), 'Proof arm-specific physical counts')
            require(decision['physical_slot_assignments_sha256'] == array_hash(counts, np.int32), 'Proof physical window checksum')
            require(np.array_equal(decision['physical_owner_assignments_per_step'], counts.reshape(16, 2, 80).sum(axis=2)), 'Proof rank token ownership')
            logical = np.bincount(c['logical'][raw].ravel(), minlength=128)*2
            require(np.array_equal(decision['current_counts'], logical), 'Proof common logical demand')
            reconstructed = np.zeros(128, dtype=np.int64)
            np.add.at(reconstructed, fixed, counts.sum(axis=0))
            require(np.array_equal(reconstructed*2, logical), 'Proof physical/logical conservation')
            compare(decision, dict(current_map=fixed.tolist(), committed_map=fixed.tolist(),
                committed_map_sha256=array_hash(fixed), allowed=False, reason='always_skip',
                commit_completed=True, transfer_sent_bytes=0, transfer_sent_bytes_by_rank=[0, 0],
                physical_window_next_write=0, physical_counter_input_multiplicity=2,
                physical_window_measured_step_range=[48, 63]), 'Proof fixed-map native decision')
            proposal = np.asarray(decision['proposed_map'])
            require(proposal.shape == (160,) and set(proposal) == set(range(128))
                    and decision['candidate_changed_slots'] == int((proposal != fixed).sum()), 'Proof native proposed map validity')
        require(np.array_equal(arm['terminal_physical_window'], c['counts'][dispatch][sequence[-16:]]), 'Proof terminal physical counts')
        trajectories.append([{k:d[k] for k in ('step', 'allowed', 'reason', 'current_counts',
            'current_map', 'proposed_map', 'committed_map')} for d in decisions])
    require(trajectories[0] == trajectories[1] == trajectories[2], 'Three-arm proof common logical/planner trajectory')
    compare(trajectories[0], [{k:row['decision_log'][0][k] for k in
        ('step', 'allowed', 'reason', 'current_counts', 'current_map', 'proposed_map', 'committed_map')}],
        'Composition proof agrees with the first measured native proposal')
    require(proof['logical_candidate_trajectory_sha256'] == digest(trajectories[0]), 'Proof common trajectory checksum')
    return dict(verified=True, batch_tokens=proof['batch_tokens'], arms=3, steps_per_arm=128,
                timed_performance_evidence=False, changed_residency_supported=False,
                logical_trajectory_sha256=proof['logical_candidate_trajectory_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--records', type=Path, nargs='+', required=True)
    parser.add_argument('--repo-root', type=Path, default=ROOT)
    parser.add_argument('--capture-root', type=Path, required=True)
    parser.add_argument('--inputs-root', type=Path, nargs='*', default=[])
    parser.add_argument('--calibration-plan', type=Path)
    parser.add_argument('--calibration-records', type=Path)
    parser.add_argument('--proof', type=Path, nargs='*')
    parser.add_argument('--summary', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        roots = [*args.inputs_root, *(p.parent/'inputs' for p in args.records)]
        if args.calibration_records:
            roots.append(args.calibration_records.parent/'inputs')
        report = audit(json.loads(args.plan.read_text()), old.read_records(args.records), args.repo_root,
            args.capture_root, roots,
            calibration_plan=json.loads(args.calibration_plan.read_text()) if args.calibration_plan else None,
            calibration_rows=old.read_records([args.calibration_records]) if args.calibration_records else None,
            calibration_raw_sha=old.original_raw_hash(args.calibration_records) if args.calibration_records else None,
            proofs=[json.loads(p.read_text()) for p in args.proof] if args.proof else None,
            summary=json.loads(args.summary.read_text()) if args.summary else None)
    except (AssertionError, KeyError, ValueError, TypeError, OSError, IndexError, OverflowError, NotImplementedError) as exc:
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

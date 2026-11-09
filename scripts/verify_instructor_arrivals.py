#!/usr/bin/env python3
"""Independent CPU audit of immutable layer-job arrivals and routing evidence.

No arrival_study module is imported. Timing arithmetic, finite-sample quantiles,
queue occupancy, calendars, matrix completeness and pairing are reimplemented.
Only the earlier independent resident-route auditor supplies its CPU oracle.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import gzip
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import random

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STATE_FIELDS = ('block_id', 'regime', 'batch_tokens')
ROW_FIELDS = (*STATE_FIELDS, 'schedule_name', 'repetition', 'arm')
ARMS = ('native', 'online')


def require(condition, label):
    if not condition:
        raise AssertionError(label)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def array_hash(value, dtype=None):
    return hashlib.sha256(np.ascontiguousarray(value, dtype=dtype).tobytes()).hexdigest()


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, label + ': integer/range')
    return value


def compare(actual, expected, label):
    """Compare exact discrete evidence and independently computed float summaries."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(expected) <= set(actual), label + ': keys')
        for key, value in expected.items():
            compare(actual[key], value, label + '/' + str(key))
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ': length')
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, label + '/' + str(i))
    elif isinstance(expected, float):
        require(type(actual) in (float, int) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9), label + ': value')
    else:
        require(type(actual) is type(expected) and actual == expected, label + ': value/type')


def quantile(values, probability):
    ordered = sorted(values)
    position = (len(ordered)-1)*probability
    lower, upper = math.floor(position), math.ceil(position)
    return float(ordered[lower] + (ordered[upper]-ordered[lower])*(position-lower))


def summarize(values):
    return dict(mean=math.fsum(values)/len(values), p50=quantile(values, .50),
                p95=quantile(values, .95), p99=quantile(values, .99), max=max(values))


def interval_occupancy(arrivals, departures, epoch, end):
    """Integrate each waiting/residence interval, and enumerate the occupancy.

    This is independent of the production event-sweep implementation. Intervals
    are half-open, including simultaneous releases and zero queue waits.
    """
    require(all(epoch <= a <= d <= end for a, d in zip(arrivals, departures)), 'Occupancy interval bounds')
    area = sum(d-a for a, d in zip(arrivals, departures))
    points = set(arrivals) | set(departures)
    peak = max((sum(a <= t < d for a, d in zip(arrivals, departures)) for t in points), default=0)
    return dict(peak_jobs=peak, jobs_at_end=0, integral_job_ns=area,
                mean_jobs=area/(end-epoch) if end > epoch else 0.)


def timing_metrics(row):
    """Complete-run accounting; incomplete attempts fail rather than lose jobs."""
    n = integer(row['n_jobs'], 'job count', 1)
    epoch = integer(row['epoch_monotonic_ns'], 'epoch')
    horizon = integer(row['horizon_ns'], 'horizon', 1)
    offsets = row['arrival_offsets_ns']
    require(len(offsets) == n, 'Calendar omitted jobs')
    offsets = [integer(x, 'arrival offset') for x in offsets]
    require(offsets[0] == 0 and offsets[-1] < horizon and offsets == sorted(offsets), 'Calendar order/horizon')
    arrivals = [epoch+x for x in offsets]
    clocks = []
    for field in ('coordinator_admit_ns', 'coordinator_start_ns', 'coordinator_finish_ns'):
        require(len(row[field]) == n, field + ': missing jobs')
        clocks.append([integer(x, field) for x in row[field]])
    admits, starts, finishes = clocks
    for i in range(n):
        require(arrivals[i] <= admits[i] <= starts[i] < finishes[i], 'Arrival/admission/service order')
        if i:
            require(finishes[i-1] <= admits[i], 'Overlapping or out-of-order FCFS service')
    end = integer(row['observation_end_ns'], 'observation end')
    require(end >= max(epoch+horizon, finishes[-1]), 'Incomplete horizon or undrained jobs')
    for field in ('per_rank_start_ns', 'per_rank_gpu_complete_ns'):
        require(len(row[field]) == 2 and all(len(v) == n for v in row[field]), 'Missing rank timing arrays')
    for rank in range(2):
        for i in range(n):
            rs = integer(row['per_rank_start_ns'][rank][i], 'rank start')
            rf = integer(row['per_rank_gpu_complete_ns'][rank][i], 'rank completion')
            require(starts[i] <= rs <= rf <= finishes[i], 'GPU completion outside acknowledged service envelope')
    vectors = dict(
        response=[f-a for f, a in zip(finishes, arrivals)],
        queue_wait=[s-a for s, a in zip(starts, arrivals)],
        service=[f-s for f, s in zip(finishes, starts)],
        admission_delay=[e-a for e, a in zip(admits, arrivals)],
        dispatch_delay=[s-e for s, e in zip(starts, admits)])
    cutoff = epoch+horizon
    done = sum(f <= cutoff for f in finishes)
    begun = sum(s <= cutoff for s in starts)
    return dict(complete=True, scheduled_jobs=n, completed_jobs=n, unfinished_jobs=0,
        evaluation_end_ns=end, horizon_observed=True,
        required_observation_extent_ns=max(horizon, finishes[-1]-epoch),
        observation_overshoot_ns=end-max(cutoff, finishes[-1]), horizon_ns=horizon,
        drain_ns=max(0, finishes[-1]-cutoff), trailing_scheduled_idle_ns=max(0, cutoff-finishes[-1]),
        completed_by_horizon=done, backlog_at_horizon=n-done, waiting_jobs_at_horizon=n-begun,
        offered_jobs_per_second=n*1e9/horizon, completed_by_horizon_per_second=done*1e9/horizon,
        completed_per_evaluation_second=n*1e9/(end-epoch),
        vectors_ns=vectors, summary_ns={key: summarize(value) for key, value in vectors.items()},
        queue_occupancy=interval_occupancy(arrivals, starts, epoch, end),
        system_occupancy=interval_occupancy(arrivals, finishes, epoch, end),
        unfinished_due_latency_lower_bound_ns=[None]*n)


def capacity(row):
    n = row['n_jobs']
    elapsed = row['coordinator_finish_ns'][-1]-row['coordinator_start_ns'][0]
    return dict(calibration_jobs=n, calibration_elapsed_ns=elapsed,
                native_capacity_jobs_per_second=n*1e9/elapsed,
                native_mean_cycle_ns=elapsed/n,
                mean_calibration_service_ns=sum(f-s for s, f in zip(
                    row['coordinator_start_ns'], row['coordinator_finish_ns']))/n)


def nearest_even(numerator, denominator):
    quotient, remainder = divmod(numerator, denominator)
    twice = 2*remainder
    return quotient + int(twice > denominator or (twice == denominator and quotient % 2 == 1))


def calendar(schedule, calibration=None):
    """Rebuild integer offsets by explicit integer rounding, not round(Fraction)."""
    require(schedule['pattern'] in ('constant', 'burst8'), 'Unknown schedule pattern')
    require(schedule['rho'] in (.5, .85) and type(schedule['rho']) is float, 'Undeclared offered load')
    require(schedule['n_jobs'] == schedule['calibration_jobs'] == 256 and schedule['burst_size'] == 8,
            'Changed bounded calendar geometry')
    elapsed = integer(schedule['calibration_elapsed_ns'], 'calibration elapsed', 1)
    if calibration is not None:
        compare(schedule['calibration_elapsed_ns'], capacity(calibration)['calibration_elapsed_ns'],
                'Calendar baseline calibration')
    rho = Fraction(str(schedule['rho']))
    numerator, denominator = elapsed*rho.denominator, 256*rho.numerator
    group = 1 if schedule['pattern'] == 'constant' else 8
    offsets = [nearest_even((i//group)*group*numerator, denominator) for i in range(256)]
    horizon = nearest_even(256*numerator, denominator)
    expected = dict(schedule_schema_version=1, pattern=schedule['pattern'], rho=schedule['rho'],
        calibration_jobs=256, calibration_elapsed_ns=elapsed,
        native_capacity_jobs_per_second=256*1e9/elapsed, native_mean_cycle_ns=elapsed/256,
        arrival_rate_jobs_per_second=float(rho)*(256*1e9/elapsed),
        n_jobs=256, burst_size=8, horizon_ns=horizon, arrival_offsets_ns=offsets,
        rounding='nearest_nanosecond_ties_to_even', arrival_unit='synthetic preassembled B-token layer job')
    expected['schedule_sha256'] = digest(expected)
    compare(schedule, expected, 'Immutable calendar')
    require(set(schedule) == set(expected), 'Unknown calendar fields')
    return expected


def window_spans(row):
    offsets = row['arrival_offsets_ns']
    start, finish = row['coordinator_start_ns'], row['coordinator_finish_ns']
    return [dict(end_step=end, jobs=16,
        scheduled_arrival_span_ns=offsets[end]-offsets[end-15],
        service_coverage_span_ns=finish[end]-start[end-15],
        completion_span_ns=finish[end]-finish[end-15]) for end in (63, 127, 191, 255)]


def read_records(paths):
    rows = []
    for path in paths:
        opener = gzip.open if Path(path).suffix == '.gz' else open
        with opener(path, 'rt') as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def load_route_oracle(repo_root):
    path = repo_root/'scripts/verify_resident_online_records.py'
    spec = importlib.util.spec_from_file_location('independent_resident_route_auditor', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def state_name(state):
    return f'{state[0]}/{state[1]}/B{state[2]}'


def plan_identity(plan, repo_root):
    require(plan['phase'] in ('calibration', 'evaluation'), 'Unsupported plan phase')
    plan_sha = digest({k: v for k, v in plan.items() if k != 'freeze'})
    compare(plan['freeze'], dict(algorithm='sha256', sha256=plan_sha), 'Frozen arrival plan')
    paths = ['arrival_study/'+name for name in ('__init__.py', 'common.py', 'schedules.py', 'runner.py')]
    paths += ['scripts/run-instructor-arrivals-kuma.sh', 'scripts/run-instructor-arrival-study-kuma.sh',
              'scripts/freeze_instructor_arrivals.py']
    files = {name: file_hash(repo_root/name) for name in paths}
    compare(plan['arrival_source_files'], files, 'Arrival source files')
    require(plan['arrival_source_sha256'] == digest(files), 'Arrival source fingerprint')
    require(plan['protocol_file_sha256'] == file_hash(repo_root/'docs/instructor-arrival-protocol.md'),
            'Protocol changed after freeze')
    if 'analysis_file_sha256' in plan:
        require(plan['analysis_file_sha256'] == file_hash(repo_root/'arrival_study/analysis.py'),
                'Production analysis source pin')
    base = json.loads((repo_root/plan['base_manifest']['path']).read_text())
    base_sha = digest({k: v for k, v in base.items() if k != 'freeze'})
    require(base['freeze']['sha256'] == plan['base_manifest']['sha256'] == base_sha,
            'Underlying frozen resident manifest')
    require(base['phase'] == 'validation' and base['split'] == 'pilot', 'Arrival inputs must be development inputs')
    for name, sha in base['execution_source_sha256'].items():
        require(file_hash(repo_root/name) == sha, 'Resident execution source changed: '+name)
    require(digest(base['execution_source_sha256']) == base['environment']['resident_dispatch_sha256'],
            'Resident execution fingerprint')
    expected_matrix = dict(block_ids=['opt-pilot-01', 'opt-pilot-02'], regimes=['domain0', 'domain1'],
                           batch_tokens=[128, 1024], arms=list(ARMS), repetitions=2)
    compare(plan['matrix'], expected_matrix, 'Fixed development matrix')
    compare(plan['jobs_per_trial'], 256, 'Fixed job budget')
    compare(plan['offered_load_fractions'], [.5, .85], 'Offered-load budget')
    compare(plan['arrival_patterns'], ['constant', 'burst8'], 'Pattern budget')
    compare(plan['burst_size'], 8, 'Burst size')
    return base, plan_sha, base_sha


def route_state(oracle, base, state, capture_root):
    entry = next(b for b in base['banks'] if b['block_id'] == state[0])
    metadata, bank = oracle.authenticated_npz(capture_root/entry['capture_path'], entry['capture_sha256'])
    documents = next(b['source_document_ids'] for b in base['blocks'] if b['block_id'] == state[0])
    expected = dict(block_id=state[0], split='pilot', source_prompt_ids=documents,
                    source_model=base['environment']['model_id'], source_revision=base['environment']['model_revision'],
                    layer=0, num_experts=128, top_k=8, hidden_size=2048, correctness_pass=True)
    compare(metadata, expected, 'Authenticated development bank metadata')
    require(bank['expert_ids'].dtype == np.int64 and bank['expert_ids'].shape == (4096, 8)
            and bank['hidden_states_bf16'].dtype == np.uint16 and bank['hidden_states_bf16'].shape == (4096, 2048)
            and bank['combining_weights'].dtype == np.float32 and bank['combining_weights'].shape == (4096, 8),
            'Captured tensor contract')
    require(all(all(p[k] is True for k in ('native_boundary_exact', 'native_router_exact',
                                          'manual_router_exact', 'bf16_roundtrip_exact'))
                for p in metadata['prompts']), 'Native capture correctness attestation')
    indices, replay_sha, sequence, samples = oracle.positions_for(base, state, bank, capture_root)
    logical = bank['expert_ids'][indices]
    counts, hashes, diagnostics, selected = {}, {}, {}, {}
    for arm in ARMS:
        rows, diags, saved, h = [], [], [], hashlib.sha256()
        for raw, ids in enumerate(logical):
            slots, c, optimum = oracle.assignment(ids, arm)
            h.update(slots.astype(np.int64).tobytes())
            rows.append(c)
            diags.append(oracle.metrics(c, optimum, base['default_configs'][str(state[2])]['BLOCK_SIZE_M']))
            if raw in samples:
                saved.append(slots.astype(np.uint8))
        counts[arm] = np.asarray(rows, dtype=np.int32)
        hashes[arm], diagnostics[arm], selected[arm] = h.hexdigest(), diags, np.stack(saved)
    provenance = dict(capture_sha256=entry['capture_sha256'], replay_sha256=replay_sha, source_prompt_ids=documents)
    arrays = dict(indices=indices, physical_to_logical_map=oracle.FIXED_MAP,
                  native_physical_counts=counts['native'], online_physical_counts=counts['online'],
                  native_sample_physical_ids=selected['native'], online_sample_physical_ids=selected['online'])
    artifact_meta = dict(**provenance, logical_routing_sha256=array_hash(logical),
                         full_physical_route_sha256=hashes, sample_raw_indices=samples)
    return dict(logical=logical, indices=indices, sequence=sequence, samples=samples, counts=counts,
                hashes=hashes, diagnostics=diagnostics, provenance=provenance,
                arrays=arrays, artifact_meta=artifact_meta)


def verify_routing_row(row, base, state_data, fixed_map):
    arm, c = row['arm'], state_data
    sequence = c['sequence']
    compare(row, dict(**c['provenance'], logical_routing_sha256=array_hash(c['logical']),
        assignment_sha256=c['hashes'][arm], source_map=fixed_map.tolist(), final_map=fixed_map.tolist(),
        resolved_kernel_config=base['default_configs'][str(row['batch_tokens'])],
        full_unique_preflight_gpu_routes_checked=48, warmup_raw_indices=list(range(16)),
        conditioning_raw_indices=[16+i%32 for i in range(64)],
        assignment_metrics=[c['diagnostics'][arm][i] for i in sequence]), 'Routing/input contract')
    decisions = row['decision_log']
    compare([d['step'] for d in decisions], [63, 127, 191], 'Native cadence')
    for d in decisions:
        raw = sequence[d['step']-15:d['step']+1]
        expected = c['counts'][arm][raw]
        require(np.array_equal(d['physical_slot_assignments_window'], expected), 'Timed physical window')
        require(d['physical_slot_assignments_sha256'] == array_hash(expected, np.int32), 'Physical window checksum')
        require(np.array_equal(d['current_counts'], np.bincount(c['logical'][raw].ravel(), minlength=128)*2),
                'Native logical count aggregation')
        require(np.array_equal(d['physical_owner_assignments_per_step'], expected.reshape(16, 2, 80).sum(axis=2)),
                'Per-rank physical work')
        compare(d, dict(current_map=fixed_map.tolist(), committed_map=fixed_map.tolist(),
            committed_map_sha256=array_hash(fixed_map), allowed=False, reason='always_skip',
            commit_completed=True, transfer_sent_bytes=0, transfer_sent_bytes_by_rank=[0, 0],
            physical_window_next_write=0, physical_counter_input_multiplicity=2,
            physical_window_measured_step_range=[d['step']-15, d['step']]), 'Native fixed-map decision')
        proposed = np.asarray(d['proposed_map'])
        require(proposed.shape == (160,) and set(proposed) == set(range(128))
                and d['candidate_changed_slots'] == int((proposed != fixed_map).sum()), 'Native proposed map validity')
    require(np.array_equal(row['terminal_physical_window'], c['counts'][arm][sequence[-16:]]),
            'Terminal physical observation window')
    trajectory = digest([{k: d[k] for k in ('step', 'current_counts', 'current_map', 'proposed_map', 'committed_map')}
                         for d in decisions])
    require(row['decision_trajectory_sha256'] == trajectory, 'Decision trajectory checksum')
    rank_hashes = row['source_weight_rank_sha256']
    require(len(rank_hashes) == 2 and all(isinstance(h, str) and len(h) == 64 for h in rank_hashes),
            'Rank resident tensor checksums')
    require(hashlib.sha256(json.dumps(rank_hashes).encode()).hexdigest() == row['source_weights_sha256'],
            'Aggregate resident tensor checksum')
    for result, count in ((row['preflight_correctness'], 48), (row, 2)):
        compare(result, dict(correctness_pass=True, correctness_atol=.02, correctness_rtol=.03,
                             full_batch_inputs_checked=count), 'Numerical-check contract')
        require(all(type(result[k]) in (float, int) and math.isfinite(result[k]) and result[k] >= 0
                    for k in ('correctness_max_abs_error', 'full_batch_max_abs_difference')), 'Numerical error values')
    checks = row['adversarial_checks']
    compare(checks, dict(passed=True, rank_agreement=True, native_recording_state_unchanged=True,
        cases_checked=9, recording_modes_per_case=2, gpu_invocations_checked=18,
        batch_tokens=row['batch_tokens']), 'Adversarial mapping checks')
    require(checks['evidence_sha256'] == digest(checks['cases']), 'Adversarial evidence checksum')


def flat_metrics(metrics):
    scalars = ('backlog_at_horizon', 'waiting_jobs_at_horizon', 'drain_ns', 'trailing_scheduled_idle_ns',
               'offered_jobs_per_second', 'completed_by_horizon_per_second', 'completed_per_evaluation_second',
               'required_observation_extent_ns', 'observation_overshoot_ns')
    result = {k: metrics[k] for k in scalars}
    exposure = metrics['required_observation_extent_ns'] + metrics['observation_overshoot_ns']
    result['coordinator_server_busy_fraction'] = sum(metrics['vectors_ns']['service'])/exposure
    for name, summaries in metrics['summary_ns'].items():
        result.update({name+'_'+key+'_ns': value for key, value in summaries.items()})
    for name in ('queue_occupancy', 'system_occupancy'):
        result.update({name+'_'+key: value for key, value in metrics[name].items()})
    return result


def contrasts(native, online):
    return {key: dict(native=value, online=online[key], online_minus_native=online[key]-value,
                      online_over_native=online[key]/value if value else None)
            for key, value in native.items()}


def check_analysis(summary, plan, rows, computed):
    require(summary['valid'] is True and not summary['missing_keys'] and not summary['errors'],
            'Production analysis is incomplete/invalid')
    compare(summary, dict(phase=plan['phase'], expected_records=len(rows), observed_records=len(rows),
        valid_records=len(rows), plan_sha256=plan['freeze']['sha256'],
        arrival_source_sha256=plan['arrival_source_sha256']), 'Analysis identity/counts')
    index = {tuple(r[k] for k in ROW_FIELDS): r for r in rows}
    trials = {tuple(r[k] for k in ROW_FIELDS): r for r in summary['trial_summaries']}
    require(len(trials) == len(summary['trial_summaries']) == len(index) and set(trials) == set(index),
            'Analysis trial matrix')
    for key, r in index.items():
        expected = dict(metrics=flat_metrics(computed[key]), observed_window_spans=window_spans(r),
                        allocation_id=r['allocation_id'], record_id=r['record_id'])
        if plan['phase'] == 'calibration':
            expected['calibration_capacity'] = capacity(r)
        compare(trials[key], expected, 'Recomputed trial report '+str(key))
    if plan['phase'] == 'calibration':
        require(not summary['paired_cases'] and not summary['repeat_averaged_cases'], 'Unexpected calibration arm comparison')
        return
    pairs = {tuple(r[k] for k in ROW_FIELDS[:-1]): r for r in summary['paired_cases']}
    means = {tuple(r[k] for k in ROW_FIELDS[:4]): r for r in summary['repeat_averaged_cases']}
    require(len(pairs) == len(summary['paired_cases']) == 64 and len(means) == len(summary['repeat_averaged_cases']) == 32,
            'Paired/averaged report matrix')
    for state_name_key in {key[:4] for key in index}:
        per_arm = {arm: [] for arm in ARMS}
        for rep in (0, 1):
            for arm in ARMS:
                per_arm[arm].append(flat_metrics(computed[(*state_name_key, rep, arm)]))
            compare(pairs[(*state_name_key, rep)]['contrasts'],
                    contrasts(per_arm['native'][rep], per_arm['online'][rep]), 'Recomputed paired report')
        averaged = {arm: {metric: math.fsum(values[metric] for values in per_arm[arm])/2
                          for metric in per_arm[arm][0]} for arm in ARMS}
        compare(means[state_name_key]['contrasts'], contrasts(averaged['native'], averaged['online']),
                'Recomputed arithmetic repeat means')


def audit(plan, rows, repo_root, capture_root, inputs_roots, *, calibration_rows=None,
          calibration_plan=None, calibration_raw_sha=None, summary=None):
    base, plan_sha, base_sha = plan_identity(plan, repo_root)
    states = list(itertools.product(plan['matrix']['block_ids'], plan['matrix']['regimes'], plan['matrix']['batch_tokens']))
    expected, schedules, cal_index = set(), {}, {}
    if plan['phase'] == 'evaluation':
        require(calibration_rows is not None and calibration_plan is not None, 'Evaluation requires raw calibration and frozen plan')
        require(plan['calibration_manifest_sha256'] == calibration_plan['freeze']['sha256'],
                'Referenced calibration manifest')
        require(plan['calibration_records_file_sha256'] == calibration_raw_sha, 'Referenced raw calibration byte checksum')
        calibration_audit = audit(calibration_plan, calibration_rows, repo_root, capture_root, inputs_roots)
        require(calibration_audit['pass'], 'Calibration independent audit failed')
        cal_index = {tuple(r[k] for k in STATE_FIELDS): r for r in calibration_rows}
        require(set(plan['schedules']) == set(plan['calibration']) == {state_name(s) for s in states},
                'Calendar/calibration state inventory')
    for state in states:
        if plan['phase'] == 'calibration':
            expected.add((*state, 'calibration_saturated', 0, 'native'))
        else:
            declared = plan['schedules'][state_name(state)]
            require(len(declared) == 4 and [(s['rho'], s['pattern']) for s in declared] ==
                    list(itertools.product((.5, .85), ('constant', 'burst8'))), 'Prespecified schedule order')
            cal = cal_index[state]
            compare(plan['calibration'][state_name(state)], dict(**capacity(cal), record_id=cal['record_id'],
                    allocation_id=cal['allocation_id'], capture_sha256=cal['capture_sha256'],
                    replay_sha256=cal['replay_sha256']), 'Calibration source association')
            for schedule in declared:
                rebuilt = calendar(schedule, cal)
                name = f"{rebuilt['pattern']}-rho{rebuilt['rho']}"
                schedules[(*state, name)] = rebuilt
                expected.update((*state, name, rep, arm) for rep, arm in itertools.product((0, 1), ARMS))
    index = {tuple(r[k] for k in ROW_FIELDS): r for r in rows}
    require(len(rows) == len(index) == len(expected) and set(index) == expected, 'Missing/duplicate/extra trial keys')
    require(len({str(r['allocation_id']) for r in rows}) == 1 and all(r['allocation_id'] for r in rows),
            'Evaluation/calibration matrix spans allocations or lacks allocation ID')
    require(len({r['hostname'] for r in rows}) == 1, 'Unexpected multiple hostnames')
    oracle = load_route_oracle(repo_root)
    artifacts, artifact_states, computed = {}, {}, {}
    for state in states:
        c = route_state(oracle, base, state, capture_root)
        state_rows = [r for key, r in index.items() if key[:3] == state]
        for r in state_rows:
            key = tuple(r[k] for k in ROW_FIELDS)
            expected_meta = dict(arrival_schema_version=1, config_hash=plan_sha, base_config_hash=base_sha,
                arrival_source_sha256=plan['arrival_source_sha256'], phase=plan['phase'], status='complete',
                policy='SKIP', evidence_eligible=False, normal_completion_verified=True,
                resident_state_verified=True, rank_assignment_agreement=True, migration_bytes=0,
                migration_sent_messages=0, terminal_proposals_suppressed=1, measurement_setup_revision=2,
                gc_enabled_during_measurement=True, n_jobs=256, world_size=2, dtype='bfloat16',
                resident_dispatch_file_sha256=base['execution_source_sha256'],
                native_vllm_commit=base['environment']['vllm_commit'])
            for field in ('legacy_harness_sha256', 'followup_sha256', 'optimization_sha256',
                          'resident_dispatch_sha256', 'dispatch_backend', 'eplb_mode'):
                expected_meta[field] = base['environment'][field]
            compare(r, expected_meta, 'Arrival record identity/source contract')
            clock = r['arrival_clock_contract']
            require(clock['monotonic'] is True and clock['adjustable'] is False
                    and 'CLOCK_MONOTONIC' in clock['implementation'], 'Shared monotonic clock contract')
            expected_id = digest({k: r[k] for k in ('config_hash', *STATE_FIELDS, 'arm', 'repetition', 'schedule_name')})
            require(r['record_id'] == expected_id, 'Record identity checksum')
            if plan['phase'] == 'calibration':
                compare(r, dict(arrival_schedule=None, arrival_offsets_ns=[0]*256, horizon_ns=1,
                                arm_order=['native']), 'Saturated native calibration')
            else:
                schedule = schedules[key[:4]]
                compare(r, dict(arrival_schedule=schedule, arrival_offsets_ns=schedule['arrival_offsets_ns'],
                                horizon_ns=schedule['horizon_ns']), 'Paired immutable calendar')
            calculated = timing_metrics(r)
            compare(r['queue_metrics'], calculated, 'Raw-timestamp queue metrics')
            compare(r['observed_window_spans'], window_spans(r), 'Timestamp-derived native window spans')
            if plan['phase'] == 'calibration':
                compare(r['calibration_capacity'], capacity(r), 'Native capacity')
            computed[key] = calculated
            sha = r['assignment_artifact']['sha256']
            if sha not in artifacts:
                artifacts[sha] = oracle.verify_cell_artifact(r, inputs_roots, c['arrays'], c['artifact_meta'])
                artifact_states[sha] = state
            require(artifact_states[sha] == state, 'Route artifact borrowed from another state')
            verify_routing_row(r, base, c, oracle.FIXED_MAP)
        for field in ('source_map', 'final_map', 'source_weights_sha256', 'source_weight_rank_sha256',
                      'logical_routing_sha256', 'decision_trajectory_sha256', 'arrival_clock_contract'):
            require(len({canonical(r[field]) for r in state_rows}) == 1, 'Within-state invariant changed: '+field)
        if plan['phase'] == 'evaluation':
            require(all(str(r['allocation_id']) == str(cal_index[state]['allocation_id']) for r in state_rows),
                    'Calibration and evaluation used different allocations')
            for field in ('capture_sha256', 'replay_sha256', 'source_weights_sha256', 'source_map',
                          'logical_routing_sha256', 'decision_trajectory_sha256', 'resolved_kernel_config'):
                require(state_rows[0][field] == cal_index[state][field], 'Calibration/evaluation state mismatch: '+field)
    require(len({r['source_weights_sha256'] for r in rows}) == 1, 'Resident weights changed across states')
    random.Random(plan['order_seed']).shuffle(states)
    execution = []
    for state_index, state in enumerate(states):
        if plan['phase'] == 'calibration':
            execution.append((*state, 'calibration_saturated', 0, 'native'))
            continue
        for case_index, schedule in enumerate(plan['schedules'][state_name(state)]):
            name = f"{schedule['pattern']}-rho{schedule['rho']}"
            for rep in (0, 1):
                order = list(ARMS if (state_index+case_index+rep) % 2 == 0 else ARMS[::-1])
                for arm in order:
                    key = (*state, name, rep, arm)
                    compare(index[key]['arm_order'], order, 'Frozen reversed arm order')
                    execution.append(key)
    require([tuple(r[k] for k in ROW_FIELDS) for r in rows] == execution, 'Raw execution order differs from frozen plan')
    if summary is not None:
        check_analysis(summary, plan, rows, computed)
    report = dict(phase=plan['phase'], plan_sha256=plan_sha, records=len(rows),
        states=len(states), jobs_checked=256*len(rows), route_artifacts=len(artifacts),
        timed_native_windows_checked=4*len(rows), allocations=sorted({str(r['allocation_id']) for r in rows}),
        hostname=rows[0]['hostname'], analysis_report_recomputed=summary is not None,
        independent_route_oracle_sha256=file_hash(repo_root/'scripts/verify_resident_online_records.py'),
        artifacts=list(artifacts.values()), limitations=[
            'Independent arithmetic within the same project team, not an external experimental replication.',
            'Captured inputs and saved GPU preflight routes are independently reconstructed; actual timed route evidence covers native observation windows.',
            'GPU numerical outputs and live expert weights are not reproduced by this CPU audit; their attested checks and recorded hash invariance are verified.',
            'Same-host clock metadata supports rank envelopes; primary response arithmetic uses the rank-0 monotonic clock only.',
            'All results concern synthetic fixed-token layer jobs on reused development inputs, not user requests or population inference.'])
    report['pass'] = True
    return report


def original_raw_hash(path):
    h = hashlib.sha256()
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--records', type=Path, nargs='+', required=True)
    parser.add_argument('--repo-root', type=Path, default=ROOT)
    parser.add_argument('--capture-root', type=Path, required=True)
    parser.add_argument('--inputs-root', type=Path, nargs='*', default=[])
    parser.add_argument('--calibration-plan', type=Path)
    parser.add_argument('--calibration-records', type=Path)
    parser.add_argument('--summary', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        plan = json.loads(args.plan.read_text())
        roots = [*args.inputs_root, *(path.parent/'inputs' for path in args.records)]
        if args.calibration_records is not None:
            roots.append(args.calibration_records.parent/'inputs')
        report = audit(plan, read_records(args.records), args.repo_root, args.capture_root, roots,
            calibration_rows=read_records([args.calibration_records]) if args.calibration_records else None,
            calibration_plan=json.loads(args.calibration_plan.read_text()) if args.calibration_plan else None,
            calibration_raw_sha=original_raw_hash(args.calibration_records) if args.calibration_records else None,
            summary=json.loads(args.summary.read_text()) if args.summary else None)
    except (AssertionError, KeyError, ValueError, TypeError, OSError, IndexError) as error:
        report = {'pass': False, 'error': f'{type(error).__name__}: {error}'}
    report['verifier_sha256'] = file_hash(Path(__file__))
    report['raw_files'] = [dict(path=str(path), file_sha256=file_hash(path),
                              original_uncompressed_sha256=original_raw_hash(path))
                           for path in args.records if path.exists()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'artifacts'}, indent=2))
    raise SystemExit(0 if report['pass'] else 1)


if __name__ == '__main__':
    main()

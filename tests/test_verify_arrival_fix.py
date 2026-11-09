"""Hand-checkable verifier fixtures; these are tests, not GPU evidence."""
import copy
from fractions import Fraction
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('independent_fix_verifier', ROOT/'scripts/verify_arrival_fix.py')
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)


def weights(values=None):
    result = dict(seed=2026110901, pattern='spread', shape=4., n_jobs=256,
                  gamma_draws_hex=[x.hex() for x in (values or [1.]*256)])
    result['weights_sha256'] = v.digest(result)
    return result


def test_uniform_gaps_exact_arithmetic_and_unused_last_gap():
    c = v.rebuild_calendar(weights(), .5, 2560)
    assert c['horizon_ns'] == 5120
    assert c['arrival_offsets_ns'] == list(range(0, 5120, 20))
    assert c['horizon_ns']-c['arrival_offsets_ns'][-1] == 20
    assert c['schedule_sha256'] == v.digest({k:x for k,x in c.items() if k != 'schedule_sha256'})


def test_nearest_even_is_explicit_for_positive_ties():
    assert v.rational_round(Fraction(5, 2)) == 2
    assert v.rational_round(Fraction(7, 2)) == 4


def test_tiny_final_gap_preserves_job_by_one_ns_rounded_cap():
    c = v.rebuild_calendar(weights([1.]*255+[float.fromhex('0x1p-100')]), .5, 2560)
    assert len(c['arrival_offsets_ns']) == 256
    assert c['arrival_offsets_ns'][-1] == c['horizon_ns']-1
    assert c['arrival_offsets_ns'] == sorted(c['arrival_offsets_ns'])


def test_exact_binary64_fraction_avoids_decimal_reinterpretation():
    values = [0.1, 0.2]+[1.]*254
    c = v.rebuild_calendar(weights(values), .85, 1000000)
    total = sum(Fraction(*x.as_integer_ratio()) for x in values)
    expected = v.rational_round(Fraction(1000000)/Fraction('0.85')*Fraction(*(.1).as_integer_ratio())/total)
    assert c['arrival_offsets_ns'][1] == expected


def test_archived_weight_mutation_rejected_even_if_valid_float():
    w = weights(); w['gamma_draws_hex'][17] = (2.).hex()
    with pytest.raises(AssertionError, match='checksum'):
        v.rebuild_calendar(w, .5, 2560)


def test_gamma_wrong_seed_cannot_be_hidden_by_rehashing():
    import random
    rng = random.Random(2026110901*2)
    w = dict(seed=2026110901, rng_seed=2026110901*2, pattern='spread', shape=4.,
        generator='Python Random.gammavariate; exact binary64 draws archived as hex',
        gamma_draws_hex=[rng.gammavariate(4., 1.).hex() for _ in range(256)], n_jobs=256)
    w['weights_sha256'] = v.digest(w)
    v.verify_weights(w, 2026110901, 'spread')
    w['gamma_draws_hex'][0] = (1.).hex()
    w['weights_sha256'] = v.digest({k:x for k,x in w.items() if k != 'weights_sha256'})
    with pytest.raises(AssertionError, match='Prespecified gamma draw'):
        v.verify_weights(w, 2026110901, 'spread')


def final_plan():
    blocks = v.expected_blocks()
    return dict(phase='final', order_seed=471421,
        matrix=dict(arrival_blocks=blocks, arms=['native', 'fast'], rhos=[.5, .85], patterns=['spread', 'cluster']),
        schedules={f"{b['id']}/B{B}":[dict(rho=rho, pattern=p) for rho in (.5, .85) for p in ('spread', 'cluster')]
                   for b in blocks for B in (128, 1024)})


def test_complete_final_order_and_two_reverse_repetitions():
    stream = v.execution_inventory(final_plan())
    assert len(stream) == 256
    assert len({item[1] for item in stream}) == 256
    for i in range(0, 256, 4):
        first, second = stream[i:i+2], stream[i+2:i+4]
        assert [x[1][-1] for x in first] == [x[1][-1] for x in second][::-1]
        assert all(x[1][-2] == 0 for x in first)
        assert all(x[1][-2] == 1 for x in second)
    assert len({(x[0][0], x[0][1]) for x in stream}) == 4


def test_calendar_order_change_is_rejected():
    p = final_plan(); first = next(iter(p['schedules']))
    p['schedules'][first] = p['schedules'][first][::-1]
    with pytest.raises(AssertionError, match='calendar order'):
        v.execution_inventory(p)


def test_calibration_stream_contains_all_eight_native_states():
    p = final_plan(); p['phase'] = 'calibration'; del p['schedules']
    stream = v.execution_inventory(p)
    assert len(stream) == len({x[0] for x in stream}) == 8
    assert all(x[1][-1] == 'native' and x[3] is None and x[4] is None for x in stream)


def hardware_row():
    return dict(hostname='kh014', physical_gpu_uuids=v.UUIDS, rank_gpu_uuids=v.UUIDS,
                world_size=2, dtype='bfloat16', arrival_clock_contract=dict(monotonic=True,
                adjustable=False, implementation='clock_gettime(CLOCK_MONOTONIC)'))


@pytest.mark.parametrize('change', ('duplicate_rank', 'wrong_pair', 'wrong_host', 'wall_clock'))
def test_hardware_requires_actual_distinct_rank_devices(change):
    row = hardware_row()
    if change == 'duplicate_rank': row['rank_gpu_uuids'] = [v.UUIDS[0]]*2
    if change == 'wrong_pair': row['physical_gpu_uuids'] = [v.UUIDS[0], 'another-gpu']
    if change == 'wrong_host': row['hostname'] = 'kh084'
    if change == 'wall_clock': row['arrival_clock_contract']['adjustable'] = True
    with pytest.raises(AssertionError): v.hardware(row)


def fast_row(n=256):
    count = n//64
    guard = dict(mode='fast', diagnostic_timing=False, pending_pair=False,
        movement_calls=count, commit_calls=count, agreement_calls=count,
        local_unchanged_calls=count, unanimously_unchanged_calls=count,
        movement_skips=count, commit_skips=count, original_movement_calls=0,
        original_commit_calls=0, fallback_calls=0, diagnostics=[])
    return dict(arm='fast', n_jobs=n, noop_guard_ranks=[guard, copy.deepcopy(guard)])


def test_guard_includes_terminal_unchanged_pair():
    v.guard_counts(fast_row())
    row = fast_row()
    for guard in row['noop_guard_ranks']:
        for name in ('movement_calls', 'commit_calls', 'agreement_calls', 'movement_skips', 'commit_skips'):
            guard[name] = 3
    with pytest.raises(AssertionError): v.guard_counts(row)


def test_native_alias_cannot_install_even_an_observe_wrapper():
    for arm in ('native', 'native_alias'):
        v.guard_counts(dict(arm=arm, n_jobs=256, noop_guard_ranks=[]))
        with pytest.raises(AssertionError, match='Unwrapped'):
            v.guard_counts(dict(arm=arm, n_jobs=256, noop_guard_ranks=fast_row()['noop_guard_ranks']))


def test_fast_primary_stage_instrumentation_rejected():
    row = fast_row(); row['noop_guard_ranks'][0]['diagnostics'] = [{}]
    with pytest.raises(AssertionError, match='stage timing'): v.guard_counts(row)


def test_recorded_rank_guard_disagreement_rejected():
    row = fast_row(); row['noop_guard_ranks'][1]['fallback_calls'] = 1
    with pytest.raises(AssertionError): v.guard_counts(row)


def test_diagnostic_window_geometry_is_128_jobs():
    row = dict(n_jobs=128, arrival_offsets_ns=list(range(128)),
               coordinator_start_ns=list(range(1000, 1128)), coordinator_finish_ns=list(range(1002, 1130)))
    assert v.spans(row) == [dict(end_step=end, jobs=16, scheduled_arrival_span_ns=15,
        service_coverage_span_ns=17, completion_span_ns=15) for end in (63, 127)]


def route_fixture(n=128):
    """Small logical fixture with actual dimensions and hand-assigned counts."""
    fixed = np.r_[np.arange(128), np.arange(32)].astype(np.int64)
    sequence = [16+i%32 for i in range(n)]
    logical = np.zeros((48, 1, 8), dtype=np.int64)
    logical[:] = np.arange(8)
    counts = np.zeros((48, 160), dtype=np.int32); counts[:, :8] = 1
    provenance = dict(capture_sha256='a'*64, replay_sha256='b'*64, source_prompt_ids=['fixture'])
    c = dict(sequence=[16+i%32 for i in range(256)], logical=logical, provenance=provenance,
             hashes={'native':'c'*64}, counts={'native':counts}, diagnostics={'native':[{} for _ in range(48)]})
    d = dict(step=63, current_map=fixed.tolist(), proposed_map=fixed.tolist(), committed_map=fixed.tolist(),
        committed_map_sha256=v.array_hash(fixed), current_counts=(np.bincount(logical[sequence[48:64]].ravel(), minlength=128)*2).tolist(),
        physical_slot_assignments_window=counts[sequence[48:64]].tolist(),
        physical_slot_assignments_sha256=v.array_hash(counts[sequence[48:64]], np.int32),
        physical_owner_assignments_per_step=counts[sequence[48:64]].reshape(16, 2, 80).sum(axis=2).tolist(),
        allowed=False, reason='always_skip', commit_completed=True, transfer_sent_bytes=0,
        transfer_sent_bytes_by_rank=[0, 0], physical_window_next_write=0, physical_counter_input_multiplicity=2,
        physical_window_measured_step_range=[48, 63], candidate_changed_slots=0)
    hashes = ['d'*64, 'e'*64]
    numerical = dict(correctness_pass=True, correctness_atol=.02, correctness_rtol=.03,
        full_batch_inputs_checked=2, correctness_max_abs_error=.001, full_batch_max_abs_difference=0.)
    row = dict(arm='fast', n_jobs=n, batch_tokens=128, **provenance, logical_routing_sha256=v.array_hash(logical),
        assignment_sha256='c'*64, source_map=fixed.tolist(), final_map=fixed.tolist(), resolved_kernel_config={'BM':64},
        warmup_raw_indices=list(range(16)), conditioning_raw_indices=[16+i%32 for i in range(64)],
        measured_raw_indices=sequence, full_unique_preflight_gpu_routes_checked=48, assignment_metrics=[{}]*n,
        decision_log=[d], terminal_physical_window=counts[sequence[-16:]].tolist(), source_weight_rank_sha256=hashes,
        source_weights_sha256=v.hashlib.sha256(v.json.dumps(hashes).encode()).hexdigest(),
        preflight_correctness={**numerical, 'full_batch_inputs_checked':48}, **numerical)
    row['decision_log'] = []
    for end in range(63, n-1, 64):
        decision = copy.deepcopy(d)
        decision.update(step=end, physical_window_measured_step_range=[end-15, end])
        row['decision_log'].append(decision)
    row['decision_trajectory_sha256'] = v.digest([{k:decision[k] for k in
        ('step', 'current_counts', 'current_map', 'proposed_map', 'committed_map')} for decision in row['decision_log']])
    return row, {'default_configs':{'128':{'BM':64}}}, c, fixed


def test_generic_route_audit_checks_128_job_diagnostic_without_fake_padding():
    args = route_fixture(); v.routing(*args)
    assert args[0]['arm'] == 'fast'  # Normalization must not mutate recorded treatment.


@pytest.mark.parametrize('mutation', ('current_counts', 'physical_window', 'terminal_window', 'committed_map'))
def test_actual_count_or_map_corruption_rejected(mutation):
    row, base, c, fixed = route_fixture()
    if mutation == 'current_counts': row['decision_log'][0]['current_counts'][0] += 1
    if mutation == 'physical_window': row['decision_log'][0]['physical_slot_assignments_window'][0][0] += 1
    if mutation == 'terminal_window': row['terminal_physical_window'][0][0] += 1
    if mutation == 'committed_map': row['decision_log'][0]['committed_map'][0] = 80
    with pytest.raises(AssertionError): v.routing(row, base, c, fixed)


def proof_fixture():
    row, base, c, fixed = route_fixture()
    row.update(native_vllm_commit='pinned-commit', assignment_artifact={'sha256':'f'*64})
    changed = fixed.copy(); changed[0], changed[80] = changed[80], changed[0]
    proof = {k:copy.deepcopy(row[k]) for k in ('batch_tokens', 'native_vllm_commit', 'capture_sha256',
        'replay_sha256', 'source_prompt_ids', 'assignment_artifact', 'logical_routing_sha256',
        'resolved_kernel_config', 'source_weights_sha256')}
    proof.update(gpu_noop_proof_schema_version=1, passed=True, diagnostic_only=True, evidence_eligible=False,
        source_restore_verified=True, native_symbols_restored=True, no_pending_async_work=True,
        numerical_raw_indices=[16, 47], proof_source_sha256=v.file_hash(ROOT/'arrival_fix/proof.py'),
        source_map_sha256=v.array_hash(fixed), changed_map_sha256=v.array_hash(changed),
        source_weight_hashes_by_rank=row['source_weight_rank_sha256'],
        restored_weight_hashes_by_rank=row['source_weight_rank_sha256'],
        restored_weights_sha256=row['source_weights_sha256'])
    guard = fast_row()['noop_guard_ranks'][0]
    guard.update(local_unchanged_calls=2, unanimously_unchanged_calls=2, movement_skips=2,
        commit_skips=2, original_movement_calls=2, original_commit_calls=2, fallback_calls=2)
    proof['guard_counters_by_rank'] = [guard, copy.deepcopy(guard)]
    proof['steps'] = []
    for name, mapping, skip in [('unchanged_before', fixed, True), ('changed_swap_0_80', changed, False),
                               ('restore_original', fixed, False), ('unchanged_after_restore', fixed, True)]:
        proof['steps'].append(dict(name=name, expected_skip=skip, map_sha256=v.array_hash(mapping),
            exact_checkpoint_weights_verified=True, inverse_map_and_replica_count_verified=True,
            transfer_payload_by_rank=[dict(rank=r, bytes=0 if skip else 123, messages=0 if skip else 1) for r in (0,1)],
            actual_transfer_bytes=0 if skip else 246, actual_transfer_messages=0 if skip else 2,
            correctness_pass=True, correctness_atol=.02, correctness_rtol=.03, full_batch_inputs_checked=2,
            correctness_max_abs_error=.001, full_batch_max_abs_difference=0.))
    proof['total_transfer_bytes'] = 492
    return proof, row


def test_gpu_proof_requires_two_real_movements_and_exact_restore():
    proof, row = proof_fixture()
    result = v.verify_proof(proof, row, ROOT)
    assert result['unchanged_skips'] == result['real_native_movements'] == 2
    assert result['actual_transfer_bytes'] == 492
    proof['steps'][1]['transfer_payload_by_rank'][1]['bytes'] += 1
    with pytest.raises(AssertionError, match='payload conservation'):
        v.verify_proof(proof, row, ROOT)


def test_proof_rejects_zero_byte_fake_changed_map_path():
    proof, row = proof_fixture()
    proof['steps'][1]['actual_transfer_bytes'] = proof['steps'][1]['actual_transfer_messages'] = 0
    for rank in proof['steps'][1]['transfer_payload_by_rank']:
        rank.update(bytes=0, messages=0)
    with pytest.raises(AssertionError, match='real movement'):
        v.verify_proof(proof, row, ROOT)


def test_full_calibration_audit_integrates_clocks_routes_order_and_guard(monkeypatch):
    """Source/file I/O is stubbed; all per-row arithmetic and route checks run."""
    from types import SimpleNamespace
    row0, base, c, fixed = route_fixture(256)
    base.update(default_configs={'128':{'BM':64}, '1024':{'BM':64}}, execution_source_sha256={})
    base['environment'] = {key:'fixture-pin' for key in ('vllm_commit', 'legacy_harness_sha256',
        'followup_sha256', 'optimization_sha256', 'resident_dispatch_sha256', 'dispatch_backend', 'eplb_mode')}
    plan = final_plan(); plan.update(phase='calibration', arrival_fix_source_sha256='fixture-source', freeze={'sha256':'fixture-plan'})
    monkeypatch.setattr(v, 'plan_identity', lambda p, repo:(base, 'fixture-plan', 'fixture-base'))
    monkeypatch.setattr(v.old, 'route_state', lambda *args:c)
    monkeypatch.setattr(v.old, 'load_route_oracle', lambda repo:SimpleNamespace(FIXED_MAP=fixed,
        verify_cell_artifact=lambda *args:{'fixture':'no-file-I/O'}))
    c.update(arrays={}, artifact_meta={})
    rows = []
    for j, (state, key, order, schedule, block) in enumerate(v.execution_inventory(plan)):
        row = copy.deepcopy(row0)
        row.update(dict(zip(v.STATE, state))); row.update(dict(zip(v.KEY, key)))
        row.update(hardware_row())
        epoch = 100000*j+1000
        starts = [epoch+1+i*10 for i in range(256)]
        finishes = [s+9 for s in starts]
        row.update(arrival_fix_schema_version=1, phase='calibration', config_hash='fixture-plan',
            base_config_hash='fixture-base', arrival_fix_source_sha256='fixture-source', status='complete',
            policy='SKIP', evidence_eligible=False, diagnostic_only=False, guard_symbols_restored=True,
            normal_completion_verified=True, resident_state_verified=True, rank_assignment_agreement=True,
            migration_bytes=0, migration_sent_messages=0, terminal_proposals_suppressed=1,
            measurement_setup_revision=2, gc_enabled_during_measurement=True,
            native_vllm_commit='fixture-pin', resident_dispatch_file_sha256={}, arm_order=order,
            arrival_seed=None, arrival_schedule=None, arrival_offsets_ns=[0]*256, horizon_ns=1,
            schedule_name='saturated', epoch_monotonic_ns=epoch,
            coordinator_admit_ns=[s-1 for s in starts], coordinator_start_ns=starts,
            coordinator_finish_ns=finishes, observation_end_ns=finishes[-1]+1,
            per_rank_start_ns=[[s+1 for s in starts]]*2, per_rank_gpu_complete_ns=[[f-1 for f in finishes]]*2,
            noop_guard_ranks=[], allocation_id='synthetic-test-only',
            assignment_artifact={'sha256':f'{j:064x}'})
        row.update({key:base['environment'][key] for key in base['environment'] if key != 'vllm_commit'})
        row['record_id'] = v.digest({key:row[key] for key in ('config_hash', 'phase', 'arrival_block_id',
            'block_id', 'regime', 'batch_tokens', 'arm', 'repetition', 'schedule_name')})
        row['queue_metrics'] = v.old.timing_metrics(row)
        row['observed_window_spans'] = v.spans(row)
        row['calibration_capacity'] = v.old.capacity(row)
        rows.append(row)
    report = v.audit(plan, rows, ROOT, ROOT, [])
    assert report['pass'] and report['records'] == 8 and report['jobs_checked'] == 2048
    assert report['timed_native_windows_checked'] == 32
    swapped = rows[:]; swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(AssertionError, match='execution order'):
        v.audit(plan, swapped, ROOT, ROOT, [])
    corrupted = copy.deepcopy(rows)
    corrupted[0]['per_rank_gpu_complete_ns'][1][0] = corrupted[0]['coordinator_finish_ns'][0]+1
    with pytest.raises(AssertionError, match='envelope'):
        v.audit(plan, corrupted, ROOT, ROOT, [])

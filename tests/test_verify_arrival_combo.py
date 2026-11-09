"""Synthetic verifier tests; none of these fixtures is experimental evidence."""
from collections import Counter
import copy
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('independent_combo_verifier', ROOT/'scripts/verify_arrival_combo.py')
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)


def plan_fixture(phase='final'):
    matrix = dict(arrival_blocks=v.expected_blocks(), batches=[128, 1024], patterns=['spread', 'cluster'],
                  rhos=[.5, .85], reps=2, arms=list(v.ARMS))
    if phase == 'pilot':
        matrix.update(arrival_blocks=[dict(id='pilot-old-calendar', seed=0, block_id='opt-pilot-01', regime='domain1')],
                      patterns=['constant', 'burst8'], rhos=[.85])
    p = dict(phase=phase, order_seed=471431, matrix=matrix)
    p['arm_orders'] = v.reconstruct_orders(matrix, p['order_seed'])
    p['schedules'] = {f"{b['id']}/B{B}":[dict(rho=rho, pattern=pattern)
        for rho, pattern in itertools.product(matrix['rhos'], matrix['patterns'])]
        for b in matrix['arrival_blocks'] for B in matrix['batches']}
    return p


def test_calendar_seeds_are_new_and_balanced_over_fixed_payload_states():
    blocks = v.expected_blocks()
    assert [b['seed'] for b in blocks] == list(range(2026110911, 2026110919))
    assert not {b['seed'] for b in blocks}.intersection(range(2026110901, 2026110909))
    assert set(Counter((b['block_id'], b['regime']) for b in blocks).values()) == {2}


@pytest.mark.parametrize('phase,count', [('pilot',24), ('calibration',8), ('final',384)])
def test_declared_execution_counts(phase, count):
    stream = v.execution_inventory(plan_fixture(phase))
    assert len(stream) == count
    # Calibration identities also include source state, unlike its generic block label.
    assert len({(x[0], x[1]) for x in stream}) == count


def test_three_arm_reversal_and_position_counterbalancing():
    p = plan_fixture()
    orders = p['arm_orders']
    assert len(orders) == 64
    positions = {arm:[0,0,0] for arm in v.ARMS}
    precedence = Counter()
    for forward, reverse in orders.values():
        assert forward == reverse[::-1]
        assert set(forward) == set(v.ARMS)
        for order in (forward, reverse):
            for arm in v.ARMS: positions[arm][order.index(arm)] += 1
            for first, second in itertools.permutations(v.ARMS, 2):
                precedence[first, second] += order.index(first) < order.index(second)
    assert set(precedence.values()) == {64}
    assert all(max(counts)-min(counts) <= 2 for counts in positions.values())
    assert any(len(set(counts)) > 1 for counts in positions.values())  # Not exact position balance.
    for rho in (.5,.85):
        rotations = Counter(tuple(value[0]) for key,value in orders.items() if key.endswith(f'rho{rho}'))
        assert sorted(rotations.values()) == [10,11,11]


def test_unknown_arm_rejected():
    with pytest.raises(AssertionError, match='Known composition arm'):
        v.dispatch_for('online_only')


def route_fixture():
    oracle = v.old.load_route_oracle(ROOT)
    fixed = oracle.FIXED_MAP
    ids = np.tile(np.arange(8, dtype=np.int64), (128, 1))
    logical = np.stack([ids]*48)
    counts, hashes, diagnostics = {}, {}, {}
    for arm in ('native', 'online'):
        slots, counter, optimum = oracle.assignment(ids, arm)
        counts[arm] = np.stack([counter]*48)
        hashes[arm] = hashlib.sha256(slots.astype(np.int64).tobytes()*48).hexdigest()
        diagnostics[arm] = [oracle.metrics(counter, optimum, 64) for _ in range(48)]
    provenance = dict(capture_sha256='a'*64, replay_sha256='b'*64, source_prompt_ids=['synthetic-fixture'])
    sequence = [16+i%32 for i in range(256)]
    c = dict(sequence=sequence, logical=logical, provenance=provenance,
             counts=counts, hashes=hashes, diagnostics=diagnostics)
    base = {'default_configs':{'128':{'BLOCK_SIZE_M':64}}}
    rank_hashes = ['d'*64, 'e'*64]
    numerical = dict(correctness_pass=True, correctness_atol=.02, correctness_rtol=.03,
        full_batch_inputs_checked=2, correctness_max_abs_error=.001, full_batch_max_abs_difference=.001)
    adversarial = dict(passed=True, rank_agreement=True, native_recording_state_unchanged=True,
        cases_checked=9, recording_modes_per_case=2, gpu_invocations_checked=18, batch_tokens=128,
        cases=[{'synthetic_case':i} for i in range(9)])
    adversarial['evidence_sha256'] = v.digest(adversarial['cases'])
    rows = []
    for arm in v.ARMS:
        dispatch = v.dispatch_for(arm)
        decisions = []
        for end in (63,127,191):
            raw = sequence[end-15:end+1]; physical = counts[dispatch][raw]
            decisions.append(dict(step=end, current_map=fixed.tolist(), proposed_map=fixed.tolist(),
                committed_map=fixed.tolist(), committed_map_sha256=v.array_hash(fixed),
                current_counts=(np.bincount(logical[raw].ravel(), minlength=128)*2).tolist(),
                physical_slot_assignments_window=physical.tolist(),
                physical_slot_assignments_sha256=v.array_hash(physical, np.int32),
                physical_owner_assignments_per_step=physical.reshape(16,2,80).sum(axis=2).tolist(),
                allowed=False, reason='always_skip', commit_completed=True, transfer_sent_bytes=0,
                transfer_sent_bytes_by_rank=[0,0], physical_window_next_write=0,
                physical_counter_input_multiplicity=2, physical_window_measured_step_range=[end-15,end],
                candidate_changed_slots=0))
        guard = dict(mode='fast', diagnostic_timing=False, pending_pair=False, movement_calls=4,
            commit_calls=4, agreement_calls=4, local_unchanged_calls=4, unanimously_unchanged_calls=4,
            movement_skips=4, commit_skips=4, original_movement_calls=0, original_commit_calls=0,
            fallback_calls=0, diagnostics=[])
        row = dict(arm=arm, n_jobs=256, batch_tokens=128, **provenance,
            logical_routing_sha256=v.array_hash(logical), assignment_sha256=hashes[dispatch],
            source_map=fixed.tolist(), final_map=fixed.tolist(), resolved_kernel_config=base['default_configs']['128'],
            full_unique_preflight_gpu_routes_checked=48, warmup_raw_indices=list(range(16)),
            conditioning_raw_indices=[16+i%32 for i in range(64)], measured_raw_indices=sequence,
            assignment_metrics=[diagnostics[dispatch][i] for i in sequence], decision_log=decisions,
            terminal_physical_window=counts[dispatch][sequence[-16:]].tolist(),
            source_weight_rank_sha256=rank_hashes,
            source_weights_sha256=hashlib.sha256(json.dumps(rank_hashes).encode()).hexdigest(),
            preflight_correctness={**numerical, 'full_batch_inputs_checked':48}, **numerical,
            adversarial_checks=copy.deepcopy(adversarial), dispatch_mode=dispatch, noop_enabled=arm!='native',
            assignment_selection_cost_included=True, online_residency_metadata_verified=arm=='combined',
            mapper_calls_measured=256 if arm=='combined' else 0,
            noop_guard_ranks=[] if arm=='native' else [guard,copy.deepcopy(guard)],
            arrival_clock_contract={'fixture':'clock'}, rank_gpu_uuids=v.fix.UUIDS)
        row['decision_trajectory_sha256'] = v.digest([{k:d[k] for k in
            ('step','current_counts','current_map','proposed_map','committed_map')} for d in decisions])
        rows.append(row)
    return rows, base, c, fixed


def test_real_distinct_native_and_online_oracles_both_pass():
    rows, base, c, fixed = route_fixture()
    for row in rows: v.routing(row, base, c, fixed)
    v.compare_invariants(rows)
    assert rows[0]['assignment_sha256'] == rows[1]['assignment_sha256']
    assert rows[0]['assignment_sha256'] != rows[2]['assignment_sha256']
    assert rows[0]['decision_trajectory_sha256'] == rows[2]['decision_trajectory_sha256']
    assert rows[2]['arm'] == 'combined'


@pytest.mark.parametrize('field,value', [('dispatch_mode','native'), ('mapper_calls_measured',255),
    ('assignment_selection_cost_included',False), ('online_residency_metadata_verified',False)])
def test_combined_dispatch_execution_contract_is_charged_and_complete(field,value):
    rows, base, c, fixed = route_fixture(); rows[2][field] = value
    with pytest.raises(AssertionError): v.routing(rows[2],base,c,fixed)


def test_combined_cannot_borrow_native_physical_window():
    rows, base, c, fixed = route_fixture()
    rows[2]['decision_log'][0]['physical_slot_assignments_window'] = rows[0]['decision_log'][0]['physical_slot_assignments_window']
    with pytest.raises(AssertionError, match='Timed physical window'):
        v.routing(rows[2],base,c,fixed)


def test_shortcut_cannot_borrow_online_assignment_hash():
    rows, base, c, fixed = route_fixture(); rows[1]['assignment_sha256'] = rows[2]['assignment_sha256']
    with pytest.raises(AssertionError): v.routing(rows[1],base,c,fixed)


def test_same_physical_history_not_required_but_common_logical_history_is():
    rows, base, c, fixed = route_fixture(); v.compare_invariants(rows)
    rows[2]['decision_trajectory_sha256'] = 'different'
    with pytest.raises(AssertionError, match='logical/resident invariant'):
        v.compare_invariants(rows)


def test_shortcut_guard_missing_or_native_baseline_wrapped_is_rejected():
    rows, base, c, fixed = route_fixture()
    rows[0]['noop_guard_ranks'] = copy.deepcopy(rows[1]['noop_guard_ranks'])
    with pytest.raises(AssertionError, match='Unwrapped'): v.routing(rows[0],base,c,fixed)
    rows, base, c, fixed = route_fixture(); rows[1]['noop_guard_ranks'] = []
    with pytest.raises(AssertionError, match='Two rank'): v.routing(rows[1],base,c,fixed)


def proof_fixture():
    rows, base, c, fixed = route_fixture()
    for row in rows:
        row.update(native_vllm_commit='fixture-pin', assignment_artifact={'sha256':'f'*64})
    row = rows[0]
    proof = {k:copy.deepcopy(row[k]) for k in ('batch_tokens','native_vllm_commit','capture_sha256',
        'replay_sha256','source_prompt_ids','assignment_artifact','logical_routing_sha256',
        'resolved_kernel_config','source_weights_sha256')}
    proof.update(gpu_combo_proof_schema_version=1, passed=True, correctness_pass=True,
        diagnostic_only=True, evidence_eligible=False, common_logical_candidate_trajectory_verified=True,
        arm_specific_physical_windows_verified=True, native_symbols_restored=True,
        reference_raw_indices=list(range(48)), fp32_raw_indices=[16,47], proof_steps_per_arm=128,
        expected_decision_steps=[63], expected_noop_pairs_per_shortcut_arm=2,
        source_map_sha256=v.array_hash(fixed), proof_source_sha256=v.file_hash(ROOT/'arrival_combo/proof.py'))
    proof['arms'] = []
    for r in rows:
        a = {k:copy.deepcopy(r[k]) for k in ('arm','source_map','final_map','source_weights_sha256',
            'source_weight_rank_sha256','assignment_sha256','online_residency_metadata_verified')}
        guards = copy.deepcopy(r['noop_guard_ranks'])
        for guard in guards:
            for key in ('movement_calls','commit_calls','agreement_calls','local_unchanged_calls',
                        'unanimously_unchanged_calls','movement_skips','commit_skips'):guard[key]=2
        a.update(routing_arm=r['dispatch_mode'], passed=True, correctness_pass=True,
            normal_completion_verified=True, exact_checkpoint_weights_verified=True,
            inverse_map_and_replica_count_verified=True, native_symbols_restored=True,
            no_pending_async_work=True, final_weights_sha256=r['source_weights_sha256'],
            final_weight_rank_sha256=r['source_weight_rank_sha256'], migration_bytes=0,
            migration_sent_messages=0, terminal_proposals_suppressed=1,
            measured_raw_indices=c['sequence'][:128], guard_counters_by_rank=guards,
            mapper_calls_measured=128 if r['arm']=='combined' else 0,
            preflight_correctness=copy.deepcopy(r['preflight_correctness']),
            postflight_correctness=copy.deepcopy(r['preflight_correctness']),
            adversarial_checks=copy.deepcopy(r['adversarial_checks']) if r['arm']=='combined' else None,
            decision_log=[copy.deepcopy(r['decision_log'][0])],
            terminal_physical_window=c['counts'][r['dispatch_mode']][c['sequence'][112:128]].tolist())
        a['completion_by_rank'] = [dict(guard=guards[i] if guards else None,
            mapper_calls_measured=a['mapper_calls_measured'],sent_bytes=0,sent_messages=0) for i in (0,1)]
        proof['arms'].append(a)
    proof['logical_candidate_trajectory_sha256'] = v.digest([{k:d[k] for k in
        ('step','allowed','reason','current_counts','current_map','proposed_map','committed_map')}
        for d in proof['arms'][0]['decision_log']])
    return proof,row,base,c


def test_complete_three_arm_composition_proof():
    proof,row,base,c=proof_fixture()
    result=v.verify_proof(proof,row,base,c,ROOT)
    assert result['verified'] and result['arms']==3 and result['steps_per_arm']==128


@pytest.mark.parametrize('mutation', ('online_window','online_calls','rank_ack','postflight_coverage','restored_weights'))
def test_composition_proof_rejects_partial_or_misattributed_evidence(mutation):
    proof,row,base,c=proof_fixture();combined=proof['arms'][2]
    if mutation=='online_window':combined['terminal_physical_window']=proof['arms'][0]['terminal_physical_window']
    if mutation=='online_calls':combined['mapper_calls_measured']=127
    if mutation=='rank_ack':combined['completion_by_rank'].pop()
    if mutation=='postflight_coverage':combined['postflight_correctness']['full_batch_inputs_checked']=2
    if mutation=='restored_weights':combined['final_weights_sha256']='changed'
    with pytest.raises(AssertionError):v.verify_proof(proof,row,base,c,ROOT)

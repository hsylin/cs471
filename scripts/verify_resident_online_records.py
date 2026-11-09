#!/usr/bin/env python3
"""Independent CPU audit of resident dispatch routes and native observations.

Uses only the standard library and NumPy; imports no production experiment code.
Reconstructs all unique assignments from authenticated captured logical routes,
checks the saved GPU preflight evidence, and checks actual timed observation
windows. It does not reproduce GPU arithmetic or infer performance significance.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import random

import numpy as np

ARMS = ('native', 'online')
IDENTITY = ('block_id', 'regime', 'batch_tokens', 'repetition', 'arm')
FIXED_MAP = np.concatenate((np.arange(128), np.arange(32))).astype(np.int64)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def array_digest(value, dtype=None):
    return hashlib.sha256(np.ascontiguousarray(value, dtype=dtype).tobytes()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def equal(actual, expected, message):
    require(np.array_equal(actual, expected), message)


def authenticated_npz(path, expected_sha):
    require(file_digest(path) == expected_sha, f'File checksum: {path}')
    with np.load(path, allow_pickle=False) as saved:
        require(len(saved.files) == len(set(saved.files)), 'Repeated NPZ entry')
        metadata = json.loads(str(saved['metadata']))
        arrays = {key: saved[key].copy() for key in saved.files
                  if key not in ('metadata', 'content_sha256')}
        if 'content_sha256' in saved:
            h = hashlib.sha256(canonical(metadata).encode())
            for key, value in sorted(arrays.items()):
                a = np.ascontiguousarray(value)
                h.update(canonical([key, a.dtype.str, list(a.shape)]).encode())
                h.update(a.tobytes())
            require(h.hexdigest() == str(saved['content_sha256']), f'Content checksum: {path}')
    return metadata, arrays


def assignment(ids, arm):
    """Independent scalar scheduling oracle for the frozen 128+32 residency.

    The active-copy optimum is found by enumerating ALL possible shared-expert
    cardinalities, rather than using the production closed-form capacity bound.
    Online tie breaking then follows the prespecified greedy policy, not a
    runtime-optimal claim. Inactive experts cannot affect any returned route.
    """
    ids = np.asarray(ids, dtype=np.int64)
    require(ids.ndim == 2 and ids.shape[1] == 8 and np.all((0 <= ids) & (ids < 128)),
            'Logical route shape/range')
    demand = np.bincount(ids.ravel(), minlength=128)
    forced = [list(range(32, 80)), list(range(80, 128))]
    fixed_active = [sum(demand[e] > 0 for e in experts) for experts in forced]
    shared = [e for e in range(32) if demand[e]]
    optimum = min(max(fixed_active[0] + count, fixed_active[1] + len(shared) - count)
                  for count in range(len(shared) + 1))
    if arm == 'native':
        # Two resident copies: the full unsigned Knuth hash reduces to its
        # parity. Reconstruct the general hash nevertheless to state semantics.
        slots = ids.copy()
        copy = np.array([((row * 2654435769) % (2**32)) % 2
                         for row in range(ids.shape[0])], dtype=np.int64)
        slots += ((ids < 32) * copy[:, None]) * 128
    else:
        require(arm == 'online', 'Unknown dispatch arm')
        homes = np.arange(128, dtype=np.int64)
        occupied = fixed_active.copy()
        loads = [sum(int(demand[e]) for e in experts) for experts in forced]
        for expert in sorted(shared, key=lambda e: (-int(demand[e]), e)):
            eligible = [rank for rank in (0, 1) if occupied[rank] < optimum]
            require(bool(eligible), 'No feasible active-copy capacity')
            owner = min(eligible, key=lambda rank: (loads[rank], rank))
            homes[expert] = expert + (128 if owner else 0)
            occupied[owner] += 1
            loads[owner] += int(demand[expert])
        slots = homes[ids]
    equal(FIXED_MAP[slots], ids, 'Physical residency changes logical route')
    counts = np.bincount(slots.ravel(), minlength=160)
    recovered = np.zeros(128, dtype=np.int64)
    np.add.at(recovered, FIXED_MAP, counts)
    equal(recovered, demand, 'Physical/logical assignment conservation')
    if arm == 'online':
        require(int((counts.reshape(2, 80) > 0).sum(axis=1).max()) == optimum,
                'Online active-copy cap differs from enumerated optimum')
        require(all(len(np.unique(slots[ids == e])) == 1 for e in shared),
                'Online used more than one home per shared expert')
    return slots, counts.astype(np.int32), int(optimum)


def metrics(counts, optimum, block_m):
    rank = counts.reshape(2, 80)
    tiles = ((rank + block_m - 1) // block_m).sum(axis=1)
    return dict(per_rank_tokens=rank.sum(axis=1).tolist(),
                per_rank_active_copies=(rank > 0).sum(axis=1).tolist(),
                per_rank_m_tiles=tiles.tolist(), per_rank_padded_rows=(tiles * block_m).tolist(),
                minimum_possible_max_rank_active_copies=optimum)


def positions_for(manifest, state, bank, capture_root):
    block, regime, batch = state
    if manifest['phase'] == 'validation':
        domain = int(regime.removeprefix('domain'))
        key = json.dumps([block, domain, batch], separators=(',', ':'))
        seed = (10471 + int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)) % 2**64
        pool = np.flatnonzero(bank['domain_ids'] == domain)
        generator = np.random.default_rng(seed)
        indices = np.stack([generator.choice(pool, batch, replace=False) for _ in range(48)])
        return indices, array_digest(indices), [16 + i % 32 for i in range(256)], [16, 47]
    entry = next(t for t in manifest['tapes'] if tuple(t[k] for k in IDENTITY[:3]) == state)
    tape, data = authenticated_npz(capture_root / entry['replay_path'], entry['replay_sha256'])
    contract = dict(block_id=block, regime=regime, batch_tokens=batch, split='final',
                    seed=20471, num_steps=544, warmup_steps=32, duration_steps=16,
                    repeat_steps=1024, phase_steps=112 if regime == 'switching' else 120,
                    capture_sha256=entry['capture_sha256'], natural_temporal_demand=False)
    require(all(tape.get(k) == v for k, v in contract.items()), 'Frozen replay metadata')
    require(set(data) == {'indices'}, 'Unexpected replay arrays')
    domains = np.zeros(544, dtype=np.int64)
    if regime == 'transient':
        domains[152:168] = 1
    elif regime == 'persistent':
        domains[152:] = 1
    elif regime == 'switching':
        for start in range(144, 544, 32):
            domains[start:start+16] = 1
    else:
        require(regime == 'stable', 'Unknown constructed regime')
    generator = np.random.default_rng(20471)
    indices = np.stack([generator.choice(2048, batch, replace=False) for _ in range(544)])
    indices += domains[:, None] * 2048
    require(data['indices'].dtype == np.int64, 'Replay index dtype')
    equal(data['indices'], indices, 'Replay differs from source-independent schedule')
    equal(tape['domain_schedule'], domains, 'Recorded domain schedule')
    return indices, entry['replay_sha256'], list(range(32, 544)), [32, 160, 288, 543]


def verify_cell_artifact(row, roots, expected, metadata):
    artifact = row['assignment_artifact']
    name = Path(artifact['path']).name
    candidates = list(dict.fromkeys(root / name for root in roots if (root / name).exists()))
    require(bool(candidates), f'Missing input artifact: {name}')
    matching = [p for p in candidates if file_digest(p) == artifact['sha256']]
    require(bool(matching), f'Input artifact checksum: {name}')
    actual_meta, actual_arrays = authenticated_npz(matching[0], artifact['sha256'])
    require(set(actual_arrays) == set(expected), f'Input artifact schema: {name}')
    for key, value in expected.items():
        require(actual_arrays[key].dtype == value.dtype, f'{name}/{key}: dtype')
        equal(actual_arrays[key], value, f'{name}/{key}: values')
    require(all(actual_meta.get(k) == v for k, v in metadata.items()), f'{name}: input provenance')
    return {'path': str(matching[0]), 'sha256': artifact['sha256']}


def verify_record(row, manifest, state, indices, logical, sequence, samples, provenance,
                  counts, route_hashes, diagnostic):
    arm = row['arm']; label = '/'.join(map(str, (*state, row['repetition'], arm)))
    config_sha = digest({k: v for k, v in manifest.items() if k != 'freeze'})
    flags = ('evidence_eligible', 'assignment_selection_cost_included', 'source_restore_verified',
             'expected_arm_weights_verified', 'commit_completed', 'logical_routing_preserved',
             'rank_assignment_agreement', 'timing_events_initialized_before_timer', 'gc_enabled_during_measurement')
    require(all(row.get(k) is True for k in flags), label + ': required attestation')
    expected = dict(config_hash=config_sha, phase=manifest['phase'], split=manifest['split'],
        status='complete', policy='SKIP', measurement_kind='fully_charged_resident_dispatch',
        steps_completed=len(sequence), source_map=FIXED_MAP.tolist(), final_map=FIXED_MAP.tolist(),
        source_map_sha256=array_digest(FIXED_MAP), assignment_sha256=route_hashes[arm],
        logical_routing_sha256=array_digest(logical), terminal_proposals_suppressed=1,
        migration_bytes=0, migration_sent_messages=0, measurement_setup_revision=2,
        full_policy_conditioning_steps=64, full_policy_conditioning_calls=1,
        full_unique_gpu_routes_checked=len(indices), gpu_mapping_and_record_checks=2*len(indices),
        physical_window_checks=len(sequence)//64, world_size=2, dtype='bfloat16',
        timing_scope=manifest['execution']['timing_scope'],
        resolved_kernel_config=manifest['default_configs'][str(state[2])],
        online_algorithm=manifest['online_algorithm'] if arm == 'online' else 'native_knuth_hash',
        native_vllm_commit=manifest['environment']['vllm_commit'],
        resident_dispatch_file_sha256=manifest['execution_source_sha256'], **provenance)
    for key in ('legacy_harness_sha256', 'followup_sha256', 'optimization_sha256',
                'resident_dispatch_sha256', 'dispatch_backend', 'eplb_mode'):
        expected[key] = manifest['environment'][key]
    require(all(row.get(k) == v for k, v in expected.items()), label + ': record contract/provenance')
    require(row['record_id'] == digest({k: row[k] for k in ('config_hash', *IDENTITY)}), label + ': record ID')
    ranks = row['per_rank_elapsed_seconds']
    require(len(ranks) == 2 and all(math.isfinite(t) and t > 0 for t in ranks), label + ': rank elapsed')
    require(row['elapsed_seconds'] == max(ranks), label + ': completed max-rank elapsed')
    event_times = np.asarray(row['step_seconds_by_rank'])
    require(event_times.shape == (2, len(sequence)) and np.isfinite(event_times).all()
            and (event_times > 0).all(), label + ': per-step events')
    expected_conditioning = [16+i%32 if manifest['phase'] == 'validation' else i%32 for i in range(64)]
    equal(row['conditioning_raw_indices'], expected_conditioning, label + ': conditioning inputs')
    for checks, n in ((row['preflight_correctness'], 48 if manifest['phase'] == 'validation' else 4),
                      (row, len(samples))):
        require(checks['correctness_pass'] is True and checks['correctness_atol'] == .02
                and checks['correctness_rtol'] == .03 and checks['full_batch_inputs_checked'] == n,
                label + ': numerical check coverage/tolerance')
        require(all(math.isfinite(checks[k]) and checks[k] >= 0
                    for k in ('correctness_max_abs_error', 'full_batch_max_abs_difference')),
                label + ': finite nonnegative numerical error')
    rank_hashes = row['source_weight_rank_sha256']
    require(len(rank_hashes) == 2 and all(isinstance(h, str) and len(h) == 64 for h in rank_hashes),
            label + ': resident tensor rank checksums')
    require(hashlib.sha256(json.dumps(rank_hashes).encode()).hexdigest() == row['source_weights_sha256'],
            label + ': aggregate resident tensor checksum')
    require(row['assignment_metrics'] == [diagnostic[arm][i] for i in sequence], label + ': per-step diagnostics')
    decisions = row['decision_log']
    equal([d['step'] for d in decisions], list(range(63, len(sequence)-1, 64)), label + ': cadence')
    for decision in decisions:
        step = decision['step']; raw = sequence[step-15:step+1]
        observed = counts[arm][raw]
        equal(decision['physical_slot_assignments_window'], observed, label + ': timed physical window')
        require(decision['physical_slot_assignments_sha256'] == array_digest(observed, np.int32), label + ': window checksum')
        equal(decision['current_counts'], np.bincount(logical[raw].ravel(), minlength=128)*2,
              label + ': native replicated-input logical aggregation')
        equal(decision['physical_owner_assignments_per_step'], observed.reshape(16, 2, 80).sum(axis=2),
              label + ': rank-owned work')
        require(decision['physical_window_measured_step_range'] == [step-15, step]
                and decision['physical_counter_input_multiplicity'] == 2
                and decision['physical_window_next_write'] == 0, label + ': window range/multiplicity')
        equal(decision['current_map'], FIXED_MAP, label + ': current map')
        equal(decision['committed_map'], FIXED_MAP, label + ': committed map')
        require(decision['committed_map_sha256'] == array_digest(FIXED_MAP)
                and decision['allowed'] is False and decision['reason'] == 'always_skip'
                and decision['commit_completed'] is True and decision['transfer_sent_bytes'] == 0
                and decision['transfer_sent_bytes_by_rank'] == [0, 0], label + ': no migration')
        proposed = np.asarray(decision['proposed_map'])
        require(proposed.shape == (160,) and set(proposed) == set(range(128))
                and decision['candidate_changed_slots'] == int((proposed != FIXED_MAP).sum()),
                label + ': proposed map validity')
    trajectory = digest([{k: d[k] for k in ('step', 'current_counts', 'current_map', 'proposed_map', 'committed_map')}
                         for d in decisions])
    require(row['decision_trajectory_sha256'] == row['logical_trajectory_sha256'] == trajectory,
            label + ': decision trajectory checksum')
    equal(row['terminal_physical_window'], counts[arm][sequence[-16:]], label + ': terminal window')
    adversarial = row['adversarial_checks']
    require(all(adversarial.get(k) is True for k in ('passed', 'rank_agreement', 'native_recording_state_unchanged'))
            and adversarial['batch_tokens'] == state[2] and adversarial['cases_checked'] == 9
            and adversarial['recording_modes_per_case'] == 2 and adversarial['gpu_invocations_checked'] == 18
            and adversarial['evidence_sha256'] == digest(adversarial['cases']), label + ': adversarial attestations')


def audit(manifest, rows, capture_root, inputs_roots, repetition=None):
    config_sha = digest({k: v for k, v in manifest.items() if k != 'freeze'})
    require(config_sha == manifest['freeze']['sha256'], 'Frozen manifest checksum')
    require(manifest['phase'] in ('validation', 'final'), 'Unsupported phase')
    matrix = manifest['matrix']
    require(matrix['repetitions'] == 2 and tuple(matrix['arms']) == ARMS, 'Frozen arm/repetition matrix')
    reps = list(range(2)) if repetition is None else [repetition]
    states = list(itertools.product(matrix['block_ids'], matrix['regimes'], matrix['batch_tokens']))
    expected_keys = {(*state, rep, arm) for state in states for rep in reps for arm in ARMS}
    index = {tuple(row[k] for k in IDENTITY): row for row in rows}
    require(len(index) == len(rows) and set(index) == expected_keys, 'Incomplete/duplicate/extra record matrix')
    source_docs = {b['block_id']: b['source_document_ids'] for b in manifest['blocks']}
    all_source_ids = list(itertools.chain.from_iterable(source_docs.values()))
    require(len(all_source_ids) == len(set(all_source_ids))
            and not set(all_source_ids).intersection(manifest.get('excluded_source_document_ids', [])), 'Source block overlap')
    banks = {}; artifacts = {}; artifact_states = {}; recomputed_routes = 0; counted_windows = 0
    for state in states:
        block, regime, batch = state
        if block not in banks:
            collection = manifest['banks'] if manifest['phase'] == 'validation' else manifest['tapes']
            entry = next(e for e in collection if e['block_id'] == block)
            bank_meta, bank = authenticated_npz(capture_root / entry['capture_path'], entry['capture_sha256'])
            require(bank_meta['correctness_pass'] is True and bank_meta['block_id'] == block
                    and bank_meta['split'] == manifest['split'] and bank_meta['source_prompt_ids'] == source_docs[block]
                    and bank_meta['source_model'] == manifest['environment']['model_id']
                    and bank_meta['source_revision'] == manifest['environment']['model_revision']
                    and bank_meta['layer'] == 0, 'Bank checkpoint/source attestation')
            require(bank['expert_ids'].dtype == np.int64 and bank['expert_ids'].shape == (4096, 8)
                    and bank['hidden_states_bf16'].dtype == np.uint16 and bank['hidden_states_bf16'].shape == (4096, 2048)
                    and bank['combining_weights'].dtype == np.float32 and bank['combining_weights'].shape == (4096, 8), 'Bank tensor contract')
            require(np.all(np.diff(np.sort(bank['expert_ids'], axis=1), axis=1) > 0), 'Repeated top-k expert')
            require(all(all(p[k] is True for k in ('native_boundary_exact', 'native_router_exact',
                                                  'manual_router_exact', 'bf16_roundtrip_exact'))
                        for p in bank_meta['prompts']), 'Native capture attestation')
            banks[block] = (bank, entry['capture_sha256'])
        bank, capture_sha = banks[block]
        indices, replay_sha, sequence, samples = positions_for(manifest, state, bank, capture_root)
        logical = bank['expert_ids'][indices]
        counts, route_hashes, diagnostic, selected = {}, {}, {}, {}
        for arm in ARMS:
            counter = []; diag = []; sample_routes = []; h = hashlib.sha256()
            for raw, ids in enumerate(logical):
                slots, physical_counts, optimum = assignment(ids, arm)
                h.update(slots.astype(np.int64).tobytes())
                counter.append(physical_counts)
                diag.append(metrics(physical_counts, optimum, manifest['default_configs'][str(batch)]['BLOCK_SIZE_M']))
                if raw in samples:
                    sample_routes.append(slots.astype(np.uint8))
                recomputed_routes += slots.size
            counts[arm] = np.asarray(counter, dtype=np.int32)
            route_hashes[arm] = h.hexdigest(); diagnostic[arm] = diag
            selected[arm] = np.stack(sample_routes)
        provenance = dict(capture_sha256=capture_sha, replay_sha256=replay_sha, source_prompt_ids=source_docs[block])
        expected_arrays = dict(indices=indices, physical_to_logical_map=FIXED_MAP,
            native_physical_counts=counts['native'], online_physical_counts=counts['online'],
            native_sample_physical_ids=selected['native'], online_sample_physical_ids=selected['online'])
        artifact_meta = dict(**provenance, logical_routing_sha256=array_digest(logical),
                             full_physical_route_sha256=route_hashes, sample_raw_indices=samples)
        state_rows = [index[(*state, rep, arm)] for rep in reps for arm in ARMS]
        for row in state_rows:
            artifact = row['assignment_artifact']
            if artifact['sha256'] not in artifacts:
                artifacts[artifact['sha256']] = verify_cell_artifact(row, inputs_roots, expected_arrays, artifact_meta)
                artifact_states[artifact['sha256']] = state
            require(artifact_states[artifact['sha256']] == state, 'Input artifact reused for a different state')
            verify_record(row, manifest, state, indices, logical, sequence, samples, provenance,
                          counts, route_hashes, diagnostic)
            counted_windows += row['physical_window_checks']
        for key in ('source_weights_sha256', 'source_weight_rank_sha256', 'decision_trajectory_sha256',
                    'logical_trajectory_sha256', 'logical_routing_sha256', 'source_map_sha256'):
            require(len({canonical(r[key]) for r in state_rows}) == 1, f'{state}: paired {key} differs')
    require(len({row['source_weights_sha256'] for row in rows}) == 1, 'Initial resident weights changed between cells')
    random.Random(manifest['execution']['order_seed']).shuffle(states)
    for n, state in enumerate(states):
        for rep in reps:
            order = list(ARMS if n % 2 == rep else ARMS[::-1])
            pair = [index[(*state, rep, arm)] for arm in ARMS]
            require(pair[0]['allocation_id'] == pair[1]['allocation_id'], 'Paired arms used different allocations')
            for arm in ARMS:
                row = index[(*state, rep, arm)]
                require(row['arm_order'] == order and row['arm_position'] == order.index(arm), 'Frozen balanced arm order')
    allocations = {rep: sorted({str(row['allocation_id']) for row in rows if row['repetition'] == rep}) for rep in reps}
    if manifest['phase'] == 'final':
        require(all(len(value) == 1 for value in allocations.values()), 'Final repetition spans allocations')
        require(len({value[0] for value in allocations.values()}) == len(reps), 'Final repetitions share allocation')
    return dict(pass_=True, phase=manifest['phase'], manifest_sha256=config_sha, rows=len(rows),
        states=len(states), repetitions=reps, allocations=allocations, captured_banks=len(banks),
        unique_artifacts=len(artifacts), artifacts=list(artifacts.values()),
        reconstructed_physical_routes=recomputed_routes, checked_timed_observation_windows=counted_windows,
        checked_timed_step_windows_per_row=(len(sequence)//64)*16,
        unique_preflight_route_inputs_per_state=len(indices),
        full_batch_numerical_inputs_preflight=48 if manifest['phase'] == 'validation' else 4,
        full_batch_numerical_inputs_after_measurement=len(samples),
        maximum_recorded_fp32_abs_error=max(max(r['correctness_max_abs_error'], r['preflight_correctness']['correctness_max_abs_error']) for r in rows),
        maximum_recorded_native_abs_difference=max(max(r['full_batch_max_abs_difference'], r['preflight_correctness']['full_batch_max_abs_difference']) for r in rows),
        limitations=[
            'Independent CPU implementation within the same project team; not external laboratory reproduction.',
            'All unique physical routes/counts are GPU preflight artifacts. Timed route evidence covers native 16-step observation windows, not every timed output.',
            'GPU numerical outputs and live resident tensors were not saved; this audit verifies their recorded attestations and hash invariance, not their arithmetic independently.',
            'FP32 references cover 16 positions at each selected reference input, not all token positions.',
            'Native proposed maps are checked for eligibility and cross-arm invariance; the native planner itself is not independently reimplemented.',
            'This audit verifies correctness evidence, not latency confidence intervals or production-serving generalization.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--records', type=Path, nargs='+', required=True)
    parser.add_argument('--capture-root', type=Path, required=True)
    parser.add_argument('--inputs-root', type=Path, nargs='*', default=[])
    parser.add_argument('--repetition', type=int, choices=(0, 1))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        rows = []
        for path in args.records:
            opener = gzip.open if path.suffix == '.gz' else open
            with opener(path, 'rt') as handle:
                rows.extend(json.loads(line) for line in handle if line.strip())
        roots = [*args.inputs_root, *(p.parent/'inputs' for p in args.records)]
        result = audit(json.loads(args.manifest.read_text()), rows, args.capture_root, roots, args.repetition)
        result['pass'] = result.pop('pass_')
    except (AssertionError, KeyError, ValueError, TypeError, OSError) as error:
        result = {'pass': False, 'error': f'{type(error).__name__}: {error}'}
    result['verifier_sha256'] = file_digest(Path(__file__))
    result['records'] = [{'path': str(p), 'sha256': file_digest(p)} for p in args.records if p.exists()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'artifacts'}, indent=2))
    raise SystemExit(0 if result['pass'] else 1)


if __name__ == '__main__':
    main()

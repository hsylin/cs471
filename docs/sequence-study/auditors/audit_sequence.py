#!/usr/bin/env python3
"""Independent sequence execution/probe audit; no production module imports.

Reuses already independent clock, stock-state and compiled-binary primitives.
This is a CPU reconstruction of artifacts and recorded GPU attestations, not a
new GPU run or an inferential analysis of the descriptive supplement.
"""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import sys
import tarfile

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'results/resident-residual/auditors'))
import audit_residual_records as residual
import residual_stock_primitives as stock
import residual_preflight_primitives as preflight
import sequence_input_primitives as inputs_oracle

require, digest = stock.require, stock.sha
ARMS = ('stockoff', 'stockon32', 'fix')
INITIAL = np.r_[np.arange(128), np.arange(32)].astype(np.int64)
EXPERT_SHA = 'ee6b83b108a0460e6308e35447434165d0778a053aa39f0c6c8ed7f36a3ddbab'


def file_sha(path):
    return digest(Path(path).read_bytes())


def raw_hash(value, dtype='<i8'):
    return digest(np.ascontiguousarray(value, dtype=dtype).tobytes())


def native_physical(ids, mapping):
    """Independent vectorized native token-index hash, arbitrary legal map."""
    ids = np.asarray(ids)
    mapping = np.asarray(mapping, dtype=np.int64)
    require(ids.shape[-2:] == (128, 8), 'Native assignment geometry')
    require(len(mapping) in (128, 160) and set(mapping.tolist()) == set(range(128)), 'Native map coverage')
    counts = np.bincount(mapping, minlength=128)
    inverse = np.zeros((128, int(counts.max())), np.int64)
    for logical in range(128):
        inverse[logical, :counts[logical]] = np.flatnonzero(mapping == logical)
    hashed = (np.arange(128, dtype=np.int64) * 2654435769) & 0xffffffff
    assigned = inverse[ids, hashed[:, None] % counts[ids]]
    require(np.array_equal(mapping[assigned], ids), 'Native reconstruction changed logical IDs')
    return assigned


def physical_work(assigned, slots):
    counts = np.bincount(np.asarray(assigned).ravel(), minlength=slots).reshape(2, slots // 2)
    tokens = counts.sum(1)
    active = np.count_nonzero(counts, axis=1)
    padded = (((counts + 63) // 64) * 64).sum(1)
    return dict(per_rank_tokens=tokens.tolist(), per_rank_active_copies=active.tolist(),
        per_rank_padded_rows=padded.tolist(), per_rank_m_tiles=(padded // 64).tolist(),
        max_to_mean_token_ratio=float(tokens.max() / tokens.mean()),
        max_to_mean_padded_ratio=float(padded.max() / padded.mean()),
        physical_counts=counts.ravel().tolist())


def routes_file(path):
    with np.load(path, allow_pickle=False) as z:
        ids, weights = z['logical_ids'], z['combining_weights']
        require(ids.shape == weights.shape == (4112, 128, 8), 'All4112 native routed batches')
        require(ids.dtype == np.dtype('int32') and weights.dtype == np.dtype('float32'), 'Native route dtypes')
        ih, wh = stock.tensor_sha(ids), stock.tensor_sha(weights)
        require(ih == str(z['logical_routing_sha256']) and wh == str(z['float32_combining_weights_sha256']), 'Native route artifact hash')
    require(np.all((ids >= 0) & (ids < 128)) and np.all(np.diff(np.sort(ids, axis=-1), axis=-1) > 0), 'Legal unique logical top8')
    require(np.isfinite(weights).all() and (weights >= 0).all() and np.allclose(weights.sum(-1), 1, atol=2e-6, rtol=0), 'Actual normalized native weights')
    return ids, weights, dict(logical_ids=ids[16:],
        logical_routing_sha256=ih, float32_combining_weights_sha256=wh,
        per_step=np.stack([np.bincount(row.ravel(), minlength=128) for row in ids[16:]]))


def check_fused(proof):
    require(proof['schema'] == 'resident_fused_adversarial_proof_v1' and proof['passed'], 'Fused adversarial proof')
    require(proof['cases_checked'] == 9 and proof['gpu_invocations_checked'] == 18 and proof['dtypes_checked'] == ['int32', 'int64'], 'Complete fused adversarial allocation')
    require(proof['evidence_sha256'] == digest(proof['cases']), 'Fused proof case hash')
    require(all(proof[k] for k in ('rank_agreement', 'stream_context_unchanged', 'internal_real_tokens_unchanged', 'input_ids_unchanged', 'external_real_tokens_unchanged')), 'Fused proof invariants')
    wanted = {(name, dtype) for name, _, _ in preflight.cases() for dtype in ('int32', 'int64')}
    seen = set()
    for row in proof['cases']:
        key = row['case'], row['dtype']
        require(key in wanted and key not in seen, 'Duplicate/unknown fused adversarial input'); seen.add(key)
        _, ids, real = next(x for x in preflight.cases() if x[0] == key[0])
        expected = preflight.expected(ids, real)
        require(row['real_tokens'] == real and row['input_ids_sha256'] == raw_hash(ids), 'Fused adversarial input changed')
        for field, value in expected.items():
            require(row[field + '_sha256'] == raw_hash(value), 'Fused adversarial output ' + field)
    require(seen == wanted, 'Missing fused adversarial cases')


def plan_check(plan, repo, archive=None):
    require(plan['schema'] == 'sequence_study_plan_v1' and plan['phase'] == 'supplement', 'Supplement plan')
    require(plan['content_sha256'] == digest(stock.unsealed(plan)), 'Plan seal')
    require(plan['source_sha256'] == digest(plan['source_files']) and len(plan['source_files']) == 116, 'Frozen116 source identity')
    if archive:
        with tarfile.open(archive) as tar:
            for name, expected in plan['source_files'].items():
                require(digest(tar.extractfile(name).read()) == expected, 'Archived source changed ' + name)
    else:
        for name, expected in plan['source_files'].items():
            require(file_sha(repo / name) == expected, 'Frozen source changed ' + name)
    require(file_sha(repo / 'docs/sequence-study/protocol-v1.md') == plan['protocol_sha256'], 'Protocol hash')
    parent = json.loads((repo / 'configs/resident-residual-final-v1.json').read_text())
    require(parent['content_sha256'] == plan['parent_plan_sha256'] == digest(stock.unsealed(parent)), 'Actual parent plan')
    require(plan['selected_fix'] == parent['selected_fix'] and plan['hardware'] == parent['hardware'], 'Changed selected fix/hardware')
    selected = []
    for dataset in ('gsm8k_train', 'wildchat'):
        selected.extend(sorted((b for b in parent['banks'] if b['dataset'] == dataset), key=lambda b: digest([471901, b['block_id']]))[:2])
    require(selected == plan['banks'], 'Outcome-independent four reused bundles')
    expected_trials = []
    permutations = list(itertools.permutations(ARMS))
    for i, entry in enumerate(selected):
        gid = entry['block_id']
        case = next(c for c in parent['cases'] if c['block_id'] == gid and c['arrival_type'] == ('code', 'conversation')[i % 2] and c['rho'] == .85)
        require(plan['calendars'][gid] == case, 'Exact reused arrival calendar'); stock.calendar_check(case)
        modes = ['contiguous', 'mixed'] + (['batch_shuffle'] if i == 0 else [])
        for rep in (0, 1):
            for mode in (modes if (i + rep) % 2 == 0 else modes[::-1]):
                order = list(permutations[(i + modes.index(mode)) % 6])
                if rep: order.reverse()
                expected_trials.extend(dict(block_id=gid, mode=mode, rep=rep, arm=a, arm_order=order) for a in order)
    require(plan['trials'] == expected_trials and len(expected_trials) == 54 and plan['arms'] == list(ARMS), 'Frozen54 trial order and reversal')
    require(plan['jobs'] == 4096 and plan['batch_tokens'] == 128 and plan['input_seed'] == 471902 and not plan['timing_selection'], 'Input/selection contract')
    return parent


def check_input_bundle(root, entry, mode, meta, bankroot, reconstruction):
    gid = entry['block_id']; stem = gid + '-' + mode
    path = root / (stem + '.npz')
    require(file_sha(path) == meta['index_artifact_sha256'], 'Exact saved input-index artifact')
    with np.load(path, allow_pickle=False) as z:
        indices, saved = z['indices'], json.loads(str(z['metadata']))
    require(all(meta.get(k) == value for k, value in saved.items()), 'Input artifact metadata/record mismatch')
    expected_indices, expected_meta = reconstruction
    require(np.array_equal(indices, expected_indices[mode]) and indices.dtype == np.dtype('int64'), 'Independent exact sequence reconstruction')
    require(all(meta.get(k) == value for k, value in expected_meta[mode].items()), 'Independent sequence numerical metadata')
    bank = bankroot / entry['capture_path']
    residual.bank_check(bank, entry, meta)
    ids, weights, artifact = routes_file(root / (stem + '-stock-routes.npz'))
    proof_path = root / (stem + '-resident-diagnostics.json')
    proof = json.loads(proof_path.read_text()); require(len(proof) == 4112, 'All4112 resident assignment checks')
    assigned = np.empty(ids.shape, np.int64)
    hashes = []
    for raw, logical in enumerate(ids):
        physical, counts = residual.oracle(logical); assigned[raw] = physical
        row = proof[raw]; ph = raw_hash(physical); hashes.append(ph)
        require(row['physical_sha256'] == ph and row['logical_sha256'] == raw_hash(logical), 'Exact physical/logical assignment bytes')
        require(row['weights_sha256'] == raw_hash(weights[raw], '<f4'), 'Exact native combining weights in resident check')
        require(all(np.array_equal(row[k], v) for k, v in counts.items()), 'Resident per-rank assignment counts')
    return dict(indices=indices, ids=ids, weights=weights, artifact=artifact, physical=assigned,
        assignment_digest=digest(hashes), metadata=meta,
        identities=dict(index=file_sha(path), routes=file_sha(root / (stem + '-stock-routes.npz')), resident=file_sha(proof_path)))


def fixed_or_native_state(row, artifact):
    if row['arm'] != 'fix':
        return stock.native_check(row, artifact)
    residual.static_check(row)
    require(all(peer['start']['final_map'] == INITIAL.tolist() and peer['end']['final_map'] == INITIAL.tolist() for peer in row['per_rank_state']), 'Fixed resident map')
    return dict(rank_windows_reconstructed=0, rank_forced_skip_decisions=0)


def check_token_identity(cache, groups):
    """Associate actual native outputs with captured-record IDs, bit for bit."""
    report = {}
    for gid in groups:
        modes = [(mode, value) for (group, mode), value in cache.items() if group == gid]
        first = modes[0][1]
        records, positions = np.unique(first['indices'][16:].ravel(), return_index=True)
        require(np.array_equal(records, np.arange(first['metadata']['source_records'])), 'Every source record represented in measured replay')
        canonical_ids = first['ids'][16:].reshape(-1, 8)[positions]
        canonical_weights = first['weights'][16:].reshape(-1, 8)[positions]
        canonical_bits = canonical_weights.view(np.uint32)
        checked = 0
        for mode, values in modes:
            # Include the common warmup as an extra check, while the canonical
            # identity is established only from measured token occurrences.
            idx = values['indices'].ravel()
            actual_ids = values['ids'].reshape(-1, 8)
            actual_bits = values['weights'].reshape(-1, 8).view(np.uint32)
            require(np.array_equal(actual_ids, canonical_ids[idx]), 'Native logical routing changed for captured record in ' + gid + '/' + mode)
            require(np.array_equal(actual_bits, canonical_bits[idx]), 'Native float32 combining-weight bytes changed for captured record in ' + gid + '/' + mode)
            checked += len(idx)
        report[gid] = dict(source_records=len(records), modes=len(modes),
            token_occurrences_checked=checked, logical_topk_exact=True,
            float32_combining_weights_bitwise_equal=True,
            canonical_logical_routes_sha256=stock.tensor_sha(canonical_ids),
            canonical_float32_weights_sha256=stock.tensor_sha(canonical_weights),
            scope='Same captured-record ID, all positions/modes/repetitions via shared authenticated input artifacts; includes warmup')
    return report


def check_probe_assignment(row, physical, logical, weights, mapping, rank):
    slots = len(mapping); local_slots = slots // 2
    counts = np.bincount(physical.ravel(), minlength=slots)
    local = counts[rank * local_slots:(rank + 1) * local_slots]
    expert_map = np.full(slots, -1, np.int64)
    expert_map[rank * local_slots:(rank + 1) * local_slots] = np.arange(local_slots)
    tokens = int(local.sum()); padded = int(np.sum((local + 63) // 64 * 64))
    expected = dict(physical_ids=physical.tolist(), physical_ids_sha256=raw_hash(physical),
        logical_ids_sha256=raw_hash(logical), combining_weights_sha256=raw_hash(weights, '<f4'),
        physical_counts=counts.tolist(), local_expert_map=expert_map.tolist(), local_counts=local.tolist(),
        local_assignments=tokens, local_active_copies=int(np.count_nonzero(local)),
        local_nonempty_m_tiles=padded // 64, local_padded_assignment_rows=padded,
        useful_two_gemm_flops_proxy=tokens * 6 * 2048 * 768,
        padded_two_gemm_flops_proxy=padded * 6 * 2048 * 768,
        useful_assignment_fraction=None if not padded else tokens / padded)
    require(all(row.get(k) == v for k, v in expected.items()), 'Actual expert arguments/work differ from CPU reconstruction')


def check_probes(path, rows, cache, entries, bankroot):
    proof = json.loads(path.read_text())
    require(set(proof) == {'stockoff', 'fix', 'stockon32', 'stockon32_initial'}, 'Four declared expert probe states')
    gid = entries[0]['block_id']; arrays = cache[gid, 'mixed']
    raw_indices = np.linspace(16, 4111, 64, dtype=int).tolist()
    with np.load(bankroot / entries[0]['capture_path'], allow_pickle=False) as z:
        hidden = z['hidden_states_bf16']
    h = hashlib.sha256()
    for raw in raw_indices:
        h.update(np.ascontiguousarray(hidden[arrays['indices'][raw]], dtype='<u2').tobytes())
    selected_hash = h.hexdigest(); totals = 0
    last_native = next(r for r in reversed(rows) if r['block_id'] == gid and r['arm'] == 'stockon32')
    for label, peers in proof.items():
        arm = 'stockon32' if label.startswith('stockon32') else label
        mapping = (np.arange(128) if arm == 'stockoff' else np.asarray(last_native['native_metadata']['final_map'])
                   if label == 'stockon32' else INITIAL)
        require(len(peers) == 2, 'Both expert diagnostic ranks')
        for rank, p in enumerate(peers):
            require(p['schema'] == 'sequence_expert_path_diagnostic_v1' and p['rank'] == rank and p['arm'] == arm, 'Expert diagnostic identity')
            require(p['label'] == ('initial_stockon32' if label.endswith('_initial') else 'post_trials_' + arm), 'Expert diagnostic label')
            require(p['raw_indices'] == raw_indices and p['selected_hidden_bf16_sha256'] == selected_hash, 'Expert diagnostic actual input bytes')
            require(p['map'] == mapping.tolist() and p['map_sha256'] == raw_hash(mapping), 'Expert diagnostic actual map')
            require(p['kernel_config'] == residual.CONFIG and p['native_experts_source_sha256'] == EXPERT_SHA, 'Expert diagnostic native implementation/config')
            required_true = ('argument_capture_pass_separate', 'correctness_pass', 'events_preinitialized', 'expert_apply_restored', 'no_controller_steps', 'observation_buffers_unchanged')
            require(all(p[k] is True for k in required_true), 'Expert diagnostic capture/restoration/state checks')
            require(p['capture_forward_calls'] == p['instrumented_forward_calls'] == 64 and p['numerical_full_batch_checks'] == 128, 'Expert diagnostic numerical scope')
            require(p['atol'] == .02 and p['rtol'] == .03 and np.isfinite(p['max_abs_difference']), 'Expert diagnostic numerical evidence')
            elapsed = np.asarray(p['expert_path_elapsed_ms'])
            require(elapsed.shape == (64,) and np.isfinite(elapsed).all() and (elapsed > 0).all(), 'Finite expert intervals')
            require(len(p['assignments']) == 64, 'All64 captured native expert arguments')
            start, end = p['start_state'], p['end_state']
            for state in (start, end):
                require(state['final_map'] == mapping.tolist() and not state['pending_result'] and not state['rebalanced'], 'Quiescent expert probe state')
                require(all(state[k] for k in ('resident_state_verified', 'rank_map_agreement', 'normal_completion_verified')), 'Expert probe weight/map attestation')
            same = ('controller_step_calls', 'native_counter', 'native_valid_steps', 'async_events', 'recorded_step_indices', 'deferred_boundary_steps', 'decisions', 'terminal_physical_window_totals_local', 'physical_window_next_write', 'migration_sent_bytes_local', 'migration_sent_messages_local')
            require(all(start[k] == end[k] for k in same), 'Expert probe changed controller/history/transport')
            require(end['mapper_calls'] - start['mapper_calls'] == (128 if arm == 'fix' else 0), 'Expert probe invocation accounting')
            if label == 'stockon32':
                for field in ('final_map', 'async_events', 'migration_sent_bytes_local', 'recorded_step_indices'):
                    require(start[field] == last_native['per_rank_state'][rank]['audit'][field], 'Final-map probe not linked to actual last native trial')
            if label.endswith('_initial'):
                require(start['native_counter'] == 2250 and start['async_events'] == [] and start['native_valid_steps'] == 0, 'Native initial-map canonical reset')
            for raw, actual in zip(raw_indices, p['assignments']):
                logical = arrays['ids'][raw]
                physical = logical.astype(np.int64) if arm == 'stockoff' else arrays['physical'][raw] if arm == 'fix' else native_physical(logical, mapping)
                require(actual['raw_index'] == raw, 'Expert argument/input association')
                check_probe_assignment(actual, physical, logical, arrays['weights'][raw], mapping, rank); totals += 1
    return dict(passed=True, file_sha256=file_sha(path), rank_input_arguments_reconstructed=totals,
        event_intervals_checked=totals, scope='Actual argument/count reconstruction and recorded numerical/state attestations; CUDA event values are not independently rerun')


def work_check(root, rows, cache):
    summaries = json.loads((root / 'summary.json').read_text())
    work_index = {(r['block_id'], r['mode'], r['rep'], r['arm']): r for r in summaries['work']}
    require(len(work_index) == len(summaries['work']) == len(rows), 'Complete unique structural work artifacts')
    routing_index = {(r['block_id'], r['mode']): r for r in summaries['routing']}
    require(len(routing_index) == len(summaries['routing']) == len(cache), 'Complete routing summaries')
    for key, a in cache.items():
        actual = routing_index[key]; ids = a['ids'][16:]
        counts = np.stack([np.bincount(batch.ravel(), minlength=128) for batch in ids])
        proportions = counts / 1024
        tv = np.sum(np.abs(proportions[1:] - proportions[:-1]), axis=1) / 2
        expected = dict(jobs=4096, logical_route_sha256=stock.tensor_sha(ids),
            source_file_sha256=a['identities']['routes'],
            mean_distinct_logical_experts_per_job=float(np.count_nonzero(counts, axis=1).mean()),
            mean_largest_expert_fraction=float(proportions.max(1).mean()),
            largest_expert_fraction_max=float(proportions.max()),
            adjacent_job_tv_mean=float(tv.mean()), adjacent_job_tv_p95=float(np.quantile(tv, .95)),
            total_logical_counts=counts.sum(0).tolist())
        require(all(actual.get(k) == v for k, v in expected.items()), 'Actual logical routing summary differs')
        for window in (16, 1000):
            n = len(counts) // window
            blocks = counts[:n * window].reshape(n, window, 128).sum(1) / (1024 * window)
            wanted = dict(nonoverlapping_full_windows=n,
                consecutive_window_tv=(np.abs(blocks[1:] - blocks[:-1]).sum(1) / 2).tolist(),
                excluded_terminal_jobs=4096 - n * window)
            require(actual['windows'][str(window)] == wanted, 'Actual routing window summary differs')
    checked = 0
    for row in rows:
        gid, mode, rep, arm = row['block_id'], row['mode'], row['technical_rep'], row['arm']
        key = gid, mode, rep, arm; index = work_index[key]
        path = root / index['file']; require(file_sha(path) == index['sha256'], 'Reconstructed work artifact hash')
        w = json.loads(path.read_text()); require(w['jobs'] == 4096 and len(w['records']) == 4096 and w['arm'] == arm, 'All work jobs')
        a = cache[gid, mode]; ids = a['ids'][16:]; mapping = np.asarray(row['native_metadata']['source_map'])
        commits = {int(e['commit_step']) + 1: np.asarray(e['committed_map']) for e in row['native_metadata']['async_events'] if not e['commit_in_completion_tail']}
        native_cache = {}
        for job, actual in enumerate(w['records']):
            if job in commits: mapping = commits[job]
            if arm == 'stockoff': physical = ids[job]
            elif arm == 'fix': physical = a['physical'][job + 16]
            else:
                mid = tuple(mapping)
                if mid not in native_cache: native_cache[mid] = native_physical(ids, mapping)
                physical = native_cache[mid][job]
            expected = physical_work(physical, len(mapping)); expected.pop('physical_counts')
            expected.update(job=job, map_sha256=stock.tensor_sha(mapping.astype(np.int64)), physical_ids_sha256=stock.tensor_sha(physical))
            require(actual == expected, 'Committed-map dispatch-work reconstruction'); checked += 1
        for field, value in (('mean_rank_tokens', 'per_rank_tokens'), ('mean_rank_active_copies', 'per_rank_active_copies'), ('mean_rank_padded_rows', 'per_rank_padded_rows')):
            require(index[field] == np.mean([r[value] for r in w['records']], axis=0).tolist(), 'Work aggregate ' + field)
    return dict(jobs_checked=checked, summary_sha256=file_sha(root / 'summary.json'), scope='Structural assignment proxies, not counters or exclusive GPU time')


def audit(plan_path, records, repo, bankroot, output_inputs=None, archive=None, binary_root=None):
    plan = json.loads(plan_path.read_text()); parent = plan_check(plan, repo, archive)
    rows, files = residual.read_records(records)
    expected = {(t['block_id'], t['mode'], t['rep'], t['arm']): t for t in plan['trials']}
    require(len(rows) == len(expected) == 54, 'Require all54 completed trials')
    root = Path(records); input_root = output_inputs or root / 'inputs'
    entries = {b['block_id']: b for b in plan['banks']}; reconstructed = {}; cache = {}; seen = set(); orders = {}; proof_ids = set(); binaries = set(); kernels = None
    record_metrics = []; cycles = 0; model_identities = set()
    for row in rows:
        key = row['block_id'], row['mode'], row['technical_rep'], row['arm']
        require(key in expected and key not in seen, 'Unexpected/duplicate sequence trial'); seen.add(key)
        gid, mode, rep, arm = key; entry = entries[gid]; declared = expected[key]
        require(row['record_id'] == digest({k: v for k, v in row.items() if k != 'record_id'}), 'Raw record identity')
        require(row['schema'] == 'sequence_study_trial_v1' and row['phase'] == 'supplement' and row['status'] == 'complete' and row['evidence_eligible'] is False and row['descriptive_supplement'] is True, 'Descriptive completed scope')
        require(row['plan_sha256'] == plan['content_sha256'] and row['source_sha256'] == plan['source_sha256'], 'Executed source/plan')
        require(row['content_group_id'] == gid and row['dataset'] == entry['dataset'] and row['arm_order'] == declared['arm_order'], 'Content/arm order')
        orders.setdefault(gid, []).append(key[1:])
        hw = plan['hardware']; require(row['hostname'].split('.')[0] == hw['hostname'] and sorted(row['gpu_uuids']) == sorted(hw['gpu_uuids']) and str(row['allocation_id']) == str(hw['allocation_id']), 'Reserved hardware')
        require(row['clock']['monotonic'] and not row['clock']['adjustable'] and 'CLOCK_MONOTONIC' in row['clock']['implementation'], 'Monotonic host clock')
        require(row['source_weights']['source_revision'] == 'ad44e777bcd18fa416d9da3bd8f70d33ebb85d39' and row['source_weights']['layer'] == 0, 'Actual expert checkpoint')
        model_identities.add(digest(row['source_weights']))
        calendar = plan['calendars'][gid]['calendar']
        require(row['calendar'] == calendar and row['arrival_offsets_ns'] == calendar['arrival_offsets_ns'] and row['horizon_ns'] == calendar['horizon_ns'], 'Same reused arrivals')
        metrics = stock.raw_clock_check(row); record_metrics.append(dict(block_id=gid, mode=mode, rep=rep, arm=arm, metrics=metrics))
        rt = row['runtime_metadata']
        require(rt['study_id'] == 'sequence_study_v1' and rt['input_seed'] == 471902 and rt['sequence_intervention'] is True, 'Sequence runtime identity')
        require(rt['native_vllm_commit'] == 'db9527a46873454610df6dbedf79a36d6bf1a7f6' and rt['model_revision'] == 'ad44e777bcd18fa416d9da3bd8f70d33ebb85d39' and rt['world_size'] == 2, 'Native checkpoint/runtime')
        require(rt['tf32_allowed'] is False and rt['float32_matmul_precision'] == 'highest' and rt['precision_per_rank'] == [dict(allow_tf32=False, matmul_precision='highest')] * 2, 'Both-rank reference precision')
        pid = digest(rt['fused_adversarial'])
        if pid not in proof_ids: check_fused(rt['fused_adversarial']); proof_ids.add(pid)
        warm = rt['native_async_preconditioning']
        require(warm['jobs'] == 750 and warm['timing_evidence'] is False and len(warm['per_rank']) == 2, 'Native process preconditioning')
        for s in warm['per_rank']:
            require(s['resident_state_verified'] and s['normal_completion_verified'] and not s['pending_result'] and not s['rebalanced'], 'Preconditioning normally completed')
            require(len(s['async_events']) == 1 and s['async_events'][0]['launch_step'] == 749 and s['async_events'][0]['commit_step'] == 749, 'Preconditioning native launch/commit')
        probes = rt['output_reduction_probe_per_rank']; require(len(probes) == 2 and probes[0]['observed_successful_backends'] == probes[1]['observed_successful_backends'], 'Native reduction agreement')
        if gid not in reconstructed:
            with np.load(bankroot / entry['capture_path'], allow_pickle=False) as z:
                reconstructed[gid] = inputs_oracle.reconstruct(z['prompt_index'], z['token_position'], gid)
        if (gid, mode) not in cache:
            cache[gid, mode] = check_input_bundle(input_root, entry, mode, row['input_meta'], bankroot, reconstructed[gid])
        a = cache[gid, mode]
        require(row['input_meta'] == a['metadata'], 'Cross-arm/repetition input identity')
        pre, post = row['preflight'], row['postflight']
        for p, count in ((pre, 4112), (post, 16)):
            require(p['correctness_pass'] and p['full_batch_inputs_checked'] == count and p['atol'] == .02 and p['rtol'] == .03 and np.isfinite(p['full_batch_max_abs_difference']), 'Full output check coverage')
        require(all(pre[k] is True for k in ('logical_ids_equal', 'combining_weights_equal', 'rank_routing_agreement', 'input_unchanged')), 'Logical identity evidence')
        require(pre['fp32_job_indices'] == [16, 765, 3765, 4111] and pre['fp32_tokens_per_job'] == 16 and post['raw_indices'] == np.linspace(16, 4111, 16, dtype=np.int64).tolist(), 'Reference sampling scope')
        require(pre['logical_routing_sha256'] == a['artifact']['logical_routing_sha256'] and pre['float32_combining_weights_sha256'] == a['artifact']['float32_combining_weights_sha256'], 'Exact actual route artifact identity')
        if arm == 'fix':
            require(pre['online_assignment_inputs_checked'] == 4112 and pre['physical_assignment_digest'] == a['assignment_digest'], 'All actual resident assignments checked')
        fixed_or_native_state(row, a['artifact'])
        k = residual.development.compiled_check(row, n=4096, arm=arm)
        if k is not None: kernels = k; binaries.add(digest(k))
        require(row['native_metadata'] == row['per_rank_state'][0]['end'], 'Rank0 end-state metadata')
        for peer in row['per_rank_state']:
            initial = list(range(128)) if arm == 'stockoff' else INITIAL.tolist()
            require(peer['start']['final_map'] == initial and peer['start']['native_valid_steps'] == 0 and peer['start']['recorded_step_indices'] == [], 'Canonical untimed-reset source state')
            if arm == 'stockon32': require(peer['start']['native_counter'] == 2250, 'Native default startup counter')
            for scope in ('start', 'end', 'audit'):
                state = peer[scope]
                require(state['resolved_kernel_config'] == residual.CONFIG and state['kernel_override_applied'] is False, 'Native tile selector unchanged')
                require(state['gate_weight_sha256'] == row['input_meta']['gate_weight_sha256'], 'Authenticated actual gate weight')
                require(state['enable_eplb'] == (arm != 'stockoff') and state['replicas'] == (0 if arm == 'stockoff' else 32)
                        and state['is_async'] == (arm == 'stockon32'), 'Actual disabled/enabled/static arm configuration')
        if arm == 'stockon32':
            first, second = row['per_rank_state']
            for scope in ('start', 'end', 'audit'):
                aevents, bevents = first[scope]['async_events'], second[scope]['async_events']
                require(len(aevents) == len(bevents), 'Native cycle count rank agreement')
                for ae, be in zip(aevents, bevents):
                    require(all(ae[k] == be[k] for k in ae if k not in ('sent_bytes_local', 'sent_bytes_at_launch')), 'Native event/map/window rank agreement')
            cycles += len(row['native_metadata']['async_events'])
    require(seen == set(expected) and len(binaries) == 1 and len(model_identities) == 1, 'Complete matrix / unchanged weights and compiled algorithm')
    for gid in entries:
        wanted = [(t['mode'], t['rep'], t['arm']) for t in plan['trials'] if t['block_id'] == gid]
        require(orders[gid] == wanted, 'Exact within-bundle execution order')
        modes = [value for (g, _), value in cache.items() if g == gid]
        require(len({value['metadata']['measured_occurrence_multiset_sha256'] for value in modes}) == 1 and len({value['metadata']['warmup_indices_sha256'] for value in modes}) == 1, 'Matched occurrence bag/conditioning')
        require(len({raw_hash(np.bincount(value['ids'][16:].ravel(), minlength=128)) for value in modes}) == 1, 'Ordering changed aggregate actual logical routing')
    token_identity = check_token_identity(cache, entries)
    probe_audit = check_probes(root / 'expert-probes.json', rows, cache, plan['banks'], bankroot)
    work_audit = work_check(root / 'work', rows, cache)
    return dict(schema='independent_sequence_execution_v1', passed=True, records=54,
        source_sha256=plan['source_sha256'], source_files_verified=len(plan['source_files']),
        plan_content_sha256=plan['content_sha256'], plan_file_sha256=file_sha(plan_path),
        source_archive_sha256=None if archive is None else file_sha(archive), raw_files=files,
        unique_content_bundles=4, unique_ordered_inputs=len(cache), prepared_batches_verified=len(cache) * 4112,
        independent_resident_assignment_batches=len(cache) * 4112, raw_jobs_verified=54 * 4096,
        native_committed_cycles=cycles, compiled_binary_verification=residual.development.check_binary_files(kernels, binary_root),
        captured_record_routing_identity=token_identity,
        inputs={gid + '-' + mode: a['identities'] for (gid, mode), a in cache.items()},
        expert_probe_audit=probe_audit, structural_work_audit=work_audit,
        auditor_sha256=file_sha(__file__), helper_sha256={Path(m.__file__).name: file_sha(m.__file__) for m in (stock, residual, residual.development, residual.development.base, preflight, inputs_oracle)},
        limits=['CPU audit of saved native routes and recorded numerical/full-weight attestations; no independent GPU rerun.',
            'Four reused content bundles and reused arrivals; technical repetitions are not independent workloads.',
            'Mixed versus contiguous changes batching and temporal order; batch_shuffle preserves batches.',
            'Expert intervals are separate instrumented GPU-stream observations, not pure GEMM residence or migration payback.',
            'No new significance tests or acceptance gate.'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'records', 'capture-root', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--repo-root', type=Path, default=REPO)
    for name in ('inputs-root', 'source-archive', 'binary-root'):
        p.add_argument('--' + name, type=Path)
    args = p.parse_args(); require(not args.output.exists(), 'Fresh audit output required')
    result = audit(args.plan, args.records, args.repo_root, args.capture_root, args.inputs_root, args.source_archive, args.binary_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('passed', 'records', 'prepared_batches_verified', 'raw_jobs_verified', 'native_committed_cycles')}))

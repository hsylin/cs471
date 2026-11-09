#!/usr/bin/env python3
"""Audit actual native policy routing windows against authenticated capture tapes.

Supports complete validation/final matrices and plain or gzip JSONL. Does not
inspect elapsed-time values or estimate performance. The NumPy physical mapper
is a transcription of the hash-pinned native token-index replica selection.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('block_id', 'regime', 'batch_tokens', 'repetition', 'kernel_variant', 'policy')
TRAJECTORY = ('step', 'allowed', 'reason', 'current_counts', 'current_map',
              'proposed_map', 'committed_map', 'transfer_sent_bytes', 'physical_slot_assignments_window')
EXPERT_BYTES = 3 * 2048 * 768 * 2


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def array_digest(value, dtype=np.int64):
    return hashlib.sha256(np.asarray(value, dtype=dtype).tobytes()).hexdigest()


def load_npz(path, sha):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha, f'File checksum mismatch: {path}'
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved['metadata']))
        arrays = {k: saved[k].copy() for k in saved.files if k not in ('metadata', 'content_sha256')}
        h = hashlib.sha256(canonical(metadata).encode())
        for name, a in sorted(arrays.items()):
            a = np.ascontiguousarray(a)
            h.update(canonical([name, a.dtype.str, list(a.shape)]).encode()); h.update(a.tobytes())
        assert h.hexdigest() == str(saved['content_sha256']), f'Content checksum mismatch: {path}'
    return metadata, arrays


def legal_map(values):
    mapping = np.asarray(values)
    assert mapping.shape == (160,) and mapping.dtype.kind in 'iu'
    assert np.all((mapping >= 0) & (mapping < 128))
    assert np.all(np.bincount(mapping, minlength=128) > 0), 'Logical expert coverage lost'
    return mapping


def native_physical_counts(ids, mapping):
    # vLLM commit db9527a46873454610df6dbedf79a36d6bf1a7f6:
    # base_router.py token_idx Knuth hash; eplb_state.py ascending replica slots.
    copies = np.bincount(mapping, minlength=128)
    inverse = np.zeros((128, int(copies.max())), np.int64)
    for logical in range(128):
        inverse[logical, :copies[logical]] = np.flatnonzero(mapping == logical)
    hashed = (np.arange(ids.shape[1], dtype=np.int64) * 2654435769) & 0xffffffff
    physical = inverse[ids, hashed[None, :, None] % copies[ids]]
    offsets = np.arange(len(ids), dtype=np.int64)[:, None, None] * 160
    return np.bincount((physical + offsets).ravel(), minlength=len(ids)*160).reshape(len(ids), 160).astype(np.int32)


def payload_senders(old, new):
    old, new = old.reshape(2, 80), new.reshape(2, 80)
    return [len(set(new[1])-set(old[1]))*EXPERT_BYTES,
            len(set(new[0])-set(old[0]))*EXPERT_BYTES]


def predicted_gain(counts, old, new):
    def peak(mapping):
        copies = np.bincount(mapping, minlength=128)
        return (counts[mapping]/copies[mapping]).reshape(2, 80).sum(axis=1).max()
    before = peak(old)
    return float((before-peak(new))/before) if before else 0.


def audit(manifest_path, records_path, capture_dir):
    manifest = json.loads(manifest_path.read_text())
    config_hash = digest({k: v for k, v in manifest.items() if k != 'freeze'})
    assert manifest['freeze']['sha256'] == config_hash
    assert manifest['optimization_schema_version'] == 1 and manifest['phase'] in ('validation', 'final')
    trace, matrix = manifest['trace'], manifest['matrix']
    assert (trace['warmup_steps'], trace['window_steps'], trace['measured_steps'], trace['raw_steps']) == (32, 16, 512, 544)
    assert manifest['policy']['rebalance_interval'] == 64 and manifest['policy']['replica_budget'] == 32
    assert set(matrix['kernel_variants']) == {'DEFAULT', 'TUNED'} and set(matrix['policies']) == {'B3', 'SKIP'}
    for field in ('block_ids', 'regimes', 'batch_tokens', 'kernel_variants', 'policies'):
        assert len(matrix[field]) == len(set(matrix[field]))
    expected = set(itertools.product(matrix['block_ids'], matrix['regimes'], matrix['batch_tokens'],
                                    range(matrix['repetitions']), matrix['kernel_variants'], matrix['policies']))
    opener = gzip.open if records_path.suffix == '.gz' else open
    with opener(records_path, 'rt') as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert len(rows) == len(expected), 'Incomplete or oversized declared matrix'
    keys = [tuple(r[k] for k in FIELDS) for r in rows]
    assert len(set(keys)) == len(keys) and set(keys) == expected, 'Missing or duplicate arm'
    assert len({r['record_id'] for r in rows}) == len(rows)
    pins = json.loads((ROOT/'skewbench/native_sources.json').read_text())
    assert pins['commit'] == manifest['environment']['vllm_commit'] == 'db9527a46873454610df6dbedf79a36d6bf1a7f6'
    assert pins['sha256']['vllm/model_executor/layers/fused_moe/router/base_router.py'] == 'c909924fcfa461ccdaa6290d215d4d35bd4062fb02feeb23619b234da993f6bf'
    assert pins['sha256']['vllm/distributed/eplb/eplb_state.py'] == 'eaac132acd438074343e5ee56c52e1df46f37a9bf7a303bd3b9373c69b513f17'
    entries = {(t['block_id'], t['regime'], t['batch_tokens']): t for t in manifest['tapes']}
    documents = {b['block_id']: b['source_document_ids'] for b in manifest['blocks']}
    banks, tapes, physical_cache = {}, {}, {}
    summaries = defaultdict(lambda: Counter(runs=0, decisions=0, accepted=0, rejected=0, sender_bytes=0,
                                          terminal_proposals_suppressed=0, final_map_changed_runs=0))
    reasons = defaultdict(Counter)
    states, quartets = {}, defaultdict(list)
    map_checks = logical_checks = physical_entries = numerical_forward_checks = 0
    maximum_error = maximum_default_delta = 0.
    for row in rows:
        assert row['config_hash'] == config_hash and row['phase'] == manifest['phase'] and row['split'] == manifest['split']
        assert row['measurement_kind'] == 'gpu_optimization' and row['status'] == 'complete'
        assert row['evidence_eligible'] is True and row['correctness_pass'] is True
        for field, value in manifest['required_record_metadata'].items(): assert row[field] == value, field
        assert row['record_id'] == digest({k: row[k] for k in ('config_hash',)+FIELDS})
        assert digest(row['optimization_file_sha256']) == row['optimization_sha256'] == manifest['environment']['optimization_sha256']
        assert digest(row['harness_file_sha256']) == row['legacy_harness_sha256']
        assert row['world_size'] == 2 and row['replica_budget_actual'] == 32
        assert row['steps_completed'] == 512 and row['observation_warmup_steps'] == 32 and row['raw_trace_steps'] == 544
        assert row['terminal_proposals_suppressed'] == 1 and row['observation_conservation_checks'] == 7
        key = (row['block_id'], row['regime'], row['batch_tokens'])
        entry = entries[key]
        assert row['capture_sha256'] == entry['capture_sha256'] and row['replay_sha256'] == entry['replay_sha256']
        assert row['source_prompt_ids'] == documents[row['block_id']]
        if entry['capture_sha256'] not in banks:
            metadata, bank = load_npz(capture_dir/entry['capture_path'], entry['capture_sha256'])
            assert metadata['correctness_pass'] is True and metadata['source_prompt_ids'] == documents[row['block_id']]
            assert (metadata['num_experts'], metadata['top_k'], metadata['hidden_size']) == (128, 8, 2048)
            assert metadata['source_model'] == manifest['environment']['model_id'] and metadata['source_revision'] == manifest['environment']['model_revision']
            ids = bank['expert_ids']
            assert ids.shape == (4096, 8) and ids.dtype == np.int64
            assert np.all((ids >= 0) & (ids < 128)) and np.all(np.diff(np.sort(ids, axis=1), axis=1) > 0)
            banks[entry['capture_sha256']] = (metadata, bank)
        metadata, bank = banks[entry['capture_sha256']]
        assert metadata['split'] == manifest['split'] and metadata['block_id'] == row['block_id']
        if key not in tapes:
            replay, data = load_npz(capture_dir/entry['replay_path'], entry['replay_sha256'])
            assert replay['capture_sha256'] == entry['capture_sha256'] and replay['source_prompt_ids'] == row['source_prompt_ids']
            assert (replay['num_steps'], replay['warmup_steps'], replay['seed']) == (544, 32, trace['seed'])
            assert replay['block_id'] == key[0] and replay['regime'] == key[1] and replay['batch_tokens'] == key[2]
            case = next(c for c in trace['cases'] if c['regime'] == key[1])
            assert all(replay[f] == case[f] for f in ('phase_steps', 'duration_steps', 'repeat_steps'))
            indices = data['indices']
            assert indices.shape == (544, row['batch_tokens']) and indices.dtype == np.int64
            assert np.all((indices >= 0) & (indices < 4096)) and np.all(np.diff(np.sort(indices, axis=1), axis=1) > 0)
            assert np.array_equal(bank['domain_ids'][indices], np.broadcast_to(np.asarray(replay['domain_schedule'])[:, None], indices.shape))
            tapes[key] = indices
        indices = tapes[key]
        initial = legal_map(row['source_map']); final = legal_map(row['final_map']); map_checks += 2
        assert np.array_equal(initial, np.r_[np.arange(128), np.arange(32)])
        assert row['source_map_sha256'] == array_digest(initial) and row['final_map_sha256'] == array_digest(final)
        log = row['decision_log']
        assert [d['step'] for d in log] == [63, 127, 191, 255, 319, 383, 447]
        assert row['decision_trajectory_sha256'] == digest([{k: d.get(k) for k in TRAJECTORY} for d in log])
        previous = initial; total_bytes = 0
        for decision in log:
            old = legal_map(decision['current_map']); proposed = legal_map(decision['proposed_map']); committed = legal_map(decision['committed_map']); map_checks += 3
            assert np.array_equal(old, previous), 'Committed map chain is broken'
            assert isinstance(decision['allowed'], bool) and decision['commit_completed'] is True
            assert np.array_equal(committed, proposed if decision['allowed'] else old)
            assert decision['committed_map_sha256'] == array_digest(committed)
            assert decision['candidate_changed_slots'] == int(np.count_nonzero(old != proposed))
            stop = 32 + decision['step'] + 1
            logical_ids = bank['expert_ids'][indices[stop-16:stop]]
            raw_counts = np.bincount(logical_ids.ravel(), minlength=128)
            counts = np.asarray(decision['current_counts'])
            assert counts.shape == (128,) and counts.dtype.kind in 'iu' and np.all(counts >= 0)
            assert np.array_equal(counts, raw_counts*2), 'Observed logical counts differ from the exact captured input window'
            assert counts.sum() == 16*row['batch_tokens']*8*2
            physical = np.asarray(decision['physical_slot_assignments_window'])
            assert physical.shape == (16, 160) and physical.dtype.kind in 'iu' and np.all(physical >= 0)
            assert np.all(physical.sum(axis=1) == row['batch_tokens']*8)
            assert decision['physical_slot_assignments_sha256'] == array_digest(physical, np.int32)
            collapsed = np.bincount(old, weights=physical.sum(axis=0), minlength=128).astype(np.int64)
            assert np.array_equal(collapsed, raw_counts), 'Physical-to-logical conservation failed'
            physical_key = (entry['replay_sha256'], decision['step'], array_digest(old))
            if physical_key not in physical_cache:
                physical_cache[physical_key] = native_physical_counts(logical_ids, old)
            assert np.array_equal(physical, physical_cache[physical_key]), 'Native replica assignment differs from exact CPU reconstruction'
            assert np.array_equal(physical.reshape(16, 2, 80).sum(axis=2), decision['physical_owner_assignments_per_step'])
            assert decision['physical_counter_input_multiplicity'] == 2
            assert decision['physical_window_measured_step_range'] == [decision['step']-15, decision['step']]
            assert decision['physical_window_next_write'] == 0
            expected_payload = payload_senders(old, committed)
            assert decision['transfer_sent_bytes_by_rank'] == expected_payload
            assert decision['transfer_sent_bytes'] == sum(expected_payload)
            if row['policy'] == 'SKIP':
                assert decision['allowed'] is False and decision['reason'] == 'always_skip' and decision['predicted_gain'] is None
            else:
                gain = predicted_gain(counts, old, proposed)
                assert np.isclose(decision['predicted_gain'], gain, rtol=0, atol=1e-14)
                allow = gain >= manifest['policy']['epsilon']
                assert decision['allowed'] == allow
                assert decision['reason'] == ('allowed' if allow else 'below_improvement_threshold')
            total_bytes += sum(expected_payload); previous = committed
            logical_checks += 1; physical_entries += physical.size
            reasons[row['policy']][decision['reason']] += 1
        assert np.array_equal(final, previous) and row['migration_bytes'] == total_bytes
        assert row['accepted'] == sum(d['allowed'] for d in log)
        assert isinstance(row['migration_sent_messages'], int) and row['migration_sent_messages'] >= 0
        if row['policy'] == 'SKIP': assert total_bytes == 0 and row['migration_sent_messages'] == 0 and np.array_equal(final, initial)
        for correctness in (row['preflight_correctness'], row):
            assert correctness['correctness_pass'] is True and correctness['correctness_atol'] == .02 and correctness['correctness_rtol'] == .03
            checks = correctness['correctness_checks']
            assert [c['raw_step'] for c in checks] == [32, 543]
            assert correctness['correctness_max_abs_error'] == max(c['fp32_max_abs_error'] for c in checks)
            for check in checks:
                assert check['reference_tokens'] == 16 and check['native_forward_batch_tokens'] == row['batch_tokens']
                assert check['reference_token_positions'] == [i*(row['batch_tokens']-1)//15 for i in range(16)]
                assert check['full_batch_default_comparison'] is True
                assert np.isfinite(check['fp32_max_abs_error']) and check['fp32_max_abs_error'] >= 0
                assert np.isfinite(check['full_batch_default_max_abs_difference']) and check['full_batch_default_max_abs_difference'] >= 0
                maximum_error = max(maximum_error, check['fp32_max_abs_error'])
                maximum_default_delta = max(maximum_default_delta, check['full_batch_default_max_abs_difference'])
                numerical_forward_checks += 1
        contract = (row['source_weights_sha256'], row['source_map_sha256'], row['final_map_sha256'],
                    row['decision_trajectory_sha256'], row['migration_bytes'], log)
        state = (*key, row['policy'])
        if state in states: assert states[state] == contract, 'Kernel/repetition changed the actual policy trajectory'
        states[state] = contract
        quartets[(*key, row['repetition'])].append(row)
        summary = summaries[(row['kernel_variant'], row['policy'])]
        summary.update(runs=1, decisions=len(log), accepted=row['accepted'], rejected=len(log)-row['accepted'],
                       sender_bytes=total_bytes, terminal_proposals_suppressed=1,
                       final_map_changed_runs=int(not np.array_equal(initial, final)))
    for quartet in quartets.values():
        assert len(quartet) == 4
        for field in ('capture_sha256', 'replay_sha256', 'source_map_sha256', 'source_weights_sha256', 'allocation_id', 'hostname', 'process_run_id'):
            assert all(r[field] == quartet[0][field] for r in quartet), field
        first = [r['decision_log'][0] for r in quartet]
        for field in ('current_counts', 'current_map', 'proposed_map', 'physical_slot_assignments_window'):
            assert all(d[field] == first[0][field] for d in first), 'First common proposal state differs'
    return {'status': 'verified', 'phase': manifest['phase'], 'split': manifest['split'],
        'manifest_sha256': config_hash, 'records_sha256': hashlib.sha256(records_path.read_bytes()).hexdigest(),
        'records': len(rows), 'matched_quartets': len(quartets), 'unique_policy_trajectories': len(states),
        'banks_verified': len(banks), 'replays_verified': len(tapes),
        'actual_logged_windows_checked': logical_checks, 'logged_observed_steps_checked': logical_checks*16,
        'logical_counter_entries_checked': logical_checks*128, 'physical_counter_entries_checked': physical_entries,
        'unique_physical_windows_independently_reconstructed': len(physical_cache),
        'complete_map_arrays_checked': map_checks, 'slots_per_map': 160, 'required_logical_expert_coverage_per_map': 128,
        'negative_logical_or_physical_count_entries': 0, 'negative_or_missing_logical_expert_map_entries': 0,
        'all_current_counts_equal_exact_captured_windows_times_two': True,
        'all_native_physical_assignments_equal_cpu_reconstruction': True,
        'all_accepted_map_chains_and_actual_sender_payloads_consistent': True,
        'all_kernel_and_repetition_policy_trajectories_exact': True,
        'terminal_step_511_proposals_absent': True, 'terminal_suppressions_verified': len(rows),
        'policy_summaries': [{'kernel_variant': k[0], 'policy': k[1], **dict(v)} for k, v in sorted(summaries.items())],
        'decision_reasons': {p: dict(c) for p, c in reasons.items()},
        'numerical': {'native_forward_check_count': numerical_forward_checks, 'reference_tokens_per_check': 16,
            'checked_raw_steps': [32, 543], 'full_batch_default_comparisons': True,
            'maximum_recorded_fp32_absolute_error': maximum_error, 'maximum_recorded_default_absolute_difference': maximum_default_delta,
            'atol': .02, 'rtol': .03},
        'allocations': sorted({r['allocation_id'] for r in rows}), 'hostnames': sorted({r['hostname'] for r in rows}),
        'performance_values_analyzed': False,
        'limits': ['Exact raw routing reconstruction covers every logged decision window: seven 16-step windows per run. It does not add an observation record for the suppressed terminal window or unmonitored intervals.',
            'Map/weight identities and completed GPU numerical checks are execution attestations; this CPU audit does not rerun expert GEMMs or independently inspect GPU weight tensors.',
            'Numerical checks use full native batches versus DEFAULT plus 16 FP32 reference tokens at each of two raw endpoints, before and after timing. They do not check every measured output.',
            'The trace is constructed repeated-prefill replay on the named replicated-input/allreduce path, not production serving traffic. No performance ratios are computed by this auditor.']}


def main():
    if not __debug__: raise SystemExit('Run without -O: validation assertions must remain enabled.')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--records', type=Path, required=True, help='Complete raw JSONL or JSONL.GZ matrix')
    p.add_argument('--capture-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='Audit JSON file; a companion .md is written')
    args = p.parse_args()
    result = audit(args.manifest, args.records, args.capture_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    lines = [f"Independent optimization routing audit: {result['status']}.", '',
        f"Verified {result['records']} runs in {result['matched_quartets']} matched quartets. Reconstructed all {result['actual_logged_windows_checked']} logged windows from actual captured logical IDs, including {result['physical_counter_entries_checked']:,} nonnegative physical counter entries. Checked {result['complete_map_arrays_checked']:,} complete maps of 160 slots, each retaining all 128 logical experts.", '',
        'Current logical counts, exact token-index replica assignments, physical-to-logical conservation, accepted map chains, native sender bytes, and cross-kernel/repetition trajectories agree. Each run suppresses the step 511 terminal proposal. No elapsed-time values were analyzed.', '',
        '| Kernel | Policy | Runs | Accepted decisions | Rejected decisions | Sender bytes |',
        '|---|---|---:|---:|---:|---:|']
    for row in result['policy_summaries']:
        lines.append(f"| {row['kernel_variant']} | {row['policy']} | {row['runs']} | {row['accepted']} | {row['rejected']} | {row['sender_bytes']:,} |")
    lines += ['', f"Maximum recorded FP32 absolute error: {result['numerical']['maximum_recorded_fp32_absolute_error']:.9g}. Numerical scope and remaining limits:", '']
    lines += ['- '+item for item in result['limits']]
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('limits',)}, indent=2))


if __name__ == '__main__': main()

"""Independent CPU audit of resident-dispatch pilot JSONL and saved route arrays.

No production assignment, runner, or analysis imports. This verifies route/state
contracts; it neither estimates speedup nor certifies runtime optimality.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

ARMS = ('native', 'native_aa', 'pre_native', 'one_copy_home', 'token_balanced')
METHOD = {a: ('native' if a in ARMS[:3] else a) for a in ARMS}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def array_sha(x):
    return hashlib.sha256(np.asarray(x, dtype=np.int64).tobytes()).hexdigest()


def canonical(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'),
                                   allow_nan=False).encode()).hexdigest()


def locate(name, directory):
    path = Path(name)
    if path.is_file():
        return path
    direct = directory / path.name
    if direct.is_file():
        return direct
    matches = list(directory.rglob(path.name))
    if len(matches) != 1:
        raise AssertionError(f'Expected one local artifact for {name}, found {len(matches)}')
    return matches[0]


def exact_native(logical, mapping):
    result = np.empty_like(logical, dtype=np.int64)
    slots = {e: [slot for slot, value in enumerate(mapping) if value == e]
             for e in set(logical.ravel().tolist())}
    # Hash token position, not route position, and reset at every input batch.
    for step in range(logical.shape[0]):
        for token in range(logical.shape[1]):
            h = (token * 2654435769) & 0xFFFFFFFF
            for choice in range(logical.shape[2]):
                copies = slots[int(logical[step, token, choice])]
                result[step, token, choice] = copies[h % len(copies)]
    return result


def count_rows(physical, slots):
    return np.stack([np.bincount(row.ravel(), minlength=slots) for row in physical])


def check_setup_metadata(row, manifest=None):
    """Check the optional second apparatus contract without rewriting old runs.

    The first frozen apparatus did not emit these fields. A revision-two
    manifest makes every revision-two attestation mandatory; omitting the row's
    revision marker must not silently downgrade the verification.
    """
    revision = row.get('measurement_setup_revision', 1)
    assert type(revision) is int and revision in (1, 2), 'Unknown setup revision'
    if manifest is not None:
        declared = manifest.get('pilot', {}).get('measurement_setup_revision', 1)
        assert revision == declared, 'Record and frozen setup revisions differ'
    if revision == 2:
        assert row['timing_events_initialized_before_timer'] is True
        assert row['full_policy_conditioning_steps'] == 64
        assert row['full_policy_conditioning_calls'] == 1
        assert row['conditioning_raw_indices'] == list(range(16, 48)) * 2
        assert row['gc_enabled_during_measurement'] is True
        duration = row['instrumentation_and_conditioning_seconds_excluded']
        assert (type(duration) in (int, float) and np.isfinite(duration)
                and duration > 0), 'Invalid excluded conditioning duration'
    return revision


def audit_artifact(path, expected_sha, batch, bank=None):
    assert sha(path) == expected_sha, f'Artifact SHA mismatch: {path}'
    with np.load(path, allow_pickle=False) as data:
        indices = data['indices'].copy()
        logical = data['logical_ids'].copy()
        mapping = data['physical_to_logical_map'].copy()
        metadata = json.loads(str(data['metadata']))
        physical = {name: data[f'{name}_physical_ids'].copy()
                    for name in ('native', 'one_copy_home', 'token_balanced')}
    assert indices.shape == (48, batch)
    assert logical.shape == (48, batch, 8)
    assert np.array_equal(mapping, np.r_[np.arange(128), np.arange(32)])
    assert np.all(logical >= 0) and np.all(logical < 128)
    assert all(len(np.unique(row)) == batch for row in indices)
    if bank is not None:
        assert sha(bank) == metadata['capture_sha256']
        with np.load(bank, allow_pickle=False) as data:
            assert np.array_equal(data['expert_ids'][indices], logical)
            source_metadata = json.loads(str(data['metadata']))
            assert source_metadata['source_prompt_ids'] == metadata['source_prompt_ids']
            assert source_metadata['correctness_pass'] is True
    assert np.array_equal(physical['native'], exact_native(logical, mapping))
    counts = {}
    for method, assignments in physical.items():
        assert assignments.shape == logical.shape
        assert np.issubdtype(assignments.dtype, np.integer)
        assert np.all(assignments >= 0) and np.all(assignments < 160)
        assert np.array_equal(mapping[assignments], logical)
        counts[method] = count_rows(assignments, 160)
        metrics = metadata['metrics'][method]
        assert len(metrics) == 48
        for step, (ids, slots, c, declared) in enumerate(zip(logical, assignments, counts[method], metrics)):
            assert int(c.sum()) == ids.size
            reconstructed = np.zeros(128, dtype=np.int64)
            for slot, number in enumerate(c):
                reconstructed[int(mapping[slot])] += number
            assert np.array_equal(reconstructed, np.bincount(ids.ravel(), minlength=128))
            rank_counts = c.reshape(2, 80)
            block_m = declared['block_size_m']
            assert block_m in (64, 128)
            computed = {'physical_counts': c.tolist(),
                        'logical_counts': reconstructed.tolist(),
                        'per_rank_tokens': rank_counts.sum(1).tolist(),
                        'per_rank_active_copies': np.count_nonzero(rank_counts, axis=1).tolist(),
                        'per_rank_m_tiles': ((rank_counts + block_m - 1)//block_m).sum(1).tolist()}
            computed['per_rank_padded_rows'] = [n*block_m for n in computed['per_rank_m_tiles']]
            for key, value in computed.items():
                assert declared[key] == value, (method, step, key)
            active = np.unique(ids)
            f0 = sum(np.all(np.flatnonzero(mapping == e) < 80) for e in active)
            f1 = sum(np.all(np.flatnonzero(mapping == e) >= 80) for e in active)
            minimum_active_peak = max(f0, f1, (len(active)+1)//2)
            assert declared['minimum_possible_max_rank_active_copies'] == minimum_active_peak
            if method == 'one_copy_home':
                assert all(len(np.unique(slots[ids == e])) == 1 for e in active)
                assert max(computed['per_rank_active_copies']) == minimum_active_peak
            if method == 'token_balanced':
                forced = [0, 0]
                flexible = 0
                for e, number in enumerate(reconstructed):
                    owners = {int(s)//80 for s in np.flatnonzero(mapping == e)}
                    if len(owners) == 1:
                        forced[next(iter(owners))] += int(number)
                    else:
                        flexible += int(number)
                optimum = min(max(forced[0]+x, forced[1]+flexible-x)
                              for x in range(flexible+1))
                assert max(computed['per_rank_tokens']) == optimum
    return {'logical': logical, 'mapping': mapping, 'physical': physical,
            'counts': counts, 'metadata': metadata, 'path': path, 'sha256': expected_sha}


def audit(records, artifacts_dir, manifest=None, banks_dir=None, allow_partial=False):
    rows = []
    for path in records:
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt') as f:
            rows += [json.loads(line) for line in f if line.strip()]
    assert rows, 'No records'
    ids, cache, trajectories, orders = set(), {}, {}, {}
    setup_revisions = {}
    manifest_digest = None
    if manifest:
        manifest_digest = canonical({k: v for k, v in manifest.items() if k != 'freeze'})
        assert manifest['freeze']['sha256'] == manifest_digest
    configurations = {r['config_hash'] for r in rows}
    assert len(configurations) == 1
    if manifest_digest:
        assert configurations == {manifest_digest}
    for row in rows:
        revision = check_setup_metadata(row, manifest)
        setup_revisions[str(revision)] = setup_revisions.get(str(revision), 0) + 1
        assert row['record_id'] not in ids, 'Duplicate record'
        ids.add(row['record_id'])
        assert row['resident_dispatch_schema_version'] == 1
        assert row['status'] == 'complete' and row['phase'] == row['split'] == 'pilot'
        assert row['measurement_kind'] == 'cyclic_fixed_placement_dispatch_headroom'
        for flag in ('evidence_eligible','correctness_pass','source_restore_verified',
                     'expected_arm_weights_verified','commit_completed','logical_routing_preserved',
                     'rank_assignment_agreement'):
            assert row[flag] is True, flag
        assert row['steps_completed'] == 256
        assert row['migration_bytes'] == row['migration_sent_messages'] == 0
        assert row['terminal_proposals_suppressed'] == 1
        assert row['correctness_atol'] == .02 and row['correctness_rtol'] == .03
        assert row['preflight_correctness']['full_batch_inputs_checked'] == 48
        assert row['full_batch_inputs_checked'] == 2
        assert row['correctness_pass'] and row['preflight_correctness']['correctness_pass']
        elapsed = row['per_rank_elapsed_seconds']
        assert len(elapsed) == 2 and all(np.isfinite(x) and x > 0 for x in elapsed)
        assert row['elapsed_seconds'] == max(elapsed)
        step_times = np.asarray(row['step_seconds_by_rank'])
        assert step_times.shape == (2,256) and np.all(np.isfinite(step_times)) and np.all(step_times > 0)
        arm = row['arm']
        assert arm in ARMS and row['assignment_method'] == METHOD[arm]
        assert row['assignment_selection_cost_included'] == (arm in ('native','native_aa'))
        artifact = row['assignment_artifact']
        key = (row['block_id'], row['domain'], row['batch_tokens'])
        if key not in cache:
            bank = None
            if banks_dir is not None:
                assert manifest is not None, 'Bank verification requires manifest'
                entry = next(e for e in manifest['banks'] if e['block_id'] == row['block_id'])
                bank = locate(entry['capture_path'], banks_dir)
                assert sha(bank) == entry['capture_sha256']
            cache[key] = audit_artifact(locate(artifact['path'], artifacts_dir),
                                       artifact['sha256'], row['batch_tokens'], bank)
        saved = cache[key]
        assert artifact['sha256'] == saved['sha256']
        assert row['capture_sha256'] == saved['metadata']['capture_sha256']
        assert row['source_prompt_ids'] == saved['metadata']['source_prompt_ids']
        method = METHOD[arm]
        assert row['assignment_sha256'] == array_sha(saved['physical'][method])
        assert row['source_map'] == row['final_map'] == saved['mapping'].tolist()
        assert row['source_map_sha256'] == array_sha(saved['mapping'])
        assert row['assignment_metrics'] == saved['metadata']['metrics'][method][16:]
        expected_window = saved['counts'][method][32:48]
        assert np.array_equal(row['terminal_physical_window'], expected_window)
        logs = row['decision_log']
        assert [d['step'] for d in logs] == [63,127,191]
        expected_logical = np.bincount(saved['logical'][32:48].ravel(), minlength=128)*2
        for d in logs:
            assert d['allowed'] is False and d['commit_completed'] is True
            assert d['current_map'] == d['committed_map'] == row['source_map']
            assert d['transfer_sent_bytes'] == 0
            assert d['physical_counter_input_multiplicity'] == 2
            assert d['physical_window_measured_step_range'] == [d['step']-15,d['step']]
            assert np.array_equal(d['current_counts'], expected_logical)
            assert np.array_equal(d['physical_slot_assignments_window'], expected_window)
            assert np.array_equal(d['physical_owner_assignments_per_step'],
                                  expected_window.reshape(16,2,80).sum(2))
        trajectory = [{k: d[k] for k in ('step','current_counts','current_map','proposed_map','committed_map')}
                      for d in logs]
        assert canonical(trajectory) == row['logical_trajectory_sha256']
        source_identity = (row['logical_trajectory_sha256'], row['source_weights_sha256'])
        if key in trajectories:
            assert trajectories[key] == source_identity
        trajectories[key] = source_identity
        order_key = (*key, row['repetition'])
        order = row['arm_order']
        assert sorted(order) == sorted(ARMS) and row['arm_position'] == order.index(arm)
        if order_key in orders:
            assert orders[order_key] == order
        orders[order_key] = order
    expected = None
    if manifest:
        m = manifest['matrix']
        expected = {(b,d,t,r,a) for b in m['block_ids'] for d in m['domains']
                    for t in m['batch_tokens'] for r in range(m['repetitions']) for a in m['arms']}
        actual = {(r['block_id'],r['domain'],r['batch_tokens'],r['repetition'],r['arm']) for r in rows}
        assert actual <= expected
        if not allow_partial:
            assert actual == expected, f'Incomplete matrix: {len(actual)}/{len(expected)}'
    elif not allow_partial:
        assert len(rows) == 80 and len(cache) == 8
    for key, order in orders.items():
        if key[-1] == 0 and (*key[:-1],1) in orders:
            assert orders[(*key[:-1],1)] == list(reversed(order))
    return {'schema_version':1, 'audit_pass':True, 'records':len(rows),
            'unique_source_cells':len(cache), 'artifact_count':len(cache),
            'expected_records':len(expected) if expected is not None else 80,
            'capture_banks_verified':banks_dir is not None,
            'measurement_setup_revision_counts':setup_revisions,
            'revision2_setup_metadata_checked':bool(setup_revisions.get('2', 0)),
            'checks':['artifact SHA','native scalar hash','logical route eligibility/conservation',
                      'one-home active-count optimum','token-only balance optimum',
                      'all per-step route diagnostics','three decision windows and terminal window',
                      'zero migration/fixed committed maps','native planning trajectory agreement',
                      'local paired order reversal','completed timer metadata'] +
                     (['revision2 initialized events,64-step conditioning and GC metadata']
                      if setup_revisions.get('2', 0) else []),
            'limitations':['No runtime optimality or speedup inference.',
                           'One-home token tie-break optimality covered by separate exhaustive unit tests.',
                           'GPU resident-weight content relies on runtime exact-check attestations; no weights copied.'],
            'artifacts':[{'path':str(v['path']),'sha256':v['sha256']} for v in cache.values()]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records',nargs='+',type=Path,required=True)
    p.add_argument('--artifacts-dir',type=Path,required=True)
    p.add_argument('--manifest',type=Path)
    p.add_argument('--banks-dir',type=Path)
    p.add_argument('--allow-partial',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    result=audit(a.records,a.artifacts_dir,
                 json.loads(a.manifest.read_text()) if a.manifest else None,
                 a.banks_dir,a.allow_partial)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f: json.dump(result,f,indent=2); f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('artifacts','limitations','checks')}))


if __name__=='__main__': main()

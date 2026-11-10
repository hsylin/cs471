#!/usr/bin/env python3
"""Independent complete-record reconstruction of the ordering supplement.

Uses only stdlib, NumPy and the archived independent auditors from the parent
study. No production runner, input builder or analysis module is imported.
This is a separate implementation within the same team, not external replay.
"""
import argparse
from collections import Counter
import hashlib
import importlib
import itertools
import json
import math
from pathlib import Path
import sys
import tarfile

import numpy as np

ARMS = ('stockoff', 'stockon32', 'fix')
METRICS = ('response_mean_ns', 'response_p95_ns', 'service_mean_ns',
           'queue_mean_ns', 'completed_elapsed_ns', 'drain_ns',
           'throughput_jobs_per_second')


def sha(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True, separators=(',', ':'),
                           allow_nan=False).encode()
    return hashlib.sha256(value).hexdigest()


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def array_sha(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(sha([a.dtype.str, list(a.shape)]).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def sealed(p):
    return p.get('content_sha256') == sha({k: v for k, v in p.items()
                                         if k != 'content_sha256'})


def assert_same(a, b, label, differences):
    if isinstance(a, dict):
        require(isinstance(b, dict) and set(a) == set(b), label + ' keys')
        for k, value in a.items():
            assert_same(value, b[k], label + '.' + k, differences)
    elif isinstance(a, list):
        require(isinstance(b, list) and len(a) == len(b), label + ' length')
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, f'{label}[{i}]', differences)
    elif isinstance(a, bool) or a is None or isinstance(a, str):
        require(type(a) is type(b) and a == b, label + ' value')
    elif isinstance(a, (int, float)):
        require(isinstance(b, (int, float)) and not isinstance(b, bool)
                and math.isfinite(a) and math.isfinite(b), label + ' numeric')
        difference = abs(a - b)
        differences.append(difference)
        if type(a) is int:
            require(type(b) is int and a == b, label + ' exact integer')
        else:
            require(math.isclose(a, b, rel_tol=2e-12, abs_tol=2e-9),
                    label + f' float difference {difference}')
    else:
        raise TypeError(label)


def verify_plan(plan, parent, root):
    require(sealed(plan) and plan['schema'] == 'sequence_study_plan_v1', 'Plan seal')
    require(sealed(parent) and parent['content_sha256'] == plan['parent_plan_sha256'],
            'Parent frozen plan')
    require(plan['phase'] == 'supplement' and plan['jobs'] == 4096
            and plan['batch_tokens'] == 128 and plan['arms'] == list(ARMS), 'Scope')
    require(plan['seed'] == 471901 and plan['input_seed'] == 471902, 'Seeds')
    require(plan['selected_fix'] == parent['selected_fix']
            and plan['hardware'] == parent['hardware'], 'Unchanged selection/hardware')
    require(plan['timing_selection'] is False and plan['technical_repetitions'] == 2,
            'No outcome selection / technical repetition count')
    require(plan['source_sha256'] == sha(plan['source_files']), 'Source manifest')
    for rel, value in plan['source_files'].items():
        require(sha((root / rel).read_bytes()) == value, 'Source changed: ' + rel)
    require(sha((root / 'docs/sequence-study/protocol-v1.md').read_bytes())
            == plan['protocol_sha256'], 'Protocol bytes')
    chosen = []
    for ds in ('gsm8k_train', 'wildchat'):
        candidates = [b for b in parent['banks'] if b['dataset'] == ds]
        chosen += sorted(candidates, key=lambda b: sha([471901, b['block_id']]))[:2]
    require(plan['banks'] == chosen, 'Hash-only content selection')
    expected = []
    permutations = list(itertools.permutations(ARMS))
    for i, bank in enumerate(chosen):
        gid = bank['block_id']
        typ = ('code', 'conversation')[i % 2]
        case = next(c for c in parent['cases'] if c['block_id'] == gid
                    and c['arrival_type'] == typ and c['rho'] == .85)
        require(plan['calendars'][gid] == case, 'Identical original calendar')
        modes = ['contiguous', 'mixed'] + (['batch_shuffle'] if i == 0 else [])
        for rep in (0, 1):
            visit = modes if (i + rep) % 2 == 0 else list(reversed(modes))
            for mode in visit:
                order = list(permutations[(i + modes.index(mode)) % 6])
                if rep:
                    order.reverse()
                expected += [dict(block_id=gid, mode=mode, rep=rep, arm=arm,
                                  arm_order=order) for arm in order]
    require(plan['trials'] == expected and len(expected) == 54, 'Frozen complete order')
    return {(t['block_id'], t['mode'], t['rep'], t['arm']): t for t in expected}


def saved_inputs(run, gid, mode, meta):
    path = run / 'inputs' / (gid + '-' + mode + '.npz')
    require(sha(path.read_bytes()) == meta['index_artifact_sha256'], 'Input file bytes')
    with np.load(path, allow_pickle=False) as z:
        indices = z['indices'].copy()
        stored = json.loads(str(z['metadata']))
    require(indices.shape == (4112, 128) and indices.dtype == np.dtype('int64'), 'Indices geometry')
    require(np.all(indices >= 0) and np.all(indices < meta['source_records']), 'Indices bounds')
    require(array_sha(indices) == meta['indices_sha256'] == stored['indices_sha256'], 'Indices digest')
    require(all(meta[k] == v for k, v in stored.items()), 'Saved index metadata')
    require(np.all(np.diff(np.sort(indices, axis=1), axis=1) > 0), 'No duplicate record within job')
    uses = np.bincount(indices[16:].ravel(), minlength=meta['source_records']).astype(np.int64)
    require(array_sha(uses) == meta['measured_occurrence_multiset_sha256']
            and uses.tolist() == meta['token_use_counts'], 'Actual occurrence multiset')
    require(array_sha(indices[:16]) == meta['warmup_indices_sha256'], 'Actual common conditioning')
    require(meta['fresh_content'] is False and meta['fresh_arrival_block'] is False
            and meta['natural_batch_trace'] is False, 'Scope labels')
    routes = run / 'inputs' / (gid + '-' + mode + '-stock-routes.npz')
    with np.load(routes, allow_pickle=False) as z:
        ids = z['logical_ids'].copy()
        weights = z['combining_weights'].copy()
        ih, wh = str(z['logical_routing_sha256']), str(z['float32_combining_weights_sha256'])
    require(ids.dtype == np.dtype('int32') and weights.dtype == np.dtype('float32')
            and ids.shape == weights.shape == (4112, 128, 8), 'Saved native routes geometry')
    require(array_sha(ids) == ih and array_sha(weights) == wh, 'Actual native route hashes')
    require(np.all((ids >= 0) & (ids < 128))
            and np.all(np.diff(np.sort(ids, axis=-1), axis=-1) > 0), 'Logical route conservation')
    require(np.isfinite(weights).all() and (weights >= 0).all()
            and np.allclose(weights.sum(-1), 1, atol=2e-6, rtol=0), 'Actual combining weights')
    return dict(logical_routing_sha256=ih, float32_combining_weights_sha256=wh,
                logical_ids=ids[16:], indices=indices, uses=uses,
                row_metadata_sha256=sha(meta),
                files={path.name: sha(path.read_bytes()), routes.name: sha(routes.read_bytes())})


def independent_statistics(index, metrics, banks):
    pairs = []
    breaks = []
    for key in index:
        gid, mode, rep, arm = key
        if arm == 'stockoff':
            continue
        off = (gid, mode, rep, 'stockoff')
        ratios = {m: metrics[key][m] / metrics[off][m]
                  if metrics[off][m] > 0 else None for m in METRICS}
        pairs.append(dict(block_id=gid, dataset=index[key]['dataset'], mode=mode,
                          rep=rep, arm=arm, ratios=ratios))
        cumulative = []
        total = 0
        for s, f, os, of in zip(index[key]['coordinator_start_ns'],
                                index[key]['coordinator_finish_ns'],
                                index[off]['coordinator_start_ns'],
                                index[off]['coordinator_finish_ns']):
            total += (f - s) - (of - os)
            cumulative.append(total)
        # Last positive prefix identifies the earliest prefix whose entire
        # remaining suffix is nonpositive, expressed independently of cumsum.
        last_positive = max((i for i, value in enumerate(cumulative) if value > 0), default=-1)
        first = last_positive + 1 if last_positive < 4095 else None
        breaks.append(dict(block_id=gid, mode=mode, rep=rep, arm=arm,
            cumulative_service_difference_ns=cumulative, terminal_difference_ns=total,
            first_job_remaining_nonpositive=first,
            interpretation='Episode-level cumulative service difference versus off; includes different controller/slot geometry; not isolated migration payback'))
    bundles = []
    for gid, mode, arm in sorted({(p['block_id'], p['mode'], p['arm']) for p in pairs}):
        selected = [p for p in pairs if (p['block_id'], p['mode'], p['arm']) == (gid, mode, arm)]
        require(len(selected) == 2 and {p['rep'] for p in selected} == {0, 1}, 'Paired repetitions')
        ratios = {}
        for m in METRICS:
            a, b = (p['ratios'][m] for p in selected)
            ratios[m] = None if a is None or b is None else math.sqrt(a * b)
        bundles.append(dict(block_id=gid, dataset=selected[0]['dataset'], mode=mode,
            arm=arm, ratios=ratios,
            response_faster_in_both_repetitions=all(p['ratios']['response_mean_ns'] < 1 for p in selected),
            response_slower_in_both_repetitions=all(p['ratios']['response_mean_ns'] > 1 for p in selected)))
    interactions = []
    for bank in banks:
        for arm in ('stockon32', 'fix'):
            selected = {p['mode']: p for p in bundles if p['block_id'] == bank['block_id'] and p['arm'] == arm}
            ratios = {}
            for m in METRICS:
                c, x = selected['contiguous']['ratios'][m], selected['mixed']['ratios'][m]
                ratios[m] = None if c is None or x is None or x == 0 else c / x
            interactions.append(dict(block_id=bank['block_id'], arm=arm,
                                     contiguous_over_mixed_relative_effect=ratios))
    return pairs, bundles, interactions, breaks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('repo-root', 'plan', 'records', 'analysis', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    root, run = a.repo_root.resolve(), a.records.resolve()
    helpers = root / 'results/resident-residual/auditors'
    sys.path.insert(0, str(helpers))
    stats = importlib.import_module('audit_residual_statistics')
    stock = importlib.import_module('residual_stock_primitives')
    resident = importlib.import_module('audit_residual_records')
    plan = json.loads(a.plan.read_text())
    parent = json.loads((root / 'configs/resident-residual-final-v1.json').read_text())
    expected = verify_plan(plan, parent, root)
    archive = run / 'provenance/executed-source.tar.gz'
    with tarfile.open(archive) as tar:
        for rel, value in plan['source_files'].items():
            require(sha(tar.extractfile(rel).read()) == value, 'Archived execution source ' + rel)
        require(sha(tar.extractfile('docs/sequence-study/protocol-v1.md').read())
                == plan['protocol_sha256'], 'Archived protocol')
    rows, raw_files = stats.read_rows(run)
    require(len(rows) == len(expected) == 54, 'Complete raw trial matrix')
    datasets = {b['block_id']: b['dataset'] for b in plan['banks']}
    banks = {b['block_id']: b for b in plan['banks']}
    index, metrics, artifacts = {}, {}, {}
    native_windows = 0
    for row in rows:
        key = (row['block_id'], row['mode'], row['technical_rep'], row['arm'])
        require(key in expected and key not in index, 'Unique complete trial identity')
        require(row['record_id'] == sha({k: v for k, v in row.items() if k != 'record_id'}), 'Raw row digest')
        require(row['schema'] == 'sequence_study_trial_v1' and row['phase'] == 'supplement'
                and row['status'] == 'complete' and row['evidence_eligible'] is False
                and row['descriptive_supplement'] is True, 'Trial scope')
        require(row['source_sha256'] == plan['source_sha256']
                and row['plan_sha256'] == plan['content_sha256'], 'Trial frozen source/plan')
        gid, mode, rep, arm = key
        require(row['dataset'] == datasets[gid] and row['content_group_id'] == gid, 'Dataset/group identity')
        require(row['arm_order'] == expected[key]['arm_order'], 'Declared arm order')
        hw = plan['hardware']
        require(row['hostname'].split('.')[0] == hw['hostname']
                and sorted(row['gpu_uuids']) == sorted(hw['gpu_uuids'])
                and str(row['allocation_id']) == str(hw['allocation_id']), 'Reserved hardware')
        cal = plan['calendars'][gid]['calendar']
        require(row['calendar'] == cal and row['arrival_offsets_ns'] == cal['arrival_offsets_ns']
                and row['horizon_ns'] == cal['horizon_ns'], 'Common absolute arrivals/horizon')
        rt = row['runtime_metadata']
        require(rt['native_vllm_commit'] == 'db9527a46873454610df6dbedf79a36d6bf1a7f6'
                and rt['model_revision'] == 'ad44e777bcd18fa416d9da3bd8f70d33ebb85d39'
                and rt['world_size'] == 2 and rt['execution_mode'] == 'eager_one_layer', 'Pinned layer/runtime')
        require(rt['tf32_allowed'] is False and rt['float32_matmul_precision'] == 'highest'
                and rt['precision_per_rank'] == [{'allow_tf32': False, 'matmul_precision': 'highest'}] * 2,
                'Reference precision')
        meta = row['input_meta']
        require(meta['mode'] == mode and meta['block_id'] == gid
                and meta['capture_sha256'] == banks[gid]['capture_sha256'], 'Actual input identity')
        if (gid, mode) not in artifacts:
            artifacts[gid, mode] = saved_inputs(run, gid, mode, meta)
        artifact = artifacts[gid, mode]
        require(sha(meta) == artifact['row_metadata_sha256'], 'Paired input metadata identity')
        for field in ('logical_routing_sha256', 'float32_combining_weights_sha256'):
            require(row['preflight'][field] == artifact[field], 'Paired actual native routes')
        require(row['preflight']['correctness_pass'] and row['preflight']['full_batch_inputs_checked'] == 4112
                and row['postflight']['correctness_pass'] and row['postflight']['full_batch_inputs_checked'] == 16,
                'Numerical coverage')
        for check in (row['preflight'], row['postflight']):
            require(check['atol'] == .02 and check['rtol'] == .03
                    and math.isfinite(check['full_batch_max_abs_difference'])
                    and check['full_batch_max_abs_difference'] >= 0, 'Numerical check contract')
        require(all(row['preflight'][name] is True for name in
                    ('logical_ids_equal', 'combining_weights_equal', 'rank_routing_agreement', 'input_unchanged')),
                'Logical and input identity checks')
        require(row['preflight']['fp32_job_indices'] == [16, 765, 3765, 4111]
                and row['preflight']['fp32_tokens_per_job'] == 16
                and row['postflight']['raw_indices'] == np.linspace(16, 4111, 16, dtype=np.int64).tolist(),
                'FP32/postflight sample positions')
        if arm == 'fix':
            kernels = resident.static_check(row)
            resident.development.check_binary_files(kernels, root / 'results/resident-residual/compiled-artifacts')
            require(row['preflight']['online_assignment_inputs_checked'] == 4112
                    and isinstance(row['preflight']['physical_assignment_digest'], str), 'Full resident assignment coverage')
        else:
            native_windows += stock.native_check(row, artifact)['rank_windows_reconstructed']
        metrics[key] = stats.clock_endpoints(row)
        index[key] = row
    require(set(index) == set(expected), 'No missing treatments')
    actual_order = [(r['block_id'], r['mode'], r['technical_rep'], r['arm'])
                    for r in sorted(rows, key=lambda r: r['epoch_monotonic_ns'])]
    require(actual_order == list(expected), 'Actual chronological trial order')
    chronological = sorted(rows, key=lambda r: r['epoch_monotonic_ns'])
    require(all(x['observation_end_ns'] < y['epoch_monotonic_ns']
                for x, y in zip(chronological, chronological[1:])), 'Sequential complete trials')
    for gid in banks:
        modes = [m for g, m in artifacts if g == gid]
        first = artifacts[gid, modes[0]]
        for mode in modes:
            require(np.array_equal(artifacts[gid, mode]['uses'], first['uses']), 'Across-mode occurrence identity')
            require(np.array_equal(artifacts[gid, mode]['indices'][:16], first['indices'][:16]), 'Across-mode warmup identity')
    pairs, bundles, interactions, breaks = independent_statistics(index, metrics, plan['banks'])
    saved = json.loads(a.analysis.read_text())
    require(saved['valid'] is True and saved['records'] == 54 and saved['no_confirmatory_test'] is True,
            'Production complete descriptive analysis')
    differences = []
    for summary in saved['trial_summaries']:
        key = (summary['block_id'], summary['mode'], summary['technical_rep'], summary['arm'])
        assert_same(metrics[key], summary['metrics'], 'trial metrics', differences)
    require(len(saved['trial_summaries']) == 54, 'Production all trials')
    # Sort only for comparison: every raw-record ordering has already been checked.
    def sort_pairs(values):
        return sorted(values, key=lambda x: (x['block_id'], x['mode'], x['rep'], x['arm']))
    assert_same(sort_pairs(pairs), sort_pairs(saved['paired_cases']), 'paired effects', differences)
    assert_same(bundles, saved['bundle_effects'], 'bundle effects', differences)
    assert_same(interactions, saved['ordering_interactions'], 'ordering interactions', differences)
    assert_same(sort_pairs(breaks), sort_pairs(saved['episode_break_even']), 'cumulative service differences', differences)
    dependency_pins = {p.name: sha(p.read_bytes()) for p in helpers.glob('*.py')}
    result = dict(schema='sequence_independent_statistics_v1', passed=True,
        plan_sha256=plan['content_sha256'], source_sha256=plan['source_sha256'], records=54,
        raw_jobs=54 * 4096, content_bundles=4, technical_repetitions=2,
        native_rank_windows_reconstructed=native_windows,
        numeric_comparisons=len(differences), maximum_absolute_numeric_difference=max(differences, default=0),
        raw_files=raw_files, input_files={name: value for item in artifacts.values() for name, value in item['files'].items()},
        compared_analysis_sha256=sha(a.analysis.read_bytes()), executed_source_archive_sha256=sha(archive.read_bytes()),
        program_sha256=sha(Path(__file__).read_bytes()), independent_helper_sha256=dependency_pins,
        bundle_effects=bundles, ordering_interactions=interactions,
        scope='Same-team independent implementation of raw-clock, paired statistical and native lifecycle checks; no production analysis imports or external GPU replication.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('passed', 'records', 'raw_jobs', 'native_rank_windows_reconstructed',
                                           'numeric_comparisons', 'maximum_absolute_numeric_difference')}))


if __name__ == '__main__':
    main()

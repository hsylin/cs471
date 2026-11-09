#!/usr/bin/env python3
"""Independent descriptive audit of completed native EPLB lifecycles.

Outside the frozen execution inventory. Standard library + NumPy only: no
production runtime, policy, analysis or inference imports. This verifies saved
evidence; it neither reruns CUDA nor estimates a treatment effect.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

PIN = 'db9527a46873454610df6dbedf79a36d6bf1a7f6'
ARMS = ('stockoff', 'stockon32', 'ours')
EXPERT_BYTES = 9437184


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(value).hexdigest()


def tensor_sha(array):
    array = np.ascontiguousarray(array)
    h = hashlib.sha256(sha([array.dtype.str, list(array.shape)]).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def load_rows(paths, allow_partial):
    rows, files = [], []
    for path in paths:
        path = Path(path)
        candidates = sorted((path/'trials').glob('*.jsonl*')) if path.is_dir() else [path]
        require(candidates, 'No record files at '+str(path))
        for file in candidates:
            stored = file.read_bytes()
            raw = gzip.decompress(stored) if file.suffix == '.gz' else stored
            discarded = 0
            if allow_partial and raw and not raw.endswith(b'\n'):
                boundary = raw.rfind(b'\n')+1
                discarded = len(raw)-boundary
                raw = raw[:boundary]
            rows.extend(json.loads(line) for line in raw.decode().splitlines() if line.strip())
            files.append(dict(path=str(file), stored_sha256=sha(stored),
                              audited_decompressed_sha256=sha(raw),
                              incomplete_trailing_bytes_excluded=discarded))
    return rows, files


def schedule(n, interval, window, initial_counter, initial_map, events):
    """Simulate pinned order: record, increment, ready-commit, then due-launch.

    A commit in step c changes the map for forward c+1. A tail commit changes no
    measured input. Native pending work defers launches without resetting cadence.
    """
    counter, current = initial_counter, tuple(initial_map)
    pending, event_index = None, 0
    recorded, deferred, active_maps, launches = [], [], [], []
    for step in range(n):
        active_maps.append(current)
        if interval-counter <= window:
            recorded.append(step)
        before = counter
        counter += 1
        if pending is not None and not pending['commit_in_completion_tail'] and pending['commit_step'] == step:
            current = tuple(pending['committed_map'])
            pending = None
        if counter >= interval:
            if pending is not None:
                deferred.append(step)
            else:
                require(event_index < len(events), 'Missing native launch at step '+str(step))
                event = events[event_index]
                require(event['launch_step'] == step, 'Native launch/cadence mismatch')
                require(event['launch_counter_before'] == before, 'Native pre-launch counter mismatch')
                require(tuple(event['source_map']) == current, 'Native source-map chain mismatch')
                selected = recorded[-window:]
                require(event['sampled_step_indices'] == selected, 'Native sampled window indices mismatch')
                require(event['window_valid_steps'] == len(selected), 'Native valid-window size mismatch')
                commit = event['commit_step']
                require(type(commit) is int and step <= commit < n, 'Invalid commit index')
                if event['commit_in_completion_tail']:
                    require(commit == n-1, 'Tail commit must retain final measured step index')
                else:
                    require(commit > step, 'Native cannot consume a newly launched cycle in the same step')
                pending = event
                event_index += 1
                counter = 0
                launches.append((event, selected.copy()))
    require(event_index == len(events), 'Extra native launch after measured inputs')
    if pending is not None:
        require(pending['commit_in_completion_tail'], 'Uncommitted in-loop cycle')
        current = tuple(pending['committed_map'])
    return dict(recorded=recorded, deferred=deferred, counter=counter,
                final_map=list(current), active_maps=active_maps, launches=launches)


def physical_counts(ids, selected, active_maps):
    """Independent native token-index Knuth hash and resident-map lookup."""
    grouped = {}
    for step in selected:
        grouped.setdefault(active_maps[step], []).append(step)
    total = np.zeros(160, np.int64)
    hashed = (np.arange(ids.shape[1], dtype=np.int64)*2654435769) & 0xffffffff
    for mapping, steps in grouped.items():
        mapping = np.asarray(mapping, np.int64)
        replicas = np.bincount(mapping, minlength=128)
        require(np.all(replicas > 0), 'Committed map lost a logical expert')
        inverse = np.full((128, int(replicas.max())), -1, np.int64)
        for expert in range(128):
            inverse[expert, :replicas[expert]] = np.flatnonzero(mapping == expert)
        selected_ids = ids[steps]
        actual = inverse[selected_ids, hashed[None, :, None] % replicas[selected_ids]]
        total += np.bincount(actual.ravel(), minlength=160)
    return total


def get_routes(inputs, row, cache):
    gid = row['content_group_id']
    if cache.get('group') == gid:
        return cache['ids'], cache['artifact']
    path = inputs/(gid+'-stock-routes.npz')
    with np.load(path, allow_pickle=False) as z:
        ids, weights = z['logical_ids'], z['combining_weights']
        require(ids.shape == weights.shape == (4112, 128, 8), 'Actual stock route array shape')
        require(ids.dtype == np.int32 and weights.dtype == np.float32, 'Actual stock route dtype')
        ih, wh = tensor_sha(ids), tensor_sha(weights)
        require(ih == str(z['logical_routing_sha256']) == row['preflight']['logical_routing_sha256'], 'Actual stock logical-ID digest')
        require(wh == str(z['float32_combining_weights_sha256']) == row['preflight']['float32_combining_weights_sha256'], 'Actual stock floating-weight digest')
        require(np.all((ids >= 0) & (ids < 128)), 'Invalid logical expert ID')
        require(np.isfinite(weights).all() and np.all(weights >= 0), 'Invalid combining weights')
    artifact = dict(path=str(path), sha256=sha(path.read_bytes()),
                    logical_routing_sha256=ih, float32_combining_weights_sha256=wh)
    # One content group at a time, rather than retaining all large route arrays.
    cache.clear()
    cache.update(group=gid, ids=ids[16:], artifact=artifact)
    return cache['ids'], artifact


def clocks(row):
    n = row['n_jobs']
    require(n == 4096, 'Unexpected measured horizon')
    require(row['per_job_completion_scope'] == 'current_inference_stream_after_native_step', 'Wrong per-job completion stream scope')
    start = np.asarray(row['coordinator_start_ns'], np.int64)
    finish = np.asarray(row['coordinator_finish_ns'], np.int64)
    rs = np.asarray(row['per_rank_start_ns'], np.int64)
    rf = np.asarray(row['per_rank_gpu_complete_ns'], np.int64)
    require(start.shape == finish.shape == (n,) and rs.shape == rf.shape == (2, n), 'Raw completion clock shape')
    require(np.all(finish >= start) and np.all(start[1:] >= finish[:-1]), 'FCFS completed-job order')
    require(np.all(rs >= start) and np.all(rf >= rs) and np.all(rf <= finish), 'Rank output-completion envelope')
    tail, q, end = (row[k] for k in ('controller_quiescence_start_ns', 'controller_quiescent_ns', 'observation_end_ns'))
    require(finish[-1] <= tail <= q <= end, 'Normal-tail clock order')
    require(row['controller_tail_elapsed_ns'] == q-int(finish[-1]), 'Tail extent accounting')
    require(row['quiescence_call_elapsed_ns'] == q-tail, 'Quiescence-call extent accounting')
    for rank in (0, 1):
        require(rf[rank,-1] <= row['per_rank_controller_tail_start_ns'][rank] <= row['per_rank_controller_quiescent_ns'][rank], 'Rank tail order')
    return start, finish


def audit_row(row, inputs, cache):
    require(row['record_id'] == sha({k:v for k,v in row.items() if k != 'record_id'}), 'Raw record identity')
    require(row['schema'] == 'stock_comparison_trial_v1' and row['status'] == 'complete', 'Incomplete/wrong-schema trial')
    require(row['runtime_metadata']['native_vllm_commit'] == PIN, 'Wrong native source pin')
    start, finish = clocks(row)
    peers, arm = row['per_rank_state'], row['arm']
    require(len(peers) == 2, 'Missing rank state')
    initial = list(range(128)) + ([] if arm == 'stockoff' else list(range(32)))
    for peer in peers:
        for label in ('start', 'end', 'audit'):
            snap = peer[label]
            require(snap['source_map'] == initial, 'Wrong initial placement')
            require(len(snap['final_map']) == len(initial) and set(snap['final_map']) == set(range(128)), 'Illegal committed map')
            require(not snap['pending_result'] and not snap['rebalanced'], 'Outstanding native state')
            require(snap['completion_disposition'] == 'normal_commit_no_discard', 'Non-normal completion')
        require(peer['start']['final_map'] == initial and peer['start']['async_events'] == [], 'Measured state was not reset')
        for flag in ('resident_state_verified', 'rank_map_agreement', 'normal_completion_verified'):
            require(peer['audit'][flag] is True, 'Missing recorded weight/map/completion assertion')
        require(peer['end']['final_map'] == peer['audit']['final_map'], 'Audit changed committed state')
    require(peers[0]['end']['final_map'] == peers[1]['end']['final_map'], 'Final rank map disagreement')
    result = dict(record_id=row['record_id'], phase=row['phase'], arm=arm,
        content_group_id=row['content_group_id'], dataset=row['dataset'],
        arrival_type=row['arrival_type'], rho=row['rho'], normal_completion=True,
        controller_tail_elapsed_ns=row['controller_tail_elapsed_ns'],
        quiescence_call_elapsed_ns=row['quiescence_call_elapsed_ns'], cycles=[])
    if arm != 'stockon32':
        require(all(p['end']['final_map'] == initial and p['end']['migration_sent_bytes_local'] == 0 for p in peers), 'Fixed-placement arm moved')
        require(all(p['end']['async_events'] == [] for p in peers), 'Non-async arm has native async events')
        if arm == 'stockoff':
            require(all(not p['end']['enable_eplb'] and p['end']['controller_step_calls'] == 0 for p in peers), 'Disabled arm has EPLB work')
        else:
            require(all([d['step'] for d in p['end']['decisions']] == list(range(63,4096,64)) for p in peers), 'Ours lost a charged boundary')
            require(all(p['end']['guard']['movement_skips'] == p['end']['guard']['commit_skips'] == 64 for p in peers), 'Ours shortcut pair accounting')
        return result
    e0, e1 = peers[0]['end'], peers[1]['end']
    require(all(p['start']['native_counter'] == 2250 and p['end']['is_async'] and p['end']['async_worker_alive'] for p in peers), 'Native startup/async worker contract')
    require(all(p['end']['window_steps'] == 1000 and p['end']['rebalance_interval'] == 3000 for p in peers), 'Native default window/cadence changed')
    require(all(p['end']['stock_policy_unmodified'] and p['end']['stock_router_unmodified'] for p in peers), 'Stock policy/router modified')
    events = e0['async_events']
    require(bool(events), 'Missing first native startup cycle')
    for event in events:
        for key in ('source_map','committed_map'):
            require(len(event[key]) == 160 and all(type(x) is int for x in event[key])
                    and set(event[key]) == set(range(128)), 'Illegal intermediate native '+key)
        require(type(event['commit_in_completion_tail']) is bool, 'Invalid native tail marker')
        require(all(type(event[k]) is int for k in ('launch_step','launch_counter_before','commit_step','window_valid_steps')), 'Non-integer native lifecycle index')
    shared_keys = ('launch_step','launch_counter_before','window_valid_steps','sampled_step_indices',
                   'source_map','committed_map','commit_step','commit_in_completion_tail','current_counts')
    require([{k:e[k] for k in shared_keys} for e in events] == [{k:e[k] for k in shared_keys} for e in e1['async_events']], 'Rank lifecycle/candidate/count disagreement')
    sim = schedule(4096,3000,1000,2250,initial,events)
    for end in (e0,e1):
        require(end['recorded_step_indices'] == sim['recorded'], 'Actual recording schedule mismatch')
        require(end['deferred_boundary_steps'] == sim['deferred'], 'Actual deferred boundaries mismatch')
        require(end['native_counter'] == sim['counter'] and end['native_valid_steps'] == len(sim['recorded']), 'Final native counters mismatch')
        require(end['final_map'] == sim['final_map'], 'Final map does not follow normal commits')
        require(sum(e['sent_bytes_local'] for e in end['async_events']) == end['migration_sent_bytes_local'], 'Per-cycle/global payload mismatch')
    ids = artifact = None
    if inputs:
        ids, artifact = get_routes(inputs,row,cache)
        terminal = physical_counts(ids,sim['recorded'][-1000:],sim['active_maps'])
        require(all(np.array_equal(p['end']['terminal_physical_window_totals_local'],terminal) for p in peers), 'Terminal actual physical history mismatch')
    for index,(event,selected) in enumerate(sim['launches']):
        require(all(type(p['end']['async_events'][index]['sent_bytes_local']) is int and p['end']['async_events'][index]['sent_bytes_local'] >= 0 for p in peers), 'Invalid payload counter')
        observed = np.asarray(event['current_counts'],np.int64).reshape(-1)
        require(observed.shape == (128,) and np.all(observed >= 0) and int(observed.sum()) == len(selected)*128*8*2, 'Window assignment conservation')
        mixed = [s for s in selected if sim['active_maps'][s] != tuple(event['source_map'])]
        row_cycle = dict(cycle=index,launch_step=event['launch_step'],commit_step=event['commit_step'],
            commit_in_completion_tail=event['commit_in_completion_tail'],
            launch_to_commit_step_distance=event['commit_step']-event['launch_step'],
            sampled_window_steps=len(selected),sampled_first_step=selected[0],sampled_last_step=selected[-1],
            mixed_map_sampled_steps=len(mixed),changed_slots=int(np.count_nonzero(np.asarray(event['source_map']) != event['committed_map'])),
            source_map_sha256=sha(event['source_map']),committed_map_sha256=sha(event['committed_map']),
            sent_bytes_per_rank=[p['end']['async_events'][index]['sent_bytes_local'] for p in peers],
            recovered_native_counts_sha256=sha(event['current_counts']))
        row_cycle['sent_bytes_total'] = sum(row_cycle['sent_bytes_per_rank'])
        require(all(x % EXPERT_BYTES == 0 for x in row_cycle['sent_bytes_per_rank']), 'Partial expert payload outside reviewed BF16 transport')
        commit_endpoint = row['controller_quiescent_ns'] if event['commit_in_completion_tail'] else int(finish[event['commit_step']])
        row_cycle['post_launch_ack_to_commit_ack_or_quiescence_ns'] = commit_endpoint-int(finish[event['launch_step']])
        if ids is not None:
            physical = physical_counts(ids,selected,sim['active_maps'])
            projected = np.bincount(event['source_map'],weights=physical,minlength=128).astype(np.int64)*2
            require(np.array_equal(observed,projected), 'Native physical-history projection mismatch')
            logical = np.bincount(ids[selected].ravel(),minlength=128)*2
            row_cycle['true_logical_histogram_matches_native'] = bool(np.array_equal(observed,logical))
            row_cycle['native_vs_true_logical_count_l1'] = int(np.abs(observed-logical).sum())
        result['cycles'].append(row_cycle)
    result.update(recorded_steps=len(sim['recorded']),deferred_boundary_steps=sim['deferred'],
                  final_counter=sim['counter'],route_artifact=artifact,
                  native_route_reconstruction=ids is not None)
    return result


def describe(values):
    if not values:
        return dict(n=0)
    a=np.asarray(values,dtype=np.float64)
    return dict(n=len(values),minimum=float(a.min()),median=float(np.median(a)),
                p95=float(np.quantile(a,.95,method='linear')),maximum=float(a.max()),sum=float(a.sum()))


def audit(plan,rows,inputs,allow_partial):
    require(plan['content_sha256'] == sha({k:v for k,v in plan.items() if k != 'content_sha256'}), 'Changed plan seal')
    require(plan['source_sha256'] == sha(plan['source_files']), 'Changed execution inventory seal')
    if plan['phase']=='calibration':
        expected={(b['block_id'],None,None,'stockoff') for b in plan['banks']}
    else:
        expected={(c['block_id'],c['calendar']['arrival_type'],c['calendar']['rho'],a) for c in plan['cases'] for a in ARMS}
    seen=set();results=[];errors=[];cache={}
    for row in rows:
        key=(row.get('content_group_id'),row.get('arrival_type'),row.get('rho'),row.get('arm'))
        try:
            require(key in expected and key not in seen, 'Unexpected/duplicate trial')
            seen.add(key)
            require(row['plan_sha256']==plan['content_sha256'] and row['source_sha256']==plan['source_sha256'], 'Trial/plan identity')
            require(row['phase']==plan['phase'], 'Trial phase mismatch')
            require(str(row['allocation_id'])==str(plan['hardware']['allocation_id']) and sorted(row['gpu_uuids'])==sorted(plan['hardware']['gpu_uuids']), 'Hardware identity mismatch')
            results.append(audit_row(row,inputs,cache))
        except (AssertionError,ValueError,KeyError,IndexError,TypeError,OSError) as exc:
            errors.append(dict(key=list(key),error=str(exc)))
    missing=expected-seen
    native=[r for r in results if r['arm']=='stockon32']
    cycles=[c for r in native for c in r['cycles']]
    return dict(schema='independent_stock_lifecycle_audit_v1',valid=not errors and (not missing or allow_partial),
        complete_matrix=not missing,scope='partial apparatus inspection' if allow_partial else 'completed-study lifecycle audit',
        phase=plan['phase'],plan_sha256=plan['content_sha256'],source_sha256=plan['source_sha256'],
        expected_records=len(expected),observed_records=len(rows),audited_records=len(results),
        errors=errors,missing=[list(x) for x in sorted(missing,key=str)],
        actual_native_route_reconstruction_requested=inputs is not None,
        summary=dict(arm_records=dict(Counter(r['arm'] for r in results)),native_trials=len(native),
            native_cycles=len(cycles),nonzero_payload_cycles=sum(c['sent_bytes_total']>0 for c in cycles),
            tail_commit_cycles=sum(c['commit_in_completion_tail'] for c in cycles),
            trials_with_deferred_boundary=sum(bool(r['deferred_boundary_steps']) for r in native),
            mixed_map_cycles=sum(c['mixed_map_sampled_steps']>0 for c in cycles),
            launch_to_commit_step_distance=describe([c['launch_to_commit_step_distance'] for c in cycles]),
            sender_payload_bytes=describe([c['sent_bytes_total'] for c in cycles]),
            ack_interval_ns=describe([c['post_launch_ack_to_commit_ack_or_quiescence_ns'] for c in cycles]),
            native_trial_tail_extent_ns=describe([r['controller_tail_elapsed_ns'] for r in native])),
        trials=results,
        limitations=['Descriptive counts and intervals only; cycles and repeated content/rates are not independent uncertainty units.',
            'This complements the separate full record/calendar/statistics auditors; it does not replace their source-file, calendar or inferential checks.',
            'Launch-to-commit ACK interval includes intervening work, arrivals and coordination; it is not exclusive planner, transfer or bandwidth time.',
            'Current-stream completion and later normal commits show allowed asynchronous progress, not independently measured GPU overlap.',
            'Resident-weight and numerical verification flags are recorded GPU assertions; this audit does not rerun CUDA.',
            'Route reconstruction authenticates saved stock routing bytes, not an independent rerun of gate/top-k.',
            'Mixed-map physical history is reproduced as native executes it; deviation from true logical demand is reported without excluding the native trial.'])


def markdown(result):
    s=result['summary']
    lines=['# Native lifecycle audit','',f"Status: {'PASS' if result['valid'] else 'FAIL'}; {result['scope']}. Audited {result['audited_records']}/{result['expected_records']} trial records.",'',
        f"Native asynchronous arm: {s['native_trials']} trials, {s['native_cycles']} launched and normally committed cycles, {s['nonzero_payload_cycles']} with nonzero sender payload. {s['tail_commit_cycles']} commits occurred in the post-output tail. {s['trials_with_deferred_boundary']} trials deferred a native boundary; {s['mixed_map_cycles']} cycles sampled history spanning a placement change.",'',
        'All interval summaries are descriptive. The ACK interval includes other work and scheduling, and is not network latency or exclusive migration cost. No arm speedup or statistical test is computed.','',
        '| Quantity | Minimum | Median | p95 | Maximum |','|---|---:|---:|---:|---:|']
    for label,key in [('Launch-to-commit step distance','launch_to_commit_step_distance'),('Sender payload bytes/cycle','sender_payload_bytes'),('Post-launch ACK to commit ACK/quiescence (ns)','ack_interval_ns'),('Native trial tail extent (ns)','native_trial_tail_extent_ns')]:
        v=s[key]
        if v['n']:lines.append('| '+label+' | '+' | '.join(f"{v[k]:.6g}" for k in ('minimum','median','p95','maximum'))+' |')
    lines+=['','## Limits','']+['- '+x for x in result['limitations']]
    if result['errors']:lines+=['','## Errors','',json.dumps(result['errors'],indent=2)]
    return '\n'.join(lines)+'\n'


class Tests(unittest.TestCase):
    def event(self,launch,commit,source,after,selected,counter,tail=False):
        return dict(launch_step=launch,commit_step=commit,source_map=source,committed_map=after,
                    sampled_step_indices=selected,window_valid_steps=len(selected),
                    launch_counter_before=counter,commit_in_completion_tail=tail)

    def test_deferred_launch_and_tail_do_not_reset_early(self):
        mapping=[0,1]
        events=[self.event(0,7,mapping,mapping,[0],3),self.event(7,9,mapping,mapping,[6,7],6,True)]
        s=schedule(10,4,2,3,mapping,events)
        self.assertEqual(s['deferred'],[4,5,6]);self.assertEqual(s['counter'],2)
        self.assertEqual(s['recorded'],[0,3,4,5,6,7])
        events[1]['launch_counter_before']=3
        with self.assertRaises(ValueError):schedule(10,4,2,3,mapping,events)

    def test_commit_changes_next_forward_and_can_mix_physical_history(self):
        initial=list(range(128))+list(range(32));after=initial.copy();after[0],after[80]=after[80],after[0]
        events=[self.event(0,3,initial,after,[0],3),self.event(4,5,after,after,[3,4],3,True)]
        s=schedule(6,4,2,3,initial,events)
        self.assertEqual(s['active_maps'][3],tuple(initial));self.assertEqual(s['active_maps'][4],tuple(after))
        ids=np.full((6,1,1),80,np.int64)
        counts=physical_counts(ids,[3,4],s['active_maps'])
        projected=np.bincount(after,weights=counts,minlength=128)
        self.assertEqual((projected[0],projected[80]),(1,1))
        self.assertEqual(np.bincount(ids[[3,4]].ravel(),minlength=128)[80],2)

    def test_native_hash_uses_token_position_not_expert_id(self):
        mapping=tuple(list(range(128))+list(range(32)))
        counts=physical_counts(np.zeros((1,4,1),np.int64),[0],[mapping])
        self.assertEqual((counts[0],counts[128]),(2,2))

    def test_no_same_step_normal_commit_or_dropped_launch(self):
        m=[0,1];e=self.event(0,0,m,m,[0],3)
        with self.assertRaises(ValueError):schedule(2,4,2,3,m,[e])
        with self.assertRaises(ValueError):schedule(2,4,2,3,m,[])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path);p.add_argument('--records',type=Path,nargs='+')
    p.add_argument('--inputs-dir',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--markdown',type=Path);p.add_argument('--allow-partial',action='store_true')
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():raise SystemExit(1)
        return
    if not (a.plan and a.records and a.output):p.error('--plan, --records and --output are required')
    for path in (a.output,a.markdown):
        if path and path.exists():raise FileExistsError('Fresh audit output required: '+str(path))
    rows,files=load_rows(a.records,a.allow_partial)
    result=audit(json.loads(a.plan.read_text()),rows,a.inputs_dir,a.allow_partial)
    result.update(auditor_sha256=sha(Path(__file__).read_bytes()),record_files=files)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if a.markdown:
        a.markdown.parent.mkdir(parents=True,exist_ok=True);a.markdown.write_text(markdown(result))
    print(json.dumps({k:result[k] for k in ('valid','complete_matrix','expected_records','observed_records','audited_records','summary')}))
    if not result['valid']:raise SystemExit(1)


if __name__=='__main__':main()

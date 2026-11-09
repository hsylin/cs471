"""Frozen components on new content and externally derived absolute calendars."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import itertools
import json
from pathlib import Path
import random
import socket
import subprocess
import time
from types import SimpleNamespace

import numpy as np

from generalization.common import validate_calendar
from generalization.common import load_plan, source_identity, canonical_gpu_uuid
from arrival_fix.noop import unchanged_map_fast_path
from generalization.inputs import prepare
from arrival_study.runner import serve
from arrival_study.schedules import calibration_capacity
from optimization.common import canonical_hash, array_hash, make_layer, manifest_hash
from optimization.kernel_config import kernel_override
from resident_dispatch.common import runtime_metadata, numerical_check
from resident_dispatch.runner import reset_observation, make_online
from skewbench.native import Runtime, Weights
from skewbench.runner import finalize_decision_log


def source_key(group):
    return f"{group['block_id']}/{group['regime']}/B{group['batch_tokens']}"


def hardware_check(plan):
    if socket.gethostname().split('.')[0] != plan['hardware']['hostname']:
        raise ValueError('Wrong physical host; no measurement permitted')
    visible = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True)
    uuids = sorted(x.strip() for x in visible.splitlines() if x.strip())
    if uuids != sorted(plan['hardware']['gpu_uuids']):
        raise ValueError(f'Wrong allocated physical GPU pair; no measurement permitted: {uuids}')
    return uuids


def groups(plan):
    result=[dict(block_id=b['block_id'],regime='newcontent',batch_tokens=128,dataset=b['dataset']) for b in plan['banks']]
    random.Random(plan['order_seed']).shuffle(result)
    return result


def cases(plan, group, state_index):
    if plan['phase']=='calibration':return [(None,None,0,['native'])]
    return [(c['source_block'],c['calendar'],0,c['arm_order']) for c in plan['cases'] if c['block_id']==group['block_id']]


def validate_guard(guard_rows, expected, diagnostic):
    if not guard_rows:
        return
    if len(guard_rows) != 2:
        raise AssertionError('Both rank fast-path records required')
    comparable = []
    for g in guard_rows:
        for name in ('movement_calls', 'commit_calls', 'agreement_calls',
                     'local_unchanged_calls', 'unanimously_unchanged_calls'):
            if g[name] != expected:
                raise AssertionError(f'Unexpected fast-path {name}')
        if g['pending_pair'] or g['fallback_calls'] or g['diagnostic_timing'] != diagnostic:
            raise AssertionError('Incomplete or inconsistent fast-path execution')
        skipped = expected if g['mode'] == 'fast' else 0
        if (g['movement_skips'] != skipped or g['commit_skips'] != skipped
                or g['original_movement_calls'] != expected-skipped
                or g['original_commit_calls'] != expected-skipped):
            raise AssertionError('Movement and normal commit did not share the agreed decision')
        comparable.append({k:v for k,v in g.items() if k != 'diagnostics'})
    if comparable[0] != comparable[1]:
        raise AssertionError('Ranks disagree on the complete fast-path trajectory')


def trial(rt, weights, base, plan, cell, group, arm, block, calendar, repetition, order, *, diagnostic=False):
    layer = make_layer(rt, weights, base, 'SKIP')
    dispatch = 'online' if arm == 'combined' else 'native'
    mapper = make_online(rt, layer, group['batch_tokens']) if dispatch == 'online' else None
    if mapper is not None:
        layer.mapping_kernel = mapper
    layer.assert_maps_and_weights()
    noop = lambda raw: None
    preflight = numerical_check(rt, layer, cell['tensors'], cell['references'], cell['reference_steps'], noop)
    if diagnostic:
        raise ValueError('No diagnostic timing arm in this composition study')
    adapter = unchanged_map_fast_path(layer) if arm in ('shortcut', 'combined') else nullcontext(None)
    n = 128 if diagnostic else 256
    sequence = cell['sequence'][:n]
    with adapter as guard:
        setup_start = time.monotonic_ns()
        layer.warming_up = True
        for i, raw in enumerate(cell['warmup']):
            layer.forward(*(t[raw] for t in cell['tensors']), observe=True)
            layer.step(i)
        layer.warming_up = False
        reset_observation(layer)
        serve(rt, layer, cell['tensors'], cell['conditioning'], [0]*64, 1, terminal=False)
        layer.close()
        layer.assert_maps_and_weights()
        if len(layer.decisions) != 1 or layer.decisions[0]['allowed'] or layer.counter.sent_bytes:
            raise AssertionError('Conditioning must retain one complete native SKIP decision')
        if guard is not None:
            guard.reset_counters()
        reset_observation(layer)
        gc.collect()
        rt.sync()
        setup_ns = time.monotonic_ns()-setup_start
        if calendar is None:
            offsets, horizon = [0]*n, 1
        else:
            validate_calendar(calendar)
            offsets, horizon = calendar['arrival_offsets_ns'], calendar['horizon_ns']
        mapper_calls_before = mapper.calls if mapper is not None else 0
        times = serve(rt, layer, cell['tensors'], sequence, offsets, horizon, terminal=True)
        mapper_calls_measured = mapper.calls-mapper_calls_before if mapper is not None else 0
        if mapper_calls_measured != (n if mapper is not None else 0):
            raise AssertionError('Dispatcher must execute exactly once per measured job')
        local_guard = guard.snapshot() if guard is not None else None
    # The native symbols have been restored before all untimed verification.
    decisions = finalize_decision_log(rt, layer)
    expected_steps = list(range(63, n-1, 64))
    if [d['step'] for d in decisions] != expected_steps:
        raise AssertionError('Native planning cadence changed')
    for decision in decisions:
        raw = sequence[decision['step']-15:decision['step']+1]
        expected = cell['counts'][dispatch][raw]
        logical = np.bincount(cell['logical'][raw].ravel(), minlength=128)*rt.world_size
        if (decision['allowed'] or not np.array_equal(decision['physical_slot_assignments_window'], expected)
                or not np.array_equal(decision['current_counts'], logical)):
            raise AssertionError('Native physical/logical observation changed')
    terminal_window = layer.ms.expert_load_window[:, 0, :].cpu().numpy()
    if not np.array_equal(terminal_window, cell['counts'][dispatch][sequence[-16:]]):
        raise AssertionError('Terminal observation changed')
    layer.assert_maps_and_weights()
    final = layer.map_snapshot()
    if (not np.array_equal(final, cell['mapping']) or layer.counter.sent_bytes
            or layer.counter.sent_messages or layer.terminal_proposals_suppressed != 1):
        raise AssertionError('Resident state/zero migration/terminal contract changed')
    if mapper is not None:
        mapper.verify_metadata(layer.layer_state.logical_to_physical_map, layer.layer_state.logical_replica_count, layer.ms.physical_to_logical_map[0])
    post = numerical_check(rt, layer, cell['tensors'], cell['references'], cell['fp32_steps'], noop)
    trajectory = canonical_hash([{k:d[k] for k in ('step', 'current_counts', 'current_map', 'proposed_map', 'committed_map')} for d in decisions])
    setup_times = rt.gather_scalars(setup_ns)
    guard_rows = [None]*rt.world_size
    rt.torch.distributed.all_gather_object(guard_rows, local_guard, group=rt.ep_group.cpu_group)
    guard_rows = [] if local_guard is None else guard_rows
    validate_guard(guard_rows, n//64, diagnostic)
    row = None
    if rt.rank == 0:
        spans = []
        for end in range(63, n, 64):
            begin = end-15
            spans.append(dict(end_step=end, jobs=16,
                scheduled_arrival_span_ns=offsets[end]-offsets[begin],
                service_coverage_span_ns=times['coordinator_finish_ns'][end]-times['coordinator_start_ns'][begin],
                completion_span_ns=times['coordinator_finish_ns'][end]-times['coordinator_finish_ns'][begin]))
        phase = 'diagnostic' if diagnostic else plan['phase']
        row = {**rt.metadata, **group, **times, 'generalization_schema_version':1,
            'phase':phase, 'config_hash':manifest_hash(plan), 'base_config_hash':manifest_hash(base),
            'execution_source_sha256':source_identity()[0], 'arm':arm, 'policy':'SKIP',
            'arrival_block_id':block['id'] if block else 'native-calibration',
            'arrival_seed':plan['arrival_seed'] if block else None, 'repetition':repetition,
            'pattern':calendar['pattern'] if calendar else 'saturated', 'rho':calendar['rho'] if calendar else None,
            'schedule_name':calendar['name'] if calendar else 'saturated',
            'arrival_schedule':calendar, 'arm_order':order, 'status':'complete',
            'evidence_eligible':phase == 'final', 'diagnostic_only':diagnostic,
            'source_map':cell['mapping'].tolist(), 'final_map':final.tolist(),
            **{k:cell[k] for k in ('capture_sha256', 'replay_sha256', 'source_prompt_ids',
                'source_weights_sha256', 'source_weight_rank_sha256', 'resolved_kernel_config')},
            'logical_routing_sha256':array_hash(cell['logical']), 'assignment_sha256':cell['route_hashes'][dispatch],
            'assignment_artifact':cell['artifact'], 'assignment_metrics':[cell['diagnostics'][dispatch][i] for i in sequence],
            'decision_log':decisions, 'decision_trajectory_sha256':trajectory,
            'terminal_physical_window':terminal_window.tolist(), 'observed_window_spans':spans,
            'migration_bytes':0, 'migration_sent_messages':0, 'terminal_proposals_suppressed':1,
            'normal_completion_verified':True, 'resident_state_verified':True, 'rank_assignment_agreement':True,
            'noop_guard_ranks':guard_rows, 'guard_symbols_restored':True,
            'dispatch_mode':dispatch, 'noop_enabled':arm in ('shortcut','combined'),
            'assignment_selection_cost_included':True, 'online_residency_metadata_verified':mapper is not None,
            'mapper_calls_measured':mapper_calls_measured, 'adversarial_checks':cell['adversarial_checks'],
            'measurement_setup_revision':2, 'per_rank_conditioning_setup_ns_excluded':setup_times,
            'warmup_raw_indices':cell['warmup'], 'conditioning_raw_indices':cell['conditioning'],
            'measured_raw_indices':sequence, 'gc_enabled_during_measurement':gc.isenabled(),
            'full_unique_preflight_gpu_routes_checked':len(cell['reference_steps']),
            'content_group_id':group['block_id'],'content_selection_sha256':plan['content_selection_sha256'],
            'job_reuse':cell['job_reuse'],'newinputmeta':cell['newinputmeta'],
            'per_rank_input_preparation_ns_excluded':cell['preparation_ns'],
            'arrival_type':block['trace'] if block else None,
            'arrival_date':block['source_date_utc'] if block else None,
            'rate_kind':calendar['rate_kind'] if calendar else 'saturated', 'preflight_correctness':preflight, **post}
        if phase == 'calibration':
            row['calibration_capacity'] = calibration_capacity(times['coordinator_start_ns'], times['coordinator_finish_ns'])
        row['record_id'] = canonical_hash({k:row[k] for k in ('config_hash', 'phase', 'arrival_block_id',
            'block_id', 'regime', 'batch_tokens', 'arm', 'repetition', 'schedule_name')})
    del layer, mapper
    gc.collect()
    rt.sync()
    rt.barrier()
    return row


def write_row(rt, path, row):
    if rt.rank == 0:
        with path.open('a') as handle:
            handle.write(json.dumps(row, allow_nan=False)+'\n')
        print(json.dumps({k:row[k] for k in ('phase', 'arrival_block_id', 'block_id', 'regime', 'batch_tokens',
                                           'arm', 'schedule_name', 'repetition', 'status')}), flush=True)
    rt.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'banks-dir', 'weights-dir', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    plan, base = load_plan(args.plan)
    uuids = hardware_check(plan)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rt = Runtime()
    try:
        weights = Weights(args.weights_dir)
        runtime_metadata(rt, weights, base)
        clock = time.get_clock_info('monotonic')
        clock_contract = dict(implementation=clock.implementation, monotonic=clock.monotonic, adjustable=clock.adjustable)
        peers = [None]*rt.world_size
        rt.torch.distributed.all_gather_object(peers, (socket.gethostname(), clock_contract, uuids), group=rt.ep_group.cpu_group)
        if rt.world_size != 2 or any(p != peers[0] for p in peers) or not clock.monotonic or clock.adjustable or 'CLOCK_MONOTONIC' not in clock.implementation:
            raise ValueError('Shared physical devices/node/monotonic clock required')
        rt.metadata.update(arrival_clock_contract=clock_contract, physical_gpu_uuids=uuids)
        rank_uuids = [None]*rt.world_size
        device_uuid = canonical_gpu_uuid(rt.torch.cuda.get_device_properties(rt.device).uuid)
        rt.torch.distributed.all_gather_object(rank_uuids, device_uuid, group=rt.ep_group.cpu_group)
        if sorted(rank_uuids) != uuids:
            raise ValueError('Actual CUDA rank devices differ from the required physical pair')
        rt.metadata['rank_gpu_uuids'] = rank_uuids
        trajectories = {}
        with kernel_override(None):
            for state_index, group in enumerate(groups(plan)):
                if plan['phase'] == 'final' and str(plan['calibration'][group['block_id']]['allocation_id']) != str(rt.metadata['allocation_id']):
                    raise ValueError('Calibration and final comparisons must share one allocation')
                preparation_start=time.monotonic_ns()
                cell = prepare(rt, weights, base, SimpleNamespace(banks_dir=args.banks_dir, output=args.output), group, plan)
                rt.sync()
                cell['preparation_ns']=rt.gather_scalars(time.monotonic_ns()-preparation_start)
                for block, calendar, repetition, order in cases(plan, group, state_index):
                    for arm in order:
                        row = trial(rt, weights, base, plan, cell, group, arm, block, calendar, repetition, order)
                        if rt.rank == 0:
                            key = source_key(group)
                            if key in trajectories and trajectories[key] != row['decision_trajectory_sha256']:
                                raise AssertionError('Fast path/calendar changed the native planning trajectory')
                            trajectories[key] = row['decision_trajectory_sha256']
                        write_row(rt, args.output, row)
                del cell
                gc.collect()
                rt.torch.cuda.empty_cache()
    finally:
        rt.close()


if __name__ == '__main__':
    main()

"""Frozen cyclic fixed-map pilot, including paired native and mapper controls."""
from __future__ import annotations
import argparse
import gc
import hashlib
import itertools
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np

from followup.payback import _materialize
from followup.router_tapes import load_router_bank
from optimization.common import agree_config, make_layer, array_hash, canonical_hash, manifest_hash
from optimization.kernel_config import kernel_override, resolve_kernel_config
from optimization.tuning import input_indices, _validate_bank
from resident_dispatch.assignments import generate_assignment, validate_assignment
from resident_dispatch.common import load_manifest, runtime_metadata, references_for, numerical_check
from skewbench.native import Runtime, Weights
from skewbench.runner import finalize_decision_log, snapshot_weight_hash

ARMS = ('native', 'native_aa', 'pre_native', 'one_copy_home', 'token_balanced')


def pilot_groups(manifest):
    m = manifest['matrix']
    if tuple(m['arms']) != ARMS or m['repetitions'] != 2:
        raise ValueError('Pilot requires the five declared arms and two paired repetitions')
    cells = list(itertools.product(m['block_ids'], m['domains'], m['batch_tokens']))
    random.Random(manifest['execution']['order_seed']).shuffle(cells)
    for index, (block, domain, batch) in enumerate(cells):
        order = list(ARMS[index % len(ARMS):] + ARMS[:index % len(ARMS)])
        for rep in range(2):
            yield dict(block_id=block, domain=domain, regime=f'domain{domain}',
                       batch_tokens=batch, repetition=rep), order if rep == 0 else list(reversed(order))


def _save_array_artifact(rt, path, **arrays):
    if rt.rank == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as f:
            np.savez_compressed(f, **arrays)
        identity = {'path':str(path), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    else: identity = None
    value = [identity]
    rt.torch.distributed.broadcast_object_list(value, src=0, group=rt.ep_group.cpu_group)
    return value[0]


def prepare_cell(rt, weights, manifest, base, output, group):
    entry = next(x for x in manifest['banks'] if x['block_id'] == group['block_id'])
    bank = load_router_bank(base / entry['capture_path'])
    _validate_bank(bank, entry, manifest, weights)
    indices, seed = input_indices(bank.domain_ids, group['block_id'], group['domain'], group['batch_tokens'])
    tensors = _materialize(rt, SimpleNamespace(bank=bank, indices=indices))
    layer = make_layer(rt, weights, manifest, 'SKIP')
    layer.assert_maps_and_weights()
    mapping = layer.map_snapshot()
    config = resolve_kernel_config(layer, group['batch_tokens'])
    if config != manifest['default_configs'][str(group['batch_tokens'])]:
        raise AssertionError('Native DEFAULT kernel changed')
    agree_config(rt, config)
    weight_hash, rank_hashes = snapshot_weight_hash(rt, (layer.w13, layer.w2))
    refs = references_for(rt, layer, tensors, (16,47))
    logical = np.asarray(bank.expert_ids)[indices]
    assignment_cpu, assignment_gpu, metrics, construction = {}, {}, {}, {}
    for method in ('native', 'one_copy_home', 'token_balanced'):
        start = time.perf_counter()
        selected = [generate_assignment(x, mapping, method=method, block_size_m=config['BLOCK_SIZE_M'])
                    for x in logical]
        assignment_cpu[method] = np.stack([x.physical_ids for x in selected])
        metrics[method] = [x.diagnostics for x in selected]
        for x, actual in zip(logical, assignment_cpu[method]):
            validate_assignment(x, actual, mapping, block_size_m=config['BLOCK_SIZE_M'])
        assignment_gpu[method] = rt.torch.from_numpy(assignment_cpu[method].copy()).to(rt.device, rt.torch.int32)
        construction[method] = time.perf_counter() - start
    # Source-derived CPU native assignment must match the executed pinned kernel.
    ls = layer.layer_state
    buffer = rt.torch.zeros_like(ls.expert_load_view)
    enabled = rt.torch.ones((), device=rt.device, dtype=rt.torch.bool)
    real = rt.torch.tensor(group['batch_tokens'], device=rt.device, dtype=rt.torch.int32)
    from resident_dispatch.gpu import PrecomputedMapper
    mapper_checks = 0
    for method in ('native','one_copy_home','token_balanced'):
        adapter = PrecomputedMapper(rt.torch, assignment_gpu[method])
        for raw in range(48):
            buffer.zero_()
            adapter.raw_index = raw
            chosen = (layer.mapping_kernel if method == 'native' else adapter)(
                tensors[1][raw], buffer, ls.logical_to_physical_map,
                ls.logical_replica_count, enabled, real)
            actual = chosen.cpu().numpy()
            if not np.array_equal(actual, assignment_cpu[method][raw]):
                raise AssertionError('GPU dispatch differs from frozen offline assignment')
            count = np.bincount(actual.ravel(), minlength=160)
            if not np.array_equal(buffer.cpu().numpy(), count):
                raise AssertionError('GPU physical recording differs from exact assignment')
            mapper_checks += 1
    # Both replicated-input ranks must agree on the global physical route tensor.
    agreement = {k:array_hash(v) for k,v in assignment_cpu.items()}
    agree_config(rt, agreement)
    name = f"{group['block_id']}-d{group['domain']}-b{group['batch_tokens']}"
    artifact = _save_array_artifact(rt, output.parent/'inputs'/f'{name}.npz',
        indices=indices, logical_ids=logical, physical_to_logical_map=mapping,
        **{f'{k}_physical_ids':v for k,v in assignment_cpu.items()},
        metadata=np.asarray(json.dumps({'capture_sha256':bank.sha256, 'seed':seed,
            'source_prompt_ids':bank.metadata['source_prompt_ids'], 'metrics':metrics,
            'offline_construction_seconds_excluded':construction}, sort_keys=True)))
    layer.close(); del layer
    return dict(tensors=tensors, references=refs, logical=logical, mapping=mapping,
        assignments=assignment_gpu, metrics=metrics, artifact=artifact,
        capture_sha256=bank.sha256, source_prompt_ids=bank.metadata['source_prompt_ids'],
        source_weights_sha256=weight_hash, source_weight_rank_sha256=rank_hashes,
        resolved_kernel_config=config, assignment_sha256=agreement,
        gpu_mapping_and_record_checks=mapper_checks)


def run_arm(rt, weights, manifest, cell, group, arm, order):
    from resident_dispatch.gpu import PrecomputedMapper
    torch = rt.torch
    layer = make_layer(rt, weights, manifest, 'SKIP')
    layer.assert_maps_and_weights()
    method = 'native' if arm in ('native','native_aa','pre_native') else arm
    adapter = None
    if arm not in ('native','native_aa'):
        adapter = PrecomputedMapper(torch, cell['assignments'][method])
        layer.mapping_kernel = adapter
    def select(raw):
        if adapter is not None: adapter.raw_index = raw
    tensors = cell['tensors']
    preflight = numerical_check(rt, layer, tensors, cell['references'], list(range(48)), select)
    setup_start = time.perf_counter()
    steps = manifest['pilot']['measured_steps']
    if steps != 256: raise ValueError('Frozen pilot requires32 inputs repeated eight times')
    events = [(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(steps)]
    # Event objects initialize their CUDA handles only at first record. Keep all
    # instrumentation setup outside the steady-state primary timer, for every arm.
    for start_event, end_event in events:
        start_event.record()
        end_event.record()
    rt.sync()
    layer.warming_up = True
    for raw in range(16):
        select(raw)
        layer.forward(*(t[raw] for t in tensors), observe=True)
        layer.step(raw)
    layer.warming_up = False
    # Exercise the entire unchanged SKIP path, including the CPU planner and
    # synchronous no-op rearrangement. This is common setup, not timed evidence.
    if layer.state.expert_rearrangement_step != 0:
        raise AssertionError('Sixteen conditioning inputs must finish the native initial period')
    for conditioning_step in range(64):
        raw = 16 + conditioning_step % 32
        select(raw)
        layer.forward(*(t[raw] for t in tensors), observe=True)
        layer.step(conditioning_step)
    layer.close()
    if len(layer.decisions) != 1 or layer.decisions[0]['allowed']:
        raise AssertionError('Full-path conditioning did not execute exactly one SKIP planner call')
    layer.assert_maps_and_weights()
    if (not np.array_equal(layer.map_snapshot(), cell['mapping']) or
            layer.counter.sent_bytes or layer.counter.sent_messages):
        raise AssertionError('Conditioning changed the fixed resident state')
    layer.state.expert_rearrangement_step = 0
    layer.state.expert_load_window_step = 0
    layer.native_valid_steps = 0
    layer.ms.expert_load_window.zero_()
    layer.ms.expert_load_pass.zero_()
    layer.observation_steps = 0
    layer.logical_ring.zero_()
    layer.logical_ring_timestamps.fill(-1)
    layer.histogram_capture_count = layer.previous_window_timestamp_checks = 0
    layer.state._update_layer_should_record(log_stats=False)
    layer.planner_seconds = layer.controller_seconds = 0.
    layer.decisions.clear()
    layer.last_accepted_step = None
    layer.end_of_input = False
    layer.terminal_proposals_suppressed = 0
    layer.counter.sent_bytes = layer.counter.sent_messages = 0
    gc.collect()  # GC remains enabled; any collection during measurement is charged.
    rt.sync()
    setup_seconds = max(rt.gather_scalars(time.perf_counter() - setup_start))
    rt.sync(); rt.barrier(); rt.sync()
    timestamp = time.time()
    started = time.perf_counter()
    for step in range(steps):
        raw = 16 + step % 32
        select(raw)
        events[step][0].record()
        layer.forward(*(t[raw] for t in tensors), observe=True)
        layer.end_of_input = step == steps-1
        layer.step(step)
        events[step][1].record()
    layer.close()
    elapsed = time.perf_counter() - started
    rank_elapsed = rt.gather_scalars(elapsed)
    rank_steps = [None]*rt.world_size
    torch.distributed.all_gather_object(rank_steps,
        [a.elapsed_time(b)/1000 for a,b in events], group=rt.ep_group.cpu_group)
    log = finalize_decision_log(rt, layer)
    if [d['step'] for d in log] != [63,127,191]:
        raise AssertionError('Unexpected observation/decision cadence')
    expected = np.bincount(cell['logical'][32:48].ravel(), minlength=128) * rt.world_size
    for d in log:
        if not np.array_equal(d['current_counts'], expected) or d['allowed']:
            raise AssertionError('Reconstructed logical demand or SKIP decision differs')
    terminal = layer.ms.expert_load_window[:,0,:].cpu().numpy()
    expected_physical = np.stack([cell['metrics'][method][i]['physical_counts'] for i in range(32,48)])
    for d in log:
        if not np.array_equal(d['physical_slot_assignments_window'], expected_physical):
            raise AssertionError('Native decision window differs from exact per-step physical dispatch')
    if not np.array_equal(terminal, expected_physical):
        raise AssertionError('Terminal native observation ring differs from physical dispatch')
    layer.assert_maps_and_weights()
    final = layer.map_snapshot()
    if not np.array_equal(final, cell['mapping']) or layer.counter.sent_bytes or layer.counter.sent_messages:
        raise AssertionError('Fixed placement moved weights or mappings')
    if layer.terminal_proposals_suppressed != 1:
        raise AssertionError('Terminal native candidate must be suppressed exactly once')
    post = numerical_check(rt, layer, tensors, cell['references'], [16,47], select)
    logical_trajectory = [{k:d[k] for k in ('step','current_counts','current_map','proposed_map','committed_map')}
                          for d in log]
    row = {**rt.metadata, **group, 'resident_dispatch_schema_version':1,
        'config_hash':manifest_hash(manifest),'phase':'pilot','split':'pilot',
        'arm':arm,'policy':'SKIP','status':'complete','evidence_eligible':True,
        'measurement_kind':'cyclic_fixed_placement_dispatch_headroom',
        'elapsed_seconds':max(rank_elapsed),'per_rank_elapsed_seconds':rank_elapsed,
        'step_seconds_by_rank':rank_steps,'steps_completed':steps,
        'measurement_setup_revision':2,'timing_events_initialized_before_timer':True,
        'full_policy_conditioning_steps':64,'full_policy_conditioning_calls':1,
        'conditioning_raw_indices':[16 + i % 32 for i in range(64)],
        'instrumentation_and_conditioning_seconds_excluded':setup_seconds,
        'gc_enabled_during_measurement':gc.isenabled(),
        'measurement_start_unix_seconds':timestamp,'arm_order':order,'arm_position':order.index(arm),
        'timing_scope':manifest['execution']['timing_scope'],
        'assignment_selection_cost_included':arm in ('native','native_aa'),
        'assignment_method':method,'assignment_artifact':cell['artifact'],
        'assignment_sha256':cell['assignment_sha256'][method],
        'assignment_metrics':cell['metrics'][method][16:],
        'capture_sha256':cell['capture_sha256'],'source_prompt_ids':cell['source_prompt_ids'],
        'logical_routing_sha256':array_hash(cell['logical']),
        'source_map':cell['mapping'].tolist(),'source_map_sha256':array_hash(cell['mapping']),
        'final_map':final.tolist(),'source_weights_sha256':cell['source_weights_sha256'],
        'source_weight_rank_sha256':cell['source_weight_rank_sha256'],
        'resolved_kernel_config':cell['resolved_kernel_config'],
        'decision_log':log,'logical_trajectory_sha256':canonical_hash(logical_trajectory),
        'decision_trajectory_sha256':canonical_hash(logical_trajectory),
        'terminal_physical_window':terminal.tolist(),'physical_window_checks':4,
        'gpu_mapping_and_record_checks':cell['gpu_mapping_and_record_checks'],
        'terminal_proposals_suppressed':1,'migration_bytes':0,'migration_sent_messages':0,
        'source_restore_verified':True,'expected_arm_weights_verified':True,
        'commit_completed':True,'logical_routing_preserved':True,'rank_assignment_agreement':True,
        'preflight_correctness':preflight, **post}
    row['record_id'] = canonical_hash({k:row[k] for k in
        ('config_hash','block_id','regime','batch_tokens','repetition','arm')})
    del layer,events
    gc.collect(); rt.sync(); rt.barrier()
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--banks-dir',type=Path,required=True)
    p.add_argument('--weights-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    manifest = load_manifest(args.manifest)
    if manifest['phase'] != 'pilot': raise ValueError('This runner is pilot-only')
    if (manifest['trace']['window_steps'] != 16 or
            manifest['policy']['rebalance_interval'] != 64 or
            manifest['pilot'] != dict(unique_conditioning_inputs=16,
                unique_measured_inputs=32, cycles=8, measured_steps=256,
                offline_selection_excluded=True, measurement_setup_revision=2,
                full_policy_conditioning_steps=64, maximum_calibration_revisions=1)):
        raise ValueError('Pilot geometry differs from its frozen execution contract')
    if args.output.exists(): raise FileExistsError('No output overwrite or partial-group splice')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    rt = Runtime()
    try:
        weights = Weights(args.weights_dir)
        runtime_metadata(rt, weights, manifest)
        previous = None
        trajectories = {}
        with kernel_override(None):
            for group,order in pilot_groups(manifest):
                key = (group['block_id'],group['domain'],group['batch_tokens'])
                if key != previous:
                    if previous is not None:
                        del cell
                        gc.collect(); rt.torch.cuda.empty_cache()
                    cell = prepare_cell(rt,weights,manifest,args.banks_dir,args.output,group)
                    previous = key
                for arm in order:
                    if rt.rank == 0:
                        print(json.dumps({'event':'run_start',**group,'arm':arm}),flush=True)
                    row = run_arm(rt,weights,manifest,cell,group,arm,order)
                    if key in trajectories and trajectories[key] != row['logical_trajectory_sha256']:
                        raise AssertionError('Dispatch changed native logical-demand planning trajectory')
                    trajectories[key] = row['logical_trajectory_sha256']
                    if rt.rank == 0:
                        with args.output.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                        print(json.dumps({'event':'run_complete','arm':arm,
                            'elapsed_seconds':row['elapsed_seconds'],'record_id':row['record_id']}),flush=True)
                    rt.barrier()
    finally: rt.close()


if __name__ == '__main__':main()

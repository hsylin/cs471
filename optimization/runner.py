"""Complete-policy replay with only the native GEMM tile configuration changed."""
from __future__ import annotations
import argparse
import gc
import itertools
import json
from pathlib import Path
import random
import time

import numpy as np

from followup.payback import _materialize
from followup.router_tapes import load_router_tape
from optimization.common import (agree_config, array_hash, build_references, canonical_hash,
    check_references, check_tape, load_manifest, make_layer, manifest_hash, runtime_metadata)
from optimization.kernel_config import (config_fingerprint, kernel_override,
                                         resolve_kernel_config)
from skewbench.native import Runtime, Weights
from skewbench.runner import finalize_decision_log, snapshot_weight_hash, warm_native_transport


def groups(manifest):
    matrix = manifest['matrix']
    arms = list(itertools.product(matrix['kernel_variants'], matrix['policies']))
    if len(arms) != 4:
        raise ValueError('Require the frozen two-kernel/two-policy design')
    keys = list(itertools.product(matrix['block_ids'], matrix['regimes'],
                                  matrix['batch_tokens'], range(matrix['repetitions'])))
    cycle = [arms[i:] + arms[:i] for i in range(4)]
    cycle += [list(reversed(order)) for order in cycle]
    orders = [cycle[i % len(cycle)] for i in range(len(keys))]
    random.Random(manifest['execution']['order_seed']).shuffle(orders)
    return [(dict(zip(('block_id','regime','batch_tokens','repetition'),key)),order)
            for key,order in zip(keys,orders)]


def trajectory_hash(log):
    fields = ('step','allowed','reason','current_counts','current_map','proposed_map',
              'committed_map','transfer_sent_bytes','physical_slot_assignments_window')
    return canonical_hash([{k:d.get(k) for k in fields} for d in log])


def validate_observations(log, tape, warmup, window, world_size):
    logical = tape.expert_ids
    for decision in log:
        stop = warmup + decision['step'] + 1
        expected = np.bincount(logical[stop-window:stop].ravel(), minlength=128) * world_size
        if not np.array_equal(expected, decision['current_counts']):
            raise AssertionError('Native observed demand differs from the exact replay')
    return len(log)


def run_arm(rt, weights, manifest, tape, tensors, references, source_hash,
            group, variant, policy, order):
    torch = rt.torch
    config_id = 'DEFAULT' if variant == 'DEFAULT' else manifest['selected_configs'][str(group['batch_tokens'])]['candidate_id']
    config = None if variant == 'DEFAULT' else manifest['selected_configs'][str(group['batch_tokens'])]['config']
    warmup = manifest['trace']['warmup_steps']
    steps = manifest['trace']['measured_steps']
    hidden, ids, combine = tensors
    construction_started = time.perf_counter()
    layer = make_layer(rt, weights, manifest, policy)
    initial_map = layer.map_snapshot().tolist()
    layer.assert_maps_and_weights()
    with kernel_override(config):
        resolved = resolve_kernel_config(layer, group['batch_tokens'])
        agree_config(rt, resolved)
        # Compilation and numerical preflight happen outside the policy timer.
        preflight = check_references(rt, layer, tensors, references)
        preflight_seconds = max(rt.gather_scalars(time.perf_counter()-construction_started))
        warm_started = time.perf_counter()
        layer.warming_up = True
        for raw in range(warmup):
            layer.forward(hidden[raw], ids[raw], combine[raw], observe=True)
            layer.step(raw)
        layer.warming_up = False
        layer.state.expert_rearrangement_step = 0
        layer.state._update_layer_should_record(log_stats=False)
        layer.planner_seconds = layer.controller_seconds = 0.
        layer.decisions.clear()
        layer.last_accepted_step = None
        layer.counter.sent_bytes = layer.counter.sent_messages = 0
        rt.sync()
        warm_seconds = max(rt.gather_scalars(time.perf_counter()-warm_started))
        events = [(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(steps)]
        torch.cuda.reset_peak_memory_stats(rt.device)
        rt.barrier(); rt.sync()
        start = time.perf_counter()
        for step in range(steps):
            raw = warmup + step
            events[step][0].record()
            layer.forward(hidden[raw], ids[raw], combine[raw], observe=True)
            layer.end_of_input = step == steps-1
            layer.step(step)
            events[step][1].record()
        layer.close()
        elapsed = time.perf_counter()-start
        rank_elapsed = rt.gather_scalars(elapsed)
        rank_steps = [None]*rt.world_size
        torch.distributed.all_gather_object(rank_steps,
            [a.elapsed_time(b)/1000 for a,b in events], group=rt.ep_group.cpu_group)
        payload = int(sum(rt.gather_scalars(layer.counter.sent_bytes)))
        messages = int(sum(rt.gather_scalars(layer.counter.sent_messages)))
        peak = max(rt.gather_scalars(torch.cuda.max_memory_allocated(rt.device)))
        log = finalize_decision_log(rt, layer)
        observation_checks = validate_observations(log,tape,warmup,manifest['trace']['window_steps'],rt.world_size)
        if [d['step'] for d in log] != list(range(63,steps-1,64)):
            raise AssertionError('Unexpected native policy decision schedule')
        layer.assert_maps_and_weights()
        correctness = check_references(rt, layer, tensors, references)
        final_map = layer.map_snapshot().tolist()
        if policy == 'SKIP' and (payload or final_map != initial_map):
            raise AssertionError('Fixed placement unexpectedly changed')
        row = {**rt.metadata, **group, 'optimization_schema_version':1,
            'config_hash':manifest_hash(manifest), 'phase':manifest['phase'], 'split':manifest['split'],
            'measurement_kind':'gpu_optimization','status':'complete','evidence_eligible':True,
            'kernel_variant':variant,'policy':policy,'kernel_config_id':config_id,
            'kernel_override_config':config,'kernel_config_sha256':config_fingerprint(config),
            'resolved_kernel_config':resolved,'kernel_config_rank_agreement':True,
            'elapsed_seconds':max(rank_elapsed),
            'per_rank_elapsed_seconds':rank_elapsed,'step_seconds_by_rank':rank_steps,
            'steps_completed':steps,'observation_warmup_steps':warmup,'raw_trace_steps':len(tape.indices),
            'warmup_seconds_excluded':warm_seconds,'source_prompt_ids':tape.metadata['source_prompt_ids'],
            'construction_compile_preflight_seconds_excluded':preflight_seconds,
            'capture_sha256':tape.bank.sha256,'replay_sha256':tape.sha256,
            'source_map':initial_map,'source_map_sha256':array_hash(initial_map),
            'source_weights_sha256':source_hash,'final_map':final_map,
            'final_map_sha256':array_hash(final_map),'decision_log':log,
            'decision_trajectory_sha256':trajectory_hash(log),
            'migration_bytes':payload,'migration_sent_messages':messages,
            'accepted':sum(d['allowed'] for d in log),
            'planner_seconds':max(rt.gather_scalars(layer.planner_seconds)),
            'controller_seconds':max(rt.gather_scalars(layer.controller_seconds)),
            'peak_gpu_memory_bytes':int(peak),'replica_budget_actual':layer.replicas,
            'observation_conservation_checks':observation_checks,
            'terminal_proposals_suppressed':layer.terminal_proposals_suppressed,
            'arm_order':order,'source_restore_verified':True,'expected_arm_weights_verified':True,
            'commit_completed':True,'logical_routing_preserved':True,
            'timing_scope':manifest['execution']['timing_scope'],
            'phase_timer_caveat':'CPU subregions overlap GPU work; not an additive decomposition',
            'preflight_correctness':preflight, **correctness}
        row['record_id'] = canonical_hash({k:row[k] for k in
            ('config_hash','block_id','regime','batch_tokens','repetition','kernel_variant','policy')})
    del layer, events
    gc.collect(); rt.sync(); rt.barrier()
    return row


def run_policy(rt, weights, manifest, args):
    warm_manifest = {'policy':manifest['policy'], 'trace':{'window_steps':manifest['trace']['window_steps']}}
    warm_native_transport(rt,weights,warm_manifest)
    inventory = {(t['block_id'],t['regime'],t['batch_tokens']):t for t in manifest['tapes']}
    state_key = None
    hashes = {}
    tensors = references = tape = None
    for group,order in groups(manifest):
        key = (group['block_id'],group['regime'],group['batch_tokens'])
        if key != state_key:
            if tensors is not None:
                del tensors,references,tape
                gc.collect(); rt.torch.cuda.empty_cache()
            entry = inventory[key]
            tape = load_router_tape(Path(args.banks_dir)/entry['replay_path'],
                                    Path(args.banks_dir)/entry['capture_path'])
            check_tape(manifest,entry,tape)
            materialization_start = time.perf_counter()
            tensors = _materialize(rt,tape)
            materialization_seconds = max(rt.gather_scalars(time.perf_counter()-materialization_start))
            layer = make_layer(rt,weights,manifest,'SKIP')
            layer.assert_maps_and_weights()
            source_hash,_ = snapshot_weight_hash(rt,[layer.w13,layer.w2])
            references = build_references(rt,layer,tensors,[manifest['trace']['warmup_steps'],len(tape.indices)-1])
            layer.close(); del layer
            state_key = key
        for variant,policy in order:
            if rt.rank == 0:
                print(json.dumps({'event':'run_start',**group,'kernel_variant':variant,'policy':policy}),flush=True)
            row = run_arm(rt,weights,manifest,tape,tensors,references,source_hash,group,variant,policy,order)
            row['state_input_materialization_seconds_excluded'] = materialization_seconds
            trajectory_key = (*key,policy)
            identity = (row['decision_trajectory_sha256'],row['final_map_sha256'],row['migration_bytes'])
            if trajectory_key in hashes and hashes[trajectory_key] != identity:
                raise AssertionError('Kernel configuration or repetition changed the policy trajectory')
            hashes[trajectory_key] = identity
            if rt.rank == 0:
                with Path(args.output).open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                print(json.dumps({'event':'run_complete','record_id':row['record_id'],'elapsed_seconds':row['elapsed_seconds']}),flush=True)
            rt.barrier()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--banks-dir',type=Path,required=True)
    p.add_argument('--weights-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile-dir',type=Path)
    args = p.parse_args()
    manifest = load_manifest(args.manifest)
    if args.output.exists():raise FileExistsError('Use a fresh output; never splice partial paired groups')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    rt = Runtime()
    try:
        weights = Weights(args.weights_dir)
        runtime_metadata(rt,weights,manifest)
        if args.profile_dir:
            if manifest['phase'] != 'validation' or manifest['split'] != 'pilot':
                raise ValueError('Profiling is a separate pilot-only measurement')
            from optimization.tuning import profile_selected
            profile_selected(rt,weights,manifest,args)
        elif manifest['phase']=='tuning':
            from optimization.tuning import run_tuning
            run_tuning(rt,weights,manifest,args)
        else:run_policy(rt,weights,manifest,args)
    finally:rt.close()


if __name__ == '__main__':main()

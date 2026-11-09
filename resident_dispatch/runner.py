"""One fully charged online resident dispatcher against unchanged native SKIP."""
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
from followup.router_tapes import load_router_bank, load_router_tape
from optimization.common import (agree_config, array_hash, canonical_hash, check_tape,
                                 make_layer, manifest_hash)
from optimization.kernel_config import kernel_override, resolve_kernel_config
from optimization.tuning import input_indices, _validate_bank
from resident_dispatch.assignments import generate_assignment, validate_assignment
from resident_dispatch.common import load_manifest, runtime_metadata, references_for, numerical_check
from resident_dispatch.pilot import _save_array_artifact
from skewbench.native import Runtime, Weights
from skewbench.runner import finalize_decision_log, snapshot_weight_hash


def groups(manifest, repetition=None):
    matrix = manifest['matrix']
    reps = range(matrix['repetitions']) if repetition is None else [repetition]
    if matrix['repetitions'] != 2 or list(matrix['arms']) != ['native','online'] or any(r not in (0,1) for r in reps):
        raise ValueError('Require one frozen online policy, native baseline and two repetitions')
    cells = list(itertools.product(matrix['block_ids'], matrix['regimes'], matrix['batch_tokens']))
    random.Random(manifest['execution']['order_seed']).shuffle(cells)
    for index,(block,regime,batch) in enumerate(cells):
        first = ['native','online'] if index % 2 == 0 else ['online','native']
        for rep in reps:
            yield dict(block_id=block,regime=regime,batch_tokens=batch,repetition=rep), first if rep==0 else list(reversed(first))


def make_online(rt, layer, batch):
    from resident_dispatch.online import OnlineMapper
    mapper = OnlineMapper(rt.torch, layer.map_snapshot(), batch_tokens=batch, top_k=8)
    mapper.verify_metadata(layer.layer_state.logical_to_physical_map,
                           layer.layer_state.logical_replica_count,
                           layer.ms.physical_to_logical_map[0])
    return mapper


def _oracle(ids, mapping, block_m, arm):
    if arm == 'native':
        return generate_assignment(ids,mapping,method='native',block_size_m=block_m).physical_ids
    from resident_dispatch.online import cpu_greedy_assignment
    result = cpu_greedy_assignment(ids,mapping)
    return result.physical_ids if hasattr(result,'physical_ids') else np.asarray(result)


def selector_diagnostic(rt, layer, tensors, mapper):
    """Separate diagnostic, never used as a selector or added to primary time."""
    torch=rt.torch;ls=layer.layer_state
    buf=torch.zeros_like(ls.expert_load_view)
    real=torch.tensor(tensors[1].shape[1],device=rt.device,dtype=torch.int32)
    rows=[]
    for enabled in (False,True):
        record=torch.tensor(enabled,device=rt.device,dtype=torch.bool)
        for name,fn in [('native',layer.mapping_kernel),('online',mapper)]:
            for i in range(16):fn(tensors[1][i],buf,ls.logical_to_physical_map,ls.logical_replica_count,record,real)
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();b.record();rt.sync();rt.barrier();rt.sync()
            a.record()
            for i in range(64):fn(tensors[1][i%len(tensors[1])],buf,ls.logical_to_physical_map,ls.logical_replica_count,record,real)
            b.record();rt.sync()
            rows.append(dict(arm=name,record=enabled,calls=64,
                per_rank_stream_seconds_per_call=rt.gather_scalars(a.elapsed_time(b)/64000),
                scope='Separate mapper-only CUDA-stream interval including launch gaps; excludes expert kernels/collectives; descriptive only'))
    return rows


def prepare(rt, weights, manifest, args, group):
    base = args.banks_dir
    if manifest['phase'] == 'validation':
        domain = int(group['regime'].removeprefix('domain'))
        entry = next(x for x in manifest['banks'] if x['block_id']==group['block_id'])
        bank = load_router_bank(base/entry['capture_path'])
        _validate_bank(bank,entry,manifest,weights)
        indices,_ = input_indices(bank.domain_ids,group['block_id'],domain,group['batch_tokens'])
        tape = SimpleNamespace(bank=bank,indices=indices)
        sequence = [16+i%32 for i in range(256)]
        warmup = list(range(16))
        conditioning = [16+i%32 for i in range(64)]
        reference_steps = list(range(48))
        fp32_steps = [16,47]
        replay_hash = array_hash(indices)
    else:
        entry = next(x for x in manifest['tapes'] if all(x[k]==group[k] for k in ('block_id','regime','batch_tokens')))
        tape = load_router_tape(base/entry['replay_path'],base/entry['capture_path'])
        check_tape(manifest,entry,tape)
        indices,bank = tape.indices,tape.bank
        sequence = list(range(32,544))
        warmup = list(range(32))
        conditioning = [i%32 for i in range(64)]
        reference_steps = fp32_steps = [32,160,288,543]
        replay_hash = tape.sha256
    tensors = _materialize(rt,tape)
    logical = np.asarray(bank.expert_ids)[indices]
    layer = make_layer(rt,weights,manifest,'SKIP')
    layer.assert_maps_and_weights()
    mapping = layer.map_snapshot()
    config = resolve_kernel_config(layer,group['batch_tokens'])
    if config != manifest['default_configs'][str(group['batch_tokens'])]:
        raise AssertionError('Native DEFAULT kernel changed')
    agree_config(rt,config)
    weight_hash,rank_hashes = snapshot_weight_hash(rt,(layer.w13,layer.w2))
    refs = references_for(rt,layer,tensors,fp32_steps,raw_steps=reference_steps)
    mapper = make_online(rt,layer,group['batch_tokens'])
    from resident_dispatch.online import verify_adversarial
    adversarial = verify_adversarial(rt,layer,mapper,group['batch_tokens'])
    native_map = layer.mapping_kernel
    ls = layer.layer_state
    buf = rt.torch.zeros_like(ls.expert_load_view)
    record = rt.torch.ones((),device=rt.device,dtype=rt.torch.bool)
    real = rt.torch.tensor(group['batch_tokens'],device=rt.device,dtype=rt.torch.int32)
    counts, route_hashes, diagnostics, snapshots = {}, {}, {}, {}
    for arm, fn in [('native',native_map),('online',mapper)]:
        raw_counts, raw_diagnostics, digester, sample = [], [], hashlib.sha256(), []
        for raw,ids_np in enumerate(logical):
            expected = _oracle(ids_np,mapping,config['BLOCK_SIZE_M'],arm)
            buf.zero_()
            actual = fn(tensors[1][raw],buf,ls.logical_to_physical_map,
                        ls.logical_replica_count,record,real).cpu().numpy()
            if not np.array_equal(actual,expected):
                raise AssertionError(f'{arm} GPU assignment differs from independent CPU oracle at input{raw}')
            diag = validate_assignment(ids_np,actual,mapping,block_size_m=config['BLOCK_SIZE_M'])
            if not np.array_equal(buf.cpu().numpy(),diag['physical_counts']):
                raise AssertionError('Actual physical counter differs from physical assignment')
            digester.update(np.asarray(actual,dtype=np.int64).tobytes())
            raw_counts.append(diag['physical_counts'])
            raw_diagnostics.append({k:diag[k] for k in ('per_rank_tokens','per_rank_active_copies','per_rank_m_tiles','per_rank_padded_rows','minimum_possible_max_rank_active_copies')})
            if raw in fp32_steps:sample.append(np.asarray(actual,dtype=np.uint8))
        counts[arm] = np.asarray(raw_counts,dtype=np.int32)
        diagnostics[arm] = raw_diagnostics
        snapshots[arm] = np.stack(sample)
        route_hashes[arm] = digester.hexdigest()
    agree_config(rt,route_hashes)
    name = f"{group['block_id']}-{group['regime']}-b{group['batch_tokens']}"
    artifact = _save_array_artifact(rt,args.output.parent/'inputs'/f'{name}.npz',
        indices=indices,physical_to_logical_map=mapping,
        native_physical_counts=counts['native'], online_physical_counts=counts['online'],
        native_sample_physical_ids=snapshots['native'],online_sample_physical_ids=snapshots['online'],
        metadata=np.asarray(json.dumps(dict(capture_sha256=bank.sha256,replay_sha256=replay_hash,
            source_prompt_ids=bank.metadata['source_prompt_ids'],logical_routing_sha256=array_hash(logical),
            full_physical_route_sha256=route_hashes,sample_raw_indices=fp32_steps,
            route_scope='All unique GPU assignments checked and hashed; exact per-step physical counts and selected full assignments saved; all assignments reconstructible from pinned current-batch CPU oracle'),sort_keys=True)))
    selector_cost=selector_diagnostic(rt,layer,tensors,mapper) if manifest['phase']=='validation' else None
    layer.close(); del layer,mapper
    return dict(tensors=tensors,logical=logical,references=refs,reference_steps=reference_steps,
        fp32_steps=fp32_steps,sequence=sequence,warmup=warmup,conditioning=conditioning,
        mapping=mapping,counts=counts,diagnostics=diagnostics,route_hashes=route_hashes,
        artifact=artifact,capture_sha256=bank.sha256,replay_sha256=replay_hash,
        source_prompt_ids=bank.metadata['source_prompt_ids'],source_weights_sha256=weight_hash,
        source_weight_rank_sha256=rank_hashes,resolved_kernel_config=config,
        adversarial_checks=adversarial,selector_diagnostic=selector_cost)


def reset_observation(layer):
    layer.state.expert_rearrangement_step = 0
    layer.state.expert_load_window_step = 0
    layer.native_valid_steps = layer.observation_steps = 0
    layer.ms.expert_load_window.zero_(); layer.ms.expert_load_pass.zero_()
    layer.logical_ring.zero_(); layer.logical_ring_timestamps.fill(-1)
    layer.histogram_capture_count = layer.previous_window_timestamp_checks = 0
    layer.state._update_layer_should_record(log_stats=False)
    layer.decisions.clear(); layer.last_accepted_step = None
    layer.planner_seconds = layer.controller_seconds = 0.
    layer.end_of_input = False; layer.terminal_proposals_suppressed = 0
    layer.counter.sent_bytes = layer.counter.sent_messages = 0
    layer.step_index = -1


def run_arm(rt,weights,manifest,cell,group,arm,order):
    torch = rt.torch
    layer = make_layer(rt,weights,manifest,'SKIP')
    layer.assert_maps_and_weights()
    mapper = make_online(rt,layer,group['batch_tokens']) if arm=='online' else None
    if mapper is not None:layer.mapping_kernel=mapper
    inputs=cell['tensors']; noop=lambda raw:None
    preflight=numerical_check(rt,layer,inputs,cell['references'],cell['reference_steps'],noop)
    setup_start=time.perf_counter()
    events=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in cell['sequence']]
    for a,b in events:a.record();b.record()
    rt.sync()
    layer.warming_up=True
    for step,raw in enumerate(cell['warmup']):
        layer.forward(*(t[raw] for t in inputs),observe=True);layer.step(step)
    layer.warming_up=False
    reset_observation(layer)
    for step,raw in enumerate(cell['conditioning']):
        layer.forward(*(t[raw] for t in inputs),observe=True);layer.step(step)
    layer.close();layer.assert_maps_and_weights()
    if len(layer.decisions)!=1 or layer.decisions[0]['allowed'] or layer.counter.sent_bytes:
        raise AssertionError('Full-path conditioning must make exactly one native SKIP decision')
    reset_observation(layer)
    gc.collect();rt.sync()
    setup=max(rt.gather_scalars(time.perf_counter()-setup_start))
    rt.barrier();rt.sync()
    timestamp=time.time();started=time.perf_counter()
    for step,raw in enumerate(cell['sequence']):
        events[step][0].record()
        layer.forward(*(t[raw] for t in inputs),observe=True)
        layer.end_of_input=step==len(cell['sequence'])-1
        layer.step(step)
        events[step][1].record()
    layer.close()
    elapsed=time.perf_counter()-started
    times=rt.gather_scalars(elapsed)
    rank_steps=[None]*rt.world_size
    torch.distributed.all_gather_object(rank_steps,[a.elapsed_time(b)/1000 for a,b in events],group=rt.ep_group.cpu_group)
    decisions=finalize_decision_log(rt,layer)
    if [d['step'] for d in decisions]!=list(range(63,len(cell['sequence'])-1,64)):
        raise AssertionError('Wrong native decision schedule')
    for d in decisions:
        raw=cell['sequence'][d['step']-15:d['step']+1]
        expected=cell['counts'][arm][raw]
        logical=np.bincount(cell['logical'][raw].ravel(),minlength=128)*rt.world_size
        if not np.array_equal(d['physical_slot_assignments_window'],expected) or not np.array_equal(d['current_counts'],logical) or d['allowed']:
            raise AssertionError('Observed native window/decision differs from exact raw dispatch')
    terminal=layer.ms.expert_load_window[:,0,:].cpu().numpy()
    if not np.array_equal(terminal,cell['counts'][arm][cell['sequence'][-16:]]):
        raise AssertionError('Terminal observation window differs')
    layer.assert_maps_and_weights()
    if mapper is not None:
        mapper.verify_metadata(layer.layer_state.logical_to_physical_map,layer.layer_state.logical_replica_count,layer.ms.physical_to_logical_map[0])
    final=layer.map_snapshot()
    if not np.array_equal(final,cell['mapping']) or layer.counter.sent_bytes or layer.counter.sent_messages or layer.terminal_proposals_suppressed!=1:
        raise AssertionError('Fixed resident-state or terminal-completion invariant failed')
    post=numerical_check(rt,layer,inputs,cell['references'],cell['fp32_steps'],noop)
    trajectory=canonical_hash([{k:d[k] for k in ('step','current_counts','current_map','proposed_map','committed_map')} for d in decisions])
    row={**rt.metadata,**group,'resident_dispatch_schema_version':1,'config_hash':manifest_hash(manifest),
        'phase':manifest['phase'],'split':manifest['split'],'arm':arm,'policy':'SKIP',
        'status':'complete','evidence_eligible':True,'measurement_kind':'fully_charged_resident_dispatch',
        'elapsed_seconds':max(times),'per_rank_elapsed_seconds':times,'step_seconds_by_rank':rank_steps,
        'steps_completed':len(cell['sequence']),'measurement_start_unix_seconds':timestamp,
        'arm_order':order,'arm_position':order.index(arm),'timing_scope':manifest['execution']['timing_scope'],
        'assignment_selection_cost_included':True,'online_algorithm':manifest['online_algorithm'] if arm=='online' else 'native_knuth_hash',
        'source_map':cell['mapping'].tolist(),'final_map':final.tolist(),'source_map_sha256':array_hash(cell['mapping']),
        **{k:cell[k] for k in ('capture_sha256','replay_sha256','source_prompt_ids','source_weights_sha256','source_weight_rank_sha256','resolved_kernel_config')},
        'logical_routing_sha256':array_hash(cell['logical']),'assignment_sha256':cell['route_hashes'][arm],
        'assignment_artifact':cell['artifact'],'assignment_metrics':[cell['diagnostics'][arm][i] for i in cell['sequence']],
        'decision_log':decisions,'decision_trajectory_sha256':trajectory,'logical_trajectory_sha256':trajectory,
        'terminal_physical_window':terminal.tolist(),'physical_window_checks':len(decisions)+1,
        'gpu_mapping_and_record_checks':len(cell['logical'])*2,'full_unique_gpu_routes_checked':len(cell['logical']),
        'adversarial_checks':cell['adversarial_checks'],'selector_diagnostic':cell['selector_diagnostic'],
        'terminal_proposals_suppressed':1,'migration_bytes':0,'migration_sent_messages':0,
        'source_restore_verified':True,'expected_arm_weights_verified':True,'commit_completed':True,
        'logical_routing_preserved':True,'rank_assignment_agreement':True,
        'measurement_setup_revision':2,'timing_events_initialized_before_timer':True,
        'full_policy_conditioning_steps':64,'full_policy_conditioning_calls':1,
        'conditioning_raw_indices':cell['conditioning'],'instrumentation_and_conditioning_seconds_excluded':setup,
        'gc_enabled_during_measurement':gc.isenabled(),'preflight_correctness':preflight,**post}
    row['record_id']=canonical_hash({k:row[k] for k in ('config_hash','block_id','regime','batch_tokens','repetition','arm')})
    del layer,mapper,events
    gc.collect();rt.sync();rt.barrier()
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','banks-dir','weights-dir','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--repetition',type=int,choices=(0,1))
    args=p.parse_args();manifest=load_manifest(args.manifest)
    if manifest['phase'] not in ('validation','final'):raise ValueError('Online runner requires a validation/final manifest')
    if manifest['phase']=='final' and args.repetition is None:raise ValueError('Run each final repetition in a separate allocation')
    if args.output.exists():raise FileExistsError('Use a new complete-matrix output')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    rt=Runtime()
    try:
        weights=Weights(args.weights_dir);runtime_metadata(rt,weights,manifest)
        previous=None;trajectories={}
        with kernel_override(None):
            for group,order in groups(manifest,args.repetition):
                key=tuple(group[k] for k in ('block_id','regime','batch_tokens'))
                if key!=previous:
                    if previous is not None:del cell;gc.collect();rt.torch.cuda.empty_cache()
                    cell=prepare(rt,weights,manifest,args,group);previous=key
                for arm in order:
                    if rt.rank==0:print(json.dumps({'event':'run_start',**group,'arm':arm}),flush=True)
                    row=run_arm(rt,weights,manifest,cell,group,arm,order)
                    if key in trajectories and trajectories[key]!=row['decision_trajectory_sha256']:
                        raise AssertionError('Dispatcher altered reconstructed logical-demand planning')
                    trajectories[key]=row['decision_trajectory_sha256']
                    if rt.rank==0:
                        with args.output.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                        print(json.dumps({'event':'run_complete','arm':arm,'elapsed_seconds':row['elapsed_seconds'],'record_id':row['record_id']}),flush=True)
                    rt.barrier()
    finally:rt.close()


if __name__=='__main__':main()

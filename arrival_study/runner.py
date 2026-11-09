"""External-clock FCFS layer jobs with explicit, charged rank coordination."""
from __future__ import annotations
import argparse
import gc
import itertools
import json
from pathlib import Path
import random
import socket
import time
from types import SimpleNamespace

import numpy as np

from arrival_study.common import load_plan, source_identity
from arrival_study.schedules import calibration_capacity, fcfs_metrics, validate_schedule
from optimization.common import canonical_hash, array_hash, make_layer, manifest_hash
from optimization.kernel_config import kernel_override
from resident_dispatch.common import runtime_metadata, numerical_check
from resident_dispatch.runner import prepare, make_online, reset_observation
from skewbench.native import Runtime, Weights
from skewbench.runner import finalize_decision_log


def state_key(group):
    return f"{group['block_id']}/{group['regime']}/B{group['batch_tokens']}"


def groups(plan):
    matrix=plan['matrix']
    if matrix['arms']!=['native','online'] or matrix['repetitions']!=2:
        raise ValueError('One unchanged two-arm contrast with two technical repetitions required')
    cells=list(itertools.product(matrix['block_ids'],matrix['regimes'],matrix['batch_tokens']))
    random.Random(plan['order_seed']).shuffle(cells)
    for block,regime,batch in cells:
        yield dict(block_id=block,regime=regime,batch_tokens=batch)


def wait_until_ns(deadline):
    while True:
        remaining=deadline-time.monotonic_ns()
        if remaining<=0:return
        if remaining>50_000:time.sleep((remaining-50_000)/1e9)


def serve(rt,layer,inputs,raw_indices,offsets,horizon_ns,*,terminal):
    """A calendar is an implicit exogenous queue, never shifted by completion.

    Rank 0 records service before the CPU command broadcast and after the CPU
    ACK barrier. The ACK follows both ranks' GPU completion. Rank 1 cannot run
    the next layer job until rank 0 has recorded completion and sent a command.
    """
    torch=rt.torch;dist=torch.distributed;cpu=rt.ep_group.cpu_group
    n=len(raw_indices)
    if len(offsets)!=n:raise ValueError('Calendar/input length mismatch')
    command=torch.zeros(1,dtype=torch.int64)
    epoch_tensor=torch.zeros(1,dtype=torch.int64)
    admits=np.zeros(n,dtype=np.int64);starts=admits.copy();finishes=admits.copy()
    rank_starts=admits.copy();rank_finishes=admits.copy()
    rt.sync();dist.barrier(group=cpu)
    if rt.rank==0:epoch_tensor[0]=time.monotonic_ns()+20_000_000
    dist.broadcast(epoch_tensor,src=0,group=cpu)
    epoch=int(epoch_tensor[0])
    for i,raw in enumerate(raw_indices):
        if rt.rank==0:
            wait_until_ns(epoch+int(offsets[i]))
            admits[i]=time.monotonic_ns()
            command[0]=i
            starts[i]=time.monotonic_ns()
        dist.broadcast(command,src=0,group=cpu)
        if int(command[0])!=i:raise AssertionError('Rank command order disagrees')
        rank_starts[i]=time.monotonic_ns()
        layer.forward(*(tensor[raw] for tensor in inputs),observe=True)
        layer.end_of_input=terminal and i==n-1
        layer.step(i)
        if terminal and i==n-1:layer.close()
        else:rt.sync()
        rank_finishes[i]=time.monotonic_ns()
        dist.barrier(group=cpu)
        if rt.rank==0:finishes[i]=time.monotonic_ns()
    if rt.rank==0:
        wait_until_ns(epoch+int(horizon_ns))
        observation_end=time.monotonic_ns()
    dist.barrier(group=cpu)
    peers=[None]*rt.world_size
    dist.all_gather_object(peers,dict(start=rank_starts.tolist(),gpu_complete=rank_finishes.tolist()),group=cpu)
    if rt.rank!=0:return None
    row=dict(n_jobs=n,arrival_offsets_ns=list(map(int,offsets)),horizon_ns=int(horizon_ns),
        epoch_monotonic_ns=epoch,observation_end_ns=observation_end,
        coordinator_admit_ns=admits.tolist(),coordinator_start_ns=starts.tolist(),
        coordinator_finish_ns=finishes.tolist(),per_rank_start_ns=[p['start'] for p in peers],
        per_rank_gpu_complete_ns=[p['gpu_complete'] for p in peers])
    row['queue_metrics']=fcfs_metrics(row)
    return row


def trial(rt,weights,base,plan,cell,group,arm,schedule,repetition,order):
    layer=make_layer(rt,weights,base,'SKIP')
    mapper=make_online(rt,layer,group['batch_tokens']) if arm=='online' else None
    if mapper is not None:layer.mapping_kernel=mapper
    layer.assert_maps_and_weights()
    noop=lambda raw:None
    preflight=numerical_check(rt,layer,cell['tensors'],cell['references'],cell['reference_steps'],noop)
    setup_start=time.monotonic_ns()
    layer.warming_up=True
    for i,raw in enumerate(cell['warmup']):
        layer.forward(*(t[raw] for t in cell['tensors']),observe=True);layer.step(i)
    layer.warming_up=False;reset_observation(layer)
    # Exercise the same CPU command/ACK and full native policy path before timing.
    serve(rt,layer,cell['tensors'],cell['conditioning'],[0]*64,1,terminal=False)
    layer.close();layer.assert_maps_and_weights()
    if len(layer.decisions)!=1 or layer.decisions[0]['allowed'] or layer.counter.sent_bytes:
        raise AssertionError('Conditioning must make one native SKIP decision and no transfer')
    reset_observation(layer);gc.collect();rt.sync()
    setup_ns=time.monotonic_ns()-setup_start
    if schedule is None:
        offsets,horizon=[0]*256,1
    else:
        validate_schedule(schedule)
        offsets,horizon=schedule['arrival_offsets_ns'],schedule['horizon_ns']
    times=serve(rt,layer,cell['tensors'],cell['sequence'],offsets,horizon,terminal=True)
    decisions=finalize_decision_log(rt,layer)
    if [d['step'] for d in decisions]!=[63,127,191]:raise AssertionError('Native cadence changed')
    for d in decisions:
        raw=cell['sequence'][d['step']-15:d['step']+1]
        expected=cell['counts'][arm][raw]
        logical=np.bincount(cell['logical'][raw].ravel(),minlength=128)*rt.world_size
        if d['allowed'] or not np.array_equal(d['physical_slot_assignments_window'],expected) or not np.array_equal(d['current_counts'],logical):
            raise AssertionError('Actual recorded routing window changed')
    terminal_window=layer.ms.expert_load_window[:,0,:].cpu().numpy()
    if not np.array_equal(terminal_window,cell['counts'][arm][cell['sequence'][-16:]]):
        raise AssertionError('Terminal physical window changed')
    layer.assert_maps_and_weights()
    final=layer.map_snapshot()
    if not np.array_equal(final,cell['mapping']) or layer.counter.sent_bytes or layer.counter.sent_messages or layer.terminal_proposals_suppressed!=1:
        raise AssertionError('Fixed resident state or normal completion violated')
    if mapper is not None:mapper.verify_metadata(layer.layer_state.logical_to_physical_map,layer.layer_state.logical_replica_count,layer.ms.physical_to_logical_map[0])
    post=numerical_check(rt,layer,cell['tensors'],cell['references'],cell['fp32_steps'],noop)
    trajectory=canonical_hash([{k:d[k] for k in ('step','current_counts','current_map','proposed_map','committed_map')} for d in decisions])
    setups=rt.gather_scalars(setup_ns)
    row=None
    if rt.rank==0:
        spans=[]
        for end in (63,127,191,255):
            begin=end-15
            spans.append(dict(end_step=end,jobs=16,
                scheduled_arrival_span_ns=offsets[end]-offsets[begin],
                service_coverage_span_ns=times['coordinator_finish_ns'][end]-times['coordinator_start_ns'][begin],
                completion_span_ns=times['coordinator_finish_ns'][end]-times['coordinator_finish_ns'][begin]))
        row={**rt.metadata,**group,**times,'arrival_schema_version':1,
            'phase':plan['phase'],'config_hash':manifest_hash(plan),'base_config_hash':manifest_hash(base),
            'arrival_source_sha256':source_identity()[0],'arm':arm,'policy':'SKIP','repetition':repetition,
            'schedule_name':'calibration_saturated' if schedule is None else f"{schedule['pattern']}-rho{schedule['rho']}",
            'arrival_schedule':schedule,'arm_order':order,'status':'complete','evidence_eligible':False,
            'interpretation':'supplementary descriptive layer-job arrival study; no held-out serving claim',
            'source_map':cell['mapping'].tolist(),'final_map':final.tolist(),
            **{k:cell[k] for k in ('capture_sha256','replay_sha256','source_prompt_ids','source_weights_sha256','source_weight_rank_sha256','resolved_kernel_config')},
            'logical_routing_sha256':array_hash(cell['logical']),'assignment_sha256':cell['route_hashes'][arm],
            'assignment_artifact':cell['artifact'],'assignment_metrics':[cell['diagnostics'][arm][i] for i in cell['sequence']],
            'adversarial_checks':cell['adversarial_checks'],'decision_log':decisions,
            'decision_trajectory_sha256':trajectory,'terminal_physical_window':terminal_window.tolist(),
            'observed_window_spans':spans,'migration_bytes':0,'migration_sent_messages':0,
            'normal_completion_verified':True,'resident_state_verified':True,'rank_assignment_agreement':True,
            'terminal_proposals_suppressed':1,'measurement_setup_revision':2,'per_rank_conditioning_setup_ns_excluded':setups,
            'conditioning_raw_indices':cell['conditioning'],'warmup_raw_indices':cell['warmup'],
            'full_unique_preflight_gpu_routes_checked':48,'gc_enabled_during_measurement':gc.isenabled(),
            'preflight_correctness':preflight,**post}
        if schedule is None:
            row['calibration_capacity']=calibration_capacity(times['coordinator_start_ns'],times['coordinator_finish_ns'])
        row['record_id']=canonical_hash({k:row[k] for k in ('config_hash','block_id','regime','batch_tokens','arm','repetition','schedule_name')})
    del layer,mapper;gc.collect();rt.sync();rt.barrier()
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','banks-dir','weights-dir','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();plan,base=load_plan(args.plan)
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    rt=Runtime()
    try:
        weights=Weights(args.weights_dir);runtime_metadata(rt,weights,base)
        peers=[None]*rt.world_size
        rt.torch.distributed.all_gather_object(peers,socket.gethostname(),group=rt.ep_group.cpu_group)
        if len(set(peers))!=1 or rt.world_size!=2:raise ValueError('Shared-node monotonic clock and two ranks required')
        clock=time.get_clock_info('monotonic')
        clock_peers=[None]*rt.world_size
        clock_contract=dict(implementation=clock.implementation,monotonic=clock.monotonic,adjustable=clock.adjustable)
        rt.torch.distributed.all_gather_object(clock_peers,clock_contract,group=rt.ep_group.cpu_group)
        if any(c!=clock_contract for c in clock_peers) or not clock.monotonic or clock.adjustable or 'CLOCK_MONOTONIC' not in clock.implementation:
            raise ValueError('Both ranks must use the shared-node CLOCK_MONOTONIC clock')
        rt.metadata['arrival_clock_contract']=clock_contract
        trajectories={}
        with kernel_override(None):
            for state_index,group in enumerate(groups(plan)):
                if plan['phase']=='evaluation' and str(plan['calibration'][state_key(group)]['allocation_id'])!=str(rt.metadata['allocation_id']):
                    raise ValueError('Calibration and evaluation must share the same allocation')
                if rt.rank==0:print(json.dumps(dict(event='prepare_state',**group)),flush=True)
                cell=prepare(rt,weights,base,SimpleNamespace(banks_dir=args.banks_dir,output=args.output),group)
                if plan['phase']=='calibration':cases=[(None,0,['native'])]
                else:
                    schedules=plan['schedules'][state_key(group)]
                    cases=[]
                    for case_index,schedule in enumerate(schedules):
                        for rep in (0,1):
                            order=['native','online'] if (state_index+case_index+rep)%2==0 else ['online','native']
                            cases.append((schedule,rep,order))
                for schedule,rep,order in cases:
                    for arm in order:
                        row=trial(rt,weights,base,plan,cell,group,arm,schedule,rep,order)
                        if rt.rank==0:
                            key=state_key(group)
                            if key in trajectories and trajectories[key]!=row['decision_trajectory_sha256']:
                                raise AssertionError('Arrival schedule or dispatcher changed logical planning')
                            trajectories[key]=row['decision_trajectory_sha256']
                            with args.output.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                            print(json.dumps(dict(event='trial_complete',state=key,arm=arm,schedule=row['schedule_name'],repetition=rep,completed=row['queue_metrics']['completed_jobs'])),flush=True)
                        rt.barrier()
                del cell;gc.collect();rt.torch.cuda.empty_cache()
    finally:rt.close()


if __name__=='__main__':main()

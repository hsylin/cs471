"""Separate charged diagnostics reusing the frozen layer and FCFS coordinator."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import socket
import time
from types import SimpleNamespace
import numpy as np
from generalization_diagnostics.plan import CONDITIONS,sequence_for,source_identity,validate
from generalization.inputs import prepare
from generalization.runner import hardware_check,validate_guard
from arrival_study.runner import serve
from arrival_fix.noop import unchanged_map_fast_path
from arrival_fix.analysis import _window_spans
from optimization.common import make_layer,canonical_hash,array_hash,manifest_hash,agree_config
from optimization.kernel_config import kernel_override
from resident_dispatch.common import runtime_metadata,numerical_check
from resident_dispatch.runner import reset_observation,make_online,_oracle
from resident_dispatch.pilot import _save_array_artifact
from skewbench.native import Runtime,Weights
from skewbench.runner import finalize_decision_log
from arrival_fix.common import canonical_gpu_uuid


class IndexedPrecomputedMapper:
    """Pointer lookup and the common copy/record adapter are fully charged.

    The frozen forward passes the exact logical tensor view into the mapper.
    No CPU readback or search over future routes occurs inside a measured call.
    """
    def __init__(self,torch,physical_ids,logical_tensors):
        from resident_dispatch.gpu import PrecomputedMapper
        self.adapter=PrecomputedMapper(torch,physical_ids)
        self.lookup={int(logical_tensors[i].data_ptr()):i for i in range(len(logical_tensors))}
        if len(self.lookup)!=len(logical_tensors):raise ValueError('Input views alias one pointer')
        self.calls=0
    def __call__(self,logical_ids,*args):
        self.adapter.raw_index=self.lookup[int(logical_ids.data_ptr())]
        self.calls+=1
        return self.adapter(logical_ids,*args)


def prepare_offline(rt,weights,base,cell,group,output):
    started=time.monotonic_ns();layer=make_layer(rt,weights,base,'SKIP')
    torch=rt.torch;arrays={};gpu={};hashes={};checks=0
    try:
        ls=layer.layer_state;load=torch.zeros_like(ls.expert_load_view)
        enabled=torch.ones((),device=rt.device,dtype=torch.bool)
        real=torch.tensor(128,device=rt.device,dtype=torch.int32)
        for dispatch in ('native','online'):
            routes=np.stack([_oracle(ids,cell['mapping'],cell['resolved_kernel_config']['BLOCK_SIZE_M'],dispatch) for ids in cell['logical']]).astype(np.int64)
            digest=hashlib.sha256(routes.tobytes(order='C')).hexdigest()
            if digest!=cell['route_hashes'][dispatch]:raise AssertionError('Offline routes differ from actual verified GPU assignments')
            arrays[dispatch]=routes;hashes[dispatch]=digest
            gpu[dispatch]=torch.from_numpy(routes.copy()).to(rt.device,dtype=cell['tensors'][1].dtype)
            adapter=IndexedPrecomputedMapper(torch,gpu[dispatch],cell['tensors'][1])
            for raw in range(272):
                load.zero_()
                actual=adapter(cell['tensors'][1][raw],load,ls.logical_to_physical_map,ls.logical_replica_count,enabled,real)
                if not np.array_equal(actual.cpu().numpy(),routes[raw]) or not np.array_equal(load.cpu().numpy(),cell['counts'][dispatch][raw]):
                    raise AssertionError('Precomputed adapter output/physical recording differs')
                checks+=1
        agree_config(rt,hashes);rt.sync()
        elapsed=rt.gather_scalars(time.monotonic_ns()-started)
        artifact=_save_array_artifact(rt,Path(output).parent/'inputs'/f"{group['block_id']}-offline-routes.npz",
            native_physical_ids=arrays['native'],online_physical_ids=arrays['online'],
            metadata=np.asarray(json.dumps(dict(route_hashes=hashes,main_assignment_artifact=cell['artifact'],
                scope='Known assignments, not deployable selector; all272 mappings and physical records checked.'),sort_keys=True)))
        return dict(gpu=gpu,route_hashes=hashes,adapter_checks=checks,preparation_ns_excluded=elapsed,artifact=artifact)
    finally:
        layer.close();del layer;gc.collect();rt.sync()


def trial(rt,weights,base,pilot,plan,cell,offline,state,condition):
    seq_name,mode=CONDITIONS[condition];sequence=sequence_for(seq_name)
    group=dict(block_id=state['content_group_id'],dataset=state['dataset'],regime='newcontent',batch_tokens=128)
    source=state['source_case']['source_block'];calendar=state['source_case']['calendar']
    layer=make_layer(rt,weights,base,'SKIP');mapper=None;dispatch='online' if mode in ('online','pre_online') else 'native'
    if mode=='online':mapper=make_online(rt,layer,128)
    elif mode.startswith('pre_'):mapper=IndexedPrecomputedMapper(rt.torch,offline['gpu'][dispatch],cell['tensors'][1])
    if mapper is not None:layer.mapping_kernel=mapper
    layer.assert_maps_and_weights()
    preflight=numerical_check(rt,layer,cell['tensors'],cell['references'],cell['reference_steps'],lambda _:None)
    with unchanged_map_fast_path(layer) as guard:
        setup_start=time.monotonic_ns();layer.warming_up=True
        for i,raw in enumerate(cell['warmup']):
            layer.forward(*(t[raw] for t in cell['tensors']),observe=True);layer.step(i)
        layer.warming_up=False;reset_observation(layer)
        serve(rt,layer,cell['tensors'],cell['conditioning'],[0]*64,1,terminal=False)
        layer.close();layer.assert_maps_and_weights()
        if len(layer.decisions)!=1 or layer.decisions[0]['allowed'] or layer.counter.sent_bytes:raise AssertionError('Conditioning path changed')
        guard.reset_counters();reset_observation(layer);gc.collect();rt.sync()
        setup_ns=time.monotonic_ns()-setup_start
        before=mapper.calls if mapper is not None else 0
        times=serve(rt,layer,cell['tensors'],sequence,calendar['arrival_offsets_ns'],calendar['horizon_ns'],terminal=True)
        calls=(mapper.calls-before) if mapper is not None else 0
        if calls!=(256 if mapper is not None else 0):raise AssertionError('Measured mapper invocation count differs')
        local_guard=guard.snapshot()
    decisions=finalize_decision_log(rt,layer)
    if [d['step'] for d in decisions]!=[63,127,191]:raise AssertionError('Native decision cadence changed')
    for d in decisions:
        raw=sequence[d['step']-15:d['step']+1]
        logical=np.bincount(cell['logical'][raw].ravel(),minlength=128)*rt.world_size
        if d['allowed'] or not np.array_equal(d['current_counts'],logical) or not np.array_equal(d['physical_slot_assignments_window'],cell['counts'][dispatch][raw]):raise AssertionError('Logical/physical window differs from exact executed inputs')
    terminal=layer.ms.expert_load_window[:,0,:].cpu().numpy()
    if not np.array_equal(terminal,cell['counts'][dispatch][sequence[-16:]]):raise AssertionError('Terminal observation differs')
    layer.assert_maps_and_weights();final=layer.map_snapshot()
    if not np.array_equal(final,cell['mapping']) or layer.counter.sent_bytes or layer.counter.sent_messages or layer.terminal_proposals_suppressed!=1:raise AssertionError('Fixed state or zero-migration contract changed')
    if mode=='online':mapper.verify_metadata(layer.layer_state.logical_to_physical_map,layer.layer_state.logical_replica_count,layer.ms.physical_to_logical_map[0])
    post=numerical_check(rt,layer,cell['tensors'],cell['references'],cell['fp32_steps'],lambda _:None)
    guard_rows=[None]*rt.world_size
    rt.torch.distributed.all_gather_object(guard_rows,local_guard,group=rt.ep_group.cpu_group)
    validate_guard(guard_rows,4,False)
    setup=rt.gather_scalars(setup_ns);row=None
    if rt.rank==0:
        row={**rt.metadata,**group,**times,'generalization_diagnostic_schema_version':1,'phase':'diagnostic',
            'diagnostic_config_hash':manifest_hash(plan),'diagnostic_source_sha256':plan['diagnostic_source_sha256'],
            'main_execution_source_sha256':pilot['execution_source_sha256'],'pilot_plan_sha256':pilot['freeze']['sha256'],
            'condition':condition,'sequence_name':seq_name,'dispatch_mode':mode,'condition_order':state['condition_order'],
            'content_group_id':group['block_id'],'arrival_block_id':source['id'],'arrival_type':'code','arrival_date':source['source_date_utc'],
            'rate_kind':'calibrated','arrival_schedule':calendar,'status':'complete','evidence_eligible':False,'diagnostic_only':True,
            'policy':'SKIP','source_map':cell['mapping'].tolist(),'final_map':final.tolist(),
            **{k:cell[k] for k in ('capture_sha256','replay_sha256','source_prompt_ids','source_weights_sha256','source_weight_rank_sha256','resolved_kernel_config')},
            'logical_routing_sha256':array_hash(cell['logical']),
            'executed_logical_routing_sha256':array_hash(cell['logical'][sequence]),
            'executed_combining_weights_sha256':array_hash(cell['tensors'][2][sequence].cpu().numpy()),
            'assignment_sha256':cell['route_hashes'][dispatch],'assignment_artifact':cell['artifact'],'offline_assignment_artifact':offline['artifact'],
            'assignment_metrics':[cell['diagnostics'][dispatch][i] for i in sequence],
            'decision_log':decisions,'decision_trajectory_sha256':canonical_hash([{k:d[k] for k in ('step','current_counts','current_map','proposed_map','committed_map')} for d in decisions]),
            'terminal_physical_window':terminal.tolist(),'observed_window_spans':_window_spans(times),
            'migration_bytes':0,'migration_sent_messages':0,'terminal_proposals_suppressed':1,'normal_completion_verified':True,
            'resident_state_verified':True,'rank_assignment_agreement':True,'guard_symbols_restored':True,'noop_guard_ranks':guard_rows,
            'online_selection_cost_included':mode=='online','precomputed_selection_cost_excluded':mode.startswith('pre_'),
            'precomputed_adapter_cost_included':mode.startswith('pre_'),'mapper_calls_measured':calls,
            'precomputed_gpu_adapter_checks':offline['adapter_checks'],'precomputed_preparation_ns_excluded':offline['preparation_ns_excluded'],
            'per_rank_conditioning_setup_ns_excluded':setup,'per_rank_input_preparation_ns_excluded':cell['preparation_ns'],
            'warmup_raw_indices':cell['warmup'],'conditioning_raw_indices':cell['conditioning'],'measured_raw_indices':sequence,
            'unique_measured_job_combinations':len(set(sequence)),'raw_job_frequency':{str(i):sequence.count(i) for i in sorted(set(sequence))},
            'full_unique_preflight_gpu_routes_checked':272,'preflight_correctness':preflight,'postflight_reference_raw_indices':cell['fp32_steps'],
            'measurement_setup_revision':2,'gc_enabled_during_measurement':gc.isenabled(),**post}
        row['record_id']=canonical_hash({k:row[k] for k in ('diagnostic_config_hash','phase','content_group_id','condition')})
    del layer,mapper;gc.collect();rt.sync();rt.barrier();return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','banks-dir','weights-dir','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();plan=json.loads(args.plan.read_text());pilot,base=validate(plan)
    uuids=hardware_check(pilot)
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True);rt=Runtime()
    try:
        weights=Weights(args.weights_dir);runtime_metadata(rt,weights,base)
        clock=time.get_clock_info('monotonic');contract=dict(implementation=clock.implementation,monotonic=clock.monotonic,adjustable=clock.adjustable)
        peers=[None]*rt.world_size
        rt.torch.distributed.all_gather_object(peers,(socket.gethostname(),contract,uuids),group=rt.ep_group.cpu_group)
        if rt.world_size!=2 or any(x!=peers[0] for x in peers) or not clock.monotonic or clock.adjustable or 'CLOCK_MONOTONIC' not in clock.implementation:raise ValueError('Shared same-host monotonic clock required')
        ranks=[None]*rt.world_size;uuid=canonical_gpu_uuid(rt.torch.cuda.get_device_properties(rt.device).uuid)
        rt.torch.distributed.all_gather_object(ranks,uuid,group=rt.ep_group.cpu_group)
        if sorted(ranks)!=sorted(uuids) or str(rt.metadata['allocation_id'])!=str(plan['hardware']['allocation_id']):raise ValueError('Physical allocation differs')
        rt.metadata.update(arrival_clock_contract=contract,physical_gpu_uuids=uuids,rank_gpu_uuids=ranks)
        with kernel_override(None):
            for state in plan['states']:
                group=dict(block_id=state['content_group_id'],dataset=state['dataset'],regime='newcontent',batch_tokens=128)
                preparation_start=time.monotonic_ns()
                cell=prepare(rt,weights,base,SimpleNamespace(banks_dir=args.banks_dir,output=args.output),group,pilot)
                rt.sync()
                cell['preparation_ns']=rt.gather_scalars(time.monotonic_ns()-preparation_start)
                offline=prepare_offline(rt,weights,base,cell,group,args.output)
                for condition in state['condition_order']:
                    row=trial(rt,weights,base,pilot,plan,cell,offline,state,condition)
                    if rt.rank==0:
                        with args.output.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                        print(json.dumps({k:row[k] for k in ('content_group_id','condition','status')}),flush=True)
                    rt.barrier()
                del cell,offline;gc.collect();rt.torch.cuda.empty_cache()
    finally:rt.close()
if __name__=='__main__':main()

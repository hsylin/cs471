"""Actual pinned stock MoE module comparison, isolated from prior harness arms."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import numpy as np
from stock_comparison.common import load_plan,ARMS,validate_calendar
from stock_comparison.data import digest
from stock_comparison.inputs import load_inputs,array_digest,WARMUP
from stock_comparison.timing import serve
from arrival_study.schedules import calibration_capacity


def progress(rt,**values):
    if rt.rank==0:print(json.dumps(values,allow_nan=False),flush=True)


def gather(rt,value):
    result=[None]*rt.world_size
    rt.torch.distributed.all_gather_object(result,value,group=rt.ep_group.cpu_group)
    return result


def check_hardware(rt,plan):
    host=socket.gethostname().split('.')[0]
    uuids=sorted(subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).split())
    expected=plan['hardware']
    if host!=expected['hostname'] or uuids!=sorted(expected['gpu_uuids']) or os.environ['SLURM_JOB_ID']!=expected['allocation_id']:
        raise ValueError('Wrong reserved hardware; measurement forbidden')
    if rt.world_size!=2 or len(set(gather(rt,host)))!=1:raise ValueError('Need two ranks on common monotonic-clock host')
    c=time.get_clock_info('monotonic')
    clock=dict(implementation=c.implementation,monotonic=c.monotonic,adjustable=c.adjustable)
    if not c.monotonic or c.adjustable or 'CLOCK_MONOTONIC' not in c.implementation or any(x!=clock for x in gather(rt,clock)):
        raise ValueError('Common CLOCK_MONOTONIC contract failed')
    return dict(hostname=host,gpu_uuids=uuids,allocation_id=os.environ['SLURM_JOB_ID'],clock=clock)


def fp32_reference(rt,weights,hidden,ids,combining):
    torch=rt.torch;result=torch.zeros_like(hidden,dtype=torch.float32)
    a=ids.cpu().numpy()
    for expert in np.unique(a):
        positions=np.argwhere(a==expert)
        rows=torch.tensor(positions[:,0],device=rt.device);choices=torch.tensor(positions[:,1],device=rt.device)
        w13=weights.tensors['w13_weight'][int(expert)].to(rt.device,torch.float32)
        w2=weights.tensors['w2_weight'][int(expert)].to(rt.device,torch.float32)
        gate,up=(hidden[rows].float()@w13.T).chunk(2,dim=-1)
        output=(torch.nn.functional.silu(gate)*up)@w2.T
        result.index_add_(0,rows,output*combining[rows,choices,None])
    return result


def prepare_references(rt,layers,weights,hidden,output,group_id):
    """Whole-input functional checks outside every primary timed trial."""
    torch=rt.torch;n=len(hidden)
    reference=torch.empty_like(hidden)
    ids_all=np.empty((n,128,8),dtype=np.int32)
    weights_all=np.empty((n,128,8),dtype=np.float32)
    fp32_indices=sorted(set([WARMUP,WARMUP+749,WARMUP+3749,n-1]))
    positions=torch.linspace(0,127,16,device=rt.device).long()
    original=hidden.clone()
    maximum=0.
    off=layers['stockoff']
    for raw in range(n):
        ids,combining=off.routing(hidden[raw])
        ids_all[raw]=ids.cpu().numpy();weights_all[raw]=combining.float().cpu().numpy()
        reference[raw].copy_(off.forward(hidden[raw],observe=False))
        if raw in fp32_indices:
            fp32=fp32_reference(rt,weights,hidden[raw,positions],ids[positions],combining[positions])
            torch.testing.assert_close(reference[raw,positions].float(),fp32,atol=.02,rtol=.03)
            maximum=max(maximum,float((reference[raw,positions].float()-fp32).abs().max()))
        if raw%1024==0:progress(rt,event='stock_reference',group=group_id,input=raw,total=n)
    if not torch.equal(hidden,original):raise AssertionError('Stock block modified replay input')
    route_hash=array_digest(ids_all);combining_hash=array_digest(weights_all)
    peers=gather(rt,(route_hash,combining_hash))
    if len(set(peers))!=1:raise AssertionError('Ranks disagree on actual logical routes/weights')
    results={}
    for arm,layer in layers.items():
        delta=0.
        assignments=[]
        if arm!='stockoff':
            for raw in range(n):
                ids,combining=layer.routing(hidden[raw])
                if not np.array_equal(ids.cpu().numpy(),ids_all[raw]) or not np.array_equal(combining.float().cpu().numpy(),weights_all[raw]):
                    raise AssertionError(f'{arm}: logical route or combining weight changed at{raw}')
                if arm=='ours':assignments.append(layer.verify_online_assignment(hidden[raw]))
                actual=layer.forward(hidden[raw],observe=False)
                torch.testing.assert_close(actual.float(),reference[raw].float(),atol=.02,rtol=.03)
                delta=max(delta,float((actual.float()-reference[raw].float()).abs().max()))
                if raw%1024==0:progress(rt,event='preflight',group=group_id,arm=arm,input=raw,total=n)
        if not torch.equal(hidden,original):raise AssertionError(f'{arm}: modified replay input')
        results[arm]=dict(correctness_pass=True,full_batch_inputs_checked=n,
            full_batch_max_abs_difference=max(gather(rt,delta)),logical_routing_sha256=route_hash,
            float32_combining_weights_sha256=combining_hash,
            logical_ids_equal=True,combining_weights_equal=True,rank_routing_agreement=True,
            input_unchanged=True,atol=.02,rtol=.03,fp32_job_indices=fp32_indices,
            online_assignment_inputs_checked=len(assignments),
            physical_assignment_digest=None if not assignments else digest([r['physical_sha256'] for r in assignments]),
            fp32_tokens_per_job=16,fp32_max_abs_error=max(gather(rt,maximum)))
        if assignments and rt.rank==0:
            (Path(output)/'inputs'/(group_id+'-resident-diagnostics.json')).write_text(json.dumps(assignments)+'\n')
        layer.reset_trial()
    if rt.rank==0:
        path=Path(output)/'inputs'/(group_id+'-stock-routes.npz')
        np.savez_compressed(path,logical_ids=ids_all,combining_weights=weights_all,
            logical_routing_sha256=np.asarray(route_hash),float32_combining_weights_sha256=np.asarray(combining_hash))
    del original
    rt.sync();rt.barrier()
    return reference,results


def postflight(rt,layer,hidden,reference):
    indices=np.linspace(WARMUP,len(hidden)-1,16,dtype=np.int64).tolist()
    delta=0.
    for raw in indices:
        actual=layer.forward(hidden[raw],observe=False)
        rt.torch.testing.assert_close(actual.float(),reference[raw].float(),atol=.02,rtol=.03)
        delta=max(delta,float((actual.float()-reference[raw].float()).abs().max()))
    return dict(correctness_pass=True,full_batch_inputs_checked=len(indices),raw_indices=indices,
        full_batch_max_abs_difference=max(gather(rt,delta)),atol=.02,rtol=.03)


def run_trial(rt,layer,hidden,reference,preflight,plan,entry,case,input_meta,hardware):
    arm=layer.arm
    setup_begin=time.monotonic_ns()
    layer.reset_trial()
    with layer.measured():
        for i in range(64):
            layer.forward(hidden[i%WARMUP],observe=True);layer.step(i);rt.sync()
        layer.complete_pending()
    layer.reset_trial()
    gc.collect();rt.sync();rt.barrier()
    with layer.measured():
        guard=layer.guard
        start=layer.snapshot()
        if case is None:
            offsets=[0]*plan['jobs'];horizon=1;calendar=None
        else:
            calendar=case['calendar'];validate_calendar(calendar,case['source_block'])
            offsets=calendar['arrival_offsets_ns'];horizon=calendar['horizon_ns']
        setup_ns=time.monotonic_ns()-setup_begin
        rt.torch.cuda.reset_peak_memory_stats()
        times=serve(rt,layer,hidden,range(WARMUP,len(hidden)),offsets,horizon)
        peak=rt.torch.cuda.max_memory_allocated()
        end=layer.snapshot()
        guard_record=guard.snapshot() if guard is not None else None
    audit=layer.audit()
    post=postflight(rt,layer,hidden,reference)
    peers=gather(rt,dict(start=start,end=end,audit=audit,shortcut_guard=guard_record,
        peak_allocated_bytes=peak,peak_memory_scope='Whole process; includes untimed prepared inputs/references and all resident comparator objects; not standalone engine memory',conditioning_setup_ns_excluded=setup_ns))
    row=None
    if rt.rank==0:
        row=dict(schema='stock_comparison_trial_v1',phase=plan['phase'],arm=arm,
            content_group_id=entry['block_id'],block_id=entry['block_id'],dataset=entry['dataset'],
            arrival_type=None if calendar is None else calendar['arrival_type'],
            arrival_date=None if calendar is None else calendar['arrival_date'],
            rho=None if calendar is None else calendar['rho'],calendar=calendar,
            arm_order=['stockoff'] if case is None else case['arm_order'],
            plan_sha256=plan['content_sha256'],source_sha256=plan['source_sha256'],
            input_meta=input_meta,preflight=preflight,postflight=post,source_weights=layer.weights.metadata,
            per_rank_state=peers,native_metadata=end,runtime_metadata=rt.metadata,
            status='complete',evidence_eligible=plan['phase']=='final',
            **hardware,**times)
        if case is None:
            row['calibration_capacity']=calibration_capacity(times['coordinator_start_ns'],times['coordinator_finish_ns'],expected_jobs=plan['jobs'])
        row['record_id']=digest(row)
    rt.barrier()
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','banks-dir','weights-dir','model-path','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();plan=load_plan(args.plan)
    target=args.output/'trials'
    from stock_comparison.native import StockRuntime,StockLayer
    from skewbench.native import Weights
    rt=StockRuntime(args.model_path)
    try:
        if rt.rank==0:
            if target.exists():raise FileExistsError(target)
            target.mkdir(parents=True)
        rt.barrier()
        hardware=check_hardware(rt,plan);weights=Weights(args.weights_dir)
        selected=['stockoff'] if plan['phase']=='calibration' else ARMS
        layers={arm:StockLayer(rt,weights,arm) for arm in selected}
        for entry in plan['banks']:
            progress(rt,event='prepare_content',group=entry['block_id'],phase=plan['phase'])
            hidden,input_meta=load_inputs(rt,entry,args.banks_dir,args.output,jobs=plan['jobs'])
            if 'stockon32' in layers and 'native_async_preconditioning' not in rt.metadata:
                on=layers['stockon32'];on.reset_trial()
                with on.measured():
                    for i in range(750):
                        on.forward(hidden[i%WARMUP],observe=True);on.step(i);rt.sync()
                    on.complete_pending()
                rt.metadata['native_async_preconditioning']=dict(
                    timing_evidence=False,source_group=entry['block_id'],jobs=750,
                    purpose='Warm native async planner/transport once; restore canonical initial state before all trials',
                    per_rank=gather(rt,on.audit()))
                on.reset_trial()
            if any(layer.gate_sha256!=input_meta['gate_weight_sha256'] for layer in layers.values()):
                raise AssertionError('Measured router differs from authenticated checkpoint capture gate')
            reference,preflights=prepare_references(rt,layers,weights,hidden,args.output,entry['block_id'])
            if 'output_reduction_probe' not in rt.metadata:
                layers['stockoff'].attest_output_reduction(hidden[WARMUP])
            cases=[None] if plan['phase']=='calibration' else [c for c in plan['cases'] if c['block_id']==entry['block_id']]
            for case in cases:
                order=['stockoff'] if case is None else case['arm_order']
                for arm in order:
                    progress(rt,event='trial_start',group=entry['block_id'],arm=arm,
                        rho=None if case is None else case['calendar']['rho'],
                        arrival=None if case is None else case['calendar']['arrival_type'])
                    row=run_trial(rt,layers[arm],hidden,reference,preflights[arm],plan,entry,case,input_meta,hardware)
                    if rt.rank==0:
                        with (target/(entry['block_id']+'.jsonl')).open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                        progress(rt,event='trial_complete',group=entry['block_id'],arm=arm,mean_response_ns=row['queue_metrics']['summary_ns']['response']['mean'],record_id=row['record_id'])
            del hidden,reference;gc.collect();rt.torch.cuda.empty_cache();rt.sync();rt.barrier()
    finally:rt.close()
if __name__=='__main__':main()

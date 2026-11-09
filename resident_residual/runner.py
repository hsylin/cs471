"""Contemporaneous native-off/current/fix arrival experiment, one actual MoE layer."""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np

from resident_residual.common import load_plan,ARMS,PREVIOUS_CLOSURE,validate_calendar
from resident_residual.inputs import load_inputs,WARMUP,array_digest
from resident_residual.native import build_layers,alignment_scope
from stock_comparison.data import digest
from stock_comparison.runner import progress,gather,check_hardware,fp32_reference,postflight
from stock_comparison.timing import serve
from arrival_study.schedules import calibration_capacity


def prepare_references(rt,layers,weights,hidden,output,group_id):
    torch=rt.torch;n=len(hidden);reference=torch.empty_like(hidden)
    ids_all=np.empty((n,128,8),np.int32);weights_all=np.empty((n,128,8),np.float32)
    fp32_indices=[WARMUP,WARMUP+749,WARMUP+3749,n-1]
    positions=torch.linspace(0,127,16,device=rt.device).long()
    original=hidden.clone();maximum=0.;off=layers['stockoff']
    for raw in range(n):
        ids,combining=off.routing(hidden[raw])
        ids_all[raw]=ids.cpu().numpy();weights_all[raw]=combining.float().cpu().numpy()
        reference[raw].copy_(off.forward(hidden[raw],observe=False))
        if raw in fp32_indices:
            fp32=fp32_reference(rt,weights,hidden[raw,positions],ids[positions],combining[positions])
            torch.testing.assert_close(reference[raw,positions].float(),fp32,atol=.02,rtol=.03)
            maximum=max(maximum,float((reference[raw,positions].float()-fp32).abs().max()))
        if raw%1024==0:progress(rt,event='reference',group=group_id,input=raw,total=n)
    if not torch.equal(hidden,original):raise AssertionError('Reference altered input')
    route_hash=array_digest(ids_all);combining_hash=array_digest(weights_all)
    if len(set(gather(rt,(route_hash,combining_hash))))!=1:raise AssertionError('Rank routing disagreement')
    results={};assignment_digests={}
    for arm,layer in layers.items():
        delta=0.;assignments=[]
        with alignment_scope(layer,probe=True) as alignment:
            for raw in range(n):
                if arm!='stockoff':
                    ids,combining=layer.routing(hidden[raw])
                    if not np.array_equal(ids.cpu().numpy(),ids_all[raw]) or not np.array_equal(combining.cpu().numpy(),weights_all[raw]):
                        raise AssertionError('Changed logical routes/combining weights')
                    assignments.append(layer.verify_online_assignment(hidden[raw]))
                actual=layer.forward(hidden[raw],observe=False)
                torch.testing.assert_close(actual.float(),reference[raw].float(),atol=.02,rtol=.03)
                delta=max(delta,float((actual.float()-reference[raw].float()).abs().max()))
                if raw%1024==0:progress(rt,event='preflight',group=group_id,arm=arm,input=raw,total=n)
        geometry=gather(rt,alignment.verify())
        geometry_file=Path(output)/'inputs'/(group_id+'-'+arm+'-alignment.json')
        if rt.rank==0:geometry_file.write_text(json.dumps(geometry)+'\n')
        geometry_summary=[{k:v for k,v in x.items() if k!='probe_records'} for x in geometry]
        rt.barrier()
        if not torch.equal(hidden,original):raise AssertionError('Treatment altered input')
        assignment_digest=None if not assignments else digest([a['physical_sha256'] for a in assignments])
        if assignment_digest is not None:assignment_digests[arm]=assignment_digest
        results[arm]=dict(alignment_probe=geometry_summary,alignment_artifact=str(geometry_file),alignment_artifact_sha256=__import__('followup.router_tapes',fromlist=['file_sha256']).file_sha256(geometry_file),correctness_pass=True,full_batch_inputs_checked=n,
            full_batch_max_abs_difference=max(gather(rt,delta)),logical_routing_sha256=route_hash,
            float32_combining_weights_sha256=combining_hash,logical_ids_equal=True,combining_weights_equal=True,
            rank_routing_agreement=True,input_unchanged=True,atol=.02,rtol=.03,fp32_job_indices=fp32_indices,
            online_assignment_inputs_checked=len(assignments),physical_assignment_digest=assignment_digest,
            fp32_tokens_per_job=16,fp32_max_abs_error=max(gather(rt,maximum)))
        if assignments and rt.rank==0:
            (Path(output)/'inputs'/(group_id+'-'+arm+'-resident-diagnostics.json')).write_text(json.dumps(assignments)+'\n')
        layer.reset_trial()
    if len(set(assignment_digests.values()))>1:raise AssertionError('Fix changed physical assignment policy')
    if rt.rank==0:
        np.savez_compressed(Path(output)/'inputs'/(group_id+'-stock-routes.npz'),logical_ids=ids_all,
            combining_weights=weights_all,logical_routing_sha256=np.asarray(route_hash),
            float32_combining_weights_sha256=np.asarray(combining_hash))
    del original;rt.sync();rt.barrier()
    return reference,results


def run_trial(rt,layer,arm,hidden,reference,preflight,plan,entry,case,input_meta,hardware):
    setup=time.monotonic_ns();layer.reset_trial()
    with alignment_scope(layer), layer.measured():
        for i in range(64):layer.forward(hidden[i%WARMUP]);layer.step(i);rt.sync()
        layer.complete_pending()
    layer.reset_trial();gc.collect();rt.sync();rt.barrier()
    with alignment_scope(layer) as alignment, layer.measured():
        start=layer.snapshot();guard=layer.guard
        if case is None:offsets=[0]*4096;horizon=1;calendar=None
        else:
            calendar=case['calendar'];validate_calendar(calendar,case['source_block'])
            offsets=calendar['arrival_offsets_ns'];horizon=calendar['horizon_ns']
        setup_ns=time.monotonic_ns()-setup
        rt.torch.cuda.reset_peak_memory_stats()
        times=serve(rt,layer,hidden,range(WARMUP,len(hidden)),offsets,horizon)
        peak=rt.torch.cuda.max_memory_allocated();end=layer.snapshot()
        guard_record=None if guard is None else guard.snapshot()
    audit=layer.audit()
    with alignment_scope(layer):post=postflight(rt,layer,hidden,reference)
    peers=gather(rt,dict(start=start,end=end,audit=audit,alignment=alignment.snapshot(),shortcut_guard=guard_record,
        peak_allocated_bytes=peak,peak_memory_scope='Whole process, all comparators and prepared inputs/references; not standalone engine',
        conditioning_setup_ns_excluded=setup_ns))
    row=None
    if rt.rank==0:
        row=dict(schema='resident_residual_trial_v1',phase=plan['phase'],arm=arm,
            content_group_id=entry['block_id'],block_id=entry['block_id'],dataset=entry['dataset'],
            arrival_type=None if calendar is None else calendar['arrival_type'],
            arrival_date=None if calendar is None else calendar['arrival_date'],
            rho=None if calendar is None else calendar['rho'],calendar=calendar,
            arm_order=['stockoff'] if case is None else case['arm_order'],
            plan_sha256=plan['content_sha256'],source_sha256=plan['source_sha256'],previous_closure_sha256=PREVIOUS_CLOSURE,
            input_meta=input_meta,preflight=preflight,postflight=post,source_weights=layer.weights.metadata,
            per_rank_state=peers,native_metadata=end,runtime_metadata=rt.metadata,status='complete',
            inference_role='main',
            evidence_eligible=plan['phase']=='final',**hardware,**times)
        if case is None:row['calibration_capacity']=calibration_capacity(times['coordinator_start_ns'],times['coordinator_finish_ns'],expected_jobs=4096)
        row['record_id']=digest(row)
    rt.barrier();return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','banks-dir','weights-dir','model-path','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();plan=load_plan(args.plan);target=args.output/'trials'
    from stock_comparison.native import StockRuntime,StockLayer
    from skewbench.native import Weights
    rt=StockRuntime(args.model_path)
    try:
        if rt.rank==0:
            if target.exists():raise FileExistsError(target)
            target.mkdir(parents=True)
        rt.barrier();hardware=check_hardware(rt,plan);weights=Weights(args.weights_dir)
        layers=({'stockoff':StockLayer(rt,weights,'stockoff')} if plan['phase']=='calibration'
                else build_layers(rt,weights,selected=plan['selected_fix']))
        rt.metadata.update(study_id='resident_residual_v1',selected_fix=plan['selected_fix'],
            tf32_allowed=rt.torch.backends.cuda.matmul.allow_tf32,
            float32_matmul_precision=rt.torch.get_float32_matmul_precision())
        precision=gather(rt,dict(allow_tf32=rt.torch.backends.cuda.matmul.allow_tf32,
            matmul_precision=rt.torch.get_float32_matmul_precision()))
        if any(x!={'allow_tf32':False,'matmul_precision':'highest'} for x in precision):
            raise AssertionError('Dense FP32 reference requires explicit highest/noTF32 execution')
        rt.metadata['precision_per_rank']=precision
        if 'fix' in layers and layers['fix'].selector_kind=='fused':
            from resident_fastpath.fused_validation import verify_adversarial_fused
            from resident_residual.alignment import verify_adversarial_alignment
            rt.metadata['fused_adversarial']=verify_adversarial_fused(rt,layers['fix'].mapper)
            rt.metadata['alignment_adversarial']=verify_adversarial_alignment(rt,layers['stockoff'],layers['fix'])
        for entry in plan['banks']:
            progress(rt,event='prepare_content',group=entry['block_id'],phase=plan['phase'])
            hidden,input_meta=load_inputs(rt,entry,args.banks_dir,args.output)
            if any(l.gate_sha256!=input_meta['gate_weight_sha256'] for l in layers.values()):raise AssertionError('Wrong gate')
            reference,preflight=prepare_references(rt,layers,weights,hidden,args.output,entry['block_id'])
            if 'output_reduction_probe_per_rank' not in rt.metadata:
                probes=gather(rt,layers['stockoff'].attest_output_reduction(hidden[WARMUP]))
                if len({digest(p['observed_successful_backends']) for p in probes})!=1:raise AssertionError('Rank reduction disagreement')
                rt.metadata['output_reduction_probe_per_rank']=probes
            cases=[None] if plan['phase']=='calibration' else [c for c in plan['cases'] if c['block_id']==entry['block_id']]
            for case in cases:
                for arm in (['stockoff'] if case is None else case['arm_order']):
                    progress(rt,event='trial_start',group=entry['block_id'],arm=arm,rho=None if case is None else case['calendar']['rho'])
                    row=run_trial(rt,layers[arm],arm,hidden,reference,preflight[arm],plan,entry,case,input_meta,hardware)
                    if rt.rank==0:
                        with (target/(entry['block_id']+'.jsonl')).open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                        progress(rt,event='trial_complete',group=entry['block_id'],arm=arm,mean_response_ns=row['queue_metrics']['summary_ns']['response']['mean'],record_id=row['record_id'])
            del hidden,reference;gc.collect();rt.torch.cuda.empty_cache();rt.sync();rt.barrier()
    finally:rt.close()


if __name__=='__main__':main()

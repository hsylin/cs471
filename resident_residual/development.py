"""One candidate, reused pilot inputs, balanced contemporaneous development."""
import argparse,gc,json,time
from pathlib import Path
import numpy as np
from followup.router_tapes import load_router_bank,file_sha256
from resident_residual.native import build_layers,alignment_scope
from resident_residual.alignment import verify_adversarial_alignment
from resident_fastpath.fused_validation import verify_adversarial_fused
from stock_comparison.common import HOST,GPU_UUIDS
from stock_comparison.data import digest
from stock_comparison.inputs import build_indices,array_digest
from stock_comparison.native import StockRuntime
from stock_comparison.runner import check_hardware,gather,progress,fp32_reference
from stock_comparison.timing import serve

ARMS=('stockoff','previous','fix')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('model-path','weights-dir','banks-dir','capture-index','plan','output'):p.add_argument('--'+k,type=Path,required=True)
    args=p.parse_args();plan=json.loads(args.plan.read_text());root=Path(__file__).resolve().parents[1]
    if plan['content_sha256']!=digest({k:v for k,v in plan.items() if k!='content_sha256'}):raise ValueError('Changed plan')
    if plan['source_sha256']!=digest(plan['source_files']) or any(file_sha256(root/rel)!=sha for rel,sha in plan['source_files'].items()):raise ValueError('Changed source')
    if plan['capture_index_sha256']!=file_sha256(args.capture_index):raise ValueError('Changed captureindex')
    if plan['arms']!=list(ARMS) or plan['measured_executions']!=24:raise ValueError('Changed candidate matrix')
    expected={'fix-'+d+'-pilot-0'+str(i) for d in ('gsm8k','wildchat') for i in (1,2)}
    if len(plan['groups'])!=4 or {g['block_id'] for g in plan['groups']}!=expected:raise ValueError('Changed diagnostic groups')
    for g in plan['groups']:
        if len(g['orders'])!=2 or sorted(g['orders'][0])!=sorted(ARMS) or g['orders'][1]!=g['orders'][0][::-1]:raise ValueError('Changed orders')
    from skewbench.native import Weights
    rt=StockRuntime(args.model_path)
    try:
        if rt.rank==0:args.output.mkdir(parents=True,exist_ok=False)
        rt.barrier();hw=check_hardware(rt,{'hardware':dict(hostname=HOST,gpu_uuids=GPU_UUIDS,allocation_id='4872248')})
        weights=Weights(args.weights_dir);layers=build_layers(rt,weights,selected=plan['candidate'])
        if rt.torch.backends.cuda.matmul.allow_tf32 or rt.torch.get_float32_matmul_precision()!='highest':raise AssertionError('FP32 precision')
        adv=verify_adversarial_fused(rt,layers['fix'].mapper)
        alignment=verify_adversarial_alignment(rt,layers['stockoff'],layers['fix'])
        if rt.rank==0:
            (args.output/'fused-adversarial.json').write_text(json.dumps(adv,indent=2)+'\n')
            (args.output/'alignment-adversarial.json').write_text(json.dumps(alignment,indent=2)+'\n')
        entries={e['block_id']:e for e in json.loads(args.capture_index.read_text())}
        for group in plan['groups']:
            entry=entries[group['block_id']];bank=load_router_bank(args.banks_dir/Path(entry['bank_file']).name)
            if bank.sha256!=entry['capture_sha256']:raise ValueError('Changed pilot bank')
            indices,meta=build_indices(len(bank.hidden_states_bf16),entry['block_id'],seed=471703,jobs=64);indices=indices[16:]
            hidden=rt.torch.empty((64,128,2048),device=rt.device,dtype=rt.torch.bfloat16)
            for i,idx in enumerate(indices):hidden[i].copy_(rt.torch.from_numpy(np.asarray(bank.hidden_states_bf16[idx],dtype=np.uint16).copy()).view(rt.torch.bfloat16))
            rt.sync();ids_all=[];combining_all=[];reference=[];physical=[];max_fp32=0.
            positions=rt.torch.linspace(0,127,16,device=rt.device).long()
            for i,h in enumerate(hidden):
                ids,combining=layers['stockoff'].routing(h)
                ids_all.append(ids.cpu().numpy());combining_all.append(combining.float().cpu().numpy())
                reference.append(layers['stockoff'].forward(h,observe=False).clone())
                signatures=[]
                for arm in ('previous','fix'):signatures.append(layers[arm].verify_online_assignment(h)['physical_sha256'])
                if len(set(signatures))!=1:raise AssertionError('Candidate changed exact physical policy')
                physical.append(layers['fix'].mapper.last_physical_ids.cpu().numpy().copy())
                if i in (0,21,42,63):
                    dense=fp32_reference(rt,weights,h[positions],ids[positions],combining[positions])
                    rt.torch.testing.assert_close(reference[-1][positions].float(),dense,atol=.02,rtol=.03)
                    max_fp32=max(max_fp32,float((reference[-1][positions].float()-dense).abs().max()))
            # Isolated submission timings are mechanism diagnostics, not service trials.
            # Complete each invocation only after stopping its host submission clock.
            if group is plan['groups'][0]:
                submission=[]
                for rep,order in enumerate((('previous','fix'),('fix','previous'))):
                    for arm in order:
                        mapper=layers[arm].mapper
                        for i in range(32):
                            ids,_=layers['stockoff'].routing(hidden[i%64]);rt.sync()
                            begin=time.monotonic_ns();mapper(ids);finish=time.monotonic_ns();rt.sync()
                            submission.append(dict(arm=arm,repeat=rep,input=i,submission_ns=finish-begin))
                submitted=gather(rt,submission)
                if rt.rank==0:(args.output/'submission-diagnostic.json').write_text(json.dumps(dict(scope='Host mapper-call submission only; excludes subsequent completion wait; not exclusive GPU time or layer service, not an independent trial',per_rank=submitted))+'\n')
            probes={}
            for arm,layer in layers.items():
                layer.reset_trial()
                with alignment_scope(layer,probe=True) as guard:
                    for i,h in enumerate(hidden):rt.torch.testing.assert_close(layer.forward(h,observe=False).float(),reference[i].float(),atol=.02,rtol=.03)
                probes[arm]=guard.verify();layer.reset_trial()
            reduction=gather(rt,layers['stockoff'].attest_output_reduction(hidden[0]))
            hashes=(array_digest(np.stack(ids_all)),array_digest(np.stack(combining_all)),array_digest(np.stack(physical)))
            if len(set(gather(rt,hashes)))!=1:raise AssertionError('Rank assignment mismatch')
            identity=dict(group=entry['block_id'],dataset=entry['dataset'],bank_sha256=bank.sha256,indices_sha256=array_digest(indices),logical_routes_sha256=hashes[0],combining_weights_sha256=hashes[1],exact_physical_assignment_sha256=hashes[2],unique_batches=64,jobs=512,sequence='Previously observed64pilotbatches cycled8times; descriptive development only',source_sha256=plan['source_sha256'],plan_sha256=file_sha256(args.plan),candidate=plan['candidate'],fp32_max_abs_error=max(gather(rt,max_fp32)),atol=.02,rtol=.03)
            gathered=gather(rt,probes)
            if rt.rank==0:
                np.savez_compressed(args.output/(entry['block_id']+'-inputs.npz'),indices=indices,logical_ids=np.stack(ids_all),combining_weights=np.stack(combining_all),physical_ids=np.stack(physical))
                (args.output/(entry['block_id']+'-geometry.json')).write_text(json.dumps(gathered)+'\n')
            for rep,order in enumerate(group['orders']):
                for arm in order:
                    layer=layers[arm];layer.reset_trial()
                    with alignment_scope(layer),layer.measured():
                        for i,h in enumerate(hidden):layer.forward(h);layer.step(i);rt.sync()
                    layer.reset_trial();gc.collect();rt.sync();rt.barrier()
                    with alignment_scope(layer) as guard,layer.measured():
                        start=layer.snapshot();timing=serve(rt,layer,hidden,[i%64 for i in range(512)],[0]*512,1);end=layer.snapshot()
                    audit=layer.audit()
                    with alignment_scope(layer):
                        for i,h in enumerate(hidden):rt.torch.testing.assert_close(layer.forward(h,observe=False).float(),reference[i].float(),atol=.02,rtol=.03)
                    states=gather(rt,dict(start=start,end=end,audit=audit,alignment=guard.snapshot(),postflight_full_inputs=64))
                    if rt.rank==0:
                        row=dict(schema='resident_residual_development_v1',**identity,variant=arm,repeat=rep,arm_order=order,per_rank_state=states,output_reduction_probes=reduction,**hw,**timing)
                        row['record_id']=digest(row)
                        with (args.output/'trials.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                        progress(rt,event='development_complete',group=entry['block_id'],arm=arm,repeat=rep,mean_service_ns=timing['queue_metrics']['summary_ns']['service']['mean'])
                    rt.barrier()
            del hidden,reference;gc.collect();rt.torch.cuda.empty_cache()
    finally:rt.close()

if __name__=='__main__':main()

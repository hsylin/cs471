"""Reused-input attribution: exact physical replay is a diagnostic, not a method."""
import argparse
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import numpy as np
from followup.router_tapes import load_router_bank,file_sha256
from resident_dispatch.online import cpu_greedy_assignment
from resident_fastpath.fused import cpu_reference
from resident_fastpath.native import StaticResidentLayer
from resident_residual.alignment import trim_native_alignment,verify_adversarial_alignment
from resident_residual.common import source_identity,PREVIOUS_CLOSURE
from stock_comparison.common import HOST,GPU_UUIDS
from stock_comparison.data import digest
from stock_comparison.inputs import build_indices,array_digest
from stock_comparison.native import StockRuntime,StockLayer
from stock_comparison.runner import check_hardware,gather,progress,fp32_reference
from stock_comparison.timing import serve

VARIANTS=('stockoff','stockoff_trim','previous','previous_trim','replay_checked','replay_checked_trim','replay_unchecked')

class ReplayMapper:
    def __init__(self,original,physical,checked):
        self.original,self.checked=original,checked
        self.torch,self.device=original.torch,original.device
        self.outputs={t:[self.torch.from_numpy(a.copy()).to(self.device,t) for a in physical]
                      for t in (self.torch.int32,self.torch.int64)}
        self.calls=self.cursor=0
        self.last_physical_ids=None
    def verify_metadata(self,*args):return self.original.verify_metadata(*args)
    def verify_context(self):return self.original.verify_context()
    def __call__(self,logical_ids,real_tokens=None):
        torch=self.torch
        if self.checked:
            if (tuple(logical_ids.shape)!=(128,8) or logical_ids.dtype not in (torch.int32,torch.int64)
                    or not logical_ids.is_contiguous() or logical_ids.device!=self.device):
                raise ValueError('Replay shape/layout/device/type changed')
            if real_tokens is not None and (real_tokens.numel()!=1 or real_tokens.dtype!=torch.int32 or real_tokens.device!=self.device):
                raise ValueError('real_tokens must be int32 scalar')
        output=self.outputs[logical_ids.dtype][self.cursor%len(self.outputs[logical_ids.dtype])]
        self.cursor+=1;self.calls+=1;self.last_physical_ids=output
        return output

@contextmanager
def replay(layer,mapper):
    original=layer.mapper
    if mapper is not None:layer.mapper=mapper
    try:yield
    finally:layer.mapper=original

def reset(layer):
    layer.reset_trial()
    if isinstance(layer.mapper,ReplayMapper):layer.mapper.cursor=0

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('model-path','weights-dir','banks-dir','capture-index','plan','output'):p.add_argument('--'+k,type=Path,required=True)
    args=p.parse_args();plan=json.loads(args.plan.read_text())
    if plan['source_sha256']!=source_identity(diagnostic=True)[0]:raise ValueError('Diagnostic source changed')
    if plan['capture_index_sha256']!=file_sha256(args.capture_index):raise ValueError('Diagnostic banks changed')
    if plan['variants']!=list(VARIANTS) or plan.get('schema')!='resident_residual_diagnostic_plan_v1':raise ValueError('Variant freeze changed')
    if (plan.get('content_sha256')!=digest({k:v for k,v in plan.items() if k!='content_sha256'})
            or plan['jobs']!=512 or plan['unique_inputs']!=64 or plan['measured_executions']!=56 or plan['seed']!=471703):raise ValueError('Unsealed diagnostic plan')
    expected_groups={'fix-'+d+'-pilot-0'+str(i) for d in ('gsm8k','wildchat') for i in (1,2)}
    if len(plan['groups'])!=4 or {g['block_id'] for g in plan['groups']}!=expected_groups:raise ValueError('Changed pilot groups')
    for group in plan['groups']:
        orders=group['orders']
        if len(orders)!=2 or sorted(orders[0])!=sorted(VARIANTS) or orders[1]!=orders[0][::-1]:raise ValueError('Unbalanced diagnostic orders')
    from skewbench.native import Weights
    rt=StockRuntime(args.model_path)
    try:
        if rt.rank==0:args.output.mkdir(parents=True,exist_ok=False)
        rt.barrier();hw=check_hardware(rt,{'hardware':dict(hostname=HOST,gpu_uuids=GPU_UUIDS,allocation_id='4872248')})
        if rt.torch.backends.cuda.matmul.allow_tf32 or rt.torch.get_float32_matmul_precision()!='highest':raise AssertionError('FP32 precision')
        weights=Weights(args.weights_dir);off=StockLayer(rt,weights,'stockoff');previous=StaticResidentLayer(rt,weights,selector='fused')
        from resident_fastpath.fused_validation import verify_adversarial_fused
        adversarial=verify_adversarial_fused(rt,previous.mapper)
        if rt.rank==0:(args.output/'fused-adversarial.json').write_text(json.dumps(adversarial,indent=2)+'\n')
        alignment_adversarial=verify_adversarial_alignment(rt,off,previous)
        if rt.rank==0:(args.output/'alignment-adversarial.json').write_text(json.dumps(alignment_adversarial,indent=2)+'\n')
        entries={e['block_id']:e for e in json.loads(args.capture_index.read_text())}
        for group in plan['groups']:
            entry=entries[group['block_id']]
            bank=load_router_bank(args.banks_dir/Path(entry['bank_file']).name)
            if bank.sha256!=entry['capture_sha256']:raise ValueError('Changed pilot bank')
            indices,meta=build_indices(len(bank.hidden_states_bf16),entry['block_id'],seed=471703,jobs=64);indices=indices[16:]
            hidden=rt.torch.empty((64,128,2048),device=rt.device,dtype=rt.torch.bfloat16)
            for i,idx in enumerate(indices):hidden[i].copy_(rt.torch.from_numpy(np.asarray(bank.hidden_states_bf16[idx],dtype=np.uint16).copy()).view(rt.torch.bfloat16))
            rt.sync();ids_all=[];combining_all=[];expected=[];reference=[];fp32_max=0.
            positions=rt.torch.linspace(0,127,16,device=rt.device).long()
            for i,h in enumerate(hidden):
                ids,combining=off.routing(h);ids_np=ids.cpu().numpy()
                oracle=cpu_greedy_assignment(ids_np,previous.initial_map)
                scalar=cpu_reference(ids_np,previous.initial_map)
                if not np.array_equal(oracle.physical_ids,scalar['physical_ids']):raise AssertionError('Independent assignment oracles differ')
                ids_all.append(ids_np);combining_all.append(combining.float().cpu().numpy());expected.append(oracle.physical_ids)
                reference.append(off.forward(h,observe=False).clone())
                previous.verify_online_assignment(h)
                if i in (0,21,42,63):
                    dense=fp32_reference(rt,weights,h[positions],ids[positions],combining[positions])
                    rt.torch.testing.assert_close(reference[-1][positions].float(),dense,atol=.02,rtol=.03)
                    fp32_max=max(fp32_max,float((reference[-1][positions].float()-dense).abs().max()))
            signatures=(array_digest(np.stack(ids_all)),array_digest(np.stack(combining_all)),array_digest(np.stack(expected)))
            if len(set(gather(rt,signatures)))!=1:raise AssertionError('Rank assignments differ')
            mappers={True:ReplayMapper(previous.mapper,expected,True),False:ReplayMapper(previous.mapper,expected,False)}
            probes={}
            for variant in VARIANTS:
                layer=off if variant.startswith('stockoff') else previous
                mapper=mappers[variant!='replay_unchecked'] if variant.startswith('replay') else None
                with replay(layer,mapper):
                    reset(layer)
                    with trim_native_alignment(layer,enabled=variant.endswith('_trim'),probe=True,policy='one_copy_home') as guard:
                        for i,h in enumerate(hidden):
                            actual=layer.forward(h,observe=False)
                            rt.torch.testing.assert_close(actual.float(),reference[i].float(),atol=.02,rtol=.03)
                            if mapper is not None and not np.array_equal(mapper.last_physical_ids.cpu().numpy(),expected[i]):raise AssertionError('Replay changed exact assignment')
                    probes[variant]=guard.verify()
                    reset(layer)
            reduction=gather(rt,off.attest_output_reduction(hidden[0]))
            identity=dict(group=entry['block_id'],dataset=entry['dataset'],bank_sha256=bank.sha256,
                indices_sha256=array_digest(indices),logical_routes_sha256=signatures[0],combining_weights_sha256=signatures[1],exact_physical_assignment_sha256=signatures[2],
                unique_batches=64,jobs=512,sequence='64 reused pilot batches cycled8 times; not fresh generalization evidence',
                source_sha256=plan['source_sha256'],previous_closure_sha256=PREVIOUS_CLOSURE,plan_sha256=file_sha256(args.plan),
                replay_scope='Known exact physical IDs: actual gate/topk and CPU lookup charged; selector and remap GPU invocation excluded; not deployable',
                fp32_max_abs_error=max(gather(rt,fp32_max)),fp32_jobs=[0,21,42,63],fp32_tokens_per_job=16,atol=.02,rtol=.03)
            if rt.rank==0:
                np.savez_compressed(args.output/(entry['block_id']+'-inputs.npz'),indices=indices,logical_ids=np.stack(ids_all),combining_weights=np.stack(combining_all),physical_ids=np.stack(expected))
            gathered_probes=gather(rt,probes)
            if rt.rank==0:(args.output/(entry['block_id']+'-geometry.json')).write_text(json.dumps(gathered_probes)+'\n')
            for rep,order in enumerate(group['orders']):
                for variant in order:
                    layer=off if variant.startswith('stockoff') else previous
                    mapper=mappers[variant!='replay_unchecked'] if variant.startswith('replay') else None
                    with replay(layer,mapper):
                        reset(layer)
                        with trim_native_alignment(layer,enabled=variant.endswith('_trim'),policy='one_copy_home'):
                            with layer.measured():
                                for i,h in enumerate(hidden):layer.forward(h);layer.step(i);rt.sync()
                        reset(layer);gc.collect();rt.sync();rt.barrier()
                        with trim_native_alignment(layer,enabled=variant.endswith('_trim'),policy='one_copy_home') as guard:
                            with layer.measured():
                                start=layer.snapshot();timing=serve(rt,layer,hidden,[i%64 for i in range(512)],[0]*512,1);end=layer.snapshot()
                        audit=layer.audit()
                        if mapper is not None:mapper.cursor=0
                        with trim_native_alignment(layer,enabled=variant.endswith('_trim'),policy='one_copy_home'):
                            for i,h in enumerate(hidden):rt.torch.testing.assert_close(layer.forward(h,observe=False).float(),reference[i].float(),atol=.02,rtol=.03)
                        states=gather(rt,dict(start=start,end=end,audit=audit,alignment=guard.snapshot(),postflight_full_inputs=64))
                    if rt.rank==0:
                        row=dict(schema='resident_residual_diagnostic_v1',**identity,variant=variant,repeat=rep,arm_order=order,per_rank_state=states,output_reduction_probes=reduction,**hw,**timing)
                        row['record_id']=digest(row)
                        with (args.output/'trials.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                        progress(rt,event='diagnostic_complete',group=entry['block_id'],variant=variant,repeat=rep,mean_service_ns=timing['queue_metrics']['summary_ns']['service']['mean'])
                    rt.barrier()
            del hidden,reference,mappers;gc.collect();rt.torch.cuda.empty_cache()
    finally:rt.close()

if __name__=='__main__':main()

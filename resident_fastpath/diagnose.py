"""Identical-assignment factorial diagnostic on previously measured pilot inputs.

Offline assignment is explicitly a diagnostic, never an online comparator.
It excludes histogram/home selection, retaining actual gate/top-k and one
physical mapping/record kernel. This measures available optimization headroom.
"""
import argparse
from contextlib import contextmanager
import gc
import json
from pathlib import Path

import numpy as np

from followup.router_tapes import load_router_bank
from resident_dispatch.online import cpu_greedy_assignment
from resident_fastpath.native import build_layers
from stock_comparison.common import HOST,GPU_UUIDS
from stock_comparison.data import digest
from stock_comparison.inputs import build_indices,array_digest
from stock_comparison.native import StockRuntime
from stock_comparison.runner import check_hardware,gather,progress
from stock_comparison.timing import serve

VARIANTS=('stockoff','current_online','current_offline','static_online','static_offline')


class OfflineMapper:
    def __init__(self, original, homes):
        self.original = original
        self.homes = homes
        self.calls = 0
        self.cursor = 0

    def verify_metadata(self,*args):
        return self.original.verify_metadata(*args)

    def __call__(self, ids, load, logical_map, replica_count, record, real):
        from resident_dispatch.online_gpu import map_and_record
        out=self.original.torch.empty_like(ids)
        home=self.homes[self.cursor % len(self.homes)]
        map_and_record[(4,)](ids,home,out,load,record,real,N=1024,TOPK=8,CHUNK=256,num_warps=4)
        self.cursor+=1;self.calls+=1
        return out


@contextmanager
def offline(layer,homes,enabled):
    original=layer.mapper
    if enabled:layer.mapper=OfflineMapper(original,homes)
    try:yield
    finally:layer.mapper=original


def reset(layer):
    layer.reset_trial()
    if isinstance(layer.mapper,OfflineMapper):layer.mapper.cursor=0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('model-path','weights-dir','banks-dir','capture-index','output'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--selector',choices=('three_kernel','fused'),default='three_kernel')
    p.add_argument('--repetitions',type=int,default=2)
    args=p.parse_args()
    if args.selector=='fused':
        variants=('stockoff','current_online','static_online')
    else:variants=VARIANTS
    if args.output.exists():raise FileExistsError(args.output)
    from skewbench.native import Weights
    rt=StockRuntime(args.model_path)
    try:
        if rt.rank==0:args.output.mkdir(parents=True)
        rt.barrier()
        hw=check_hardware(rt,{'hardware':dict(hostname=HOST,gpu_uuids=GPU_UUIDS,allocation_id='4872248')})
        weights=Weights(args.weights_dir);layers=build_layers(rt,weights,selector=args.selector)
        if args.selector=='fused':
            from resident_fastpath.fused_validation import verify_adversarial_fused
            evidence=verify_adversarial_fused(rt,layers['fix'].mapper)
            if rt.rank==0:(args.output/'fused-adversarial.json').write_text(json.dumps(evidence,indent=2)+'\n')
        banks=json.loads(args.capture_index.read_text())
        for gi,entry in enumerate(banks):
            bank=load_router_bank(args.banks_dir/Path(entry['bank_file']).name)
            if bank.sha256!=entry['capture_sha256']:raise ValueError('Changed diagnostic pilot bank')
            indices,meta=build_indices(len(bank.hidden_states_bf16),entry['block_id'],jobs=64)
            indices=indices[16:]
            hidden=rt.torch.empty((64,128,2048),device=rt.device,dtype=rt.torch.bfloat16)
            for i,idx in enumerate(indices):
                bits=np.asarray(bank.hidden_states_bf16[idx],dtype=np.uint16).copy()
                hidden[i].copy_(rt.torch.from_numpy(bits).view(rt.torch.bfloat16))
            rt.sync()
            homes=[];ids_all=[];expected=[];reference=[]
            for h in hidden:
                ids,combining=layers['stockoff'].routing(h)
                ids_np=ids.cpu().numpy()
                oracle=cpu_greedy_assignment(ids_np,layers['current'].initial_map)
                homes.append(rt.torch.from_numpy(oracle.homes.copy()).to(rt.device,rt.torch.int32))
                ids_all.append(ids_np);expected.append(oracle.physical_ids)
                reference.append(layers['stockoff'].forward(h,observe=False).clone())
                layers['current'].verify_online_assignment(h)
                layers['fix'].verify_online_assignment(h)
            for variant in variants:
                layer=layers['stockoff' if variant=='stockoff' else 'current' if variant.startswith('current') else 'fix']
                with offline(layer,homes,variant.endswith('_offline')):
                    reset(layer)
                    for i,h in enumerate(hidden):
                        actual=layer.forward(h,observe=False)
                        rt.torch.testing.assert_close(actual.float(),reference[i].float(),atol=.02,rtol=.03)
                    reset(layer)
            # Probe outside timing, restored immediately, as in the stock study.
            probe=gather(rt,layers['stockoff'].attest_output_reduction(hidden[0]))
            if len({digest(x['observed_successful_backends']) for x in probe})!=1:raise AssertionError('Reduction differs')
            identity=dict(group=entry['block_id'],dataset=entry['dataset'],bank_sha256=bank.sha256,
                indices_sha256=array_digest(indices),logical_routes_sha256=array_digest(np.stack(ids_all)),
                exact_physical_assignment_sha256=array_digest(np.stack(expected)),
                unique_batches=64,jobs=512,sequence='64 distinct pilot batches cycled8 times; not fresh final evidence',
                selector=args.selector,offline_scope='Exact prior-known homes; real gate/topk and map/record kernel charged; selection cost excluded')
            if rt.rank==0:
                np.savez_compressed(args.output/(entry['block_id']+'-inputs.npz'),indices=indices,
                    logical_ids=np.stack(ids_all),physical_ids=np.stack(expected))
            for rep in range(args.repetitions):
                shift=gi%len(variants)
                order=list(variants[shift:]+variants[:shift])
                if rep%2:order.reverse()
                for variant in order:
                    layer=layers['stockoff' if variant=='stockoff' else 'current' if variant.startswith('current') else 'fix']
                    with offline(layer,homes,variant.endswith('_offline')):
                        reset(layer)
                        with layer.measured():
                            for i,h in enumerate(hidden):layer.forward(h);layer.step(i);rt.sync()
                        reset(layer);gc.collect();rt.sync();rt.barrier()
                        with layer.measured():
                            start=layer.snapshot()
                            timing=serve(rt,layer,hidden,[i%64 for i in range(512)],[0]*512,1)
                            end=layer.snapshot()
                        audit=layer.audit()
                        # Offline cursor is restored before postflight exact-input checks.
                        if isinstance(layer.mapper,OfflineMapper):layer.mapper.cursor=0
                        for i,h in enumerate(hidden):
                            actual=layer.forward(h,observe=False)
                            rt.torch.testing.assert_close(actual.float(),reference[i].float(),atol=.02,rtol=.03)
                        states=gather(rt,dict(start=start,end=end,audit=audit))
                    if rt.rank==0:
                        row=dict(schema='resident_fix_diagnostic_v1',**identity,variant=variant,repeat=rep,
                            arm_order=order,per_rank_state=states,output_reduction_probes=probe,**hw,**timing)
                        row['record_id']=digest(row)
                        with (args.output/'trials.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                        progress(rt,event='diagnostic_complete',group=entry['block_id'],variant=variant,repeat=rep,
                            mean_service_ns=timing['queue_metrics']['summary_ns']['service']['mean'])
                    rt.barrier()
            del hidden,reference,homes;gc.collect();rt.torch.cuda.empty_cache()
    finally:rt.close()


if __name__=='__main__':main()

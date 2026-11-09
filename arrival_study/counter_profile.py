"""Bounded development-only Nsight Compute collector; never latency evidence.

Run the lightweight --probe under NCU first. The actual two-rank collection uses
the immutable resident validation manifest and one old input (raw16) per batch.
Only the native fused_experts call is inside cudaProfilerStart/Stop; its physical
mapping and output NCCL allreduce execute outside that range.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


METRICS = (
    'gpu__time_duration.sum',
    'dram__bytes_read.sum',
    'dram__bytes_write.sum',
    'lts__t_bytes.sum',
    'sm__throughput.avg.pct_of_peak_sustained_elapsed',
    'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed',
    'sm__warps_active.avg.pct_of_peak_sustained_active',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity():
    root=Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root)):digest(p) for p in
            (Path(__file__),root/'scripts/profile-instructor-counters-kuma.sh')}


def probe(output):
    """One four-megabyte fill; no vLLM, model load or distributed initialization."""
    import torch
    torch.cuda.set_device(0)
    values=torch.empty(1024*1024,device='cuda',dtype=torch.float32)
    values.fill_(1)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    try:
        values.fill_(2)
        torch.cuda.synchronize()
    finally:
        torch.cuda.profiler.stop()
    if float(values[0].item()) != 2:
        raise AssertionError('Probe fill failed')
    output.write_text(json.dumps(dict(probe_completed=True,probe_kernel='float32 fill',
        logical_bytes=4*1024*1024,torch_version=torch.__version__,gpu=torch.cuda.get_device_name(0),
        scope='Permission/compatibility probe only; not performance evidence'),indent=2)+'\n')


@contextmanager
def capture_experts_only(layer, label, before_capture):
    """Restore the instance callable even on exceptions; no collective is inside."""
    original=layer.fused_experts
    torch=layer.torch
    calls=[]

    def wrapped(*args,**kwargs):
        before_capture(args,kwargs)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(label)
        started=False
        try:
            torch.cuda.profiler.start();started=True
            result=original(*args,**kwargs)
            torch.cuda.synchronize()
            calls.append(label)
            return result
        finally:
            if started:torch.cuda.profiler.stop()
            torch.cuda.nvtx.range_pop()

    layer.fused_experts=wrapped
    try:
        yield calls
    finally:
        layer.fused_experts=original


def collect(args):
    import gc
    import numpy as np

    from optimization.common import agree_config,make_layer,manifest_hash
    from optimization.kernel_config import kernel_override
    from resident_dispatch.common import load_manifest,numerical_check,runtime_metadata
    from resident_dispatch.runner import make_online,prepare
    from skewbench.native import Runtime,Weights
    from skewbench.runner import snapshot_weight_hash

    manifest=load_manifest(args.manifest)
    if manifest['phase']!='validation' or manifest['split']!='pilot':
        raise ValueError('Counter collection requires the immutable old-input validation manifest')
    if 'opt-pilot-01' not in manifest['matrix']['block_ids']:
        raise ValueError('Missing fixed development block')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    rt=Runtime()
    rows=[]
    try:
        weights=Weights(args.weights_dir)
        runtime_metadata(rt,weights,manifest)
        preparation_args=SimpleNamespace(banks_dir=args.banks_dir,output=args.output_dir/'preparation.jsonl')
        with kernel_override(None):
            for batch in (128,1024):
                group=dict(block_id='opt-pilot-01',regime='domain0',batch_tokens=batch,repetition=0)
                cell=prepare(rt,weights,manifest,preparation_args,group)
                for arm in ('native','online'):
                    layer=make_layer(rt,weights,manifest,'SKIP')
                    layer.assert_maps_and_weights()
                    mapper=make_online(rt,layer,batch) if arm=='online' else None
                    if mapper is not None:layer.mapping_kernel=mapper
                    raw=16
                    inputs=cell['tensors']
                    pre=numerical_check(rt,layer,inputs,cell['references'],[raw],lambda _:None)
                    # Identical old inputs and warmup count; the assignment arm itself differs.
                    for warm in range(16):
                        layer.forward(*(t[warm] for t in inputs),observe=False)
                    rt.sync();rt.barrier();rt.sync()
                    evidence={}

                    def inspect_physical(expert_args,expert_kwargs):
                        physical=expert_args[4].cpu().numpy()
                        counts=np.bincount(physical.ravel(),minlength=160)
                        if not np.array_equal(counts,cell['counts'][arm][raw]):
                            raise AssertionError('Captured expert input differs from verified physical counts')
                        if not np.array_equal(cell['mapping'][physical],cell['logical'][raw]):
                            raise AssertionError('Captured physical input changed logical routing')
                        evidence.update(physical_ids_sha256=hashlib.sha256(np.asarray(physical,dtype=np.int64).tobytes()).hexdigest(),
                            physical_counts=counts.tolist(),
                            physical_assignment_metrics=cell['diagnostics'][arm][raw],
                            global_num_experts=expert_kwargs['global_num_experts'])

                    label=f'instructor/B{batch}/{arm}/raw16/rank{rt.rank}'
                    with capture_experts_only(layer,label,inspect_physical) as calls:
                        output=layer.forward(*(t[raw] for t in inputs),observe=False)
                    # The output allreduce has executed after cudaProfilerStop.
                    rt.sync()
                    if len(calls)!=1:
                        raise AssertionError('Exactly one local expert call must be captured')
                    rt.torch.testing.assert_close(output.float(),cell['references'][raw]['native'].float(),atol=.02,rtol=.03)
                    agree_config(rt,evidence)
                    post=numerical_check(rt,layer,inputs,cell['references'],[raw],lambda _:None)
                    layer.assert_maps_and_weights()
                    current_hash,_=snapshot_weight_hash(rt,(layer.w13,layer.w2))
                    if current_hash!=cell['source_weights_sha256'] or layer.counter.sent_bytes:
                        raise AssertionError('Counter capture altered resident weights')
                    arrays={
                        'hidden_bf16_bits':inputs[0][raw].view(rt.torch.uint16).cpu().numpy(),
                        'logical_ids':inputs[1][raw].cpu().numpy(),
                        'combining_weights':inputs[2][raw].cpu().numpy(),
                    }
                    row={**rt.metadata,**group,'arm':arm,'rank':rt.rank,'raw_index':raw,
                        'counter_profile_schema_version':1,'phase':'development_counter_diagnostic',
                        'evidence_eligible':False,'measurement_kind':'ncu_kernel_counters',
                        'config_hash':manifest_hash(manifest),'collector_source_sha256':source_identity(),
                        'profile_label':label,'expected_selected_gemm_launches':2,
                        'warmup_inputs':list(range(16)),'capture_scope':'fused_experts only; physical mapping and output NCCL allreduce excluded',
                        'replay_mode':'kernel','cache_control':'none','clock_control':'none',
                        'metrics_requested':list(METRICS),'normal_native_counter_recording':False,
                        'capture_sha256':cell['capture_sha256'],'replay_sha256':cell['replay_sha256'],
                        'source_prompt_ids':cell['source_prompt_ids'],'source_map':cell['mapping'].tolist(),
                        'source_weights_sha256':cell['source_weights_sha256'],
                        'resolved_kernel_config':cell['resolved_kernel_config'],
                        'input_array_sha256':{k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in arrays.items()},
                        'preflight_correctness':pre,**post,**evidence,'migration_bytes':0,
                        'limit':'Counter replay is not primary latency and does not measure selector/collective overhead.'}
                    rows.append(row)
                    print(json.dumps(dict(event='counter_profile_case_complete',rank=rt.rank,batch_tokens=batch,arm=arm)),flush=True)
                    layer.close();del layer,mapper;gc.collect();rt.sync();rt.barrier()
                del cell;gc.collect();rt.torch.cuda.empty_cache()
        path=args.output_dir/f'counter-metadata-rank{rt.rank}.json'
        if path.exists():raise FileExistsError(path)
        path.write_text(json.dumps(dict(completed=True,rank=rt.rank,expected_selected_gemm_launches=8,
            source_identity=source_identity(),profiles=rows),indent=2,allow_nan=False)+'\n')
        rt.barrier()
    finally:
        rt.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe',action='store_true')
    parser.add_argument('--probe-output',type=Path)
    for name in ('manifest','banks-dir','weights-dir','output-dir'):
        parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    if args.probe:
        if args.probe_output is None or args.probe_output.exists():
            parser.error('Probe requires a fresh --probe-output')
        probe(args.probe_output)
    else:
        if any(getattr(args,k) is None for k in ('manifest','banks_dir','weights_dir','output_dir')):
            parser.error('Collection requires manifest, banks-dir, weights-dir and output-dir')
        collect(args)


if __name__=='__main__':main()

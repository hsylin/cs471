"""Same-eight-GPU native AgRs primitive replay using observed shape metadata.

Synthetic rank-labelled payloads, not model inference. No serving arm bypasses
communication. Isolates collective latency without experts or rank compute skew;
does not estimate end-to-end critical-path time by subtracting kernel sums.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import time


def select_shapes(root):
    rows={}
    for p in Path(root).rglob('cost-events-rank0.jsonl'):
        for line in p.read_text().splitlines():
            r=json.loads(line)
            if r['kind']!='activation_collective_shape' or r['extra_shapes']:continue
            sizes=tuple(r['sizes'])
            if len(sizes)!=8 or min(sizes)<=0:continue
            key=(sizes,r['hidden_dtype'],r['weights_dtype'],r['ids_dtype'])
            rows[key]={k:r[k] for k in ('sizes','hidden_dtype','weights_dtype','ids_dtype')}
    assert rows,'No observed collective shape metadata'
    ordered=sorted(rows.values(),key=lambda r:(sum(r['sizes']),r['sizes'],r['ids_dtype']))
    selected=[ordered[i] for i in sorted({0,len(ordered)//2,len(ordered)-1})]
    return {'status':'selected_without_timing','selection':'Smallest, median-index and largest observed positive-row size/dtype combinations; source includes graph construction, not execution frequency.',
            'shapes':selected,'source_files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(root).rglob('cost-events-rank0.jsonl')}}


def worker(spec,output,master):
    import torch
    import torch.distributed as dist
    import vllm
    from vllm.distributed.parallel_state import init_distributed_environment,GroupCoordinator
    rank=int(os.environ['SLURM_PROCID']);assert int(os.environ['SLURM_NTASKS'])==8
    assert os.environ['SLURM_JOB_ID']=='4882284'
    assert socket.gethostname().split('.')[0] in ('kh074','kh077')
    for name,digest in spec['native_source_sha256'].items():
        assert hashlib.sha256((Path(vllm.__file__).parent/name).read_bytes()).hexdigest()==digest,name
    # Match the serving workers' four-visible-device namespace. On this pinned
    # NCCL stack, one-visible-device tasks fail P2P/CUMEM peer setup with CUDA101.
    local_rank=int(os.environ['SLURM_LOCALID'])
    assert torch.cuda.device_count()==4 and local_rank==rank%4
    torch.cuda.set_device(local_rank)
    init_distributed_environment(world_size=8,rank=rank,local_rank=local_rank,distributed_init_method=master,backend='nccl')
    group=GroupCoordinator([list(range(8))],local_rank,'nccl',True,group_name='cost_replay',use_all2all=False)
    records=[]
    for shape in spec['shapes']:
        sizes=shape['sizes'];n=sizes[rank];total=sum(sizes)
        dtype=lambda name:getattr(torch,name.removeprefix('torch.'))
        inputs=[torch.full((n,2048),rank+1,device='cuda',dtype=dtype(shape['hidden_dtype'])),
                torch.full((n,8),rank+1,device='cuda',dtype=dtype(shape['weights_dtype'])),
                torch.full((n,8),rank+1,device='cuda',dtype=dtype(shape['ids_dtype']))]
        partial=torch.full((total,2048),rank+1,device='cuda',dtype=dtype(shape['hidden_dtype']))
        def gather():return group.all_gatherv(inputs,dim=0,sizes=sizes)
        def scatter():return group.reduce_scatterv(partial,dim=0,sizes=sizes)
        gathered=gather();combined=scatter();torch.cuda.synchronize()
        for tensor in gathered:
            start=0
            for peer,count in enumerate(sizes):
                assert bool(torch.all(tensor[start:start+count]==peer+1));start+=count
        assert bool(torch.all(combined==36))
        timings={}
        for name,fn in (('dispatch_3_tensors',gather),('combine_1_tensor',scatter)):
            for _ in range(5):fn()
            torch.cuda.synchronize();dist.barrier(group=group.cpu_group)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):captured=fn()
            graph.replay();torch.cuda.synchronize()
            if name=='dispatch_3_tensors':
                for tensor in captured:
                    offset=0
                    for peer,count in enumerate(sizes):
                        assert bool(torch.all(tensor[offset:offset+count]==peer+1))
                        offset+=count
            else:
                assert bool(torch.all(captured==36))
            for mode,call in (('eager',fn),('cuda_graph',graph.replay)):
                samples=[]
                for repeat in range(3):
                    dist.barrier(group=group.cpu_group);torch.cuda.synchronize()
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    wall=time.perf_counter_ns();start.record()
                    for _ in range(50):call()
                    end.record();end.synchronize()
                    samples.append({'event_us_per_call':start.elapsed_time(end)*1000/50,'wall_us_per_call':(time.perf_counter_ns()-wall)/1000/50})
                timings[name+'/'+mode]=samples
        records.append({'shape':shape,'rank':rank,'correct_rank_payloads_and_sum':True,'timings':timings,
                        'per_rank_local_dispatch_input_bytes':sum(t.numel()*t.element_size() for t in inputs),
                        'per_rank_combine_input_bytes':partial.numel()*partial.element_size()})
    output.mkdir(parents=True,exist_ok=True)
    (output/f'rank{rank}.json').write_text(json.dumps({'status':'passed','rank':rank,'local_rank':local_rank,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'hostname':socket.gethostname(),'job':os.environ['SLURM_JOB_ID'],'step':os.environ['SLURM_STEP_ID'],'gpu':torch.cuda.get_device_name(),'records':records,
      'native_source_sha256':spec['native_source_sha256'],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'backend':'Pinned vLLM GroupCoordinator.all_gatherv/reduce_scatterv, same primitives called by AgRs; NCCL device communicator',
      'scope':'Synthetic payload, observed sizes; component-only, excludes experts, compute imbalance and serving scheduler. Both eager and graph replay charged; no bandwidth-saturation or serving-speedup claim.'},indent=2)+'\n')
    dist.barrier(group=group.cpu_group);dist.destroy_process_group()


def summarize(root,spec,source_sha256):
    """Pair repeats across all ranks; retain slowest-rank latency, never fastest."""
    import math
    import statistics
    ranks=[json.loads((Path(root)/f'rank{rank}.json').read_text()) for rank in range(8)]
    assert len({r['step'] for r in ranks})==1
    assert {r['hostname'].split('.')[0] for r in ranks}=={'kh074','kh077'}
    for rank,r in enumerate(ranks):
        assert r['status']=='passed' and r['rank']==rank and r['job']=='4882284'
        assert 'H100' in r['gpu']
        assert r['source_sha256']==source_sha256
        assert r['native_source_sha256']==spec['native_source_sha256']
        assert len(r['records'])==len(spec['shapes'])
        for record,shape in zip(r['records'],spec['shapes']):
            assert record['shape']==shape and record['rank']==rank
            assert record['correct_rank_payloads_and_sum']
    shapes=[]
    for i,shape in enumerate(spec['shapes']):
        timings={}
        expected={op+'/'+mode for op in ('dispatch_3_tensors','combine_1_tensor') for mode in ('eager','cuda_graph')}
        assert all(set(r['records'][i]['timings'])==expected for r in ranks)
        for key in sorted(expected):
            samples=[r['records'][i]['timings'][key] for r in ranks]
            assert all(len(s)==3 for s in samples)
            fields={}
            for field in ('event_us_per_call','wall_us_per_call'):
                assert all(math.isfinite(x[field]) and x[field]>0 for s in samples for x in s)
                maxima=[max(s[rep][field] for s in samples) for rep in range(3)]
                fields[field]={'all_rank_samples':[[x[field] for x in s] for s in samples],
                               'max_rank_each_repeat':maxima,'median_max_rank':statistics.median(maxima)}
            timings[key]=fields
        shapes.append({'shape':shape,'timings':timings})
    return {'status':'passed','ranks':8,'source_sha256':source_sha256,'shapes':shapes,
            'scope':'Synthetic collective replay, three technical repeats. Maximum across ranks for each paired repeat; no independent-run uncertainty. Peer waiting is still included; not a wire-time lower bound or an end-to-end speedup.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('select','run'));p.add_argument('--root',type=Path);p.add_argument('--spec',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--master');a=p.parse_args()
    if a.mode=='select':a.output.write_text(json.dumps(select_shapes(a.root),indent=2)+'\n')
    else:worker(json.loads(a.spec.read_text()),a.output,a.master)
if __name__=='__main__':main()

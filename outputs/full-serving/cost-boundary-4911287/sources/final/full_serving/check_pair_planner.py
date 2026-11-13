"""Correctness and GPU-graph cost of equivalent old/new resident planners."""
import argparse
import json
from pathlib import Path
import socket
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import numpy as np
    import torch
    from full_serving.resident_policy import ResidentMapper, balanced_map, cpu_reference
    from full_serving.resident_pair import PairMapper
    torch.manual_seed(4711111)
    device=torch.device('cuda',0); cases=[];timings=[]
    for rows in (1,8,16,32,64,128,512,513,2048,8192,65536):
        rng=np.random.default_rng(471+rows)
        for dtype in (torch.int32,torch.int64):
            ids=torch.empty((rows,8),dtype=dtype,device=device)
            mappers={'original':ResidentMapper(device,balanced_map()),'pair_parallel':PairMapper(device,balanced_map())}
            graphs={};outputs={}
            for name,mapper in mappers.items():
                ids.fill_(-1)
                for _ in range(3):mapper(ids)
                torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g): outputs[name]=mapper(ids)
                graphs[name]=g
            for variant in ('uniform','shared','padding','skew','ties'):
                values=rng.integers(0,128,size=(rows,8),dtype=np.int64)
                if variant=='shared':values=(values//16)*16+values%4
                if variant=='padding':values[::2]=-1
                if variant=='skew':values[:max(1,rows*3//4)]=np.arange(8)
                if variant=='ties':values=np.tile(np.arange(8),(rows,1))+32
                ids.copy_(torch.tensor(values,dtype=dtype,device=device));expected=cpu_reference(values)['physical_ids']
                for name,g in graphs.items():
                    g.replay();assert np.array_equal(outputs[name].cpu().numpy(),expected),(rows,dtype,variant,name)
                cases.append({'rows':rows,'dtype':str(dtype),'variant':variant,'exact_cpu_and_original':True})
                if dtype==torch.int32 and variant in ('uniform','skew'):
                    # Repeated device launches in one captured graph avoid host submission gaps.
                    batches={}
                    for name,mapper in mappers.items():
                        batch=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(batch):
                            for _ in range(64):mapper(ids)
                        batches[name]=batch
                    samples={name:[] for name in mappers}
                    for repeat in range(6):
                        order=list(mappers) if repeat%2==0 else list(reversed(mappers))
                        for name in order:
                            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                            start.record();batches[name].replay();end.record();end.synchronize()
                            samples[name].append(start.elapsed_time(end)*1000/64)
                    timings.append({'rows':rows,'variant':variant,'us_per_call_samples':samples})
    result={'status':'passed','host':socket.gethostname(),'gpu':torch.cuda.get_device_name(),'cases':cases,'timings':timings,'scope':'Synthetic GPU mapper costs with actual H100; not full-model latency or authentic routing. Six technical repetitions, no inferential claim.'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'passed','cases':len(cases)}))
if __name__=='__main__':main()

"""GPU exact-history oracle and charged host/device microbenchmarks.

Checks changes of placement, duplicate copies, padding, dummy steps, ring wraps,
unaligned proposal offsets and snapshot independence. Not serving performance.
"""
import argparse
import json
from pathlib import Path
import statistics
import time


def run():
    import numpy as np
    import torch
    from full_serving.logical_history_gpu import LogicalHistory
    torch.manual_seed(4711111)
    device='cuda'
    checked=0
    for rolling in (False,True):
        for window,layers,experts,physical,steps in ((7,3,11,16,45),(1000,48,128,160,7051)):
            h=LogicalHistory(device,window=window,layers=layers,experts=experts,physical=physical,rolling=rolling)
            ring=np.zeros((2*window,layers,experts+1),dtype=np.int64)
            rng=np.random.default_rng(4711111)
            mapping=np.tile(np.arange(physical)%experts,(layers,1));mapping[:,-1]=-1
            counts=rng.integers(0,100,size=(layers,physical),dtype=np.int32)
            m=torch.tensor(mapping,dtype=torch.int64,device=device);c=torch.tensor(counts,device=device)
            checkpoints={0,window-1,window,2*window-1,2*window,steps-1,749,3749,6749}
            saved=None
            for step in range(steps):
                if step in (window-1,2*window+3):
                    mapping=np.roll(mapping,1,axis=1);m.copy_(torch.tensor(mapping,device=device))
                dummy=step%13==0
                h.update(m,c,dummy=dummy)
                row=np.zeros((layers,experts+1),dtype=np.int64)
                if not dummy:
                    for layer in range(layers):np.add.at(row[layer],np.where(mapping[layer]<0,experts,mapping[layer]),counts[layer])
                ring[step%(2*window)]=row
                if step in checkpoints:
                    a,b=h.snapshot();order=(np.arange(2*window)+step+1)%(2*window)
                    expected=(ring[order[:window],:,:experts].sum(0),ring[order[window:],:,:experts].sum(0))
                    actual_a=a.cpu().numpy();actual_b=b.cpu().numpy()
                    if not np.array_equal(actual_a,expected[0]) or not np.array_equal(actual_b,expected[1]):
                        actual_ring=h.ring.cpu().numpy()
                        raise AssertionError({'rolling':rolling,'window':window,'step':step,'ring_equal':bool(np.array_equal(actual_ring,ring)),'previous_max_error':int(np.abs(actual_a-expected[0]).max()),'recent_max_error':int(np.abs(actual_b-expected[1]).max()),'example':[(int(l),int(e),int(actual_a[l,e]),int(expected[0][l,e])) for l,e in zip(*np.nonzero(actual_a!=expected[0]))][:8]})
                    assert np.array_equal(b.cpu().numpy(),expected[1]),(rolling,window,step,'recent')
                    checked+=1
                    if saved is not None:assert torch.equal(saved[0],saved[1])
                    saved=(a,a.clone())
            print(json.dumps({'rolling':rolling,'window':window,'checked':checked}),flush=True)
    # Same actual geometry and operation sequences; no map/count generation timed.
    W,L,E,P=1000,48,128,160
    m=torch.arange(P,device=device).remainder(E).expand(L,-1).contiguous().to(torch.int64)
    c=torch.randint(0,100,(L,P),device=device,dtype=torch.int32)
    def original_update(ring,step):
        row=ring[step%(2*W)];row.zero_()
        ids=m.long().masked_fill(m<0,128)
        row.scatter_add_(1,ids,c.to(torch.int64))
    def original_snapshot(ring,seen):
        order=(torch.arange(2*W,device=device)+seen)%(2*W)
        ordered=ring.index_select(0,order)
        return ordered[:W,:,:E].sum(0),ordered[W:,:,:E].sum(0)
    timings={}
    for kind in ('original','rolling','reduction'):
        ring=torch.zeros((2*W,L,E+1),device=device,dtype=torch.int64)
        history=LogicalHistory(device,rolling=kind=='rolling')
        def update(i):
            if kind=='original':
                original_update(ring,history.seen)
                history.seen+=1
            else:history.update(m,c)
        snapshot=(lambda:original_snapshot(ring,history.seen)) if kind=='original' else history.snapshot
        for i in range(40):update(i)
        snapshot();torch.cuda.synchronize()
        repeats=[]
        for repeat in range(3):
            torch.cuda.synchronize();t=time.perf_counter_ns()
            # Native interval is 3000; include one snapshot and CPU enqueue cost.
            for i in range(3000):update(i)
            snapshot();torch.cuda.synchronize()
            repeats.append((time.perf_counter_ns()-t)/1e6)
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        for i in range(100):update(i)
        end.record();end.synchronize()
        timings[kind]={'three_cycle_wall_ms':repeats,'median_cycle_wall_ms':statistics.median(repeats),'update_device_us_including_host_enqueue_gaps':start.elapsed_time(end)*10}
    return {'status':'passed','exact_snapshot_cases':checked,'timings':timings,
            'selection_rule':'Choose the correct fused variant with lower median charged cycle time; do not use final serving traces for tuning.',
            'selected':min(('rolling','reduction'),key=lambda k:timings[k]['median_cycle_wall_ms']),
            'limitations':'Synthetic counter state; component timing includes Python enqueue and GPU synchronization. Not a full-model latency prediction.',
            'torch':torch.__version__,'gpu':torch.cuda.get_device_name()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=run();a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()

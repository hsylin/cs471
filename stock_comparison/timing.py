"""FCFS external-clock layer jobs; native async commit tail charged separately."""
import time
import numpy as np
from arrival_study.runner import wait_until_ns
from arrival_study.schedules import fcfs_metrics

def serve(rt,layer,hidden,raw_indices,offsets,horizon_ns):
    torch=rt.torch;dist=torch.distributed;cpu=rt.ep_group.cpu_group
    n=len(raw_indices)
    if len(offsets)!=n:raise ValueError('Calendar/input length mismatch')
    command=torch.zeros(1,dtype=torch.int64);epoch_tensor=command.clone()
    admits=np.zeros(n,dtype=np.int64);starts=admits.copy();finishes=admits.copy()
    rank_starts=admits.copy();rank_finishes=admits.copy()
    rt.sync();dist.barrier(group=cpu)
    if rt.rank==0:epoch_tensor[0]=time.monotonic_ns()+20_000_000
    dist.broadcast(epoch_tensor,src=0,group=cpu);epoch=int(epoch_tensor[0])
    for i,raw in enumerate(raw_indices):
        if rt.rank==0:
            wait_until_ns(epoch+int(offsets[i]));admits[i]=time.monotonic_ns()
            command[0]=i;starts[i]=time.monotonic_ns()
        dist.broadcast(command,src=0,group=cpu)
        if int(command[0])!=i:raise AssertionError('Rank command order disagrees')
        rank_starts[i]=time.monotonic_ns()
        layer.forward(hidden[raw],observe=True)
        layer.step(i)
        # Output/default-stream dependencies are complete; unrelated native
        # async transfer-stream work must remain concurrent. Device-wide sync
        # belongs to setup and the separately charged normal-commit tail.
        torch.cuda.current_stream(rt.device).synchronize()
        rank_finishes[i]=time.monotonic_ns()
        dist.barrier(group=cpu)
        if rt.rank==0:finishes[i]=time.monotonic_ns()
    # The final job ACK means completed output. Remaining native migration uses
    # ordinary all-rank readiness/move-to-workspace; never drain/discard.
    tail_start=time.monotonic_ns()
    layer.complete_pending()
    rt.sync();dist.barrier(group=cpu)
    quiescent=time.monotonic_ns()
    if rt.rank==0:
        wait_until_ns(epoch+int(horizon_ns));observation_end=time.monotonic_ns()
    dist.barrier(group=cpu)
    peers=[None]*rt.world_size
    dist.all_gather_object(peers,dict(start=rank_starts.tolist(),gpu_complete=rank_finishes.tolist(),
        tail_start=tail_start,quiescent=quiescent),group=cpu)
    if rt.rank!=0:return None
    row=dict(n_jobs=n,arrival_offsets_ns=list(map(int,offsets)),horizon_ns=int(horizon_ns),
        per_job_completion_scope='current_inference_stream_after_native_step',
        epoch_monotonic_ns=epoch,observation_end_ns=observation_end,
        coordinator_admit_ns=admits.tolist(),coordinator_start_ns=starts.tolist(),
        coordinator_finish_ns=finishes.tolist(),per_rank_start_ns=[p['start'] for p in peers],
        per_rank_gpu_complete_ns=[p['gpu_complete'] for p in peers],
        per_rank_controller_tail_start_ns=[p['tail_start'] for p in peers],
        per_rank_controller_quiescent_ns=[p['quiescent'] for p in peers],
        controller_quiescent_ns=quiescent,controller_tail_elapsed_ns=quiescent-int(finishes[-1]),
        controller_quiescence_start_ns=tail_start,quiescence_call_elapsed_ns=quiescent-tail_start,
        job_completed_elapsed_ns=int(finishes[-1])-epoch,
        completed_elapsed_ns=observation_end-epoch,
        drain_including_controller_ns=max(0,quiescent-epoch-int(horizon_ns)),
        completion_contract='Jobresponse ends after output GPU completion and common ACK; native async normal commit tail charged into quiescent/completed elapsed separately')
    row['queue_metrics']=fcfs_metrics(row)
    return row

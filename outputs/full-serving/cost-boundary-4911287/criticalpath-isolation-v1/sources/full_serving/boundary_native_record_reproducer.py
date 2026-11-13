"""Small component reproducer, not a serving benchmark or complete B4 worker.

Uses native48map/record kernels inside graphs, LogicalHistory and CpuGpuEvent. Omits native planner/weight transfer,
MoE/attention/NCCL, allocator pressure and serving request scheduling.
"""
import argparse,concurrent.futures,contextlib,json,os,queue,socket,threading,time,traceback
from pathlib import Path
FACTORS={
 'map_record_off':dict(graph=True,async_output=True,worker=True,sync_events=True,recording=False,blocking_output=False),
 'map_record_on':dict(graph=True,async_output=True,worker=True,sync_events=True,recording=True,blocking_output=False),
 'blocking_output':dict(graph=True,async_output=True,worker=True,sync_events=True,recording=True,blocking_output=True),
 'eager_record':dict(graph=False,async_output=True,worker=True,sync_events=True,recording=True,blocking_output=False),
}
PARENTS={'map_record_off':None,'map_record_on':'map_record_off','blocking_output':'map_record_on','eager_record':'map_record_on'}
def changed_factors(name):
 parent=PARENTS[name]
 return [k for k,v in FACTORS[name].items() if parent and v!=FACTORS[parent][k]]
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--variant',choices=FACTORS,required=True);args=p.parse_args()
 import torch,torch.distributed as dist
 from full_serving.logical_history_gpu import LogicalHistory
 from vllm.distributed.eplb.eplb_utils import CpuGpuEvent,device_stream
 from full_serving.boundary_cuda_event_links import install
 rank=int(os.environ['SLURM_PROCID']);assert int(os.environ['SLURM_NTASKS'])==8
 assert torch.cuda.device_count()==1;torch.cuda.set_device(0);cfg=FACTORS[args.variant]
 args.output.mkdir(exist_ok=True);log=args.output/f'events-rank{rank}.jsonl';assert not log.exists()
 def emit(kind,**fields):
  with log.open('a') as f:f.write(json.dumps(dict(kind=kind,rank=rank,wall_ns=time.time_ns(),**fields))+'\n')
 emit('start',variant=args.variant,factors=cfg,hostname=socket.gethostname(),torch=torch.__version__,cuda=torch.version.cuda,scope='native48maprecord_graph_history_event_components_not_full_B4')
 dist.init_process_group('gloo',init_method='tcp://kh021:29619',rank=rank,world_size=8)
 worker_group=dist.new_group(list(range(8)),backend='gloo')
 compute=torch.cuda.Stream();copy=torch.cuda.Stream();ready=torch.cuda.Event();done=torch.cuda.Event(blocking=cfg['blocking_output'])
 x=torch.full((512,512),.01,device='cuda',dtype=torch.float32);y=x.clone();out=torch.empty_like(x);host=torch.empty_like(x,device='cpu',pin_memory=True)
 mapping=(torch.arange(160,device='cuda')%128).repeat(48,1).contiguous();counts=torch.ones((48,160),device='cuda',dtype=torch.int32)
 from vllm.model_executor.layers.fused_moe.router.base_router import eplb_map_to_physical_and_record
 import hashlib,inspect
 emit('native_mapping_source',sha256=hashlib.sha256(Path(inspect.getsourcefile(eplb_map_to_physical_and_record)).read_bytes()).hexdigest())
 logical_ids=(torch.arange(512,device='cuda',dtype=torch.int32)%128).reshape(64,8)
 inverse=torch.full((128,2),-1,device='cuda',dtype=torch.int64);inverse[:,0]=torch.arange(128,device='cuda');inverse[:32,1]=torch.arange(128,160,device='cuda')
 replicas=torch.ones(128,device='cuda',dtype=torch.int64);replicas[:32]=2
 record_enabled=torch.tensor(int(cfg['recording']),device='cuda',dtype=torch.int32)
 expected_ids=[]
 for i in range(512):
  logical=i%128;replica=((i//8)*2654435769 & 0xffffffff)%(2 if logical<32 else 1);expected_ids.append(logical+128 if replica else logical)
 expected_ids_cpu=torch.tensor(expected_ids,dtype=torch.int32).reshape(64,8)
 expected_counts=torch.zeros(160,dtype=torch.int32)+ (torch.bincount(expected_ids_cpu.flatten().long(),minlength=160).int() if cfg['recording'] else 0)
 routed=[]
 def body():
  nonlocal routed
  torch.mm(x,y,out=out);counts.fill_(0)
  routed=[eplb_map_to_physical_and_record(logical_ids,counts[layer],inverse,replicas,record_enabled) for layer in range(48)]
 hist=LogicalHistory('cuda',window=32,layers=48,experts=128,physical=160)
 compute.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(compute):
  for _ in range(3):body()
  hist.update(mapping,counts);hist.snapshot()
 torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph,stream=compute):body()
 # All variants construct identical graphs; only replay vs eager execution differs.
 current=[None];links=[];_,ledger=install(torch,lambda:current[0],links.append)
 jobs=queue.Queue();results=queue.Queue();errors=queue.Queue();worker_cycles=[]
 def worker():
  try:
   torch.cuda.set_device(0);stream=torch.cuda.Stream()
   while True:
    job=jobs.get()
    if job is None:return
    cycle,start,snapshot=job;start.wait(stream=stream)
    with device_stream(stream):cpu=snapshot.cpu()
    flag=torch.tensor([1],dtype=torch.int32);dist.all_reduce(flag,group=worker_group);assert flag.item()==8
    assert torch.isfinite(cpu).all() and cpu.sum()>=0
    for layer in range(48):
     with device_stream(stream):scratch=snapshot[0,layer].clone()
     stream.synchronize()
     consumed=CpuGpuEvent();results.put((cycle,layer,consumed));consumed.wait(stream=stream)
     stream.synchronize()
    worker_cycles.append(cycle)
  except BaseException:errors.put(traceback.format_exc())
 thread=threading.Thread(target=worker,daemon=True)
 if cfg['worker']:thread.start()
 output_pool=concurrent.futures.ThreadPoolExecutor(max_workers=1,initializer=lambda:torch.cuda.set_device(0))
 output_future=None
 def consume():
  done.synchronize();assert abs(float(host[0,0])-.0512)<1e-5
 pending=None;cycles=0;commits=0
 def step(i):
  nonlocal pending,cycles,commits,output_future
  if not errors.empty():raise RuntimeError(errors.get())
  with torch.profiler.record_function(f'cs471::component_step::{i}'):
   with torch.cuda.stream(compute):
    if cfg['graph']:graph.replay()
    else:body()
   # Prior CPU consumer must finish before reusing its pinned host buffer/event.
   if output_future is not None:output_future.result(timeout=30);output_future=None
   ready.record(compute);ready.wait(copy)
   with torch.cuda.stream(copy):host.copy_(out,non_blocking=True)
   done.record(copy)
   if cfg['async_output']:output_future=output_pool.submit(consume)
   else:consume()
   # Preserve graph-output reuse order in both variants.
   done.wait(compute)
   if cfg['worker']:
    with torch.cuda.stream(compute):hist.update(mapping,counts)
    if i%128==0:
     assert pending is None
     with torch.cuda.stream(compute):snapshot=torch.stack(hist.snapshot())
     start=CpuGpuEvent();start.record(compute);pending=i;cycles+=1;jobs.put((i,start,snapshot))
    if pending is not None:
     try:cycle,layer,consumed=results.get(timeout=30)
     except queue.Empty:raise RuntimeError('worker result timeout')
     assert cycle==pending
     # A no-move result still retires the native consumed event after main work.
     consumed.record(compute);commits+=1
     if layer==47:pending=None
  if i%64==63:
   if output_future is not None:output_future.result(timeout=30);output_future=None
   done.synchronize();assert torch.allclose(host,torch.full_like(host,.0512),rtol=1e-4,atol=1e-6)
   assert all(torch.equal(value.cpu(),expected_ids_cpu) for value in routed)
   assert torch.equal(counts.cpu(),expected_counts.repeat(48,1))
   emit('progress',iteration=i+1,commits=commits,native_mapping_and_record_counts_checked=True)
 activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]
 prof=torch.profiler.profile(activities=activities,record_shapes=False,with_stack=False,profile_memory=False,experimental_config=torch._C._profiler._ExperimentalConfig(enable_cuda_sync_events=cfg['sync_events'],profile_all_threads=False))
 # Warm all components before profiler; native worker jobs occur again inside capture.
 for i in range(128):step(i)
 torch.cuda.synchronize();dist.barrier();current[0]='continuous';prof.start();emit('capture_start')
 for i in range(128,512):step(i)
 if output_future is not None:output_future.result(timeout=30)
 output_pool.shutdown(wait=True)
 torch.cuda.synchronize();dist.barrier();prof.stop();current[0]=None
 if cfg['worker']:
  jobs.put(None);thread.join(timeout=30);assert not thread.is_alive() and errors.empty()
  assert cycles==4 and commits==192 and len(worker_cycles)==4
 path=args.output/f'trace-rank{rank}.json.gz';assert not path.exists();prof.export_chrome_trace(str(path))
 (args.output/f'event-links-rank{rank}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in links))
 emit('completed',steps=512,captured_steps=384,cycles=cycles,commits=commits,worker_cycles=worker_cycles,output_checked=True,changed_factors=changed_factors(args.variant))
 dist.destroy_process_group()
if __name__=='__main__':main()

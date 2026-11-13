"""Eight-rank native metadata equivalence and CUDA graph-reader contract.

Synthetic metadata only: separate full-model/all-weight pilot admission required.
Not a dispatcher speed benchmark or numerical model-quality test.
"""
import argparse,hashlib,json,os,socket
from pathlib import Path
from types import SimpleNamespace
from full_serving.boundary_noop_commit import CycleProof,commit_with_proof

def main():
 import torch
 import torch.distributed as dist
 import vllm
 from vllm.distributed.eplb import eplb_state as native
 p=argparse.ArgumentParser();p.add_argument("--protocol",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--master",required=True);a=p.parse_args()
 root=Path(__file__).resolve().parents[1];spec=json.loads(a.protocol.read_text())
 pins=json.loads((root/spec["native_pins"]).read_text())
 for name,sha in pins.items():assert hashlib.sha256((Path(vllm.__file__).parent/name).read_bytes()).hexdigest()==sha
 assert os.environ["SLURM_JOB_ID"]=="4911287"
 assert socket.gethostname().split(".")[0] in ("kh021","kh023")
 rank=int(os.environ["SLURM_PROCID"]);local=int(os.environ["SLURM_LOCALID"])
 assert int(os.environ["SLURM_NTASKS"])==8 and torch.cuda.device_count()==4 and local==rank%4
 torch.cuda.set_device(local)
 dist.init_process_group("gloo",init_method=a.master,rank=rank,world_size=8)
 base=(torch.arange(160,dtype=torch.int64)%128).repeat(48,1)
 def make():
  logical,counts=native.compute_logical_maps(base,128)
  return SimpleNamespace(physical_to_logical_map=base.cuda(),
    logical_to_physical_map=torch.nn.functional.pad(logical,(0,33-logical.shape[-1]),value=-1).cuda(),
    logical_replica_count=counts.cuda())
 ref=make();candidate=make();fields=("physical_to_logical_map","logical_to_physical_map","logical_replica_count")
 pointers={f:getattr(candidate,f).data_ptr() for f in fields}
 # Graphs read the same persistent map allocations, including a zero-token rank.
 graphs=[]
 for n in (0 if rank==0 else 1,1,17,128):
  ids=(torch.arange(n,device="cuda",dtype=torch.int64)+rank)%128
  def reader(ms):
   return (ms.logical_to_physical_map[:,ids,0]+ms.logical_replica_count[:,ids]+ms.physical_to_logical_map[:,:n])
  # n<=128, so all three views have shape [48,n].
  for _ in range(3):reader(ref);reader(candidate)
  torch.cuda.synchronize()
  gr=torch.cuda.CUDAGraph();gc=torch.cuda.CUDAGraph()
  with torch.cuda.graph(gr):yr=reader(ref)
  with torch.cuda.graph(gc):yc=reader(candidate)
  graphs.append((gr,gc,yr,yc,n))
 torch.cuda.synchronize()
 prior=base.clone();events=[];cycles=[]
 for cycle in range(6):
  target=prior.clone()
  if cycle in (1,3):target[cycle::7]=target[cycle::7].roll(1,dims=1)
  digest=hashlib.sha256(prior.numpy().tobytes()+target.numpy().tobytes()).hexdigest()
  agreed=[None]*8;dist.all_gather_object(agreed,digest);assert len(set(agreed))==1
  candidate.cs471_noop_commit_proof=CycleProof(cycle,prior.tolist(),target.tolist())
  if cycle==4:
   # Out-of-band full-map write invalidates the proof; every layer falls back.
   candidate.cs471_noop_commit_proof.invalidate()
   native._commit_eplb_maps(candidate,target)
  before=len(events)
  for layer in range(48):
   native._commit_eplb_maps_for_layer(ref,target[layer],layer)
   commit_with_proof(native._commit_eplb_maps_for_layer,candidate,target[layer],layer,
     lambda kind,**kw:events.append(dict(kind=kind,**kw)))
  torch.cuda.synchronize()
  for f in fields:
   assert torch.equal(getattr(ref,f),getattr(candidate,f)),(cycle,f)
   assert pointers[f]==getattr(candidate,f).data_ptr(),f
  for gr,gc,yr,yc,n in graphs:
   gr.replay();gc.replay();torch.cuda.synchronize();assert torch.equal(yr,yc),(cycle,n)
  cycles.append(dict(cycle=cycle,unchanged_layers=int((prior==target).all(dim=1).sum()),
                     elided_layers=len(events)-before,full_write_invalidation=cycle==4))
  prior=target
 assert cycles[0]["elided_layers"]==48 and cycles[1]["elided_layers"]<48 and cycles[4]["elided_layers"]==0
 record=dict(status="passed",rank=rank,node=socket.gethostname(),device=torch.cuda.get_device_name(),
  cycles=cycles,graph_token_counts=[g[-1] for g in graphs],all_metadata_exact=True,
  stable_graph_map_addresses=True,native_pins_sha256=hashlib.sha256((root/spec["native_pins"]).read_bytes()).hexdigest(),
  scope="Synthetic native metadata and graph-reader equivalence; no model or full-weight proof, no performance inference")
 a.output.mkdir(parents=True,exist_ok=True)
 (a.output/f"rank{rank}.json").write_text(json.dumps(record,indent=2)+"\n")
 dist.barrier();dist.destroy_process_group()
if __name__=="__main__":main()

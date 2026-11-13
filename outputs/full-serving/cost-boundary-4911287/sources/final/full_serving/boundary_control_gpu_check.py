"""Synthetic mapper exactness and graph checks; separate from serving timing."""
import argparse,hashlib,json,os,socket
from pathlib import Path
def main():
 import numpy as np
 import torch
 import torch.distributed as dist
 from full_serving.resident_pair import PairMapper
 from full_serving.resident_policy import balanced_map,cpu_reference
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--master",required=True);a=p.parse_args()
 rank=int(os.environ["SLURM_PROCID"]);local=int(os.environ["SLURM_LOCALID"])
 assert os.environ["SLURM_JOB_ID"]=="4911287" and int(os.environ["SLURM_NTASKS"])==8
 assert socket.gethostname().split(".")[0] in ("kh021","kh023") and torch.cuda.device_count()==4 and local==rank%4
 torch.cuda.set_device(local)
 dist.init_process_group("gloo",init_method=a.master,world_size=8,rank=rank)
 rng=np.random.default_rng(471+rank);records=[]
 for mode in ("native_hash","static_home","resident_v1"):
  mapper=PairMapper(torch.device("cuda",local),balanced_map(),mode=mode)
  for n in (0 if rank==0 else 8,1,17,128,513,4097):
   ids=rng.integers(-1,128,(n,8),dtype=np.int32)
   if n:ids[n//2:]=-1
   tensor=torch.as_tensor(ids,device="cuda")
   expected=cpu_reference(ids,mode=mode)["physical_ids"]
   np.testing.assert_array_equal(mapper(tensor).cpu().numpy(),expected)
   for _ in range(3):mapper(tensor)
   torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):out=mapper(tensor)
   graph.replay();torch.cuda.synchronize()
   np.testing.assert_array_equal(out.cpu().numpy(),expected)
   changed=rng.integers(0,128,(n,8),dtype=np.int32)
   tensor.copy_(torch.as_tensor(changed,device="cuda"))
   graph.replay();torch.cuda.synchronize()
   np.testing.assert_array_equal(out.cpu().numpy(),cpu_reference(changed,mode=mode)["physical_ids"])
   tensor.fill_(-1);graph.replay();torch.cuda.synchronize()
   assert bool(torch.all(out==-1))
   records.append(dict(mode=mode,rows=n,eager_exact=True,graph_exact=True,changed_inputs_exact=True,all_padding_exact=True))
 a.output.mkdir(parents=True,exist_ok=True)
 (a.output/f"rank{rank}.json").write_text(json.dumps(dict(status="passed",rank=rank,node=socket.gethostname(),
  cases=records,map_sha256=hashlib.sha256(json.dumps(balanced_map()).encode()).hexdigest(),
  scope="Synthetic mapping-only contract including zero-token rank and graph replay; actual full-model weights and expert numerics require separate startup pilots. No serving gain inferred."),indent=2)+"\n")
 dist.barrier();dist.destroy_process_group()
if __name__=="__main__":main()

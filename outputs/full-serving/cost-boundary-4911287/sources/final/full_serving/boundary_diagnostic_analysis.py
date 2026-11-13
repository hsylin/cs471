"""Independent checks for the paired host-probe diagnostic, not a speed claim."""
import json,hashlib,math
from pathlib import Path
from full_serving.sustained_analysis import raw_phase,runtime,migrations
from full_serving.proposal_analysis import gate_audit
from full_serving.report_pilots import metric_deltas
from full_serving.custom_report import phase_balance

def audit(root,dataset,probe):
 root=Path(root);data=json.loads((root/"data-manifest.json").read_text())
 status=json.loads((root/"status.json").read_text())
 assert status["state"]=="comparison_completed"
 assert status["completed_sessions"]==[dataset+"/b4"]
 directory=root/dataset/"b4";phases={}
 calendar=json.loads((root/(dataset+"-calendar.json")).read_text())
 for phase in ("warmup-00","warmup-01","measured"):
  item=data["datasets"][dataset]["items"]["measured" if phase=="measured" else "warmup"]
  targets=json.loads((root/(dataset+"-"+phase+"-lengths.json")).read_text())
  phases[phase]=raw_phase(directory/(phase+".jsonl"),item,calendar=calendar if phase=="measured" else None,targets=targets)[0]
  assert phases[phase]["header"]["http_force_close"]
 ident=runtime(directory,"b4");migration=migrations(directory);gate=gate_audit(directory,"b4")
 stats=metric_deltas(directory/"measured-metrics-before.txt",directory/"measured-metrics-after.txt")
 result=dict(status="passed",dataset=dataset,probe=probe,phases=phases,runtime=ident,migrations=migration,gate=gate,
  phase_balance=phase_balance(stats,request_count=512),
  scope="One paired diagnostic group. Observer treatment may alter batching/routes. Host intervals overlap and are not GPU durations or a causal critical-path decomposition.")
 files=list(directory.glob("boundary-host-rank*.jsonl"))
 if probe=="off":assert not files,"Uninstrumented control unexpectedly emitted probe data"
 else:
  assert len(files)==8
  start=phases["measured"]["timing"]["epoch_wall_ns_approx"]
  end=start+phases["measured"]["timing"]["completed_elapsed_ns"]
  peers=[]
  for rank in range(8):
   p=directory/f"boundary-host-rank{rank}.jsonl";rows=[json.loads(s) for s in p.read_text().splitlines()]
   assert all(r["rank"]==rank for r in rows)
   bounds=[r for r in rows if r["kind"]=="bound"];assert bounds
   totals={};used=0;cross=0;previous=bounds[-1]["wall_ns"]
   for r in rows:
    if r["kind"]!="host_intervals":continue
    if previous>=start and r["wall_ns"]<=end:
     used+=1
     for k,v in r["intervals"].items():
      assert not v["failed_calls"]
      t=totals.setdefault(k,dict(calls=0,total_ns=0,max_ns=0))
      t["calls"]+=v["calls"];t["total_ns"]+=v["total_ns"];t["max_ns"]=max(t["max_ns"],v["max_ns"])
    elif previous<end and r["wall_ns"]>start:cross+=1
    previous=r["wall_ns"]
   shapes=[r for r in rows if r["kind"]=="scheduler_shape" and start<=r["wall_ns"]<=end]
   assert used>0 and shapes
   # Dummy calls have synthetic scheduler shapes; keep them separate.
   real=[r for r in shapes if r["dummy_kwarg"] is False and r["profile_kwarg"] is not True]
   peers.append(dict(rank=rank,whole_intervals_inside_measurement=used,crossing_intervals_excluded=cross,
    intervals=totals,scheduler_samples=len(shapes),real_scheduler_samples=len(real),
    sampled_real_tokens=sorted(set(r["scheduled_tokens"] for r in real)),
    sample_limit="CPU scheduled tokens, not gathered/padded GEMM shape; dummy and unknown excluded",
    sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
  result["host_probe"]=peers
 return result
if __name__=="__main__":
 import argparse
 p=argparse.ArgumentParser();p.add_argument("root",type=Path);p.add_argument("--dataset",required=True);p.add_argument("--probe",choices=("off","host"),required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 result=audit(a.root,a.dataset,a.probe);a.output.write_text(json.dumps(result,indent=2)+"\n")
 print(json.dumps({"status":result["status"],"dataset":a.dataset,"probe":a.probe}))

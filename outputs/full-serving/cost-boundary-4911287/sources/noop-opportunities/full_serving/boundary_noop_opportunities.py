"""Reconstruct exact unchanged-map opportunities from existing B4 logs, CPU only."""
import argparse,hashlib,json,subprocess
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def reconstruct(peers,start,end):
 assert len(peers)==8 and start<end
 decisions=[];enqueued=[];completed=[]
 for rank,rows in enumerate(peers):
  assert all(r["rank"]==rank for r in rows)
  dec={r["step"]:r for r in rows if r["kind"]=="decision"}
  assert len(dec)==sum(r["kind"]=="decision" for r in rows)
  for r in dec.values():
   assert r["rank_agreement"] is True
   for key in ("old_map","admitted_map"):
    assert len(r[key])==48 and all(len(row)==160 and all(type(x) is int and 0<=x<128 for x in row) for row in r[key])
  decisions.append(dec)
  pairs=[]
  for kind in ("commit_enqueued","commit_completed"):
   rr=[r for r in rows if r["kind"]==kind];keys={(r["cycle_step"],r["layer"]) for r in rr}
   assert len(keys)==len(rr)
   assert all(step in dec and 0<=layer<48 for step,layer in keys)
   pairs.append(keys)
  assert pairs[1]<=pairs[0];enqueued.append(pairs[0]);completed.append(pairs[1])
 assert all(set(d)==set(decisions[0]) for d in decisions)
 totals={phase:{"proposal_cycles":0,"layer_opportunities":0,"unchanged_layer_opportunities":0,
  "unchanged_all_rank_enqueued":0,"unchanged_all_rank_completed":0,"unchanged_censored_or_unenqueued":0,
  "changed_layer_opportunities":0} for phase in ("before_measurement","during_measurement","after_measurement")}
 cycles=[]
 for step,r in sorted(decisions[0].items()):
  for d in decisions[1:]:
   other=d[step]
   assert all(other[k]==r[k] for k in ("old_map","admitted_map","digest"))
  unchanged=[i for i,(old,new) in enumerate(zip(r["old_map"],r["admitted_map"])) if old==new]
  phase="before_measurement" if r["wall_ns"]<start else ("during_measurement" if r["wall_ns"]<=end else "after_measurement")
  eq=sum(all((step,i) in p for p in enqueued) for i in unchanged)
  done=sum(all((step,i) in p for p in completed) for i in unchanged)
  t=totals[phase];t["proposal_cycles"]+=1;t["layer_opportunities"]+=48;t["unchanged_layer_opportunities"]+=len(unchanged)
  t["unchanged_all_rank_enqueued"]+=eq;t["unchanged_all_rank_completed"]+=done
  t["unchanged_censored_or_unenqueued"]+=len(unchanged)-done;t["changed_layer_opportunities"]+=48-len(unchanged)
  cycles.append(dict(step=step,proposal_phase=phase,unchanged_layers=unchanged,
                     all_rank_enqueued_unchanged_layers=eq,all_rank_completed_unchanged_layers=done))
 return {"partitions_by_proposal_time":totals,"cycles":cycles,
  "limits":["Exact old/admitted row identity agreed on all8 ranks is eligibility only, not a serving-time estimate.",
   "Observed commits are recorded per full session; partition by proposal time does not imply completion inside that phase.",
   "Changed and unknown states must retain native commit. This analysis does not exercise elision or alter runtime.",
   "A saved inverse-map rebuild still retains native transfer/readiness/retirement work in the proposed fix.",
   "No confidence interval, causal payback, critical-path attribution or final fix selection is made."]}

def review(root,dataset,control):
 root=Path(root);assert control in ("b4_observed","b4_gate_only")
 audit=json.loads((root.parent/"audit.json").read_text())
 assert audit["status"]=="passed" and not audit["pilot"] and audit["dataset"]==dataset and audit["control"]==control
 assert json.loads((root/"status.json").read_text())["state"]=="comparison_completed"
 source=json.loads((root/"controller-source.json").read_text())["declared_git_revision"]
 assert source=="c05186025dc152f103f61a7c2703b65fa74f3afe"
 timing=audit["phases"]["measured"]["timing"];start=timing["epoch_wall_ns_approx"];end=start+timing["completed_elapsed_ns"]
 paths=[root/dataset/control/f"gate-events-rank{rank}.jsonl" for rank in range(8)]
 hashes={p.name:sha(p) for p in paths}
 result=reconstruct([[json.loads(line) for line in p.read_text().splitlines()] for p in paths],start,end)
 assert all(sha(p)==hashes[p.name] for p in paths)
 return dict(status="passed",dataset=dataset,control=control,gpu_rerun=False,runtime_source=source,
  input_audit_sha256=sha(root.parent/"audit.json"),input_gate_sha256=hashes,evidence=result,selection_made=False)
def main():
 p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--dataset",required=True)
 p.add_argument("--control",required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 result=review(a.root,a.dataset,a.control)
 result["analysis_source_revision"]=subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).resolve().parents[1],text=True).strip()
 with a.output.open("x") as f:json.dump(result,f,indent=2);f.write("\n")
if __name__=="__main__":main()

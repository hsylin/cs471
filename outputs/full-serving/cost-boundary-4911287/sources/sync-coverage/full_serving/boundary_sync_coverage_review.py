"""CPU-only temporal coverage review of completed native-DP aligned diagnostics."""
import argparse, hashlib, json, math, subprocess
from pathlib import Path
from full_serving.boundary_shape_review import distribution, inside

SOURCE="620cbb0170c8ea275229092a88948356935c0686"
PROTOCOL="733cac8b193afc81b861699d7492ce80e32d25b04a640c443c3dcba57a7ea15a"
LIMITS=[
 "Native DP-ordinal systematic samples are not an unbiased frequency estimate.",
 "Time coverage uses rank0 wall time relative to the client's approximate wall epoch; it is not cross-node clock synchronization proof.",
 "Unobserved time bins and missing rank observations are not zero work.",
 "CPU token and attention-pair summaries are not exact FLOPs, KV occupancy, kernel costs or exposed critical-path attribution.",
 "Active samples include explicit empty ranks. All-empty observations are counted separately and have no max/mean work ratio.",
 "Prefill-only, decode-only and mixed samples remain separate; dataset names do not establish phase dominance.",
 "Broad query-only CUDA spans include host gaps and peer waits. No serving speedup or causal migration payback is inferred."
]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ratio(values):
 assert len(values)==8 and all(type(v) is int and v>=0 for v in values)
 return max(values)/(sum(values)/8) if sum(values) else None

def coverage(samples, rank0_rows, timing):
 start=timing["epoch_wall_ns_approx"]; horizon=timing["offered_horizon_ns"]/1e9
 end=start+timing["completed_elapsed_ns"]
 assert horizon>0 and timing["completed_elapsed_ns"]>=timing["offered_horizon_ns"]
 by_ordinal={}
 for r in rank0_rows:
  if r["kind"] in ("actual_batch","explicit_empty_batch") and inside(r,start,end):
   n=r["sync_call"];assert n not in by_ordinal
   by_ordinal[n]=r
 bins=[{"start_s":i*horizon/10,"end_s":(i+1)*horizon/10,"observed":0,"active":0,"all_empty":0} for i in range(10)]
 drain={"observed":0,"active":0,"all_empty":0}; offsets=[]; active_offsets=[]
 groups={k:{"real_token_max_over_mean":[],"attention_proxy_max_over_mean":[],"active_ranks":[],"padded_tokens_per_real_token":[]} for k in ("prefill_only","decode_only","mixed")}
 seen=set()
 for sample in samples:
  n=sample["sync_call"];assert n not in seen and n in by_ordinal;seen.add(n)
  row=by_ordinal[n];assert row["sample_policy"]=="native_dp_sync_ordinal" and not row["profile"]
  real=sample["real_tokens_per_rank"];pre=sample["prefill_tokens_per_rank"];dec=sample["decode_tokens_per_rank"]
  assert len(real)==len(pre)==len(dec)==8 and all(a+b==c for a,b,c in zip(pre,dec,real))
  assert sample["total_real_tokens"]==sum(real)
  dummy=sample["dummy_ranks"];assert len(dummy)==len(set(dummy)) and all(0<=r<8 and real[r]==0 for r in dummy)
  elapsed=(row["execute_start_wall_ns"]-start)/1e9;assert elapsed>=0
  offsets.append(elapsed)
  bucket=bins[min(9,int(elapsed/horizon*10))] if elapsed<horizon else drain
  bucket["observed"]+=1
  if sum(real)==0:
   bucket["all_empty"]+=1
   assert sample["real_token_max_over_mean"] is None
   continue
  bucket["active"]+=1;active_offsets.append(elapsed)
  kind="mixed" if sum(pre)>0 and sum(dec)>0 else ("prefill_only" if sum(pre)>0 else "decode_only")
  computed=ratio(real);assert math.isclose(computed,sample["real_token_max_over_mean"],rel_tol=1e-12)
  assert row["real_tokens"]==real[0]
  groups[kind]["real_token_max_over_mean"].append(computed)
  attention=ratio(sample["attention_pairs_upper_per_rank"])
  if attention is not None:groups[kind]["attention_proxy_max_over_mean"].append(attention)
  groups[kind]["active_ranks"].append(sum(x>0 for x in real))
  padded=sample["padded_tokens_across_dp"]
  assert len(padded)==8 and all(type(p) is int and p>=v for p,v in zip(padded,real))
  groups[kind]["padded_tokens_per_real_token"].append(sum(padded)/sum(real))
 for b in bins:b["observed_active_work"]=True if b["active"] else (False if b["observed"] else None)
 active_bins=sum(b["active"]>0 for b in bins)
 return {"aligned_samples":len(samples),"active_samples":len(active_offsets),
  "rank0_first_sample_elapsed_s":min(offsets) if offsets else None,
  "rank0_last_sample_elapsed_s":max(offsets) if offsets else None,
  "rank0_first_active_elapsed_s":min(active_offsets) if active_offsets else None,
  "rank0_last_active_elapsed_s":max(active_offsets) if active_offsets else None,
  "offered_horizon_s":horizon,"elapsed_s":timing["completed_elapsed_ns"]/1e9,
  "offered_decile_bins":bins,"drain_samples":drain,"active_offered_bins":active_bins,
  "sampled_active_work_in_all_offered_deciles":active_bins==10,
  "sampled_work_classes":{kind:{metric:distribution(v) for metric,v in values.items()} for kind,values in groups.items()},
  "scope":"Coverage and sample distributions only; no automatic bottleneck selection or execution-frequency estimate."}

def review(root,dataset):
 root=Path(root);assert dataset in ("longbench","numinamath")
 audit_path=root.parent/"audit.json";a=json.loads(audit_path.read_text())
 assert a["status"]=="passed" and a["sync_shape_probe_passed"] and a["dataset"]==dataset and not a["pilot"]
 assert json.loads((root/"status.json").read_text())["state"]=="comparison_completed"
 assert json.loads((root/"controller-source.json").read_text())["declared_git_revision"]==SOURCE
 assert sha(root/"protocol.json")==PROTOCOL
 directory=root/dataset/"b4"
 for name,h in a["input_sha256"].items():assert sha(directory/name)==h
 timing=a["ordinary"]["ordinary"]["phases"]["measured"]["timing"]
 rows=[json.loads(x) for x in (directory/"boundary-shapes-rank0.jsonl").read_text().splitlines()]
 result=coverage(a["aligned"]["samples"],rows,timing)
 assert result["active_samples"]==a["active_aligned_samples"]
 for name,h in a["input_sha256"].items():assert sha(directory/name)==h
 return {"status":"passed","dataset":dataset,"gpu_rerun":False,"runtime_source_revision":SOURCE,
  "runtime_protocol_sha256":PROTOCOL,"input_audit_sha256":sha(audit_path),"input_record_sha256":a["input_sha256"],
  "coverage":result,"limitations":LIMITS,"automatic_fix_selection":False}

def main():
 p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
 p.add_argument("--dataset",choices=("longbench","numinamath"),required=True);p.add_argument("--output",type=Path,required=True)
 a=p.parse_args();result=review(a.root,a.dataset)
 result["analysis_source_revision"]=subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).resolve().parents[1],text=True).strip()
 with a.output.open("x") as f:json.dump(result,f,indent=2);f.write("\n")
if __name__=="__main__":main()

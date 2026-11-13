"""Audit actual runtime shapes separately from every request's timing."""
import argparse,hashlib,json,statistics
from pathlib import Path
from full_serving.sustained_analysis import pilot as ordinary_pilot
from full_serving.boundary_diagnostic_analysis import audit as paired_audit
from full_serving.sustained_analysis import raw_phase,runtime,migrations
from full_serving.report_pilots import metric_deltas
from full_serving.custom_report import phase_balance
from full_serving.proposal_analysis import gate_audit

def audit(root,pilot=False,dataset=None):
 root=Path(root);directory=root/("pilot" if pilot else dataset)/"b4"
 if pilot:
  ordinary=ordinary_pilot(root,"b4");start=0;end=2**63
 else:
  data=json.loads((root/"data-manifest.json").read_text())
  status=json.loads((root/"status.json").read_text());assert status["state"]=="comparison_completed"
  assert status["completed_sessions"]==[dataset+"/b4"]
  phases={}
  for phase in ("warmup-00","warmup-01","measured"):
   item=data["datasets"][dataset]["items"]["measured" if phase=="measured" else "warmup"]
   targets=json.loads((root/(dataset+"-"+phase+"-lengths.json")).read_text())
   calendar=json.loads((root/(dataset+"-calendar.json")).read_text()) if phase=="measured" else None
   phases[phase]=raw_phase(directory/(phase+".jsonl"),item,targets=targets,calendar=calendar)[0]
  ordinary=dict(status="passed",phases=phases,runtime=runtime(directory,"b4"),migrations=migrations(directory),
      gate=gate_audit(directory,"b4"),phase_balance=phase_balance(metric_deltas(directory/"measured-metrics-before.txt",directory/"measured-metrics-after.txt"),request_count=512))
  start=phases["measured"]["timing"]["epoch_wall_ns_approx"]
  end=start+phases["measured"]["timing"]["completed_elapsed_ns"]
 gauges={}
 for phase in ordinary["phases"]:
  path=directory/(phase+"-engine-gauges.jsonl")
  samples=[json.loads(x) for x in path.read_text().splitlines()]
  observed_samples=[r for r in samples if r["status"]=="observed"]
  assert observed_samples and all({g["engine"] for g in r["gauges"] if g["metric"]=="kv_cache_usage_perc"}==set(range(8)) for r in observed_samples)
  gauges[phase]=dict(samples=len(samples),unobserved=len(samples)-len(observed_samples),
    kv_cache_usage_peak={str(rank):max(g["value"] for r in observed_samples for g in r["gauges"] if g["metric"]=="kv_cache_usage_perc" and g["engine"]==rank) for rank in range(8)},
    scope="Native cached gauges sampled every5 seconds. Engine labels0..7; telemetry lag and nonsimultaneous engine refresh prevent exact per-iteration occupancy claims.",
    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
 peers=[];syncs=[]
 for rank in range(8):
  p=directory/f"boundary-shapes-rank{rank}.jsonl"
  rows=[json.loads(x) for x in p.read_text().splitlines()]
  assert rows and all(r["rank"]==rank for r in rows)
  enqueue={r["execute_call"]:r for r in rows if r["kind"]=="gpu_span_enqueued"}
  observed={r["execute_call"]:r for r in rows if r["kind"]=="gpu_span_observed"}
  assert len(enqueue)==sum(r["kind"]=="gpu_span_enqueued" for r in rows)
  assert len(observed)==sum(r["kind"]=="gpu_span_observed" for r in rows)
  assert set(observed)<=set(enqueue)
  assert not any(r["failed"] for r in enqueue.values())
  inside={k:r for k,r in enqueue.items() if start<=r["execute_start_wall_ns"] and r["wall_ns"]<=end}
  measured=[r for k,r in observed.items() if k in inside and r["wall_ns"]<=end]
  real=[r for r in rows if r["kind"]=="actual_batch" and start<=r["execute_start_wall_ns"] and r["wall_ns"]<=end]
  assert real and all(not r["dummy"] and not r["profile"] for r in real)
  assert all(r["real_tokens"]<=r["padded_tokens"] and r["prefill_tokens"]+r["decode_tokens"]==r["real_tokens"] for r in real)
  agreed={r["sync_call"]:r for r in rows if r["kind"]=="agreed_batch" and start<=r["execute_start_wall_ns"] and r["wall_ns"]<=end}
  syncs.append(agreed)
  durations=[r["gpu_stream_ms"] for r in measured];assert durations and min(durations)>=0
  peers.append(dict(rank=rank,actual_samples=len(real),real_token_shapes=sorted({r["real_tokens"] for r in real}),
    padded_token_shapes=sorted({r["padded_tokens"] for r in real}),
    sampled_prefill_tokens=sum(r["prefill_tokens"] for r in real),
    sampled_decode_tokens=sum(r["decode_tokens"] for r in real),
    max_cpu_sequence_upper=max(r["sequence_cpu_upper_max"] for r in real),
    sampled_attention_pairs_upper_proxy=sum(r["attention_pairs_cpu_upper_proxy"] for r in real),
    gpu_spans_observed=len(measured),gpu_spans_censored=sum(k not in {r["execute_call"] for r in measured} for k in inside),
    gpu_stream_ms_mean=statistics.fmean(durations),gpu_stream_ms_max=max(durations),
    sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 shared=set.intersection(*(set(x) for x in syncs))
 agreement=[]
 for index in sorted(shared):
  sizes=[x[index]["padded_tokens_across_dp"] for x in syncs]
  assert all(s==sizes[0] for s in sizes),"Shared DP ordinal has inconsistent agreement; cannot align ranks"
  agreement.append(dict(sync_call=index,padded_tokens_across_dp=sizes[0]))
 return dict(status="passed",shape_probe_passed=True,pilot=pilot,dataset=dataset,ordinary=ordinary,
  engine_gauges=gauges,ranks=peers,all_rank_shared_samples=len(agreement),shared_dp_agreements=agreement,
  limits=["Sampling only, not an unbiased execution-frequency estimate: first8 real calls and every64 calls",
  "CPU sequence lengths and attention-pair count are upper-bound proxies, not exact KV occupancy or FLOPs",
  "Current-stream event intervals include host gaps and peer waiting, exclude later EPLB step; not kernel duration sums or causal critical-path proof",
  "Unknown/unobserved final events and crossing-phase spans are censored",
  "No private prompts, token IDs, request IDs or output arrays collected",
  "Diagnostic overhead charged in this run; timings are not final speed evidence"])
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("root",type=Path);p.add_argument("--pilot",action="store_true");p.add_argument("--dataset");p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 d=audit(a.root,a.pilot,a.dataset);a.output.write_text(json.dumps(d,indent=2)+"\n");print(json.dumps({"status":d["status"],"shape_probe_passed":True}))

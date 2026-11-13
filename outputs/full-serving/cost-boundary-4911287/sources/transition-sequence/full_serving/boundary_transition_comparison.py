"""Cross-run comparability for frozen-candidate controls, never causal by default."""
import argparse
import json
from pathlib import Path

def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]

def completed_prefix(left,right,left_timing,right_timing,left_cutoff,right_cutoff):
    assert len(left)==len(right)
    common=matched=contiguous=0;prefix_open=True
    for a,b in zip(left,right):
        assert a["request_id"]==b["request_id"] and a["input_sha256"]==b["input_sha256"]
        before=False
        if left_cutoff is not None and right_cutoff is not None:
            ends=[t["epoch_wall_ns_approx"]+r["end_ns"]-t["epoch_monotonic_ns"]
                  for r,t in ((a,left_timing),(b,right_timing))]
            before=ends[0]<left_cutoff and ends[1]<right_cutoff
        if before:
            common+=1;matched+=a["output_sha256"]==b["output_sha256"]
        if prefix_open and before:contiguous+=1
        else:prefix_open=False
    return dict(completed_before_both_proposals=common,matching_output_hashes=matched,
                contiguous_completed_request_prefix=contiguous,
                every_comparable_completed_output_matches=matched==common if common else None,
                timestamp_scope="Completion wall times use client epoch_wall_ns_approx; active KV and in-flight output prefixes are unobserved.")

def compare_main(keep_root,charged_root,dataset):
    roots={"keep":Path(keep_root),"charged":Path(charged_root)}
    plans={};reports={}
    for side,root in roots.items():
        assert json.loads((root/"status.json").read_text())["state"]=="comparison_completed"
        reports[side]=json.loads((root.parent/"audit.json").read_text())
        assert reports[side]["status"]=="passed" and reports[side]["control"]==side
        plans[side]=[{r["step"]:r for r in rows(root/dataset/side/f"transition-rank{rank}.jsonl") if r["kind"]=="proposal"} for rank in range(8)]
    targets=[[step for step,r in p.items() if r["triggered"]] for p in plans["charged"]]
    assert all(t==targets[0] for t in targets)
    step=targets[0][0] if len(targets[0])==1 else None
    opportunity=step is not None and all(step in p for peers in plans.values() for p in peers)
    checks=[];cutoffs={"keep":None,"charged":None}
    if opportunity:
        for rank in range(8):
            a,b=plans["keep"][rank][step],plans["charged"][rank][step]
            checks.append(dict(rank=rank,old_map_equal=a["old_map_sha256"]==b["old_map_sha256"],
                               live_demand_equal=a["live_demand_sha256"]==b["live_demand_sha256"]))
        for side in roots:cutoffs[side]=min(p[step]["wall_ns"] for p in plans[side])
    prefixes={};same_counts=same_outputs=requests=0
    for phase in ("warmup-00","warmup-01","measured"):
        rr={}
        for side,root in roots.items():
            rr[side]=sorted((r for r in rows(root/dataset/side/(phase+".jsonl")) if r.get("type")=="request"),key=lambda r:r["request_index"])
        left,right=rr["keep"],rr["charged"]
        prefixes[phase]=completed_prefix(left,right,reports["keep"]["phases"][phase]["timing"],
                                         reports["charged"]["phases"][phase]["timing"],cutoffs["keep"],cutoffs["charged"])
        for a,b in zip(left,right):
            assert a["usage"]["completion_tokens"]==b["usage"]["completion_tokens"]
            same_counts+=1;same_outputs+=a["output_sha256"]==b["output_sha256"];requests+=1
    return dict(status="passed",dataset=dataset,target_step=step,opportunity_all_ranks=opportunity,
                state_checks=checks,completed_predecision_prefixes=prefixes,requests_including_warmups=requests,
                equal_output_counts=same_counts,equal_output_hashes=same_outputs,
                causal_payback_established=False,
                reasons=["Independent sessions do not clone active KV state or in-flight output prefixes.",
                         "Fixed output counts do not fix tokens, routes, batching or floating-point reductions.",
                         "One ordered pair cannot estimate run-to-run variation.",
                         "PREINSTALLED has different preceding warmup state and excludes startup migration cost.",
                         "Use observed commit records; draining alone proves no commit."])

def pilot_logprobs(run,private):
    # Uses existing private token arrays in place; returns aggregate diagnostics only.
    from full_serving.pilot_output_diagnostic import compare_outputs
    run,private=Path(run),Path(private);out={}
    for other in ("preinstalled","charged"):
        for ds in ("longbench","numinamath"):
            label=ds+"-warmup-00"
            args=[]
            for side in ("keep",other):
                args += [run/side/"results/pilot"/side/(label+".jsonl"),
                         private/side/"pilot"/side/(label+"-outputs.jsonl")]
            result=compare_outputs(*args)
            out[other+"/"+ds]=dict(integrity_status=result["integrity_status"],
                                  aggregate=result["aggregate"],scope=result["scope"],
                                  conditioning=result["conditioning"],limitations=result["limitations"])
    return dict(status="passed",comparisons=out,
                numerical_equivalence_established=False,
                scope="Token-prefix selected-logprob diagnostic, not full logits or model-quality proof. Raw text and token arrays remain private.")

if __name__=="__main__":
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest="mode",required=True)
    a=sub.add_parser("main");a.add_argument("--keep",type=Path,required=True);a.add_argument("--charged",type=Path,required=True);a.add_argument("--dataset",required=True)
    a=sub.add_parser("pilot");a.add_argument("--run",type=Path,required=True);a.add_argument("--private",type=Path,required=True)
    for a in sub.choices.values():a.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    report=compare_main(a.keep,a.charged,a.dataset) if a.mode=="main" else pilot_logprobs(a.run,a.private)
    with a.output.open("x") as f:f.write(json.dumps(report,indent=2)+"\n")

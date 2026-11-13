"""Audit sync-ordinal shape coverage using actual-batch or explicit empty records."""
import argparse,hashlib,json
from pathlib import Path
from full_serving.boundary_shape_analysis import audit as ordinary_audit
from full_serving.boundary_shape_review import rank_review,inside
from full_serving.boundary_sync_shape_contract import aligned_summary

def audit(root,pilot=False,dataset=None):
    root=Path(root);directory=root/("pilot" if pilot else dataset)/"b4"
    ordinary=ordinary_audit(root,pilot,dataset)
    ordinary["limits"]=[s.replace("first8 real calls and every64 calls","first8 native DP dispatches and every64 native DP dispatches") for s in ordinary["limits"]]
    for rank in ordinary["ranks"]:
        rank["legacy_mixed_gpu_stream_ms_mean"]=rank.pop("gpu_stream_ms_mean")
        rank["legacy_mixed_gpu_stream_ms_max"]=rank.pop("gpu_stream_ms_max")
    if pilot:start,end=0,2**63
    else:
        timing=ordinary["ordinary"]["phases"]["measured"]["timing"]
        start=timing["epoch_wall_ns_approx"];end=start+timing["completed_elapsed_ns"]
    peers=[];rank_stats=[];hashes={}
    for rank in range(8):
        path=directory/f"boundary-shapes-rank{rank}.jsonl";content=path.read_bytes()
        hashes[path.name]=hashlib.sha256(content).hexdigest()
        rows=[json.loads(x) for x in content.splitlines()]
        rank_stats.append(rank_review(rows,rank,start,end)[0])
        agreed={r["sync_call"]:r for r in rows if r["kind"]=="agreed_batch" and inside(r,start,end)}
        batches=[r for r in rows if r["kind"] in ("actual_batch","explicit_empty_batch") and inside(r,start,end)]
        merged={}
        for batch in batches:
            ordinal=batch["sync_call"]
            assert ordinal not in merged and ordinal in agreed
            sync=agreed[ordinal]
            assert batch["execute_call"]==sync["execute_call"]
            assert batch["sample_policy"]==sync["sample_policy"]=="native_dp_sync_ordinal"
            assert batch["expected_sync_call"]==sync["expected_sync_call"]==ordinal
            assert batch["profile"] is False
            if batch["kind"]=="explicit_empty_batch":
                assert batch["dummy"] is True and batch["real_tokens"]==0 and batch["no_actual_request_batch"]
            else:
                assert batch["dummy"] is False
                assert batch["real_tokens"]==batch["prefill_tokens"]+batch["decode_tokens"]
            merged[ordinal]=dict(batch,padded_tokens_across_dp=sync["padded_tokens_across_dp"])
        peers.append(merged)
    aligned=aligned_summary(peers)
    active=[r for r in aligned["samples"] if r["total_real_tokens"]>0]
    assert len(active)>=(1 if pilot else 8),"Insufficient aligned active-work samples; preserve run, no bottleneck claim"
    empty=sum(len(r["dummy_ranks"]) for r in active)
    if pilot:assert empty>0,"Admission did not exercise explicit empty-rank observations"
    for name,digest in hashes.items():assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==digest
    return {"status":"passed","sync_shape_probe_passed":True,"pilot":pilot,"dataset":dataset,
            "ordinary":ordinary,"aligned":aligned,"real_dummy_separated_ranks":rank_stats,
            "active_aligned_samples":len(active),"explicit_empty_rank_observations":empty,
            "input_sha256":hashes,
            "limits":["Uses source-pinned native dummy branch to establish zero actual request work, not dummy scheduler descriptors.",
                      "Systematic samples are not an unbiased execution-frequency estimator.",
                      "Padded token vectors, CPU attention proxies and cached KV gauges are not exact FLOPs or KV occupancy.",
                      "Event spans retain host gaps and peer waiting; no kernel or exposed critical-path attribution.",
                      "Only a fresh unprofiled final comparison may establish serving benefit."]}
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);p.add_argument("--pilot",action="store_true")
    p.add_argument("--dataset");p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    value=audit(a.root,a.pilot,a.dataset)
    with a.output.open("x") as f:f.write(json.dumps(value,indent=2)+"\n")
    print(json.dumps({"status":value["status"],"sync_shape_probe_passed":True,"active_aligned_samples":value["active_aligned_samples"]}))

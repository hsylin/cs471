"""CPU-only review of existing shape records; no GPU reruns or runtime edits."""
import argparse,collections,hashlib,json,math,statistics,time
from pathlib import Path

def distribution(values):
    if not values:return {"n":0,"mean":None,"p95":None,"max":None}
    assert all(math.isfinite(x) and x>=0 for x in values)
    ordered=sorted(values);pos=(len(values)-1)*.95;lo=int(pos);hi=min(lo+1,len(values)-1)
    return {"n":len(values),"mean":math.fsum(values)/len(values),
            "p95":ordered[lo]+(ordered[hi]-ordered[lo])*(pos-lo),"max":max(values)}

def inside(row,start,end):
    return start<=row["execute_start_wall_ns"]<=row["wall_ns"]<=end

def rank_review(rows,rank,start,end):
    assert all(r["rank"]==rank for r in rows)
    enqueued={r["execute_call"]:r for r in rows if r["kind"]=="gpu_span_enqueued"}
    observed={r["execute_call"]:r for r in rows if r["kind"]=="gpu_span_observed"}
    assert len(enqueued)==sum(r["kind"]=="gpu_span_enqueued" for r in rows)
    assert len(observed)==sum(r["kind"]=="gpu_span_observed" for r in rows)
    assert observed.keys()<=enqueued.keys()
    actual=[r for r in rows if r["kind"]=="actual_batch" and inside(r,start,end)]
    actual_calls={r["execute_call"] for r in actual}
    spans={}
    for label in ("real_batch","dummy","non_dummy_without_batch"):
        dummy=label=="dummy"
        eligible={k:r for k,r in enqueued.items() if inside(r,start,end) and r["dummy"] is dummy and r["profile"] is False
                  and (dummy or ((k in actual_calls)==(label=="real_batch")))}
        assert all(not r["failed"] for r in eligible.values())
        done={k:observed[k] for k in eligible if k in observed and observed[k]["wall_ns"]<=end}
        for k,r in done.items():
            assert r["dummy"] is dummy and r["profile"] is False and r["execute_start_wall_ns"]==eligible[k]["execute_start_wall_ns"]
        spans[label]={"enqueued_samples":len(eligible),
            "observed_samples":len(done),"censored_samples":len(eligible)-len(done),
            "gpu_stream_ms":distribution([r["gpu_stream_ms"] for r in done.values()])}
    assert all(r["dummy"] is False and r["profile"] is False for r in actual)
    for r in actual:
        assert r["real_tokens"]<=r["padded_tokens"]
        assert r["prefill_tokens"]+r["decode_tokens"]==r["real_tokens"]
    agreed={r["sync_call"]:r for r in rows if r["kind"]=="agreed_batch" and inside(r,start,end) and r["profile"] is False}
    assert len(agreed)==sum(r["kind"]=="agreed_batch" and inside(r,start,end) and r["profile"] is False for r in rows)
    actual_by_sync={r["sync_call"]:r for r in actual if "sync_call" in r}
    result={"rank":rank,"real_shape_samples":len(actual),"gpu_spans_separated":spans,
            "sampled_real_token_histogram":dict(sorted(collections.Counter(r["real_tokens"] for r in actual).items())),
            "sampled_padded_token_histogram":dict(sorted(collections.Counter(r["padded_tokens"] for r in actual).items())),
            "sampled_real_requests":distribution([r["real_requests"] for r in actual]),
            "sampled_cpu_sequence_upper_max":distribution([r["sequence_cpu_upper_max"] for r in actual]),
            "sampled_attention_pairs_upper_proxy":distribution([r["attention_pairs_cpu_upper_proxy"] for r in actual]),
            "sampled_prefill_tokens":sum(r["prefill_tokens"] for r in actual),
            "sampled_decode_tokens":sum(r["decode_tokens"] for r in actual),
            "agreed_dp_vector_samples":len(agreed),
            "real_shapes_without_sync_ordinal":len(actual)-len(actual_by_sync)}
    return result,agreed,actual_by_sync

def cross_rank(agreed,actual):
    assert len(agreed)==len(actual)==8
    groups={}
    for rank,peer in enumerate(agreed):
        for ordinal,row in peer.items():groups.setdefault(ordinal,[]).append((rank,row))
    coverage=collections.Counter();coobserved=0
    for ordinal,peer in groups.items():
        coverage[len(peer)]+=1
        vectors=[r["padded_tokens_across_dp"] for _,r in peer]
        assert all(v is not None and len(v)==8 and all(type(x) is int and x>=0 for x in v) for v in vectors)
        assert all(v==vectors[0] for v in vectors),"Observed common DP ordinal disagrees"
        if len(peer)>1:coobserved+=1
    shared=set.intersection(*(set(peer) for peer in agreed))
    shared_real=set.intersection(*(set(peer) for peer in actual))
    real_skews=[]
    for ordinal in sorted(shared_real):
        values=[peer[ordinal]["real_tokens"] for peer in actual]
        mean=statistics.fmean(values)
        real_skews.append(max(values)/mean if mean else 1.)
    return {"unique_sampled_sync_ordinals":len(groups),
            "rank_coverage_histogram":dict(sorted(coverage.items())),
            "multi_rank_coobserved_ordinals":coobserved,"coobserved_vectors_agree":True,
            "all_eight_dp_vector_samples":len(shared),
            "all_eight_real_shape_samples":len(shared_real),
            "simultaneous_real_work_skew_evidence_available":bool(shared_real),
            "sampled_all_rank_real_token_max_over_mean":distribution(real_skews),
            "scope":"Native DP-sync ordinals are compared only when actually observed on multiple ranks. Missing rank samples are not zero work. Agreed padded vectors do not reveal unpadded work or KV occupancy."}

def review(root,dataset):
    root=Path(root);directory=root/dataset/"b4";audit_path=root.parent/"audit.json"
    audit=json.loads(audit_path.read_text());assert audit["status"]=="passed" and audit["shape_probe_passed"]
    assert audit["dataset"]==dataset and not audit["pilot"]
    assert json.loads((root/"status.json").read_text())["state"]=="comparison_completed"
    timing=audit["ordinary"]["phases"]["measured"]["timing"]
    start=timing["epoch_wall_ns_approx"];end=start+timing["completed_elapsed_ns"]
    hashes={};results=[];agreements=[];actual=[]
    for rank in range(8):
        path=directory/f"boundary-shapes-rank{rank}.jsonl"
        content=path.read_bytes();hashes[str(path)]=hashlib.sha256(content).hexdigest()
        rows=[json.loads(line) for line in content.splitlines()]
        result,a,r=rank_review(rows,rank,start,end)
        results.append(result);agreements.append(a);actual.append(r)
    for path,digest in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    cross=cross_rank(agreements,actual)
    return {"status":"passed","scope":"CPU reanalysis of completed shape diagnostic; not new GPU execution",
            "dataset":dataset,"gpu_rerun":False,"original_audit_sha256":hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            "input_hashes":hashes,"ranks":results,"cross_rank_coverage":cross,
            "original_all_rank_shared_samples":audit["all_rank_shared_samples"],
            "engine_gauges":audit["engine_gauges"]["measured"],
            "limitations":["Original aggregate GPU spans mix real and dummy calls; use the separately labeled distributions here.",
                "Dummy calls participate in padded execution but are not real request batches. Non-dummy spans without an observed actual batch are separately labeled and are not counted as real batch spans.",
                "Samples are first8 real calls plus every64 local execute calls, not unbiased execution frequencies.",
                "Local execute counters diverge across ranks, so sampling does not ensure simultaneous eight-rank coverage.",
                "Approximate client wall-phase boundaries are not cross-node clock synchronization proof.",
                "Event spans include host gaps/preparation/forward/peer waiting and exclude later EPLB; not kernel-only or exposed critical path.",
                "Pending, late-observed and boundary-crossing intervals are not used as completed real spans.",
                "Cached KV gauges and CPU sequence upper bounds do not identify exact per-iteration KV, attention FLOPs, or bottleneck."]}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--dataset",required=True)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args();result=review(a.root,a.dataset)
    with a.output.open("x") as f:f.write(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"status":result["status"],"dataset":a.dataset,"coverage":result["cross_rank_coverage"]}))

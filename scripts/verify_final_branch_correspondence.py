"""Read-only branch/full-policy correspondence audit, preferably beside raw NPZ files."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
import numpy as np


def main():
    p = argparse.ArgumentParser()
    for name in ("manifest", "records", "branches"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--trace-dir", type=Path)
    a = p.parse_args(); m = json.loads(a.manifest.read_text()); d = m["diagnostics"]
    digest = hashlib.sha256(json.dumps({k:v for k,v in m.items() if k!="freeze"},sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    assert digest == m["freeze"]["sha256"] and m["split"] == "final"
    full = [json.loads(x) for x in a.records.read_text().splitlines() if x.strip()]
    b3 = {(r["regime"],r["trace_variant"],r["batch_tokens"],r["seed"],r["repetition"]):r for r in full if r["method"]=="B3"}
    keys = [(c["regime"],c["trace_variant"],batch,seed,rep,step) for seed,rep,c,batch,step in itertools.product(
        d["seeds"],range(d["repetitions"]),d["cases"],d["batch_tokens"],d["decision_steps"])]
    assert len(keys)==8
    orders = [["allow","skip"] if i%2==0 else ["skip","allow"] for i in range(8)]
    random.Random(d.get("order_seed",471)).shuffle(orders); expected_orders=dict(zip(keys,orders))
    rows=[json.loads(x) for x in a.branches.read_text().splitlines() if x.strip()]
    assert len(rows)==8; seen=set(); checked=[]
    def tape(record):
        artifact=record["trace_artifact"]; path=Path(artifact["path"])
        if a.trace_dir: path=a.trace_dir/path.name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==artifact["sha256"]
        with np.load(path,allow_pickle=False) as z:
            ids=z["expert_ids"].copy(); weights=z["combining_weights"].copy(); hidden=z["hidden_states"].copy()
            h=hashlib.sha256(str(z["metadata"]).encode());h.update(memoryview(ids).cast('B'));h.update(memoryview(weights).cast('B'))
            assert h.hexdigest()==record["trace_sha256"]==str(z["trace_sha256"])
        return ids,weights,hidden
    for r in rows:
        k=tuple(r[x] for x in ("regime","trace_variant","batch_tokens","seed","repetition","decision_step"))
        assert k in expected_orders and k not in seen;seen.add(k)
        mainrow=b3[k[:-1]]; decision=next(x for x in mainrow["decision_log"] if x["step"]==k[-1])
        for item in (r,mainrow):
            assert item["config_hash"]==digest and item["harness_sha256"]==m["environment"]["harness_sha256"]
            assert item["split"]=="final" and item["status"]=="complete" and item["correctness_pass"] is True and item["evidence_eligible"] is True
        assert r["weights_source"]==mainrow["weights_source"] and r["policy_config"]==mainrow["policy_config"]==m["policy"]
        for branch_field,full_field in (("current_map","current_map"),("proposed_map","proposed_map"),("current_counts","current_counts")):
            assert r[branch_field]==decision[full_field], (k,branch_field)
        assert r["b3_would_allow"]==decision["allowed"] and r["b3_reason"]==decision["reason"]
        w=m["trace"]["window_steps"]; end=2*w+k[-1]+1; stop=end+d["horizon_steps"]
        fids,fw,fh=tape(mainrow); bids,bw,bh=tape(r)
        assert len(fids)>=stop and len(bids)>=stop
        assert np.array_equal(fids[:stop],bids[:stop]) and np.array_equal(fw[:stop],bw[:stop]) and np.array_equal(fh,bh)
        for field,lo,hi in (("current_counts",end-w,end),("previous_counts",end-2*w,end-w)):
            expected=np.bincount(fids[lo:hi].ravel(),minlength=m["trace"]["num_experts"])*r["world_size"]
            assert np.array_equal(expected,r[field]),(k,field)
        assert r["future_steps"]==d["horizon_steps"] and r["future_rebalancing_enabled"] is False
        assert r["branch_order"]==expected_orders[k]==[arm["branch"] for arm in r["branches"]]
        assert len(r["branches"])==2 and {arm["branch"] for arm in r["branches"]}=={"allow","skip"}
        for arm in r["branches"]:
            assert arm["correctness_pass"] is True and arm["future_steps"]==d["horizon_steps"]
            for field in ("source_map_sha256","candidate_map_sha256","source_weights_sha256","trace_sha256"):
                assert arm[field]==r[field] and len(r[field])==64
        checked.append({"state":k,"branch_order":r["branch_order"],"full_B3_snapshot_matches":True,
            "raw_prefix_and_future_routes_match":True,"hidden_inputs_match":True,"exact_previous_window_verified":True,
            "tape_fingerprints_identical":r["trace_sha256"]==mainrow["trace_sha256"]})
    assert seen==set(keys) and sum(r["branch_order"][0]=="allow" for r in rows)==4
    identical_history = []
    for batch in d["batch_tokens"]:
        burst = next(r for r in rows if r["regime"]=="burst" and r["trace_variant"]=="core" and r["batch_tokens"]==batch)
        persistent = next(r for r in rows if r["regime"]=="persistent" and r["trace_variant"]=="core" and r["batch_tokens"]==batch)
        for field in ("seed","repetition","decision_step","current_counts","previous_counts","current_map","proposed_map",
                      "source_map_sha256","candidate_map_sha256","source_weights_sha256","b3_would_allow","b3_reason","gate_decision"):
            assert burst[field]==persistent[field], (batch,field,"identical-history check")
        ids_b,weights_b,hidden_b=tape(burst); ids_p,weights_p,hidden_p=tape(persistent)
        assert np.array_equal(hidden_b,hidden_p)
        config=burst["trace_config"]; prefix=config["warmup_steps"]
        decision_end=prefix+burst["decision_step"]+1
        burst_end=prefix+config["phase_steps"]+config["hotspot_lifetime_steps"]
        stop=decision_end+burst["future_steps"]
        assert decision_end<burst_end<stop
        assert np.array_equal(ids_b[:burst_end],ids_p[:burst_end]) and np.array_equal(weights_b[:burst_end],weights_p[:burst_end])
        different=np.any(ids_b[decision_end:stop]!=ids_p[decision_end:stop],axis=(1,2)) | np.any(weights_b[decision_end:stop]!=weights_p[decision_end:stop],axis=(1,2))
        assert different.any()
        first_change=decision_end+int(np.flatnonzero(different)[0])
        assert first_change>=burst_end
        def payback(record):
            arms={x["branch"]:x for x in record["branches"]}
            return 1000*(arms["skip"]["elapsed_seconds"]-arms["allow"]["elapsed_seconds"])
        identical_history.append({"batch_tokens":batch,"seed":burst["seed"],"decision_step":burst["decision_step"],
            "identical_past_routes_weights_windows_source_and_candidate":True,"identical_gate_decision":True,
            "same_future_steps_before_divergence":first_change-decision_end,
            "first_different_future_measured_step":first_change-prefix,
            "burst_payback_ms":payback(burst),"persistent_payback_ms":payback(persistent),
            "interpretation":"The past observations do not distinguish these different futures. Payback signs are measured, not assumed or required by this check."})
    print(json.dumps({"status":"verified","pairs":8,"fixed_horizon":d["horizon_steps"],"allow_first":4,"skip_first":4,
        "rows":checked,"identical_history_different_future":identical_history,
        "scope":"Raw replay arrays and actual full-policy B3 proposal states checked; numerical/weight equality passes remain recorded runtime attestations."},indent=2))


if __name__=="__main__":main()

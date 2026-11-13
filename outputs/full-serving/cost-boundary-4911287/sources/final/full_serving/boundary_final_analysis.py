"""Final-cell audits from raw records; no speed claim without all six cells."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from full_serving.boundary_final_contract import TOTALS, TREATMENTS, verify_launch, validate_targets
from full_serving.sustained_analysis import raw_phase, runtime, migrations
from full_serving.proposal_analysis import gate_audit, gate_phase_evidence
from full_serving.report_pilots import metric_deltas
from full_serving.custom_report import phase_balance


def readrows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def arithmetic(records):
    assert len(records) > 0 and all(r["success"] for r in records)
    def distribution(values):
        assert all(math.isfinite(x) and x >= 0 for x in values)
        ordered = sorted(values)
        position = (len(ordered)-1)*.95
        lower = int(position)
        upper = min(lower+1, len(ordered)-1)
        return dict(mean=math.fsum(values)/len(values),
                    p95=ordered[lower]+(ordered[upper]-ordered[lower])*(position-lower))
    response = [(r["end_ns"]-r["scheduled_ns"])/1e6 for r in records]
    assert all(r["first_content_ns"] is not None for r in records)
    ttft = [(r["first_content_ns"]-r["scheduled_ns"])/1e6 for r in records]
    return dict(response_ms=distribution(response), ttft_ms=distribution(ttft),
                output_tokens=sum(r["usage"]["completion_tokens"] for r in records),
                ceiling_hits=sum(r["usage"]["completion_tokens"] == 8192 for r in records),
                requests=len(records), failed_requests=0)


def noop_evidence(directory, expected):
    directory=Path(directory)
    files=list(directory.glob("noop-commit-rank*.jsonl"))
    if not expected:
        assert not files
        return None
    # No proposal can legitimately mean no no-op log. Gate logs decide coverage.
    peers=[];cross={}
    for rank in range(8):
        gate=readrows(directory/f"gate-events-rank{rank}.jsonl")
        decisions={r["step"]:r for r in gate if r["kind"]=="decision"}
        path=directory/f"noop-commit-rank{rank}.jsonl"
        rr=readrows(path) if path.exists() else []
        assert all(r["rank"]==rank for r in rr)
        assert not any(r["kind"] in ("map_shadow_passed","weight_shadow_passed") for r in rr)
        proofs=[r for r in rr if r["kind"]=="unchanged_proof"]
        assert len(proofs)==len(decisions)
        eligible={}
        for proof in proofs:
            step=proof["cycle_step"];d=decisions[step]
            assert proof["shadow"] is False and d["rank_agreement"]
            layers=tuple(i for i,(old,new) in enumerate(zip(d["old_map"],d["admitted_map"])) if old==new)
            assert tuple(proof["unchanged_layers"])==layers and step not in eligible
            eligible[step]=set(layers)
            fingerprint=(d["digest"],layers)
            assert cross.setdefault(step,fingerprint)==fingerprint
        elided=[(r["cycle_step"],r["layer"]) for r in rr if r["kind"]=="map_commit_elided"]
        assert len(set(elided))==len(elided)
        assert all(step in eligible and layer in eligible[step] for step,layer in elided)
        moves=readrows(directory/f"migration-events-rank{rank}.jsonl")
        enqueued={(r["cycle_step"],r["layer"]) for r in moves if r["kind"]=="commit_enqueued"}
        completed={(r["cycle_step"],r["layer"]) for r in moves if r["kind"]=="commit_completed"}
        assert completed <= enqueued and set(elided) <= enqueued
        peers.append(dict(rank=rank,proposal_cycles=len(decisions),
                          eligible_layer_opportunities=sum(map(len,eligible.values())),
                          elided_map_commits=len(elided),observed_elided_commits=len(set(elided)&completed),
                          censored_elided_commits=len(set(elided)-completed),
                          unelided_eligible_layer_opportunities=sum(map(len,eligible.values()))-len(elided)))
    assert len({p["proposal_cycles"] for p in peers})==1
    return dict(ranks=peers,scope="All phases; eligibility is not GPU completion. No observed completion is inferred from drain.")


def audit(root,dataset,treatment):
    root=Path(root);arm,fix=TREATMENTS[treatment]
    status=json.loads((root/"status.json").read_text())
    assert status["state"]=="comparison_completed" and status["mode"]=="matched"
    assert status["completed_sessions"]==[dataset+"/"+arm] and status["pilot_only"] is False
    data=json.loads((root/"data-manifest.json").read_text())
    validate_targets(root)
    directory=root/dataset/arm
    for role in ("primary","worker"):
        verify_launch(json.loads((directory/f"effective-launch-{role}.json").read_text()),treatment)
    for pattern in ("boundary-host-rank*.jsonl","boundary-shapes-rank*.jsonl","resident-startup-diagnostic*"):
        assert not list(directory.glob(pattern)),pattern
    identity=runtime(directory,arm)
    phases={}
    for phase in ("warmup-00","warmup-01","measured"):
        item=data["datasets"][dataset]["items"]["measured" if phase=="measured" else "warmup"]
        targets=json.loads((root/(dataset+"-"+phase+"-lengths.json")).read_text())
        calendar=json.loads((root/(dataset+"-calendar.json")).read_text()) if phase=="measured" else None
        checked,records=raw_phase(directory/(phase+".jsonl"),item,calendar=calendar,targets=targets)
        assert checked["header"]["http_force_close"] and checked["header"]["arm"]==arm
        independent=arithmetic(records)
        summary=checked["summary"]
        for local,remote in (("response_ms","response_from_schedule_ms"),("ttft_ms","ttft_from_schedule_ms")):
            for metric in ("mean","p95"):
                assert math.isclose(independent[local][metric],summary[remote][metric],rel_tol=1e-12,abs_tol=1e-9)
        phases[phase]=dict(arithmetic=independent,summary=summary,timing=checked["timing"])
    assert phases["measured"]["arithmetic"]["output_tokens"]==TOTALS[dataset]
    native=metric_deltas(directory/"measured-metrics-before.txt",directory/"measured-metrics-after.txt")
    balance=phase_balance(native,request_count=512);assert balance["available"]
    report=dict(status="passed",dataset=dataset,treatment=treatment,runtime=identity,
                phases=phases,native_metrics=native,phase_balance=balance,
                no_op=noop_evidence(directory,fix=="noop_maps"),
                limitations="One matched group, descriptive only. Equal counts do not imply equal generated tokens/routes/quality. No critical-path or causal individual-migration payback inference.")
    if arm=="b4":
        for rank in range(8):
            assert not any(r["kind"]=="history_shadow_passed" for r in readrows(directory/f"gate-events-rank{rank}.jsonl"))
        report["migrations"]=migrations(directory)
        report["gate"]=gate_audit(directory,"b4")
        report["gate"]["phase_evidence"]=gate_phase_evidence(directory)
        report["gate"]["phase_evidence"]["scope"]=report["gate"]["phase_evidence"]["scope"].replace("kh074","kh021")
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);p.add_argument("--dataset",choices=tuple(TOTALS),required=True)
    p.add_argument("--treatment",choices=tuple(TREATMENTS),required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();result=audit(a.root,a.dataset,a.treatment)
    with a.output.open("x") as f:f.write(json.dumps(result,indent=2)+"\n")

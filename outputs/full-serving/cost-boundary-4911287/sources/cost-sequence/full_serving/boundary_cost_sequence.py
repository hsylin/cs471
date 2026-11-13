"""Sequential sustained cost controls. Orchestration only; frozen runtime is untouched."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

BASE = Path("/work/cs-471/skewbench-hlin/full-serving")
SOURCE = "c05186025dc152f103f61a7c2703b65fa74f3afe"
PROTOCOL_SHA = "ea76f5baf684fb50952ee6f456ccd59a484c003d95f2357f9c671e066bd573e6"
DEPLOY = BASE / "deployments/cost-boundary-4911287-controls-v2"
PROTOCOL = Path("results/full-serving/cost-boundary-4911287/controls-v2/protocol.json")
REFERENCE = BASE / "full-sustained-512-20261111-v3/natural/results"
PYTHON = "/work/cs-471/skewbench-hlin/venv/bin/python"
ROOT = BASE / "cost-boundary-4911287"
CONTROLS = ("fixed32_idle", "fixed32_record", "fixed32_history", "native_hash",
            "static_home", "resident_charged", "b4_gate_only", "b4_observed")
TOTALS = {"longbench": 75246, "numinamath": 741753}
LIMITS = [
    "Diagnostic interventions; not the final OFF/B4/fix comparison.",
    "One group per dataset: descriptive ratios only, no confidence intervals or significance.",
    "Matched counts do not fix generated tokens, routes, quality, or active KV state.",
    "Host and collective durations are not bandwidth or exposed GPU critical-path evidence.",
    "No observed commit is inferred from drain; missing observations remain censored.",
    "Contrasts include the stated interventions and run-to-run variation; not additive causal cost decomposition."
]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text())

def write_new(path, value):
    with Path(path).open("x") as f: f.write(json.dumps(value, indent=2)+"\n")

def plan(spec):
    assert spec["arms"] == list(CONTROLS)
    assert spec["orders"]["longbench"] == list(CONTROLS)
    assert spec["orders"]["numinamath"] == list(reversed(CONTROLS))
    return [(ds, arm) for ds in TOTALS for arm in spec["orders"][ds]]

def verify_admission(directory, protocol_sha=PROTOCOL_SHA, source=SOURCE):
    directory = Path(directory)
    assert (directory/"completed.txt").read_text().strip() == "sequence_exit=0"
    admission = read(directory/"admission.json")
    assert admission["status"] == "passed"
    assert admission["controls"] == list(CONTROLS)
    assert admission["source_revision"] == source and admission["protocol_sha256"] == protocol_sha
    assert set(admission["audit_sha256"]) == set(CONTROLS)
    for arm in CONTROLS:
        p = directory/arm/"audit.json"; audit = read(p)
        assert sha(p) == admission["audit_sha256"][arm]
        assert audit["status"] == "passed" and audit["pilot"] and audit["control"] == arm
    return admission

def verify_review(review, admission):
    assert review["status"] == "passed" and review["scope"] == "cost_controls_main_admission"
    assert review["runtime_source"] == SOURCE and review["protocol_sha256"] == PROTOCOL_SHA
    assert review["admission_sha256"] == sha(Path(admission)/"admission.json")
    assert review["all_eight_pilots_reviewed"] is True
    assert review["main_measurements_used_for_admission"] is False

def verify_deployment(deploy=DEPLOY):
    deploy = Path(deploy)
    assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=deploy,text=True).strip() == SOURCE
    assert sha(deploy/PROTOCOL) == PROTOCOL_SHA
    spec = read(deploy/PROTOCOL)
    for name, digest in spec["files"].items(): assert sha(deploy/name) == digest, name
    assert sha(deploy/spec["data_manifest"]) == spec["data_manifest_sha256"]
    plan(spec)
    return spec

def verify_predecessor(directory):
    directory = Path(directory)
    assert (directory/"completed.txt").read_text().strip() == "sequence_exit=0"
    value = read(directory/"admission.json")
    assert value["status"] == "passed"
    # This gate only serializes resource use; transition evidence is reviewed separately.
    return {"admission_sha256": sha(directory/"admission.json"),
            "completed_sha256": sha(directory/"completed.txt")}

def arithmetic(records):
    assert records and all(r["success"] for r in records)
    def dist(values):
        assert all(math.isfinite(x) and x >= 0 for x in values)
        values = sorted(values); position = (len(values)-1)*.95
        lower = int(position); upper = min(lower+1,len(values)-1)
        return {"mean":math.fsum(values)/len(values),
                "p95":values[lower]+(position-lower)*(values[upper]-values[lower])}
    assert all(r["first_content_ns"] is not None for r in records)
    return {"requests":len(records), "failed_requests":0,
            "response_ms":dist([(r["end_ns"]-r["scheduled_ns"])/1e6 for r in records]),
            "ttft_ms":dist([(r["first_content_ns"]-r["scheduled_ns"])/1e6 for r in records]),
            "output_tokens":sum(r["usage"]["completion_tokens"] for r in records),
            "ceiling_hits":sum(r["usage"]["completion_tokens"]==8192 for r in records)}

def verify_arithmetic(value, summary):
    for local, remote in (("response_ms","response_from_schedule_ms"),("ttft_ms","ttft_from_schedule_ms")):
        for metric in ("mean","p95"):
            assert math.isclose(value[local][metric], summary[remote][metric], rel_tol=1e-12, abs_tol=1e-9)

def controller_command(run, dataset, arm, admission, deploy=DEPLOY):
    run=Path(run); cell=run/(dataset+"-"+arm)
    return [str(x) for x in [
        "srun","--jobid=4911287","--mpi=none","--exact","--nodes=1","--ntasks=1",
        "--nodelist=kh021","--cpus-per-task=4","--gpus=0","--mem=16G","--input=none",
        "--output="+str(cell/"controller.log"),"--error="+str(cell/"controller.err"),
        PYTHON,"-m","full_serving.boundary_control_experiment","run","--protocol",deploy/PROTOCOL,
        "--private-data","/work/cs-471/skewbench-hlin/private/sustained-512-v1-inputs",
        "--snapshot","/work/cs-471/skewbench-hlin/cache/huggingface/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
        "--output",cell/"results","--private-output",
        Path("/work/cs-471/skewbench-hlin/private/cost-boundary-4911287")/run.name/(dataset+"-"+arm),
        "--source-revision",SOURCE,"--mode","matched","--reference",REFERENCE,
        "--pilot-admission",Path(admission)/"admission.json","--dataset",dataset,
        "--arm",arm,"--host-probe","off","--map-commit","native"]]

def audit_cell(run, dataset, arm, deploy=DEPLOY):
    cell=Path(run)/(dataset+"-"+arm); out=cell/"audit.json"
    assert not out.exists()
    subprocess.run([PYTHON,"-m","full_serving.boundary_control_analysis",str(cell/"results"),
                    "--control",arm,"--dataset",dataset,"--output",str(out)],cwd=deploy,check=True)
    result=read(out); assert result["status"]=="passed" and not result["pilot"]

def runtime_signature(identity, checkpoint, checkpoint_sha):
    assert checkpoint["complete"] is True and checkpoint["kind"]=="full_checkpoint_sha256"
    assert len(identity)==2
    assert all(r["checkpoint_manifest_sha256"]==checkpoint_sha for r in identity)
    # The manifest includes a per-check timestamp. Verify each exact digest first,
    # then compare every content field except that timestamp across sessions.
    return json.loads(json.dumps({"runtime":[{k:v for k,v in role.items() if k!="checkpoint_manifest_sha256"} for role in identity],
            "checkpoint":{k:v for k,v in checkpoint.items() if k!="checked_at"}}))

def cell_summary(run, dataset, arm, spec, deploy=DEPLOY):
    cell=Path(run)/(dataset+"-"+arm); root=cell/"results"; directory=root/dataset/arm
    audit=read(cell/"audit.json")
    assert audit["status"]=="passed" and not audit["pilot"]
    assert audit["control"]==arm and audit["dataset"]==dataset
    assert read(root/"status.json")["state"]=="comparison_completed"
    assert sha(root/"protocol.json")==PROTOCOL_SHA
    assert sha(root/"data-manifest.json")==spec["data_manifest_sha256"]
    source=read(root/"controller-source.json")
    assert source["declared_git_revision"]==SOURCE
    for name,digest in source["files"].items():
        blob=subprocess.check_output(["git","show",SOURCE+":"+name],cwd=deploy)
        assert hashlib.sha256(blob).hexdigest()==digest==sha(deploy/name),name
    phases={}; fingerprint=None
    for phase in ("warmup-00","warmup-01","measured"):
        rows=[json.loads(x) for x in (directory/(phase+".jsonl")).read_text().splitlines()]
        records=sorted([r for r in rows if r.get("type")=="request"],key=lambda r:r["request_index"])
        checked=audit["phases"][phase]; value=arithmetic(records)
        verify_arithmetic(value,checked["summary"])
        assert value["requests"]==(512 if phase=="measured" else 8)
        targets=read(root/(dataset+"-"+phase+"-lengths.json"))
        assert set(targets)=={r["request_id"] for r in records}
        assert all(targets[r["request_id"]]==r["usage"]["completion_tokens"] for r in records)
        phases[phase]={"arithmetic":value, "elapsed_s":checked["timing"]["completed_elapsed_ns"]/1e9,
                       "offered_horizon_s":checked["timing"]["offered_horizon_ns"]/1e9,
                       "drain_s":checked["timing"]["drain_ns"]/1e9}
        if phase=="measured":
            assert value["output_tokens"]==TOTALS[dataset]
            fingerprint=[(r["request_id"],r["input_sha256"],targets[r["request_id"]],
                          r["scheduled_ns"]-checked["timing"]["epoch_monotonic_ns"]) for r in records]
    seal=read(root/"matched-targets-seal.json"); assert seal["frozen_before_server_start"] is True
    for name,digest in seal["files"].items(): assert sha(root/name)==digest
    # Preserve hashes of BOTH datasets' frozen warmup/measured targets and selected calendar.
    targets=dict(seal["files"]); targets["calendar"]=sha(root/(dataset+"-calendar.json"))
    assert audit["phase_balance"]["available"]
    result={"dataset":dataset,"control":arm,"phases":phases,"native_phase_balance":audit["phase_balance"],
            "runtime":audit["runtime"],"audit_sha256":sha(cell/"audit.json"),
            "runtime_signature":runtime_signature(audit["runtime"],read(root/"checkpoint-verified.json"),
                                                   sha(root/"checkpoint-verified.json"))}
    for pattern in ("boundary-host-rank*.jsonl","boundary-shapes-rank*.jsonl","resident-startup-diagnostic*"):
        assert not list(directory.glob(pattern)),pattern
    if arm.startswith("b4_"):
        from full_serving.proposal_analysis import gate_phase_evidence
        evidence=gate_phase_evidence(directory)
        evidence["scope"]=evidence["scope"].replace("kh074","kh021")
        result["gate_phase_evidence"]=evidence
        for rank in range(8):
            gate=[json.loads(x) for x in (directory/f"gate-events-rank{rank}.jsonl").read_text().splitlines()]
            assert not any(r["kind"]=="history_shadow_passed" for r in gate)
    for key in ("gate","additional_observer"):
        if key in audit: result[key]=audit[key]
    return result, fingerprint, targets

def ratio(left,right):
    assert left>=0 and right>0
    return {"ratio":left/right,"change_percent":100*(left/right-1)}

def combine(cells, spec):
    assert set(cells)=={(ds,arm) for ds,arm in plan(spec)}
    identities=[cells[key]["runtime_signature"] for key in plan(spec)]
    assert all(identity==identities[0] for identity in identities), "Hardware/runtime changed between cells"
    assert [r["hostname"] for r in identities[0]["runtime"]]==["kh021","kh023"]
    devices=[d for r in identities[0]["runtime"] for d in r["devices"]]
    assert len(devices)==8 and len({d["uuid"] for d in devices})==8
    assert all(d["name"]=="NVIDIA H100" for d in devices)
    datasets={}
    for dataset in TOTALS:
        peers={arm:cells[(dataset,arm)] for arm in CONTROLS}
        contrasts={}
        for label,interpretation in spec["diagnostic"]["contrasts"].items():
            numerator,denominator=label.split("/")
            left=peers[numerator]["phases"]["measured"]["arithmetic"]
            right=peers[denominator]["phases"]["measured"]["arithmetic"]
            contrasts[label]={"intervention":interpretation,"metrics":{
                metric:{stat:ratio(left[metric][stat],right[metric][stat]) for stat in ("mean","p95")}
                for metric in ("response_ms","ttft_ms")}}
        datasets[dataset]={"cells":peers,"contrasts":contrasts}
    return {"status":"passed","scope":"cost_controls_main","sessions":16,"measured_requests":8192,
            "runtime_source":SOURCE,"protocol_sha256":PROTOCOL_SHA,"datasets":datasets,"limitations":LIMITS}

def raw_hashes(run):
    # PRIVATE receipt only. Logs and raw records must never be published.
    return {str(p.relative_to(run)):sha(p) for p in sorted(Path(run).glob("*/results/**/*")) if p.is_file()}

def report(run, output, deploy=DEPLOY):
    run=Path(run); output=Path(output); assert not output.exists();output.mkdir()
    spec=verify_deployment(deploy); before=raw_hashes(run); cells={}; fingerprints={};targets={}
    for dataset,arm in plan(spec):
        value, fingerprint, lengths=cell_summary(run,dataset,arm,spec,deploy)
        assert fingerprints.setdefault(dataset,fingerprint)==fingerprint
        assert targets.setdefault(dataset,lengths)==lengths
        cells[(dataset,arm)]=value
    combined=combine(cells,spec)
    time.sleep(2)
    after=raw_hashes(run);assert before==after
    write_new(output/"stable-records-private.json",{"status":"passed","files":after,"private":True})
    write_new(output/"comparison-review-required.json",combined)
    write_new(output/"completion.json",{"status":"passed","sessions":16,"measured_requests":8192,
        "runtime_source":SOURCE,"protocol_sha256":PROTOCOL_SHA,"publication_review_required":True,
        "report_sha256":sha(output/"comparison-review-required.json"),
        "stable_records_private_sha256":sha(output/"stable-records-private.json")})

def execute(args):
    assert not args.run.exists(), "Use a NEW run directory; completed GPU work is never rerun."
    assert not any(k.startswith(("FULL_SERVING_PAYBACK","FULL_SERVING_BOUNDARY","CS471_")) for k in os.environ)
    spec=verify_deployment()
    verify_admission(args.admission); review=read(args.review);verify_review(review,args.admission)
    args.run.mkdir()
    lock=(args.run/"launcher.lock").open("w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    code=1
    try:
        write_new(args.run/"plan.json",{"cells":plan(spec),"source":SOURCE,
            "protocol_sha256":PROTOCOL_SHA,"review_sha256":sha(args.review),
            "predecessor":str(args.predecessor),"profiles":False})
        (args.run/"active-stage.txt").write_text("waiting_existing_transition_admission\n")
        while not (args.predecessor/"completed.txt").exists():time.sleep(30)
        write_new(args.run/"predecessor-receipt.json",verify_predecessor(args.predecessor))
        for dataset,arm in plan(spec):
            verify_deployment()
            cell=args.run/(dataset+"-"+arm);cell.mkdir()
            (args.run/"active-stage.txt").write_text(dataset+"/"+arm+"\n")
            cmd=controller_command(args.run,dataset,arm,args.admission)
            write_new(cell/"controller-command.json",cmd)
            subprocess.run(cmd,cwd=DEPLOY,check=True)
            audit_cell(args.run,dataset,arm)
        report(args.run,args.run/"reconstruction-v1")
        (args.run/"active-stage.txt").write_text("completed_pending_interpretation_review\n")
        code=0
    finally:
        (args.run/"completed.txt").write_text("sequence_exit="+str(code)+"\n")

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest="action",required=True)
    run=sub.add_parser("run");run.add_argument("--run",type=Path,required=True)
    run.add_argument("--review",type=Path,required=True)
    run.add_argument("--admission",type=Path,default=ROOT/"controls-admission-v2")
    run.add_argument("--predecessor",type=Path,default=ROOT/"transition-admission-v1")
    analysis=sub.add_parser("report");analysis.add_argument("--run",type=Path,required=True)
    analysis.add_argument("--output",type=Path,required=True)
    sub.add_parser("plan")
    args=p.parse_args()
    if args.action=="run":execute(args)
    elif args.action=="report":report(args.run,args.output)
    else:print(json.dumps(plan(verify_deployment())))
if __name__=="__main__": main()

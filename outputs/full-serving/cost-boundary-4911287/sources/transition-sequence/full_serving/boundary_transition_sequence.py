"""Six matched frozen-candidate diagnostics, serialized after existing cost controls."""
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import time
from full_serving.boundary_sequence_common import (
    sha,read,write_new,arithmetic,verify_arithmetic,runtime_signature,ratio,raw_hashes)

BASE=Path("/work/cs-471/skewbench-hlin/full-serving")
ROOT=BASE/"cost-boundary-4911287"
DEPLOY=BASE/"deployments/cost-boundary-4911287-transition-v1"
SOURCE="31d2d651f31c3443bf2e36b3277ffc7c01da468f"
PROTOCOL=Path("results/full-serving/cost-boundary-4911287/transition-v1/protocol.json")
PROTOCOL_SHA="61ec42f0ce1b0974aa520ea6fecd35e6b87d8ab032abd01dff6ec0ad9fb50beb"
CANDIDATE_SHA="87bfb97a85ac9eb770e9be7c54d00014293fb4e572d7c816f8f13a3a4a30634d"
MODES=("keep","preinstalled","charged")
TOTALS={"longbench":75246,"numinamath":741753}
REFERENCE=BASE/"full-sustained-512-20261111-v3/natural/results"
PYTHON="/work/cs-471/skewbench-hlin/venv/bin/python"
LIMITS=[
    "Frozen pilot candidate with native B2 recording, window, readiness, transfer, commit and observer; not online B4 or final selected fix.",
    "Same first changed pilot candidate for both datasets; no fastest or best-candidate selection.",
    "PREINSTALLED changes preceding warmup/KV state and excludes startup migration from request latency.",
    "Equal output counts do not imply equal tokens, routes, batches, reduction order, active KV or quality.",
    "One group per dataset/mode supports descriptive ratios only; no confidence intervals or significance.",
    "CPU/collective/GPU-copy intervals do not isolate exposed serving critical path; drain is not commit proof.",
    "Independent sessions do not establish causal individual-migration payback even when observed predecision checks agree."
]

def plan(spec):
    assert spec["arms"]==list(MODES)
    assert spec["orders"]=={"longbench":list(MODES),"numinamath":list(reversed(MODES))}
    return [(ds,mode) for ds in TOTALS for mode in spec["orders"][ds]]

def verify_deployment(deploy=DEPLOY):
    assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=deploy,text=True).strip()==SOURCE
    assert sha(deploy/PROTOCOL)==PROTOCOL_SHA
    spec=read(deploy/PROTOCOL);plan(spec)
    assert spec["candidate_sha256"]==CANDIDATE_SHA==sha(deploy/spec["candidate"])
    for name,digest in spec["files"].items():assert sha(deploy/name)==digest,name
    assert sha(deploy/spec["data_manifest"])==spec["data_manifest_sha256"]
    return spec

def verify_admission(directory):
    directory=Path(directory)
    assert (directory/"completed.txt").read_text().strip()=="sequence_exit=0"
    admission=read(directory/"admission.json")
    assert admission["status"]=="passed" and admission["controls"]==list(MODES)
    assert admission["source_revision"]==SOURCE and admission["protocol_sha256"]==PROTOCOL_SHA
    assert admission["candidate_sha256"]==CANDIDATE_SHA
    assert set(admission["audit_sha256"])==set(MODES)
    for mode,digest in admission["audit_sha256"].items():
        path=directory/mode/"audit.json";value=read(path)
        assert sha(path)==digest and value["status"]=="passed" and value["pilot"] and value["control"]==mode
        assert value["candidate_sha256"]==CANDIDATE_SHA
        assert len(value["ranks"])==8
        if mode=="charged":
            assert all(r["full_weight_shadow_layers"]==48 for r in value["ranks"])
    path=directory/"pilot-logprob-diagnostic.json"
    assert sha(path)==admission["logprob_diagnostic_sha256"] and read(path)["status"]=="passed"
    return admission

def verify_review(review,admission):
    assert review["status"]=="passed" and review["scope"]=="transition_main_admission"
    assert review["runtime_source"]==SOURCE and review["protocol_sha256"]==PROTOCOL_SHA
    assert review["candidate_sha256"]==CANDIDATE_SHA
    assert review["admission_sha256"]==sha(Path(admission)/"admission.json")
    for key in ("all_three_pilots_reviewed","all_rank_graph_weight_metadata_reviewed",
                "charged_target_all48_observed_commits_reviewed","pilot_logprob_limitations_reviewed"):
        assert review[key] is True
    assert review["main_measurements_used_for_admission"] is False
    assert review["causal_payback_established"] is False

def verify_predecessor(directory):
    directory=Path(directory)
    assert (directory/"completed.txt").read_text().strip()=="sequence_exit=0"
    completion=directory/"reconstruction-v1/completion.json";value=read(completion)
    assert value["status"]=="passed" and value["sessions"]==16 and value["measured_requests"]==8192
    assert value["runtime_source"]=="c05186025dc152f103f61a7c2703b65fa74f3afe"
    assert sha(completion.parent/"comparison-review-required.json")==value["report_sha256"]
    assert sha(completion.parent/"stable-records-private.json")==value["stable_records_private_sha256"]
    return {"completion_sha256":sha(completion),"scope":"Resource dependency only; not automatic cost interpretation."}

def verify_serial_predecessor(directory):
    directory=Path(directory)
    assert (directory/"completed.txt").read_text().strip()=="sequence_exit=0"
    admission=read(directory/"admission.json")
    assert admission["status"]=="passed" and admission["sync_shape_probe_passed"] is True
    assert admission["source_revision"]=="620cbb0170c8ea275229092a88948356935c0686"
    assert admission["protocol_sha256"]=="733cac8b193afc81b861699d7492ce80e32d25b04a640c443c3dcba57a7ea15a"
    assert sha(directory/"pilot/audit.json")==admission["pilot_audit_sha256"]
    audits={}
    for label in ("pilot","longbench","numinamath"):
        path=directory/label/"audit.json";audit=read(path)
        assert audit["status"]=="passed" and audit["sync_shape_probe_passed"] is True
        assert audit["active_aligned_samples"]>=(1 if label=="pilot" else 8)
        assert audit["pilot"] is (label=="pilot")
        if label=="pilot":assert audit["explicit_empty_rank_observations"]>0
        else:assert audit["dataset"]==label
        root=directory/label/"results"
        assert sha(root/"protocol.json")==admission["protocol_sha256"]
        assert read(root/"controller-source.json")["declared_git_revision"]==admission["source_revision"]
        audits[label]=sha(path)
    assert read(directory/"stable-records-private.json")["private"] is True
    return {"status":"passed","admission_sha256":sha(directory/"admission.json"),"audit_sha256":audits,
            "scope":"Serialize resource use after aligned diagnostic; interpretation remains separate."}

def controller_command(run,dataset,mode,admission,deploy=DEPLOY):
    cell=Path(run)/(dataset+"-"+mode)
    return list(map(str,[
      "srun","--jobid=4911287","--mpi=none","--exact","--nodes=1","--ntasks=1","--nodelist=kh021",
      "--cpus-per-task=4","--gpus=0","--mem=16G","--input=none",
      "--output="+str(cell/"controller.log"),"--error="+str(cell/"controller.err"),
      PYTHON,"-m","full_serving.boundary_transition_experiment","run","--protocol",deploy/PROTOCOL,
      "--private-data","/work/cs-471/skewbench-hlin/private/sustained-512-v1-inputs",
      "--snapshot","/work/cs-471/skewbench-hlin/cache/huggingface/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
      "--output",cell/"results","--private-output",
      Path("/work/cs-471/skewbench-hlin/private/cost-boundary-4911287")/run.name/(dataset+"-"+mode),
      "--source-revision",SOURCE,"--mode","matched","--reference",REFERENCE,
      "--pilot-admission",Path(admission)/"admission.json","--dataset",dataset,"--arm",mode,
      "--host-probe","off","--map-commit","native"]))

def audit_cell(run,dataset,mode):
    cell=Path(run)/(dataset+"-"+mode)
    assert not (cell/"audit.json").exists()
    subprocess.run([PYTHON,"-m","full_serving.boundary_transition_analysis",str(cell/"results"),
                    "--control",mode,"--dataset",dataset,"--output",str(cell/"audit.json")],cwd=DEPLOY,check=True)

def cell_summary(run,dataset,mode,spec,deploy=DEPLOY):
    import json
    cell=Path(run)/(dataset+"-"+mode);root=cell/"results";directory=root/dataset/mode
    audit=read(cell/"audit.json");status=read(root/"status.json")
    assert audit["status"]=="passed" and not audit["pilot"] and audit["control"]==mode and audit["dataset"]==dataset
    assert status["state"]=="comparison_completed" and status["mode"]=="matched"
    assert status["completed_sessions"]==[dataset+"/"+mode] and not status["pilot_only"]
    assert sha(root/"protocol.json")==PROTOCOL_SHA and sha(root/"candidate.json")==CANDIDATE_SHA
    assert sha(root/"data-manifest.json")==spec["data_manifest_sha256"]
    source=read(root/"controller-source.json");assert source["declared_git_revision"]==SOURCE
    for name,digest in source["files"].items():
        blob=subprocess.check_output(["git","show",SOURCE+":"+name],cwd=deploy)
        assert hashlib.sha256(blob).hexdigest()==digest==sha(deploy/name),name
    phases={};fingerprint=None
    for phase in ("warmup-00","warmup-01","measured"):
        rows=[json.loads(x) for x in (directory/(phase+".jsonl")).read_text().splitlines()]
        rr=sorted([r for r in rows if r.get("type")=="request"],key=lambda r:r["request_index"])
        checked=audit["phases"][phase];value=arithmetic(rr);verify_arithmetic(value,checked["summary"])
        assert value["requests"]==(512 if phase=="measured" else 8)
        targets=read(root/(dataset+"-"+phase+"-lengths.json"))
        assert set(targets)=={r["request_id"] for r in rr}
        assert all(targets[r["request_id"]]==r["usage"]["completion_tokens"] for r in rr)
        timing=checked["timing"]
        phases[phase]={"arithmetic":value,"elapsed_s":timing["completed_elapsed_ns"]/1e9,
                       "offered_horizon_s":timing["offered_horizon_ns"]/1e9,"drain_s":timing["drain_ns"]/1e9}
        if phase=="measured":
            assert value["output_tokens"]==TOTALS[dataset]
            fingerprint=[(r["request_id"],r["input_sha256"],targets[r["request_id"]],
                          r["scheduled_ns"]-timing["epoch_monotonic_ns"]) for r in rr]
    seal=read(root/"matched-targets-seal.json");assert seal["frozen_before_server_start"]
    for name,digest in seal["files"].items():assert sha(root/name)==digest
    target_hashes=dict(seal["files"]);target_hashes["calendar"]=sha(root/(dataset+"-calendar.json"))
    assert audit["phase_balance"]["available"]
    for pattern in ("boundary-host-rank*.jsonl","boundary-shapes-rank*.jsonl","resident-startup-diagnostic*"):
        assert not list(directory.glob(pattern)),pattern
    result={"dataset":dataset,"control":mode,"phases":phases,"native_phase_balance":audit["phase_balance"],
            "runtime":audit["runtime"],"ranks":audit["ranks"],"migrations":audit["migrations"],
            "candidate_sha256":CANDIDATE_SHA,"audit_sha256":sha(cell/"audit.json"),
            "runtime_signature":runtime_signature(audit["runtime"],read(root/"checkpoint-verified.json"),sha(root/"checkpoint-verified.json"))}
    return result,fingerprint,target_hashes

def combine(cells,comparisons,spec):
    assert set(cells)==set(plan(spec)) and set(comparisons)==set(TOTALS)
    signatures=[cells[key]["runtime_signature"] for key in plan(spec)]
    assert all(s==signatures[0] for s in signatures)
    roles=signatures[0]["runtime"];assert [r["hostname"] for r in roles]==["kh021","kh023"]
    devices=[d for r in roles for d in r["devices"]]
    assert len(devices)==8 and len({d["uuid"] for d in devices})==8 and all(d["name"]=="NVIDIA H100" for d in devices)
    datasets={}
    meanings={"preinstalled/keep":"Candidate placement preinstalled before warmup; startup migration excluded, preceding state differs.",
              "charged/keep":"Frozen candidate with measured native transition versus retaining old placement, including online costs.",
              "charged/preinstalled":"Transition timing and warmup-state difference; not isolated migration overhead."}
    for ds in TOTALS:
        peers={m:cells[(ds,m)] for m in MODES};contrasts={}
        for label,meaning in meanings.items():
            a,b=label.split("/")
            left,right=(peers[x]["phases"]["measured"]["arithmetic"] for x in (a,b))
            contrasts[label]={"scope":meaning,"metrics":{
                metric:{stat:ratio(left[metric][stat],right[metric][stat]) for stat in ("mean","p95")}
                for metric in ("response_ms","ttft_ms")}}
        assert comparisons[ds]["status"]=="passed" and comparisons[ds]["causal_payback_established"] is False
        datasets[ds]={"cells":peers,"contrasts":contrasts,"predecision_comparability":comparisons[ds]}
    return {"status":"passed","scope":"frozen_candidate_transition_main","sessions":6,"measured_requests":3072,
            "runtime_source":SOURCE,"protocol_sha256":PROTOCOL_SHA,"candidate_sha256":CANDIDATE_SHA,
            "datasets":datasets,"causal_payback_established":False,"limitations":LIMITS}

def report(run,output):
    run,output=Path(run),Path(output);assert not output.exists();output.mkdir()
    spec=verify_deployment();before=raw_hashes(run);cells={};fingerprints={};targets={}
    for ds,mode in plan(spec):
        value,fingerprint,lengths=cell_summary(run,ds,mode,spec)
        assert fingerprints.setdefault(ds,fingerprint)==fingerprint
        assert targets.setdefault(ds,lengths)==lengths
        cells[(ds,mode)]=value
    comparisons={}
    for ds in TOTALS:
        out=output/(ds+"-predecision-comparison.json")
        subprocess.run([PYTHON,"-m","full_serving.boundary_transition_comparison","main",
                        "--keep",str(run/(ds+"-keep")/"results"),"--charged",str(run/(ds+"-charged")/"results"),
                        "--dataset",ds,"--output",str(out)],cwd=DEPLOY,check=True)
        comparisons[ds]=read(out)
    combined=combine(cells,comparisons,spec)
    time.sleep(2);after=raw_hashes(run);assert before==after
    write_new(output/"stable-records-private.json",{"status":"passed","private":True,"files":after})
    write_new(output/"comparison-review-required.json",combined)
    write_new(output/"completion.json",{"status":"passed","sessions":6,"measured_requests":3072,
        "runtime_source":SOURCE,"protocol_sha256":PROTOCOL_SHA,"candidate_sha256":CANDIDATE_SHA,
        "publication_review_required":True,"causal_payback_established":False,
        "report_sha256":sha(output/"comparison-review-required.json"),
        "stable_records_private_sha256":sha(output/"stable-records-private.json")})

def execute(args):
    assert not args.run.exists(),"Use a NEW directory; never repeat completed GPU work."
    assert not any(k.startswith(("FULL_SERVING_PAYBACK","FULL_SERVING_BOUNDARY","CS471_")) for k in os.environ)
    spec=verify_deployment();verify_admission(args.admission);verify_review(read(args.review),args.admission)
    args.run.mkdir();lock=(args.run/"launcher.lock").open("w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    code=1
    try:
        write_new(args.run/"plan.json",{"cells":plan(spec),"source":SOURCE,"protocol_sha256":PROTOCOL_SHA,
            "candidate_sha256":CANDIDATE_SHA,"review_sha256":sha(args.review),"predecessor":str(args.predecessor),"serial_predecessor":str(args.serial_predecessor)})
        (args.run/"active-stage.txt").write_text("waiting_existing_cost_main\n")
        while not (args.predecessor/"completed.txt").exists():time.sleep(30)
        write_new(args.run/"predecessor-receipt.json",verify_predecessor(args.predecessor))
        (args.run/"active-stage.txt").write_text("waiting_existing_sync_shapes\n")
        while not (args.serial_predecessor/"completed.txt").exists():time.sleep(30)
        write_new(args.run/"serial-predecessor-receipt.json",verify_serial_predecessor(args.serial_predecessor))
        for ds,mode in plan(spec):
            verify_deployment();cell=args.run/(ds+"-"+mode);cell.mkdir()
            (args.run/"active-stage.txt").write_text(ds+"/"+mode+"\n")
            cmd=controller_command(args.run,ds,mode,args.admission)
            write_new(cell/"controller-command.json",cmd)
            subprocess.run(cmd,cwd=DEPLOY,check=True);audit_cell(args.run,ds,mode)
        report(args.run,args.run/"reconstruction-v1")
        (args.run/"active-stage.txt").write_text("completed_pending_model_review\n");code=0
    finally:(args.run/"completed.txt").write_text("sequence_exit="+str(code)+"\n")

def main():
    import json
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest="action",required=True)
    a=sub.add_parser("run");a.add_argument("--run",type=Path,required=True);a.add_argument("--review",type=Path,required=True)
    a.add_argument("--admission",type=Path,default=ROOT/"transition-admission-v1")
    a.add_argument("--predecessor",type=Path,default=ROOT/"cost-main-v1")
    a.add_argument("--serial-predecessor",type=Path,default=ROOT/"sync-shapes-run-v1")
    a=sub.add_parser("report");a.add_argument("--run",type=Path,required=True);a.add_argument("--output",type=Path,required=True)
    sub.add_parser("plan");args=p.parse_args()
    if args.action=="run":execute(args)
    elif args.action=="report":report(args.run,args.output)
    else:print(json.dumps(plan(verify_deployment())))
if __name__=="__main__":main()

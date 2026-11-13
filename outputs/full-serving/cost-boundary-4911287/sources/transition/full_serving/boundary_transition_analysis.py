"""Audit frozen-candidate interventions; individual causal payback stays separate."""
import argparse
import json
from pathlib import Path
from full_serving.boundary_transition_contract import MODES, canonical, validate_spec
from full_serving.boundary_transition_experiment import server_arguments
from full_serving.sustained_analysis import raw_phase,migrations,logged_config,logged_replica_count
from full_serving.custom_runtime import runtime_identity,engine_configuration,errors_before_shutdown
from full_serving.report_pilots import metric_deltas
from full_serving.custom_report import phase_balance

def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def audit(root,control,pilot=False,dataset=None):
    root=Path(root);assert control in MODES
    spec=json.loads((root/"candidate.json").read_text());validate_spec(spec)
    protocol=json.loads((root/"protocol.json").read_text())
    status=json.loads((root/"status.json").read_text())
    assert status["state"]=="comparison_completed"
    assert status["completed_sessions"]==[("pilot" if pilot else dataset)+"/"+control]
    data=json.loads((root/"data-manifest.json").read_text())
    directory=root/("pilot" if pilot else dataset)/control
    texts={f"server-{n}.{s}":(directory/f"server-{n}.{s}").read_text() for n in range(2) for s in ("log","err")}
    assert errors_before_shutdown(texts)==0
    assert engine_configuration(texts["server-0.log"])==engine_configuration(texts["server-1.log"])
    cfg=logged_config(texts["server-0.log"]);assert cfg["enable_eplb"] and logged_replica_count(cfg)==32
    identity=[]
    for role in ("primary","worker"):
        identity.append(runtime_identity(json.loads((directory/f"preflight-{role}.json").read_text()),role))
        launch=json.loads((directory/f"effective-launch-{role}.json").read_text());cmd=launch["command"]
        assert launch["arm"]==launch["transition_mode"]==control and launch["plugin"]=="cs471_transition"
        assert launch["transition_shadow"] is pilot and launch["candidate_sha256"]==protocol["candidate_sha256"]
        assert cmd==["vllm",*server_arguments(control,cmd[2],cmd[cmd.index("--data-parallel-address")+1],role)]
    phases={}
    for ds in ("longbench","numinamath") if pilot else (dataset,):
        for phase in ("warmup-00","warmup-01","pilot") if pilot else ("warmup-00","warmup-01","measured"):
            label=ds+"-"+phase if pilot else phase
            item=data["datasets"][ds]["items"]["measured" if phase=="measured" else "warmup"]
            targets=None if pilot else json.loads((root/(ds+"-"+phase+"-lengths.json")).read_text())
            calendar=json.loads((root/(ds+"-calendar.json")).read_text()) if phase=="measured" else None
            value,_=raw_phase(directory/(label+".jsonl"),item,calendar=calendar,targets=targets)
            assert value["header"]["arm"]==control and value["header"]["http_force_close"]
            phases[label]=value
    peers=[];cross={};triggered_steps=[]
    for rank in range(8):
        rr=rows(directory/f"transition-rank{rank}.jsonl")
        assert all(r["rank"]==rank and r["mode"]==control for r in rr)
        bound=[r for r in rr if r["kind"]=="bound"];assert len(bound)==1
        b=bound[0];assert b["full_weight_slots_checked"]==48*160 and b["metadata_all_elements_checked"]
        assert b["startup_gpu_completion_observed"] and b["shadow"] is pilot
        assert b["startup_graph_reader_layers"]==(48 if pilot else 0)
        assert b["spec_sha256"]==protocol["candidate_sha256"]
        assert b["initial_map_sha256"]==spec["candidate_map_sha256" if control=="preinstalled" else "old_map_sha256"]
        proposals=[r for r in rr if r["kind"]=="proposal"]
        assert len({r["step"] for r in proposals})==len(proposals)
        for r in proposals:
            assert r["rank_agreement"]
            fp=tuple(r[k] for k in ("triggered","measured_marker_agreed","old_map_sha256","admitted_map_sha256","live_demand_sha256","changed_slots"))
            assert cross.setdefault(r["step"],fp)==fp
            if r["triggered"]:
                assert control=="charged" and r["measured_marker_agreed"]
                assert r["old_map_sha256"]==spec["old_map_sha256"]
                assert r["admitted_map_sha256"]==spec["candidate_map_sha256"]
                assert r["changed_slots"]==spec["changed_slots"]>0
        triggers=[r["step"] for r in proposals if r["triggered"]]
        assert len(triggers)==(1 if control=="charged" else 0),"Target opportunity absent or repeated; do not retune/retry for performance"
        triggered_steps.append(triggers)
        shadows=[r for r in rr if r["kind"]=="charged_weight_shadow_passed"]
        if pilot and control=="charged":
            assert len(shadows)==48 and {r["layer"] for r in shadows}==set(range(48))
            assert all(r["cycle_step"]==triggers[0] and r["full_local_experts_checked"]==20 and r["metadata_all_elements_checked"] and r["graph_reader_checked"] for r in shadows)
        else:assert not shadows
        peers.append(dict(rank=rank,proposals=len(proposals),trigger_steps=triggers,
                          full_weight_shadow_layers=len(shadows),startup_transition_wall_ns=b["startup_transition_wall_ns"]))
    assert all(t==triggered_steps[0] for t in triggered_steps)
    migration=migrations(directory)
    if pilot and control=="charged":
        target=[c for c in migration["cycles"] if c["step"]==triggered_steps[0][0]]
        assert len(target)==1 and target[0]["all_rank_completed_layers"]==48,"Pilot target commit completion not fully observed"
    result=dict(status="passed",control=control,pilot=pilot,dataset=dataset,phases=phases,runtime=identity,
                ranks=peers,migrations=migration,candidate_source_step=spec["source_step"],
                candidate_sha256=protocol["candidate_sha256"],
                limitations="Diagnostic frozen placement intervention with native B2 bookkeeping/observer, not online B4. Same pilot-frozen candidate for both datasets. PREINSTALLED changes warmup and KV state; equal lengths do not fix tokens/routes. Compare completed predecision prefix hashes and live demand/map hashes across KEEP/CHARGED separately; passing this audit alone does not establish causal payback.")
    if not pilot:
        delta=metric_deltas(directory/"measured-metrics-before.txt",directory/"measured-metrics-after.txt")
        result["phase_balance"]=phase_balance(delta,request_count=512)
        assert result["phase_balance"]["available"]
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);p.add_argument("--control",choices=MODES,required=True)
    p.add_argument("--pilot",action="store_true");p.add_argument("--dataset");p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    with a.output.open("x") as f:f.write(json.dumps(audit(a.root,a.control,a.pilot,a.dataset),indent=2)+"\n")

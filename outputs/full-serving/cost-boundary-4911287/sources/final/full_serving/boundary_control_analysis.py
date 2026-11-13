"""Audit every diagnostic request and rank; never label controls final policies."""
import argparse,json
from pathlib import Path
from full_serving.sustained_analysis import raw_phase,migrations,logged_config,logged_replica_count
from full_serving.custom_runtime import runtime_identity,engine_configuration,errors_before_shutdown
from full_serving.proposal_analysis import gate_audit
from full_serving.report_pilots import metric_deltas
from full_serving.custom_report import phase_balance
from full_serving.boundary_control_experiment import server_arguments
from full_serving.resident_policy import balanced_map

def readrows(p):return [json.loads(x) for x in p.read_text().splitlines()]
def audit(root,control,pilot=False,dataset=None):
 root=Path(root);status=json.loads((root/"status.json").read_text())
 assert status["state"]=="comparison_completed"
 assert status["completed_sessions"]==[("pilot" if pilot else dataset)+"/"+control]
 data=json.loads((root/"data-manifest.json").read_text())
 directory=root/("pilot" if pilot else dataset)/control
 texts={f"server-{n}.{s}":(directory/f"server-{n}.{s}").read_text() for n in range(2) for s in ("log","err")}
 assert errors_before_shutdown(texts)==0
 assert engine_configuration(texts["server-0.log"])==engine_configuration(texts["server-1.log"])
 cfg=logged_config(texts["server-0.log"]);assert cfg["enable_eplb"] and logged_replica_count(cfg)==32
 identities=[]
 for role in ("primary","worker"):
  identities.append(runtime_identity(json.loads((directory/f"preflight-{role}.json").read_text()),role))
  launch=json.loads((directory/f"effective-launch-{role}.json").read_text())
  cmd=launch["command"]
  assert launch["arm"]==control and launch["plugin"]=="cs471_boundary"
  assert cmd==["vllm",*server_arguments(control,cmd[2],cmd[cmd.index("--data-parallel-address")+1],role)]
 phases={}
 for ds in ("longbench","numinamath") if pilot else (dataset,):
  for phase in ("warmup-00","warmup-01","pilot") if pilot else ("warmup-00","warmup-01","measured"):
   label=ds+"-"+phase if pilot else phase
   item=data["datasets"][ds]["items"]["measured" if phase=="measured" else "warmup"]
   calendar=json.loads((root/(ds+"-calendar.json")).read_text()) if phase=="measured" else None
   targets=None if pilot else json.loads((root/(ds+"-"+phase+"-lengths.json")).read_text())
   result,_=raw_phase(directory/(label+".jsonl"),item,calendar=calendar,targets=targets)
   assert result["header"]["http_force_close"] and result["header"]["arm"]==control
   phases[label]=result
 ranks=[]
 for rank in range(8):
  bound=[r for r in readrows(directory/f"boundary-control-rank{rank}.jsonl") if r["kind"]=="bound"]
  assert len(bound)==1 and bound[0]["rank"]==rank and bound[0]["control"]==control
  b=bound[0]
  if not control.startswith("b4_"):
   assert b["fixed_map"] and b["map"]==balanced_map() and not b["worker"] and not b["migrations"]
  if control.startswith("fixed32_"):
   assert b["native_bookkeeping"] and b["recording"]==(control!="fixed32_idle")
   assert b["logical_history"]==(control=="fixed32_history")
   if pilot:
    cost=readrows(directory/f"cost-events-rank{rank}.jsonl")
    assert any(r["kind"]=="map_record_oracle" and r["status"]=="passed" for r in cost)
    assert any(r["kind"]=="fixed_full_weight_check" and r["status"]=="passed" for r in readrows(directory/f"boundary-control-rank{rank}.jsonl"))
  elif control in ("native_hash","static_home","resident_charged"):
   binding=json.loads((directory/f"resident-bindings-rank{rank}.json").read_text())
   assert binding["physical_to_logical_map"]==balanced_map() and binding["layer_count"]==48 and not binding["native_control_step_active"]
   if pilot:
    d=json.loads((directory/f"resident-startup-diagnostic-rank{rank}.json").read_text())
    assert d["status"]=="passed" and d["weight_check"]["full_bf16_replica_hashes_equal"]
    assert len(d["native_expert_numerical_checks"])==9
  else:
   assert b["additional_sustained_observer"]==(control=="b4_observed")
  ranks.append(b)
 result=dict(status="passed",control=control,pilot=pilot,dataset=dataset,runtime=identities,phases=phases,ranks=ranks,
  limitations="Diagnostic interventions, same replica placement for fixed controls; one group descriptive. Matched counts do not fix tokens/routes. Native hash versus static home changes routing policy. Native-bookkeeping contrast also changes mapping/recording kernel.")
 if control.startswith("b4_"):
  result["gate"]=gate_audit(directory,"b4")
  if control=="b4_observed":result["additional_observer"]=migrations(directory)
  else:assert not list(directory.glob("migration-events-rank*.jsonl"))
 if not pilot:
  delta=metric_deltas(directory/"measured-metrics-before.txt",directory/"measured-metrics-after.txt")
  result["phase_balance"]=phase_balance(delta,request_count=512)
 return result
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("root",type=Path);p.add_argument("--control",required=True);p.add_argument("--pilot",action="store_true");p.add_argument("--dataset");p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 d=audit(a.root,a.control,a.pilot,a.dataset);a.output.write_text(json.dumps(d,indent=2)+"\n");print(json.dumps({"status":d["status"],"control":a.control}))

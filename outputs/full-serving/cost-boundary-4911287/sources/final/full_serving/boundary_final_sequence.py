"""Unprofiled final six-session runner and audited report; requires reviewed freeze."""
import argparse,fcntl,hashlib,html,json,os,subprocess,time
from pathlib import Path
from full_serving.boundary_final_contract import TOTALS,TREATMENTS,validate_admission
from full_serving.boundary_sequence_common import sha,read,write_new,runtime_signature,ratio,raw_hashes

BASE=Path("/work/cs-471/skewbench-hlin/full-serving")
REFERENCE=BASE/"full-sustained-512-20261111-v3/natural/results"
PYTHON="/work/cs-471/skewbench-hlin/venv/bin/python"
LIMITS=[
 "One group per dataset/arm; descriptive results, no confidence intervals or significance claims.",
 "Matched output counts do not guarantee identical generated tokens, routes, quality, batching or active KV state.",
 "Counts at8192 are matched targets inherited from designated natural OFF output counts; they do not establish natural EOS truncation for the matched arms.",
 "Content and arrivals are independent sources; cohorts overlap previous studies and are not fresh content.",
 "Full response and TTFT include connection setup and scheduled waiting; no POST retry.",
 "Native phase metrics and host/copy timings are not exposed critical-path or causal individual-migration payback evidence.",
 "Only observed GPU commit completions count; unenqueued/unobserved commits remain censored.",
 "Earlier failed attempts and negative findings are retained. No fastest-attempt selection.",
 "Raw logs, prompts, outputs and token arrays remain private. This generated report requires review before publication."
]

def plan(spec):
 assert spec["treatments"]==list(TREATMENTS)
 expected={"longbench":list(TREATMENTS),"numinamath":list(reversed(TREATMENTS))}
 assert spec["orders"]==expected
 return [(ds,treatment) for ds in TOTALS for treatment in expected[ds]]

def verify(deploy,protocol,admission,source):
 deploy,protocol,admission=Path(deploy),Path(protocol),Path(admission)
 assert len(source)==40 and subprocess.check_output(["git","rev-parse","HEAD"],cwd=deploy,text=True).strip()==source
 spec=read(protocol);plan(spec)
 from full_serving.custom_compare import canonical_hash
 assert spec["seal"]==canonical_hash({k:v for k,v in spec.items() if k!="seal"})
 for name,digest in spec["files"].items():assert sha(deploy/name)==digest,name
 assert sha(deploy/spec["data_manifest"])==spec["data_manifest_sha256"]
 for treatment in TREATMENTS:validate_admission(admission,protocol,source,treatment,deploy)
 return spec

def verify_predecessor(directory):
 directory=Path(directory)
 assert (directory/"completed.txt").read_text().strip()=="sequence_exit=0"
 completion=directory/"reconstruction-v1/completion.json";value=read(completion)
 assert value["status"]=="passed" and value["sessions"]==6 and value["measured_requests"]==3072
 assert value["causal_payback_established"] is False
 assert sha(completion.parent/"comparison-review-required.json")==value["report_sha256"]
 assert sha(completion.parent/"stable-records-private.json")==value["stable_records_private_sha256"]
 return {"completion_sha256":sha(completion),"scope":"Resource serialization after all frozen-candidate controls."}

def controller_command(run,ds,treatment,deploy,protocol,admission,source):
 arm,fix=TREATMENTS[treatment];cell=Path(run)/(ds+"-"+treatment)
 return list(map(str,[
  "srun","--jobid=4911287","--mpi=none","--exact","--nodes=1","--ntasks=1","--nodelist=kh021",
  "--cpus-per-task=4","--gpus=0","--mem=16G","--input=none",
  "--output="+str(cell/"controller.log"),"--error="+str(cell/"controller.err"),
  PYTHON,"-m","full_serving.boundary_final_experiment","run","--protocol",protocol,
  "--private-data","/work/cs-471/skewbench-hlin/private/sustained-512-v1-inputs",
  "--snapshot","/work/cs-471/skewbench-hlin/cache/huggingface/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
  "--output",cell/"results","--private-output",
  Path("/work/cs-471/skewbench-hlin/private/cost-boundary-4911287")/Path(run).name/(ds+"-"+treatment),
  "--source-revision",source,"--mode","matched","--reference",REFERENCE,"--pilot-admission",admission,
  "--dataset",ds,"--treatment",treatment,"--arm",arm,"--map-commit",fix,"--host-probe","off"]))

def audit_cell(run,ds,treatment,deploy):
 cell=Path(run)/(ds+"-"+treatment);assert not (cell/"audit.json").exists()
 subprocess.run([PYTHON,"-m","full_serving.boundary_final_analysis",str(cell/"results"),
                 "--dataset",ds,"--treatment",treatment,"--output",str(cell/"audit.json")],cwd=deploy,check=True)

def cell_summary(run,ds,treatment,deploy,protocol,admission,source,spec):
 arm,_=TREATMENTS[treatment];cell=Path(run)/(ds+"-"+treatment);root=cell/"results";directory=root/ds/arm
 value=read(cell/"audit.json");assert value["status"]=="passed" and value["dataset"]==ds and value["treatment"]==treatment
 assert sha(root/"protocol.json")==sha(protocol)
 assert sha(root/"pilot-admission.json")==sha(admission)
 assert sha(root/"data-manifest.json")==spec["data_manifest_sha256"]
 controller=read(root/"controller-source.json");assert controller["declared_git_revision"]==source
 for name,digest in controller["files"].items():
  blob=subprocess.check_output(["git","show",source+":"+name],cwd=deploy)
  assert hashlib.sha256(blob).hexdigest()==digest==sha(deploy/name),name
 assert value["phases"]["measured"]["arithmetic"]["requests"]==512
 assert value["phases"]["measured"]["arithmetic"]["failed_requests"]==0
 assert value["phases"]["measured"]["arithmetic"]["output_tokens"]==TOTALS[ds]
 rows=[json.loads(x) for x in (directory/"measured.jsonl").read_text().splitlines()]
 records=sorted((r for r in rows if r.get("type")=="request"),key=lambda r:r["request_index"])
 targets=read(root/(ds+"-measured-lengths.json"));timing=value["phases"]["measured"]["timing"]
 fingerprint=[(r["request_id"],r["input_sha256"],targets[r["request_id"]],
                r["scheduled_ns"]-timing["epoch_monotonic_ns"]) for r in records]
 seal=read(root/"matched-targets-seal.json");assert seal["frozen_before_server_start"]
 for name,digest in seal["files"].items():assert sha(root/name)==digest
 target_hashes=dict(seal["files"]);target_hashes["calendar"]=sha(root/(ds+"-calendar.json"))
 value["runtime_signature"]=runtime_signature(value["runtime"],read(root/"checkpoint-verified.json"),sha(root/"checkpoint-verified.json"))
 value["audit_sha256"]=sha(cell/"audit.json")
 return value,fingerprint,target_hashes

def comparisons(peers):
 out={}
 for left,right in (("b4","off"),("b4_noop","off"),("b4_noop","b4")):
  a,b=(peers[x]["phases"]["measured"]["arithmetic"] for x in (left,right))
  metrics={m:{stat:ratio(a[m][stat],b[m][stat]) for stat in ("mean","p95")} for m in ("response_ms","ttft_ms")}
  gain=100*(1-a["response_ms"]["mean"]/b["response_ms"]["mean"])
  out[left+"/"+right]={"metrics":metrics,"mean_response_gain_percent":gain,
                       "useful_mean_gain_at_least5percent":a["response_ms"]["mean"]<=.95*b["response_ms"]["mean"],
                       "mean_response_regression_exceeds2percent":a["response_ms"]["mean"]>1.02*b["response_ms"]["mean"]}
 return out

def combine(cells,spec,source,protocol_sha,admission_sha):
 assert set(cells)==set(plan(spec))
 signatures=[cells[key]["runtime_signature"] for key in plan(spec)]
 assert all(s==signatures[0] for s in signatures)
 roles=signatures[0]["runtime"];assert [r["hostname"] for r in roles]==["kh021","kh023"]
 devices=[d for r in roles for d in r["devices"]]
 assert len(devices)==8 and len({d["uuid"] for d in devices})==8 and all(d["name"]=="NVIDIA H100" for d in devices)
 datasets={}
 for ds in TOTALS:
  peers={t:cells[(ds,t)] for t in TREATMENTS}
  datasets[ds]={"cells":peers,"comparisons":comparisons(peers)}
 return {"status":"passed","scope":"unprofiled_final_matched_OFF_originalB4_selected_noop_fix",
         "sessions":6,"measured_requests":3072,"source_revision":source,"protocol_sha256":protocol_sha,
         "admission_sha256":admission_sha,"selected_fix":spec["selected_fix"],"groups_per_dataset":1,
         "datasets":datasets,"criteria":{"useful_mean_gain_percent":5,"regression_guard_percent":2,
         "application":"Shown per dataset and comparator; no pooled weighting or statistical significance."},
         "limitations":LIMITS,"publication_review_required":True}

def render_html(report):
 def esc(x):return html.escape(str(x))
 def number(x):return f"{x:,.3f}"
 rows=[];contrasts=[]
 for ds,data in report["datasets"].items():
  for treatment,cell in data["cells"].items():
   phase=cell["phases"]["measured"];a=phase["arithmetic"];t=phase["timing"]
   values=[ds,treatment,a["requests"],a["failed_requests"],number(a["response_ms"]["mean"]),number(a["response_ms"]["p95"]),
           number(a["ttft_ms"]["mean"]),number(a["ttft_ms"]["p95"]),number(t["completed_elapsed_ns"]/1e9),
           number(t["offered_horizon_ns"]/1e9),number(t["drain_ns"]/1e9),a["output_tokens"],a["ceiling_hits"]]
   rows.append("<tr>"+"".join("<td>"+esc(v)+"</td>" for v in values)+"</tr>")
  for name,c in data["comparisons"].items():
   contrasts.append("<tr>"+"".join("<td>"+esc(v)+"</td>" for v in
      (ds,name,number(c["mean_response_gain_percent"])+"%",c["useful_mean_gain_at_least5percent"],c["mean_response_regression_exceeds2percent"]))+"</tr>")
 details=[]
 for ds,data in report["datasets"].items():
  for treatment,cell in data["cells"].items():
   # Numeric aggregate evidence only. Never embed raw request text, token arrays or logs.
   evidence={k:cell[k] for k in ("phase_balance","native_metrics","gate","migrations","no_op") if k in cell}
   details.append("<details><summary>"+esc(ds+" / "+treatment+"：原生階段及 commit 證據")+"</summary><pre>"+esc(json.dumps(evidence,ensure_ascii=False,indent=2))+"</pre></details>")
 heads=["資料集","方案","請求","失敗","平均回應 ms","p95 回應 ms","平均 TTFT ms","p95 TTFT ms","總時間 s","到達窗口 s","排空 s","輸出 tokens","8192-token 固定目標數"]
 table="<table><thead><tr>"+"".join("<th>"+esc(h)+"</th>" for h in heads)+"</tr></thead><tbody>"+"".join(rows)+"</tbody></table>"
 return """<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CS471 matched serving comparison</title><style>body{font:16px/1.6 system-ui,sans-serif;max-width:1280px;margin:40px auto;padding:0 24px;color:#17212b}table{border-collapse:collapse;font-size:14px;width:100%}td,th{border:1px solid #ccd3da;padding:8px;text-align:right}th:first-child,td:first-child,td:nth-child(2){text-align:left}.scroll{overflow:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:16px}details{margin:16px 0}h1{font-size:28px}</style>
<h1>CS471：固定輸出長度的端到端比較</h1><p><strong>審閱用報告，尚未發布。</strong>同一組八張 H100；LongBench 與 NuminaMath 各512筆、一組；每個方案使用全新服務程序。回應及TTFT從預定到達時計算，包含連線建立成本。</p>
<p>方案：OFF、原始 B4、B4 的已選修正。正增益表示平均完整回應時間降低。單組結果只支持描述性比較。</p>
"""+"<p>來源："+esc(report["source_revision"])+"；協定 SHA256："+esc(report["protocol_sha256"])+"</p><div class=scroll>"+table+"</div><h2>描述性比較</h2><table><tr><th>資料集</th><th>比較</th><th>平均回應增益</th><th>達5%</th><th>退步超2%</th></tr>"+"".join(contrasts)+"</table><h2>量測與解讀限制</h2><ul>"+"".join("<li>"+esc(x)+"</li>" for x in report["limitations"])+"</ul><h2>原生階段及提案／commit</h2>"+"".join(details)+"</html>"

def report(run,output,deploy,protocol,admission,source):
 run,output=Path(run),Path(output);assert not output.exists();output.mkdir()
 spec=verify(deploy,protocol,admission,source);before=raw_hashes(run);cells={};fingerprints={};targets={}
 for ds,treatment in plan(spec):
  value,fingerprint,lengths=cell_summary(run,ds,treatment,deploy,protocol,admission,source,spec)
  assert fingerprints.setdefault(ds,fingerprint)==fingerprint
  assert targets.setdefault(ds,lengths)==lengths
  cells[(ds,treatment)]=value
 combined=combine(cells,spec,source,sha(protocol),sha(admission))
 time.sleep(2);after=raw_hashes(run);assert before==after
 write_new(output/"stable-records-private.json",{"status":"passed","private":True,"files":after})
 write_new(output/"comparison-review-required.json",combined)
 with (output/"comparison-review-required.html").open("x") as f:f.write(render_html(combined))
 write_new(output/"completion.json",{"status":"passed","sessions":6,"measured_requests":3072,"source_revision":source,
  "protocol_sha256":sha(protocol),"admission_sha256":sha(admission),"publication_review_required":True,
  "report_sha256":sha(output/"comparison-review-required.json"),"html_sha256":sha(output/"comparison-review-required.html"),
  "stable_records_private_sha256":sha(output/"stable-records-private.json")})

def execute(args):
 assert not args.run.exists(),"Use new run; never repeat completed GPU measurements."
 assert not any(k.startswith(("FULL_SERVING_PAYBACK","FULL_SERVING_BOUNDARY","CS471_")) for k in os.environ)
 spec=verify(args.deploy,args.protocol,args.admission,args.source_revision)
 args.run.mkdir();lock=(args.run/"launcher.lock").open("w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);code=1
 try:
  write_new(args.run/"plan.json",{"cells":plan(spec),"source_revision":args.source_revision,
    "protocol_sha256":sha(args.protocol),"admission_sha256":sha(args.admission),"predecessor":str(args.predecessor)})
  (args.run/"active-stage.txt").write_text("waiting_existing_transition_main\n")
  while not (args.predecessor/"completed.txt").exists():time.sleep(30)
  write_new(args.run/"predecessor-receipt.json",verify_predecessor(args.predecessor))
  for ds,treatment in plan(spec):
   verify(args.deploy,args.protocol,args.admission,args.source_revision)
   cell=args.run/(ds+"-"+treatment);cell.mkdir();(args.run/"active-stage.txt").write_text(ds+"/"+treatment+"\n")
   cmd=controller_command(args.run,ds,treatment,args.deploy,args.protocol,args.admission,args.source_revision)
   write_new(cell/"controller-command.json",cmd)
   subprocess.run(cmd,cwd=args.deploy,check=True);audit_cell(args.run,ds,treatment,args.deploy)
  report(args.run,args.run/"reconstruction-v1",args.deploy,args.protocol,args.admission,args.source_revision)
  (args.run/"active-stage.txt").write_text("completed_pending_model_privacy_and_publication_review\n");code=0
 finally:(args.run/"completed.txt").write_text("sequence_exit="+str(code)+"\n")

def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest="action",required=True)
 for action in ("run","report"):
  a=sub.add_parser(action)
  for key in ("run","deploy","protocol","admission"):a.add_argument("--"+key,type=Path,required=True)
  a.add_argument("--source-revision",required=True)
  if action=="run":a.add_argument("--predecessor",type=Path,required=True)
  else:a.add_argument("--output",type=Path,required=True)
 args=p.parse_args()
 if args.action=="run":execute(args)
 else:report(args.run,args.output,args.deploy,args.protocol,args.admission,args.source_revision)
if __name__=="__main__":main()

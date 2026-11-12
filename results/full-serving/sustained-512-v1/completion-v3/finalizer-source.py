"""Read-only durable collection; no GPU job restarts or result selection."""
import datetime,fcntl,hashlib,json,shlex,subprocess,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parent;REPO=BASE/'cs471'
REMOTE='hlin@kuma.hpc.epfl.ch';RUN='/work/cs-471/skewbench-hlin/full-serving/full-sustained-512-20261111-v3'
LOCAL=REPO/'results/full-serving/sustained-512-v1/run-v3';AUDIT=LOCAL.parent/'completion-v3'
STATE=BASE/'sustained-v3-finalizer-state.json'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',REMOTE]

def run(args,timeout=600):return subprocess.run(args,cwd=REPO,check=True,capture_output=True,text=True,timeout=timeout).stdout

def state(status,**kw):
 data={'status':status,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw}
 temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(STATE)

def inventory():
 code="import pathlib,json,hashlib;p=pathlib.Path("+repr(RUN)+");print(json.dumps({str(f.relative_to(p)):{'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in p.rglob('*') if f.is_file()}))"
 return json.loads(run(SSH+['python3 -c '+shlex.quote(code)],timeout=900))

def main():
 lock=(BASE/'sustained-v3-finalizer.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 LOCAL.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(REPO))
 deadline=time.monotonic()+20*3600
 while time.monotonic()<deadline:
  try:
   code="from pathlib import Path;import json;p=Path("+repr(RUN)+");print(json.dumps({'stage':(p/'active-stage.txt').read_text() if (p/'active-stage.txt').exists() else None,'exit':(p/'completed.txt').read_text() if (p/'completed.txt').exists() else None}))"
   info=json.loads(run(SSH+['python3 -c '+shlex.quote(code)],timeout=60))
   run(['rsync','-a','--timeout=120',REMOTE+':'+RUN+'/',str(LOCAL)+'/'],timeout=900)
   state('collecting',**info)
   if info['exit']:
    if info['exit']!='sequence_exit=0\n':state('sequence_failed_preserve_all_attempts',**info);return 1
    break
  except (OSError,subprocess.SubprocessError,ValueError) as exc:
   state('read_only_collection_retry',error=repr(exc))
  time.sleep(120)
 else:state('deadline_no_completion_claim');return 1
 state('stable_byte_and_git_audit')
 before=inventory();run(['rsync','-a','--timeout=120',REMOTE+':'+RUN+'/',str(LOCAL)+'/'],timeout=900);after=inventory()
 assert before==after,'Final remote records still changing'
 files={str(p.relative_to(LOCAL)):p for p in LOCAL.rglob('*') if p.is_file()};assert set(files)==set(after)
 for n,p in files.items():assert {'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}==after[n],n
 for raw in LOCAL.rglob('*.jsonl'):
  with raw.open() as f:
   first=f.readline()
  if first:
   header=json.loads(first)
   if header.get('type')=='run_header':assert header.get('http_force_close') is True,str(raw)
 for source in LOCAL.rglob('controller-source.json'):
  obj=json.loads(source.read_text());rev=obj['declared_git_revision']
  for n,sha in obj['files'].items():
   raw=subprocess.check_output(['git','show',rev+':'+n],cwd=REPO);assert hashlib.sha256(raw).hexdigest()==sha,n
  spec=json.loads(source.with_name('protocol.json').read_text())
  for n,sha in spec['files'].items():
   raw=subprocess.check_output(['git','show',rev+':'+n],cwd=REPO);assert hashlib.sha256(raw).hexdigest()==sha,n
 from full_serving.sustained_analysis import report,pilot
 from full_serving.sustained_payback import analyze
 # Compare independently re-executed local CPU arithmetic to cluster analysis.
 result={}
 for arm in ('off','b2','b4'):
  rebuilt=pilot(LOCAL/('pilot-'+arm)/'results',arm)
  assert rebuilt==json.loads((LOCAL/('pilot-'+arm)/'audit.json').read_text())
 for mode in ('natural','matched'):
  rebuilt=report(LOCAL/mode/'results');remote=json.loads((LOCAL/mode/'analysis.json').read_text())
  # Only machine-dependent metric source labels differ; verify exact expected
  # locations before canonicalizing. All other fields must remain exactly equal.
  for dataset in rebuilt['datasets']:
   for arm in rebuilt['datasets'][dataset]['arms']:
    for boundary in ('before','after'):
     suffix=f'{dataset}/{arm}/measured-metrics-{boundary}.txt'
     lm=rebuilt['datasets'][dataset]['arms'][arm]['native_metrics']
     rm=remote['datasets'][dataset]['arms'][arm]['native_metrics']
     assert lm[boundary]==str(LOCAL/mode/'results'/suffix)
     assert rm[boundary]==RUN+'/'+mode+'/results/'+suffix
     lm[boundary]=rm[boundary]=suffix
  assert rebuilt==remote,'CPU/cluster reconstruction differs after verified path normalization'
  result[mode]=rebuilt
 result['payback_diagnostic']=analyze(LOCAL)
 AUDIT.mkdir(parents=True,exist_ok=True)
 (AUDIT/'remote-inventory.json').write_text(json.dumps({'before':before,'after':after},indent=2)+'\n')
 output=REPO/'outputs/full-serving/sustained-512-comparison.json';output.write_text(json.dumps(result,indent=2)+'\n')
 (AUDIT/'completion.json').write_text(json.dumps({'status':'completed_and_audited','sessions':24,'measured_requests':12288,'public_raw_files':len(files),'stable_remote_local_bytes':True,'git_sources':True,'independent_cpu_cluster_reconstruction':True,'report':str(output),'public_privacy_review_and_final_delivery':'pending'},indent=2)+'\n')
 state('completed_and_audited_pending_publication',report=str(output))
 return 0

if __name__=='__main__':
 try:sys.exit(main())
 except Exception as exc:state('finalization_failed_no_completion_claim',error=repr(exc),stderr=getattr(exc,'stderr',None));raise

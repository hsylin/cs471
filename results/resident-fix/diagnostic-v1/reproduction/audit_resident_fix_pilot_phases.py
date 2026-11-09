"""Descriptive old-development service split; not a causal stage attribution."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def run(root):
 files=sorted(root.glob('trials/*.jsonl')); rows=[]
 for p in files: rows.extend(json.loads(x) for x in p.read_text().splitlines() if x.strip())
 selected={}
 for r in rows:
  if r['arm'] not in ('ours','stockoff'): continue
  k=(r['content_group_id'],r['arrival_type'],r['rho'])
  if r['arm'] in selected.setdefault(k,{}): raise AssertionError('Duplicate pair')
  selected[k][r['arm']]=r
 pairs=[]
 for k,arms in sorted(selected.items()):
  assert set(arms)=={'ours','stockoff'}
  n=arms['ours']['n_jobs']; assert n==4096 and arms['stockoff']['n_jobs']==n
  phase=np.arange(n)%64
  classes={'nonrecording_ordinary':phase<48,'recording_ordinary':(phase>=48)&(phase<63),'boundary':phase==63}
  assert np.array_equal(arms['ours']['native_metadata']['recorded_step_indices'],np.flatnonzero(phase>=48))
  service={a:(np.asarray(r['coordinator_finish_ns'],dtype=np.int64)-np.asarray(r['coordinator_start_ns'],dtype=np.int64)) for a,r in arms.items()}
  assert all(np.all(v>0) for v in service.values())
  cell={'content_group_id':k[0],'dataset':arms['ours']['dataset'],'arrival_type':k[1],'rho':k[2],'phases':{}}
  for c,m in classes.items():
   cell['phases'][c]={'jobs':int(m.sum()),'ours_mean_ms':float(service['ours'][m].mean()/1e6),'stockoff_mean_ms':float(service['stockoff'][m].mean()/1e6),'paired_extra_mean_ms':float((service['ours'][m]-service['stockoff'][m]).mean()/1e6),'paired_extra_total_ms':float((service['ours'][m]-service['stockoff'][m]).sum()/1e6)}
  pairs.append(cell)
 summaries={}
 for dataset in ['all']+sorted({r['dataset'] for r in pairs}):
  subset=[r for r in pairs if dataset=='all' or r['dataset']==dataset]
  summaries[dataset]={'paired_trials':len(subset),'unique_content_groups':len({r['content_group_id'] for r in subset}),'phases':{c:{key:float(np.mean([r['phases'][c][key] for r in subset])) for key in ('jobs','ours_mean_ms','stockoff_mean_ms','paired_extra_mean_ms','paired_extra_total_ms')} for c in classes}}
 return {'schema':'resident_fix_old_pilot_phase_association_v1','source_scope':'Old six development content groups only; 24 matched ours/stockoff trials; no new final demand','timing':'Coordinator finish minus start; includes inference-stream completion and coordinator protocol','causal_claim':False,'limitations':['Recording phase is associated with elapsed service, not randomized independently of input, calendar, queue, or controller state.','Phase differences do not isolate a kernel, synchronization, memory operation or Python component.','The new matched control/selector ablation is needed for component attribution.','Repeated jobs and arrival conditions are not independent workload samples.'],'phase_definition':{'nonrecording_ordinary':'measured index modulo64 0..47','recording_ordinary':'48..62','boundary':'63, includes recording+scheduled planner/no-op commit'},'source_files':{str(p):sha(p) for p in files},'script_sha256':sha(Path(__file__)),'summaries':summaries,'pairs':pairs}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--records-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--markdown',type=Path,required=True);a=p.parse_args()
 result=run(a.records_root);a.output.write_text(json.dumps(result,indent=2)+'\n')
 lines=['# Old-pilot phase association','',result['source_scope']+'.','', 'Service is coordinator finish minus start. These are descriptive paired associations, not a causal decomposition.','', '| Phase | Jobs per trial | Ours mean ms | Stock-off mean ms | Extra ms/job | Extra ms/trial |','|---|---:|---:|---:|---:|---:|']
 for c,r in result['summaries']['all']['phases'].items(): lines.append(f"| {c} | {r['jobs']:.0f} | {r['ours_mean_ms']:.6f} | {r['stockoff_mean_ms']:.6f} | {r['paired_extra_mean_ms']:.6f} | {r['paired_extra_total_ms']:.3f} |")
 lines+=['','All 24 ours records independently match the archived native observation indices: phases48–63 are recorded. The boundary class is separated from phases48–62.','']+[f'- {s}' for s in result['limitations']]
 a.markdown.write_text('\n'.join(lines)+'\n')
 print(json.dumps(result['summaries']['all'],indent=2))

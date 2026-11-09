"""Independent, exploratory identical-input factorial diagnostic analysis.

No production analysis imports. Average technical repetitions within content
groups. No confidence intervals or tests. Offline homes are headroom diagnostics.
"""
from __future__ import annotations
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np

VARIANTS=('stockoff','current_online','current_offline','static_online','static_offline')

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def digest(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def array_digest(value):
 a=np.ascontiguousarray(value)
 h=hashlib.sha256(digest([a.dtype.str,list(a.shape)]).encode())
 h.update(memoryview(a).cast('B'))
 return h.hexdigest()

def read(path):
 opener=gzip.open if path.suffix=='.gz' else open
 with opener(path,'rt') as f: return [json.loads(line) for line in f if line.strip()]

def extract(row):
 if row.get('record_id')!=digest({k:v for k,v in row.items() if k!='record_id'}):
  raise ValueError('Raw record identity differs')
 n=row['n_jobs']
 if n!=512 or row['jobs']!=512 or row['unique_batches']!=64 or row['selector']!='three_kernel':
  raise ValueError('Not the declared 64-by8 three-kernel factorial diagnostic')
 starts=np.asarray(row['coordinator_start_ns'],dtype=np.int64)
 finishes=np.asarray(row['coordinator_finish_ns'],dtype=np.int64)
 admits=np.asarray(row['coordinator_admit_ns'],dtype=np.int64)
 if any(v.shape!=(n,) for v in (starts,finishes,admits)):
  raise ValueError('Wrong clock length')
 if np.any(starts<admits) or np.any(finishes<=starts) or np.any(starts[1:]<finishes[:-1]):
  raise ValueError('Invalid sequential raw clocks')
 if row['arrival_offsets_ns']!=[0]*n: raise ValueError('Diagnostic is not saturated same-arrival input')
 service=(finishes-starts)/1e6
 if abs(row['queue_metrics']['summary_ns']['service']['mean']/1e6-float(service.mean()))>1e-10:
  raise ValueError('Stored service metric differs from raw clocks')
 variant=row['variant']
 if variant not in VARIANTS: raise ValueError('Unknown factorial arm')
 if row['arm_order'].count(variant)!=1 or set(row['arm_order'])!=set(VARIANTS):
  raise ValueError('Invalid diagnostic arm order')
 states=row['per_rank_state']
 if len(states)!=2: raise ValueError('Need both native rank states')
 for state in states:
  begin,end,audit=state['start'],state['end'],state['audit']
  if begin['final_map']!=end['final_map'] or end['final_map']!=audit['final_map']:
   raise ValueError('Diagnostic moved the fixed placement')
  if end['migration_sent_bytes_local'] or end['migration_sent_messages_local']:
   raise ValueError('Unexpected diagnostic movement')
  if not audit['resident_state_verified'] or not audit['normal_completion_verified']:
   raise ValueError('Missing completed weight/map audit')
  if begin['mapper_calls']!=0 or end['mapper_calls']!=(0 if variant=='stockoff' else n):
   raise ValueError('Incorrect measured mapper calls')
  if variant.startswith('current'):
   if end['controller_step_calls']!=n or len(end['decisions'])!=8:
    raise ValueError('Current control must charge all eight cadence boundaries')
  elif variant.startswith('static'):
   if end['controller_step_calls']!=0 or end['static_step_calls']!=n or not audit['static_state_unchanged']:
    raise ValueError('Static state/control contract failed')
  elif end['controller_step_calls']!=0: raise ValueError('Stock-off unexpectedly used a controller')
 phase=np.arange(n)%64
 return dict(mean_service_ms=float(service.mean()),ordinary_nonrecording_ms=float(service[phase<48].mean()),
  ordinary_recording_ms=float(service[(phase>=48)&(phase<63)].mean()),boundary_ms=float(service[phase==63].mean()),
  total_service_ms=float(service.sum()),completed_elapsed_ms=row['completed_elapsed_ns']/1e6)

def summarize(rows):
 cells={};identity={};orders={};group_dataset={}
 for row in rows:
  if row['schema']!='resident_fix_diagnostic_v1': raise ValueError('Wrong schema')
  key=(row['group'],int(row['repeat']),row['variant'])
  if key in cells: raise ValueError('Duplicate diagnostic cell')
  if key[1] not in (0,1): raise ValueError('Expected exactly two technical repetitions')
  cells[key]=extract(row)
  ident=tuple(row[k] for k in ('dataset','bank_sha256','indices_sha256','logical_routes_sha256','exact_physical_assignment_sha256'))
  if key[0] in identity and identity[key[0]]!=ident: raise ValueError('Paired inputs or exact assignment changed')
  identity[key[0]]=ident;group_dataset[key[0]]=row['dataset']
  orderkey=key[:2];order=tuple(row['arm_order'])
  if orderkey in orders and orders[orderkey]!=order: raise ValueError('Inconsistent logged order')
  orders[orderkey]=order
 groups=sorted(identity)
 if len(groups)!=6 or len(cells)!=60: raise ValueError('Require complete six-group × two-repetition × five-arm matrix')
 if set(cells)!={(g,r,v) for g in groups for r in (0,1) for v in VARIANTS}: raise ValueError('Incomplete matrix')
 for g in groups:
  if orders[g,1]!=orders[g,0][::-1]: raise ValueError('Technical repetition order is not reversed')
 # Collapse the two technical repetitions before computing paired group effects.
 means={g:{v:{metric:float(np.mean([cells[g,r,v][metric] for r in (0,1)]))
              for metric in next(iter(cells.values()))} for v in VARIANTS} for g in groups}
 m={v:np.array([means[g][v]['mean_service_ms'] for g in groups]) for v in VARIANTS}
 def effect(values):
  return {'point':float(values.mean()),'observed_content_group_range':[float(values.min()),float(values.max())],
          'per_group':{g:float(x) for g,x in zip(groups,values)}}
 def ratio(treatment,control):
  logs=np.log(m[treatment]/m[control]);e=effect(logs)
  return {'ratio':float(np.exp(e['point'])),'observed_content_group_range':np.exp(e['observed_content_group_range']).tolist(),
          'per_group':{g:float(np.exp(x)) for g,x in e['per_group'].items()}}
 contrasts={}
 for name,t,c in [('control_removal_online','static_online','current_online'),
                  ('control_removal_known_homes','static_offline','current_offline'),
                  ('selection_removed_current','current_offline','current_online'),
                  ('selection_removed_static','static_offline','static_online'),
                  ('static_online_over_stockoff','static_online','stockoff'),
                  ('static_known_homes_over_stockoff','static_offline','stockoff')]:
  contrasts[name]={'treatment':t,'control':c,'service_ratio':ratio(t,c),
                   'saved_ms_per_job':effect(m[c]-m[t])}
 control_effect=((m['current_online']-m['static_online'])+(m['current_offline']-m['static_offline']))/2
 selection_effect=((m['current_online']-m['current_offline'])+(m['static_online']-m['static_offline']))/2
 interaction=(m['current_online']-m['current_offline'])-(m['static_online']-m['static_offline'])
 return dict(schema='resident_fix_factorial_analysis_v1',complete=True,trials=60,content_groups=6,technical_repetitions=2,
  jobs_per_trial=512,unique_batches_per_group=64,phase='old_development_exploratory',
  estimand='Arithmetic mean coordinator service per trial; average technical repetitions within group; equally weight six content groups',
  uncertainty={'unit':'content group','descriptive_only':True,'confidence_intervals':False,'hypothesis_tests':False,'range':'Observed minimum and maximum of six content-group effects after averaging repetitions'},
  limitations=['Only six previously measured development groups; this is not fresh held-out evidence.',
   'Matched ablations change the declared control or selection package; differences do not identify individual Python/GPU operations.',
   'Offline homes use prior knowledge and are never an online-production comparison.',
   'Clock intervals include coordinator protocol and GPU completion; not device-kernel residence.',
   'Reported ranges are the six observed content-group effects, not uncertainty intervals or advancement tests.',
   'Floating-point correctness comes from runner assertions; this CPU analysis does not rerun GPU arithmetic.'],
  contrasts=contrasts,main_effect_control_cost_ms=effect(control_effect),main_effect_selection_cost_ms=effect(selection_effect),
  interaction_ms=effect(interaction),interaction_definition='(current_online-current_offline)-(static_online-static_offline); positive means greater selection package cost with current control',
  group_means=means,group_datasets=group_dataset,
  arm_position_counts={v:[sum(order[i]==v for order in orders.values()) for i in range(5)] for v in VARIANTS},
  source_pair_identity={g:dict(zip(('dataset','bank_sha256','indices_sha256','logical_routes_sha256','exact_physical_assignment_sha256'),identity[g])) for g in groups})

def audit_artifacts(root,rows,result):
 plan_path=root/'plan.json';plan=json.loads(plan_path.read_text())
 for path,expected in plan['diagnostic_files'].items():
  if sha(root/'source'/path)!=expected: raise ValueError('Archived diagnostic source changed: '+path)
 declared={(c['group'],c['repeat']):c['order'] for c in plan['cases']}
 for row in rows:
  if row['arm_order']!=declared[row['group'],row['repeat']]: raise ValueError('Executed order differs from predeclared order')
 blocks={entry['block_id']:entry for entry in plan['groups']}
 artifacts={}
 for group,identity in result['source_pair_identity'].items():
  if identity['bank_sha256']!=blocks[group]['capture_sha256']: raise ValueError('Declared bank hash differs')
  path=root/(group+'-inputs.npz')
  with np.load(path,allow_pickle=False) as data:
   indices,logical,physical=(data[k] for k in ('indices','logical_ids','physical_ids'))
   if indices.shape!=(64,128) or logical.shape!=(64,128,8) or physical.shape!=logical.shape:
    raise ValueError('Wrong saved input dimensions')
   for array,key in ((indices,'indices_sha256'),(logical,'logical_routes_sha256'),(physical,'exact_physical_assignment_sha256')):
    if array_digest(array)!=identity[key]: raise ValueError('Saved input digest differs: '+key)
   if np.any(indices<0) or np.any(indices>=blocks[group]['tokens']): raise ValueError('Invalid selected bank row')
   if any(len(set(i))!=128 for i in indices) or len({tuple(sorted(i)) for i in indices})!=64:
    raise ValueError('Repeated records within a batch or repeated batch combination')
   # Independent integer greedy assignment; no production policy import.
   for ids,actual in zip(logical,physical):
    if np.any(ids<0) or np.any(ids>=128): raise ValueError('Invalid logical expert')
    counts=np.bincount(ids.ravel(),minlength=128)
    active=[np.count_nonzero(counts[32:80]),np.count_nonzero(counts[80:128])]
    loads=[int(counts[32:80].sum()),int(counts[80:128].sum())]
    shared=sorted([e for e in range(32) if counts[e]],key=lambda e:(-int(counts[e]),e))
    cap=max(*active,(sum(active)+len(shared)+1)//2)
    homes=np.arange(128)
    for e in shared:
     rank=min([r for r in (0,1) if active[r]<cap],key=lambda r:(loads[r],r))
     homes[e]=e+128*rank;loads[rank]+=int(counts[e]);active[rank]+=1
    if not np.array_equal(actual,homes[ids]): raise ValueError('Saved physical assignment differs from independent greedy oracle')
   artifacts[group]={'file_sha256':sha(path),'batches_verified':64,'routes_verified':int(logical.size)}
 return {'plan_sha256':sha(plan_path),'archived_source_hashes_verified':plan['diagnostic_files'],
  'orders_match_plan':True,'input_artifacts':artifacts,'independent_saved_assignment_oracle':True,
  'scope':'Saved assignments and recorded execution contracts; no external GPU rerun'}

def self_test():
 import copy
 rows=[]
 for gi in range(6):
  group='group'+str(gi);order=list(VARIANTS[gi%5:]+VARIANTS[:gi%5])
  for rep in (0,1):
   for variant in VARIANTS:
    costs={'stockoff':0,'current_online':3,'current_offline':1,'static_online':2,'static_offline':0}
    duration=int((10+gi+costs[variant]+(-.5 if rep==0 else .5))*1e6)
    starts=(np.arange(512,dtype=np.int64)*(duration+1000)+100000).tolist();finishes=[x+duration for x in starts]
    begin={'mapper_calls':0,'final_map':[0]}
    end={'mapper_calls':0 if variant=='stockoff' else 512,'final_map':[0],
      'migration_sent_bytes_local':0,'migration_sent_messages_local':0,
      'controller_step_calls':512 if variant.startswith('current') else 0,
      'static_step_calls':512,'decisions':[{}]*8}
    audit={'final_map':[0],'resident_state_verified':True,'normal_completion_verified':True,'static_state_unchanged':True}
    row={'schema':'resident_fix_diagnostic_v1','group':group,'repeat':rep,'variant':variant,'n_jobs':512,'jobs':512,
     'unique_batches':64,'selector':'three_kernel','coordinator_start_ns':starts,'coordinator_finish_ns':finishes,
     'coordinator_admit_ns':starts,'arrival_offsets_ns':[0]*512,
     'queue_metrics':{'summary_ns':{'service':{'mean':duration}}},'completed_elapsed_ns':finishes[-1],
     'arm_order':order if rep==0 else order[::-1],'per_rank_state':[dict(start=begin,end=end,audit=audit)]*2,
     'dataset':'d'+str(gi//2),'bank_sha256':'a','indices_sha256':'b','logical_routes_sha256':'c','exact_physical_assignment_sha256':'d'}
    row['record_id']=digest(row);rows.append(row)
 result=summarize(rows)
 assert result['main_effect_control_cost_ms']['point']==1
 assert result['main_effect_selection_cost_ms']['point']==2
 assert result['interaction_ms']['point']==0
 for mutation in ('missing','clock','identity','boundaries'):
  bad=copy.deepcopy(rows)
  if mutation=='missing':bad.pop()
  elif mutation=='clock':bad[0]['coordinator_finish_ns'][0]+=10
  elif mutation=='identity':bad[0]['logical_routes_sha256']='changed'
  else:bad[1]['per_rank_state'][0]['end']['decisions']=[]
  if mutation!='missing':bad[0 if mutation!='boundaries' else 1]['record_id']=digest({k:v for k,v in bad[0 if mutation!='boundaries' else 1].items() if k!='record_id'})
  try:summarize(bad)
  except ValueError:pass
  else:raise AssertionError('Mutation accepted: '+mutation)
 print('five substantive self-tests passed')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--records',type=Path,nargs='+');p.add_argument('--output',type=Path);p.add_argument('--markdown',type=Path);p.add_argument('--self-test',action='store_true');p.add_argument('--artifact-root',type=Path);args=p.parse_args()
 if args.self_test:self_test();raise SystemExit
 if not args.records or args.output is None:p.error('--records and --output required')
 rows=[r for path in args.records for r in read(path)];result=summarize(rows)
 if args.artifact_root is not None:result['artifact_audit']=audit_artifacts(args.artifact_root,rows,result)
 result.update(source_files={str(path):sha(path) for path in args.records},script_sha256=sha(Path(__file__)),numpy_version=np.__version__)
 args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 if args.markdown:
  lines=['# Identical-input factorial diagnostic','',result['estimand']+'.','',
   '| Matched contrast | Service ratio | Observed group range | Saved µs/job |',
   '|---|---:|---:|---:|']
  for name,c in result['contrasts'].items():
   ratio=c['service_ratio'];ci=ratio['observed_content_group_range']
   lines.append(f"| {name} | {ratio['ratio']:.6f} | [{ci[0]:.6f}, {ci[1]:.6f}] | {1000*c['saved_ms_per_job']['point']:.3f} |")
  lines+=['',f"Control-package main effect: {1000*result['main_effect_control_cost_ms']['point']:.3f}µs/job. Selection-package main effect: {1000*result['main_effect_selection_cost_ms']['point']:.3f}µs/job. Interaction: {1000*result['interaction_ms']['point']:.3f}µs/job.",'']+[f'- {s}' for s in result['limitations']]
  args.markdown.write_text('\n'.join(lines)+'\n')
 print(json.dumps({'complete':result['complete'],'trials':result['trials'],'contrasts':result['contrasts']},indent=2))

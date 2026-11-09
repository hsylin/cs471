"""Independent descriptive audit of the bounded three-arm fused development run.

Imports only a prior independent audit helper, not production policy/analysis.
Technical repetitions collapse before group comparisons. No CI or tests.
"""
import argparse,copy,json
from pathlib import Path
import numpy as np
import analyze_resident_fix_factorial as base

ARMS=('stockoff','current_online','static_online')


def extract(row):
 if row.get('record_id')!=base.digest({k:v for k,v in row.items() if k!='record_id'}):raise ValueError('Changed record digest')
 if row['schema']!='resident_fix_diagnostic_v1' or row['selector']!='fused' or row['n_jobs']!=512 or row['jobs']!=512 or row['unique_batches']!=64:raise ValueError('Wrong fused trial geometry')
 a=row['variant']
 if a not in ARMS or set(row['arm_order'])!=set(ARMS) or len(row['arm_order'])!=3:raise ValueError('Wrong arms/order')
 begin,admit,finish=[np.asarray(row[k],np.int64) for k in ('coordinator_start_ns','coordinator_admit_ns','coordinator_finish_ns')]
 if any(x.shape!=(512,) for x in (begin,admit,finish)) or np.any(begin<admit) or np.any(finish<=begin) or np.any(begin[1:]<finish[:-1]):raise ValueError('Invalid raw clocks')
 if row['arrival_offsets_ns']!=[0]*512:raise ValueError('Diagnostic must be saturated')
 service=(finish-begin)/1e6
 completed=(int(row['observation_end_ns'])-int(row['epoch_monotonic_ns']))/1e6
 if int(row['completed_elapsed_ns'])!=int(row['observation_end_ns'])-int(row['epoch_monotonic_ns']):raise ValueError('Completed elapsed differs from raw clocks')
 if row['job_completed_elapsed_ns']!=int(finish[-1])-int(row['epoch_monotonic_ns']):raise ValueError('Job completion differs')
 if not finish[-1]<=row['controller_quiescent_ns']<=row['observation_end_ns']:raise ValueError('Controller completion outside declared elapsed')
 if abs(float(service.mean())-row['queue_metrics']['summary_ns']['service']['mean']/1e6)>1e-10:raise ValueError('Stored service differs from raw clocks')
 if len(row['per_rank_state'])!=2:raise ValueError('Missing rank state')
 for state in row['per_rank_state']:
  start,end,audit=(state[k] for k in ('start','end','audit'))
  if start['final_map']!=end['final_map'] or end['final_map']!=audit['final_map']:raise ValueError('Placement changed')
  if end['migration_sent_bytes_local'] or end['migration_sent_messages_local'] or end['pending_result'] or end['rebalanced']:raise ValueError('Unexpected movement or pending work')
  if not all(audit[k] for k in ('resident_state_verified','normal_completion_verified','rank_map_agreement')):raise ValueError('Missing full-slot native state audit')
  if start['mapper_calls']!=0 or end['mapper_calls']!=(0 if a=='stockoff' else 512):raise ValueError('Wrong measured mapper count')
  if a=='current_online':
   if end['controller_step_calls']!=512 or len(end['decisions'])!=8:raise ValueError('Current control not retained')
   for k in ('movement_calls','commit_calls','agreement_calls','movement_skips','commit_skips'):
    if end['guard'][k]!=8:raise ValueError('Current shortcut event count differs')
  elif a=='static_online':
   if end['controller_step_calls'] or end['static_step_calls']!=512 or end['selector_kind']!='fused' or not audit['static_state_unchanged']:raise ValueError('Fused static contract differs')
  elif end['controller_step_calls'] or end['physical_slots']!=128:raise ValueError('Stock-off contract differs')
 phase=np.arange(512)%64
 return {'mean_service_ms':float(service.mean()),'completed_elapsed_ms':completed,
  'ordinary_nonrecording_ms':float(service[phase<48].mean()),'ordinary_recording_ms':float(service[(phase>=48)&(phase<63)].mean()),'boundary_ms':float(service[phase==63].mean())}


def summarize(rows):
 cells={};identities={};orders={}
 for r in rows:
  key=(r['group'],r['repeat'],r['variant'])
  if key in cells:raise ValueError('Duplicate trial')
  cells[key]=extract(r)
  identity={k:r[k] for k in ('dataset','bank_sha256','indices_sha256','logical_routes_sha256','exact_physical_assignment_sha256')}
  if key[0] in identities and identities[key[0]]!=identity:raise ValueError('Paired input/assignment changed')
  identities[key[0]]=identity
  if key[:2] in orders and orders[key[:2]]!=r['arm_order']:raise ValueError('Inconsistent order')
  orders[key[:2]]=r['arm_order']
 groups=sorted(identities)
 if len(groups)!=6 or set(cells)!={(g,r,a) for g in groups for r in (0,1) for a in ARMS}:raise ValueError('Require all36 declared trials')
 for g in groups:
  if orders[g,1]!=orders[g,0][::-1]:raise ValueError('Repetition order not reversed')
 positions={a:[sum(order[i]==a for order in orders.values()) for i in range(3)] for a in ARMS}
 if any(counts!=[4,4,4] for counts in positions.values()):raise ValueError('Arm-position balance differs')
 means={g:{a:{m:float(np.mean([cells[g,r,a][m] for r in (0,1)])) for m in next(iter(cells.values()))} for a in ARMS} for g in groups}
 contrasts={}
 for control in ('current_online','stockoff'):
  by_metric={}
  for metric in ('mean_service_ms','completed_elapsed_ms'):
   ratios={g:means[g]['static_online'][metric]/means[g][control][metric] for g in groups}
   savings={g:means[g][control][metric]-means[g]['static_online'][metric] for g in groups}
   by_metric[metric]={'ratio':float(np.exp(np.mean(np.log(list(ratios.values()))))),
    'observed_group_ratio_range':[min(ratios.values()),max(ratios.values())],'per_group_ratios':ratios,
    'arithmetic_mean_saved_ms':float(np.mean(list(savings.values()))),'per_group_saved_ms':savings}
  contrasts['static_fused_over_'+control]=by_metric
 return dict(schema='resident_fix_fused_development_analysis_v1',complete=True,trials=36,content_groups=6,
  technical_repetitions=2,descriptive_only=True,confidence_intervals=False,hypothesis_tests=False,
  estimand='Arithmetic average of two technical repetitions within each content group, then geometric mean of six paired group ratios',
  group_means=means,arm_position_counts=positions,source_pair_identity=identities,contrasts=contrasts,
  arithmetic_arm_mean_ms={a:{m:float(np.mean([means[g][a][m] for g in groups])) for m in next(iter(cells.values()))} for a in ARMS},
  limitations=['Old development inputs only; not a held-out confirmation.',
   'The current-vs-fused matched contrast changes both controller and selector implementation, while preserving the physical-assignment policy.',
   'Fused160-slot residency and stock-off128-slot residency have different memory budgets.',
   'Service/completed intervals include common host/distributed orchestration; they are not GPU-kernel residence.',
   'Recorded GPU checks are audited, not externally rerun by this CPU script.'])


def expected_adversarial():
 b=128
 ties=np.tile(np.arange(8,dtype=np.int32),(b,1));single=np.tile(np.array([32,33,34,35,80,81,82,83],np.int32),(b,1))
 asym=np.empty((b,8),np.int32)
 for row in range(b):asym[row,0]=row%32;asym[row,1:]=32+(row*7+np.arange(7))%48
 boundary=np.tile(np.arange(32,40,dtype=np.int32),(b,1))
 for c,n in enumerate((63,64,65)):boundary[:n,c]=c
 hot=np.tile(np.array([0,32,33,34,80,81,82,83],np.int32),(b,1));mixed=ties.copy();mixed[1::2]=single[1::2]
 partial=ties.copy();partial[64:]=-1
 return dict(shared_count_ties=(ties,128),singleton_only=(single,128),asymmetric_singleton_owners=(asym,128),tile_boundary_counts=(boundary,128),
  hot_shared_with_forced_owners=(hot,128),shared_and_singleton_ties=(mixed,128),all_logical_experts=((np.arange(1024).reshape(128,8)%128).astype(np.int32),128),
  partial_real_rows=(partial,64),zero_real_rows=(np.full((128,8),-1,np.int32),0))


def oracle(ids,real):
 counts=np.bincount(ids[:real].ravel(),minlength=128);fixed=[int(np.count_nonzero(counts[32:80])),int(np.count_nonzero(counts[80:128]))]
 active=fixed.copy();loads=[int(counts[32:80].sum()),int(counts[80:128].sum())]
 shared=sorted([e for e in range(32) if counts[e]],key=lambda e:(-int(counts[e]),e));cap=max(*active,(sum(active)+len(shared)+1)//2);homes=np.arange(128)
 for e in shared:
  rank=min([r for r in (0,1) if active[r]<cap],key=lambda r:(loads[r],r));homes[e]=e+128*rank;active[rank]+=1;loads[rank]+=int(counts[e])
 physical=np.full(ids.shape,-1,np.int64);valid=(ids>=0)&(ids<128);physical[valid]=homes[ids[valid]]
 return dict(physical_ids=physical,homes=homes,logical_counts=counts,summary=np.asarray([cap,*fixed,len(shared),*loads,*active]))


def proof_audit(path):
 import hashlib
 proof=json.loads(path.read_text());cases=expected_adversarial()
 for k in ('passed','rank_agreement','stream_context_unchanged','internal_real_tokens_unchanged','input_ids_unchanged','external_real_tokens_unchanged'):
  if proof[k] is not True:raise ValueError('Adversarial proof flag failed: '+k)
 if proof['gpu_invocations_checked']!=18 or proof['internal_real_tokens']!=128 or proof['reference_count']!=2:raise ValueError('Wrong proof extent')
 if proof['evidence_sha256']!=base.digest(proof['cases']):raise ValueError('Changed adversarial proof digest')
 seen=set()
 for result in proof['cases']:
  key=(result['case'],result['dtype'])
  if key in seen or key[0] not in cases or key[1] not in ('int32','int64'):raise ValueError('Invalid proof case')
  seen.add(key);ids,real=cases[key[0]]
  if result['real_tokens']!=real:raise ValueError('Proof mask differs')
  for field,array in dict(input_ids=ids,**oracle(ids,real)).items():
   h=hashlib.sha256(np.ascontiguousarray(array,dtype='<i8').tobytes()).hexdigest()
   if result[field+'_sha256']!=h:raise ValueError('Adversarial field differs from independent oracle: '+field)
 if len(seen)!=18:raise ValueError('Incomplete adversarial proof')
 return {'passed':True,'cases':18,'field_hashes_independently_recomputed':90,'proof_sha256':base.sha(path)}


def cross_run(fused,previous):
 if set(fused['group_means'])!=set(previous['group_means']):raise ValueError('Cross-run groups differ')
 for g in fused['group_means']:
  if fused['source_pair_identity'][g]!=previous['source_pair_identity'][g]:raise ValueError('Cross-run input/assignment identity differs')
 output={}
 for anchor in ('stockoff','current_online'):
  pairs={g:(fused['group_means'][g]['static_online']['mean_service_ms']/fused['group_means'][g][anchor]['mean_service_ms'])/(previous['group_means'][g]['static_online']['mean_service_ms']/previous['group_means'][g][anchor]['mean_service_ms']) for g in fused['group_means']}
  shifts={g:fused['group_means'][g][anchor]['mean_service_ms']/previous['group_means'][g][anchor]['mean_service_ms'] for g in pairs}
  output[anchor]={'normalized_fused_over_three_kernel_ratio':float(np.exp(np.mean(np.log(list(pairs.values()))))),
   'observed_group_range':[min(pairs.values()),max(pairs.values())],'per_group':pairs,'anchor_between_run_ratios':shifts}
 return {'clean_matched_comparison':False,'description':'Ratio of within-run relative service ratios; adjustment assumes anchor drift is informative and cannot eliminate run/order/thermal confounding.',
  'same_saved_input_and_assignment_hashes':True,'anchors':output}


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--records',type=Path,required=True);p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--previous-factorial',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--markdown',type=Path);a=p.parse_args()
 rows=base.read(a.records);result=summarize(rows)
 result['artifact_audit']=base.audit_artifacts(a.artifact_root,rows,result)
 result['adversarial_proof_audit']=proof_audit(a.artifact_root/'fused-adversarial.json')
 if a.previous_factorial:result['cross_run_three_kernel_context']=cross_run(result,json.loads(a.previous_factorial.read_text()))
 result.update(records_sha256=base.sha(a.records),script_sha256=base.sha(Path(__file__)),independent_helper_sha256=base.sha(Path(base.__file__)),numpy_version=np.__version__)
 a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 if a.markdown:
  lines=['# Fused development audit','',result['estimand']+'. No confidence intervals or tests.','',
   '| Matched contrast | Service ratio | Completed-elapsed ratio | Observed service group range |','|---|---:|---:|---:|']
  for name,stats in result['contrasts'].items():
   r=stats['mean_service_ms'];c=stats['completed_elapsed_ms'];lo,hi=r['observed_group_ratio_range']
   lines.append(f"| {name} | {r['ratio']:.6f} | {c['ratio']:.6f} | [{lo:.6f}, {hi:.6f}] |")
  lines+=['','All36 records, both ranks, archived source, input/assignment identities and18 adversarial case/dtype checks passed the independent CPU audit.','']
  if a.previous_factorial:
   lines+=['Cross-run three-kernel context is not a clean matched comparison:']
   for anchor,stats in result['cross_run_three_kernel_context']['anchors'].items():lines.append(f"- {anchor}-normalized fused/three-kernel service ratio: {stats['normalized_fused_over_three_kernel_ratio']:.6f}.")
   lines+=['']
  lines += [f'- {s}' for s in result['limitations']]
  a.markdown.write_text('\n'.join(lines)+'\n')
 print(json.dumps({'complete':True,'trials':36,'contrasts':result['contrasts']},indent=2))

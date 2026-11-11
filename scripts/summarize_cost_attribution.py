import json,hashlib,collections,statistics
from pathlib import Path
root=Path('results/full-serving/cost-isolation-v1');pilots=json.loads((root/'pilot-audit-v1/pilot-summary.json').read_text());profiles=json.loads((root/'pilot-audit-v1/profile-summary.json').read_text())
assert pilots['status']=='completed_pilots'
ref=pilots['arms']['off']['runtime']['identities'];reference_checkpoint=None
for arm,data in pilots['arms'].items():
 p=root/'pilot-v1'/arm/'checkpoint-verified.json';checkpoint=json.loads(p.read_text());checkpoint.pop('checked_at')
 if reference_checkpoint is None:reference_checkpoint=checkpoint
 assert checkpoint==reference_checkpoint
 for i,d in enumerate(data['runtime']['identities']):
  assert d['checkpoint_manifest_sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
  assert {k:v for k,v in d.items() if k!='checkpoint_manifest_sha256'}=={k:v for k,v in ref[i].items() if k!='checkpoint_manifest_sha256'}
rows=[]
for arm,data in pilots['arms'].items():
 for ds in ('longbench','numinamath'):
  p=data['phases'][ds+'-warmup-01'];s=p['summary'];assert p['native_phase']['available']
  rows.append({'arm':arm,'dataset':ds,'phase':'warmup-01','timing_admissible_for_attribution':data['timing_admissible_for_attribution'],'requests':s['completed_requests'],'input_tokens':s['completed_input_tokens'],'output_tokens':s['completed_output_tokens'],'mean_response_ms':s['response_from_schedule_ms']['mean'],'p95_response_ms':s['response_from_schedule_ms']['p95'],'completed_elapsed_s':s['completed_elapsed_s'],'prefill_fraction':p['native_phase']['prefill_fraction']})
contrasts={}
for ds in ('longbench','numinamath'):
 m={r['arm']:r for r in rows if r['dataset']==ds};contrasts[ds]={}
 for pair in ('fixed0_record/fixed0_off','fixed32_record/fixed32_off','fixed0_off/off','fixed32_off/fixed0_off','balanced32_off/fixed32_off','resident_v2/balanced32_off'):
  a,b=pair.split('/');contrasts[ds][pair]={k:m[a][k]/m[b][k] for k in ('mean_response_ms','p95_response_ms','completed_elapsed_s')}
groups=collections.defaultdict(list)
for r in profiles['files']:
 a=r['file'].split('/pilot-v1/')[1].split('/')[0];groups[(a,r['dataset'])].append(r)
aggregates=[]
for (a,ds),rs in sorted(groups.items()):
 assert len(rs)==8 and {r['rank'] for r in rs}==set(range(8)) and ds in ('longbench','numinamath')
 aggregates.append({'arm':a,'dataset':ds,'ranks':8,'mean_rank_kernel_union_us':statistics.mean(r['kernel_union_us'] for r in rs),'mean_rank_summed_kernel_us':statistics.mean(r['summed_kernel_us'] for r in rs),'mean_rank_category_sums_us':{c:statistics.mean(r['categories'][c]['sum_us'] for r in rs) for c in rs[0]['categories']},'scope':'Instrumented16 engine iterations; not same batch progression across arms. Not cross-rank critical-path decomposition.'})
out={'status':'pilot_evidence_only','runtime_identity_audit':'All hardware, software, native source and transport fingerprints equal; checkpoint manifests differ only in checked_at. All checkpoint asset/shard records equal and manifest digests matched per arm.','latency_phase':'warmup-01: logprobs disabled; no profiler active, but common Python diagnostic hooks remain. warmup-00 requests logprobs; profile phase has profiler overhead.','rows':rows,'contrasts':contrasts,'profile_aggregates':aggregates,'causal_contrasts':pilots['causal_contrasts'],'limitations':['One eight-request pilot group, fixed sequential arm order; no uncertainty or unique causal decomposition of every timing delta.','Equal output counts do not force token/routing identity. LongBench timings vary widely and do not show a consistent recording penalty.','B4_light retains original-history shadow and is excluded from cost timing contrasts.','NCCL kernel durations include rank arrival skew, peer waiting and synchronization. Collective replay removes expert compute but remains a component benchmark, not wire-time proof.','The proposed packed optimization applies to OFF too; any general transport benefit must not be attributed to the stability gate or resident dispatch policy.'], 'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'pilot-audit-v1/pilot-summary.json',root/'pilot-audit-v1/profile-summary.json',root/'pilot-audit-v1/profile-inventory.json',root/'communication-v1/packed-replay-summary.json']}}
(root/'pilot-audit-v1/attribution-summary.json').write_text(json.dumps(out,indent=2)+'\n');print('Runtime/checkpoint payloads match;16arm-dataset profiles cover all8ranks; summary written')

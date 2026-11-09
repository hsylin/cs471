"""Descriptive raw service, actual tile configuration and assignment-work proxies.

No production imports, confidence intervals, gates, or causal component timing.
The separate independent record/statistics audits remain required.
"""
import argparse,gzip,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

ARMS=('stockoff','current','fix')
DEFAULT={'BLOCK_SIZE_K':64,'BLOCK_SIZE_M':64,'BLOCK_SIZE_N':128,'GROUP_SIZE_M':1,'SPLIT_K':1,'num_stages':3,'num_warps':4}


def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def array_digest(v):
 a=np.ascontiguousarray(v);h=hashlib.sha256(digest([a.dtype.str,list(a.shape)]).encode());h.update(memoryview(a).cast('B'));return h.hexdigest()
def read(paths):
 rows=[];files=[]
 for path in paths:
  files.extend(sorted(path.rglob('*.jsonl'))+sorted(path.rglob('*.jsonl.gz')) if path.is_dir() else [path])
 for path in sorted(set(files)):
  with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f:rows.extend(json.loads(x) for x in f if x.strip())
 return rows,{str(p):sha(p) for p in sorted(set(files))}


def clock_summary(r):
 if r['n_jobs']!=4096:raise ValueError('Expected4096 measured jobs')
 starts,finishes=[np.asarray(r[k],np.int64) for k in ('coordinator_start_ns','coordinator_finish_ns')]
 if starts.shape!=(4096,) or finishes.shape!=starts.shape or np.any(finishes<=starts) or np.any(starts[1:]<finishes[:-1]):raise ValueError('Invalid raw service clocks')
 if r['completed_elapsed_ns']!=r['observation_end_ns']-r['epoch_monotonic_ns']:raise ValueError('Completed elapsed differs')
 service=(finishes-starts)/1e6;phase=np.arange(4096)%64
 masks={'nonrecording_ordinary':phase<48,'recording_ordinary':(phase>=48)&(phase<63),'boundary':phase==63,'all':np.ones(4096,bool)}
 if abs(service.mean()-r['queue_metrics']['summary_ns']['service']['mean']/1e6)>1e-10:raise ValueError('Stored service differs from clocks')
 return {name:{'jobs':int(mask.sum()),'mean_ms':float(service[mask].mean()),'total_ms':float(service[mask].sum())} for name,mask in masks.items()} | {'completed_elapsed_ms':r['completed_elapsed_ns']/1e6}


def metrics(counts,config):
 if counts.ndim!=2 or counts.shape[1] not in (128,160) or not np.issubdtype(counts.dtype,np.integer) or np.any(counts<0) or np.any(counts.sum(1)!=1024):raise ValueError('Invalid physical counts')
 bm,bn,bk=(config[k] for k in ('BLOCK_SIZE_M','BLOCK_SIZE_N','BLOCK_SIZE_K'))
 local=counts.shape[1]//2;rank=counts.reshape(len(counts),2,local)
 active=(rank>0).sum(2);tiles=((rank+bm-1)//bm).sum(2);tokens=rank.sum(2)
 # Two matmuls: [M,H]×[H,2I], then[M,I]×[I,H].
 h,i=2048,768
 tile_flops=2*bm*((2*i+bn-1)//bn*bn)*((h+bk-1)//bk*bk)+2*bm*((h+bn-1)//bn*bn)*((i+bk-1)//bk*bk)
 useful=1024*2*(h*2*i+i*h)
 total_tiles=tiles.sum(1);padded=total_tiles*bm
 return {'global_active_copies':active.sum(1),'max_rank_active_copies':active.max(1),
  'global_nonempty_m_tiles':total_tiles,'max_rank_nonempty_m_tiles':tiles.max(1),
  'max_rank_routes':tokens.max(1),'global_padded_rows':padded,
  'counted_tile_flops_proxy':total_tiles*tile_flops,'useful_matmul_flops':np.full(len(counts),useful,dtype=np.int64),
  'useful_to_counted_tile_flops_fraction':useful/(total_tiles*tile_flops)}


def summarize_vectors(values):
 return {k:{'mean':float(np.mean(v)),'min':float(np.min(v)),'max':float(np.max(v))} for k,v in values.items()}


def architecture(root,gid,records,configs):
 route_path=root/(gid+'-stock-routes.npz')
 with np.load(route_path,allow_pickle=False) as z:ids=z['logical_ids']
 if ids.shape!=(4112,128,8) or not np.issubdtype(ids.dtype,np.integer) or np.any(ids<0) or np.any(ids>=128):raise ValueError('Invalid actual route tape')
 if any(r['preflight']['logical_routing_sha256']!=array_digest(ids) for r in records):raise ValueError('Actual native route identity differs')
 logical=np.bincount((ids.astype(np.int64)+128*np.arange(4112)[:,None,None]).ravel(),minlength=4112*128).reshape(4112,128)
 counts={'stockoff':logical[16:]};artifacts={str(route_path):sha(route_path)}
 physical_ids={}
 for arm in ('current','fix'):
  path=root/(gid+'-'+arm+'-resident-diagnostics.json');data=json.loads(path.read_text())
  if len(data)!=4112:raise ValueError('Missing assignment diagnostics')
  full=np.asarray([d['physical_counts'] for d in data])
  if full.shape!=(4112,160) or not np.issubdtype(full.dtype,np.integer):raise ValueError('Wrong resident-count matrix')
  collapsed=full[:,:128].copy();collapsed[:,:32]+=full[:,128:]
  if not np.array_equal(collapsed,logical):raise ValueError('Physical counts do not preserve actual logical demand')
  physical_ids[arm]=[d['physical_sha256'] for d in data]
  for r in records:
   if r['arm']==arm and r['preflight']['physical_assignment_digest']!=digest(physical_ids[arm]):raise ValueError('Full assignment attestation differs')
  counts[arm]=full[16:];artifacts[str(path)]=sha(path)
 if physical_ids['current']!=physical_ids['fix'] or not np.array_equal(counts['current'],counts['fix']):raise ValueError('Fix changed physical assignments')
 vectors={a:metrics(counts[a],configs[a]) for a in ARMS}
 same={k:bool(np.array_equal(vectors['stockoff'][k],vectors['current'][k]) and np.array_equal(vectors['current'][k],vectors['fix'][k])) for k in ('global_active_copies','global_nonempty_m_tiles','global_padded_rows','counted_tile_flops_proxy')}
 return {'measured_batches':4096,'current_fix_exact_assignment_identity':True,'all_arms_equal_each_batch':same,
  'per_arm':{a:summarize_vectors(vectors[a]) for a in ARMS},
  'current_over_stockoff_mean_ratios':{k:float(vectors['current'][k].mean()/vectors['stockoff'][k].mean()) for k in vectors['current']},
  'artifact_hashes':artifacts}


def summarize(plan,rows,inputs):
 if plan['content_sha256']!=digest({k:v for k,v in plan.items() if k!='content_sha256'}):raise ValueError('Plan seal differs')
 if plan['phase'] not in ('pilot','final'):raise ValueError('Descriptive treatment summary requires pilot or final plan')
 expected={(c['block_id'],c['arrival_type'],c['rho'],a) for c in plan['cases'] for a in ARMS};seen={};configs={};inventory=Counter();trials=[];by_group=defaultdict(list)
 for r in rows:
  k=tuple(r[x] for x in ('content_group_id','arrival_type','rho','arm'))
  if k not in expected or k in seen:raise ValueError('Unexpected/duplicate trial')
  if r['schema']!='resident_fix_trial_v1' or r['status']!='complete' or r['source_sha256']!=plan['source_sha256'] or r['plan_sha256']!=plan['content_sha256']:raise ValueError('Wrong trial identity')
  if r['record_id']!=digest({k:v for k,v in r.items() if k!='record_id'}):raise ValueError('Record digest differs')
  seen[k]=r;arm=r['arm'];by_group[k[0]].append(r)
  for rank,p in enumerate(r['per_rank_state']):
   for scope in ('start','end','audit'):
    s=p[scope];cfg=s['resolved_kernel_config']
    if s['kernel_override_applied'] or cfg!=DEFAULT:raise ValueError('Observed native config differs from verified B128 defaults; do not reuse equal-work interpretation')
    if s['local_slots']!=(64 if arm=='stockoff' else 80):raise ValueError('Changed physical geometry')
    if arm in configs and configs[arm]!=cfg:raise ValueError('Config varies between arms/ranks/records')
    configs[arm]=cfg
    if scope=='end':inventory[(arm,rank,s['local_slots'])]+=1
  trials.append(dict(content_group_id=k[0],dataset=r['dataset'],arrival_type=k[1],rho=k[2],arm=arm,**clock_summary(r)))
 if set(seen)!=expected:raise ValueError('Require complete declared treatment matrix')
 grouped={}
 for scope in ('main','mbpp_sentinel'):
  subset=[t for t in trials if (t['dataset']=='mbpp')==(scope=='mbpp_sentinel')]
  grouped[scope]={}
  for arm in ARMS:
   group=[t for t in subset if t['arm']==arm]
   grouped[scope][arm]={'trials':len(group),'content_groups':len({t['content_group_id'] for t in group}),
    **{phase:{'jobs_per_trial':group[0][phase]['jobs'],'mean_service_ms':float(np.mean([t[phase]['mean_ms'] for t in group])),
      'mean_total_service_ms':float(np.mean([t[phase]['total_ms'] for t in group]))} for phase in ('all','boundary','recording_ordinary','nonrecording_ordinary')}}
  grouped[scope]['descriptive_phase_differences']={control:{phase:{'fix_minus_control_ms_per_job':grouped[scope]['fix'][phase]['mean_service_ms']-grouped[scope][control][phase]['mean_service_ms'],
   'fix_minus_control_ms_per_trial':grouped[scope]['fix'][phase]['mean_total_service_ms']-grouped[scope][control][phase]['mean_total_service_ms']} for phase in ('all','boundary','recording_ordinary','nonrecording_ordinary')} for control in ('stockoff','current')}
 architectural={g:architecture(inputs,g,rs,configs) for g,rs in sorted(by_group.items())}
 launch={}
 for a in ARMS:
  e=128 if a=='stockoff' else 160;bm,bn=64,128;allocated=1024+e*(bm-1);mp=(allocated+bm-1)//bm
  launch[a]={'global_physical_experts':e,'allocated_sorted_id_elements_source_derived':allocated,
   'allocated_m_grid_positions_source_derived':mp,'gemm1_grid_programs_per_rank_source_derived':mp*((1536+bn-1)//bn),
   'gemm2_grid_programs_per_rank_source_derived':mp*((2048+bn-1)//bn),
   'early_return_programs_included':True,'measured_profiler_launch_counts':False}
 return dict(schema='resident_fix_descriptive_service_architecture_v1',phase=plan['phase'],complete=True,trials=len(rows),
  descriptive_only=True,confidence_intervals=False,hypothesis_tests=False,plan_sha256=plan['content_sha256'],source_sha256=plan['source_sha256'],
  observed_kernel_configs=configs,kernel_metadata_records=[{'arm':a,'rank':rank,'local_experts':e,'records':n} for (a,rank,e),n in sorted(inventory.items())],
  group_summaries=grouped,per_trial_service=trials,architecture_by_group=architectural,source_derived_launch_extent=launch,
  metric_definitions={'service':'coordinator_finish minus coordinator_start, includes completion/ACK',
   'phase':'Index modulo64;0..47 nonrecording ordinary,48..62 recording ordinary,63 boundary of CURRENT controller only. Other arms use matched positions.',
   'global_nonempty_m_tiles':'Sum over physical experts ceil(route_count / observed BLOCK_SIZE_M), summed once over owner ranks.',
   'counted_tile_flops_proxy':'Two matmuls with observed M/N/K tiling;2*tileM*(rounded2I*roundedH+roundedH*roundedI), summed over nonempty M tiles.',
   'useful_matmul_flops':'1024 routed rows *2*(2048*1536+768*2048); excludes activations/router/reduction.',
   'allocated_launch_extent':'Pinned source formula, includes early-return programs; not observed Nsight grid measurements.'},
  limitations=['Descriptive phase associations are not exclusive causal stage timings.',
   'Current/fix same assignment guarantees equal work proxies only with the attested common tile configuration.',
   'Equal global nonempty expert tiles do not imply equal launched programs, cache traffic, hardware instructions or runtime.',
   'Proxy FLOPs and useful fraction are not SM/tensor-core utilization, HBM bytes or roofline measurements.',
   'MBPP sentinel is reported separately and never pooled into the main summary.',
   'Independent execution/statistics audits remain required; this script adds no decision gate.'])


def self_test():
 c=np.zeros((1,128),np.int64);c[0,0]=1024
 p=np.zeros((1,160),np.int64);p[0,128]=1024
 a,b=metrics(c,DEFAULT),metrics(p,DEFAULT)
 assert a['global_nonempty_m_tiles'][0]==b['global_nonempty_m_tiles'][0]==16
 assert a['counted_tile_flops_proxy'][0]==a['useful_matmul_flops'][0]==9663676416
 c=np.full((1,128),8,np.int64);p=np.zeros((1,160),np.int64);p[:,:128]=c
 assert metrics(c,DEFAULT)['global_padded_rows'][0]==8192
 assert metrics(c,DEFAULT)['useful_to_counted_tile_flops_fraction'][0]==.125
 changed={**DEFAULT,'BLOCK_SIZE_M':32}
 assert metrics(c,changed)['global_padded_rows'][0]==4096
 print('Three work-proxy boundary tests passed')


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path);p.add_argument('--records',type=Path,nargs='+');p.add_argument('--inputs-root',type=Path);p.add_argument('--output',type=Path);p.add_argument('--markdown',type=Path);p.add_argument('--self-test',action='store_true');a=p.parse_args()
 if a.self_test:self_test();raise SystemExit
 if any(x is None for x in (a.plan,a.records,a.inputs_root,a.output)):p.error('--plan --records --inputs-root --output required')
 if a.output.exists() or (a.markdown and a.markdown.exists()):raise FileExistsError('Fresh output paths required')
 rows,files=read(a.records);result=summarize(json.loads(a.plan.read_text()),rows,a.inputs_root)
 result.update(records_files=files,script_sha256=sha(Path(__file__)),numpy_version=np.__version__)
 a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 if a.markdown:
  lines=['# Descriptive service and work proxies','',f"Complete {result['phase']} matrix: {result['trials']} trials. No confidence intervals or new tests.",'', '| Scope | Arm | All jobs ms | Boundary ms | Recording ordinary ms | Nonrecording ordinary ms |','|---|---|---:|---:|---:|---:|']
  for scope,group in result['group_summaries'].items():
   for arm in ARMS:
    t=group[arm];lines.append(f"| {scope} | {arm} | {t['all']['mean_service_ms']:.6f} | {t['boundary']['mean_service_ms']:.6f} | {t['recording_ordinary']['mean_service_ms']:.6f} | {t['nonrecording_ordinary']['mean_service_ms']:.6f} |")
  equal=all(all(g['all_arms_equal_each_batch'].values()) for g in result['architecture_by_group'].values())
  lines+=['',f'All compared arms have equal per-batch global active-copy/nonempty-tile/padded-row/count-proxy values: {equal}. See JSON for actual configuration checks and rank-imbalance summaries.','']+[f'- {x}' for x in result['limitations']]
  a.markdown.write_text('\n'.join(lines)+'\n')
 print(json.dumps({'complete':True,'trials':len(rows),'phase':result['phase']}))

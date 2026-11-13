"""Conservative linkage of Python event generations to native CUDA wait activity."""
import argparse,collections,gzip,json
from pathlib import Path
from full_serving.boundary_critical_contract import digest,write_new

def gpu_work(events):
 return [e for e in events if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset') and e.get('ph')=='X']

def api_for_scope(scope,apis,operation):
 needle={'record':'EventRecord','wait':'StreamWaitEvent','synchronize':'EventSynchronize'}[operation]
 matches=[a for a in apis if needle in a.get('name','') and a.get('pid')==scope.get('pid') and a.get('tid')==scope.get('tid') and a['ts']>=scope['ts'] and a['ts']+a.get('dur',0)<=scope['ts']+scope.get('dur',0)]
 return matches[0] if len(matches)==1 else None

def checked_record(wait,records):
 rec=records.get(wait.get('record_call_id'))
 if not rec:return None,'record_not_observed'
 if rec['event_token']!=wait['event_token']:return None,'identity_mismatch'
 if rec.get('end_outside_window') or wait.get('end_outside_window'):return None,'capture_boundary_crossing'
 if not rec.get('success') or not wait.get('success') or rec.get('overlap') or wait.get('overlap'):return None,'failed_or_concurrent'
 if rec['end_wall_ns']>wait['start_wall_ns']:return None,'record_not_completed_before_wait'
 newer=[r for r in records.values() if r['event_token']==wait['event_token'] and r['call_id']!=rec['call_id'] and r['start_wall_ns']<wait['end_wall_ns'] and r['end_wall_ns']>rec['end_wall_ns']]
 if newer:return None,'later_or_overlapping_record'
 return rec,None

def stream_identity_map(identities):
 out=collections.defaultdict(set);bounds=collections.defaultdict(set)
 for r in identities:
  key=(r['stream_device'],r['stream_handle']);bounds[key].add(r.get('identity_boundary'))
  if r.get('status')=='resolved':out[key].add((r['device'],r['context'],r['stream']))
  else:out[key].add(('unknown',None,None))
 return {k:v for k,v in out.items() if bounds[k]>={'start','end'} and all(x[0]!='unknown' for x in v)}

def analyze_values(events,links,synthetic=False):
 identities=[r for r in links if r.get('kind')=='cuda_stream_identity']
 links=[r for r in links if r.get('kind')!='cuda_stream_identity']
 queried=stream_identity_map(identities)
 scopes=collections.defaultdict(list)
 for e in events:
  if e.get('cat')=='user_annotation' and e.get('ph')=='X':scopes[e['name']].append(e)
 apis=[e for e in events if e.get('cat') in ('cuda_runtime','cuda_driver') and e.get('ph')=='X']
 api_by_corr=collections.defaultdict(list)
 for a in apis:api_by_corr[a.get('args',{}).get('correlation')].append(a)
 syncs=collections.defaultdict(list)
 for e in events:
  if e.get('cat')=='cuda_sync' and e.get('name')=='Stream Wait Event':syncs[e.get('args',{}).get('correlation')].append(e)
 rows={r['call_id']:r for r in links};assert len(rows)==len(links),'duplicate call identity'
 records={k:r for k,r in rows.items() if r['operation']=='record'}
 mapped={};native={};stream_map=collections.defaultdict(set);unknown=collections.Counter();known_ground_truth={}
 for call,r in rows.items():
  ss=scopes.get(r['annotation'],[])
  if len(ss)!=1:continue
  a=api_for_scope(ss[0],apis,r['operation'])
  if a is None:continue
  mapped[call]=(ss[0],a)
  if synthetic:
   enclosing=[s for name,vals in scopes.items() if name.startswith(('cs471::known_record::','cs471::known_wait::','cs471::return_record::','cs471::return_wait::')) for s in vals if s['pid']==ss[0]['pid'] and s['tid']==ss[0]['tid'] and s['ts']<=ss[0]['ts'] and s['ts']+s['dur']>=ss[0]['ts']+ss[0]['dur']]
   if len(enclosing)==1:known_ground_truth[call]=enclosing[0]['name']
  if r['operation']=='wait':
   ssync=syncs.get(a.get('args',{}).get('correlation'),[])
   if len(ssync)!=1:continue
   event=ssync[0];v=event['args'];native[call]=event
   key=(r['stream_device'],r['stream_handle'])
   if key in queried:
    q=queried[key]
    if len(q)!=1 or next(iter(q))[1:]!=(v.get('context'),v.get('stream')):
     native.pop(call);continue
   else:stream_map[key].add((v.get('device'),v.get('context'),v.get('stream')))
 stream_map.update(queried)
 activities=gpu_work(events)  # cuda_sync describes API functions, not GPU execution spans.
 edges=[];truth_checked=0
 for call,w in rows.items():
  if w['operation']!='wait':continue
  rec,reason=checked_record(w,records)
  if reason:unknown[reason]+=1;continue
  if call not in mapped or rec['call_id'] not in mapped or call not in native:unknown['missing_unique_API_or_native_wait']+=1;continue
  _,ra=mapped[rec['call_id']];_,wa=mapped[call]
  if ra['ts']+ra.get('dur',0)>wa['ts']:unknown['CPU_API_order_unresolved']+=1;continue
  ss=stream_map.get((rec['stream_device'],rec['stream_handle']),set())
  dest=native[call]['args'];source=next(iter(ss)) if len(ss)==1 else None
  dq=queried.get((w['stream_device'],w['stream_handle']),set())
  destination=next(iter(dq)) if len(dq)==1 else (dest.get('device'),dest.get('context'),dest.get('stream'))
  source_work=[]
  if source:
   for e in activities:
    v=e.get('args',{})
    if (v.get('device'),v.get('context'),v.get('stream'))!=source:continue
    candidates=api_by_corr.get(v.get('correlation'),[])
    if len(candidates)==1 and candidates[0]['ts']+candidates[0].get('dur',0)<=ra['ts']:source_work.append(e)
  pred=max(source_work,key=lambda e:e['ts']+e.get('dur',0)) if source_work else None
  edge=dict(event_token=w['event_token'],record_call_id=rec['call_id'],wait_call_id=call,
      record_CPU_api_interval_us=[ra['ts'],ra['ts']+ra.get('dur',0)],
      wait_CPU_api_interval_us=[wa['ts'],wa['ts']+wa.get('dur',0)],
      native_sync_CPU_interval_us=[native[call]['ts'],native[call]['ts']+native[call].get('dur',0)],
      source_activity_interval_us=[pred['ts'],pred['ts']+pred.get('dur',0)] if pred else None,
      source_activity_correlation=pred.get('args',{}).get('correlation') if pred else None,
      source_activity_name=pred.get('name') if pred else None,
      source_work_still_running_at_wait_API=pred['ts']+pred.get('dur',0)>wa['ts'] if pred else None,
      native_ordinal=w.get('ordinal'),native_thread=w.get('thread'),
      record_correlation=ra['args']['correlation'],wait_correlation=wa['args']['correlation'],
      source_stream=source,destination_stream=destination,source_identity_method='CUPTI_stream_query' if (rec['stream_device'],rec['stream_handle']) in queried else 'native_wait_observation',
      source_activity_observed=pred is not None,source_activity_category=pred.get('cat') if pred else None,
      native_record_correlation=dest.get('wait_on_cuda_event_record_corr_id'),
      cross_stream=source is not None and source!=destination)
  if synthetic:
   wn=known_ground_truth.get(call);rn=known_ground_truth.get(rec['call_id'])
   assert wn and rn and wn.replace('_wait::','_record::')==rn,(wn,rn)
   truth_checked+=1
  edges.append(edge)
 waits=sum(r['operation']=='wait' for r in links)
 return dict(observed_event_calls=len(links),observed_wait_calls=waits,linked_waits=len(edges),unknown_waits=dict(unknown),
  cross_stream_links=sum(e['cross_stream'] for e in edges),source_activity_observed=sum(e['source_activity_observed'] for e in edges),
  synthetic_ground_truth_checked=truth_checked,edges=edges,
  limits=['CUPTI cuda_sync timestamps describe synchronization API functions, not GPU stall intervals.',
   'Only Python-issued observed event generations; native/CUDA-graph internal events can be absent.',
   'Source stream mapping derives from native wait correlation, never numerical equality of stream handle and CUPTI ID.',
   'Latest source activity is a submitted predecessor candidate, not a complete dependency graph or removable time.',
   'Missing, ambiguous, concurrent and out-of-window records remain unknown; no residual latency attribution.'])

def analyze(trace,sidecar,synthetic=False):
 with gzip.open(trace,'rt') if trace.name.endswith('.gz') else trace.open() as f:events=json.load(f)['traceEvents']
 links=[json.loads(x) for x in sidecar.read_text().splitlines() if x.strip()]
 if sidecar.name.startswith('critical-rank'):
  import re
  window=re.fullmatch(r'trace-(.+)-rank[0-7]\.json(?:\.gz)?',trace.name).group(1)
  links=[r for r in links if r.get('kind') in ('cuda_event_link','cuda_stream_identity') and r.get('window')==window]
 out=analyze_values(events,links,synthetic)
 out['inputs']={str(p):dict(sha256=digest(p),bytes=p.stat().st_size) for p in (trace,sidecar)}
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('trace',type=Path);p.add_argument('sidecar',type=Path);p.add_argument('--synthetic',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 write_new(a.output,analyze(a.trace,a.sidecar,a.synthetic))
if __name__=='__main__':main()

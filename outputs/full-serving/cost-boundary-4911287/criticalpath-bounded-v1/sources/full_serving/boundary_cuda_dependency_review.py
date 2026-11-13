"""Observed local CUDA event dependency order; never a removable latency estimate."""
import argparse,collections,gzip,json
from pathlib import Path
from full_serving.boundary_cuda_event_analysis import analyze,gpu_work
from full_serving.boundary_critical_contract import digest,write_new

def stream(e):
 a=e.get('args',{});return (a.get('device'),a.get('context'),a.get('stream'))

def make_index(events):
 apis=collections.defaultdict(list)
 for e in events:
  if e.get('cat') in ('cuda_runtime','cuda_driver') and e.get('ph')=='X':apis[e.get('args',{}).get('correlation')].append(e)
 indexed=collections.defaultdict(list)
 for e in gpu_work(events):
  candidates=apis.get(e.get('args',{}).get('correlation'),[])
  if len(candidates)==1:
   a=candidates[0];indexed[stream(e)].append((a['ts'],a['ts']+a.get('dur',0),e))
 return indexed

def inspect_edge(edge,events,all_edges,index=None):
 if edge.get('source_activity_interval_us') is None:return dict(status='unknown_source_activity')
 source=edge.get('source_stream');destination=edge.get('destination_stream')
 if not source or not destination or source[0:2]!=destination[0:2]:return dict(status='unknown_or_different_device_context')
 indexed=make_index(events) if index is None else index
 before=[];after=[]
 lo,hi=edge['wait_CPU_api_interval_us']
 for item in indexed.get(tuple(destination),[]):
  if item[1]<=lo:before.append(item)
  elif item[0]>=hi:after.append(item)
 if not after:return dict(status='next_destination_activity_unobserved')
 first_api=min(a[0] for a in after);nexts=[a[2] for a in after if a[0]==first_api];nxt=min(nexts,key=lambda e:e['ts'])
 prior=max((a[2]['ts']+a[2].get('dur',0) for a in before),default=None)
 extra=[e for e in all_edges if e is not edge and e.get('destination_stream')==destination and hi<=e['wait_CPU_api_interval_us'][0]<first_api]
 source_end=edge['source_activity_interval_us'][1];next_start=nxt['ts']
 return dict(status='observed_dependency_order' if next_start>=source_end else 'timing_order_inconsistent',
  source_end_us=source_end,wait_API_return_us=hi,destination_previous_observed_activity_end_us=prior,
  next_destination_activity=dict(name=nxt['name'],category=nxt['cat'],start_us=next_start,end_us=next_start+nxt.get('dur',0),correlation=nxt.get('args',{}).get('correlation')),
  source_unfinished_at_wait_submission=source_end>hi,
  source_finishes_after_previous_observed_destination_work=source_end>prior if prior is not None else None,
  intervening_linked_event_waits=len(extra),
  exclusive_dependency_established=False,removable_time_established=False,
  limits='A recorded event orders stream work, but scheduling, graph-internal/unassigned activity and other dependencies can also delay the successor. Missing prior destination activity is unknown, never idle.')

def review(trace,sidecar,links_path=None):
 links=json.loads(links_path.read_text()) if links_path else analyze(trace,sidecar)
 for p in (trace,sidecar):assert links['inputs'][str(p)]['sha256']==digest(p)
 with gzip.open(trace,'rt') if trace.name.endswith('.gz') else trace.open() as f:events=json.load(f)['traceEvents']
 index=make_index(events)
 details=[dict(event_token=e['event_token'],wait_call_id=e['wait_call_id'],native_ordinal=e.get('native_ordinal'),cross_stream=e['cross_stream'],**inspect_edge(e,events,links['edges'],index)) for e in links['edges']]
 counts=collections.Counter(d['status'] for d in details)
 return dict(status='descriptive_dependency_review',counts=dict(counts),linked_events=links['linked_waits'],cross_stream_links=links['cross_stream_links'],
  source_unfinished_at_submission=sum(d.get('source_unfinished_at_wait_submission') is True for d in details),
  source_after_previous_observed_destination=sum(d.get('source_finishes_after_previous_observed_destination_work') is True for d in details),
  exclusive_dependency_established=False,residual_latency_attributed=False,details=details,inputs=links['inputs'],link_receipt_sha256=digest(links_path) if links_path else None)

def main():
 p=argparse.ArgumentParser();p.add_argument('trace',type=Path);p.add_argument('sidecar',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--links',type=Path);a=p.parse_args();write_new(a.output,review(a.trace,a.sidecar,a.links))
if __name__=='__main__':main()

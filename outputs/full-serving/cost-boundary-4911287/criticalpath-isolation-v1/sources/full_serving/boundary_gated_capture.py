"""Single bounded continuous profiler session, logical windows only; no serving policy changes."""
import gzip,json
from collections import Counter
from full_serving.boundary_critical_contract import digest,write_new

def slice_events(events,start_name,end_name):
    anchors={n:[e for e in events if e.get('cat')=='user_annotation' and e.get('name')==n and e.get('ph')=='X'] for n in (start_name,end_name)}
    assert all(len(x)==1 for x in anchors.values()),'Missing or duplicate CPU anchors'
    start=anchors[start_name][0]['ts'];last=anchors[end_name][0];end=last['ts']+last['dur'];assert start<end
    cpu=[e for e in events if e.get('ph')=='X' and e.get('cat') not in ('kernel','gpu_memcpy','gpu_memset','cuda_sync') and start<=e['ts'] and e['ts']+e.get('dur',0)<=end]
    keys={(e.get('pid'),e.get('args',{}).get('correlation')) for e in cpu if e.get('cat') in ('cuda_runtime','cuda_driver') and e.get('args',{}).get('correlation') is not None}
    # CUPTI GPU pid differs from host pid; require unique host pid for a correlation.
    host_pid_counts=Counter(c for _,c in keys)
    corr={c for c,count in host_pid_counts.items() if count==1}
    activity=[e for e in events if e.get('ph')=='X' and e.get('cat') in ('kernel','gpu_memcpy','gpu_memset','cuda_sync') and (e.get('args',{}).get('correlation') in corr or (e['ts']<end and e['ts']+e.get('dur',0)>start))]
    metadata=[e for e in events if e.get('ph')=='M']
    return metadata+cpu+activity,dict(start_us=start,end_us=end,cpu_events=len(cpu),activity_events=len(activity),rule='Complete CPU scopes between anchors; associated GPU activity including overhang plus all overlapping unassigned activity; no cuda_sync GPU-duration interpretation')

def export_slices(root,rank,session,windows):
    with gzip.open(session,'rt') as f:doc=json.load(f)
    receipts={}
    for w in windows:
        selected,bounds=slice_events(doc['traceEvents'],w['start_anchor'],w['end_anchor'])
        path=root/f"trace-{w['name']}-rank{rank}.json.gz";assert not path.exists()
        value={k:v for k,v in doc.items() if k!='traceEvents'};value['traceEvents']=selected
        with gzip.open(path,'xt') as f:json.dump(value,f)
        receipts[w['name']]=dict(path=path.name,sha256=digest(path),**bounds)
    write_new(root/f'trace-slice-receipt-rank{rank}.json',dict(session_sha256=digest(session),windows=receipts,raw_private=True))
    return receipts

def gated_capture_class(Base):
 class GatedCapture(Base):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw);self.finished_windows=[];self.session_closed=False
    def start(self,window):
        assert not self.session_closed and self.active is None and self.ordinal==window['start_ordinal']
        self.boundary('start_agreed',window=window['name'],end_ordinal=window['end_ordinal']);self.barrier()
        self.used.add(window['name']);self.active=window
        activities=[self.torch.profiler.ProfilerActivity.CPU,self.torch.profiler.ProfilerActivity.CUDA]
        if self.profile is None:
            self.profile=self.torch.profiler.profile(activities=activities,record_shapes=False,profile_memory=False,with_stack=False,with_flops=False,experimental_config=self.torch._C._profiler._ExperimentalConfig(enable_cuda_sync_events=False,profile_all_threads=False))
            self.profile.start();self.boundary('session_started',window=window['name'])
        else:pass # Same continuous capture; never re-enable collection.
        if hasattr(self,'event_ledger'):
            from full_serving.boundary_cupti_streams import StreamIdentity
            if not hasattr(self,'stream_resolver'):self.stream_resolver=StreamIdentity()
            with self.event_ledger.lock:streams=list(self.event_ledger.streams)
            for device,handle in streams:self.emit('cuda_stream_identity',window=window['name'],identity_boundary='start',**self.stream_resolver.resolve(device,handle))
        self.anchor();window['start_anchor']=f'cs471::clock_anchor::{self.anchor_id}'
        self.boundary('start_complete',window=window['name']);self.emit('collection_enabled',window=window['name'])
    def stop(self):
        assert self.ordinal==self.active['end_ordinal']
        window=dict(self.active);name=window['name'];self.boundary('stop_agreed',window=name);self.barrier()
        self.torch.cuda.synchronize() # diagnostic boundary only, outside execute scopes
        if hasattr(self,'event_ledger'):
            with self.event_ledger.lock:streams=list(self.event_ledger.streams)
            for device,handle in streams:self.emit('cuda_stream_identity',window=name,identity_boundary='end',**self.stream_resolver.resolve(device,handle))
        self.anchor();window['end_anchor']=f'cs471::clock_anchor::{self.anchor_id}'
        activities=[self.torch.profiler.ProfilerActivity.CPU,self.torch.profiler.ProfilerActivity.CUDA]
        # Logical boundary only: no activity collection toggle.
        self.active=None;self.finished_windows.append(window);self.boundary('logical_window_complete',window=name)
        self.flush()
        if all(w['name'] in self.used for w in self.plan['windows']):
            self.profile.stop();self.session_closed=True
            session=self.root/f'capture-session-rank{self.rank}.json.gz';assert not session.exists();self.profile.export_chrome_trace(str(session))
            self.boundary('session_stopped',windows=len(self.finished_windows))
            receipts=export_slices(self.root,self.rank,session,self.finished_windows)
            for w in self.finished_windows:
                self.boundary('export_complete',window=w['name'],capture_end_ordinal=w['end_ordinal'],trace=receipts[w['name']]['path'],deferred_session_export=True)
            self.profile=None
        self.barrier()
    def tick(self):
        if self.session_closed:self.ordinal+=1;return
        super().tick()
 return GatedCapture

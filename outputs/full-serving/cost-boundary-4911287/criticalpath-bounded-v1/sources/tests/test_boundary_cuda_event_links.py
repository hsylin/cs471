from full_serving.boundary_cuda_event_links import EventLedger
class Event:pass

def test_reused_event_wait_binds_latest_completed_record():
 l=EventLedger();e=Event();r=l.begin(e,'record');l.end(e,r,True)
 w=l.begin(e,'wait');assert w['record_call_id']==r['call_id'];l.end(e,w,True)
 r2=l.begin(e,'record');l.end(e,r2,True);w2=l.begin(e,'wait');assert w2['record_call_id']==r2['call_id']

def test_unrecorded_and_failed_event_unknown():
 l=EventLedger();e=Event();w=l.begin(e,'wait');assert w['record_call_id'] is None;l.end(e,w,True)
 r=l.begin(e,'record');l.end(e,r,False);w=l.begin(e,'wait');assert w['record_call_id'] is None

def test_concurrent_record_wait_invalidates_link_both_directions():
 l=EventLedger();e=Event();r=l.begin(e,'record');w=l.begin(e,'wait');l.end(e,r,True)
 assert l.end(e,w,True)['record_call_id'] is None
 w=l.begin(e,'wait');assert w['record_call_id'] is None;l.end(e,w,True)
 r=l.begin(e,'record');l.end(e,r,True);w=l.begin(e,'wait');r2=l.begin(e,'record')
 assert l.end(e,w,True)['record_call_id'] is None;l.end(e,r2,True)

def test_event_identity_distinct_and_not_retained():
 import weakref,gc
 l=EventLedger();a=Event();b=Event();x=l.begin(a,'record');l.end(a,x,True);y=l.begin(b,'record');l.end(b,y,True)
 assert x['event_token']!=y['event_token'];ref=weakref.ref(a);del a;gc.collect();assert ref() is None

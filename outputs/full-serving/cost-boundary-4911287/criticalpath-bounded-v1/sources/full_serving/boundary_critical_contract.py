"""Bounded, private profiling diagnostics; never a final performance comparison."""
import hashlib
import json

CONTROLS = ('off', 'b4', 'fixed_idle', 'fixed_charged')
WINDOWS = (('onset', 0.0), ('early', .15), ('middle', .50), ('late', .85))

def windows(epoch_ns, horizon_ns, pilot=False):
    assert epoch_ns > 0 and horizon_ns >= 0
    if pilot:
        return [dict(name=n,start_ns=epoch_ns+offset,end_ns=epoch_ns+offset+3_000_000_000,native_iterations=192 if n=='repeat2' else 64) for n,offset in (('admission',0),('repeat1',0),('repeat2',0))]
    assert horizon_ns > 60_000_000_000
    return [dict(name=n, start_ns=epoch_ns+int(f*horizon_ns),
                 end_ns=epoch_ns+int(f*horizon_ns)+3_000_000_000) for n,f in WINDOWS]

def selected_window(plan, now, used):
    return next((w for w in plan if w['name'] not in used and w['start_ns'] <= now < w['end_ns']), None)

def interval_union(intervals):
    result=[]
    for a,b in sorted(intervals):
        assert b >= a
        if result and a <= result[-1][1]: result[-1][1]=max(b,result[-1][1])
        else: result.append([a,b])
    return result

def clock_interval(sent, remote, received):
    assert sent <= received
    # Remote clock minus controller clock; no symmetric-network assumption.
    return [remote-received,remote-sent]

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def write_new(path, value):
    with open(path,'x') as f: json.dump(value,f,indent=2); f.write('\n')


def requested_window(plan, now, used, proposal_pending=False):
    """Rank0 selects one due window; all ranks execute the agreed ordinal command."""
    if plan is None or now<plan['epoch_wall_ns']:return 0
    for i,w in enumerate(plan['windows'],1):
        if w['name'] not in used and now>=w['start_ns']:return i
    if proposal_pending and 'proposal' not in used:return len(plan['windows'])+1
    return 0

def agreed_window(plan, choice, ordinal):
    assert ordinal>0 and ordinal%64==0
    assert 1<=choice<=len(plan['windows'])+1
    if choice<=len(plan['windows']):
        w=dict(plan['windows'][choice-1]);length=int(w.get('native_iterations',64));assert length in (64,192)
    else:w=dict(name='proposal');length=192
    return dict(w,start_ordinal=ordinal,end_ordinal=ordinal+length,
                capture_bound='native_dispatch_iterations',diagnostic_only=True)

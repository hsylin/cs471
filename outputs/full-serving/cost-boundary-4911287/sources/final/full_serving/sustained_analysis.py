"""Independent raw arithmetic, runtime identity and censored migration evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from collections import Counter
from full_serving.analyze import load_run, quantile
from full_serving.report_pilots import metric_deltas
from full_serving.custom_report import phase_balance
from full_serving.proposal_analysis import gate_audit, gate_phase_evidence, logged_config, logged_replica_count
from full_serving.sustained_experiment import ARMS, server_arguments


def runtime(root,arm):
    from full_serving.custom_runtime import runtime_identity,engine_configuration,errors_before_shutdown
    texts={f'server-{n}.{s}':(root/f'server-{n}.{s}').read_text() for n in range(2) for s in ('log','err')}
    assert errors_before_shutdown(texts)==0
    identities=[runtime_identity(json.loads((root/f'preflight-{r}.json').read_text()),r) for r in ('primary','worker')]
    assert engine_configuration(texts['server-0.log'])==engine_configuration(texts['server-1.log'])
    cfg=logged_config(texts['server-0.log'])
    assert bool(cfg.get('enable_eplb',False))==(arm!='off')
    if arm!='off':assert logged_replica_count(cfg)==(0 if arm=='b1' else 32)
    for role in ('primary','worker'):
        launch=json.loads((root/f'effective-launch-{role}.json').read_text());cmd=launch['command']
        assert launch['arm']==arm and launch['plugin']==(None if arm=='off' else 'cs471_sustained')
        assert cmd==['vllm',*server_arguments(arm,cmd[2],cmd[cmd.index('--data-parallel-address')+1],role)]
    if arm=='resident_v2':
        for rank in range(8):
            b=json.loads((root/f'resident-bindings-rank{rank}.json').read_text())
            assert b['ep_rank']==rank and b['layer_count']==48 and not b['native_control_step_active']
    return identities


def raw_phase(path,item,cap=8192,calendar=None,targets=None):
    rows=[json.loads(x) for x in path.read_text().splitlines()]
    h=[r for r in rows if r.get('type')=='run_header'];t=[r for r in rows if r.get('type')=='run_summary']
    rr=sorted([r for r in rows if r.get('type')=='request'],key=lambda r:r['request_index'])
    assert len(h)==len(t)==1 and len(rr)==item['request_count'] and len(rows)==len(rr)+2
    h,t=h[0],t[0]
    assert h['requests_sha256']==item['requests_sha256'] and h['max_tokens']==cap
    assert h['ignore_eos']==(targets is not None)
    assert [r['request_id'] for r in rr]==item['request_ids']
    offsets=calendar['arrival_offsets_ns'] if calendar else [0]*len(rr)
    if calendar:assert h['calendar_sha256']==item['calendar_sha256']
    assert t['valid'] and not t['failed_requests']
    for r,o in zip(rr,offsets):
        assert r['success'] and r['saw_done'] and r['http_status']==200
        assert r['scheduled_ns']==t['epoch_monotonic_ns']+o
        assert r['response_from_schedule_ns']==r['end_ns']-r['scheduled_ns']
        assert r['response_from_send_ns']==r['end_ns']-r['send_ns']
        assert r['expected_input_tokens']==r['usage']['prompt_tokens']
        assert 0<r['usage']['completion_tokens']<=cap and r['usage']['total_tokens']==r['usage']['prompt_tokens']+r['usage']['completion_tokens']
        assert r['finish_reason'] in ('length','stop')
        if targets:assert r['usage']['completion_tokens']==targets[r['request_id']]==r['requested_output_tokens'] and r['finish_reason']=='length'
        elif r['finish_reason']=='length':assert r['usage']['completion_tokens']==cap
    last=max(r['end_ns'] for r in rr);horizon=calendar['horizon_ns'] if calendar else 0
    elapsed=max(last,t['epoch_monotonic_ns']+horizon)-t['epoch_monotonic_ns']
    drain=max(0,last-t['epoch_monotonic_ns']-horizon)
    assert elapsed==t['completed_elapsed_ns'] and drain==t['drain_ns']
    summary=load_run(path)['summary'];assert summary['valid']
    mean=math.fsum((r['end_ns']-r['scheduled_ns'])/1e6 for r in rr)/len(rr)
    assert math.isclose(mean,summary['response_from_schedule_ms']['mean'],rel_tol=1e-12)
    return {'header':h,'timing':t,'summary':summary,'independent_arithmetic':'passed'},rr


def migrations(root):
    peers=[];bytes_received=0;copies=[]
    for rank in range(8):
        rows=[json.loads(x) for x in (root/f'migration-events-rank{rank}.jsonl').read_text().splitlines()]
        plans={r['step']:r for r in rows if r['kind']=='proposal'}
        enq={(r['cycle_step'],r['layer']):r for r in rows if r['kind']=='commit_enqueued'}
        done={(r['cycle_step'],r['layer']):r for r in rows if r['kind']=='commit_completed'}
        assert done.keys()<=enq.keys()
        for key,r in done.items():
            assert r['map_sha256']==enq[key]['map_sha256'] and r['wall_ns']>=enq[key]['wall_ns']
            import numpy as np
            expected=hashlib.sha256(np.asarray(plans[key[0]]['admitted'][key[1]],dtype=np.int64).tobytes()).hexdigest()
            assert r['map_sha256']==expected
            copies.append(r['gpu_copy_ms'])
        bytes_received+=sum(r['remote_received_tensor_bytes'] for r in enq.values())
        peers.append({'plans':plans,'enqueued':enq,'completed':done})
    for p in peers[1:]:
        assert p['plans'].keys()==peers[0]['plans'].keys()
        for k in p['plans']:assert p['plans'][k]['admitted']==peers[0]['plans'][k]['admitted']
    cycles=[]
    for step,plan in peers[0]['plans'].items():
        common=set.intersection(*[set(p['completed']) for p in peers]);keys=[k for k in common if k[0]==step]
        done_rank0=[peers[0]['completed'][k]['wall_ns'] for k in keys]
        cycles.append({'step':step,'proposal_wall_ns_rank0':plan['wall_ns'],
            'changed_slots':plan['changed_slots'],'all_rank_completed_layers':len(keys),
            'uncompleted_or_unobserved_layers':48-len(keys),
            'last_observed_commit_wall_ns_rank0':max(done_rank0) if done_rank0 else None,
            'host_plan_ns_rank0':plan['host_plan_ns'],
            'proposal_to_last_observed_commit_ms_rank0':(max(done_rank0)-plan['wall_ns'])/1e6 if done_rank0 else None})
    return {'cycles':cycles,'remote_received_tensor_bytes_all_ranks':bytes_received,
        'rank_layer_gpu_copy_ms_sum':sum(copies),'rank_agreement':True,
        'scope':'GPU-copy durations may overlap across ranks; not additive wall-time cost. Proposal-to-commit includes background waiting. Unobserved commits censored; drain is not commit proof.'}


def pilot(root,arm):
    spec=json.loads((root/'protocol.json').read_text());data=json.loads((root/'data-manifest.json').read_text())
    directory=root/'pilot'/arm;identity=runtime(directory,arm);phases={}
    for dataset in spec['datasets']:
        for label in ('warmup-00','warmup-01','pilot'):
            key=dataset+'-'+label
            phases[key]=raw_phase(directory/(key+'.jsonl'),data['datasets'][dataset]['items']['warmup'])[0]
    result={'status':'passed','runtime':identity,'phases':phases}
    if arm in ('b1','b2','b3','b4'):result['migrations']=migrations(directory)
    if arm in ('b3','b4'):
        result['gate']=gate_audit(directory,arm)
        for rank in range(8):
            rows=[json.loads(x) for x in (directory/f'gate-events-rank{rank}.jsonl').read_text().splitlines()]
            assert sum(r['kind']=='history_shadow_passed' for r in rows)==result['gate']['proposal_cycles']>0
    return result


def report(root):
    spec=json.loads((root/'protocol.json').read_text());data=json.loads((root/'data-manifest.json').read_text());status=json.loads((root/'status.json').read_text())
    assert status['state']=='comparison_completed' and len(status['completed_sessions'])==12
    result={'status':'completed_and_audited','mode':status['mode'],'datasets':{},'sessions':12,'measured_requests':6144,
        'uncertainty':'One group/session per dataset and arm; descriptive only; no confidence intervals or significance.',
        'payback_scope':'Observed policy-level cumulative latency savings include all online overhead but are not causal per-migration execution payback. Matched branch diagnostics must separately establish comparable predecision state and future work.'}
    for dataset,meta in data['datasets'].items():
        calendar=json.loads((root/(dataset+'-calendar.json')).read_text());arms={};records={}
        for arm in ARMS:
            directory=root/dataset/arm;identity=runtime(directory,arm)
            for phase in ('warmup-00','warmup-01','measured'):
                item=meta['items']['measured' if phase=='measured' else 'warmup']
                tp=root/(dataset+'-'+phase+'-lengths.json');targets=json.loads(tp.read_text()) if status['mode']=='matched' else None
                value,rr=raw_phase(directory/(phase+'.jsonl'),item,calendar=calendar if phase=='measured' else None,targets=targets)
                if phase=='measured':arms[arm]=value;records[arm]=rr
            arms[arm]['runtime']=identity
            metrics=metric_deltas(directory/'measured-metrics-before.txt',directory/'measured-metrics-after.txt')
            arms[arm]['phase_balance']=phase_balance(metrics,request_count=512)
            assert arms[arm]['phase_balance']['available']
            arms[arm]['native_metrics']=metrics
            if arm in ('b1','b2','b3','b4'):arms[arm]['migrations']=migrations(directory)
            if arm in ('b3','b4'):arms[arm]['gate']=gate_audit(directory,arm);arms[arm]['gate']['phase_evidence']=gate_phase_evidence(directory)
        contrasts={}
        for arm in ARMS:
            off=records['off'];rr=records[arm]
            assert [r['input_sha256'] for r in rr]==[r['input_sha256'] for r in off]
            cumulative=[];total=0
            for i,(a,b) in enumerate(zip(off,rr)):
                total+=(a['response_from_schedule_ns']-b['response_from_schedule_ns'])/1e6
                cumulative.append({'request_prefix':i+1,'cumulative_response_saved_ms':total,
                    'elapsed_to_complete_prefix_s':(max(r['end_ns'] for r in rr[:i+1])-arms[arm]['timing']['epoch_monotonic_ns'])/1e9})
            aa=arms[arm]['summary'];bb=arms['off']['summary']
            contrasts[arm+'/off']={'mean_response_ratio':aa['response_from_schedule_ms']['mean']/bb['response_from_schedule_ms']['mean'],
                'p95_response_ratio':aa['response_from_schedule_ms']['p95']/bb['response_from_schedule_ms']['p95'],
                'elapsed_ratio':aa['completed_elapsed_s']/bb['completed_elapsed_s'],
                'equal_output_counts':sum(a['usage']['completion_tokens']==b['usage']['completion_tokens'] for a,b in zip(off,rr)),
                'equal_output_hashes':sum(a['output_sha256']==b['output_sha256'] for a,b in zip(off,rr)),
                'policy_cumulative_savings':cumulative}
        result['datasets'][dataset]={'arms':arms,'contrasts':contrasts}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('pilot','final'));p.add_argument('root',type=Path);p.add_argument('--arm');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    value=pilot(a.root,a.arm) if a.mode=='pilot' else report(a.root)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps({'status':value['status'],'output':str(a.output)}))

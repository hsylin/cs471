"""Audit proposal decisions and equal-length results without private payloads."""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics


def gate_audit(directory,mode):
    import numpy as np
    from full_serving.proposal_gate import decide
    root=Path(directory);peers=[];counts=Counter();completed=0;enqueued=0;bytes_received=0
    for rank in range(8):
        path=root/f'gate-events-rank{rank}.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()]
        assert rows and all(r['rank']==rank and r['mode']==mode for r in rows)
        assert len([r for r in rows if r['kind']=='bound'])==1
        decisions=[r for r in rows if r['kind']=='decision'];by_step={r['step']:r for r in decisions};assert len(by_step)==len(decisions)
        for record in decisions:
            assert record['rank_agreement'] and record['window']==1000
            seen=record['seen_steps'];assert record['window_bounds']=={'previous':[seen-2000,seen-1000],'recent':[seen-1000,seen]}
            for layer in range(48):
                a=record['previous_counts'][layer];b=record['recent_counts'][layer];old=record['old_map'][layer];candidate=record['native_candidate'][layer]
                d=decide(a,b,old,candidate,require_stability=mode=='b4',recent_complete=seen>=1000,previous_complete=seen>=2000)
                assert d==record['layers'][layer],(rank,record['step'],layer,'decision reconstruction')
                assert record['admitted_map'][layer]==(candidate if d['allow'] else old)
                if rank==0:counts[d['reason']]+=1
        eq={};done={}
        for r in rows:
            if r['kind'] not in ('commit_enqueued','commit_completed'):continue
            key=(r['cycle_step'],r['layer']);target=eq if r['kind']=='commit_enqueued' else done
            assert key not in target;target[key]=r
            d=by_step[key[0]];expected=d['admitted_map'][key[1]]
            assert r['map_sha256']==hashlib.sha256(np.asarray(expected,dtype=np.int64).tobytes()).hexdigest()
            if not d['layers'][key[1]]['allow']:
                assert r['remote_received_experts']==0 and r['remote_received_tensor_bytes']==0 and r['unchanged_local_slots']==20
        assert done.keys()<=eq.keys()
        for k,r in done.items():
            assert r['wall_ns']>=eq[k]['wall_ns']
            assert all(r[x]==eq[k][x] for x in ('map_sha256','remote_received_experts','remote_received_tensor_bytes'))
        enqueued+=len(eq);completed+=len(done);bytes_received+=sum(r['remote_received_tensor_bytes'] for r in eq.values())
        peers.append({'rank':rank,'decisions':[(r['step'],r['seen_steps'],r['digest']) for r in decisions],'enqueued':sorted(eq),'completed':sorted(done)})
    assert all(p['decisions']==peers[0]['decisions'] for p in peers)
    shared_commits=set(map(tuple,peers[0]['completed']))
    for p in peers[1:]:shared_commits&=set(map(tuple,p['completed']))
    return {'status':'passed','proposal_cycles':len(peers[0]['decisions']),'layer_decisions_rank0':dict(counts),'rank_layer_commits_enqueued':enqueued,'rank_layer_commits_gpu_completed':completed,'all_rank_layer_commits_completed':len(shared_commits),'completion_not_observed_count':enqueued-completed,'remote_expert_tensor_bytes_received_all_ranks':bytes_received,'rank_agreement':True,'decisions_independently_reconstructed':True,'scope':'Commit events follow native move_from_buffer and map-copy enqueue; completion observed by CUDA event query. Does not hash final full-model weights. Unobserved completion remains censored; drain is not a commit. Rejected proposals retain native no-op bookkeeping.'}


def profile_summary(directory):
    results=[]
    for path in sorted(Path(directory).rglob('*.pt.trace.json.gz')):
        with gzip.open(path,'rt') as f:trace=json.load(f)
        kernels=[e for e in trace.get('traceEvents',[]) if e.get('cat')=='kernel' and isinstance(e.get('dur'),(float,int))]
        by_name=Counter();counts=Counter()
        for e in kernels:by_name[e['name']]+=e['dur'];counts[e['name']]+=1
        mapper_names=[n for n in by_name if any(k in n for k in ('fused_resident_map','plan_from_partial','partial_histogram','apply_homes'))]
        total=sum(by_name.values());mapper=sum(by_name[n] for n in mapper_names)
        results.append({'file':str(path),'kernel_events':len(kernels),'summed_kernel_us':total,'mapper_kernel_us':mapper,'mapper_fraction_of_summed_kernel_time':mapper/total if total else None,'collective_kernel_us':sum(v for n,v in by_name.items() if 'nccl' in n.lower()),'collective_fraction_of_summed_kernel_time':sum(v for n,v in by_name.items() if 'nccl' in n.lower())/total if total else None,'moe_kernel_us':sum(v for n,v in by_name.items() if 'fused_moe_kernel' in n),'mapper_kernels':[{'name':n,'calls':counts[n],'sum_us':by_name[n]} for n in mapper_names],'largest_kernels':[{'name':n,'calls':counts[n],'sum_us':v} for n,v in by_name.most_common(12)]})
    return {'files':results,'scope':'Profiler perturbs execution. Per-trace summed CUDA kernel times may overlap; not a critical-path or unprofiled latency decomposition. Pilot inputs only.'}


def gate_phase_evidence(directory,summary):
    """Distinguish measured decisions from warmup using same-host rank0 clock."""
    begin=summary['epoch_wall_ns_approx'];end=begin+summary['completed_elapsed_ns']
    rows=[json.loads(x) for x in (Path(directory)/'gate-events-rank0.jsonl').read_text().splitlines()]
    partitions={name:[] for name in ('before_measurement','during_measurement','after_measurement')}
    for row in rows:
        key='before_measurement' if row['wall_ns']<begin else 'during_measurement' if row['wall_ns']<=end else 'after_measurement'
        partitions[key].append(row)
    result={}
    for name,part in partitions.items():
        decisions=[r for r in part if r['kind']=='decision']
        result[name]={'proposal_cycles':len(decisions),'proposal_steps':[r['step'] for r in decisions],
            'layer_decisions':dict(Counter(d['reason'] for r in decisions for d in r['layers'])),
            'rank0_commit_enqueued':sum(r['kind']=='commit_enqueued' for r in part),
            'rank0_commit_gpu_completed':sum(r['kind']=='commit_completed' for r in part)}
    return {'approximate_measured_wall_bounds_ns':[begin,end],'partitions':result,
        'scope':'Rank0 and client run on kh074; approximate wall-clock alignment, not a cross-node clock synchronization proof. Counts classify event observation time: a completion during measurement can belong to an earlier proposal. Full-session all-rank commit audit is reported separately. Warmup decisions may affect measured placement.'}


def logged_config(text):
    entries=[line.split("non-default args: ",1)[1] for line in text.splitlines() if "non-default args: " in line]
    assert len(entries)==1, "Expected one primary API configuration record"
    class Config(ast.NodeTransformer):
        def visit_Call(self,node):
            assert isinstance(node.func,ast.Name) and node.func.id in ('EPLBConfig','ProfilerConfig')
            assert not node.args and all(k.arg is not None for k in node.keywords)
            return ast.Dict(keys=[ast.Constant(k.arg) for k in node.keywords],values=[self.visit(k.value) for k in node.keywords])
    return ast.literal_eval(Config().visit(ast.parse(entries[0],mode='eval')))


def runtime(directory,arm):
    from full_serving.custom_runtime import runtime_identity,logged_arguments,engine_configuration,errors_before_shutdown
    root=Path(directory);texts={f'server-{n}.{suffix}':(root/f'server-{n}.{suffix}').read_text() for n in range(2) for suffix in ('log','err')}
    assert errors_before_shutdown(texts)==0,'Runtime error before shutdown'
    identities=[runtime_identity(json.loads((root/f'preflight-{role}.json').read_bytes()),role) for role in ('primary','worker')]
    # Headless worker does not emit API non-default args; its actual engine
    # initialization and exact issued CLI are checked separately below.
    configs=[logged_config(texts['server-0.log'])]
    engines=[engine_configuration(texts[f'server-{n}.log']) for n in range(2)];assert engines[0]==engines[1]
    for cfg in configs:
        if arm=='off':assert cfg.get('enable_eplb',False) is False
        else:
            assert cfg['enable_eplb'] is True
            assert cfg['eplb_config']['num_redundant_experts']==(0 if arm=='b1' else 32)
    joined='\n'.join(texts.values())
    for marker in ('Using AgRsAll2AllManager','Using V2 Model Runner','Using FlashAttention version 3','Using TRITON Unquantized MoE'):assert marker in joined,marker
    if arm in ('b1','b2','b3','b4'):assert 'TorchDistGlooStagedEplbCommunicator' in joined
    else:assert 'Initialized EPLB communicator:' not in joined
    for role in ('primary','worker'):
        launch=json.loads((root/f'effective-launch-{role}.json').read_bytes())
        assert launch['arm']==arm
        from full_serving.proposal_experiment import server_arguments
        command=launch['command'];assert command[0]=='vllm'
        ip=command[command.index('--data-parallel-address')+1]
        expected=['vllm',*server_arguments(arm,command[2],ip,role)]
        assert command[:len(expected)]==expected
        if launch['pilot'] and arm.startswith('resident'):
            assert command[len(expected)]=='--profiler-config'
            profiler=json.loads(command[len(expected)+1]);assert profiler['profiler']=='torch' and profiler['max_iterations']==16
        else:assert command==expected
        if arm in ('off','b1','b2'):assert launch['plugin'] is None
        else:assert launch['plugin']=='cs471_proposal'
    if arm.startswith('resident'):
        for rank in range(8):
            b=json.loads((root/f'resident-bindings-rank{rank}.json').read_bytes());assert b['ep_rank']==rank and b['layer_count']==48 and b['native_control_step_active'] is False
    return {'status':'passed','identities':identities,'native_source_configuration_checked':True}


def final_report(root):
    from full_serving.proposal_experiment import ARMS
    from full_serving.report_pilots import read_phase,metric_deltas,eplb_log_evidence
    from full_serving.custom_report import METRICS,at,phase_balance,cache_counter_deltas,expected_generation,output_length_issues,IDENTITY_KEYS
    from full_serving.audit_final_statistics import reconstruct
    from full_serving.audit_pilots import audit_counters
    root=Path(root);spec=json.loads((root/'protocol.json').read_bytes());data=json.loads((root/'data-protocol.json').read_bytes());status=json.loads((root/'status.json').read_bytes())
    expected=[f'{d}/{a}' for d in spec['datasets'] for a in spec['orders'][d]]
    assert status['state']=='comparison_completed' and status['completed_sessions']==expected
    controller=json.loads((root/'controller-source.json').read_bytes());client_sha=controller['files']['full_serving/client.py']
    request_admission=json.loads((root/'request-admission.json').read_bytes())['requests'];datasets=[]
    for dataset in data['datasets']:
        name=dataset['dataset'];block=dataset['blocks'][0];calendar=json.loads((root/'calendars'/name/(block['block_id']+'.json')).read_bytes());arms={};records={}
        targets=json.loads((root/block['output_lengths_file']).read_bytes())
        for arm in ARMS:
            p=root/name/arm;identity=runtime(p,arm);item,rows=read_phase(p/'measured.jsonl');assert item['valid'],item['issues']
            assert item['header']['client_source_sha256']==client_sha
            assert expected_generation(item['header'],data,False,name)
            assert not output_length_issues(item['header'],rows,block,data,root)
            independent,signatures=reconstruct(p/'measured.jsonl',arm,block,calendar,data['generation_by_dataset'][name]['max_tokens'],targets,block['output_lengths_sha256'])
            summary=item['summary'];primary={k:at(summary,v) for k,v in METRICS.items()}
            for k in ('mean_response_ms','p95_response_ms','completed_elapsed_s','drain_s'):assert math.isclose(primary[k],independent[k],rel_tol=1e-10,abs_tol=1e-8)
            for row in rows:assert row['input_sha256']==request_admission[row['request_id']]['client_input_sha256']
            for phase in ('warmup-00','warmup-01'):
                warm,wrows=read_phase(p/f'{phase}.jsonl');assert warm['valid'] and len(wrows)==8
                assert warm['header']['arm']==arm and warm['header']['client_source_sha256']==client_sha
                assert warm['header']['requests_sha256']==dataset['warmup']['requests_sha256']
                assert expected_generation(warm['header'],data,phase=='warmup-00',name)
                assert not output_length_issues(warm['header'],wrows,dataset['warmup'],data,root)
                for row in wrows:assert row['input_sha256']==request_admission[row['request_id']]['client_input_sha256']
            counters=audit_counters(p/'measured-metrics-before.txt',p/'measured-metrics-after.txt',16,summary['finish_reason_counts']);assert counters['status']=='matched'
            metrics=metric_deltas(p/'measured-metrics-before.txt',p/'measured-metrics-after.txt');balance=phase_balance(metrics)
            assert balance['available']
            eplb=eplb_log_evidence(p);assert not eplb['async_error_log_entries'] and not eplb['coordinated_stop_log_entries']
            arms[arm]={'metrics':primary,'summary':summary,'runtime':identity,'phase_balance':balance,'native_metrics':metrics,'cache':cache_counter_deltas(p/'measured-metrics-before.txt',p/'measured-metrics-after.txt'),'eplb':eplb,'independent_arithmetic':'passed','header':item['header'],'signatures':signatures}
            if arm in ('b3','b4'):
                arms[arm]['gate']=gate_audit(p,arm)
                arms[arm]['gate']['phase_evidence']=gate_phase_evidence(p,summary)
            records[arm]=rows
        reference=arms['off']
        for arm in ARMS:
            assert arms[arm]['signatures']==reference['signatures']
            assert all(arms[arm]['header'].get(k)==reference['header'].get(k) for k in IDENTITY_KEYS)
        contrasts={}
        for contrast in spec['primary_comparisons']+spec['secondary_comparisons']:
            a,b=contrast.split('/');contrasts[contrast]={'ratios':{k:arms[a]['metrics'][k]/arms[b]['metrics'][k] for k in METRICS},'equal_output_counts':all(x['usage']['completion_tokens']==y['usage']['completion_tokens'] for x,y in zip(records[a],records[b])),'equal_output_hashes':sum(x['output_sha256']==y['output_sha256'] for x,y in zip(records[a],records[b]))}
        datasets.append({'dataset':name,'arms':arms,'contrasts':contrasts})
    return {'status':'completed_and_audited','schema':'cs471_proposal_followup_results_v1','sessions':14,'measured_requests':224,'datasets':datasets,'uncertainty':'One group, descriptive only; no inferential intervals or significance. Equal token counts do not fix token identity or routing.','timing':'Scheduled arrival to final response; startup and warmup separate. Completed elapsed includes arrival horizon and drain.','automatic_speedup_claim':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('profiles','gate','final'));p.add_argument('root',type=Path);p.add_argument('--gate-mode',choices=('b3','b4'));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=profile_summary(a.root) if a.mode=='profiles' else gate_audit(a.root,a.gate_mode) if a.mode=='gate' else final_report(a.root)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result.get('status'),'output':str(a.output)}))
if __name__=='__main__':main()

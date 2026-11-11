"""Audit completed attribution pilots; keep instrumentation and causal limits explicit."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path


def runtime(directory,arm):
    from full_serving.custom_runtime import runtime_identity,engine_configuration,errors_before_shutdown
    from full_serving.monitoring_analysis import logged_config,logged_replica_count
    from full_serving.cost_experiment import server_arguments
    p=Path(directory)
    texts={f'server-{n}.{s}':(p/f'server-{n}.{s}').read_text() for n in range(2) for s in ('log','err')}
    assert errors_before_shutdown(texts)==0
    identities=[runtime_identity(json.loads((p/f'preflight-{role}.json').read_text()),role) for role in ('primary','worker')]
    assert engine_configuration(texts['server-0.log'])==engine_configuration(texts['server-1.log'])
    config=logged_config(texts['server-0.log'])
    if arm=='off':assert not config.get('enable_eplb',False)
    else:
        assert config['enable_eplb']
        assert logged_replica_count(config)==(0 if arm.startswith('fixed0') else 32)
    for role in ('primary','worker'):
        launch=json.loads((p/f'effective-launch-{role}.json').read_text());cmd=launch['command']
        assert launch['arm']==arm and launch['pilot'] and not launch['diagnostic']
        assert launch['plugin']=='cs471_cost'
        ip=cmd[cmd.index('--data-parallel-address')+1]
        base=['vllm',*server_arguments(arm,cmd[2],ip,role)]
        assert cmd[:len(base)]==base and cmd[len(base)]=='--profiler-config' and len(cmd)==len(base)+2
        assert json.loads(cmd[-1])['max_iterations']==16
    for marker in ('Using AgRsAll2AllManager','Using V2 Model Runner','Using FlashAttention version 3','Using TRITON Unquantized MoE'):
        assert marker in '\n'.join(texts.values()),marker
    for rank in range(8):
        events=[json.loads(l) for l in (p/f'cost-events-rank{rank}.jsonl').read_text().splitlines()]
        assert any(r['kind']=='activation_collective_shape' for r in events)
        assert all(r['rank']==rank and r['arm']==arm for r in events)
        if arm.startswith(('fixed','balanced')):
            bound=[r for r in events if r['kind']=='fixed_bound'];assert len(bound)==1
            assert bound[0]['recording']==arm.endswith('_record') and not bound[0]['migration_enabled']
            assert not bound[0]['async_worker_started']
            checks=[r for r in events if r['kind']=='map_record_oracle'];assert len(checks)==1 and checks[0]['status']=='passed' and checks[0]['cases']==12
            maps=bound[0]['physical_to_logical_map'];assert len(maps)==48 and all(m==maps[0] for m in maps)
            if rank==0:reference=maps
            else:assert maps==reference
    return {'status':'passed','identities':identities,'native_configuration_and_all_rank_diagnostics':'passed'}


def pilots(root):
    from full_serving.cost_experiment import ARMS
    from full_serving.report_pilots import read_phase,metric_deltas
    from full_serving.custom_report import phase_balance
    root=Path(root);out={}
    for arm in ARMS:
        run=root/arm;p=run/'pilot'/arm
        if not (run/'status.json').exists():continue
        s=json.loads((run/'status.json').read_text())
        if s['state']!='comparison_completed':continue
        assert (root/f'{arm}-completed.txt').read_text()=='controller_exit=0\n'
        identity=runtime(p,arm);data=json.loads((run/'data-protocol.json').read_text());phases={}
        for dataset in data['datasets']:
            name=dataset['dataset'];warm=dataset['warmup']
            targets=json.loads((run/'output-lengths'/f'{name}-warmup.json').read_text())
            for phase in ('warmup-00','warmup-01','profile'):
                label=name+'-'+phase;item,rows=read_phase(p/(label+'.jsonl'))
                assert item['valid'] and len(rows)==8
                assert item['header']['requests_sha256']==warm['requests_sha256']
                assert [r['request_id'] for r in rows]==warm['request_ids']
                assert all(r['requested_output_tokens']==targets[r['request_id']]==r['usage']['completion_tokens'] and r['finish_reason']=='length' for r in rows)
                phases[label]={'summary':item['summary'],'native_phase':phase_balance(metric_deltas(p/(label+'-metrics-before.txt'),p/(label+'-metrics-after.txt')))}
        out[arm]={'runtime':identity,'phases':phases,'timing_admissible_for_attribution':arm!='b4_light',
                  'scope':'Pilot-only, same eight inputs and exact lengths; two warmups are technical passes, not independent groups. B4-light retains reference history shadow and is a diagnostic coverage control, not a fair cost baseline.'}
    return {'status':'completed_pilots' if len(out)==len(ARMS) else 'partial','arms':out,
            'causal_contrasts':{'recording':'fixed0_record/fixed0_off and fixed32_record/fixed32_off include router atomics and native window copy/reset; map and no-migration policy fixed',
              'mapping_and_bookkeeping':'fixed0_off/OFF includes map kernel, prepare_forward and native step; cannot uniquely attribute to one without profile/replay',
              'replica_capacity':'fixed32_off/fixed0_off changes32extra slots, ownership and expert work, not a pure memory allocation cost',
              'replica_layout':'balanced32_off/fixed32_off holds32extra copies and native hash dispatch, changes placement AND which logical experts are replicated',
              'dispatcher':'resident_v2/balanced32_off uses same resident map but also removes native per-step state; not a pure mapper-only contrast'},
            'limits':'Generated tokens/routes may differ despite exact lengths. Profiling perturbs execution. Final speed claims require separately frozen uninstrumented sessions.'}


def union_us(events):
    spans=sorted((r['ts'],r['ts']+r['dur']) for r in events if isinstance(r.get('ts'),(int,float)) and r.get('dur',0)>0)
    total=0;end=None
    for a,b in spans:
        if end is None or a>=end:total+=b-a
        elif b>end:total+=b-end
        end=b if end is None else max(end,b)
    return total


def profiles(root):
    files=[]
    for p in sorted(Path(root).rglob('*.pt.trace.json.gz')):
        with gzip.open(p,'rt') as f:trace=json.load(f)
        events=trace.get('traceEvents',[])
        kernels=[e for e in events if e.get('cat')=='kernel' and isinstance(e.get('dur'),(int,float))]
        base=trace.get('baseTimeNanoseconds',0)
        bounds=[base+min(e['ts'] for e in kernels)*1000,base+max(e['ts']+e['dur'] for e in kernels)*1000] if kernels else None
        matched=[];phase_intervals={}
        for dataset in ('longbench','numinamath'):
            raw=p.parent.parent/(dataset+'-profile.jsonl')
            if not raw.exists():continue
            summaries=[r for line in raw.read_text().splitlines() if (r:=json.loads(line)).get('type')=='run_summary']
            assert len(summaries)==1
            summary=summaries[0];start=summary['epoch_wall_ns_approx'];end=start+summary['completed_elapsed_ns']
            phase_intervals[dataset]=[start,end]
            if bounds and bounds[0]<end and bounds[1]>start:matched.append(dataset)
        names=Counter();counts=Counter()
        for e in kernels:names[e['name']]+=e['dur'];counts[e['name']]+=1
        cats={
            'collective':[e for e in kernels if 'nccl' in e['name'].lower()],
            'native_map_record':[e for e in kernels if '_eplb_map_and_record_i32_kernel' in e['name']],
            'resident_mapper':[e for e in kernels if any(k in e['name'] for k in ('resident_map','plan_from_partial','partial_histogram','apply_homes','pair_plan'))],
            'logical_history':[e for e in kernels if any(k in e['name'] for k in ('update_history','reduce_history'))],
            'expert_gemm':[e for e in kernels if 'fused_moe_kernel' in e['name']],
        }
        files.append({'file':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'rank':trace.get('distributedInfo',{}).get('rank'),'host':trace.get('host_name'),
          'approximate_kernel_wall_bounds_ns':bounds,'client_profile_intervals_ns':phase_intervals,'dataset':matched[0] if len(matched)==1 else 'unclassified',
          'phase_classification_limit':'Approximate host wall clocks; ambiguous/non-overlapping traces remain unclassified.',
          'kernels':len(kernels),'kernel_union_us':union_us(kernels),'summed_kernel_us':sum(names.values()),
          'categories':{name:{'calls':len(es),'sum_us':sum(e['dur'] for e in es),'union_us':union_us(es)} for name,es in cats.items()},
          'top_kernels':[{'name':n,'calls':counts[n],'sum_us':v} for n,v in names.most_common(20)]})
    return {'files':files,'scope':'Per-trace device timeline union and sums, not synchronized cross-rank critical path. NCCL durations include peer waiting; do not call them wire time or bandwidth saturation. Python hooks catalogue graph construction/eager shapes, not actual graph replay frequency.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('pilots','profiles'));p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=pilots(a.root) if a.mode=='pilots' else profiles(a.root)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result.get('status'),'output':str(a.output)}))
if __name__=='__main__':main()

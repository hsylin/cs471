"""Bounded migration intervention diagnostics, with explicit state/work limits."""
import argparse
import json
from pathlib import Path
from full_serving.sustained_analysis import migrations,raw_phase,runtime


def analyze(root):
    root=Path(root);sides={}
    for side in ('allow','skip'):
        run=root/('payback-'+side)/'results';directory=run/'pilot'/'b2'
        status=json.loads((run/'status.json').read_text());assert status['state']=='comparison_completed'
        manifest=json.loads((run/'data-manifest.json').read_text());phases=[]
        for ds in ('longbench','numinamath'):
            for phase in ('warmup-00','warmup-01','pilot'):
                label=ds+'-'+phase;targets=json.loads((run/(label+'-lengths.json')).read_text())
                value,rows=raw_phase(directory/(label+'.jsonl'),manifest['datasets'][ds]['items']['warmup'],targets=targets)
                phases.append({'label':label,'metrics':value,'records':rows})
        plans={};events={}
        for rank in range(8):
            rr=[json.loads(x) for x in (directory/f'migration-events-rank{rank}.jsonl').read_text().splitlines()]
            plans[rank]={r['step']:r for r in rr if r['kind']=='proposal'};events[rank]=rr
        sides[side]={'runtime':runtime(directory,'b2'),'migrations':migrations(directory),'phases':phases,'plans':plans,'events':events}
    a,b=sides['allow'],sides['skip'];opportunity=all(3750 in s['plans'][r] for s in sides.values() for r in range(8))
    checks={}
    if opportunity:
        for r in range(8):
            left,right=a['plans'][r][3750],b['plans'][r][3750]
            checks[str(r)]={k:left[k]==right[k] for k in ('old_map_sha256','diagnostic_native_demand_sha256','candidate')}
    totals={'allow':0.,'skip':0.};curve=[];prefix=[]
    for ap,bp in zip(a['phases'],b['phases']):
        assert ap['label']==bp['label']
        for ar,br in zip(ap['records'],bp['records']):
            assert ar['request_id']==br['request_id'] and ar['input_sha256']==br['input_sha256']
            assert ar['usage']['completion_tokens']==br['usage']['completion_tokens']
            # Single-inflight diagnostic: compare cumulative actual HTTP execution
            # time, excluding artificial client semaphore wait, including serving
            # stalls encountered by requests. Not GPU-only time.
            for side,row in [('allow',ar),('skip',br)]:totals[side]+=row['response_from_send_ns']/1e9
            entry={'phase':ap['label'],'request_id':ar['request_id'],
                'allow_cumulative_request_execution_s':totals['allow'],
                'skip_cumulative_request_execution_s':totals['skip'],
                'net_saved_s':totals['skip']-totals['allow'],
                'equal_generated_output_hash':ar['output_sha256']==br['output_sha256']}
            curve.append(entry)
            if opportunity:
                before=[]
                for side,row,phase in [('allow',ar,ap),('skip',br,bp)]:
                    timing=phase['metrics']['timing']
                    end_wall=timing['epoch_wall_ns_approx']+row['end_ns']-timing['epoch_monotonic_ns']
                    before.append(end_wall<sides[side]['plans'][0][3750]['wall_ns'])
                if all(before):prefix.append(entry['equal_generated_output_hash'])
    # No automatic causal declaration: active-request KV cache is not cloned or
    # hashed, and natural autoregression can diverge even at equal output counts.
    return {'status':'completed_and_audited','target_engine_step':3750,
        'opportunity_observed_all_ranks':opportunity,'predecision_checks':checks,
        'completed_predecision_request_hash_matches':sum(prefix),'completed_predecision_requests':len(prefix),
        'equal_generated_outputs':sum(x['equal_generated_output_hash'] for x in curve),'requests':len(curve),
        'cumulative_curve':curve,'net_request_execution_saved_s':totals['skip']-totals['allow'],
        'allow_migrations':a['migrations'],'skip_migrations':b['migrations'],
        'causal_payback_established':False,
        'limitations':['Separate restarted full-model sessions, no cloned/full-hashed KV state.',
            'Map, demand and completed-prefix agreement are necessary diagnostics, not proof of identical active-request state.',
            'Different generated outputs/routes can change work despite exact lengths.',
            'If the target is absent, candidate is unchanged or states disagree, migration-specific payback is inconclusive.',
            'Only one ordered allow/skip pair; timing drift unestimated. Net execution contrast is descriptive, not isolated transfer cost.']}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.write_text(json.dumps(analyze(a.root),indent=2)+'\n')

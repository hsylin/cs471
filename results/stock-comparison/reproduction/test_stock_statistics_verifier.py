"""Synthetic validation only; no GPU measurements or experimental evidence."""
import copy,importlib.util,itertools,json,math
from pathlib import Path
import numpy as np
import pytest
spec=importlib.util.spec_from_file_location('independent_stock',Path(__file__).with_name('verify_stock_final_statistics.py'))
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)


def reseal(x):x['content_sha256']=v.digest({k:y for k,y in x.items() if k!='content_sha256'});return x

def rehash(x):x['record_id']=v.digest({k:y for k,y in x.items() if k!='record_id'});return x

def fixture(phase='final',service_fn=None,date_fn=None):
    n=6 if phase=='final' else 2;protocol=b'Synthetic test protocol; not an experiment.'
    banks=[{'block_id':f'{d}-{i:02d}','dataset':d,'split':phase} for d in v.DATASETS for i in range(n)]
    offsets=[i*2000000 for i in range(4096)];horizon=4096*2000000
    plan={'schema':'stock_comparison_plan_v1','phase':phase,'split':phase,'arms':list(v.ARMS),'jobs':4096,'batch_tokens':128,'protocol_sha256':v.digest(protocol),'source_sha256':'1'*64,'hardware':{'allocation_id':'synthetic'},'banks':banks,'cases':[]}
    perms=list(itertools.permutations(v.ARMS))
    for di,d in enumerate(v.DATASETS):
        for rho in v.LOADS:
            for j,(i,t) in enumerate(itertools.product(range(n),v.TYPES)):
                gid=f'{d}-{i:02d}';day=date_fn(d,i,t) if date_fn else f'2024-05-{12+i%5 if t=="code" else 14+(i+2)%5:02d}'
                source={'trace':t,'source_date_utc':day,'jobs':4096,'original_offsets_ns':offsets,'original_horizon_ns':horizon}
                cal=reseal({'arrival_type':t,'arrival_date':day,'rho':rho,'n_jobs':4096,'arrival_offsets_ns':offsets,'horizon_ns':horizon,'stockoff_calibration_elapsed_ns':int(horizon*rho)})
                order=perms[(j if phase=='final' else di*4+j)%6]
                plan['cases'].append({'block_id':gid,'dataset':d,'source_block':source,'calendar':cal,'arm_order':list(order)})
    reseal(plan);rows=[]
    for ci,c in enumerate(plan['cases']):
        gid=c['block_id'];d=c['dataset'];i=int(gid[-2:]);t=c['calendar']['arrival_type'];rho=c['calendar']['rho']
        for pos,arm in enumerate(c['arm_order']):
            duration=service_fn(d,i,t,rho,arm) if service_fn else {'stockoff':100000,'stockon32':110000,'ours':85000}[arm]
            epoch=10**15+ci*10**13+pos*10**12
            starts=[epoch+x for x in offsets];finishes=[s+duration for s in starts];q=finishes[-1]+10;end=max(epoch+horizon,q)
            row={'content_group_id':gid,'block_id':gid,'dataset':d,'arrival_type':t,'rho':rho,'arm':arm,'status':'complete','phase':phase,'evidence_eligible':phase=='final','plan_sha256':plan['content_sha256'],'source_sha256':plan['source_sha256'],'allocation_id':'synthetic','calendar':c['calendar'],'arrival_date':c['calendar']['arrival_date'],'arrival_offsets_ns':offsets,'horizon_ns':horizon,'arm_order':c['arm_order'],'input_meta':{'capture_sha256':v.digest(gid),'indices_sha256':v.digest([gid,'indices']),'source_prompt_ids':[gid],'gate_weight_sha256':'2'*64},'preflight':{'logical_routing_sha256':v.digest([gid,'route']),'float32_combining_weights_sha256':v.digest([gid,'weights']),'correctness_pass':True},'postflight':{'correctness_pass':True},'n_jobs':4096,'per_job_completion_scope':'current_inference_stream_after_native_step','epoch_monotonic_ns':epoch,'observation_end_ns':end,'coordinator_admit_ns':starts,'coordinator_start_ns':starts,'coordinator_finish_ns':finishes,'per_rank_start_ns':[starts,starts],'per_rank_gpu_complete_ns':[finishes,finishes],'controller_quiescence_start_ns':finishes[-1]+5,'controller_quiescent_ns':q,'completed_elapsed_ns':end-epoch,'job_completed_elapsed_ns':finishes[-1]-epoch,'controller_tail_elapsed_ns':10,'quiescence_call_elapsed_ns':5,'drain_including_controller_ns':0,'queue_metrics':{'complete':True,'completed_jobs':4096,'unfinished_jobs':0,'vectors_ns':{'response':[duration]*4096},'summary_ns':{'response':{'mean':float(duration),'p95':float(duration)}}}}
            rows.append(rehash(row))
    return plan,rows,protocol


def test_exact_constants_complete_matrix_and_default_draw_digest():
    p,rows,proto=fixture();out=v.reconstruct(p,rows,proto)
    x=out['confirmatory'];assert out['trial_count']==216 and out['matched_triplets']==72 and out['paired_comparisons']==216
    assert x['primary']['ratio']==pytest.approx(.85)
    assert x['primary']['ci95']==pytest.approx([.85,.85])
    assert x['secondary_stockon32']['ratio']==pytest.approx(85000/110000)
    assert x['acceptance']=='pass' and len(x['guards'])==14
    assert all(g['one_sided_confidence_level']==1-.05/14 for g in x['guards'])
    rng=np.random.Generator(np.random.PCG64(471612))
    tables=[rng.integers(0,6,size=(100000,6),dtype=np.int64) for _ in range(3)]
    expected=v.digest(b''.join(t.astype('<i8').tobytes() for t in tables))
    assert x['bootstrap']['draw_indices_sha256']==expected
    assert x['calendar_date_robustness']['date_count']==7


def test_pilot_has_no_confirmatory_inference():
    p,rows,proto=fixture('pilot');x=v.reconstruct(p,rows,proto)
    assert x['trial_count']==72 and x['matched_triplets']==24 and x['confirmatory'] is None


def test_secondary_cannot_rescue_primary_or_failed_guard():
    p,rows,proto=fixture(service_fn=lambda d,i,t,r,a:{'stockoff':100000,'stockon32':120000,'ours':96000}[a])
    x=v.reconstruct(p,rows,proto,draw_count=500)['confirmatory']
    assert not x['primary']['passes'] and x['secondary_stockon32']['passes'] is None
    assert x['secondary_stockon32']['status']=='not_tested_gate_closed'
    p,rows,proto=fixture(service_fn=lambda d,i,t,r,a:{'stockoff':100000,'stockon32':110000,'ours':105000 if d=='mbpp' and r==.5 else 85000}[a])
    x=v.reconstruct(p,rows,proto,draw_count=500)['confirmatory']
    assert x['primary']['passes'] and not next(g for g in x['guards'] if g['name']=='stockoff_mbpp_rho0.5_mean_response')['passes']
    assert x['secondary_stockon32']['ci95'] is None


def test_secondary_itself_must_clear_target():
    p,rows,proto=fixture(service_fn=lambda d,i,t,r,a:{'stockoff':100000,'stockon32':88000,'ours':85000}[a])
    x=v.reconstruct(p,rows,proto,draw_count=300)['confirmatory']
    assert x['primary_acceptance']=='pass' and x['secondary_stockon32']['confirmatory']
    assert not x['secondary_stockon32']['passes'] and x['acceptance']=='not_established'


def test_date_deletion_missing_dataset_fails_without_reweighting_survivors():
    p,rows,proto=fixture(date_fn=lambda d,i,t:'2024-05-12' if d=='gsm8k' else f'2024-05-{14+i%5:02d}')
    x=v.reconstruct(p,rows,proto,draw_count=300)['confirmatory']
    deleted=[r for r in x['calendar_date_robustness']['results'] if r['excluded_date']=='2024-05-12']
    assert all(r['missing_datasets']==['gsm8k'] and r['ratio'] is None and not r['passes'] for r in deleted)
    assert x['secondary_stockon32']['ci95'] is None


@pytest.mark.parametrize('mutation',['missing','duplicate','early_finish','rank_outside','changed_response','wrong_protocol','changed_calendar','input_identity','arm_epoch_order'])
def test_raw_mutations_fail_closed(mutation):
    p,rows,proto=fixture('pilot')
    if mutation=='missing':rows.pop()
    elif mutation=='duplicate':rows.append(rows[0])
    elif mutation=='wrong_protocol':proto=b'wrong'
    else:
        row=copy.deepcopy(rows[0]);rows[0]=row
        if mutation=='early_finish':row['coordinator_finish_ns'][5]=row['coordinator_start_ns'][5]-1
        elif mutation=='rank_outside':row['per_rank_gpu_complete_ns'][1][7]+=10000000
        elif mutation=='changed_response':row['queue_metrics']['summary_ns']['response']['p95']+=10
        elif mutation=='changed_calendar':row['arrival_offsets_ns'][3]+=1
        elif mutation=='input_identity':row['input_meta']['indices_sha256']='9'*64
        elif mutation=='arm_epoch_order':row['epoch_monotonic_ns']+=10**12
        rehash(row)
    with pytest.raises((ValueError,KeyError)):v.reconstruct(p,rows,proto,draw_count=10)


def summary_from_fixture(out):
    return {'valid':True,'phase':out['phase'],'observed_records':out['trial_count'],'expected_records':out['trial_count'],'valid_records':out['trial_count'],'matched_triplets':out['matched_triplets'],'paired_comparisons':out['paired_comparisons'],'pairs_per_contrast':out['pairs_per_contrast'],'missing_keys':[],'errors':[],'trial_summaries':[dict(zip(('content_group_id','arrival_type','rho','arm'),k),metrics=value) for k,value in out['trial_metrics'].items()],'paired_cases':copy.deepcopy(out['paired_cases']),'confirmatory':copy.deepcopy(out['confirmatory'])}


@pytest.mark.parametrize('mutation',['trial_stat','one_ns_elapsed','pair_stat','draw_digest','guard','verdict'])
def test_report_mutations_do_not_drive_recomputed_decision(mutation):
    p,rows,proto=fixture();out=v.reconstruct(p,rows,proto,draw_count=100);summary=summary_from_fixture(out)
    assert v.compare_summary(out,summary)['passed']
    if mutation=='trial_stat':summary['trial_summaries'][0]['metrics']=dict(summary['trial_summaries'][0]['metrics'],response_mean_ns=1.)
    elif mutation=='one_ns_elapsed':summary['trial_summaries'][0]['metrics']=dict(summary['trial_summaries'][0]['metrics'],completed_elapsed_ns=summary['trial_summaries'][0]['metrics']['completed_elapsed_ns']+1)
    elif mutation=='pair_stat':summary['paired_cases'][0]['ratios']['response_p95_ns']+=.1
    elif mutation=='draw_digest':summary['confirmatory']['bootstrap']['draw_indices_sha256']='f'*64
    elif mutation=='guard':summary['confirmatory']['guards'][0]['passes']=False
    elif mutation=='verdict':summary['confirmatory']['acceptance']='not_established'
    with pytest.raises(ValueError):v.compare_summary(out,summary)

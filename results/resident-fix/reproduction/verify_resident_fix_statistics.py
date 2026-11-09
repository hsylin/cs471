#!/usr/bin/env python3
"""Independent CPU reconstruction of the frozen resident-fix statistics.

Only stdlib/NumPy are imported. Production summaries are comparison targets,
never inputs to clock metrics, paired effects, bootstrap draws or decisions.
This is independent implementation within the same team, not external replication.
"""
from collections import Counter,defaultdict
from fractions import Fraction
import argparse,gzip,hashlib,itertools,json,math
from pathlib import Path
import numpy as np

DATASETS=('gsm8k','wildchat')
SENTINEL='mbpp'
TYPES=('code','conversation')
LOADS=(.5,.85)
ARMS=('stockoff','current','fix')
CONTRASTS={'fix_over_stockoff':('fix','stockoff'),'fix_over_current':('fix','current'),'current_over_stockoff':('current','stockoff')}
METRICS=('response_mean_ns','response_p95_ns')
DRAWS=100000
SEED=471712


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(value):return hashlib.sha256(value if isinstance(value,bytes) else canonical(value)).hexdigest()
def check(condition,message):
    if not condition:raise ValueError(message)
def sealed(value):return value.get('content_sha256')==digest({k:v for k,v in value.items() if k!='content_sha256'})
def finite_positive(x):return not isinstance(x,bool) and isinstance(x,(int,float)) and math.isfinite(x) and x>0

def linear_quantile(values,q):
    a=sorted(values);index=(len(a)-1)*q;lo=math.floor(index);fraction=index-lo
    if not fraction:return float(a[lo])
    # Same mathematical linear convention, implemented without np.quantile.
    return float(a[lo]+(a[lo+1]-a[lo])*fraction)


def read_rows(path):
    path=Path(path)
    files=sorted((path/'trials').glob('*.jsonl*')) if path.is_dir() else [path]
    check(bool(files),'No raw trial files')
    rows=[];pins=[]
    for file in files:
        stored=file.read_bytes();raw=gzip.decompress(stored) if file.suffix=='.gz' else stored
        rows.extend(json.loads(line) for line in raw.splitlines() if line.strip())
        pins.append({'file':file.name,'stored_sha256':digest(stored),'uncompressed_sha256':digest(raw)})
    return rows,pins


def clock_endpoints(row):
    n=4096
    check(row.get('n_jobs')==n,'Job count differs')
    check(row.get('per_job_completion_scope')=='current_inference_stream_after_native_step','Completion scope differs')
    fields=('arrival_offsets_ns','coordinator_admit_ns','coordinator_start_ns','coordinator_finish_ns')
    offsets,admits,starts,finishes=(row[k] for k in fields)
    for field,arr in zip(fields,(offsets,admits,starts,finishes)):
        check(len(arr)==n and all(type(v) is int for v in arr),'Incomplete or noninteger clocks: '+field)
    epoch=row['epoch_monotonic_ns'];horizon=row['horizon_ns'];end=row['observation_end_ns']
    check(all(type(v) is int for v in (epoch,horizon,end)) and horizon>0,'Invalid epoch/horizon')
    check(offsets[0]==0 and offsets[-1]<horizon and all(0<=a<=b for a,b in zip(offsets,offsets[1:])),'Changed/invalid arrival order')
    responses=[];service=[];queue=[]
    for i,(offset,admit,start,finish) in enumerate(zip(offsets,admits,starts,finishes)):
        arrival=epoch+offset
        check(arrival<=admit<=start<finish,'Invalid job timing order')
        if i:check(finishes[i-1]<=admit,'FCFS overlap or dropped ordering')
        responses.append(finish-arrival);service.append(finish-start);queue.append(start-arrival)
    check(end>=max(epoch+horizon,finishes[-1]),'Censored offered horizon or output')
    for field in ('per_rank_start_ns','per_rank_gpu_complete_ns'):
        check(len(row[field])==2 and all(len(a)==n and all(type(v) is int for v in a) for a in row[field]),'Missing rank clocks')
    for rank in range(2):
        check(all(s<=rs<=rf<=f for s,rs,rf,f in zip(starts,row['per_rank_start_ns'][rank],row['per_rank_gpu_complete_ns'][rank],finishes)),'Rank completion not ACK-enclosed')
    tail_start=row['controller_quiescence_start_ns'];q=row['controller_quiescent_ns']
    check(type(tail_start) is int and type(q) is int and finishes[-1]<=tail_start<=q<=end,'Invalid controller tail')
    elapsed={'completed_elapsed_ns':end-epoch,'job_completed_elapsed_ns':finishes[-1]-epoch,'controller_tail_elapsed_ns':q-finishes[-1],'quiescence_call_elapsed_ns':q-tail_start,'drain_including_controller_ns':max(0,q-epoch-horizon)}
    check(all(row.get(k)==v for k,v in elapsed.items()),'Changed elapsed/tail accounting')
    vectors={'response':responses,'service':service,'queue_wait':queue,
        'admission_delay':[a-epoch-o for a,o in zip(admits,offsets)],'dispatch_delay':[s-a for s,a in zip(starts,admits)]}
    values=dict(elapsed)
    for label,vector in vectors.items():
        name='queue' if label=='queue_wait' else label
        summaries=dict(mean=sum(vector)/n,p50=linear_quantile(vector,.5),p95=linear_quantile(vector,.95),p99=linear_quantile(vector,.99),max=max(vector))
        values.update({name+'_'+stat+'_ns':v for stat,v in summaries.items()})
    cutoff=epoch+horizon;completed_by=sum(f<=cutoff for f in finishes);started_by=sum(t<=cutoff for t in starts)
    values.update(drain_ns=max(0,finishes[-1]-cutoff),backlog_at_horizon=n-completed_by,waiting_jobs_at_horizon=n-started_by,
        throughput_jobs_per_second=n*1e9/(end-epoch),by_horizon_throughput_jobs_per_second=completed_by*1e9/horizon,
        completed_by_horizon=completed_by,horizon_ns=horizon,coordinator_busy_fraction=sum(service)/(end-epoch))
    stored=row['queue_metrics']
    check(stored['complete'] is True and stored['completed_jobs']==n and stored['unfinished_jobs']==0,'Incomplete stored trial')
    check(all(stored['vectors_ns'][k]==v for k,v in vectors.items()),'Stored response/service/queue vectors differ from immutable arrivals')
    check(stored['queue_occupancy']['integral_job_ns']==sum(queue) and stored['system_occupancy']['integral_job_ns']==sum(responses),'Occupancy integral accounting')
    for metric,stat in (('response_mean_ns','mean'),('response_p95_ns','p95')):
        check(math.isclose(values[metric],stored['summary_ns']['response'][stat],rel_tol=2e-14,abs_tol=1e-6),'Stored response statistic differs')
    return values


def validate_inputs(plan,rows,protocol_bytes):
    check(sealed(plan) and plan['schema']=='resident_fix_plan_v1','Changed/wrong frozen plan')
    phase=plan['phase'];split=plan['split'];n=8 if split=='final' else 2
    check(phase in ('calibration','pilot','final') and split in ('pilot','final') and (phase=='calibration' or phase==split),'Phase/split')
    check(plan['arms']==list(ARMS) and plan['jobs']==4096 and plan['batch_tokens']==128,'Scope')
    check(digest(protocol_bytes)==plan['protocol_sha256'],'Frozen protocol')
    check(plan['source_sha256']==digest(plan['source_files']),'Source inventory digest')
    check(plan['old_closure_sha256']=='9c6d6a7e1e94f4be769d435fbf7652e0cbb86501637c046ae078a0ccef1015d4','Old source closure')
    check(plan['selected_fix_content_sha256']==digest(plan['selected_fix']),'Selected implementation content')
    groups={b['block_id']:b['dataset'] for b in plan['banks']}
    check(len(groups)==len(plan['banks'])==2*n+1 and Counter(groups.values())==Counter({'gsm8k':n,'wildchat':n,'mbpp':1}),'Main/sentinel allocation')
    cases={};orders={};source_ids=defaultdict(set)
    for c in plan['cases']:
        cal=c['calendar'];key=(c['content_group_id'],c['arrival_type'],c['rho']);source=c['source_block']
        check(key not in cases and key[0] in groups and c['block_id']==key[0] and c['dataset']==groups[key[0]],'Duplicate/mislabeled case')
        check(sealed(cal) and cal['arrival_type']==key[1] and cal['rho']==key[2],'Calendar identity')
        factor=Fraction(cal['stockoff_calibration_elapsed_ns'],1)/Fraction(str(cal['rho']))/source['original_horizon_ns']
        check(cal['arrival_offsets_ns']==[round(x*factor) for x in source['original_offsets_ns']] and cal['horizon_ns']==round(source['original_horizon_ns']*factor),'Calendar rational scaling')
        check(cal['arrival_type']==source['trace'] and cal['arrival_date']==source['source_date_utc'] and cal['n_jobs']==source['jobs']==4096,'Calendar source/geometry')
        check(sorted(c['arm_order'])==sorted(ARMS),'Arm permutation')
        p=plan['calibration_provenance'][key[0]]
        check(p['calibration_elapsed_ns']==cal['stockoff_calibration_elapsed_ns'] and p['source_sha256']==plan['source_sha256'],'Calibration provenance')
        check(str(p['allocation_id'])==str(plan['hardware']['allocation_id']),'Calibration allocation')
        cases[key]=c;orders[key]=tuple(c['arm_order']);source_ids[key[:2]].add(digest(source))
    if phase=='calibration':check(not cases,'Calibration treatment cases')
    else:
        check(set(cases)==set(itertools.product(groups,TYPES,LOADS)) and all(len(s)==1 for s in source_ids.values()),'Complete paired calendar matrix')
        for g,rho in itertools.product(groups,LOADS):check(orders[g,'code',rho]==orders[g,'conversation',rho][::-1],'Within-group reverse order')
        perms=list(itertools.permutations(ARMS))
        for ds,rho in itertools.product((DATASETS,('gsm8k',),('wildchat',)),LOADS):
            selected=[o for (g,t,r),o in orders.items() if groups[g] in ds and r==rho];counts=Counter(selected)
            check(max(counts[x] for x in perms)-min(counts[x] for x in perms)<=1,'Near-balanced order counts')
            for a,b in itertools.combinations(ARMS,2):check(sum(o.index(a)<o.index(b) for o in selected)*2==len(selected),'Pairwise precedence')
            if phase=='final':
                for typ in TYPES:
                    counts=Counter(o for (g,t,r),o in orders.items() if groups[g] in ds and r==rho and t==typ)
                    check(max(counts[x] for x in perms)-min(counts[x] for x in perms)<=1,'Per-arrival-type balance')
    expected={(g,None,None,'stockoff') for g in groups} if phase=='calibration' else {(*k,arm) for k in cases for arm in ARMS}
    metrics={};raw={};identities=defaultdict(set)
    for row in rows:
        key=tuple(row[k] for k in ('content_group_id','arrival_type','rho','arm'))
        check(key in expected and key not in raw,'Unexpected/duplicate row')
        check(row['record_id']==digest({k:v for k,v in row.items() if k!='record_id'}),'Raw content hash')
        check(row['schema']=='resident_fix_trial_v1' and row['phase']==phase and row['status']=='complete' and row['evidence_eligible']==(phase=='final'),'Trial disposition')
        check(row['dataset']==groups[key[0]] and row['block_id']==key[0],'Dataset/content')
        check(row['inference_role']==('sentinel_descriptive' if row['dataset']=='mbpp' else 'main'),'Sentinel classification')
        check(row['plan_sha256']==plan['content_sha256'] and row['source_sha256']==plan['source_sha256'] and row['old_closure_sha256']==plan['old_closure_sha256'],'Source/plan pins')
        hw=plan['hardware'];check(str(row['allocation_id'])==str(hw['allocation_id']) and row['hostname'].split('.')[0]==hw['hostname'] and sorted(row['gpu_uuids'])==sorted(hw['gpu_uuids']),'Hardware identity')
        if phase=='calibration':
            check(row['calendar'] is None and row['arrival_offsets_ns']==[0]*4096 and row['horizon_ns']==1,'Saturated calibration')
            check(row['calibration_capacity']['calibration_elapsed_ns']==row['coordinator_finish_ns'][-1]-row['coordinator_start_ns'][0],'Capacity denominator')
        else:
            c=cases[key[:3]];cal=c['calendar']
            check(row['calendar']==cal and row['arrival_date']==cal['arrival_date'] and row['arrival_offsets_ns']==cal['arrival_offsets_ns'] and row['horizon_ns']==cal['horizon_ns'] and row['arm_order']==c['arm_order'],'Frozen calendar/order')
        meta=row['input_meta'];pre=row['preflight'];check(pre['correctness_pass'] and row['postflight']['correctness_pass'],'Numerical assertion')
        identities[key[0]].add(digest([meta['capture_sha256'],meta['indices_sha256'],meta['gate_weight_sha256'],pre['logical_routing_sha256'],pre['float32_combining_weights_sha256']]))
        metrics[key]=clock_endpoints(row);raw[key]=row
    check(set(raw)==expected and all(len(v)==1 for v in identities.values()),'Missing rows or changed paired inputs')
    for key,c in cases.items():
        check(sorted(ARMS,key=lambda arm:raw[(*key,arm)]['epoch_monotonic_ns'])==c['arm_order'],'Actual arm order')
        check(raw[(*key,'current')]['preflight']['physical_assignment_digest']==raw[(*key,'fix')]['preflight']['physical_assignment_digest'],'Physical assignment policy changed')
    return {d:sorted(g for g in groups if groups[g]==d) for d in (*DATASETS,SENTINEL)},cases,metrics


def infer_pairs(paired,group_ids,cases,*,draw_count=DRAWS,seed=SEED):
    """Independent main+sentinel inference from independently measured pairs."""
    rng=np.random.default_rng(seed);tables=[rng.integers(0,8,(draw_count,8)) for _ in DATASETS]
    draw_hash=hashlib.sha256(b''.join(np.asarray(t,dtype='<i8').tobytes(order='C') for t in tables)).hexdigest()
    def scores(contrast,rho,metric,excluded=None,datasets=DATASETS):
        out={}
        for d in datasets:
            for g in group_ids[d]:
                terms=[paired[g,t,rho,contrast][metric] for t in TYPES if cases[g,t,rho]['calendar']['arrival_date']!=excluded]
                if terms:out[g]=math.fsum(terms)/len(terms)
        return out
    def point(contrast,rho,metric,datasets=DATASETS):
        values=scores(contrast,rho,metric,datasets=datasets)
        return math.exp(math.fsum(math.fsum(values[g] for g in group_ids[d])/len(group_ids[d]) for d in datasets)/len(datasets)),values
    def endpoint(contrast,rho,metric,datasets=DATASETS,guard=False):
        ratio,values=point(contrast,rho,metric,datasets);boot=[]
        for d in datasets:
            a=np.array([values[g] for g in group_ids[d]],float)
            boot.append(np.take(a,tables[DATASETS.index(d)]).sum(axis=1)/8)
        distribution=np.exp(np.add.reduce(boot)/len(boot));q=.995 if guard else .975
        upper=float(np.quantile(distribution,q,method='linear'));limit=1.02 if guard else .95
        passed=upper<=limit if guard else upper<limit
        out=dict(contrast=contrast,rho=rho,metric=metric,datasets=list(datasets),ratio=ratio,content_group_log_ratios=values,
            upper_confidence_bound=upper,ratio_limit=limit,passes=bool(passed),status='pass' if passed else 'not_established')
        if guard:out.update(one_sided_confidence_level=.995,guard_family_size=10)
        else:out['ci95']=[float(np.quantile(distribution,.025,method='linear')),upper]
        return out
    primary=endpoint('fix_over_stockoff',.85,'response_mean_ns');guards=[]
    for control in ('stockoff','current'):
        for d,rho in itertools.product(DATASETS,LOADS):guards.append(dict(name=f'{control}_{d}_rho{rho}_mean_response',**endpoint('fix_over_'+control,rho,'response_mean_ns',(d,),True)))
        guards.append(dict(name=f'{control}_high_load_p95_response',**endpoint('fix_over_'+control,.85,'response_p95_ns',guard=True)))
    main=set(group_ids['gsm8k']+group_ids['wildchat'])
    dates=sorted({c['calendar']['arrival_date'] for k,c in cases.items() if k[0] in main});deletions=[]
    for day,control in itertools.product(dates,('stockoff','current')):
        values=scores('fix_over_'+control,.85,'response_mean_ns',day);means=[];retained={};missing=[]
        for d in DATASETS:
            x=[values[g] for g in group_ids[d] if g in values];retained[d]=len(x)
            if x:means.append(math.fsum(x)/len(x))
            else:missing.append(d)
        ratio=None if missing else math.exp(math.fsum(means)/2)
        deletions.append(dict(excluded_date=day,control=control,ratio=ratio,retained_content_groups=retained,missing_datasets=missing,
            removed_arrival_bundles=sum(c['calendar']['arrival_date']==day for k,c in cases.items() if k[0] in main and k[2]==.85),passes=ratio is not None and ratio<=1.02))
    ranges={control:[min(v),max(v)] if (v:=[x['ratio'] for x in deletions if x['control']==control and x['ratio'] is not None]) else None for control in ('stockoff','current')}
    robust=all(x['passes'] for x in deletions);gate=primary['passes'] and all(g['passes'] for g in guards) and robust
    if gate:secondary={**endpoint('fix_over_current',.85,'response_mean_ns'), 'confirmatory':True,'gate_passed':True}
    else:
        ratio,values=point('fix_over_current',.85,'response_mean_ns')
        secondary=dict(contrast='fix_over_current',rho=.85,metric='response_mean_ns',ratio=ratio,content_group_log_ratios=values,ratio_limit=.95,
            ci95=None,upper_confidence_bound=None,passes=None,status='not_tested_gate_closed',confirmatory=False,gate_passed=False)
    checks=[]
    for control in ('stockoff','current'):
        for rho,metric in ((.5,'response_mean_ns'),(.85,'response_mean_ns'),(.85,'response_p95_ns')):
            ratio,values=point('fix_over_'+control,rho,metric,('mbpp',))
            checks.append(dict(control=control,contrast='fix_over_'+control,rho=rho,metric=metric,ratio=ratio,ratio_limit=1.02,
                passes=ratio<=1.02,ci95=None,inferential=False,content_group_log_ratios=values,status='pass' if ratio<=1.02 else 'point_limit_exceeded'))
    main_pass=bool(gate and secondary['passes']);sentinel_pass=all(x['passes'] for x in checks);advance=main_pass and sentinel_pass
    cells=[]
    for d,t,rho,c in itertools.product((*DATASETS,SENTINEL),TYPES,LOADS,CONTRASTS):
        cells.append(dict(dataset=d,arrival_type=t,rho=rho,contrast=c,content_groups=len(group_ids[d]),
            ratios={m:math.exp(math.fsum(paired[g,t,rho,c][m] for g in group_ids[d])/len(group_ids[d])) for m in METRICS}))
    return dict(primary=primary,guards=guards,secondary_current=secondary,
        calendar_date_robustness=dict(dates=dates,date_count=len(dates),ratio_limit=1.02,passes=robust,results=deletions,ratio_ranges=ranges),
        mbpp_sentinel=dict(dataset='mbpp',content_group_id=group_ids['mbpp'][0],content_groups=1,treatment_trials=12,passes=sentinel_pass,checks=checks),
        descriptive_cells=cells,primary_acceptance='pass' if gate else 'not_established',main_acceptance='pass' if main_pass else 'not_established',
        overall_advance='pass' if advance else 'do_not_advance',acceptance='pass' if advance else 'not_established',
        total_content_groups=17,main_content_groups=16,sentinel_content_groups=1,total_treatment_trials=204,main_treatment_trials=192,sentinel_treatment_trials=12,
        paired_comparisons=len(paired),matched_triplets=len(cases),bootstrap=dict(samples=draw_count,seed=seed,rng='numpy.random.default_rng/PCG64',
            strata=list(DATASETS),groups_per_stratum=8,quantile_method='linear',draw_indices_sha256=draw_hash))


def reconstruct(plan,rows,protocol_bytes,*,draw_count=DRAWS,seed=SEED):
    groups,cases,metrics=validate_inputs(plan,rows,protocol_bytes);paired={};details=[]
    for (g,t,rho),c in sorted(cases.items()):
        for contrast,(numerator,denominator) in CONTRASTS.items():
            a,b=metrics[g,t,rho,numerator],metrics[g,t,rho,denominator]
            logs={m:math.log(a[m])-math.log(b[m]) for m in METRICS};paired[g,t,rho,contrast]=logs
            details.append(dict(content_group_id=g,arrival_type=t,rho=rho,contrast=contrast,
                log_ratios={m:math.log(a[m])-math.log(b[m]) for m in a if a[m]>0 and b[m]>0},
                ratios={m:a[m]/b[m] if b[m]>0 else None for m in a},differences={m:a[m]-b[m] for m in a},control=b,treatment=a))
    return dict(phase=plan['phase'],trial_count=len(metrics),matched_triplets=len(cases),paired_comparisons=len(paired),
        pairs_per_contrast={c:len(cases) for c in CONTRASTS} if cases else {},trial_metrics=metrics,paired_cases=details,
        confirmatory=infer_pairs(paired,groups,cases,draw_count=draw_count,seed=seed) if plan['phase']=='final' else None)


def compare_summary(actual,summary):
    """Compare only after independent reconstruction; no summaries drive it."""
    check(summary.get('valid') is True,'Production analysis declares invalid')
    check(summary['phase']==actual['phase'],'Analysis phase differs')
    numeric=[];compared=0
    def compare(a,b,path):
        nonlocal compared
        if isinstance(a,dict):
            check(isinstance(b,dict),path+' object missing')
            for k,v in a.items():check(k in b,path+'.'+k+' missing');compare(v,b[k],path+'.'+k)
        elif isinstance(a,list):
            check(isinstance(b,list) and len(a)==len(b),path+' length differs')
            for i,(x,y) in enumerate(zip(a,b)):compare(x,y,f'{path}[{i}]')
        elif isinstance(a,bool) or a is None or isinstance(a,str):
            check(type(a) is type(b) and a==b,path+' differs');compared+=1
        elif isinstance(a,(int,float)):
            check(not isinstance(b,bool) and isinstance(b,(int,float)) and math.isfinite(b),path+' invalid number')
            delta=abs(a-b);numeric.append(delta)
            if type(a) is int:
                check(type(b) is int and a==b,path+f' exact integer difference {delta}')
            else:
                check(math.isclose(a,b,rel_tol=2e-12,abs_tol=2e-10),path+f' numerical difference {delta}')
            compared+=1
        else:raise TypeError(path)
    for k,v in [('observed_records',actual['trial_count']),('expected_records',actual['trial_count']),('valid_records',actual['trial_count']),('matched_triplets',actual['matched_triplets']),('paired_comparisons',actual['paired_comparisons']),('pairs_per_contrast',actual['pairs_per_contrast'])]:compare(v,summary[k],k)
    check(summary['missing_keys']==[] and summary['errors']==[],'Analysis excludes or rejects trials')
    trial_index={tuple(r[k] for k in ('content_group_id','arrival_type','rho','arm')):r for r in summary['trial_summaries']}
    check(len(trial_index)==len(summary['trial_summaries'])==actual['trial_count'],'Analysis trial duplication')
    for key,values in actual['trial_metrics'].items():compare(values,trial_index[key]['metrics'],f'trial[{key}]')
    pair_index={tuple(r[k] for k in ('content_group_id','arrival_type','rho','contrast')):r for r in summary['paired_cases']}
    check(len(pair_index)==len(summary['paired_cases'])==actual['paired_comparisons'],'Analysis pair duplication')
    for pair in actual['paired_cases']:
        key=tuple(pair[k] for k in ('content_group_id','arrival_type','rho','contrast'));compare(pair,pair_index[key],f'pair[{key}]')
    compare(actual['confirmatory'],summary['confirmatory'],'confirmatory')
    return dict(passed=True,scalar_comparisons=compared,numeric_comparisons=len(numeric),maximum_absolute_numeric_difference=max(numeric,default=0.))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','records','analysis','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[3])
    a=p.parse_args();plan_bytes=a.plan.read_bytes();plan=json.loads(plan_bytes)
    protocol=(a.repo_root/'docs/resident-fix/protocol-v1.md').read_bytes()
    for rel,expected in plan['source_files'].items():check(digest((a.repo_root/rel).read_bytes())==expected,'Changed frozen source '+rel)
    selected=(a.repo_root/'configs/resident-fix-selected-v1.json').read_bytes()
    check(digest(selected)==plan['selected_fix_sha256'] and json.loads(selected)==plan['selected_fix'],'Selected fix file')
    rows,pins=read_rows(a.records);actual=reconstruct(plan,rows,protocol)
    summary_bytes=a.analysis.read_bytes();comparison=compare_summary(actual,json.loads(summary_bytes))
    actual.pop('trial_metrics');actual.pop('paired_cases')
    result=dict(schema='resident_fix_independent_statistics_v1',passed=True,plan_sha256=plan['content_sha256'],source_sha256=plan['source_sha256'],
        old_closure_sha256=plan['old_closure_sha256'],compared_analysis_sha256=digest(summary_bytes),protocol_sha256=digest(protocol),
        raw_record_files=pins,comparison=comparison,recomputed=actual,program_sha256=digest(Path(__file__).read_bytes()),
        scope='Independent implementation within same team; main conditional content-bundle uncertainty; MBPP descriptive only. No production analysis/inference imports.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True,phase=actual['phase'],trials=actual['trial_count'],comparison=comparison)))
if __name__=='__main__':main()

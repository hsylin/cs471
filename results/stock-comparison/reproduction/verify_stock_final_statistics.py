#!/usr/bin/env python3
"""Independent CPU reconstruction of the frozen stock-comparison statistics.

Only stdlib/NumPy are imported. Production summaries are comparison targets,
never inputs to clock metrics, paired effects, bootstrap draws or decisions.
This is independent implementation within the same team, not external replication.
"""
from collections import Counter,defaultdict
from fractions import Fraction
import argparse,gzip,hashlib,itertools,json,math
from pathlib import Path
import numpy as np

DATASETS=('gsm8k','mbpp','wildchat')
TYPES=('code','conversation')
LOADS=(.5,.85)
ARMS=('stockoff','stockon32','ours')
CONTRASTS={'ours_over_stockoff':('ours','stockoff'),'ours_over_stockon32':('ours','stockon32'),'stockon32_over_stockoff':('stockon32','stockoff')}
METRICS=('response_mean_ns','response_p95_ns')
DRAWS=100000
SEED=471612


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
    values={'response_mean_ns':sum(responses)/n,'response_p95_ns':linear_quantile(responses,.95),'service_mean_ns':sum(service)/n,'queue_mean_ns':sum(queue)/n,**elapsed}
    stored=row['queue_metrics']
    check(stored['complete'] is True and stored['completed_jobs']==n and stored['unfinished_jobs']==0,'Incomplete stored trial')
    check(stored['vectors_ns']['response']==responses,'Stored responses differ from immutable arrivals')
    for metric,stat in (('response_mean_ns','mean'),('response_p95_ns','p95')):
        check(math.isclose(values[metric],stored['summary_ns']['response'][stat],rel_tol=2e-14,abs_tol=1e-6),'Stored response statistic differs')
    return values


def validate_inputs(plan,rows,protocol_bytes):
    check(sealed(plan),'Frozen plan content hash differs')
    check(plan.get('schema')=='stock_comparison_plan_v1' and plan.get('phase') in ('pilot','final'),'Expected pilot/final plan')
    phase=plan['phase'];n=6 if phase=='final' else 2
    check(plan.get('split')==phase and plan.get('arms')==list(ARMS) and plan.get('jobs')==4096 and plan.get('batch_tokens')==128,'Frozen scope differs')
    check(digest(protocol_bytes)==plan['protocol_sha256'],'Frozen protocol bytes differ')
    group_list=plan['banks'];groups={b['block_id']:b['dataset'] for b in group_list}
    check(len(groups)==len(group_list)==3*n,'Missing/duplicate groups')
    check(Counter(groups.values())==Counter({d:n for d in DATASETS}),'Unbalanced dataset allocation')
    declared=set(itertools.product(groups,TYPES,LOADS));cases={};orders={};source_identities=defaultdict(set)
    for case in plan['cases']:
        cal=case['calendar'];key=(case['block_id'],cal['arrival_type'],cal['rho'])
        check(key in declared and key not in cases and case['dataset']==groups[key[0]],'Unexpected/duplicate case')
        check(sealed(cal),'Calendar content hash differs')
        source=case['source_block'];elapsed=cal['stockoff_calibration_elapsed_ns']
        factor=Fraction(elapsed,1)/Fraction(str(cal['rho']))/source['original_horizon_ns']
        check(cal['arrival_offsets_ns']==[round(x*factor) for x in source['original_offsets_ns']],'Calendar scaling differs')
        check(cal['horizon_ns']==round(source['original_horizon_ns']*factor),'Calendar horizon differs')
        check(cal['arrival_type']==source['trace'] and cal['arrival_date']==source['source_date_utc'],'Source calendar association differs')
        check(cal['n_jobs']==source['jobs']==4096,'Calendar job count differs')
        check(sorted(case['arm_order'])==sorted(ARMS),'Invalid arm order')
        cases[key]=case;orders[key]=tuple(case['arm_order']);source_identities[key[:2]].add(digest(source))
    check(set(cases)==declared,'Incomplete case matrix')
    check(all(len(v)==1 for v in source_identities.values()),'Source calendar changed across offered loads')
    perms=set(itertools.permutations(ARMS))
    for d,rho in itertools.product(DATASETS if phase=='final' else (None,),LOADS):
        counts=Counter(order for key,order in orders.items() if key[2]==rho and (d is None or groups[key[0]]==d))
        check(set(counts)==perms and len(set(counts.values()))==1,'Unbalanced six-way order within dataset/load')
    expected={(*key,arm) for key in declared for arm in ARMS};metrics={};raw_index={};identities=defaultdict(set)
    for row in rows:
        key=tuple(row[k] for k in ('content_group_id','arrival_type','rho','arm'))
        check(key in expected and key not in raw_index,'Duplicate/unexpected trial')
        check(row.get('record_id')==digest({k:v for k,v in row.items() if k!='record_id'}),'Raw record hash differs')
        check(row.get('status')=='complete' and row.get('phase')==phase and row.get('evidence_eligible')==(phase=='final'),'Invalid trial disposition')
        check(row['dataset']==groups[key[0]] and row['block_id']==key[0],'Wrong content/dataset labels')
        check(row['plan_sha256']==plan['content_sha256'] and row['source_sha256']==plan['source_sha256'],'Plan/source pin differs')
        check(str(row['allocation_id'])==str(plan['hardware']['allocation_id']),'Different allocation')
        case=cases[key[:3]];cal=case['calendar']
        check(row['calendar']==cal and row['arrival_date']==cal['arrival_date'] and row['arrival_offsets_ns']==cal['arrival_offsets_ns'] and row['horizon_ns']==cal['horizon_ns'],'Row calendar differs from frozen case')
        check(row['arm_order']==case['arm_order'],'Row arm-order declaration differs')
        meta=row['input_meta'];pre=row['preflight']
        identity={'capture':meta['capture_sha256'],'indices':meta['indices_sha256'],'source_prompts':meta['source_prompt_ids'],'gate':meta['gate_weight_sha256'],'routes':pre['logical_routing_sha256'],'combining':pre['float32_combining_weights_sha256']}
        identities[key[0]].add(digest(identity))
        check(pre['correctness_pass'] is True and row['postflight']['correctness_pass'] is True,'Recorded correctness failed')
        metrics[key]=clock_endpoints(row);raw_index[key]=row
    check(set(raw_index)==expected,'Missing trials; no exclusions allowed')
    check(all(len(v)==1 for v in identities.values()),'Unpaired input/gate/logical-route identities')
    for key,case in cases.items():
        order=sorted(ARMS,key=lambda a:raw_index[(*key,a)]['epoch_monotonic_ns'])
        check(order==case['arm_order'],'Actual epoch arm order differs')
    group_ids={d:sorted(g for g in groups if groups[g]==d) for d in DATASETS}
    return group_ids,cases,metrics


def reconstruct(plan,rows,protocol_bytes,*,draw_count=DRAWS,seed=SEED):
    group_ids,cases,metrics=validate_inputs(plan,rows,protocol_bytes)
    triplets=len(cases);paired={};pair_details=[]
    for (g,typ,rho),case in sorted(cases.items()):
        for contrast,(numerator,denominator) in CONTRASTS.items():
            a=metrics[g,typ,rho,numerator];b=metrics[g,typ,rho,denominator]
            logs={m:math.log(a[m])-math.log(b[m]) for m in METRICS}
            paired[g,typ,rho,contrast]=logs
            pair_details.append(dict(content_group_id=g,arrival_type=typ,rho=rho,contrast=contrast,log_ratios=logs,ratios={m:a[m]/b[m] for m in METRICS},differences={m:a[m]-b[m] for m in METRICS},control={m:b[m] for m in METRICS},treatment={m:a[m] for m in METRICS}))
    result={'phase':plan['phase'],'trial_count':len(metrics),'matched_triplets':triplets,'paired_comparisons':len(paired),'pairs_per_contrast':{c:triplets for c in CONTRASTS},'trial_metrics':metrics,'paired_cases':pair_details,'confirmatory':None}
    if plan['phase']=='pilot':return result
    check(draw_count>0 and type(draw_count) is int,'Invalid draw count')
    rng=np.random.default_rng(seed)
    resamples=[rng.integers(0,6,(draw_count,6)) for _ in DATASETS]
    draw_hash=hashlib.sha256(b''.join(np.asarray(x,dtype='<i8').tobytes(order='C') for x in resamples)).hexdigest()
    def group_scores(contrast,rho,metric,excluded_date=None):
        value={}
        for d in DATASETS:
            for g in group_ids[d]:
                terms=[paired[g,t,rho,contrast][metric] for t in TYPES if cases[g,t,rho]['calendar']['arrival_date']!=excluded_date]
                if terms:value[g]=math.fsum(terms)/len(terms)
        return value
    def estimate(contrast,rho,metric,datasets=DATASETS,guard=False):
        scores=group_scores(contrast,rho,metric);tables=[];stratum_means=[];logs={}
        for d in datasets:
            vals=np.array([scores[g] for g in group_ids[d]],dtype=np.float64)
            stratum_means.append(float(np.mean(vals)));logs.update({g:scores[g] for g in group_ids[d]})
            # Whole bundle resampling: one index carries every nested condition.
            tables.append(np.take(vals,resamples[DATASETS.index(d)]).sum(axis=1)/6)
        boot=np.exp(np.add.reduce(tables)/len(tables))
        upper=float(np.quantile(boot,1-.05/14 if guard else .975,method='linear'))
        point=math.exp(math.fsum(stratum_means)/len(stratum_means));limit=1.02 if guard else .95
        passed=upper<=limit if guard else upper<limit
        out={'contrast':contrast,'rho':rho,'metric':metric,'datasets':list(datasets),'ratio':point,'content_group_log_ratios':logs,'upper_confidence_bound':upper,'ratio_limit':limit,'passes':bool(passed),'status':'pass' if passed else 'not_established'}
        if guard:out.update(one_sided_confidence_level=1-.05/14,guard_family_size=14)
        else:out['ci95']=[float(np.quantile(boot,.025,method='linear')),upper]
        return out
    primary=estimate('ours_over_stockoff',.85,'response_mean_ns')
    guards=[]
    for control in ('stockoff','stockon32'):
        for d,rho in itertools.product(DATASETS,LOADS):
            guards.append({'name':f'{control}_{d}_rho{rho}_mean_response',**estimate('ours_over_'+control,rho,'response_mean_ns',(d,),True)})
        guards.append({'name':f'{control}_high_load_p95_response',**estimate('ours_over_'+control,.85,'response_p95_ns',guard=True)})
    dates=sorted({case['calendar']['arrival_date'] for case in cases.values()});deletions=[]
    for day,control in itertools.product(dates,('stockoff','stockon32')):
        score=group_scores('ours_over_'+control,.85,'response_mean_ns',day);means=[];retained={};missing=[]
        for d in DATASETS:
            values=[score[g] for g in group_ids[d] if g in score];retained[d]=len(values)
            if values:means.append(math.fsum(values)/len(values))
            else:missing.append(d)
        ratio=math.exp(math.fsum(means)/3) if not missing else None
        removed=sum(case['calendar']['arrival_date']==day for key,case in cases.items() if key[2]==.85)
        deletions.append(dict(excluded_date=day,control=control,ratio=ratio,retained_content_groups=retained,missing_datasets=missing,removed_arrival_bundles=removed,passes=ratio is not None and ratio<=1.02))
    robust=all(d['passes'] for d in deletions)
    ranges={c:[min(v),max(v)] if (v:=[d['ratio'] for d in deletions if d['control']==c and d['ratio'] is not None]) else None for c in ('stockoff','stockon32')}
    gate=primary['passes'] and all(g['passes'] for g in guards) and robust
    if gate:
        secondary=estimate('ours_over_stockon32',.85,'response_mean_ns');secondary.update(confirmatory=True,gate_passed=True)
    else:
        logs=group_scores('ours_over_stockon32',.85,'response_mean_ns')
        point=math.exp(math.fsum(math.fsum(logs[g] for g in group_ids[d])/6 for d in DATASETS)/3)
        secondary=dict(contrast='ours_over_stockon32',rho=.85,metric='response_mean_ns',ratio=point,content_group_log_ratios=logs,ratio_limit=.95,ci95=None,upper_confidence_bound=None,passes=None,status='not_tested_gate_closed',confirmatory=False,gate_passed=False)
    cells=[]
    for d,t,rho,c in itertools.product(DATASETS,TYPES,LOADS,CONTRASTS):
        cells.append(dict(dataset=d,arrival_type=t,rho=rho,contrast=c,content_groups=6,ratios={m:math.exp(math.fsum(paired[g,t,rho,c][m] for g in group_ids[d])/6) for m in METRICS}))
    result['confirmatory']=dict(primary=primary,guards=guards,secondary_stockon32=secondary,calendar_date_robustness=dict(date_count=len(dates),dates=dates,ratio_limit=1.02,passes=robust,results=deletions,ratio_ranges=ranges),descriptive_cells=cells,primary_acceptance='pass' if gate else 'not_established',acceptance='pass' if gate and secondary['passes'] else 'not_established',worst_stock_relative_point_ratio=max(primary['ratio'],secondary['ratio']),content_groups=18,paired_comparisons=len(paired),matched_triplets=triplets,treatment_trials=len(metrics),bootstrap=dict(samples=draw_count,seed=seed,rng='numpy.random.default_rng/PCG64',strata=list(DATASETS),groups_per_stratum=6,quantile_method='linear',draw_indices_sha256=draw_hash))
    return result


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
    for name in ('plan','records','protocol','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--summary','--analysis',dest='summary',type=Path,required=True)
    args=p.parse_args();check(not args.output.exists(),'Fresh output required')
    plan_bytes=args.plan.read_bytes();protocol_bytes=args.protocol.read_bytes();plan=json.loads(plan_bytes)
    rows,record_files=read_rows(args.records)
    actual=reconstruct(plan,rows,protocol_bytes)
    summary_bytes=args.summary.read_bytes();summary=json.loads(summary_bytes)
    check(summary['plan_sha256']==plan['content_sha256'] and summary['source_sha256']==plan['source_sha256'],'Analysis plan/source identity differs')
    comparison=compare_summary(actual,summary)
    actual.pop('trial_metrics');actual.pop('paired_cases')
    result=dict(schema='stock_independent_statistics_audit_v1',passed=True,plan_sha256=plan['content_sha256'],plan_file_sha256=digest(plan_bytes),source_sha256=plan['source_sha256'],protocol_sha256=digest(protocol_bytes),raw_record_files=record_files,compared_analysis_sha256=digest(summary_bytes),comparison=comparison,recomputed=actual,program_sha256=digest(Path(__file__).read_bytes()),implementation='Independent stdlib/NumPy reconstruction; no production imports. Same team, not external-laboratory reproduction.',scope='Conditional whole-content-bundle variation; no arrival-population, new-day, full-serving or hardware-population CI. Execution/routes/bank audits remain separately required.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'passed':True,'phase':actual['phase'],'trials':actual['trial_count'],'triplets':actual['matched_triplets'],'comparison':comparison}))

if __name__=='__main__':main()

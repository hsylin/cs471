"""Prespecified conditional inference against two genuine stock configurations.

The verified trial matrix is reduced to whole-content paired log effects. All
jobs, arrival types, loads and arms stay nested in a content bundle. Intervals do
not estimate new-day, deployment or hardware-population uncertainty.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from datetime import date
import hashlib
import itertools
import math
import statistics
import numpy as np

DATASETS=('gsm8k','mbpp','wildchat')
ARRIVAL_TYPES=('code','conversation')
LOADS=(.5,.85)
CONTRASTS=('ours_over_stockoff','ours_over_stockon32','stockon32_over_stockoff')
METRICS=('response_mean_ns','response_p95_ns')
BOOTSTRAP_SAMPLES=100_000
BOOTSTRAP_SEED=471612
GROUPS_PER_DATASET=6
GUARD_FAMILY_SIZE=14


def _validate(pairs,groups):
    by_dataset=defaultdict(list);seen=set()
    for group in groups:
        gid,d=group['id'],group['dataset']
        if not isinstance(gid,str) or not gid or gid in seen or d not in DATASETS:raise ValueError('Unique declared content groups required')
        seen.add(gid);by_dataset[d].append(gid)
    if set(by_dataset)!=set(DATASETS) or any(len(by_dataset[d])!=6 for d in DATASETS):raise ValueError('Exactly six final content groups per dataset required')
    by_dataset={d:sorted(by_dataset[d]) for d in DATASETS};lookup={g:d for d,gs in by_dataset.items() for g in gs}
    expected=set(itertools.product(lookup,ARRIVAL_TYPES,LOADS,CONTRASTS));index={};source_dates={}
    for pair in pairs:
        key=tuple(pair[k] for k in ('content_group_id','arrival_type','rho','contrast'))
        if key not in expected or key in index or pair['dataset']!=lookup[key[0]]:raise ValueError('Duplicate, unexpected or mislabeled paired unit')
        day=pair['arrival_date']
        if not isinstance(day,str) or len(day)!=10 or date.fromisoformat(day).isoformat()!=day:raise ValueError('ISO source date required')
        if key[:2] in source_dates and source_dates[key[:2]]!=day:raise ValueError('Calendar date changed across arms or loads')
        source_dates[key[:2]]=day
        if any(not isinstance(pair['log_ratios'][m],(int,float,np.number)) or not math.isfinite(pair['log_ratios'][m]) for m in METRICS):raise ValueError('Finite paired log ratios required')
        index[key]=pair
    if set(index)!=expected:raise ValueError('Complete216 paired comparisons required for final inference')
    # These three ratios derive from the same three measured arm outcomes.
    for gid,typ,rho in itertools.product(lookup,ARRIVAL_TYPES,LOADS):
        for metric in METRICS:
            off=index[gid,typ,rho,CONTRASTS[0]]['log_ratios'][metric]
            on=index[gid,typ,rho,CONTRASTS[1]]['log_ratios'][metric]
            stock=index[gid,typ,rho,CONTRASTS[2]]['log_ratios'][metric]
            if not math.isclose(off,on+stock,rel_tol=0,abs_tol=1e-10):raise ValueError('Three paired contrasts violate shared-outcome identity')
    return by_dataset,index


def _logs(index,groups,contrast,rho,metric,exclude_date=None):
    values={}
    for gid in groups:
        selected=[index[gid,t,rho,contrast] for t in ARRIVAL_TYPES]
        selected=[p for p in selected if p['arrival_date']!=exclude_date]
        if selected:values[gid]=statistics.mean(p['log_ratios'][metric] for p in selected)
    return values


def infer_endpoints(pairs,groups,*,bootstrap_samples=BOOTSTRAP_SAMPLES,bootstrap_seed=BOOTSTRAP_SEED):
    """Final-only pure helper; production calls use the frozen default arguments.

    Optional draw controls permit inexpensive independent CPU oracle tests. The
    selected controls, exact draw digest and confidence levels are in output.
    """
    if isinstance(bootstrap_samples,bool) or not isinstance(bootstrap_samples,int) or bootstrap_samples<1:raise ValueError('Positive integer bootstrap count required')
    by_dataset,index=_validate(pairs,groups)
    rng=np.random.default_rng(bootstrap_seed)
    draws={d:rng.integers(0,6,size=(bootstrap_samples,6)) for d in DATASETS}
    def endpoint(name,contrast,rho,metric,datasets=DATASETS,guard=False):
        points=[];samples=[];logs={}
        for d in datasets:
            values=_logs(index,by_dataset[d],contrast,rho,metric);logs.update(values)
            a=np.asarray([values[g] for g in by_dataset[d]],dtype=float)
            points.append(float(a.mean()));samples.append(a[draws[d]].mean(axis=1))
        boot=np.exp(np.mean(samples,axis=0));q=1-.05/GUARD_FAMILY_SIZE if guard else .975
        upper=float(np.quantile(boot,q,method='linear'));limit=1.02 if guard else .95
        out=dict(name=name,contrast=contrast,rho=rho,metric=metric,datasets=list(datasets),
            ratio=math.exp(statistics.mean(points)),content_group_log_ratios=logs,upper_confidence_bound=upper,
            ratio_limit=limit,passes=upper<=limit if guard else upper<limit,
            criterion='upper <= limit' if guard else 'upper < limit',
            interval_scope='Conditional content-bundle bootstrap; not production-arrival or hardware-population uncertainty')
        if guard:out.update(one_sided_confidence_level=q,guard_family_size=GUARD_FAMILY_SIZE)
        else:out['ci95']=[float(np.quantile(boot,.025,method='linear')),upper]
        out['status']='pass' if out['passes'] else 'not_established'
        return out
    primary=endpoint('high_load_mean_response_vs_stockoff','ours_over_stockoff',.85,'response_mean_ns')
    guards=[]
    for control in ('stockoff','stockon32'):
        contrast='ours_over_'+control
        for d,rho in itertools.product(DATASETS,LOADS):
            guards.append(endpoint(f'{control}_{d}_rho{rho}_mean_response',contrast,rho,'response_mean_ns',(d,),True))
        guards.append(endpoint(f'{control}_high_load_p95_response',contrast,.85,'response_p95_ns',guard=True))
    dates=sorted({p['arrival_date'] for p in index.values()});deletions=[]
    for day,control in itertools.product(dates,('stockoff','stockon32')):
        points=[];retained={};missing=[]
        for d in DATASETS:
            values=_logs(index,by_dataset[d],'ours_over_'+control,.85,'response_mean_ns',day)
            retained[d]=len(values)
            if values:points.append(statistics.mean(values.values()))
            else:missing.append(d)
        ratio=math.exp(statistics.mean(points)) if not missing else None
        deletions.append(dict(excluded_date=day,control=control,ratio=ratio,retained_content_groups=retained,
            missing_datasets=missing,removed_arrival_bundles=sum(index[g,t,.85,'ours_over_stockoff']['arrival_date']==day for d in DATASETS for g in by_dataset[d] for t in ARRIVAL_TYPES),
            passes=ratio is not None and ratio<=1.02))
    robustness_pass=all(x['passes'] for x in deletions)
    ranges={}
    for control in ('stockoff','stockon32'):
        values=[x['ratio'] for x in deletions if x['control']==control and x['ratio'] is not None]
        ranges[control]=[min(values),max(values)] if values else None
    robustness=dict(method='Delete one actual source calendar date, linking both services; point sensitivity only',
        date_count=len(dates),dates=dates,ratio_limit=1.02,passes=robustness_pass,results=deletions,ratio_ranges=ranges,
        weighting='Remaining types within each surviving group; groups within dataset; three datasets equally. Missing dataset fails.',
        criterion='Every date-deletion point ratio <= 1.02 against both stock controls; no CI')
    gate=primary['passes'] and all(g['passes'] for g in guards) and robustness_pass
    if gate:
        secondary=endpoint('high_load_mean_response_vs_stockon32','ours_over_stockon32',.85,'response_mean_ns')
        secondary.update(confirmatory=True,gate_passed=True)
    else:
        points=[];logs={}
        for d in DATASETS:
            values=_logs(index,by_dataset[d],'ours_over_stockon32',.85,'response_mean_ns');logs.update(values);points.append(statistics.mean(values.values()))
        secondary=dict(name='high_load_mean_response_vs_stockon32',contrast='ours_over_stockon32',rho=.85,metric='response_mean_ns',
            ratio=math.exp(statistics.mean(points)),content_group_log_ratios=logs,ratio_limit=.95,ci95=None,upper_confidence_bound=None,
            passes=None,status='not_tested_gate_closed',confirmatory=False,gate_passed=False)
    cells=[]
    for d,t,rho,c in itertools.product(DATASETS,ARRIVAL_TYPES,LOADS,CONTRASTS):
        cells.append(dict(dataset=d,arrival_type=t,rho=rho,contrast=c,content_groups=6,
            ratios={m:math.exp(statistics.mean(index[g,t,rho,c]['log_ratios'][m] for g in by_dataset[d])) for m in METRICS},
            scope='descriptive; no cell-level CI or significance test'))
    return dict(schema='stock_comparison_inference_v1',primary=primary,guards=guards,secondary_stockon32=secondary,
        calendar_date_robustness=robustness,descriptive_cells=cells,
        primary_acceptance='pass' if gate else 'not_established',
        acceptance='pass' if gate and secondary['passes'] else 'not_established',
        claim='Useful high-load mean-response gain against both tested stock configurations, conditional on all prespecified safeguards',
        worst_stock_relative_point_ratio=max(primary['ratio'],secondary['ratio']),
        content_groups=18,paired_comparisons=len(index),matched_triplets=len(index)//3,treatment_trials=216,
        bootstrap=dict(samples=bootstrap_samples,seed=bootstrap_seed,rng='numpy.random.default_rng/PCG64',
            strata=list(DATASETS),groups_per_stratum=6,unit='whole content bundle with two arrival types, two loads, three arms',
            quantile_method='linear',draw_indices_sha256=hashlib.sha256(b''.join(np.asarray(draws[d],dtype='<i8').tobytes(order='C') for d in DATASETS)).hexdigest(),
            draw_indices_encoding='Concatenate complete [samples,6] tables in DATASETS order; little-endian signed int64 C-order bytes, no header.',
            scope='Conditional on selected external trace material, content construction and one hardware allocation; not uncertainty across new days or deployments'),
        hierarchy='Stockoff primary, all14 guards and both-control date-deletion checks gate the stockon32 secondary; gain against a weaker comparator cannot rescue failure.')

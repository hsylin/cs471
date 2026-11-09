"""Prespecified fresh-content inference for one residual-path repair.

Inference uses exactly eight GSM8K-train and eight WildChat whole-content bundles.
The prior fused method is a control; there is no MBPP allocation in this study.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import date
import hashlib
import itertools
import math
import statistics
import numpy as np

DATASETS=('gsm8k_train','wildchat')
ARRIVAL_TYPES=('code','conversation')
LOADS=(.5,.85)
CONTRASTS=('fix_over_stockoff','fix_over_previous','previous_over_stockoff')
METRICS=('response_mean_ns','response_p95_ns')
BOOTSTRAP_SAMPLES=100_000
BOOTSTRAP_SEED=471812
GROUPS_PER_DATASET=8
GUARD_FAMILY_SIZE=10


def _validate(pairs,groups):
    by=defaultdict(list);lookup={}
    for group in groups:
        gid,d=group['id'],group['dataset']
        if not isinstance(gid,str) or not gid or gid in lookup or d not in DATASETS:raise ValueError('Unique declared main groups required')
        lookup[gid]=d;by[d].append(gid)
    if set(by)!=set(DATASETS) or any(len(by[d])!=8 for d in DATASETS):raise ValueError('Exactly8GSM8K-train+8WildChat final groups required')
    by={d:sorted(by[d]) for d in DATASETS}
    expected=set(itertools.product(lookup,ARRIVAL_TYPES,LOADS,CONTRASTS));index={};dates={}
    for pair in pairs:
        key=tuple(pair[k] for k in ('content_group_id','arrival_type','rho','contrast'))
        if key not in expected or key in index or pair['dataset']!=lookup[key[0]]:raise ValueError('Duplicate, unexpected or mislabeled paired unit')
        day=pair['arrival_date']
        if not isinstance(day,str) or len(day)!=10 or date.fromisoformat(day).isoformat()!=day:raise ValueError('ISO actual source date required')
        if key[:2] in dates and dates[key[:2]]!=day:raise ValueError('Calendar date changed across arms/loads')
        dates[key[:2]]=day
        if any(isinstance(pair['log_ratios'][m],(bool,np.bool_)) or not isinstance(pair['log_ratios'][m],(int,float,np.number)) or not math.isfinite(pair['log_ratios'][m]) for m in METRICS):raise ValueError('Finite paired log ratios required')
        index[key]=pair
    if set(index)!=expected:raise ValueError('Complete192 paired comparisons required')
    for gid,t,rho in itertools.product(lookup,ARRIVAL_TYPES,LOADS):
        for m in METRICS:
            a,b,c=(index[gid,t,rho,contrast]['log_ratios'][m] for contrast in CONTRASTS)
            if not math.isclose(a,b+c,rel_tol=0,abs_tol=1e-10):raise ValueError('Three contrasts violate shared-outcome identity')
    return by,index


def _logs(index,groups,contrast,rho,metric,exclude_date=None):
    out={}
    for gid in groups:
        pairs=[index[gid,t,rho,contrast] for t in ARRIVAL_TYPES]
        pairs=[p for p in pairs if p['arrival_date']!=exclude_date]
        if pairs:out[gid]=statistics.mean(p['log_ratios'][metric] for p in pairs)
    return out


def infer_endpoints(pairs,groups,*,bootstrap_samples=BOOTSTRAP_SAMPLES,bootstrap_seed=BOOTSTRAP_SEED):
    """Final-only helper. Production must use frozen default draw arguments.

    Optional controls enable cheap deterministic CPU tests; the chosen count,
    seed, encoding and exact draw digest are always emitted for verification.
    """
    if isinstance(bootstrap_samples,bool) or not isinstance(bootstrap_samples,int) or bootstrap_samples<1:raise ValueError('Positive integer bootstrap count required')
    by,index=_validate(pairs,groups)
    rng=np.random.default_rng(bootstrap_seed)
    draws={d:rng.integers(0,8,size=(bootstrap_samples,8)) for d in DATASETS}
    def point(contrast,rho,metric,datasets=DATASETS):
        values={d:_logs(index,by[d],contrast,rho,metric) for d in datasets}
        return math.exp(statistics.mean(statistics.mean(v.values()) for v in values.values())),values
    def endpoint(name,contrast,rho,metric,datasets=DATASETS,guard=False):
        ratio,values=point(contrast,rho,metric,datasets);resampled=[]
        for d in datasets:
            a=np.asarray([values[d][gid] for gid in by[d]],float)
            resampled.append(a[draws[d]].mean(axis=1))
        boot=np.exp(np.mean(resampled,axis=0));q=1-.05/GUARD_FAMILY_SIZE if guard else .975
        upper=float(np.quantile(boot,q,method='linear'));limit=1.02 if guard else .95
        result=dict(name=name,contrast=contrast,rho=rho,metric=metric,datasets=list(datasets),ratio=ratio,
            content_group_log_ratios={g:x for v in values.values() for g,x in v.items()},upper_confidence_bound=upper,
            ratio_limit=limit,passes=upper<=limit if guard else upper<limit,
            criterion='upper <= limit' if guard else 'upper < limit',
            interval_scope='Conditional main-content-bundle bootstrap; no new-day or hardware-population inference')
        if guard:result.update(one_sided_confidence_level=q,guard_family_size=GUARD_FAMILY_SIZE)
        else:result['ci95']=[float(np.quantile(boot,.025,method='linear')),upper]
        result['status']='pass' if result['passes'] else 'not_established'
        return result
    primary=endpoint('high_load_mean_response_vs_stockoff','fix_over_stockoff',.85,'response_mean_ns')
    guards=[]
    for control in ('stockoff','previous'):
        contrast='fix_over_'+control
        for d,rho in itertools.product(DATASETS,LOADS):
            guards.append(endpoint(f'{control}_{d}_rho{rho}_mean_response',contrast,rho,'response_mean_ns',(d,),True))
        guards.append(endpoint(f'{control}_high_load_p95_response',contrast,.85,'response_p95_ns',guard=True))
    main_ids=[g for d in DATASETS for g in by[d]]
    actual_dates=sorted({index[g,t,.85,'fix_over_stockoff']['arrival_date'] for g,t in itertools.product(main_ids,ARRIVAL_TYPES)})
    deleted=[]
    for day,control in itertools.product(actual_dates,('stockoff','previous')):
        points=[];retained={};missing=[]
        for d in DATASETS:
            logs=_logs(index,by[d],'fix_over_'+control,.85,'response_mean_ns',day);retained[d]=len(logs)
            if logs:points.append(statistics.mean(logs.values()))
            else:missing.append(d)
        ratio=math.exp(statistics.mean(points)) if not missing else None
        deleted.append(dict(excluded_date=day,control=control,ratio=ratio,retained_content_groups=retained,missing_datasets=missing,
            removed_arrival_bundles=sum(index[g,t,.85,'fix_over_stockoff']['arrival_date']==day for g,t in itertools.product(main_ids,ARRIVAL_TYPES)),
            passes=ratio is not None and ratio<=1.02))
    ranges={}
    for control in ('stockoff','previous'):
        values=[x['ratio'] for x in deleted if x['control']==control and x['ratio'] is not None]
        ranges[control]=[min(values),max(values)] if values else None
    robustness=dict(method='Delete one actual main source calendar date, linking services; point sensitivity only',
        dates=actual_dates,date_count=len(actual_dates),ratio_limit=1.02,passes=all(x['passes'] for x in deleted),results=deleted,ratio_ranges=ranges,
        weighting='Remaining arrival types within group, surviving groups within dataset, two main datasets equally; missing dataset fails.',
        scope='Main datasets only; no additional confidence interval or arrival-population claim')
    main_gate=primary['passes'] and all(g['passes'] for g in guards) and robustness['passes']
    if main_gate:
        secondary=endpoint('high_load_mean_response_vs_previous','fix_over_previous',.85,'response_mean_ns')
        secondary.update(confirmatory=True,gate_passed=True)
    else:
        ratio,values=point('fix_over_previous',.85,'response_mean_ns')
        secondary=dict(name='high_load_mean_response_vs_previous',contrast='fix_over_previous',rho=.85,metric='response_mean_ns',ratio=ratio,
            content_group_log_ratios={g:x for v in values.values() for g,x in v.items()},ratio_limit=.95,
            ci95=None,upper_confidence_bound=None,passes=None,status='not_tested_gate_closed',confirmatory=False,gate_passed=False)
    main_pass=main_gate and secondary['passes'] is True
    cells=[]
    for d,t,rho,contrast in itertools.product(DATASETS,ARRIVAL_TYPES,LOADS,CONTRASTS):
        cells.append(dict(dataset=d,arrival_type=t,rho=rho,contrast=contrast,content_groups=len(by[d]),
            ratios={m:math.exp(statistics.mean(index[g,t,rho,contrast]['log_ratios'][m] for g in by[d])) for m in METRICS},
            scope='Descriptive; no cell-level CI or significance test'))
    return dict(schema='resident_residual_inference_v1',primary=primary,guards=guards,secondary_previous=secondary,
        calendar_date_robustness=robustness,descriptive_cells=cells,
        primary_acceptance='pass' if main_gate else 'not_established',main_acceptance='pass' if main_pass else 'not_established',
        overall_advance='pass' if main_pass else 'do_not_advance',acceptance='pass' if main_pass else 'not_established',
        main_claim='Useful high-load mean-response gain versus stockoff and previous on the two main datasets, conditional on main safeguards',
        total_content_groups=16,main_content_groups=16,total_treatment_trials=192,
        main_treatment_trials=192,paired_comparisons=len(index),matched_triplets=len(index)//3,
        bootstrap=dict(samples=bootstrap_samples,seed=bootstrap_seed,rng='numpy.random.default_rng/PCG64',strata=list(DATASETS),
            groups_per_stratum=8,unit='Whole main content bundle: both arrival types, loads and all arms',quantile_method='linear',
            draw_indices_sha256=hashlib.sha256(b''.join(np.asarray(draws[d],dtype='<i8').tobytes(order='C') for d in DATASETS)).hexdigest(),
            draw_indices_encoding='Concatenate complete [samples,8] tables in GSM8K-train,WildChat order; little-endian signed int64 C-order; no header',
            scope='Conditional on selected trace material, content construction, fixed payload mix and one allocation; excludes new-day/deployment variation'),
        hierarchy='Stockoff primary,10guards and date checks gate previous-fused secondary; both hierarchical useful-gain endpoints are required for advance.')


def build_arm_orders(groups,*,seed,phase):
    """Frozen code/conversation reversal; near-balanced six permutations.

    Returns normalized case dictionaries, independent of calendars/outcomes.
    Caller merges them by (content_group_id, arrival_type, rho).
    """
    import random
    if phase not in ('pilot','final'):raise ValueError('Order phase must be pilot/final')
    n=2 if phase=='pilot' else 8;by=defaultdict(list)
    for g in groups:by[g['dataset']].append(g['id'])
    if set(by)!=set(DATASETS) or any(len(by[d])!=n for d in DATASETS):raise ValueError('Wrong phase group allocation')
    rng=random.Random(seed);arms=('stockoff','previous','fix')
    bases=((arms[0],arms[1],arms[2]),(arms[0],arms[2],arms[1]),(arms[1],arms[0],arms[2]))
    out=[]
    for di,d in enumerate(DATASETS):
        for li,rho in enumerate(LOADS):
            deficient=(di+li)%3
            roster=[]
            for pi,base in enumerate(bases):
                count=(0 if pi==deficient else 1) if phase=='pilot' else (2 if pi==deficient else 3)
                for j in range(count):roster.append(base if (j+di+li+pi)%2==0 else base[::-1])
            ids=sorted(by[d]);rng.shuffle(ids);rng.shuffle(roster)
            for gid,order in zip(ids,roster):
                for typ,permutation in (('code',order),('conversation',order[::-1])):
                    out.append(dict(content_group_id=gid,arrival_type=typ,rho=rho,arm_order=list(permutation)))
    validate_arm_orders(out,groups,phase=phase)
    return out


def validate_arm_orders(cases,groups,*,phase):
    """Exact within-content precedence balance, not impossible equal positions."""
    from collections import Counter
    if phase not in ('pilot','final'):raise ValueError('Order phase must be pilot/final')
    n=2 if phase=='pilot' else 8;by=defaultdict(list);lookup={}
    for g in groups:
        if g['id'] in lookup:raise ValueError('Duplicate order group')
        lookup[g['id']]=g['dataset'];by[g['dataset']].append(g['id'])
    if set(by)!=set(DATASETS) or any(len(by[d])!=n for d in DATASETS):raise ValueError('Wrong phase order allocation')
    arms=('stockoff','previous','fix');perms=set(itertools.permutations(arms));index={}
    for c in cases:
        k=(c['content_group_id'],c['arrival_type'],c['rho']);order=tuple(c['arm_order'])
        if k in index or order not in perms:raise ValueError('Duplicate case or invalid arm permutation')
        index[k]=order
    if set(index)!=set(itertools.product(lookup,ARRIVAL_TYPES,LOADS)):raise ValueError('Incomplete order matrix')
    for gid,rho in itertools.product(lookup,LOADS):
        if index[gid,'code',rho]!=index[gid,'conversation',rho][::-1]:raise ValueError('Arrival-type orders must be exact reversals within content/load')
    diagnostics=[]
    for rho in LOADS:
        for selected in (DATASETS,*( (d,) for d in DATASETS)):
            orders=[order for (gid,t,load),order in index.items() if load==rho and lookup[gid] in selected]
            counts=Counter(orders);allcounts=[counts[p] for p in perms]
            if max(allcounts)-min(allcounts)>1:raise ValueError('Permutation counts not near-balanced within main dataset/load or pooled load')
            for a,b in itertools.combinations(arms,2):
                if sum(o.index(a)<o.index(b) for o in orders)*2!=len(orders):raise ValueError('Treatment pair precedence unbalanced')
            if phase=='final':
                for typ in ARRIVAL_TYPES:
                    ct=Counter(order for (gid,t,load),order in index.items() if load==rho and t==typ and lookup[gid] in selected)
                    if max(ct[p] for p in perms)-min(ct[p] for p in perms)>1:raise ValueError('Arrival-type permutation counts not near-balanced')
            diagnostics.append(dict(datasets=list(selected),rho=rho,cases=len(orders),
                permutation_counts={'/'.join(p):counts[p] for p in sorted(perms)},
                arm_position_counts={a:[sum(o[pos]==a for o in orders) for pos in range(3)] for a in arms},
                pairwise_precedence_balanced=True))
    return diagnostics

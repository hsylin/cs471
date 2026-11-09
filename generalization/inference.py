"""Prespecified inference conditional on the selected external trace material.

Inputs are verified paired trial statistics, not individual jobs. Resampling a
whole content bundle preserves both arrival types and both rate definitions.
Calendar dates are fixed design attributes; these intervals do not estimate
uncertainty over new production days, services, or hardware allocations.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import itertools
import math
import statistics
import numpy as np

DATASETS = ('gsm8k', 'mbpp', 'wildchat')
ARRIVAL_TYPES = ('code', 'conversation')
RATE_KINDS = ('calibrated', 'common')
CONTRASTS = ('combined_over_shortcut', 'combined_over_native', 'shortcut_over_native')
BOOTSTRAP_SAMPLES = 100_000
BOOTSTRAP_SEED = 471512
ASSIGNMENT_SEED = 471513


def _rank(*parts):
    return hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()


def _groups(groups, allowed_counts=(2, 6, 8)):
    by_dataset = defaultdict(list)
    seen = set()
    for group in groups:
        gid, dataset = group['id'], group['dataset']
        if not isinstance(gid, str) or not gid or gid in seen or dataset not in DATASETS:
            raise ValueError('unique content IDs and the three declared datasets required')
        seen.add(gid); by_dataset[dataset].append(gid)
    counts = {len(by_dataset[d]) for d in DATASETS}
    if len(counts) != 1 or next(iter(counts)) not in allowed_counts:
        raise ValueError('equal declared content-group counts per dataset required')
    return {d: sorted(by_dataset[d]) for d in DATASETS}


def select_content_groups(groups, n, *, seed=ASSIGNMENT_SEED):
    """Select six or eight groups without reading captures or outcomes."""
    by_dataset = _groups(groups, (8,))
    if n not in (6, 8):
        raise ValueError('final allocation is six or eight groups per dataset')
    selected = {gid for d in DATASETS for gid in sorted(
        by_dataset[d], key=lambda x: (_rank(seed, 'select', d, x), x))[:n]}
    return sorted((dict(g) for g in groups if g['id'] in selected), key=lambda g: (g['dataset'], g['id']))


def assign_calendars(groups, blocks, *, phase, seed=ASSIGNMENT_SEED):
    """Balanced dates, unique blocks, independent of measured/routing outcomes.

    Final: dataset d, hash-ranked group j maps to date (j+d*n+rotation)%5;
    rotations are zero for code and two for conversation. Within each date use
    source hash rank. Pilot: each dataset gets its preselected one block/day.
    """
    if phase not in ('pilot', 'final'):
        raise ValueError('pilot or final phase required')
    by_dataset = _groups(groups, (2,) if phase == 'pilot' else (6, 8))
    n = len(by_dataset[DATASETS[0]])
    pool = defaultdict(list); seen = set()
    for block in blocks:
        if block['split'] != phase:
            continue
        bid = block['id']
        if bid in seen or block['trace'] not in ARRIVAL_TYPES:
            raise ValueError('duplicate block or unexpected service')
        seen.add(bid)
        pool[block['trace'], block['source_date_utc']].append(block)
    dates = {t: sorted(date for typ,date in pool if typ == t) for t in ARRIVAL_TYPES}
    if any(len(ds) != (2 if phase == 'pilot' else 5) for ds in dates.values()):
        raise ValueError('need two pilot or five final dates per service')
    for key in pool:
        pool[key].sort(key=lambda b: (b['hash_rank_score'], b['id']))
    used = set(); result = []
    for di,dataset in enumerate(DATASETS):
        ordered = sorted(by_dataset[dataset], key=lambda gid: (_rank(seed, 'assignment', phase, dataset, gid), gid))
        for j,gid in enumerate(ordered):
            for typ in ARRIVAL_TYPES:
                rotation = 0 if typ == 'code' else 2
                date = dates[typ][(j+di*n+rotation) % len(dates[typ])]
                eligible = [b for b in pool[typ,date] if b['id'] not in used and
                    (phase != 'pilot' or b.get('pilot_dataset_assignment') == dataset)]
                if not eligible:
                    raise ValueError('insufficient distinct blocks for declared assignment')
                block = eligible[0]; used.add(block['id'])
                result.append(dict(content_group_id=gid,dataset=dataset,arrival_type=typ,
                    arrival_date=date,arrival_block_id=block['id'],source_rows_sha256=block['source_rows_sha256']))
    for typ in ARRIVAL_TYPES:
        counts = Counter(r['arrival_date'] for r in result if r['arrival_type'] == typ)
        if max(counts.values())-min(counts.values()) > 1:
            raise AssertionError('global calendar balance violated')
    return sorted(result,key=lambda r:(r['dataset'],r['content_group_id'],r['arrival_type']))


def _index_pairs(pairs, groups, allowed_counts=(2, 6, 8)):
    by_dataset = _groups(groups, allowed_counts)
    lookup = {gid:d for d,gids in by_dataset.items() for gid in gids}
    expected = set(itertools.product(lookup,ARRIVAL_TYPES,RATE_KINDS,CONTRASTS))
    index = {}; dates = {}
    for pair in pairs:
        key = tuple(pair[k] for k in ('content_group_id','arrival_type','rate_kind','contrast'))
        if key not in expected or key in index or pair['dataset'] != lookup[key[0]]:
            raise ValueError('duplicate, unexpected or mislabeled paired unit')
        date = pair['arrival_date']
        if not isinstance(date,str) or len(date) != 10:
            raise ValueError('ISO calendar date required')
        dkey = key[:2]
        if dkey in dates and dates[dkey] != date:
            raise ValueError('arrival date differs across paired arms or rates')
        dates[dkey] = date
        if any(not math.isfinite(pair['log_ratios'][m]) for m in ('response_mean_ns','response_p95_ns')):
            raise ValueError('finite log-response ratios required')
        index[key] = pair
    if set(index) != expected:
        raise ValueError('complete three-contrast, two-arrival, two-rate content bundles required')
    return by_dataset,index


def _bundle_logs(index, groups, *, contrast, rate, metric, exclude_date=None):
    result = {}
    for gid in groups:
        rows = [index[gid,t,rate,contrast] for t in ARRIVAL_TYPES]
        rows = [p for p in rows if p['arrival_date'] != exclude_date]
        if rows:
            result[gid] = statistics.mean(p['log_ratios'][metric] for p in rows)
    return result


def choose_final_sample_size(pairs, groups):
    """The v1 variance-only heuristic, not a power or CI-width guarantee."""
    by_dataset,index = _index_pairs(pairs,groups,(2,))
    logs = {gid:statistics.mean(index[gid,t,r,'combined_over_shortcut']['log_ratios']['response_mean_ns']
        for t,r in itertools.product(ARRIVAL_TYPES,RATE_KINDS)) for d in DATASETS for gid in by_dataset[d]}
    sd = statistics.stdev(logs.values())
    widths = {n:2.571*sd/math.sqrt(3*n) for n in (6,8)}
    n = 6 if widths[6] <= .03 else 8
    return dict(groups_per_dataset=n,final_treatment_trials=3*n*2*2*3,
        pilot_content_groups=6,pilot_bundle_log_ratios=logs,pooled_sample_sd=sd,
        dataset_sample_sd={d:statistics.stdev(logs[g] for g in by_dataset[d]) for d in DATASETS},
        projected_log_halfwidth={str(k):v for k,v in widths.items()},
        rule='six if 2.571*s/sqrt(18) <= 0.03; otherwise eight',
        uses_mean_benefit_for_selection=False,
        limitation='Six pilot bundles; heuristic only. Does not guarantee primary or seven-guard power, or quantify calendar-population variation.')


def infer_endpoints(pairs, groups, *, bootstrap_samples=BOOTSTRAP_SAMPLES, bootstrap_seed=BOOTSTRAP_SEED):
    """Final-only inference on a verified complete paired matrix.

    Optional sampling controls support CPU oracle tests. Production must pass the
    frozen defaults, which are returned in output for independent verification.
    """
    if not isinstance(bootstrap_samples,int) or bootstrap_samples < 1:
        raise ValueError('positive integer bootstrap sample count required')
    by_dataset,index = _index_pairs(pairs,groups,(6,8))
    rng = np.random.default_rng(bootstrap_seed)
    n = len(by_dataset[DATASETS[0]])
    # One shared draw table per stratum, in fixed dataset order; all endpoints
    # and contrasts use these same tables, preserving dependence.
    draws = {d:rng.integers(0,n,size=(bootstrap_samples,n)) for d in DATASETS}
    def endpoint(name,contrast,rate,metric,datasets=DATASETS,guard=False):
        stratum_points=[]; distributions=[]; logs={}
        for d in datasets:
            values = _bundle_logs(index,by_dataset[d],contrast=contrast,rate=rate,metric=metric)
            logs.update(values)
            arr = np.asarray([values[g] for g in by_dataset[d]],dtype=float)
            stratum_points.append(float(arr.mean()))
            distributions.append(arr[draws[d]].mean(axis=1))
        samples = np.exp(np.mean(distributions,axis=0))
        q = 1-.05/7 if guard else .975
        upper = float(np.quantile(samples,q,method='linear'))
        limit = 1.02 if guard else .95
        result = dict(name=name,contrast=contrast,rate_kind=rate,metric=metric,datasets=list(datasets),
            ratio=math.exp(statistics.mean(stratum_points)),content_group_log_ratios=logs,
            upper_confidence_bound=upper,ratio_limit=limit,passes=upper<=limit if guard else upper<limit,
            criterion='upper <= limit' if guard else 'upper < limit',
            interval_scope='conditional on selected external trace material; not arrival-population uncertainty')
        if guard:
            result.update(one_sided_confidence_level=q,guard_family_size=7)
        else:
            result.update(ci95=[float(np.quantile(samples,.025,method='linear')),upper])
        result['status']='pass' if result['passes'] else 'not_established'
        return result
    primary = endpoint('calibrated_mean_response','combined_over_shortcut','calibrated','response_mean_ns')
    guards = [endpoint(f'{d}_{r}_mean_response','combined_over_shortcut',r,'response_mean_ns',(d,),True)
        for d,r in itertools.product(DATASETS,RATE_KINDS)]
    guards.append(endpoint('calibrated_p95_response','combined_over_shortcut','calibrated','response_p95_ns',guard=True))
    dates = sorted({p['arrival_date'] for p in index.values()})
    lodo=[]
    for date in dates:
        stratum_points=[]; retained={}; absent=[]
        for d in DATASETS:
            logs=_bundle_logs(index,by_dataset[d],contrast='combined_over_shortcut',rate='calibrated',
                metric='response_mean_ns',exclude_date=date)
            retained[d]=len(logs)
            if not logs:
                absent.append(d)
            else:
                stratum_points.append(statistics.mean(logs.values()))
        ratio=math.exp(statistics.mean(stratum_points)) if not absent else None
        lodo.append(dict(excluded_date=date,ratio=ratio,retained_content_groups=retained,
            removed_arrival_bundles=sum(index[g,t,'calibrated','combined_over_shortcut']['arrival_date']==date
                for d in DATASETS for g in by_dataset[d] for t in ARRIVAL_TYPES),
            passes=ratio is not None and ratio<=1.02,missing_datasets=absent))
    robustness_pass=all(x['passes'] for x in lodo)
    valid_lodo=[x['ratio'] for x in lodo if x['ratio'] is not None]
    robustness=dict(method='leave one actual calendar date out, linking services on shared dates',
        ratio_limit=1.02,criterion='every available point ratio <= 1.02; no CI or population claim',
        date_count=len(dates),dates=dates,results=lodo,passes=robustness_pass,
        ratio_range=[min(valid_lodo),max(valid_lodo)] if valid_lodo else None,
        weighting='Average remaining arrival types within surviving group, groups within dataset, then three datasets equally; absent dataset fails.')
    gate=primary['passes'] and all(x['passes'] for x in guards) and robustness_pass
    if gate:
        secondary=endpoint('calibrated_mean_response_baseline','combined_over_native','calibrated','response_mean_ns')
        secondary.update(confirmatory=True,gate_passed=True)
    else:
        means=[]; logs={}
        for d in DATASETS:
            part=_bundle_logs(index,by_dataset[d],contrast='combined_over_native',rate='calibrated',metric='response_mean_ns')
            logs.update(part);means.append(statistics.mean(part.values()))
        secondary=dict(name='calibrated_mean_response_baseline',contrast='combined_over_native',
            ratio=math.exp(statistics.mean(means)),content_group_log_ratios=logs,ci95=None,
            upper_confidence_bound=None,ratio_limit=.95,passes=None,status='not_tested_gate_closed',confirmatory=False,gate_passed=False)
    descriptive=[]
    for d,t,r,c in itertools.product(DATASETS,ARRIVAL_TYPES,RATE_KINDS,CONTRASTS):
        descriptive.append(dict(dataset=d,arrival_type=t,rate_kind=r,contrast=c,
            ratios={metric:math.exp(statistics.mean(index[g,t,r,c]['log_ratios'][metric] for g in by_dataset[d]))
                for metric in ('response_mean_ns','response_p95_ns')},content_groups=n))
    return dict(schema='generalization_inference_v2',primary=primary,guards=guards,
        calendar_date_robustness=robustness,secondary_baseline=secondary,descriptive_cells=descriptive,
        incremental_acceptance='pass' if gate else 'not_established',
        acceptance='pass' if gate and secondary['passes'] else 'not_established',
        content_groups=3*n,paired_comparisons=len(index),treatment_trials=3*n*2*2*3,
        bootstrap=dict(samples=bootstrap_samples,seed=bootstrap_seed,rng='numpy.random.default_rng/PCG64',
            strata=list(DATASETS),groups_per_stratum=n,unit='entire content bundle with two types and two rates',
            quantile_method='linear',scope='content-bundle resampling conditional on selected external trace material; does not estimate production-arrival/calendar-day or hardware-population uncertainty'))

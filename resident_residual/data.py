"""Fresh document/arrival selection for the fixed-residency fast-path study.

No GPU or performance observations participate. Earlier selector primitives are
imported unchanged; this module only declares the new split and source linkage.
"""
from collections import defaultdict
from stock_comparison.data import digest, normalized, exclude_and_deduplicate, scan_fresh_arrivals

CONTENT_SEED=471801
ARRIVAL_SEED=471802
DATASETS=('gsm8k_train','wildchat')
ALLOCATION={d:{'pilot':2,'final':8} for d in DATASETS}
DOCUMENTS_PER_GROUP=32
STOCK_SELECTION='6020e89fc0b814460c6e2b4c2e9423a6956f390ca850cfe96d45cfe5687ef20f'


def split_sources(records,*,seed=CONTENT_SEED):
    blocks=[]
    for dataset in DATASETS:
        ordered=sorted((r for r in records if r['dataset']==dataset),key=lambda r:(digest([seed,dataset,r['source_id'],r['text_sha256']]),r['source_id']))
        need=sum(ALLOCATION[dataset].values())*DOCUMENTS_PER_GROUP
        if len(ordered)<need:
            raise ValueError(f'{dataset}: only{len(ordered)} fresh documents, need{need}; no reuse or silent split reduction')
        offset=0
        for split,count in ALLOCATION[dataset].items():
            for i in range(count):
                chosen=ordered[offset:offset+DOCUMENTS_PER_GROUP];offset+=DOCUMENTS_PER_GROUP
                blocks.append(dict(block_id=f'residual-{dataset}-{split}-{i+1:02d}',dataset=dataset,split=split,
                    inference_role='main',sources=chosen))
    keys=[(r['dataset'],r['source_id']) for b in blocks for r in b['sources']]
    hashes=[r['normalized_text_sha256'] for b in blocks for r in b['sources']]
    if len(keys)!=len(set(keys)) or len(hashes)!=len(set(hashes)):raise ValueError('Source repeated across fresh groups')
    return blocks


def combine_prior_arrivals(manifests):
    """Keep every earlier selected boundary, including unused held-out blocks."""
    if len(manifests)!=3 or [len(x['blocks']) for x in manifests]!=[92,48,44]:
        raise ValueError('Require all92 original,48 stock,and44 resident-fix arrival blocks')
    rows=[b for m in manifests for b in m['blocks']]
    for trace in ('code','conversation'):
        intervals=sorted((b['row_start'],b['boundary_row']+1) for b in rows if b['trace']==trace)
        if len(intervals)!=92 or any(hi>next_lo for (_,hi),(next_lo,_) in zip(intervals,intervals[1:])):
            raise ValueError('Prior selected intervals overlap or have missing rows')
        if len({b['source_sha256'] for b in rows if b['trace']==trace})!=1:raise ValueError('Prior source-week pins disagree')
    return rows


def assign_arrivals(scans,groups,*,seed=ARRIVAL_SEED):
    result=[]
    for scan in sorted(scans,key=lambda x:x['trace']):
        dates=sorted(scan['retained_candidates']);used=defaultdict(int)
        if len(dates)!=7:raise ValueError('Exactly seven source dates required')
        for split,days in (('pilot',dates[:2]),('final',dates[2:])):
            ordinal=0
            for dataset in DATASETS:
                selected=sorted((g for g in groups if g['dataset']==dataset and g['split']==split),key=lambda g:(digest([seed,'assignment',scan['trace'],g['block_id']]),g['block_id']))
                if len(selected)!=ALLOCATION[dataset][split]:raise ValueError('Declared group allocation differs')
                for group in selected:
                    rotation=0 if scan['trace']=='code' else 2
                    day=days[(ordinal+rotation)%len(days)];ordinal+=1
                    position=used[day];candidates=scan['retained_candidates'][day]
                    if position>=len(candidates):raise ValueError('Insufficient prespecified daily candidates')
                    chosen=candidates[position];used[day]+=1
                    if chosen['jobs']!=4096 or chosen['boundary_row']!=chosen['row_start']+4096:
                        raise ValueError('Need4096 contiguous jobs plus boundary')
                    if chosen['original_horizon_ns']<=0 or chosen['terminal_gap_ns']<=0:
                        raise ValueError('Hash-selected invalid horizon; no silent replacement')
                    result.append({**chosen,'id':f"residual-azure-{scan['trace']}-{day}-{position+1:02d}",
                        'content_group_id':group['block_id'],'dataset':dataset,'split':split,
                        'inference_role':group['inference_role'],'source_sha256':scan['source_sha256'],'calendar_date_cluster':day})
    if len(result)!=40:raise ValueError('Expected40 source intervals')
    for trace in ('code','conversation'):
        intervals=sorted((b['row_start'],b['boundary_row']+1) for b in result if b['trace']==trace)
        if any(hi>lo for (_,hi),(lo,_) in zip(intervals,intervals[1:])):raise ValueError('Fresh intervals overlap')
    return result

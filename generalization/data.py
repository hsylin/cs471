"""Pure provenance/splitting and absolute-arrival construction contracts."""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import re
import unicodedata


def digest(data):
    if not isinstance(data, bytes):
        data = json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def shingles(text, width=5):
    words = re.findall(r'\w+|[^\w\s]', normalized(text))
    return {tuple(words[i:i+width]) for i in range(max(1, len(words)-width+1))}


def deduplicate(records, threshold=0.8):
    """Global before splitting: IDs, normalized exact matches, then 5-gram Jaccard.

    Deterministic input order fixes which representative survives. No timing,
    routing, or model-output information enters this filter.
    """
    if not 0 < threshold <= 1:
        raise ValueError('Invalid duplicate threshold')
    kept, dropped, ids, hashes = [], [], set(), {}
    index, sets = defaultdict(set), []
    for record in records:
        key = (record['dataset'], record['source_id'])
        text = normalized(record['text'])
        h = digest(text.encode())
        reason, duplicate = None, None
        if not text:
            reason = 'empty'
        elif key in ids:
            reason = 'same_document_or_conversation'
        elif h in hashes:
            reason, duplicate = 'normalized_exact', hashes[h]
        else:
            s = shingles(text)
            candidates = set().union(*(index[g] for g in s))
            for k in sorted(candidates):
                other = sets[k]
                score = len(s & other) / len(s | other)
                if score >= threshold:
                    reason, duplicate = 'near_duplicate', kept[k]['source_id']
                    break
        ids.add(key)
        if reason:
            dropped.append({'dataset':record['dataset'], 'source_id':record['source_id'],
                            'reason':reason, 'duplicate_of':duplicate})
            continue
        i = len(kept)
        kept.append({**record, 'text_sha256':digest(record['text'].encode()),
                     'normalized_text_sha256':h})
        hashes[h] = record['source_id']; sets.append(s)
        for g in s:index[g].add(i)
    return kept, dropped


def deduplicate_captured_prefixes(records, threshold=0.8):
    """Filter the actual model-visible view as well as full source documents."""
    views=[{**r,'text':r['captured_prefix_text']} for r in records]
    kept,dropped=deduplicate(views,threshold=threshold)
    keys={(r['dataset'],r['source_id']) for r in kept}
    result=[]
    for r in records:
        if (r['dataset'],r['source_id']) in keys:
            result.append({k:v for k,v in r.items() if k!='captured_prefix_text'})
            result[-1]['captured_prefix_normalized_sha256']=digest(normalized(r['captured_prefix_text']).encode())
    return result,dropped


def split_records(records, *, seed=471501, pilot_blocks=2, final_blocks=8,
                  pilot_documents=16, final_documents=32):
    """Hash-ranked document sampling; all splits chosen before capture/timing."""
    datasets = sorted({r['dataset'] for r in records})
    blocks=[]
    for dataset in datasets:
        eligible=sorted((r for r in records if r['dataset']==dataset),
                        key=lambda r:digest([seed,dataset,r['source_id'],r['text_sha256']]))
        needed=pilot_blocks*pilot_documents+final_blocks*final_documents
        if len(eligible)<needed:
            raise ValueError(f'{dataset}: {len(eligible)} eligible documents, need {needed}; no silent reuse')
        cursor=0
        for split,nblocks,ndocs in [('pilot',pilot_blocks,pilot_documents),('final',final_blocks,final_documents)]:
            for i in range(nblocks):
                block_id=f'{dataset}-{split}-{i+1:02d}'
                sources=eligible[cursor:cursor+ndocs];cursor+=ndocs
                blocks.append({'block_id':block_id,'dataset':dataset,'split':split,'sources':sources})
    validate_splits(blocks)
    return blocks


def validate_splits(blocks):
    seen=set(); hashes=set()
    for block in blocks:
        for r in block['sources']:
            key=(r['dataset'],r['source_id'])
            if key in seen or r['normalized_text_sha256'] in hashes:
                raise ValueError('Document/normalized content reused across blocks')
            if digest(r['text'].encode())!=r['text_sha256']:
                raise ValueError('Text changed after selection')
            seen.add(key);hashes.add(r['normalized_text_sha256'])
    return {'blocks':len(blocks),'documents':len(seen),'disjoint':True}


def timestamp_ns(value):
    # Parse exactly, never through floating-point epoch seconds.
    m=re.fullmatch(r'(\d{4}-\d\d-\d\d)[ T](\d\d:\d\d:\d\d)(?:\.(\d{1,9}))?(Z|[+-]\d\d:\d\d)?',value)
    if not m:raise ValueError(f'Unsupported timestamp {value!r}')
    sec=datetime.fromisoformat(m[1]+'T'+m[2]+(m[4] or '+00:00').replace('Z','+00:00'))
    epoch=datetime(1970,1,1,tzinfo=timezone.utc)
    delta=sec-epoch
    return (delta.days*86400+delta.seconds)*10**9+int((m[3] or '').ljust(9,'0'))


def select_arrival_blocks(rows, *, trace, seed=471502, count=32, jobs=256):
    """Nonoverlapping contiguous row blocks, stable time order; ties retained.

    Candidates are disjoint fixed-size bins. No burstiness/load-based selection.
    Each block keeps its following timestamp as the offered-horizon boundary.
    """
    times=[timestamp_ns(r['TIMESTAMP']) for r in rows]
    if any(b<a for a,b in zip(times,times[1:])):
        raise ValueError('Trace must be monotonically ordered; never silently sort')
    starts=list(range(0,len(rows)-jobs,jobs+1))
    starts.sort(key=lambda start:digest([seed,trace,start]))
    if len(starts)<count:raise ValueError('Insufficient nonoverlapping external blocks')
    result=[]
    for i,start in enumerate(starts[:count]):
        zero=times[start];offsets=[t-zero for t in times[start:start+jobs]]
        horizon=times[start+jobs]-zero
        if horizon<=0:raise ValueError('Zero-span candidate; revise source protocol explicitly')
        result.append({'id':f'azure-{trace}-{i+1:02d}','split':'pilot' if i<8 else 'final',
                       'trace':trace,'row_start':start,'row_stop_exclusive':start+jobs,
                       'boundary_row':start+jobs,'source_origin_ns':zero,
                       'original_offsets_ns':offsets,'original_horizon_ns':horizon,
                       'context_tokens':[int(r['ContextTokens']) for r in rows[start:start+jobs]],
                       'generated_tokens':[int(r['GeneratedTokens']) for r in rows[start:start+jobs]]})
    return result


def scaled_calendar(block, *, numerator=1, denominator=1):
    scale=Fraction(numerator,denominator)
    if scale<=0:raise ValueError('Positive time scale required')
    offsets=[round(x*scale) for x in block['original_offsets_ns']]
    horizon=round(block['original_horizon_ns']*scale)
    if horizon<=0 or any(x<0 or x>=horizon for x in offsets):
        raise ValueError('Scaling collapsed horizon or terminal gap; no silent clipping')
    return {'source_block':block['id'],'arrival_offsets_ns':offsets,'horizon_ns':horizon,
            'scale_numerator':scale.numerator,'scale_denominator':scale.denominator,
            'mapping':'one original invocation timestamp to one fixed-token layer job; sizes not executed',
            'absolute_arrivals':True,'drop_policy':'none; drain every admitted job'}

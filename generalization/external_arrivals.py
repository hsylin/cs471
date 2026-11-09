"""Streaming, timestamp-exact selection of external arrival blocks.

This module selects source rows without consulting prompt content, routing,
latency, burstiness, or any treatment outcome. Each candidate consumes 256 job
rows plus one following boundary row, so even boundary rows are not reused.
"""
from __future__ import annotations
import csv
from datetime import datetime, timezone
import hashlib
import heapq
import io
import json
from pathlib import Path
import re
from collections import Counter

from generalization.data import digest

DATASETS = ('gsm8k', 'mbpp', 'wildchat')
STAMP = re.compile(r'^(\d{4}-\d{2}-\d{2})[ T](\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,9}))?(Z|\+00:00)$')


class _HashingReader(io.RawIOBase):
    def __init__(self, path):
        self.file = open(path, 'rb'); self.sha = hashlib.sha256(); self.bytes_read = 0
    def readable(self): return True
    def readinto(self, b):
        n = self.file.readinto(b)
        if n:
            self.sha.update(memoryview(b)[:n]); self.bytes_read += n
        return n
    def close(self):
        self.file.close(); super().close()


def parse_timestamp(value, day_cache=None):
    """UTC source timestamps only; return exact ns, date, fractional precision."""
    m = STAMP.fullmatch(value)
    if not m: raise ValueError(f'Unsupported source timestamp {value!r}; expected explicit UTC')
    day, hour, minute, second, fraction, _ = m.groups()
    hour, minute, second = int(hour), int(minute), int(second)
    if hour >= 24 or minute >= 60 or second >= 60: raise ValueError('Invalid time of day')
    cache = day_cache if day_cache is not None else {}
    if day not in cache:
        dt = datetime.strptime(day, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        delta = dt - datetime(1970, 1, 1, tzinfo=timezone.utc)
        cache[day] = delta.days * 86_400 * 10**9
    fraction = fraction or ''
    ns = cache[day] + (hour*3600 + minute*60 + second)*10**9 + int(fraction.ljust(9, '0'))
    return ns, day, len(fraction)


def candidate_score(seed, trace, day, row_start):
    return digest([seed, trace, day, row_start])


def _candidate(trace, day, chunk, seed, jobs):
    start = chunk[0]['row_index']; zero = chunk[0]['timestamp_ns']
    score = candidate_score(seed, trace, day, start)
    rows, boundary = chunk[:jobs], chunk[jobs]
    offsets = [r['timestamp_ns']-zero for r in rows]
    out = {
        'trace':trace, 'source_date_utc':day, 'row_start':start,
        'row_stop_exclusive':start+jobs, 'boundary_row':start+jobs,
        'hash_rank_score':score, 'jobs':jobs, 'source_origin_ns':zero,
        'original_offsets_ns':offsets,
        'original_horizon_ns':boundary['timestamp_ns']-zero,
        'raw_timestamps':[r['raw_timestamp'] for r in rows],
        'boundary_timestamp':boundary['raw_timestamp'],
        'boundary_timestamp_ns':boundary['timestamp_ns'],
        'context_tokens':[r['context_tokens'] for r in rows],
        'generated_tokens':[r['generated_tokens'] for r in rows],
        'boundary_context_tokens':boundary['context_tokens'],
        'boundary_generated_tokens':boundary['generated_tokens'],
        'within_block_timestamp_ties':sum(a == b for a,b in zip(offsets, offsets[1:])),
        'terminal_gap_ns':boundary['timestamp_ns']-rows[-1]['timestamp_ns'],
    }
    out['source_rows_sha256'] = digest([[r['raw_timestamp'],r['context_tokens'],r['generated_tokens']] for r in chunk])
    return out


def scan_trace(path, *, trace, expected_sha256, seed=471502, jobs=256,
               keep_per_day=8, inventory_stream=None):
    """Read a source once, retaining only O(days*keep_per_day*jobs) records.

    The optional text stream receives all candidate identities/scores, not just
    winners. Inversions are counted and retained in the audit; selection rejects
    them. The input is never sorted or repaired.
    """
    if jobs < 1 or keep_per_day < 1: raise ValueError('Positive jobs and keep count required')
    path = Path(path)
    if not re.fullmatch('[0-9a-f]{64}', expected_sha256): raise ValueError('Pinned source SHA256 required')
    raw = _HashingReader(path)
    days = {}; heaps = {}; cache = {}; precisions = Counter()
    previous = None; active_day = None; chunk = []; rows_count = 0
    inversions = []; inversion_count = 0; ties = 0; day_reentries = 0
    with io.TextIOWrapper(io.BufferedReader(raw, buffer_size=1024*1024), encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['TIMESTAMP','ContextTokens','GeneratedTokens']:
            raise ValueError(f'Unexpected CSV schema: {reader.fieldnames!r}')
        for index, row in enumerate(reader):
            if None in row or any(v is None for v in row.values()): raise ValueError(f'Malformed row {index}')
            ns, day, precision = parse_timestamp(row['TIMESTAMP'], cache)
            context, generated = int(row['ContextTokens']), int(row['GeneratedTokens'])
            if context < 0 or generated < 0: raise ValueError(f'Negative token metadata at row {index}')
            precisions[precision] += 1
            if previous is not None:
                if ns < previous:
                    inversion_count += 1
                    if len(inversions) < 20: inversions.append({'row_index':index,'previous_ns':previous,'timestamp_ns':ns})
                elif ns == previous: ties += 1
            previous = ns; rows_count += 1
            if active_day != day:
                if active_day is not None: days[active_day]['discarded_partial_bin_rows'] += len(chunk)
                if day in days: day_reentries += 1
                active_day = day; chunk = []
            if day not in days:
                days[day] = {'date_utc':day,'row_start':index,'row_stop_exclusive':index,
                    'rows':0,'first_timestamp':row['TIMESTAMP'],'last_timestamp':row['TIMESTAMP'],
                    'first_timestamp_ns':ns,'last_timestamp_ns':ns,'candidate_bins':0,
                    'discarded_partial_bin_rows':0,'fractional_precision_counts':Counter()}
                heaps[day] = []
            stat = days[day]; stat['rows'] += 1; stat['row_stop_exclusive'] = index+1
            stat['last_timestamp'] = row['TIMESTAMP']; stat['last_timestamp_ns'] = ns
            stat['fractional_precision_counts'][precision] += 1
            chunk.append({'row_index':index,'raw_timestamp':row['TIMESTAMP'],'timestamp_ns':ns,
                          'context_tokens':context,'generated_tokens':generated})
            if len(chunk) == jobs+1:
                block = _candidate(trace, day, chunk, seed, jobs); stat['candidate_bins'] += 1
                if inventory_stream is not None:
                    inv = {k:block[k] for k in ('trace','source_date_utc','row_start','row_stop_exclusive',
                        'boundary_row','hash_rank_score','source_origin_ns','original_horizon_ns',
                        'terminal_gap_ns','within_block_timestamp_ties','source_rows_sha256')}
                    inventory_stream.write(json.dumps(inv,sort_keys=True,separators=(',',':'))+'\n')
                entry = (-int(block['hash_rank_score'],16),-block['row_start'],block)
                heap = heaps[day]
                if len(heap) < keep_per_day: heapq.heappush(heap,entry)
                elif entry > heap[0]: heapq.heapreplace(heap,entry)
                chunk = []
        if active_day is not None: days[active_day]['discarded_partial_bin_rows'] += len(chunk)
    actual_sha = raw.sha.hexdigest()
    if actual_sha != expected_sha256: raise ValueError(f'{trace}: source SHA256 differs: {actual_sha}')
    if rows_count == 0: raise ValueError('Empty arrival trace')
    for day,stat in days.items():
        stat['fractional_precision_counts'] = {str(k):v for k,v in sorted(stat['fractional_precision_counts'].items())}
        stat['retained_candidates'] = sorted((e[2] for e in heaps[day]),key=lambda b:(b['hash_rank_score'],b['row_start']))
    return {'schema':'external_arrival_scan_v1','trace':trace,'source_file':path.name,
            'source_path':str(path.resolve()),'source_sha256':actual_sha,'source_bytes':raw.bytes_read,
            'rows':rows_count,'seed':seed,'jobs':jobs,'keep_per_day':keep_per_day,
            'fractional_precision_counts':{str(k):v for k,v in sorted(precisions.items())},
            'timestamp_ties':ties,'timestamp_inversions':inversion_count,'first_inversions':inversions,
            'date_reentries':day_reentries,'days':[days[d] for d in sorted(days)],
            'selection_uses_timing_outcomes':False,'source_order_preserved':True,
            'binning':'reset at each UTC date; disjoint jobs+1 row bins; last row supplies horizon boundary; partial bins discarded'}


def select_service_days(scans, *, pilot_days=2, final_days=5, pilot_blocks_per_day=3,
                        final_blocks_per_day=8):
    """Service-specific temporal holdout; shared calendar dates remain clusters."""
    if len(scans) != 2 or {x['trace'] for x in scans} != {'code','conversation'}:
        raise ValueError('Exactly code and conversation scans are required')
    if min(pilot_days,final_days,pilot_blocks_per_day,final_blocks_per_day) < 1:
        raise ValueError('Positive selection counts required')
    result=[]; service_dates={}; sets={}
    for scan in sorted(scans,key=lambda x:x['trace']):
        if scan['timestamp_inversions'] or scan['date_reentries']:
            raise ValueError(f"{scan['trace']}: nonmonotone source; no sorting permitted")
        dates=[d['date_utc'] for d in scan['days']]
        required=pilot_days+final_days
        if len(dates)<required: raise ValueError(f"{scan['trace']}: {len(dates)} dates, need {required}")
        selected=scan['days'][:required]
        pilot=dates[:pilot_days]; final=dates[pilot_days:required]
        service_dates[scan['trace']]={'all_dates':dates,'pilot_dates':pilot,'final_dates':final,'unused_dates':dates[required:]}
        sets[scan['trace']]=set(dates)
        for day_index,stat in enumerate(selected):
            split='pilot' if day_index<pilot_days else 'final'
            count=pilot_blocks_per_day if split=='pilot' else final_blocks_per_day
            available=stat['retained_candidates']
            if len(available)<count: raise ValueError(f"{scan['trace']}/{stat['date_utc']}: need {count} candidate bins")
            for rank,block in enumerate(available[:count]):
                if block['original_horizon_ns']<=0 or block['terminal_gap_ns']<=0:
                    raise ValueError('Hash-selected block has zero span/terminal tie; requires protocol amendment, never resample')
                if any(x<0 or x>=block['original_horizon_ns'] for x in block['original_offsets_ns']):
                    raise ValueError('Invalid selected offsets')
                result.append({**block,'id':f"azure2024-{scan['trace']}-{stat['date_utc']}-{rank+1:02d}",
                    'split':split,'day_hash_rank':rank+1,'calendar_date_cluster':stat['date_utc'],
                    'source_sha256':scan['source_sha256'],
                    'pilot_dataset_assignment':DATASETS[rank] if split=='pilot' and count==3 else None})
    code=service_dates['code']; conv=service_dates['conversation']
    final_union=sorted(set(code['final_dates'])|set(conv['final_dates']))
    overlap=sorted((set(code['pilot_dates'])|set(conv['pilot_dates'])) & set(final_union))
    payload={'schema':'external_arrivals_v1','protocol':'generalization protocol-v2, service-specific date holdout',
        'selection':{'pilot_days_per_service':pilot_days,'final_days_per_service':final_days,
                     'pilot_blocks_per_day':pilot_blocks_per_day,'final_blocks_per_day':final_blocks_per_day,
                     'seed':scans[0]['seed'],'uses_burstiness_filter':False,'uses_performance_outcomes':False},
        'services':service_dates,'common_available_dates':sorted(sets['code']&sets['conversation']),
        'union_available_dates':sorted(sets['code']|sets['conversation']),
        'final_calendar_date_clusters':final_union,'cross_service_pilot_final_calendar_overlap':overlap,
        'global_calendar_future_holdout':not bool(overlap),'within_service_temporal_holdout':True,
        'mapping':'Each original invocation timestamp becomes one fixed-token layer job; original request sizes remain metadata, not executed.',
        'blocks':result}
    if len({s['seed'] for s in scans}) != 1 or len({s['jobs'] for s in scans}) != 1:
        raise ValueError('Sources use inconsistent seed or job count')
    payload['content_sha256']=digest(payload)
    return payload

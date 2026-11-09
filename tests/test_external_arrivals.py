import csv
import hashlib
import io
import json
from pathlib import Path
import pytest
from generalization.data import digest, timestamp_ns
from generalization.external_arrivals import parse_timestamp, scan_trace, select_service_days


def csv_file(tmp_path, name, first_day=10, days=7, per_day=27, mutate=None):
    rows=[]
    for d in range(first_day,first_day+days):
        for n in range(per_day):
            # Alternating ties are real arrivals, never merged.
            rows.append([f'2024-05-{d:02d} 00:00:{n//2:02d}.123456+00:00',n,n+1])
    if mutate:mutate(rows)
    p=tmp_path/name
    with p.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['TIMESTAMP','ContextTokens','GeneratedTokens']);w.writerows(rows)
    return p,hashlib.sha256(p.read_bytes()).hexdigest()


def scan(tmp_path,trace,first_day=10,**kwargs):
    p,h=csv_file(tmp_path,trace+'.csv',first_day=first_day,**kwargs)
    sink=io.StringIO()
    r=scan_trace(p,trace=trace,expected_sha256=h,jobs=2,keep_per_day=8,inventory_stream=sink)
    return r,[json.loads(x) for x in sink.getvalue().splitlines()]


@pytest.mark.parametrize('stamp',['2024-05-10 12:13:14.123456789+00:00','2024-05-10T00:00:00Z','2024-05-10 23:59:59.000001+00:00'])
def test_exact_timestamp(stamp):
    assert parse_timestamp(stamp)[0]==timestamp_ns(stamp)


@pytest.mark.parametrize('stamp',['2024-05-10 24:00:00Z','2024-05-10 00:00:00','2024-05-10 00:00:00+01:00','2024-02-30 00:00:00Z'])
def test_timestamp_fail_closed(stamp):
    with pytest.raises(ValueError):parse_timestamp(stamp)


def test_all_candidates_and_hash_rank_are_independent_of_arrival_shape(tmp_path):
    r,all_candidates=scan(tmp_path,'code',days=1,per_day=28)
    day=r['days'][0]
    assert r['rows']==28 and r['timestamp_ties']==14
    assert day['candidate_bins']==9 and day['discarded_partial_bin_rows']==1
    assert len(all_candidates)==9
    expected=sorted(all_candidates,key=lambda b:b['hash_rank_score'])[:8]
    assert [x['row_start'] for x in day['retained_candidates']]==[x['row_start'] for x in expected]
    assert all(c['boundary_row']==c['row_start']+2 for c in all_candidates)
    assert all_candidates[1]['row_start']==all_candidates[0]['boundary_row']+1
    for c in day['retained_candidates']:
        assert c['original_offsets_ns']==[timestamp_ns(t)-c['source_origin_ns'] for t in c['raw_timestamps']]
        assert c['context_tokens']==list(range(c['row_start'],c['row_stop_exclusive']))


def test_sha_and_malformed_metadata_rejected(tmp_path):
    p,h=csv_file(tmp_path,'one.csv',days=1)
    with pytest.raises(ValueError,match='SHA256 differs'):
        scan_trace(p,trace='code',expected_sha256='0'*64,jobs=2)
    p,h=csv_file(tmp_path,'bad.csv',days=1,mutate=lambda rows:rows[0].__setitem__(1,-1))
    with pytest.raises(ValueError,match='Negative token'):
        scan_trace(p,trace='code',expected_sha256=h,jobs=2)


def test_inversion_counted_and_selection_never_sorts(tmp_path):
    code,_=scan(tmp_path,'code',mutate=lambda rows:rows.__setitem__(slice(0,3),rows[2::-1]))
    conv,_=scan(tmp_path,'conversation',first_day=12)
    assert code['timestamp_inversions']==1
    with pytest.raises(ValueError,match='nonmonotone'):select_service_days([code,conv])


def test_service_temporal_split_does_not_claim_global_holdout(tmp_path):
    # Odd row count plus alternate ties can leave a hash-selected terminal tie;
    # remove ties here to test split design independently of terminal validation.
    def unique(rows):
        for i,r in enumerate(rows):r[0]=r[0][:17]+f'{i%27:02d}'+r[0][19:]
    code,_=scan(tmp_path,'code',mutate=unique)
    conv,_=scan(tmp_path,'conversation',first_day=12,mutate=unique)
    result=select_service_days([code,conv])
    assert len(result['blocks'])==92
    assert sum(b['split']=='pilot' for b in result['blocks'])==12
    assert result['common_available_dates']==[f'2024-05-{d:02d}' for d in range(12,17)]
    assert result['cross_service_pilot_final_calendar_overlap']==['2024-05-12','2024-05-13']
    assert result['final_calendar_date_clusters']==[f'2024-05-{d:02d}' for d in range(12,19)]
    assert result['within_service_temporal_holdout'] and not result['global_calendar_future_holdout']
    assert result['content_sha256']==digest({k:v for k,v in result.items() if k!='content_sha256'})
    for service in result['services'].values():assert max(service['pilot_dates'])<min(service['final_dates'])
    intervals={}
    for b in result['blocks']:
        used=set(range(b['row_start'],b['boundary_row']+1));key=b['trace']
        assert not used & intervals.setdefault(key,set());intervals[key]|=used


def test_too_few_days_is_explicit_failure(tmp_path):
    code,_=scan(tmp_path,'code',days=6);conv,_=scan(tmp_path,'conversation',first_day=12)
    with pytest.raises(ValueError,match='6 dates, need 7'):select_service_days([code,conv])


def test_selected_terminal_tie_is_not_replaced(tmp_path):
    code,_=scan(tmp_path,'code');conv,_=scan(tmp_path,'conversation',first_day=12)
    # All source candidates still appear in the inventory; selection must fail
    # if one of the fixed hash winners has a zero terminal gap.
    with pytest.raises(ValueError,match='terminal tie'):select_service_days([code,conv])

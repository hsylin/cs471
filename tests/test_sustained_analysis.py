import json
from pathlib import Path
import pytest
from full_serving.sustained_analysis import raw_phase


def test_raw_arithmetic_on_previous_verified_matched_episode(tmp_path):
    root=Path(__file__).resolve().parents[1]
    run=root/'results/full-serving/cost-isolation-v1/packed-run-v1'
    path=run/'numinamath/off/measured.jsonl'
    if not path.exists():pytest.skip('archived raw record not present')
    rows=[json.loads(x) for x in path.read_text().splitlines()]
    h=next(r for r in rows if r['type']=='run_header')
    rr=sorted([r for r in rows if r['type']=='request'],key=lambda r:r['request_index'])
    t=next(r for r in rows if r['type']=='run_summary')
    item={'request_count':16,'request_ids':[r['request_id'] for r in rr],'requests_sha256':h['requests_sha256'],'calendar_sha256':h['calendar_sha256']}
    calendar={'arrival_offsets_ns':[r['arrival_offset_ns'] for r in rr],'horizon_ns':t['offered_horizon_ns']}
    targets={r['request_id']:r['usage']['completion_tokens'] for r in rr}
    value,_=raw_phase(path,item,cap=h['max_tokens'],calendar=calendar,targets=targets)
    assert value['summary']['valid'] and value['independent_arithmetic']=='passed'
    rr[0]['response_from_schedule_ns']+=1
    damaged=tmp_path/'damaged.jsonl';damaged.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    with pytest.raises(AssertionError):raw_phase(damaged,item,cap=h['max_tokens'],calendar=calendar,targets=targets)

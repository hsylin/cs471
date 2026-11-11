import json
from pathlib import Path
import pytest
from full_serving.sustained_experiment import ARMS, server_arguments, clean, copy_inputs
from full_serving.data import digest


def test_six_arms_hold_native_transport_and_replica_contract():
    assert len(ARMS)==6
    for arm in ARMS:
        args=server_arguments(arm,'snapshot','127.0.0.1','primary')
        assert args[args.index('--all2all-backend')+1]=='allgather_reducescatter'
        assert args[args.index('--data-parallel-size')+1]=='8'
        if arm=='off':assert '--enable-eplb' not in args
        else:assert json.loads(args[args.index('--eplb-config')+1])['num_redundant_experts']==(0 if arm=='b1' else 32)


def test_pilot_copy_never_opens_final_payload(tmp_path):
    source=tmp_path/'source';source.mkdir()
    row={'id':'p','messages':[{'role':'user','content':'pilot'}],'input_tokens':30}
    raw=(json.dumps(row)+'\n').encode();(source/'pilot.jsonl').write_bytes(raw)
    import hashlib
    data={'datasets':{'longbench':{'requests':[{'id':'p','messages_sha256':digest(row['messages'])}],
        'items':{'warmup':{'requests_file':'pilot.jsonl','requests_sha256':hashlib.sha256(raw).hexdigest(),'request_ids':['p']},
                 'measured':{'requests_file':'does-not-exist'}}}}}
    result=copy_inputs(data,tmp_path/'manifest.json',source,tmp_path/'private',tmp_path,True)
    assert result['longbench']['warmup']['requests'].read_bytes()==raw
    row['messages'][0]['content']='changed'
    (source/'pilot.jsonl').write_text(json.dumps(row)+'\n')
    with pytest.raises(AssertionError):copy_inputs(data,tmp_path/'manifest.json',source,tmp_path/'other',tmp_path,True)


def test_final_cohort_is_512_distinct_and_pilot_disjoint():
    p=Path(__file__).resolve().parents[1]/'results/full-serving/sustained-512-v1/data-v1/manifest.json'
    manifest=json.loads(p.read_text())
    assert manifest['seal']==digest({k:v for k,v in manifest.items() if k!='seal'})
    for ds in manifest['datasets'].values():
        rows=ds['requests'];assert len(rows)==520 and len({r['group_id'] for r in rows})==520
        assert all(r['input_tokens']+8192<=32768 for r in rows)
        assert len(ds['items']['measured']['request_ids'])==512
        assert not set(ds['items']['measured']['request_ids'])&set(ds['items']['warmup']['request_ids'])

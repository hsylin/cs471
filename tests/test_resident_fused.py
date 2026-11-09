"""CPU equivalence checks for the one-launch static selector prototype.

These establish policy identity, not GPU compilation or performance. GPU
preflight must additionally compare all four kernel output arrays to the
independent reference before a measured use.
"""
import builtins
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from resident_dispatch.online import adversarial_inputs, cpu_greedy_assignment
from resident_fastpath.fused import FusedMapper, POLICY_ID, cpu_reference, fixed_map, validate_map
from resident_fastpath.fused_validation import check_snapshot, expected_fields


def compare(ids, real=None):
    saved=ids.copy()
    old=cpu_greedy_assignment(ids,fixed_map(),real_tokens=real)
    new=cpu_reference(ids,fixed_map(),real_tokens=real)
    for field in ('physical_ids','homes','logical_counts'):
        np.testing.assert_array_equal(new[field],getattr(old,field))
    for field in ('summary','physical_counts'):
        np.testing.assert_array_equal(new[field],old.diagnostics[field])
    np.testing.assert_array_equal(ids,saved)
    assert new['policy_id']==old.diagnostics['policy_id']==POLICY_ID
    assert new['runtime_optimality_claimed'] is False
    return new


@pytest.mark.parametrize('dtype',[np.int32,np.int64])
@pytest.mark.parametrize('case',adversarial_inputs(128),ids=lambda case:case['name'])
def test_frozen_policy_equivalence_on_adversarial_cases(case,dtype):
    compare(case['logical_ids'].astype(dtype),case['real_tokens'])


@pytest.mark.parametrize('seed',range(8))
@pytest.mark.parametrize('real',[0,1,64,128])
def test_random_full_partial_and_zero_batches_match_frozen_policy(seed,real):
    ids=np.random.default_rng(seed).integers(0,128,(128,8),dtype=np.int64)
    # Valid and invalid padded rows must never affect the real histogram.
    if real<128:
        ids[real::2]=-1
        ids[real+1::4]=128
    result=compare(ids,real)
    assert int(result['logical_counts'].sum())==real*8
    assert int(result['physical_counts'].sum())==real*8


def test_shared_ties_choose_expert_order_then_rank_zero():
    result=compare(np.tile(np.arange(8),(128,1)))
    assert result['homes'][:8].tolist()==[0,129,2,131,4,133,6,135]
    assert result['summary'].tolist()==[4,0,0,8,512,512,4,4]


def test_capacity_overrides_token_load_preference():
    ids=np.asarray([0]*100+[1]*4+[2]*3+[3]*2+[32,33,34,35]).reshape(-1,1)
    result=compare(ids)
    assert result['homes'][:4].tolist()==[128,129,130,131]
    assert result['summary'][4:].tolist()==[4,109,4,4]


def test_padding_maps_valid_ids_without_counting_them():
    ids=np.tile(np.arange(8),(128,1))
    ids[64:]=np.arange(80,88)
    result=compare(ids,64)
    assert np.all(result['logical_counts'][80:88]==0)
    np.testing.assert_array_equal(result['physical_ids'][64:],ids[64:])
    assert result['summary'][4:6].tolist()==[256,256]


@pytest.mark.parametrize('mutation',['slot_swap','redundant_copy','float','shape'])
def test_same_size_residency_changes_fail_closed(mutation):
    mapping=fixed_map()
    if mutation=='slot_swap': mapping[[0,80]]=mapping[[80,0]]
    elif mutation=='redundant_copy': mapping[-1]=32
    elif mutation=='float': mapping=mapping.astype(float)
    else: mapping=mapping.reshape(2,80)
    with pytest.raises(ValueError): validate_map(mapping)


@pytest.mark.parametrize('ids,real',[(np.array([[0.]]),None),(np.array([0]),None),
    (np.array([[-1]]),1),(np.array([[128]]),1),(np.array([[0]]),-1),
    (np.array([[0]]),2),(np.array([[0]]),True)])
def test_invalid_real_inputs_fail_closed(ids,real):
    with pytest.raises(ValueError): cpu_reference(ids,fixed_map(),real_tokens=real)


@pytest.mark.parametrize('mutation',['none','logical','replicas','float','padding'])
def test_metadata_content_guard_is_untimed_and_rejects_changed_maps(mutation):
    mapper=object.__new__(FusedMapper)
    mapper.physical_to_logical_map=validate_map(fixed_map())
    calls=[]
    mapper.verify_context=lambda:calls.append(True)
    logical=np.full((128,33),-1,dtype=np.int64)
    logical[:,0]=np.arange(128)
    logical[:32,1]=np.arange(128,160)
    replicas=np.r_[np.full(32,2),np.ones(96)].astype(np.int64)
    if mutation=='logical': logical[0,1]=129
    elif mutation=='replicas': replicas[0]=1
    elif mutation=='float': logical=logical.astype(float)
    elif mutation=='padding': logical[0,32]=0
    if mutation=='none':
        assert mapper.verify_metadata(logical,replicas,fixed_map())
        assert calls==[True]
    else:
        with pytest.raises(ValueError): mapper.verify_metadata(logical,replicas,fixed_map())
        assert not calls


def test_cpu_import_does_not_import_gpu_modules(monkeypatch):
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.split('.')[0] in ('torch','triton','vllm'):
            raise AssertionError('GPU dependency imported by CPU reference')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    path=Path(__file__).parents[1]/'resident_fastpath'/'fused.py'
    spec=importlib.util.spec_from_file_location('fused_cpu_import_test',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.POLICY_ID==POLICY_ID


@pytest.mark.parametrize('field',['physical_ids','homes','logical_counts','summary'])
def test_adversarial_proof_rejects_corruption_in_every_kernel_output(field):
    case=adversarial_inputs(128)[0]
    expected=expected_fields(case['logical_ids'],fixed_map(),128)
    snapshot={key:value.copy() for key,value in expected.items()}
    assert len(check_snapshot(snapshot,expected))==4
    snapshot[field].flat[0]+=1
    with pytest.raises(AssertionError,match=field): check_snapshot(snapshot,expected)


def test_adversarial_proof_rejects_float_output_even_when_integer_values_match():
    expected=expected_fields(np.tile(np.arange(8),(128,1)),fixed_map(),128)
    snapshot={key:value.copy() for key,value in expected.items()}
    snapshot['physical_ids']=snapshot['physical_ids'].astype(np.float32)
    with pytest.raises(AssertionError,match='physical_ids'): check_snapshot(snapshot,expected)

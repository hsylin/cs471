"""Mutations against the independent raw-execution auditor; no GPU required."""
import copy
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('independent_generalization_records',ROOT/'scripts/verify_generalization_records.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)


@pytest.fixture(scope='module')
def captured():
    count=256
    ids=np.stack([(np.arange(8)+i)%128 for i in range(count)]).astype(np.int64)
    bank=dict(expert_ids=ids,hidden_states_bf16=np.zeros((count,2048),np.uint16),
        combining_weights=np.full((count,8),.125,np.float32),token_ids=np.arange(count,dtype=np.int64),
        prompt_index=np.repeat(np.arange(64,dtype=np.int32),4),token_position=np.tile(np.arange(4,dtype=np.int32),64),
        domain_ids=np.repeat(np.arange(64,dtype=np.int8)%2,4))
    sources=[];prompts=[]
    for i in range(64):
        tokens=bank['token_ids'][i*4:i*4+4]
        source=dict(dataset='gsm8k',source_id=str(i),captured_token_count=4,full_token_count=4,
            token_ids_sha256=v.digest(tokens.tolist()),captured_prefix_normalized_sha256=v.digest(['prefix',i]))
        sources.append(source)
        prompts.append(dict(source,prompt_id=str(i),position_start=0,token_offset=0,captured_tokens=4,total_source_tokens=4,
            token_ids_sha256=v.array_hash(tokens),native_boundary_exact=True,native_router_exact=True,manual_router_exact=True,bf16_roundtrip_exact=True))
    names=['gsm8k:'+str(i) for i in range(64)]
    base=dict(environment=dict(model_id='model',model_revision='rev'),default_configs={'128':{'BLOCK_SIZE_M':64}})
    plan=dict(split='pilot',content_selection_sha256='a'*64)
    entry=dict(block_id='g',dataset='gsm8k',tokens=count,source_prompt_ids=names)
    meta=dict(block_id='g',dataset='gsm8k',split='pilot',source_prompt_ids=names,source_model='model',source_revision='rev',
        layer=0,num_experts=128,top_k=8,hidden_size=2048,correctness_pass=True,study_id='generalization_v1',native_serving_trace=False,
        timing_evidence=False,origin='prompt_causal_prefill_layer0',selection_content_sha256='a'*64,special_tokens_added=False,chat_template=None,prompts=prompts)
    return meta,bank,entry,dict(sources=sources),plan,base


def test_captured_bank_valid(captured):
    v.bank_identity(*captured)


@pytest.mark.parametrize('field,value',[('split','final'),('selection_content_sha256','b'*64),('timing_evidence',True)])
def test_wrong_bank_identity(captured,field,value):
    args=copy.deepcopy(captured);args[0][field]=value
    with pytest.raises(AssertionError):v.bank_identity(*args)


def test_rehashed_token_cannot_bypass_public_prefix(captured):
    args=copy.deepcopy(captured);meta,bank,*_=args
    bank['token_ids'][0]=999
    meta['prompts'][0]['token_ids_sha256']=v.array_hash(bank['token_ids'][:4])
    with pytest.raises(AssertionError,match='public canonical token'):v.bank_identity(*args)


def test_fulltext_identity_does_not_replace_prefix_identity(captured):
    args=copy.deepcopy(captured);args[0]['prompts'][0]['captured_prefix_normalized_sha256']='b'*64
    with pytest.raises(AssertionError,match='prefix hints'):v.bank_identity(*args)


def test_not_legacy48_and_no_repeated_job_combinations():
    a,_=v.job_indices(256,'group-a');b,_=v.job_indices(256,'group-a');other,_=v.job_indices(256,'group-b')
    assert a.shape==(272,128) and np.array_equal(a,b) and not np.array_equal(a,other)
    assert len({tuple(sorted(x)) for x in a})==272
    assert np.all(np.diff(np.sort(a,axis=1),axis=1)>0)
    assert not np.array_equal(a[16:48],a[48:80])


@pytest.fixture(scope='module')
def route_fixture(captured):
    _,bank,entry,_,_,base=captured
    indices,_=v.job_indices(256,'g');logical=bank['expert_ids'][indices]
    counts,hashes,ds,samples={},{},{},{}
    for mode in ('native','online'):
        c=[];d=[];s=[];h=hashlib.sha256()
        for i,ids in enumerate(logical):
            physical,co,opt=v.routes.assignment(ids,mode);c.append(co);d.append(v.routes.metrics(co,opt,64));h.update(physical.tobytes())
            if i in (16,271):s.append(physical.astype(np.uint8))
        counts[mode]=np.asarray(c,np.int32);hashes[mode]=h.hexdigest();ds[mode]=d;samples[mode]=np.stack(s)
    provenance=dict(capture_sha256='a'*64,replay_sha256=v.array_hash(indices),source_prompt_ids=entry['source_prompt_ids'])
    state=dict(logical=logical,counts=counts,hashes=hashes,diagnostics=ds,sequence=list(range(16,272)),provenance=provenance,
        reuse=v.reuse_summary(indices,256),input_meta={'seed':471503},arrays={'indices':indices,'online_physical_counts':counts['online']},artifact_meta={'seed':471503})
    row=dict(arm='combined',**provenance,logical_routing_sha256=v.array_hash(logical),assignment_sha256=hashes['online'],
        resolved_kernel_config=base['default_configs']['128'],source_map=v.MAP.tolist(),final_map=v.MAP.tolist(),job_reuse=state['reuse'],newinputmeta=state['input_meta'],
        assignment_metrics=copy.deepcopy(ds['online'][16:]),decision_log=[],terminal_physical_window=counts['online'][-16:].tolist())
    for step in (63,127,191):
        raw=state['sequence'][step-15:step+1];co=counts['online'][raw]
        row['decision_log'].append(dict(step=step,physical_slot_assignments_window=co.tolist(),physical_slot_assignments_sha256=v.array_hash(co),
            current_counts=(np.bincount(logical[raw].ravel(),minlength=128)*2).tolist(),physical_owner_assignments_per_step=co.reshape(16,2,80).sum(2).tolist(),
            current_map=v.MAP.tolist(),committed_map=v.MAP.tolist(),committed_map_sha256=v.array_hash(v.MAP),allowed=False,reason='always_skip',commit_completed=True,
            transfer_sent_bytes=0,transfer_sent_bytes_by_rank=[0,0],physical_window_next_write=0,physical_counter_input_multiplicity=2,
            physical_window_measured_step_range=[step-15,step],proposed_map=v.MAP.tolist(),candidate_changed_slots=0))
    row['decision_trajectory_sha256']=v.digest([{k:d[k] for k in ('step','current_counts','current_map','proposed_map','committed_map')} for d in row['decision_log']])
    ranks=['a'*64,'b'*64];row.update(source_weight_rank_sha256=ranks,source_weights_sha256=hashlib.sha256(json.dumps(ranks).encode()).hexdigest())
    numeric=dict(correctness_pass=True,correctness_atol=.02,correctness_rtol=.03,full_batch_inputs_checked=2,correctness_max_abs_error=.001,full_batch_max_abs_difference=.001)
    row.update(numeric);row['preflight_correctness']=dict(numeric,full_batch_inputs_checked=272)
    row['adversarial_checks']=dict(passed=True,rank_agreement=True,native_recording_state_unchanged=True,cases_checked=9,
        recording_modes_per_case=2,gpu_invocations_checked=18,batch_tokens=128,cases=[],evidence_sha256=v.digest([]))
    return row,state,base


def test_all_route_windows_and_late_input_checked(route_fixture):
    v.routing(*route_fixture)


@pytest.mark.parametrize('mutation',['shift_window','wrong_multiplicity','terminal_old','late_metric','coverage48','pending_commit'])
def test_route_mutations(route_fixture,mutation):
    row,state,base=copy.deepcopy(route_fixture)
    if mutation=='shift_window':row['decision_log'][0]['physical_slot_assignments_window']=state['counts']['online'][63:79].tolist()
    elif mutation=='wrong_multiplicity':row['decision_log'][0]['current_counts']=[x//2 for x in row['decision_log'][0]['current_counts']]
    elif mutation=='terminal_old':row['terminal_physical_window']=state['counts']['online'][16:32].tolist()
    elif mutation=='late_metric':row['assignment_metrics'][255]['per_rank_tokens'][0]+=1
    elif mutation=='coverage48':row['preflight_correctness']['full_batch_inputs_checked']=48
    else:row['decision_log'][0]['commit_completed']=False
    with pytest.raises(AssertionError):v.routing(row,state,base)


def test_rehashed_artifact_late_input_mutation(tmp_path,route_fixture):
    row,state,_=copy.deepcopy(route_fixture);p=tmp_path/'inputs.npz'
    arrays=copy.deepcopy(state['arrays']);arrays['indices'][271,[0,1]]=arrays['indices'][271,[1,0]]
    np.savez(p,**arrays,metadata=np.asarray(json.dumps(state['artifact_meta'])))
    row['assignment_artifact']=dict(path='/server/inputs.npz',sha256=v.file_hash(p))
    with pytest.raises(AssertionError,match='artifact indices'):v.artifact(row,state,[tmp_path])


def test_unchanged_guard_requires_terminal_pair():
    guard=dict(mode='fast',diagnostic_timing=False,pending_pair=False,movement_calls=4,commit_calls=4,agreement_calls=4,
        local_unchanged_calls=4,unanimously_unchanged_calls=4,movement_skips=4,commit_skips=4,original_movement_calls=0,
        original_commit_calls=0,fallback_calls=0,diagnostics=[])
    row=dict(arm='fast',n_jobs=256,noop_guard_ranks=[guard,copy.deepcopy(guard)])
    v.fix.guard_counts(row)
    row['noop_guard_ranks'][1]['commit_calls']=3
    with pytest.raises(AssertionError):v.fix.guard_counts(row)


def test_global_balance_does_not_hide_within_rate_confound():
    permutations=list(itertools.permutations(v.ARMS))
    bad=[dict(arm_order=list(permutations[(2*i+r)%6]),calendar={'rate_kind':rate})
         for i in range(12) for r,rate in enumerate(('calibrated','common'))]
    with pytest.raises(AssertionError,match='within rate'):v.order_balance(bad)
    good=[dict(arm_order=list(p),calendar={'rate_kind':rate})
          for rate in ('calibrated','common') for p in permutations for _ in range(2)]
    v.order_balance(good)


def test_absolute_clocks_detect_unacknowledged_rank_completion():
    row=dict(n_jobs=256,epoch_monotonic_ns=100,horizon_ns=1,arrival_offsets_ns=[0]*256,
        coordinator_admit_ns=[100+i*10 for i in range(256)],coordinator_start_ns=[101+i*10 for i in range(256)],
        coordinator_finish_ns=[108+i*10 for i in range(256)],observation_end_ns=2670,
        per_rank_start_ns=[[102+i*10 for i in range(256)]]*2,per_rank_gpu_complete_ns=[[107+i*10 for i in range(256)]]*2)
    v.old.timing_metrics(row)
    row=copy.deepcopy(row);row['per_rank_gpu_complete_ns'][1][255]=row['coordinator_finish_ns'][255]+1
    with pytest.raises(AssertionError,match='acknowledged'):v.old.timing_metrics(row)


def test_full_calibration_inventory_and_source_overlap():
    groups=[{'id':f'{d}-{i}','dataset':d} for d in v.DATASETS for i in (1,2)]
    blocks=[dict(block_id=g['id'],dataset=g['dataset'],split='pilot',sources=[dict(dataset=g['dataset'],source_id=f"{g['id']}-{i}") for i in range(64)]) for g in groups]
    banks=[dict(block_id=b['block_id'],dataset=b['dataset'],split='pilot',documents=64,
        source_prompt_ids=[s['dataset']+':'+s['source_id'] for s in b['sources']]) for b in blocks]
    plan=dict(groups=groups,banks=banks,phase='calibration',split='pilot',order_seed=471514,cases=[])
    result=v.matrix(plan,dict(blocks=blocks),dict(blocks=[]))
    assert len(result[3])==6 and all(x[1:]==(None,'saturated','native') for x in result[3])
    bad=copy.deepcopy(plan);bad['banks'].pop()
    with pytest.raises(AssertionError,match='capture inventory'):v.matrix(bad,dict(blocks=blocks),dict(blocks=[]))
    blocks=copy.deepcopy(blocks);blocks[1]['sources'][0]=blocks[0]['sources'][0]
    with pytest.raises(AssertionError,match='Disjoint'):v.matrix(plan,dict(blocks=blocks),dict(blocks=[]))


def test_coherently_rehashed_shifted_calendar_rejected():
    block=dict(id='arrival',trace='code',source_date_utc='2024-05-12',source_rows_sha256='f'*64,
        original_horizon_ns=512,original_offsets_ns=list(range(0,512,2)))
    plan=dict(calibration={'g':{'calibration_elapsed_ns':2560}})
    from fractions import Fraction
    horizon=round(Fraction(2560*100,85));scale=Fraction(horizon,512)
    cal=dict(rate_kind='calibrated',arrival_offsets_ns=[round(x*scale) for x in block['original_offsets_ns']],horizon_ns=horizon,
        source_block='arrival',scale_numerator=scale.numerator,scale_denominator=scale.denominator,absolute_arrivals=True,
        drop_policy='none; drain every admitted job',name='arrival/calibrated',pattern='code',rho=float(Fraction(2560,horizon)),
        source_date_utc='2024-05-12',source_rows_sha256='f'*64)
    cal['sha256']=v.digest(cal);v.calendar(cal,block,plan,'g')
    cal['arrival_offsets_ns'][200]+=1;cal['sha256']=v.digest({k:x for k,x in cal.items() if k!='sha256'})
    with pytest.raises(AssertionError,match='scaled calendar'):v.calendar(cal,block,plan,'g')

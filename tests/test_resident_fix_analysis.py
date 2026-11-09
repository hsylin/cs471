import copy
import itertools
import numpy as np
import pytest
from resident_fastpath.analysis import validate_plan,fix_native_issues,precision_issues
from resident_fastpath.common import OLD_CLOSURE,seal,scale_calendar
from resident_fastpath.inference import build_arm_orders
from stock_comparison.data import digest


def plan(phase='pilot',split=None):
    split=split or phase;n=2 if split=='pilot' else 8
    groups=[dict(id=f'{d}-{i}',dataset=d) for d,k in [('gsm8k',n),('wildchat',n),('mbpp',1)] for i in range(k)]
    banks=[dict(block_id=g['id'],dataset=g['dataset'],split=split,capture_sha256='f'*64) for g in groups]
    cases=[]
    if phase!='calibration':
        for c in build_arm_orders(groups,seed=471711,phase=phase):
            gid=c['content_group_id'];typ=c['arrival_type'];source=dict(id=f'{gid}/{typ}',trace=typ,source_date_utc='2024-05-14',source_rows_sha256='a'*64,
                jobs=4096,original_offsets_ns=list(range(4096)),original_horizon_ns=4096)
            cases.append(dict(**c,block_id=gid,dataset=next(g['dataset'] for g in groups if g['id']==gid),source_block=source,
                calendar=scale_calendar(source,81920,rho=c['rho'])))
    files={'module.py':'a'*64};source=digest(files);selected={'selector':'three_kernel','diagnostic_basis':'synthetic test fixture'}
    return seal(dict(schema='resident_fix_plan_v1',phase=phase,split=split,arms=['stockoff','current','fix'],jobs=4096,batch_tokens=128,
        old_closure_sha256=OLD_CLOSURE,source_files=files,source_sha256=source,selected_fix=selected,
        selected_fix_content_sha256=digest(selected),selected_fix_sha256='c'*64,banks=banks,cases=cases,hardware={'allocation_id':'1'},
        calibration_record_ids={g['id']:'b'*64 for g in groups},
        calibration_provenance={g['id']:dict(calibration_elapsed_ns=81920,allocation_id='1',source_sha256=source,record_id='b'*64) for g in groups}))


def fixed_row():
    mapping=list(range(128))+list(range(32));gate='d'*64
    base=dict(schema_version=1,arm='fix',enable_eplb=True,replicas=32,physical_slots=160,local_slots=80,static_resident_mode=True,
        controller_active=False,upstream_eplb_disabled=False,selector_kind='three_kernel',logical_router_computation_included=True,
        native_output_reduction=True,stock_router_unmodified=False,stock_policy_unmodified=False,custom_all_reduce_disabled=False,
        source_map=mapping,final_map=mapping,controller_step_calls=0,native_counter=0,native_valid_steps=0,
        recorded_step_indices=[],deferred_boundary_steps=[],decisions=[],async_events=[],physical_window_next_write=0,pending_result=False,
        rebalanced=False,migration_sent_bytes_local=0,migration_sent_messages_local=0,terminal_proposals_suppressed=0,
        completion_disposition='normal_commit_no_discard',guard=None,terminal_physical_window_totals_local=[0]*160,gate_weight_sha256=gate)
    start={**base,'static_step_calls':0,'mapper_calls':0};end={**base,'static_step_calls':4096,'mapper_calls':4096}
    audited={**end,**{k:True for k in ('resident_state_verified','rank_map_agreement','normal_completion_verified','static_state_unchanged','observation_and_migration_inactive')}}
    peer=dict(start=start,end=end,audit=audited,shortcut_guard=None)
    pre=dict(correctness_pass=True,full_batch_inputs_checked=4112,atol=.02,rtol=.03,full_batch_max_abs_difference=0.,
        logical_ids_equal=True,combining_weights_equal=True,rank_routing_agreement=True,input_unchanged=True,
        fp32_job_indices=[16,765,3765,4111],fp32_tokens_per_job=16,online_assignment_inputs_checked=4112,
        logical_routing_sha256='a'*64,float32_combining_weights_sha256='b'*64,physical_assignment_digest='c'*64)
    post=dict(correctness_pass=True,full_batch_inputs_checked=16,atol=.02,rtol=.03,full_batch_max_abs_difference=0.,raw_indices=np.linspace(16,4111,16,dtype=np.int64).tolist())
    return dict(arm='fix',input_meta={'gate_weight_sha256':gate},preflight=pre,postflight=post,per_rank_state=[copy.deepcopy(peer),copy.deepcopy(peer)])


@pytest.mark.parametrize('phase,split,expected',(('calibration','pilot',5),('calibration','final',17),('pilot','pilot',5),('final','final',17)))
def test_new_matrix_scope_and_feasible_order_contract(phase,split,expected):
    groups,cases,balance=validate_plan(plan(phase,split))
    assert len(groups)==expected
    assert len(cases)==(0 if phase=='calibration' else expected*4)


def test_selection_change_cannot_hide_in_resealed_plan():
    p=plan();p['selected_fix']['selector']='fused';p=seal(p)
    with pytest.raises(ValueError,match='Selected fix content'):validate_plan(p)


def test_changed_old_closure_is_rejected():
    p=plan();p['old_closure_sha256']='a'*64;p=seal(p)
    with pytest.raises(ValueError,match='Old frozen'):validate_plan(p)


def test_calibrated_calendar_and_flat_case_cannot_disagree():
    p=plan();p['cases'][0]['rho']=.4;p=seal(p)
    with pytest.raises(ValueError,match='Flattened'):validate_plan(p)


def test_valid_static_state_has_no_controller_recording_or_transfer():
    assert fix_native_issues(fixed_row(),'three_kernel')==[]


@pytest.mark.parametrize('field,value',(('controller_step_calls',1),('native_counter',1),('native_valid_steps',1),('migration_sent_bytes_local',1),('recorded_step_indices',[0]),('mapper_calls',4095),('static_step_calls',4095),('upstream_eplb_disabled',True)))
def test_static_state_mutations_are_rejected(field,value):
    row=fixed_row();row['per_rank_state'][0]['end'][field]=value
    assert fix_native_issues(row,'three_kernel')


def test_static_window_recording_even_without_counter_change_is_rejected():
    row=fixed_row();row['per_rank_state'][1]['end']['terminal_physical_window_totals_local'][0]=1
    assert any('recording_active' in x for x in fix_native_issues(row,'three_kernel'))


def test_wrong_selected_selector_and_missing_fused_proof_rejected():
    row=fixed_row();issues=fix_native_issues(row,'fused')
    assert any('static_state' in x for x in issues) and 'fused_adversarial_proof' in issues


def fused_row():
    row=fixed_row()
    for rank,peer in enumerate(row['per_rank_state']):
        for scope in ('start','end','audit'):peer[scope]['selector_kind']='fused'
        peer['audit']['fused_stream_context']=dict(device=f'cuda:{rank}',stream_handle=100+rank,sequential_single_stream_required=True,reused_output_storage=True)
    row['runtime_metadata']={'fused_adversarial':dict(passed=True,cases_checked=9,gpu_invocations_checked=18,reference_count=2,rank_agreement=True,stream_context_unchanged=True,input_ids_unchanged=True,internal_real_tokens=128)}
    return row

def test_fused_context_accepts_distinct_valid_rank_streams():
    assert fix_native_issues(fused_row(),'fused')==[]

@pytest.mark.parametrize('field,value',(('device','cuda:0'),('stream_handle',-1),('stream_handle',True),('sequential_single_stream_required',False),('reused_output_storage',False)))
def test_fused_context_contract_rejects_wrong_rank_or_scope(field,value):
    row=fused_row();row['per_rank_state'][1]['audit']['fused_stream_context'][field]=value
    assert 'rank1_fused_stream_context' in fix_native_issues(row,'fused')

def test_dense_reference_precision_requires_both_ranks_and_global_flags():
    rt=dict(tf32_allowed=False,float32_matmul_precision='highest',precision_per_rank=[dict(allow_tf32=False,matmul_precision='highest') for _ in range(2)])
    assert not precision_issues(rt)
    for changed in ({**rt,'tf32_allowed':True},{**rt,'precision_per_rank':rt['precision_per_rank'][:1]},{**rt,'float32_matmul_precision':'high'}):assert precision_issues(changed)
    rt['precision_per_rank'][1]['allow_tf32']=True
    assert precision_issues(rt)

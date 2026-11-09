import copy
import importlib.util
from pathlib import Path
import pytest
from resident_residual.analysis import validate_plan,resident_native_issues,geometry_issues
from resident_residual.common import PREVIOUS_CLOSURE,seal,scale_calendar
from resident_residual.inference import build_arm_orders
from resident_residual.alignment import EXPECTED_CONFIG,SOURCE_SHA256
from stock_comparison.data import digest


def plan(phase='pilot',split=None):
    split=split or phase;n=2 if split=='pilot' else 8
    groups=[dict(id=f'{d}-{i}',dataset=d) for d in ('gsm8k_train','wildchat') for i in range(n)]
    banks=[dict(block_id=g['id'],dataset=g['dataset'],split=split,capture_sha256='a'*64) for g in groups]
    cases=[]
    if phase!='calibration':
        for c in build_arm_orders(groups,seed=471804,phase=phase):
            gid=c['content_group_id'];typ=c['arrival_type'];source=dict(id=f'{gid}/{typ}',trace=typ,source_date_utc='2024-05-14',source_rows_sha256='a'*64,jobs=4096,original_offsets_ns=list(range(4096)),original_horizon_ns=4096)
            cases.append(dict(**c,block_id=gid,dataset=next(g['dataset'] for g in groups if g['id']==gid),source_block=source,calendar=scale_calendar(source,81920,rho=c['rho'])))
    files={'synthetic.py':'a'*64};src=digest(files);selected=dict(selector='fused',selector_launch='jit',alignment_trim=True,mapper_input_checks=True)
    return seal(dict(schema='resident_residual_plan_v1',phase=phase,split=split,arms=['stockoff','previous','fix'],jobs=4096,batch_tokens=128,
        previous_closure_sha256=PREVIOUS_CLOSURE,source_files=files,source_sha256=src,selected_fix=selected,selected_fix_content_sha256=digest(selected),selected_fix_sha256='b'*64,banks=banks,cases=cases,hardware={'allocation_id':'1'},
        calibration_record_ids={g['id']:'b'*64 for g in groups},calibration_provenance={g['id']:dict(calibration_elapsed_ns=81920,allocation_id='1',source_sha256=src,record_id='b'*64) for g in groups}))


def resident_row(arm='fix'):
    path=Path(__file__).parent/'test_resident_fix_analysis.py'
    spec=importlib.util.spec_from_file_location('prior_static_fixture',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    row=mod.fused_row();row['arm']=arm
    trim=arm=='fix'
    for p in row['per_rank_state']:
        for scope in ('start','end','audit'):p[scope].update(arm=arm,selector_launch='jit',alignment_trim=trim,mapper_input_checks=True)
        p['alignment']=dict(schema='native_alignment_extent_v1',enabled=trim,probe_only=False,alignment_calls=4096 if trim else 0,native_allocation_preserved=True,native_alignment_algorithm_preserved=True,capacity_bound=5056,active_local_cap=64,global_physical_experts=160,native_capacity=11104,config=EXPECTED_CONFIG,native_source_sha256=SOURCE_SHA256,runtime_gpu_readback=False,actual_padding_verified=False)
    row['preflight']['alignment_probe']=[dict(row['per_rank_state'][i]['alignment'],probe_only=True,actual_padding_verified=True,alignment_calls=4112) for i in range(2)]
    return row


@pytest.mark.parametrize('phase,split,n',(('calibration','pilot',4),('calibration','final',16),('pilot','pilot',4),('final','final',16)))
def test_exact_main_matrix(phase,split,n):
    groups,cases,orders=validate_plan(plan(phase,split));assert len(groups)==n and len(cases)==(0 if phase=='calibration' else 4*n)


@pytest.mark.parametrize('arm',('previous','fix'))
def test_public_arm_and_measured_alignment_contract(arm):
    row=resident_row(arm);p=plan()
    assert resident_native_issues(row,p)==[]
    assert geometry_issues(row,p)==[]


@pytest.mark.parametrize('field,value',(('arm','fix'),('alignment_trim',True),('mapper_input_checks',False)))
def test_previous_cannot_silently_receive_selected_changes(field,value):
    row=resident_row('previous');row['per_rank_state'][0]['end'][field]=value
    assert resident_native_issues(row,plan())


@pytest.mark.parametrize('field,value',(('alignment_calls',4095),('capacity_bound',4992),('probe_only',True),('runtime_gpu_readback',True),('native_allocation_preserved',False)))
def test_trim_timing_scope_or_bound_mutation_rejected(field,value):
    row=resident_row();row['per_rank_state'][0]['alignment'][field]=value
    assert geometry_issues(row,plan())


def test_missing_or_partial_actual_geometry_preflight_is_rejected():
    row=resident_row();row['preflight']['alignment_probe'][1]['alignment_calls']=64
    assert 'rank1_alignment_preflight' in geometry_issues(row,plan())
    row['preflight']['alignment_probe'].pop()
    assert 'missing_rank_alignment_preflight' in geometry_issues(row,plan())


def test_selected_options_must_be_actual_booleans_and_bound_content():
    p=plan();p['selected_fix']['alignment_trim']=1;p['selected_fix_content_sha256']=digest(p['selected_fix']);p=seal(p)
    with pytest.raises(ValueError,match='implementation'):validate_plan(p)
    p=plan();p['selected_fix']['mapper_input_checks']=False;p=seal(p)
    with pytest.raises(ValueError,match='Selected fix content'):validate_plan(p)


def test_test_split_label_and_old_source_cannot_pass_as_new_study():
    p=plan();p['banks'][0]['dataset']='gsm8k';p=seal(p)
    with pytest.raises(ValueError,match='allocation'):validate_plan(p)
    p=plan();p['previous_closure_sha256']='0'*64;p=seal(p)
    with pytest.raises(ValueError,match='closure'):validate_plan(p)


def test_selected_launch_path_cannot_be_missing_or_silently_changed():
    p=plan();del p['selected_fix']['selector_launch'];p['selected_fix_content_sha256']=digest(p['selected_fix']);p=seal(p)
    with pytest.raises(ValueError,match='implementation'):validate_plan(p)
    row=resident_row('previous');row['per_rank_state'][1]['end']['selector_launch']='prebound_compiled'
    assert 'rank1_end_selector_launch' in resident_native_issues(row,plan())


def compiled_row_and_plan():
    from resident_residual.compiled import TRITON_SOURCE_SHA256,GPU_SOURCE_SHA256
    p=plan();p['selected_fix'].update(selector_launch='prebound_compiled',alignment_trim=False);p['selected_fix_content_sha256']=digest(p['selected_fix']);p=seal(p)
    row=resident_row('fix')
    kernels={}
    for dtype in ('int32','int64'):
        signature=dict(ids='*i32' if dtype=='int32' else '*i64',real_tokens='*i32',output='*i32' if dtype=='int32' else '*i64',homes='*i32',logical_counts='*i32',summary='*i32',N='constexpr',TOPK='constexpr')
        kernels[dtype]=dict(compiled_hash='a'*64,cubin_sha256='b'*64,ptx_sha256='c'*64,signature=[[repr(k),repr(v)] for k,v in sorted(signature.items(),key=lambda item:repr(item[0]))],constants=[['(6,)','1024'],['(7,)','8']],pointer_specialization_attributes=[['(0,)',"[['tt.divisibility', 16]]"]],num_warps=4,num_ctas=1,arch=90,shared=128,baseline_jit_same_compiled_object=True)
    for rank,peer in enumerate(row['per_rank_state']):
        for scope in ('start','end','audit'):
            peer[scope].update(alignment_trim=False,selector_launch='prebound_compiled')
            peer[scope]['compiled_launch']=dict(schema='resident_prebound_compiled_launch_v1',mode='prebound_compiled',triton_version='3.7.1',gpu_algorithm_changed=False,input_checks_retained=True,pointer_alignment_guard=True,device=f'cuda:{rank}',stream_handle=100+rank,fixed_argument_storage_verified=True,hook_contract_verified=True,dynamic_launch_metadata_hooks_preserved=True,jit_prerun_hooks_required_empty=True,grid=[1,1,1],constexpr_args=[1024,8],gpu_source_sha256=GPU_SOURCE_SHA256,calls=0 if scope=='start' else 4096,compiler_source_sha256=copy.deepcopy(TRITON_SOURCE_SHA256),kernels=copy.deepcopy(kernels))
        peer['alignment'].update(enabled=False,alignment_calls=0)
        row['preflight']['alignment_probe'][rank]['enabled']=False
    return row,p


def test_prebound_launcher_retains_exact_kernel_specializations_and_calls():
    row,p=compiled_row_and_plan()
    assert resident_native_issues(row,p)==[] and geometry_issues(row,p)==[]


@pytest.mark.parametrize('mutation',('count','kernel','constant','signature','source','checks','stream','hooks','launchgrid'))
def test_compiled_launch_mutations_cannot_pass_by_name_alone(mutation):
    row,p=compiled_row_and_plan();m=row['per_rank_state'][1]['end']['compiled_launch']
    if mutation=='count':m['calls']=4095
    elif mutation=='kernel':m['kernels']['int32']['cubin_sha256']='0'*64
    elif mutation=='constant':m['kernels']['int64']['constants'][0][1]='512'
    elif mutation=='signature':m['kernels']['int32']['signature'].pop()
    elif mutation=='source':m['compiler_source_sha256']['runtime/jit.py']='0'*64
    elif mutation=='checks':m['input_checks_retained']=False
    elif mutation=='stream':m['stream_handle']=-1
    elif mutation=='hooks':m['jit_prerun_hooks_required_empty']=False
    else:m['grid']=[2,1,1]
    assert resident_native_issues(row,p)

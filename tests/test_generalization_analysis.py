"""Synthetic validator fixtures only; these are not measured performance data."""
import copy
import itertools
import math
from pathlib import Path
import numpy as np
import pytest
from generalization.analysis import analyze,canonical_hash,_sha,_window_spans,ROOT,ARMS
from arrival_study.schedules import fcfs_metrics,calibration_capacity
from tests.test_arrival_schedules import row_fixture


def fixture(phase='pilot'):
    n=2 if phase!='final' else 6
    groups=[dict(id=f'{d}-{i}',dataset=d) for d in ('gsm8k','mbpp','wildchat') for i in range(n)]
    source_files={'generalization/runner.py':_sha(ROOT/'generalization/runner.py')}
    mapping=list(range(128))+list(range(32))
    plan=dict(phase=phase,split='pilot' if phase!='final' else 'final',generalization_schema_version=1,
        groups=groups,banks=[dict(block_id=g['id'],capture_sha256='capture-'+g['id']) for g in groups],arms=list(ARMS),batch_tokens=128,
        physical_to_logical_map=mapping,analysis_sha256=_sha(ROOT/'generalization/analysis.py'),inference_sha256=_sha(ROOT/'generalization/inference.py'),
        execution_source_files=source_files,execution_source_sha256=canonical_hash(source_files),protocol_files={},
        base_manifest={'sha256':'base'},content_selection_sha256='selection',arrival_seed=471502,
        hardware=dict(hostname='fixture-host',gpu_uuids=['gpu0','gpu1'],allocation_id='fixture-allocation'),
        calibration={g['id']:dict(allocation_id='fixture-allocation') for g in groups},cases=[])
    if phase!='calibration':
        for i,g in enumerate(groups):
            for t,r in itertools.product(('code','conversation'),('calibrated','common')):
                day=(12 if t=='code' else 14)+i%5
                b=dict(id=g['id']+'-'+t,trace=t,source_date_utc=f'2024-05-{day:02d}',source_rows_sha256='source-'+g['id']+t)
                c=dict(arrival_offsets_ns=[j*10000 for j in range(256)],horizon_ns=2560000,pattern=t,rho=.85,rate_kind=r,
                    name=b['id']+'/'+r,source_date_utc=b['source_date_utc'],source_rows_sha256=b['source_rows_sha256'])
                c['sha256']=canonical_hash(c)
                plan['cases'].append(dict(block_id=g['id'],dataset=g['dataset'],source_block=b,calendar=c,arm_order=list(list(itertools.permutations(ARMS))[len(plan['cases'])%6])))
    plan['freeze']={'sha256':canonical_hash(plan)}
    counts=np.zeros((16,160),dtype=np.int64);counts[:,:128]=8
    rows=[]
    work=plan['cases'] if phase!='calibration' else [dict(block_id=g['id'],dataset=g['dataset']) for g in groups]
    for case in work:
        g=next(g for g in groups if g['id']==case['block_id'])
        cal=case.get('calendar');block=case.get('source_block')
        for arm in (case['arm_order'] if phase!='calibration' else ['native']):
            row=row_fixture(cal['arrival_offsets_ns'] if cal else [0]*256,[{'native':1000,'shortcut':800,'combined':700}[arm]]*256,horizon=cal['horizon_ns'] if cal else 1)
            row['per_rank_start_ns']=[row['coordinator_start_ns'][:]]*2;row['per_rank_gpu_complete_ns']=[row['coordinator_finish_ns'][:]]*2
            row.update(content_group_id=g['id'],block_id=g['id'],dataset=g['dataset'],arrival_type=block['trace'] if block else None,
                arrival_date=block['source_date_utc'] if block else None,rate_kind=cal['rate_kind'] if cal else 'saturated',arm=arm,
                status='complete',phase=phase,config_hash=plan['freeze']['sha256'],generalization_schema_version=1,
                execution_source_sha256=plan['execution_source_sha256'],base_config_hash='base',regime='newcontent',batch_tokens=128,repetition=0,
                correctness_pass=True,evidence_eligible=phase=='final',diagnostic_only=False,guard_symbols_restored=True,
                normal_completion_verified=True,resident_state_verified=True,rank_assignment_agreement=True,migration_bytes=0,migration_sent_messages=0,
                terminal_proposals_suppressed=1,policy='SKIP',capture_sha256='capture-'+g['id'],content_selection_sha256='selection',source_map=mapping,final_map=mapping,
                dispatch_mode='online' if arm=='combined' else 'native',noop_enabled=arm!='native',assignment_selection_cost_included=True,
                online_residency_metadata_verified=arm=='combined',mapper_calls_measured=256 if arm=='combined' else 0,
                measurement_setup_revision=2,gc_enabled_during_measurement=True,warmup_raw_indices=list(range(16)),conditioning_raw_indices=[i%16 for i in range(64)],
                measured_raw_indices=list(range(16,272)),full_unique_preflight_gpu_routes_checked=272,
                correctness_atol=.02,correctness_rtol=.03,full_batch_inputs_checked=2,correctness_max_abs_error=.001,full_batch_max_abs_difference=.001,
                preflight_correctness=dict(correctness_pass=True,correctness_atol=.02,correctness_rtol=.03,full_batch_inputs_checked=272,correctness_max_abs_error=.001,full_batch_max_abs_difference=.001),
                allocation_id='fixture-allocation',hostname='fixture-host',physical_gpu_uuids=['gpu0','gpu1'],rank_gpu_uuids=['gpu0','gpu1'],
                arrival_clock_contract=dict(monotonic=True,adjustable=False,implementation='clock_gettime(CLOCK_MONOTONIC)'),
                assignment_metrics=[{'per_rank_tokens':[512,512]}]*256,job_reuse=dict(unique_all_job_combinations=272,unique_measured_job_combinations=256),
                arrival_schedule=cal,arrival_block_id=block['id'] if block else 'native-calibration',arrival_seed=471502 if block else None,
                schedule_name=cal['name'] if cal else 'saturated',pattern=cal['pattern'] if cal else 'saturated',rho=.85 if cal else None,
                arm_order=case['arm_order'] if cal else ['native'],replay_sha256='replay-'+g['id'],source_prompt_ids=['source-'+g['id']],
                source_weights_sha256='weights',source_weight_rank_sha256=['w0','w1'],resolved_kernel_config={},logical_routing_sha256='logical-'+g['id'],
                assignment_sha256=('online-' if arm=='combined' else 'native-')+g['id'])
            guard=dict(mode='fast',diagnostic_timing=False,pending_pair=False,movement_calls=4,commit_calls=4,agreement_calls=4,
                local_unchanged_calls=4,unanimously_unchanged_calls=4,movement_skips=4,commit_skips=4,original_movement_calls=0,original_commit_calls=0,fallback_calls=0,diagnostics=[])
            row['noop_guard_ranks']=[] if arm=='native' else [guard.copy(),guard.copy()]
            row['decision_log']=[dict(step=s,allowed=False,current_map=mapping,committed_map=mapping,proposed_map=mapping,
                physical_slot_assignments_window=counts.tolist(),current_counts=[256]*128) for s in (63,127,191)]
            row['decision_trajectory_sha256']=canonical_hash([{k:d[k] for k in ('step','current_counts','current_map','proposed_map','committed_map')} for d in row['decision_log']])
            row['terminal_physical_window']=counts.tolist();row['queue_metrics']=fcfs_metrics(row);row['observed_window_spans']=_window_spans(row)
            if phase=='calibration':row['calibration_capacity']=calibration_capacity(row['coordinator_start_ns'],row['coordinator_finish_ns'])
            row['record_id']=canonical_hash({k:row[k] for k in ('config_hash','phase','arrival_block_id','block_id','regime','batch_tokens','arm','repetition','schedule_name')})
            rows.append(row)
    return plan,rows


def test_pilot_only_descriptive_and_variance_allocation_no_final_interval():
    plan,rows=fixture();result=analyze(plan,rows)
    assert result['valid'],result['errors']
    assert result['expected_records']==72 and result['matched_triplets']==24
    assert result['confirmatory'] is None and result['allocation_decision']['groups_per_dataset']==6
    pair=result['paired_cases'][0]
    assert pair['ratios']['queue_wait_mean_ns'] is None and pair['differences']['queue_wait_mean_ns']==0


def test_calibration_capacity_recomputed_not_trusted():
    plan,rows=fixture('calibration');result=analyze(plan,rows)
    assert result['valid'],result['errors']
    assert result['expected_records']==6 and len(result['calibration'])==6
    rows[0]['calibration_capacity']['native_capacity_jobs_per_second']+=1
    assert not analyze(plan,rows)['valid']


def test_final_complete_contract_calls_frozen_inference():
    plan,rows=fixture('final');result=analyze(plan,rows)
    assert result['valid'],result['errors']
    assert result['expected_records']==216 and result['confirmatory']['bootstrap']['seed']==471512
    assert result['confirmatory']['primary']['ratio']==pytest.approx(.875)
    assert result['allocation_decision'] is None


@pytest.mark.parametrize('mutation',['missing','duplicate','clock','source','wrong_map','counter','route','order','preflight','calendar'])
def test_material_mutations_invalidate_matrix(mutation):
    plan,rows=fixture()
    if mutation=='missing':rows.pop()
    elif mutation=='duplicate':rows.append(copy.deepcopy(rows[0]))
    elif mutation=='clock':rows[0]['coordinator_finish_ns'][2]=rows[0]['coordinator_start_ns'][2]-1
    elif mutation=='source':rows[0]['execution_source_sha256']='other'
    elif mutation=='wrong_map':rows[0]['final_map']=[0]*160
    elif mutation=='counter':rows[0]['decision_log'][0]['physical_slot_assignments_window'][0][0]+=1
    elif mutation=='route':rows[1]['assignment_sha256']='different-native'
    elif mutation=='order':rows[0],rows[1]=rows[1],rows[0]
    elif mutation=='preflight':rows[0]['preflight_correctness']['full_batch_inputs_checked']=2
    elif mutation=='calendar':rows[0]['arrival_offsets_ns']=rows[0]['arrival_offsets_ns'][:];rows[0]['arrival_offsets_ns'][4]+=1
    assert not analyze(plan,rows)['valid']


def test_same_logical_counts_cannot_hide_native_shortcut_physical_mismatch():
    plan,rows=fixture()
    # Experts0 and128 are two physical copies of logical0: transfer one count
    # between them preserves logical collapse and total, but violates identity.
    window=rows[1]['decision_log'][0]['physical_slot_assignments_window']
    window[0][0]-=1;window[0][128]+=1
    result=analyze(plan,rows)
    assert not result['valid']
    assert any('native_shortcut_physical_windows' in e.get('issues',[]) for e in result['errors'])


def test_unbalanced_plan_rejected_even_if_rows_match_declared_orders():
    plan,rows=fixture()
    for case in plan['cases']:case['arm_order']=list(ARMS)
    plan['freeze']['sha256']=canonical_hash({k:v for k,v in plan.items() if k!='freeze'})
    for row in rows:
        row['arm_order']=list(ARMS);row['config_hash']=plan['freeze']['sha256']
    with pytest.raises(ValueError,match='six arm permutations'):
        analyze(plan,rows)

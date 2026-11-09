import copy
import itertools
import math
import numpy as np
import pytest
from arrival_study.schedules import fcfs_metrics
from stock_comparison.analysis import clock_metrics, paired_values, validate_plan, validate_async_lifecycle
from stock_comparison.common import ARMS, scale_calendar, seal


def clocks():
    n=4096;epoch=10**12
    arrivals=[20*i for i in range(n)]
    starts=[epoch+x+2 for x in arrivals];finishes=[s+10 for s in starts]
    q=finishes[-1]+80;end=q+1
    row=dict(n_jobs=n,arrival_offsets_ns=arrivals,horizon_ns=20*n,
        epoch_monotonic_ns=epoch,observation_end_ns=end,
        coordinator_admit_ns=[s-1 for s in starts],coordinator_start_ns=starts,coordinator_finish_ns=finishes,
        per_rank_start_ns=[[s+1 for s in starts],[s+2 for s in starts]],
        per_rank_gpu_complete_ns=[[f-2 for f in finishes],[f-1 for f in finishes]],
        controller_quiescence_start_ns=finishes[-1]+5,controller_quiescent_ns=q,
        per_rank_controller_tail_start_ns=[finishes[-1]+5,finishes[-1]+6],
        per_rank_controller_quiescent_ns=[q,q+3],completed_elapsed_ns=end-epoch,
        job_completed_elapsed_ns=finishes[-1]-epoch,controller_tail_elapsed_ns=80,
        quiescence_call_elapsed_ns=75,drain_including_controller_ns=q-epoch-20*n)
    row['queue_metrics']=fcfs_metrics(row)
    return row


def plan(split='pilot'):
    n=2 if split=='pilot' else 6;banks=[];cases=[];perms=list(itertools.permutations(ARMS))
    for di,dataset in enumerate(('gsm8k','mbpp','wildchat')):
        ids=[f'{dataset}-{i}' for i in range(n)]
        banks.extend(dict(block_id=g,dataset=dataset,split=split) for g in ids)
        for li,rho in enumerate((.5,.85)):
            j=di*4 if split=='pilot' else 0
            for gid,service in itertools.product(ids,('code','conversation')):
                source=dict(id=f'{gid}/{service}',trace=service,source_date_utc='2024-05-14',
                    source_rows_sha256='a'*64,jobs=4096,original_offsets_ns=list(range(4096)),original_horizon_ns=4096)
                cases.append(dict(block_id=gid,dataset=dataset,source_block=source,
                    calendar=scale_calendar(source,81920,rho=rho),arm_order=list(perms[(j+3*li)%6])))
                j+=1
    return seal(dict(schema='stock_comparison_plan_v1',phase=split,split=split,
        arms=list(ARMS),jobs=4096,batch_tokens=128,banks=banks,cases=cases,
        hardware={'allocation_id':'1'},source_sha256='a'*64,
        calibration_record_ids={b['block_id']:'b'*64 for b in banks},
        calibration_provenance={b['block_id']:dict(calibration_elapsed_ns=81920,allocation_id='1',
            source_sha256='a'*64,record_id='b'*64) for b in banks}))


def test_tail_charged_without_altering_job_response():
    row=clocks();m=clock_metrics(row)
    assert m['response_mean_ns']==12
    assert m['service_mean_ns']==10
    assert m['controller_tail_elapsed_ns']==80
    assert m['completed_elapsed_ns']>m['job_completed_elapsed_ns']
    assert m['drain_ns']==0 and m['drain_including_controller_ns']>0
    assert m['throughput_jobs_per_second']==4096*1e9/m['completed_elapsed_ns']
    # Shared barrier host-return skew does not turn rank1's later host timestamp
    # into additional rank0 observation work or an invalid GPU completion.
    assert row['per_rank_controller_quiescent_ns'][1]>row['observation_end_ns']


@pytest.mark.parametrize('field',('completed_elapsed_ns','job_completed_elapsed_ns','controller_tail_elapsed_ns','quiescence_call_elapsed_ns','drain_including_controller_ns'))
def test_tampered_tail_or_completed_extent_rejected(field):
    row=clocks();row[field]+=1
    with pytest.raises(ValueError,match='accounting'):clock_metrics(row)


def test_rank_completion_outside_service_rejected():
    row=clocks();row['per_rank_gpu_complete_ns'][1][100]=row['coordinator_finish_ns'][100]+1
    with pytest.raises(ValueError,match='outside'):clock_metrics(row)


def test_precomputed_summary_cannot_hide_service_spike():
    row=clocks();row['coordinator_finish_ns'][100]+=1
    with pytest.raises(ValueError,match='Stored metrics'):clock_metrics(row)


@pytest.mark.parametrize('split',('pilot','final'))
def test_declared_counterbalance_matches_frozen_design(split):
    p=plan(split);groups,cases=validate_plan(p)
    assert len(groups)==(6 if split=='pilot' else 18)
    assert len(cases)==4*len(groups)


def test_globally_balanced_but_load_confounded_order_rejected():
    p=plan();perms=list(itertools.permutations(ARMS))
    for i,c in enumerate(p['cases']):c['arm_order']=list(perms[2*(i//2)%6+(c['calendar']['rho']==.85)])
    p=seal(p)
    with pytest.raises(ValueError,match='balanced'):validate_plan(p)


def test_mutated_calendar_rejected_even_if_outer_plan_resealed():
    p=plan();p['cases'][0]['calendar']['arrival_offsets_ns'][1]+=1;p=seal(p)
    with pytest.raises(ValueError,match='calendar'):validate_plan(p)


def test_zero_queue_or_drain_has_no_fabricated_log_ratio():
    base={'response_mean_ns':2.,'response_p95_ns':4.,'service_mean_ns':1.,'completed_elapsed_ns':8.,'queue_mean_ns':0.,'drain_ns':0.}
    target={**base,'response_mean_ns':1.8,'queue_mean_ns':1.}
    out=paired_values(base,target)
    assert out['ratios']['queue_mean_ns'] is None
    assert 'queue_mean_ns' not in out['log_ratios']
    assert out['differences']['queue_mean_ns']==1
    assert math.isclose(math.exp(out['log_ratios']['response_mean_ns']),.9)


def test_slow_native_pending_work_is_preserved_not_excluded():
    initial=list(range(128))+list(range(32));recorded=list(range(750))+list(range(2750,4096))
    state=dict(source_map=initial,final_map=initial,recorded_step_indices=recorded,
        deferred_boundary_steps=list(range(3749,4096)),native_counter=3346,native_valid_steps=len(recorded),
        migration_sent_bytes_local=128,
        async_events=[dict(launch_step=749,launch_counter_before=2999,commit_step=4095,
            commit_in_completion_tail=True,source_map=initial,committed_map=initial,
            window_valid_steps=750,sampled_step_indices=list(range(750)),sent_bytes_local=128)])
    validate_async_lifecycle(state)
    bad=copy.deepcopy(state);bad['deferred_boundary_steps'].pop()
    with pytest.raises(ValueError,match='deferral'):validate_async_lifecycle(bad)

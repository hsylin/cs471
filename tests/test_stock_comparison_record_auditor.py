"""Independent auditor mutation tests; no production analysis imported."""
import copy
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('stock_raw_audit',Path(__file__).resolve().parents[1]/'scripts/audit_stock_comparison_records.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def clock_fixture():
    n=4096;e=10**12;offset=np.arange(n,dtype=np.int64)*20
    a=e+offset;s=a+2;f=s+10;q=int(f[-1])+80;end=q+1
    vectors=dict(response=f-a,queue_wait=s-a,service=f-s,admission_delay=s-1-a,dispatch_delay=np.ones(n,dtype=np.int64))
    summaries={name:dict(mean=float(v.mean()),p50=float(np.quantile(v,.5)),p95=float(np.quantile(v,.95)),p99=float(np.quantile(v,.99)),max=int(v.max())) for name,v in vectors.items()}
    return dict(n_jobs=n,epoch_monotonic_ns=e,horizon_ns=n*20,observation_end_ns=end,
        arrival_offsets_ns=offset.tolist(),coordinator_admit_ns=(s-1).tolist(),coordinator_start_ns=s.tolist(),coordinator_finish_ns=f.tolist(),
        per_rank_start_ns=[(s+1).tolist(),(s+2).tolist()],per_rank_gpu_complete_ns=[(f-2).tolist(),(f-1).tolist()],
        controller_quiescent_ns=q,controller_quiescence_start_ns=int(f[-1])+5,
        per_rank_controller_tail_start_ns=[int(f[-1])+5,int(f[-1])+6],per_rank_controller_quiescent_ns=[q,q+3],
        completed_elapsed_ns=end-e,job_completed_elapsed_ns=int(f[-1])-e,controller_tail_elapsed_ns=80,
        quiescence_call_elapsed_ns=75,drain_including_controller_ns=q-e-n*20,
        queue_metrics=dict(complete=True,completed_jobs=n,unfinished_jobs=0,
            vectors_ns={k:v.tolist() for k,v in vectors.items()},summary_ns=summaries,
            queue_occupancy={'integral_job_ns':int(vectors['queue_wait'].sum())},system_occupancy={'integral_job_ns':int(vectors['response'].sum())},
            completed_by_horizon=n,backlog_at_horizon=0,drain_ns=0,completed_per_evaluation_second=n*1e9/(end-e)))


def test_independent_clock_keeps_controller_tail_separate():
    flat=audit.raw_clock_check(clock_fixture())
    assert flat['response_mean_ns']==12 and flat['service_mean_ns']==10
    assert flat['controller_tail_elapsed_ns']==80 and flat['drain_ns']==0
    assert flat['drain_including_controller_ns']>0


@pytest.mark.parametrize('mutation',('tail','omission','cached_percentile'))
def test_independent_clock_rejects_hidden_work(mutation):
    row=clock_fixture()
    if mutation=='tail':row['completed_elapsed_ns']=row['job_completed_elapsed_ns']
    elif mutation=='omission':row['coordinator_finish_ns'].pop()
    else:row['queue_metrics']['summary_ns']['response']['p95']=11
    with pytest.raises(ValueError):audit.raw_clock_check(row)


def test_float_bytes_identity_distinguishes_fractional_weights():
    x=np.array([.25,.75],dtype='<f4');y=np.array([.5,.5],dtype='<f4')
    assert audit.tensor_sha(x)!=audit.tensor_sha(y)
    assert audit.tensor_sha(x)!=audit.tensor_sha(x.astype('<f8'))

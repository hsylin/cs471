"""Read-only accounting replays of archived clocks; never predicted GPU timings."""
from pathlib import Path
import gzip,json,hashlib,itertools,statistics
import numpy as np
import argparse
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--records',type=Path,default=ROOT/'results/instructor/arrivals-v1/evaluation/records.jsonl.gz')
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
BOUNDARIES=np.array([63,127,191,255]); ORDINARY=np.array([i for i in range(256) if i not in BOUNDARIES])

def replay(arrivals,service,gap):
    previous=0; result=[]
    for arrival,duration,extra in zip(arrivals,service,gap):
        previous=max(int(arrival),previous)+int(extra)+int(duration)
        result.append(previous)
    return np.array(result,dtype=np.int64)

def q(v,p):return float(np.quantile(v,p,method='linear'))
def m(v):return float(np.mean(v))/1e6

def diagnose(row):
    origin=row['epoch_monotonic_ns']; a=np.array(row['arrival_offsets_ns'],dtype=np.int64)
    e=np.array(row['coordinator_admit_ns'],dtype=np.int64)-origin
    s=np.array(row['coordinator_start_ns'],dtype=np.int64)-origin
    f=np.array(row['coordinator_finish_ns'],dtype=np.int64)-origin
    ready=np.maximum(a,np.r_[0,f[:-1]])
    d=f-s; ga=e-ready; gd=s-e; g=ga+gd
    assert (ga>=0).all() and (gd>=0).all()
    assert np.array_equal(replay(a,d,g),f)
    assert np.array_equal((ready-a)+ga+gd+d,f-a)
    median=int(round(float(np.median(d[ORDINARY]))))
    capped=d.copy(); capped[BOUNDARIES]=np.minimum(capped[BOUNDARIES],median)
    variants={'observed':f,'zero_gaps':replay(a,d,np.zeros(256,dtype=np.int64)),
      'boundary_capped':replay(a,capped,g),'boundary_capped_zero_gaps':replay(a,capped,np.zeros(256,dtype=np.int64)),
      'zero_admission_gap':replay(a,d,gd),'zero_dispatch_gap':replay(a,d,ga)}
    first=g.copy();first[0]=0;variants['zero_first_start_gap']=replay(a,d,first)
    means={k:m(v-a) for k,v in variants.items()}
    response=f-a
    top_service=sorted(ORDINARY.tolist(),key=lambda i:int(d[i]),reverse=True)[:5]
    top_gap=sorted(range(256),key=lambda i:int(g[i]),reverse=True)[:5]
    return {**{k:row[k] for k in ('block_id','regime','batch_tokens','schedule_name','repetition','arm')},
      'pattern':row['arrival_schedule']['pattern'],'rho':row['arrival_schedule']['rho'],
      'response_mean_ms':m(response),'response_p95_ms':q(response,.95)/1e6,
      'service_mean_ms':m(d),'ordinary_service_mean_ms':m(d[ORDINARY]),'ordinary_service_median_ms':median/1e6,
      'boundary_service_ms':{str(i):int(d[i])/1e6 for i in BOUNDARIES},
      'boundary_service_total_ms':float(d[BOUNDARIES].sum())/1e6,
      'boundary_service_fraction':float(d[BOUNDARIES].sum()/d.sum()),
      'boundary_excess_total_ms':float((d-capped).sum())/1e6,
      'mean_direct_queue_carried_ms':m(ready-a),'mean_direct_admission_gap_ms':m(ga),'mean_direct_dispatch_gap_ms':m(gd),
      'mean_direct_gap_ms':m(g),'max_gap_ms':int(g.max())/1e6,'gap_p95_ms':q(g,.95)/1e6,
      'first_job_admission_lateness_ms':int(ga[0])/1e6,'first_job_start_gap_ms':int(g[0])/1e6,
      'busy_arrival_jobs':int(np.count_nonzero(np.r_[0,f[:-1]]>a)),
      'idle_ready_admission_gap_mean_ms':m(ga[np.r_[0,f[:-1]]<=a]),
      'busy_ready_admission_gap_mean_ms':m(ga[np.r_[0,f[:-1]]>a]) if (np.r_[0,f[:-1]]>a).any() else None,
      'replay_response_mean_ms':means,
      'gap_effect_with_observed_service_ms':means['observed']-means['zero_gaps'],
      'gap_effect_after_boundary_capping_ms':means['boundary_capped']-means['boundary_capped_zero_gaps'],
      'boundary_effect_with_observed_gaps_ms':means['observed']-means['boundary_capped'],
      'boundary_effect_after_gap_removal_ms':means['zero_gaps']-means['boundary_capped_zero_gaps'],
      'gap_boundary_interaction_ms':means['observed']-means['zero_gaps']-means['boundary_capped']+means['boundary_capped_zero_gaps'],
      'first_start_gap_effect_ms':means['observed']-means['zero_first_start_gap'],
      'nonboundary_service_over_2ms_jobs':int(np.count_nonzero(d[ORDINARY]>2_000_000)),
      'largest_nonboundary_service':[dict(step=i,service_ms=int(d[i])/1e6,response_ms=int(response[i])/1e6,gap_ms=int(g[i])/1e6) for i in top_service],
      'largest_start_gaps':[dict(step=i,gap_ms=int(g[i])/1e6,admission_gap_ms=int(ga[i])/1e6,dispatch_gap_ms=int(gd[i])/1e6) for i in top_gap]}

path=args.records
with gzip.open(path,'rt') as handle: rows=[json.loads(line) for line in handle if line.strip()]
reports=[diagnose(row) for row in rows]
metrics=['response_mean_ms','response_p95_ms','service_mean_ms','ordinary_service_mean_ms','boundary_service_total_ms','boundary_service_fraction','boundary_excess_total_ms',
 'mean_direct_queue_carried_ms','mean_direct_admission_gap_ms','mean_direct_dispatch_gap_ms','mean_direct_gap_ms',
 'gap_effect_with_observed_service_ms','gap_effect_after_boundary_capping_ms','boundary_effect_with_observed_gaps_ms','boundary_effect_after_gap_removal_ms','gap_boundary_interaction_ms','first_start_gap_effect_ms']
groups=[]
for batch,pattern,rho,arm in itertools.product((128,1024),('constant','burst8'),(.5,.85),('native','online')):
    selected=[r for r in reports if (r['batch_tokens'],r['pattern'],r['rho'],r['arm'])==(batch,pattern,rho,arm)]
    groups.append(dict(batch_tokens=batch,pattern=pattern,rho=rho,arm=arm,trials=len(selected),
      **{key:statistics.mean(r[key] for r in selected) for key in metrics},
      replay_response_mean_ms={key:statistics.mean(r['replay_response_mean_ms'][key] for r in selected) for key in selected[0]['replay_response_mean_ms']}))
result=dict(archive_records_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),raw_rows=len(rows),exact_recurrence_checks_passed=len(reports),
  definition='f[i]=max(a[i],f[i-1])+admission_residual[i]+dispatch_delay[i]+service[i]; all in recorded integer nanoseconds',
  caveat='Conditional accounting replays freeze measured service and/or residual gaps under a changed queue. They are not causal performance estimates; removing boundary excess is an optimistic surrogate, not an implemented path.',
  boundary_capping_rule='At indices63,127,191,255 cap service to the rounded median of the other252 durations in the same trial. Keep arrivals and ordering unchanged.',
  groups=groups,trials=reports)
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(result,indent=2)+'\n')
print('Exact recurrence',len(reports))
for r in groups:
 print(r['batch_tokens'],r['pattern'],r['rho'],r['arm'], {k:round(r[k],5) for k in ('response_mean_ms','service_mean_ms','mean_direct_gap_ms','gap_effect_with_observed_service_ms','boundary_effect_after_gap_removal_ms')},'replays',{k:round(v,5) for k,v in r['replay_response_mean_ms'].items()})

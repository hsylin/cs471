"""Post-measurement accounting; never changes or filters the frozen experiment."""
import argparse
import hashlib
import json
from pathlib import Path


def diagnose(report_path, run):
    report_path,run=Path(report_path),Path(run)
    report=json.loads(report_path.read_bytes());assert report['status']=='completed_and_audited'
    output={'status':'descriptive_post_measurement_diagnostics','report_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest(),
        'scope':'Native request RUNNING/WAITING intervals are not GPU service time. Their residual against HTTP response includes other boundaries and is not an identified CPU, network, or scheduler cost. No run/request is excluded.', 'datasets':[]}
    for dataset in report['datasets']:
        name=dataset['dataset'];result={'dataset':name,'native_interval_accounting':{},'per_request_vs_off':{},'gate_cycles':{}}
        records={}
        for arm,data in dataset['arms'].items():
            series=data['native_metrics']['series'];means={}
            for metric in sorted({r['metric'] for r in series}):
                rows=[r for r in series if r['metric']==metric];assert all(r['available'] for r in rows)
                count=sum(r['delta_count'] for r in rows);assert count==16
                means[metric]=1000*sum(r['delta_sum_seconds'] for r in rows)/count
            response=data['metrics']['mean_response_ms'];running=means['vllm:request_inference_time_seconds'];waiting=means['vllm:request_queue_time_seconds']
            result['native_interval_accounting'][arm]={'mean_response_ms':response,'native_mean_ms':means,'residual_response_minus_running_minus_waiting_ms':response-running-waiting}
            raw=[json.loads(l) for l in (run/name/arm/'measured.jsonl').read_text().splitlines()]
            records[arm]=sorted([r for r in raw if r['type']=='request'],key=lambda r:r['request_index'])
        for arm in ('resident_v1','resident_v2'):
            result['per_request_vs_off'][arm]=[
                {'request_index':a['request_index'],'request_id':a['request_id'],'input_tokens':a['expected_input_tokens'],
                 'response_difference_ms':((a['end_ns']-a['scheduled_ns'])-(b['end_ns']-b['scheduled_ns']))/1e6,
                 'ttft_difference_ms':((a['first_content_ns']-a['scheduled_ns'])-(b['first_content_ns']-b['scheduled_ns']))/1e6,
                 'send_lag_ms':(a['send_ns']-a['scheduled_ns'])/1e6}
                for a,b in zip(records[arm],records['off'])]
        for arm in ('b3','b4'):
            ranks=[[json.loads(l) for l in (run/name/arm/f'gate-events-rank{rank}.jsonl').read_text().splitlines()] for rank in range(8)]
            cycles=[]
            for decision in [r for r in ranks[0] if r['kind']=='decision']:
                step=decision['step'];enqueued=[{r['layer'] for r in rows if r['kind']=='commit_enqueued' and r['cycle_step']==step} for rows in ranks]
                completed=[{r['layer'] for r in rows if r['kind']=='commit_completed' and r['cycle_step']==step} for rows in ranks]
                cycles.append({'step':step,'allowed_layers':sum(d['allow'] for d in decision['layers']),
                    'all_rank_layer_enqueues':len(set.intersection(*enqueued)),
                    'all_rank_layer_GPU_completions_observed':len(set.intersection(*completed)),
                    'rank0_layers_without_commit_enqueue':sorted(set(range(48))-enqueued[0]),
                    'rank0_enqueued_without_observed_completion':sorted(enqueued[0]-completed[0]),
                    'remote_received_tensor_bytes_all_ranks':sum(r['remote_received_tensor_bytes'] for rows in ranks for r in rows if r['kind']=='commit_enqueued' and r['cycle_step']==step)})
            result['gate_cycles'][arm]=cycles
        output['datasets'].append(result)
    return output


def main():
    p=argparse.ArgumentParser()
    for name in ('report','run','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();result=diagnose(a.report,a.run);a.output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()

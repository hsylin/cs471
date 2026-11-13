"""Read-only diagnostic boundaries; timing instrumentation is not a final arm."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(root):
    result={'status':'diagnostic_only','sessions':{},'limitations':[
        'Instrumentation adds overhead; use uninstrumented sessions for final speed comparisons.',
        'First engine output is not guaranteed to contain the first text token.',
        'API boundaries share one process clock; client-to-API differences require same-host validation.',
        'Event-loop lag identifies a stall, not its CPU/GIL/network cause.',
    ]}
    root=Path(root)
    for measured in sorted(root.glob('*/*/measured.jsonl')):
        folder=measured.parent
        if not (folder/'status.json').exists():continue
        records=[json.loads(l) for l in measured.read_text().splitlines()]
        requests=[r for r in records if r['type']=='request']
        probes={}
        for path in folder.glob('latency-api-*.jsonl'):
            for line in path.read_text().splitlines():
                row=json.loads(line)
                first=[e for e in row['events'] if e['event']=='generate.first_output']
                if first:
                    rid=first[0]['engine_request_id']
                    assert rid not in probes
                    probes[rid]=row
        rows=[]
        spans={
            'render_ms':('OpenAIServingChat.render_chat_request.begin','OpenAIServingChat.render_chat_request.end'),
            'pre_render_ms':('http.received','OpenAIServingChat.render_chat_request.begin'),
            'render_to_submit_ms':('OpenAIServingChat.render_chat_request.end','DPAsyncMPClient.add_request_async.begin'),
            'submit_ms':('DPAsyncMPClient.add_request_async.begin','DPAsyncMPClient.add_request_async.end'),
            'submit_to_engine_output_ms':('DPAsyncMPClient.add_request_async.end','generate.first_output'),
            'api_to_engine_output_ms':('http.received','generate.first_output'),
        }
        for req in requests:
            p=probes[req['server_request_id']]
            events={e['event']:e for e in p['events']}
            row={'request_index':req['request_index'],'server_request_id':req['server_request_id'],
                 'host':p['host'],'engine_index':events['dp_selection']['engine_index'],
                 'response_ms':(req['end_ns']-req['scheduled_ns'])/1e6,
                 'ttft_ms':(req['first_content_ns']-req['scheduled_ns'])/1e6,
                 'event_loop_max_lag_ms':max((x['late_ns']/1e6 for x in p['event_loop_lag']),default=0)}
            for key,(a,b) in spans.items():
                row[key]=(events[b]['mono_ns']-events[a]['mono_ns'])/1e6
                assert row[key]>=0,(key,row)
            rows.append(row)
        assert len(rows)==16 and all(r['success'] for r in requests)
        result['sessions'][str(folder.relative_to(root))]={
            'requests':rows,'mean':{k:statistics.mean(r[k] for r in rows) for k in [*spans,'response_ms','ttft_ms','event_loop_max_lag_ms']}}
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.write_text(json.dumps(summarize(a.root),indent=2)+'\n')
if __name__=='__main__':main()

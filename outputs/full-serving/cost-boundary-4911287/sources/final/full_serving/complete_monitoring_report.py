"""Close the eighteen-session run only after stable byte and source verification."""
import argparse
import csv
import hashlib
import html
import io
import json
from pathlib import Path
import subprocess
from full_serving.monitoring_analysis import final_report


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def complete(run,wrapper,inventory,repo,output,audits):
    run,wrapper,inventory,repo,output,audits=map(Path,(run,wrapper,inventory,repo,output,audits))
    assert (wrapper/'completed.txt').read_text()=='controller_exit=0\n'
    manifest=json.loads(inventory.read_bytes());assert manifest['before']==manifest['after']
    expected=manifest['after'];actual={'results/'+str(p.relative_to(run)):p for p in run.rglob('*') if p.is_file()}
    actual.update({n:wrapper/n for n in ('controller.log','controller.err','completed.txt')});assert actual.keys()==expected.keys()
    for name,p in actual.items():assert p.stat().st_size==expected[name]['bytes'] and digest(p)==expected[name]['sha256'],name
    source=json.loads((run/'controller-source.json').read_bytes());revision=source['declared_git_revision'];assert len(revision)==40
    for name,sha in source['files'].items():
        raw=subprocess.run(['git','show',revision+':'+name],cwd=repo,check=True,capture_output=True).stdout
        assert hashlib.sha256(raw).hexdigest()==sha,name
    spec=json.loads((run/'protocol.json').read_bytes())
    from full_serving.custom_compare import canonical_hash
    assert spec['seal']==canonical_hash({k:v for k,v in spec.items() if k!='seal'}) and spec['status']=='frozen'
    protocol_name=spec['protocol_repository_path']
    assert not Path(protocol_name).is_absolute() and '..' not in Path(protocol_name).parts
    original_protocol=subprocess.run(['git','show',revision+':'+protocol_name],cwd=repo,check=True,capture_output=True).stdout
    assert original_protocol==(run/'protocol.json').read_bytes(),'Frozen protocol differs from declared Git source'
    admission=subprocess.run(['git','show',revision+':'+spec['pilot_admission']],cwd=repo,check=True,capture_output=True).stdout
    assert hashlib.sha256(admission).hexdigest()==spec['pilot_admission_sha256']
    for name,sha in spec['files'].items():
        raw=subprocess.run(['git','show',revision+':'+name],cwd=repo,check=True,capture_output=True).stdout
        assert hashlib.sha256(raw).hexdigest()==sha,name
    report=final_report(run)
    output.parent.mkdir(parents=True,exist_ok=True);audits.mkdir(parents=True,exist_ok=True)
    target=output.with_suffix('.json');target.write_text(json.dumps(report,indent=2)+'\n')
    rows=[]
    for d in report['datasets']:
        for arm,a in d['arms'].items():
            rows.append({'dataset':d['dataset'],'arm':arm,**a['metrics'],'output_tokens':a['summary']['completed_output_tokens'],'prefill_fraction':a['phase_balance']['prefill_fraction']})
    csvpath=output.with_suffix('.csv')
    with csvpath.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    cols=('dataset','arm','mean_response_ms','p95_response_ms','mean_ttft_schedule_ms','completed_elapsed_s','drain_s','output_tokens')
    body=['<!doctype html><html lang="en"><meta charset="utf-8"><title>Proposal and resident follow-up results</title><style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:20px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}</style><h1>Equal-output full-model comparison</h1><p>One group per dataset; descriptive ratios only. Startup and warmup excluded from serving timing. Equal output lengths do not prove identical tokens, routing or quality.</p><table><tr>']
    body+=['<th>'+html.escape(c)+'</th>' for c in cols];body+=['</tr>']
    for row in rows:
        body+=['<tr>']
        for c in cols:
            value=row[c];body+=['<td>'+html.escape(f'{value:.3f}' if isinstance(value,float) else str(value))+'</td>']
        body+=['</tr>']
    body+=['</table>']
    for d in report['datasets']:
        body+=['<h2>'+html.escape(d['dataset'])+'</h2>']
        for name,v in d['contrasts'].items():body+=['<p>'+html.escape(name)+f": mean response {(v['ratios']['mean_response_ms']-1)*100:+.2f}%, p95 {(v['ratios']['p95_response_ms']-1)*100:+.2f}%, elapsed {(v['ratios']['completed_elapsed_s']-1)*100:+.2f}%. Identical output hashes {v['equal_output_hashes']}/16.</p>"]
        for arm in ('b3','b4','b3_light','b4_light'):body+=['<pre>'+html.escape(json.dumps({arm:d['arms'][arm]['gate']},indent=2))+'</pre>']
    body+=['<p>Recorded CUDA-event completion is distinct from drain. It does not establish full-weight or all-logit correctness. No automatic superiority or capacity claim.</p></html>']
    hp=output.with_suffix('.html');hp.write_text(''.join(body)+'\n')
    receipt={'status':'completed_and_audited','sessions':18,'measured_requests':288,'experiment_revision':revision,'verified_raw_files':len(actual),'verified_controller_source_files':len(source['files']),'verified_protocol_source_files':len(spec['files']),'independent_raw_arithmetic':'passed all sessions','artifacts':{str(p):digest(p) for p in (target,csvpath,hp)},'normal_native_b1_b2_commits':'not proven by native logs','automatic_superiority_claim':False}
    receipt['analysis_files_sha256']={name:digest(repo/name) for name in
        ('full_serving/monitoring_analysis.py','full_serving/complete_monitoring_report.py','full_serving/audit_final_statistics.py')}
    # Receipt last, and never silently overwrite an earlier closure.
    with (audits/'completion.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    return receipt


def main():
    p=argparse.ArgumentParser()
    for name in ('run','wrapper','inventory','repo','output','audits'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(complete(a.run,a.wrapper,a.inventory,a.repo,a.output,a.audits)))
if __name__=='__main__':main()

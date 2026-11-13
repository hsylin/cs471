"""Freeze the fourteen-session run only after all four pilot servers pass.

No rates, prompts, targets, thresholds, orders or algorithms are selected here.
A correct negative pilot is allowed. No fastest attempt or output subset selection.
"""
import argparse
import json
from pathlib import Path
from full_serving.proposal_experiment import load
from full_serving.proposal_analysis import runtime,gate_audit
from full_serving.report_pilots import read_phase
from full_serving.custom_compare import canonical_hash
from full_serving.orchestrate import sha


def freeze(pilot_root,development_protocol,output):
    root=Path(pilot_root);development_protocol=Path(development_protocol);output=Path(output)
    spec,data,data_path=load(development_protocol)
    assert spec['status']=='development'
    assert (root/'completed.txt').read_text()=='pilot_sequence_exit=0\n'
    pilots={}
    for arm in ('resident_v1','resident_v2','b3','b4'):
        run=root/arm;p=run/'pilot'/arm
        assert (root/f'{arm}-completed.txt').read_text()=='controller_exit=0\n'
        status=json.loads((run/'status.json').read_bytes());assert status['state']=='comparison_completed' and status['pilot_only'] is True
        assert status['completed_sessions']==[f'pilot/{arm}']
        identity=runtime(p,arm);records=[]
        for dataset in data['datasets']:
            name=dataset['dataset'];warm=dataset['warmup'];targets=json.loads((run/'output-lengths'/f'{name}-warmup.json').read_bytes())
            assert sha(run/'output-lengths'/f'{name}-warmup.json')==warm['output_lengths_sha256']
            for phase in ('warmup-00','warmup-01','profile'):
                item,rows=read_phase(p/f'{name}-{phase}.jsonl');assert item['valid'] and len(rows)==8
                h=item['header'];assert h['requests_sha256']==warm['requests_sha256'] and h['ignore_eos'] is True and h['output_lengths_sha256']==warm['output_lengths_sha256']
                assert [r['request_id'] for r in rows]==warm['request_ids']
                assert all(r['requested_output_tokens']==targets[r['request_id']]==r['usage']['completion_tokens'] and r['finish_reason']=='length' for r in rows)
                assert h['logprobs'] is (phase=='warmup-00')
                records.append({'dataset':name,'phase':phase,'sha256':sha(p/f'{name}-{phase}.jsonl'),'completed':8})
        evidence={'status':'passed','runtime':identity,'phases':records,'final_payload_opened':False}
        if arm.startswith('resident'):
            for rank in range(8):
                d=json.loads((p/f'resident-startup-diagnostic-rank{rank}.json').read_bytes())
                assert d['status']=='passed' and d['ep_rank']==rank and d['weight_check']['full_bf16_replica_hashes_equal']
                assert len(d['native_expert_numerical_checks'])==9
            profile=root/f'{arm}-profile-summary.json';stats=json.loads(profile.read_bytes())
            assert len(stats['files'])>=8 and all(x['kernel_events']>0 for x in stats['files'])
            assert any(x['mapper_kernel_us']>0 for x in stats['files'])
            evidence['profile_summary_sha256']=sha(profile)
            evidence['representative_numerical_and_all_replica_weight_checks']='passed; not all-logit equivalence'
        else:
            evidence['gate']=gate_audit(p,arm)
            assert evidence['gate']['proposal_cycles']>=1,'Pilot never exercised native async gate'
        pilots[arm]=evidence
    admission={'status':'passed','source_files':spec['files'],'pilots':pilots,'scope':'Correctness/feasibility pilot, no requirement to beat OFF; full-logit equivalence and long-horizon payback not established.'}
    output.mkdir(parents=True,exist_ok=False);ap=output/'pilot-admission.json';ap.write_text(json.dumps(admission,indent=2)+'\n')
    from full_serving.proposal_experiment import ROOT
    spec.update(status='frozen',pilot_admission=str(ap.resolve().relative_to(ROOT)),pilot_admission_sha256=sha(ap))
    spec.pop('seal');spec['seal']=canonical_hash(spec);protocol=output/'protocol.json';protocol.write_text(json.dumps(spec,indent=2,sort_keys=True)+'\n')
    load(protocol)
    return protocol


def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot-root',type=Path,required=True);p.add_argument('--development-protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(freeze(a.pilot_root,a.development_protocol,a.output))
if __name__=='__main__':main()

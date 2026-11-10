"""Freeze the supplement before any supplemental GPU outcome is measured."""
import argparse,json,itertools
from pathlib import Path
from sequence_study.common import source_identity,ARMS
from stock_comparison.common import seal
from stock_comparison.data import digest
from followup.router_tapes import file_sha256

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    from resident_residual.common import load_plan
    old=load_plan('configs/resident-residual-final-v1.json')
    banks=[]
    for dataset in ('gsm8k_train','wildchat'):
        chosen=sorted((b for b in old['banks'] if b['dataset']==dataset),key=lambda b:digest([471901,b['block_id']]))[:2]
        banks.extend(chosen)
    trials=[];calendars={};perms=list(itertools.permutations(ARMS))
    for i,b in enumerate(banks):
        gid=b['block_id'];kind=('code','conversation')[i%2]
        calendars[gid]=next(c for c in old['cases'] if c['block_id']==gid and c['arrival_type']==kind and c['rho']==.85)
        modes=['contiguous','mixed']+(['batch_shuffle'] if i==0 else [])
        for rep in (0,1):
            order=modes if (i+rep)%2==0 else list(reversed(modes))
            for mode in order:
                armorder=list(perms[(i+modes.index(mode))%6])
                if rep:armorder.reverse()
                for arm in armorder:trials.append(dict(block_id=gid,mode=mode,rep=rep,arm=arm,arm_order=armorder))
    sha,files=source_identity()
    plan=seal(dict(schema='sequence_study_plan_v1',phase='supplement',arms=list(ARMS),jobs=4096,batch_tokens=128,
        seed=471901,input_seed=471902,source_sha256=sha,source_files=files,hardware=old['hardware'],banks=banks,trials=trials,
        calendars=calendars,parent_plan_sha256=old['content_sha256'],selected_fix=old['selected_fix'],
        protocol_sha256=file_sha256('docs/sequence-study/protocol-v1.md'),
        timing_selection=False,independent_content_units=4,technical_repetitions=2,
        evidence_scope='New ordering of reused captured final content and reused absolute calendars; separate descriptive sensitivity study, not fresh-content confirmation.'))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(sha=plan['content_sha256'],source=sha,banks=[b['block_id'] for b in banks],trials=len(trials))))

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Freeze explicit new-content calibrations and paired external-arrival cases."""
from pathlib import Path
import argparse,gzip,itertools,json,math,random,sys
from fractions import Fraction
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from generalization.common import source_identity, ARMS, ROOT, validate_calendar, load_plan
from generalization.data import scaled_calendar
from generalization.inference import assign_calendars,select_content_groups
from arrival_fix.common import GPU_UUIDS,HARDWARE_HOST,file_hash
from optimization.common import canonical_hash,manifest_hash


def rows(path):
    op=gzip.open if str(path).endswith('.gz') else open
    with op(path,'rt') as f:return [json.loads(line) for line in f if line.strip()]


def calibration_table(records,expected,calibration_plan,content_sha,execution_sha):
    if calibration_plan['phase']!='calibration' or calibration_plan['content_selection_sha256']!=content_sha or calibration_plan['execution_source_sha256']!=execution_sha:
        raise ValueError('Stale or ineligible calibration plan')
    bank_sha={b['block_id']:b['capture_sha256'] for b in calibration_plan['banks']}
    if set(bank_sha)!=set(expected):raise ValueError('Calibration plan inventory changed')
    result={}
    for r in records:
        if r['phase']!='calibration' or r['arm']!='native' or r['block_id'] in result:
            raise ValueError('One native calibration per content group required')
        if (r['status']!='complete' or not r['correctness_pass'] or not r['resident_state_verified'] or not r['normal_completion_verified'] or r['migration_bytes']):
            raise ValueError('Invalid native calibration')
        if str(r['allocation_id'])!='4872248':raise ValueError('Calibration allocation changed')
        if (r.get('config_hash')!=manifest_hash(calibration_plan) or r.get('execution_source_sha256')!=execution_sha
                or r.get('content_selection_sha256')!=content_sha or r.get('capture_sha256')!=bank_sha.get(r['block_id'])
                or r.get('batch_tokens')!=128 or r.get('policy')!='SKIP'):
            raise ValueError('Calibration content/capture/source/plan identity changed')
        c=r['calibration_capacity']
        starts,finishes=r['coordinator_start_ns'],r['coordinator_finish_ns']
        if len(starts)!=256 or len(finishes)!=256 or c['calibration_elapsed_ns']!=finishes[-1]-starts[0]:
            raise ValueError('Calibration elapsed boundary changed')
        if c['native_mean_cycle_ns']!=c['calibration_elapsed_ns']/256:
            raise ValueError('Calibration capacity differs from raw clocks')
        if c['calibration_jobs']!=256 or c['calibration_elapsed_ns']<=0:raise ValueError('Calibration boundary changed')
        result[r['block_id']]={**c,'allocation_id':r['allocation_id'],'record_id':r['record_id'],'config_hash':r['config_hash']}
    if set(result)!=set(expected):raise ValueError('Incomplete calibration inventory')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['calibration','pilot','final'],required=True)
    p.add_argument('--split',choices=['pilot','final'],required=True)
    p.add_argument('--capture-root',type=Path,required=True)
    p.add_argument('--content-manifest',type=Path,default=Path('results/generalization/content-v2/source-manifest.json'))
    p.add_argument('--arrivals',type=Path,default=Path('results/generalization/arrivals-v1/arrival-manifest.json'))
    p.add_argument('--calibration-records',type=Path)
    p.add_argument('--calibration-plan',type=Path)
    p.add_argument('--pilot-calibration-records',type=Path)
    p.add_argument('--pilot-calibration-plan',type=Path)
    p.add_argument('--allocation',type=Path,help='Pilot analyzer sample-size decision; required for final split')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    content=json.loads(args.content_manifest.read_text());external=json.loads(args.arrivals.read_text())
    if content['public_manifest_sha256']!=canonical_hash({k:v for k,v in content.items() if k!='public_manifest_sha256'}):raise ValueError('Changed public content manifest')
    if not content['selection'].get('captured_prefix_deduplicated'):raise ValueError('Full-text-only split is ineligible')
    if external['content_sha256']!=canonical_hash({k:v for k,v in external.items() if k!='content_sha256'}):
        raise ValueError('Changed external arrival manifest')
    allgroups=[dict(id=b['block_id'],dataset=b['dataset']) for b in content['blocks'] if b['split']==args.split]
    selection=None
    if args.split=='final':
        if not args.allocation:p.error('Final split requires pilot precision allocation')
        selection=json.loads(args.allocation.read_text());n=selection['groups_per_dataset']
        groups=select_content_groups(allgroups,n)
    else:groups=allgroups
    ids={g['id'] for g in groups}
    index=json.loads((args.capture_root/'capture-index.json').read_text())
    banks=[]
    for g in groups:
        matches=[b for b in index if b['block_id']==g['id']]
        if len(matches)!=1:raise ValueError('Capture inventory differs from selection')
        b=matches[0]
        if not b['correctness_pass'] or b['split']!=args.split:raise ValueError('Wrong capture split')
        source_block=next(x for x in content['blocks'] if x['block_id']==g['id'])
        banks.append(dict(block_id=g['id'],dataset=g['dataset'],split=args.split,source_prompt_ids=[x['dataset']+':'+x['source_id'] for x in source_block['sources']],capture_path='banks/'+g['id']+'.npz',capture_sha256=b['capture_sha256'],tokens=b['tokens'],documents=b['documents']))
    basepath='configs/resident-online-validation-frozen-v1.json';base=json.loads((ROOT/basepath).read_text())
    execution,files=source_identity()
    plan={'generalization_schema_version':1,'phase':args.phase,'split':args.split,'arms':ARMS,'batch_tokens':128,
          'groups':groups,'banks':banks,'input_seed':471503,'arrival_seed':471502,'order_seed':471514,
          'content_selection_sha256':content['content_sha256'],'content_public_manifest_sha256':content['public_manifest_sha256'],
          'external_arrival_manifest_sha256':file_hash(args.arrivals),'base_manifest':{'path':basepath,'sha256':base['freeze']['sha256']},
          'hardware':{'hostname':HARDWARE_HOST,'gpu_uuids':GPU_UUIDS,'allocation_id':'4872248'},
          'physical_to_logical_map':list(range(128))+list(range(32)),
          'execution_source_sha256':execution,'execution_source_files':files,
          'protocol_files':{x:file_hash(ROOT/x) for x in ['docs/generalization/protocol-v1.md','docs/generalization/protocol-v2.md','docs/generalization/protocol-v3.md']},
          'inference_sha256':file_hash(ROOT/'generalization/inference.py'),'analysis_sha256':file_hash(ROOT/'generalization/analysis.py'),
          'allocation_decision':selection,'cases':[], 'interval_scope':'conditional on selected external trace material; content-bundle bootstrap; date deletion is sensitivity only'}
    if args.phase!='calibration':
        if args.phase!=args.split:p.error('Treatment phase must equal selected split')
        if not args.calibration_records or not args.calibration_plan:p.error('Treatment requires native calibration plan and records')
        cp,_=load_plan(args.calibration_plan)
        calibrations=calibration_table(rows(args.calibration_records),ids,cp,content['content_sha256'],execution)
        if {b['block_id']:b['capture_sha256'] for b in cp['banks']}!={b['block_id']:b['capture_sha256'] for b in banks}:raise ValueError('Calibration captures differ from target bank inventory')
        pilot_path=args.pilot_calibration_records or (args.calibration_records if args.split=='pilot' else None)
        if pilot_path is None:p.error('Final common absolute rate requires original pilot calibrations')
        pilotrows=rows(pilot_path)
        pilotids={b['block_id'] for b in content['blocks'] if b['split']=='pilot'}
        pilotplanpath=args.pilot_calibration_plan or (args.calibration_plan if args.split=='pilot' else None)
        if pilotplanpath is None:p.error('Original pilot calibration plan required')
        pp,_=load_plan(pilotplanpath)
        pilot=calibration_table(pilotrows,pilotids,pp,content['content_sha256'],execution)
        common_cycle_ns=math.exp(sum(math.log(c['native_mean_cycle_ns']) for c in pilot.values())/len(pilot))
        plan['calibration']=calibrations
        plan['calibration_records_sha256']=file_hash(args.calibration_records)
        plan['calibration_plan_sha256']=manifest_hash(cp)
        plan['common_rate']={'native_pilot_cycle_ns_hex':common_cycle_ns.hex(),'jobs_per_second':.85e9/common_cycle_ns,'pilot_calibration_sha256':file_hash(pilot_path),'source':'geometric mean of six native pilot cycles, multiplied by0.85; frozen across final content'}
        byid={b['id']:b for b in external['blocks']}
        assignments=assign_calendars(groups,external['blocks'],phase=args.phase)
        permutations=list(itertools.permutations(ARMS))
        for i,a in enumerate(assignments):
            block=byid[a['arrival_block_id']]
            c=calibrations[a['content_group_id']]
            for ri,kind in enumerate(['calibrated','common']):
                cycle=Fraction(c['calibration_elapsed_ns'],256) if kind=='calibrated' else Fraction.from_float(common_cycle_ns)
                horizon=round(256*cycle/Fraction(85,100))
                cal=scaled_calendar(block,numerator=horizon,denominator=block['original_horizon_ns'])
                cal.update(name=block['id']+'/'+kind,pattern=block['trace'],rho=float(Fraction(c['calibration_elapsed_ns'],horizon)),rate_kind=kind,
                           source_date_utc=block['source_date_utc'],source_rows_sha256=block['source_rows_sha256'])
                cal['sha256']=canonical_hash(cal);validate_calendar(cal)
                order=[]
                plan['cases'].append({'block_id':a['content_group_id'],'dataset':a['dataset'],'source_block':block,'calendar':cal,'arm_order':order})
        for ri,kind in enumerate(['calibrated','common']):
            selected=[case for case in plan['cases'] if case['calendar']['rate_kind']==kind]
            random.Random(plan['order_seed']+ri).shuffle(selected)
            if len(selected)%6:raise ValueError('Each rate requires a multiple of six paired cases')
            for order_index,case in enumerate(selected):case['arm_order']=list(permutations[order_index%6])
        random.Random(plan['order_seed']+2).shuffle(plan['cases'])
    plan['freeze']={'sha256':manifest_hash(plan),'stage':'before execution of this plan'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({'phase':plan['phase'],'groups':len(groups),'cases':len(plan['cases']),'trials':len(groups) if args.phase=='calibration' else 3*len(plan['cases']),'sha256':plan['freeze']['sha256']}))

if __name__=='__main__':main()

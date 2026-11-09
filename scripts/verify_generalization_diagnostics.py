#!/usr/bin/env python3
"""Small independent descriptive audit of the24 development diagnostics.

Reuses independent raw-clock and scalar route auditors, never the diagnostic
runner, plan, analysis or any production inference/preparation module.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import statistics

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('independent_generalization_main',ROOT/'scripts/verify_generalization_records.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
require,compare,digest=v.require,v.compare,v.digest
PAIRS=('main','cycle0','cycle16','offline')
CONDITIONS={'main_shortcut':('main','native'),'main_combined':('main','online'),
    'cycle0_shortcut':('cycle0','native'),'cycle0_combined':('cycle0','online'),
    'cycle16_shortcut':('cycle16','native'),'cycle16_combined':('cycle16','online'),
    'offline_native':('main','pre_native'),'offline_online':('main','pre_online')}
COMPARISONS={p:(['offline_native','offline_online'] if p=='offline' else [p+'_shortcut',p+'_combined']) for p in PAIRS}
COMPARISONS.update(cycle0_over_main_shortcut=['main_shortcut','cycle0_shortcut'],
    cycle0_over_main_combined=['main_combined','cycle0_combined'],
    cycle16_over_cycle0_shortcut=['cycle0_shortcut','cycle16_shortcut'],
    cycle16_over_cycle0_combined=['cycle0_combined','cycle16_combined'],
    online_over_precomputed_online=['offline_online','main_combined'],native_over_precomputed_native=['offline_native','main_shortcut'])


def sequence(name):
    if name=='main':return list(range(16,272))
    require(name in ('cycle0','cycle16'),'Known sequence')
    phase={'cycle0':0,'cycle16':16}[name]
    return [16+(step+phase)%32 for step in range(256)]


def states(pilot):
    pairs=list(PAIRS);random.Random(471520).shuffle(pairs);result=[]
    for di,dataset in enumerate(v.DATASETS):
        available=[g for g in pilot['groups'] if g['dataset']==dataset]
        require(len(available)==2,'Two development groups per dataset')
        group=min(available,key=lambda g:digest([471520,'select',dataset,g['id']]))
        source=[c for c in pilot['cases'] if c['block_id']==group['id'] and c['source_block']['trace']=='code' and c['calendar']['rate_kind']=='calibrated']
        require(len(source)==1,'Unique original code/calibrated case')
        order=[]
        for pair in pairs[di:]+pairs[:di]:
            names=['offline_native','offline_online'] if pair=='offline' else [pair+'_shortcut',pair+'_combined']
            order.extend(names[::-1] if (di+PAIRS.index(pair))%2 else names)
        result.append(dict(content_group_id=group['id'],dataset=dataset,source_case=source[0],condition_order=order))
    random.Random(471521).shuffle(result)
    return result


def offline_artifact(row,c,roots):
    a=row['offline_assignment_artifact'];name=Path(a['path']).name
    candidates=[root/name for root in roots if (root/name).is_file()]
    paths=[p for p in candidates if v.file_hash(p)==a['sha256']]
    require(paths,'Missing or changed precomputed route artifact')
    meta,arrays=v.routes.authenticated_npz(paths[0],a['sha256'])
    require(set(arrays)=={'native_physical_ids','online_physical_ids'},'Precomputed route array inventory')
    for mode in ('native','online'):
        physical=arrays[mode+'_physical_ids']
        require(physical.dtype==np.int64 and physical.shape==(272,128,8),'Full precomputed route tensor')
        require(v.array_hash(physical)==c['hashes'][mode],'Precomputed routes equal independently reconstructed assignment')
    compare(meta,dict(route_hashes=c['hashes'],main_assignment_artifact=row['assignment_artifact']),'Precomputed source identity')
    return dict(path=str(paths[0]),sha256=a['sha256'])


def check_routes(row,c,base,bank_weights):
    seq_name,mode=CONDITIONS[row['condition']];seq=sequence(seq_name)
    arm='online' if mode in ('online','pre_online') else 'native'
    weights=bank_weights[c['indices']][seq]
    weight_identity=check_weight_digest(row['executed_combining_weights_sha256'],weights,
        row.get('executed_combining_weights_encoding'))
    compare(row,dict(sequence_name=seq_name,dispatch_mode=mode,measured_raw_indices=seq,
        unique_measured_job_combinations=len(set(seq)),raw_job_frequency={str(k):n for k,n in Counter(seq).items()},
        logical_routing_sha256=v.array_hash(c['logical']),executed_logical_routing_sha256=v.array_hash(c['logical'][seq]),
        assignment_sha256=c['hashes'][arm],
        assignment_metrics=[c['diagnostics'][arm][i] for i in seq],resolved_kernel_config=base['default_configs']['128'],
        **c['provenance']),'Actual diagnostic sequence and route identity')
    compare([d['step'] for d in row['decision_log']],[63,127,191],'Diagnostic proposal cadence')
    for d in row['decision_log']:
        raw=seq[d['step']-15:d['step']+1];physical=c['counts'][arm][raw]
        compare(d,dict(current_map=v.MAP.tolist(),committed_map=v.MAP.tolist(),committed_map_sha256=v.array_hash(v.MAP),
            allowed=False,reason='always_skip',commit_completed=True,transfer_sent_bytes=0,transfer_sent_bytes_by_rank=[0,0],
            current_counts=(np.bincount(c['logical'][raw].ravel(),minlength=128)*2).tolist(),
            physical_slot_assignments_window=physical.tolist(),physical_slot_assignments_sha256=v.array_hash(physical,np.int32),
            physical_owner_assignments_per_step=physical.reshape(16,2,80).sum(2).tolist(),
            physical_counter_input_multiplicity=2,physical_window_next_write=0,
            physical_window_measured_step_range=[d['step']-15,d['step']]),'Executed physical/logical observation')
        proposed=np.asarray(d['proposed_map'])
        require(proposed.shape==(160,) and set(proposed)==set(range(128)) and d['candidate_changed_slots']==int((proposed!=v.MAP).sum()),'Legal uncommitted candidate')
    require(np.array_equal(row['terminal_physical_window'],c['counts'][arm][seq[-16:]]),'Actual terminal cycle phase')
    require(row['decision_trajectory_sha256']==digest([{k:d[k] for k in ('step','current_counts','current_map','proposed_map','committed_map')} for d in row['decision_log']]),'Per-sequence native trajectory')
    v.fix.numerical(row['preflight_correctness'],272);v.fix.numerical(row,2)
    hashes=row['source_weight_rank_sha256']
    require(len(hashes)==2 and all(v.sha(h) for h in hashes) and hashlib.sha256(json.dumps(hashes).encode()).hexdigest()==row['source_weights_sha256'],'Resident weight attestation digest')
    return weight_identity


WEIGHT_ENCODING='little-endian float32; C-order; no header'


def check_weight_digest(recorded,weights,encoding):
    """Require the actual float representation; v1's integer digest must fail."""
    require(weights.dtype==np.float32 and np.isfinite(weights).all(),'Float32 source weights')
    expected=hashlib.sha256(np.ascontiguousarray(weights,dtype='<f4').tobytes(order='C')).hexdigest()
    require(recorded==expected,'Executed combining weights must hash float32 bytes')
    require(encoding==WEIGHT_ENCODING,'Explicit combining-weight byte encoding')
    return dict(executed_float32_sha256=recorded,encoding=encoding,shape=list(weights.shape),
        scope='Recorded executed float bytes match authenticated bank and index reconstruction; GPU digest is a recorded attestation, not an independent GPU rerun.')


def audit(plan,rows,capture_root,roots,summary=None):
    require(plan['freeze']['sha256']==digest({k:x for k,x in plan.items() if k!='freeze'}),'Diagnostic freeze')
    require(plan['diagnostic_source_sha256']==digest(plan['diagnostic_source_files']),'Diagnostic source digest')
    for name,h in plan['diagnostic_source_files'].items():require(v.file_hash(v.relative_file(ROOT,name))==h,'Frozen diagnostic dependency '+name)
    path=v.relative_file(ROOT,plan['pilot_plan']['path']);pilot=json.loads(path.read_text())
    require(v.file_hash(path)==plan['pilot_plan']['file_sha256'] and pilot['freeze']['sha256']==plan['pilot_plan']['sha256'],'Pinned original pilot')
    base,public,external=v.identities(pilot,ROOT);v.matrix(pilot,public,external)
    expected_states=states(pilot)
    require(plan['states']==expected_states,'Identifier-only selected states and exact order')
    compare(plan,dict(generalization_diagnostic_schema_version=1,instrumentation_revision=2,phase='diagnostic',seed=471520,expected_trials=24,
        measured_jobs=256,evidence_eligible=False,selection_uses_treatment_outcomes=False,selection_uses_final_inputs=False,
        hardware=pilot['hardware'],physical_to_logical_map=v.MAP.tolist(),main_execution_source_sha256=pilot['execution_source_sha256'],
        comparison_pairs={k:COMPARISONS[k] for k in PAIRS},
        sequence_definitions={k:sequence(k) for k in ('main','cycle0','cycle16')}),'Fixed diagnostic design')
    require([(r['content_group_id'],r['condition']) for r in rows]==[(s['content_group_id'],c) for s in expected_states for c in s['condition_order']],'Complete24-condition execution order')
    artifacts=[];computed={};trials=[];weight_identities=[]
    blocks={b['block_id']:b for b in public['blocks']}
    for state in expected_states:
        gid=state['content_group_id'];members=[r for r in rows if r['content_group_id']==gid]
        entry=next(b for b in pilot['banks'] if b['block_id']==gid)
        c=v.route_state(pilot,base,entry,blocks[gid],capture_root)
        _,bank=v.routes.authenticated_npz(capture_root/entry['capture_path'],entry['capture_sha256'])
        artifacts.extend([v.artifact(members[0],c,roots),offline_artifact(members[0],c,roots)])
        cal=state['source_case']['calendar'];block=state['source_case']['source_block']
        for row in members:
            name,mode=CONDITIONS[row['condition']]
            compare(row,dict(generalization_diagnostic_schema_version=1,instrumentation_revision=2,phase='diagnostic',status='complete',evidence_eligible=False,diagnostic_only=True,
                diagnostic_config_hash=plan['freeze']['sha256'],diagnostic_source_sha256=plan['diagnostic_source_sha256'],
                main_execution_source_sha256=pilot['execution_source_sha256'],pilot_plan_sha256=pilot['freeze']['sha256'],
                block_id=gid,dataset=state['dataset'],regime='newcontent',batch_tokens=128,policy='SKIP',condition_order=state['condition_order'],
                world_size=2,dtype='bfloat16',native_vllm_commit=base['environment']['vllm_commit'],
                **{k:base['environment'][k] for k in ('legacy_harness_sha256','followup_sha256','optimization_sha256','resident_dispatch_sha256','dispatch_backend','eplb_mode')},
                arrival_schedule=cal,arrival_offsets_ns=cal['arrival_offsets_ns'],horizon_ns=cal['horizon_ns'],arrival_type='code',
                arrival_date=block['source_date_utc'],arrival_block_id=block['id'],rate_kind='calibrated',n_jobs=256,
                source_map=v.MAP.tolist(),final_map=v.MAP.tolist(),migration_bytes=0,migration_sent_messages=0,terminal_proposals_suppressed=1,
                normal_completion_verified=True,resident_state_verified=True,rank_assignment_agreement=True,guard_symbols_restored=True,
                online_selection_cost_included=mode=='online',precomputed_selection_cost_excluded=mode.startswith('pre_'),
                precomputed_adapter_cost_included=mode.startswith('pre_'),mapper_calls_measured=0 if mode=='native' else 256,
                precomputed_gpu_adapter_checks=544,full_unique_preflight_gpu_routes_checked=272,postflight_reference_raw_indices=[16,271],
                warmup_raw_indices=list(range(16)),conditioning_raw_indices=[i%16 for i in range(64)],measurement_setup_revision=2,gc_enabled_during_measurement=True),
                'Charged diagnostic execution and scope')
            require(str(row['allocation_id'])==str(plan['hardware']['allocation_id']) and row['hostname'].split('.')[0]==plan['hardware']['hostname'],'Hardware allocation')
            for key in ('physical_gpu_uuids','rank_gpu_uuids'):require(sorted(row[key])==sorted(plan['hardware']['gpu_uuids']),'Physical rank identity')
            clock=row['arrival_clock_contract'];require(clock['monotonic'] is True and clock['adjustable'] is False and 'CLOCK_MONOTONIC' in clock['implementation'],'Monotonic source clocks')
            for key in ('per_rank_input_preparation_ns_excluded','per_rank_conditioning_setup_ns_excluded','precomputed_preparation_ns_excluded'):
                require(len(row[key])==2 and all(isinstance(x,(float,int)) and np.isfinite(x) and x>0 for x in row[key]),'Recorded excluded setup '+key)
            v.fix.guard_counts(dict(row,arm='fast'))
            weight_identities.append(dict(content_group_id=gid,condition=row['condition'],
                **check_routes(row,c,base,bank['combining_weights'])))
            require(row['record_id']==digest({k:row[k] for k in ('diagnostic_config_hash','phase','content_group_id','condition')}),'Diagnostic record identity')
            m=v.old.timing_metrics(row);compare(row['queue_metrics'],m,'Raw-clock metrics');compare(row['observed_window_spans'],v.fix.spans(row),'Window clock spans')
            flat=v.fix.flat(row,m);service=m['vectors_ns']['service'];flat.update(ordinary_step_service_mean_ns=statistics.mean(x for i,x in enumerate(service) if i not in (63,127,191,255)),nonterminal_boundary_service_mean_ns=statistics.mean(service[i] for i in (63,127,191)),terminal_service_ns=service[255])
            computed[gid,row['condition']]=flat
            trials.append(dict(content_group_id=gid,condition=row['condition'],metrics=flat))
        for name in ('capture_sha256','replay_sha256','source_prompt_ids','source_weights_sha256','source_weight_rank_sha256','resolved_kernel_config','logical_routing_sha256','assignment_artifact','offline_assignment_artifact'):
            require(len({v.canonical(r[name]) for r in members})==1,'Shared state/artifact '+name)
        for seq in ('main','cycle0','cycle16'):
            subset=[r for r in members if r['sequence_name']==seq]
            for name in ('executed_logical_routing_sha256','executed_combining_weights_sha256','decision_trajectory_sha256'):
                require(len({v.canonical(r[name]) for r in subset})==1,'Same-sequence invariant '+name)
    if summary is not None:
        compare(summary,dict(valid=True,instrumentation_revision=2,expected_records=24,observed_records=24,valid_records=24,errors=[],missing_keys=[],
            diagnostic_config_hash=plan['freeze']['sha256'],diagnostic_source_sha256=plan['diagnostic_source_sha256'],confidence_intervals=None,acceptance=None),'Descriptive summary identity')
        saved={(r['content_group_id'],r['condition']):r for r in summary['trial_summaries']}
        require(len(saved)==24,'Summary complete condition matrix')
        for key,metrics in computed.items():compare(saved[key]['metrics'],metrics,'Independent per-condition arithmetic')
        keys=[(p['content_group_id'],p['comparison']) for p in summary['paired_cases']]
        require(len(keys)==30 and len(set(keys))==30 and set(keys)=={(s['content_group_id'],name) for s in expected_states for name in COMPARISONS},'All30 declared descriptive comparisons')
        for pair in summary['paired_cases']:
            require([pair['control_condition'],pair['treatment_condition']]==COMPARISONS[pair['comparison']],'Declared comparison direction')
            a=computed[pair['content_group_id'],pair['control_condition']];b=computed[pair['content_group_id'],pair['treatment_condition']]
            compare(pair,dict(control=a,treatment=b,ratios={k:b[k]/a[k] if a[k]>0 else None for k in a},differences={k:b[k]-a[k] for k in a}),'Independent descriptive paired arithmetic')
    return dict(schema_version=1,passed=True,trials=24,development_states=3,reconstructed_assignments=1632,
        measured_jobs=6144,diagnostic_plan_sha256=plan['freeze']['sha256'],artifacts=artifacts,
        combining_weight_identity=weight_identities,confidence_intervals=None,acceptance=None,
        limitations=['GPU arithmetic and weight integrity are recorded attestations, not independently rerun.',
            'Postflight checks raw16/271, the main-reference endpoints; cycle terminal inputs differ.',
            'No exclusive selector/launch/cache attribution; no production or final-data inference.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('plan','records','capture-root','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--summary',type=Path);p.add_argument('--inputs-root',type=Path,nargs='+',default=[])
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    raw=args.records.read_bytes();payload=gzip.decompress(raw) if raw.startswith(b'\x1f\x8b') else raw
    rows=[json.loads(line) for line in payload.splitlines() if line.strip()]
    summary=json.loads(args.summary.read_text()) if args.summary else None
    if summary is not None:require(summary['records_jsonl_payload_sha256']==hashlib.sha256(payload).hexdigest(),'Summary original record identity')
    result=audit(json.loads(args.plan.read_text()),rows,args.capture_root,[args.records.parent/'inputs',args.records.parent.parent/'inputs',*args.inputs_root],summary)
    result.update(verifier_sha256=v.file_hash(Path(__file__)),records_file_sha256=hashlib.sha256(raw).hexdigest(),records_jsonl_payload_sha256=hashlib.sha256(payload).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('passed','trials','development_states','measured_jobs')}))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Independent authenticated-input, physical-route and clock audit.

Imports only earlier INDEPENDENT auditors, never the experiment runner,
preparation, analysis or inference modules. GPU arithmetic/weight preservation
are checked as recorded attestations, not independently re-executed here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import gzip
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import random
import re
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def independent_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fix = independent_module('generalization_independent_fix', 'verify_arrival_fix.py')
old = fix.old
routes = independent_module('generalization_independent_routes', 'verify_resident_online_records.py')
require, compare, digest = old.require, old.compare, old.digest
file_hash, array_hash, canonical = old.file_hash, old.array_hash, old.canonical
ARMS = ('native', 'shortcut', 'combined')
DATASETS = ('gsm8k', 'mbpp', 'wildchat')
KEY = ('content_group_id', 'arrival_type', 'rate_kind', 'arm')
MAP = routes.FIXED_MAP
CONTRASTS = {'combined_over_shortcut': ('shortcut', 'combined'),
             'combined_over_native': ('native', 'combined'),
             'shortcut_over_native': ('native', 'shortcut')}


def sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def relative_file(repo, name):
    p = (repo/name).resolve()
    require(p.is_relative_to(repo.resolve()) and p.is_file(), 'Local source/manifest path: '+name)
    return p


def identities(plan, repo, source_revision=None):
    """Authenticate every declared execution/protocol file and public selection."""
    require(plan['freeze']['sha256'] == digest({k:v for k,v in plan.items() if k != 'freeze'}), 'Plan freeze')
    compare(plan, dict(generalization_schema_version=1, arms=list(ARMS), batch_tokens=128,
        physical_to_logical_map=MAP.tolist(), input_seed=471503, arrival_seed=471502,
        order_seed=471514), 'Fixed study contract')
    require(plan['phase'] in ('calibration', 'pilot', 'final') and plan['split'] in ('pilot', 'final'), 'Phase/split')
    require(plan['phase'] == 'calibration' or plan['phase'] == plan['split'], 'Treatment split')
    files = plan['execution_source_files']
    require(files and digest(files) == plan['execution_source_sha256'], 'Execution inventory digest')
    required = {'generalization/'+n for n in ('__init__.py', 'data.py', 'inputs.py', 'common.py', 'runner.py', 'external_arrivals.py')}
    required |= {'scripts/freeze_generalization.py', 'scripts/run-generalization-kuma.sh', 'arrival_fix/noop.py'}
    require(required <= set(files), 'Mandatory execution source inventory')
    pinned = {**files, **plan['protocol_files'], 'generalization/analysis.py':plan['analysis_sha256'],
              'generalization/inference.py':plan['inference_sha256']}
    if source_revision is not None:
        require(bool(re.fullmatch('[0-9a-f]{7,40}',source_revision)), 'Explicit hexadecimal archived source revision')
    for name, expected in pinned.items():
        path = relative_file(repo, name)
        if source_revision is None:
            actual=file_hash(path)
        else:
            actual=hashlib.sha256(subprocess.check_output(['git','show',source_revision+':'+name],cwd=repo)).hexdigest()
        require(sha(expected) and actual == expected, 'Frozen file: '+name)
    base = json.loads(relative_file(repo, plan['base_manifest']['path']).read_text())
    require(base['freeze']['sha256'] == plan['base_manifest']['sha256'] ==
            digest({k:v for k,v in base.items() if k != 'freeze'}), 'Frozen resident manifest')
    resident=base['execution_source_sha256']
    require({'resident_dispatch/online.py','resident_dispatch/online_gpu.py'} <= set(resident)
            and digest(resident)==base['environment']['resident_dispatch_sha256'],'Resident source inventory')
    for name,expected in resident.items():
        path=relative_file(repo,name)
        payload=path.read_bytes() if source_revision is None else subprocess.check_output(['git','show',source_revision+':'+name],cwd=repo)
        require(hashlib.sha256(payload).hexdigest()==expected,'Frozen resident source: '+name)
    public = json.loads((repo/'results/generalization/content-v2/source-manifest.json').read_text())
    require(public['public_manifest_sha256'] == plan['content_public_manifest_sha256'] ==
            digest({k:v for k,v in public.items() if k != 'public_manifest_sha256'}), 'Public content identity')
    require(public['content_sha256'] == plan['content_selection_sha256'] and
            public['selection']['captured_prefix_deduplicated'] is True, 'Captured-prefix deduplicated selection')
    external_path = repo/'results/generalization/arrivals-v1/arrival-manifest.json'
    require(file_hash(external_path) == plan['external_arrival_manifest_sha256'], 'External arrival file identity')
    external = json.loads(external_path.read_text())
    require(external['content_sha256'] == digest({k:v for k,v in external.items() if k != 'content_sha256'}), 'External arrival content identity')
    return base, public, external


def matrix(plan, public, external):
    """Rebuild complete row and execution order inventory from frozen design."""
    groups = {g['id']:g['dataset'] for g in plan['groups']}
    counts = Counter(groups.values())
    n = next(iter(counts.values()))
    require(len(groups) == len(plan['groups']) and set(counts) == set(DATASETS)
            and len(set(counts.values())) == 1 and n in ((2,) if plan['split']=='pilot' else (6,8)), 'Balanced content matrix')
    public_blocks = {b['block_id']:b for b in public['blocks']}
    eligible = [b for b in public['blocks'] if b['split'] == plan['split']]
    if plan['split'] == 'pilot':
        chosen = {b['block_id'] for b in eligible}
    else:
        rank = lambda d,g: hashlib.sha256(f'471513|select|{d}|{g}'.encode()).hexdigest()
        chosen = {g for d in DATASETS for g in sorted([b['block_id'] for b in eligible if b['dataset']==d], key=lambda x:(rank(d,x),x))[:n]}
        require(plan['allocation_decision']['groups_per_dataset'] == n, 'Final allocation selection')
    require(set(groups) == chosen, 'Prespecified selected source groups')
    banks = {b['block_id']:b for b in plan['banks']}
    require(len(banks) == len(plan['banks']) and set(banks) == set(groups), 'Complete capture inventory')
    sources = set()
    for gid, dataset in groups.items():
        block, bank = public_blocks[gid], banks[gid]
        names = [s['dataset']+':'+s['source_id'] for s in block['sources']]
        require(len(names) == len(set(names)) == 64 and not sources.intersection(names), 'Disjoint64-document source bundles')
        sources.update(names)
        compare(bank, dict(dataset=dataset, split=plan['split'], documents=64, source_prompt_ids=names), 'Bank public source identity')
    cases = {}
    blocks = {b['id']:b for b in external['blocks']}
    for case in plan['cases']:
        gid, block, cal = case['block_id'], case['source_block'], case['calendar']
        key = (gid, block['trace'], cal['rate_kind'])
        require(key not in cases and groups[gid] == case['dataset'], 'Unique correctly labeled case')
        require(block == blocks[block['id']] and block['split'] == plan['split'], 'Exact selected external block')
        require(sorted(case['arm_order']) == sorted(ARMS), 'Three distinct arms')
        calendar(cal, block, plan, gid)
        cases[key] = case
    if plan['phase'] == 'calibration':
        require(not cases, 'Saturated calibration has no treatment cases')
    else:
        require(set(cases) == set(itertools.product(groups, ('code','conversation'), ('calibrated','common'))), 'Four cases per source group')
        order_balance(list(cases.values()))
        pairs = {(gid, typ):[cases[gid,typ,r]['source_block']['id'] for r in ('calibrated','common')]
                 for gid in groups for typ in ('code','conversation')}
        require(all(a==b for a,b in pairs.values()), 'Same source block across rate definitions')
        require(len({v[0] for v in pairs.values()}) == len(pairs), 'Arrival blocks not reused across content bundles')
    ordered = [b['block_id'] for b in plan['banks']]
    random.Random(plan['order_seed']).shuffle(ordered)
    stream = []
    for gid in ordered:
        if plan['phase'] == 'calibration':
            stream.append((gid,None,'saturated','native'))
        else:
            for c in plan['cases']:
                if c['block_id'] == gid:
                    stream.extend((gid,c['source_block']['trace'],c['calendar']['rate_kind'],a) for a in c['arm_order'])
    return groups, banks, cases, stream, public_blocks


def order_balance(cases):
    for rate in (None,'calibrated','common'):
        selected=[c for c in cases if rate is None or c['calendar']['rate_kind']==rate]
        require(len(selected)>0 and len(selected)%6==0 and Counter(tuple(c['arm_order']) for c in selected)==
            Counter({p:len(selected)//6 for p in itertools.permutations(ARMS)}),
            'All six arm orders balanced '+('globally' if rate is None else 'within rate '+rate))


def calendar(cal, block, plan, gid):
    require(cal['sha256'] == digest({k:v for k,v in cal.items() if k != 'sha256'}), 'Calendar hash')
    c = plan['calibration'][gid]
    cycle = Fraction(c['calibration_elapsed_ns'],256) if cal['rate_kind']=='calibrated' else Fraction.from_float(float.fromhex(plan['common_rate']['native_pilot_cycle_ns_hex']))
    horizon = round(256*cycle/Fraction(85,100))
    original = block['original_horizon_ns']
    offsets = [round(Fraction(x*horizon, original)) for x in block['original_offsets_ns']]
    scale=Fraction(horizon,original)
    compare(cal, dict(arrival_offsets_ns=offsets, horizon_ns=horizon, source_block=block['id'],
        scale_numerator=scale.numerator,scale_denominator=scale.denominator,absolute_arrivals=True,
        drop_policy='none; drain every admitted job',
        name=block['id']+'/'+cal['rate_kind'], pattern=block['trace'],
        rho=float(Fraction(c['calibration_elapsed_ns'],horizon)), source_date_utc=block['source_date_utc'],
        source_rows_sha256=block['source_rows_sha256']), 'Exact rationally scaled calendar')
    require(len(offsets)==256 and offsets[0]==0 and offsets==sorted(offsets) and offsets[-1]<horizon, 'Calendar order, ties and final boundary')


def job_indices(count, gid):
    """Independently reconstruct the per-job PCG64 full-bank permutation."""
    seen, result, retries = set(), [], 0
    require(count>128, 'Enough captured records for distinct jobs')
    for raw in range(272):
        for retry in range(1024):
            seed = int(digest([471503,gid,128,raw,retry])[:16],16)
            values = np.random.Generator(np.random.PCG64(seed)).permutation(count)[:128].astype(np.int64)
            key = tuple(sorted(map(int,values)))
            if key not in seen:
                result.append(values); seen.add(key); retries += retry
                break
        else:
            raise AssertionError('Input combination collision retry limit')
    return np.stack(result), retries


def reuse_summary(indices, count):
    def describe(a):
        uses = np.bincount(a.ravel(),minlength=count)
        return dict(token_record_occurrences=int(a.size), distinct_captured_records_used=int(np.count_nonzero(uses)),
            unused_captured_records=int((uses==0).sum()),minimum_uses_including_unused=int(uses.min()),
            maximum_uses=int(uses.max()),record_uses_sha256=array_hash(uses))
    return dict(bank_records=count,unique_all_job_combinations=len({tuple(sorted(x)) for x in indices}),
        unique_measured_job_combinations=len({tuple(sorted(x)) for x in indices[16:]}),
        all_jobs=describe(indices),warmup_jobs=describe(indices[:16]),measured_jobs=describe(indices[16:]))


def bank_identity(meta, bank, entry, public_block, plan, base):
    env = base['environment']; count = entry['tokens']
    compare(meta, dict(block_id=entry['block_id'],dataset=entry['dataset'],split=plan['split'],
        source_prompt_ids=entry['source_prompt_ids'], source_model=env['model_id'],source_revision=env['model_revision'],
        layer=0,num_experts=128,top_k=8,hidden_size=2048,correctness_pass=True,study_id='generalization_v1',
        native_serving_trace=False,timing_evidence=False,origin='prompt_causal_prefill_layer0',
        selection_content_sha256=plan['content_selection_sha256'],special_tokens_added=False,chat_template=None), 'Captured provenance')
    specs = {'hidden_states_bf16':(np.uint16,(count,2048)), 'expert_ids':(np.int64,(count,8)),
        'combining_weights':(np.float32,(count,8)), 'token_ids':(np.int64,(count,)),
        'prompt_index':(np.int32,(count,)), 'token_position':(np.int32,(count,)), 'domain_ids':(np.int8,(count,))}
    require(set(bank)==set(specs),'Captured array inventory')
    for key,(dtype,shape) in specs.items():
        require(bank[key].dtype==dtype and bank[key].shape==shape,'Captured tensor '+key)
    ids, weights = bank['expert_ids'],bank['combining_weights']
    require(((ids>=0)&(ids<128)).all() and (np.diff(np.sort(ids,axis=1),axis=1)>0).all(), 'Captured top8 IDs')
    require(np.isfinite(weights).all() and (weights>=0).all() and np.allclose(weights.sum(1),1,rtol=0,atol=.004), 'Captured BF16-rounded combining weights')
    require(not ((bank['hidden_states_bf16'] & 0x7f80)==0x7f80).any(),'Finite captured BF16 hidden')
    require(len(meta['prompts'])==64 and set(bank['prompt_index'])==set(range(64)), 'Every source document represented')
    require(np.array_equal(bank['domain_ids'],bank['prompt_index']%2),'Parity metadata remains provenance only')
    for i,(p,source) in enumerate(zip(meta['prompts'],public_block['sources'])):
        selected = bank['prompt_index']==i; token_ids=bank['token_ids'][selected]
        expected = {k:v for k,v in source.items() if k!='token_ids_sha256'}
        expected.update(prompt_id=source['source_id'],position_start=0,token_offset=0,
            captured_tokens=source['captured_token_count'],total_source_tokens=source['full_token_count'])
        compare(p,expected,'Captured source/prefix hints')
        compare(p,{k:True for k in ('native_boundary_exact','native_router_exact','manual_router_exact','bf16_roundtrip_exact')},'Native capture attestations')
        require(1<=len(token_ids)<=512 and len(token_ids)==source['captured_token_count'], 'Actual prefix length')
        require(np.array_equal(bank['token_position'][selected],np.arange(len(token_ids))), 'Causal prefix positions')
        require(p['token_ids_sha256']==array_hash(token_ids) and source['token_ids_sha256']==digest(token_ids.tolist()), 'Captured raw-byte and public canonical token hashes')


def route_state(plan, base, entry, public_block, capture_root, repo=ROOT):
    meta, bank = routes.authenticated_npz(capture_root/entry['capture_path'],entry['capture_sha256'])
    bank_identity(meta,bank,entry,public_block,plan,base)
    capture_freeze=json.loads((repo/'results/generalization/content-v2/capture-freeze.json').read_text())
    compare(capture_freeze,dict(content_selection_sha256=plan['content_selection_sha256'],
        public_manifest_sha256=plan['content_public_manifest_sha256'],capture_script_sha256=meta['capture_script_sha256']),
        'Separate pre-capture content/source freeze')
    require(file_hash(repo/'scripts/capture_generalization_inputs.py')==meta['capture_script_sha256'],'Pinned capture implementation')
    indices,retries = job_indices(len(bank['expert_ids']),entry['block_id'])
    logical=bank['expert_ids'][indices]; counts, hashes, diagnostics, samples = {},{},{},{}
    for arm in ('native','online'):
        cs,ds,ss,h = [],[],[],hashlib.sha256()
        for raw,ids in enumerate(logical):
            physical,c,optimum = routes.assignment(ids,arm)
            h.update(np.asarray(physical,np.int64).tobytes()); cs.append(c)
            ds.append(routes.metrics(c,optimum,base['default_configs']['128']['BLOCK_SIZE_M']))
            if raw in (16,271):ss.append(np.asarray(physical,np.uint8))
        counts[arm]=np.asarray(cs,np.int32); hashes[arm]=h.hexdigest(); diagnostics[arm]=ds; samples[arm]=np.stack(ss)
    reuse = reuse_summary(indices,len(bank['expert_ids']))
    provenance = dict(capture_sha256=entry['capture_sha256'],replay_sha256=array_hash(indices),source_prompt_ids=entry['source_prompt_ids'])
    imeta=dict(schema='generalization_layer_jobs_v1',block_id=entry['block_id'],seed=471503,batch_tokens=128,jobs=272,
        warmup_range_half_open=[0,16],measured_range_half_open=[16,272],combination_collision_retries=retries,
        selection_independent_of_arm_calendar_repetition=True,natural_request_or_batch_trace=False,
        indices_sha256=array_hash(indices),job_reuse=reuse,source_block=entry['block_id'],dataset=entry['dataset'],
        split=plan['split'],source_documents=64,captured_records=len(bank['expert_ids']),
        content_selection_sha256=plan['content_selection_sha256'],authentic_record_identity_verified=True,
        capture_sha256=entry['capture_sha256'],source_prompt_ids=entry['source_prompt_ids'])
    arrays=dict(indices=indices,physical_to_logical_map=MAP,selected_prompt_index=bank['prompt_index'][indices],
        selected_token_position=bank['token_position'][indices],selected_token_ids=bank['token_ids'][indices],
        all_record_use_counts=np.bincount(indices.ravel(),minlength=len(bank['expert_ids'])),
        measured_record_use_counts=np.bincount(indices[16:].ravel(),minlength=len(bank['expert_ids'])),
        native_physical_counts=counts['native'],online_physical_counts=counts['online'],
        native_sample_physical_ids=samples['native'],online_sample_physical_ids=samples['online'])
    ameta={**imeta,**provenance,'logical_routing_sha256':array_hash(logical),'full_physical_route_sha256':hashes,
        'sample_raw_indices':[16,271],'reference_raw_indices':list(range(272))}
    return dict(logical=logical,indices=indices,counts=counts,hashes=hashes,diagnostics=diagnostics,provenance=provenance,
        arrays=arrays,artifact_meta=ameta,input_meta=imeta,reuse=reuse,sequence=list(range(16,272)))


def artifact(row, state, roots):
    a = row['assignment_artifact']; name = Path(a['path']).name
    candidates = [p/name for p in roots if (p/name).is_file()]
    require(candidates,'Missing assignment NPZ: '+name)
    good=[p for p in candidates if file_hash(p)==a['sha256']]
    require(good,'Assignment NPZ checksum: '+name)
    meta, arrays = routes.authenticated_npz(good[0],a['sha256'])
    require(set(arrays)==set(state['arrays']),'Assignment artifact array inventory')
    for key,expected in state['arrays'].items():
        require(arrays[key].dtype==expected.dtype and np.array_equal(arrays[key],expected),'Assignment artifact '+key)
    compare(meta,state['artifact_meta'],'Assignment artifact metadata')
    return dict(path=str(good[0]),sha256=a['sha256'])


def routing(row, state, base):
    arm='online' if row['arm']=='combined' else 'native'; seq=state['sequence']
    compare(row,dict(**state['provenance'],logical_routing_sha256=array_hash(state['logical']),
        assignment_sha256=state['hashes'][arm],resolved_kernel_config=base['default_configs']['128'],
        source_map=MAP.tolist(),final_map=MAP.tolist(),job_reuse=state['reuse'],newinputmeta=state['input_meta'],
        assignment_metrics=[state['diagnostics'][arm][i] for i in seq]),'Routing/input contract')
    compare([d['step'] for d in row['decision_log']],[63,127,191],'Three nonterminal proposal boundaries')
    for d in row['decision_log']:
        raw=seq[d['step']-15:d['step']+1]; expected=state['counts'][arm][raw]
        compare(d,dict(physical_slot_assignments_window=expected.tolist(),
            physical_slot_assignments_sha256=array_hash(expected,np.int32),
            current_counts=(np.bincount(state['logical'][raw].ravel(),minlength=128)*2).tolist(),
            physical_owner_assignments_per_step=expected.reshape(16,2,80).sum(2).tolist(),
            current_map=MAP.tolist(),committed_map=MAP.tolist(),committed_map_sha256=array_hash(MAP),
            allowed=False,reason='always_skip',commit_completed=True,transfer_sent_bytes=0,transfer_sent_bytes_by_rank=[0,0],
            physical_window_next_write=0,physical_counter_input_multiplicity=2,
            physical_window_measured_step_range=[d['step']-15,d['step']]),'Actual measured physical/logical window')
        proposed=np.asarray(d['proposed_map'])
        require(proposed.shape==(160,) and np.issubdtype(proposed.dtype,np.integer) and set(proposed)==set(range(128))
                and d['candidate_changed_slots']==int((proposed!=MAP).sum()),'Legal native proposal')
    trajectory=digest([{k:d[k] for k in ('step','current_counts','current_map','proposed_map','committed_map')} for d in row['decision_log']])
    require(row['decision_trajectory_sha256']==trajectory,'Logical decision trajectory digest')
    require(np.array_equal(row['terminal_physical_window'],state['counts'][arm][seq[-16:]]),'Terminal physical window')
    rank_hashes=row['source_weight_rank_sha256']
    require(len(rank_hashes)==2 and all(map(sha,rank_hashes)) and hashlib.sha256(json.dumps(rank_hashes).encode()).hexdigest()==row['source_weights_sha256'],'Recorded resident-weight checksum aggregation')
    fix.numerical(row['preflight_correctness'],272); fix.numerical(row,2)
    a=row['adversarial_checks']
    compare(a,dict(passed=True,rank_agreement=True,native_recording_state_unchanged=True,cases_checked=9,
        recording_modes_per_case=2,gpu_invocations_checked=18,batch_tokens=128),'Adversarial GPU attestation')
    require(a['evidence_sha256']==digest(a['cases']),'Adversarial evidence digest')


def row_contract(row, plan, base, group, case):
    combined=row['arm']=='combined'
    expected=dict(generalization_schema_version=1,phase=plan['phase'],config_hash=plan['freeze']['sha256'],
        base_config_hash=plan['base_manifest']['sha256'],execution_source_sha256=plan['execution_source_sha256'],
        content_selection_sha256=plan['content_selection_sha256'],block_id=row['content_group_id'],dataset=group,
        regime='newcontent',batch_tokens=128,repetition=0,status='complete',policy='SKIP',
        evidence_eligible=plan['phase']=='final',diagnostic_only=False,n_jobs=256,
        normal_completion_verified=True,resident_state_verified=True,rank_assignment_agreement=True,
        migration_bytes=0,migration_sent_messages=0,terminal_proposals_suppressed=1,guard_symbols_restored=True,
        dispatch_mode='online' if combined else 'native',noop_enabled=row['arm']!='native',assignment_selection_cost_included=True,
        online_residency_metadata_verified=combined,mapper_calls_measured=256 if combined else 0,
        measurement_setup_revision=2,gc_enabled_during_measurement=True,full_unique_preflight_gpu_routes_checked=272,
        warmup_raw_indices=list(range(16)),conditioning_raw_indices=[i%16 for i in range(64)],measured_raw_indices=list(range(16,272)),
        world_size=2,dtype='bfloat16')
    env=base['environment']
    expected.update({k:env[k] for k in ('legacy_harness_sha256','followup_sha256','optimization_sha256','resident_dispatch_sha256','dispatch_backend','eplb_mode')})
    expected['native_vllm_commit']=env['vllm_commit']
    if case is None:
        expected.update(arrival_schedule=None,arrival_offsets_ns=[0]*256,horizon_ns=1,arm_order=['native'],arrival_date=None,
            arrival_block_id='native-calibration',arrival_seed=None,schedule_name='saturated',pattern='saturated',rho=None,arrival_type=None,rate_kind='saturated')
    else:
        cal,block=case['calendar'],case['source_block']
        expected.update(arrival_schedule=cal,arrival_offsets_ns=cal['arrival_offsets_ns'],horizon_ns=cal['horizon_ns'],
            arm_order=case['arm_order'],arrival_date=block['source_date_utc'],arrival_block_id=block['id'],arrival_seed=plan['arrival_seed'],
            schedule_name=cal['name'],pattern=cal['pattern'],rho=cal['rho'],arrival_type=block['trace'],rate_kind=cal['rate_kind'])
        require(str(row['allocation_id'])==str(plan['calibration'][row['block_id']]['allocation_id']),'Calibration/measurement same allocation')
    compare(row,expected,'Execution record identity and contract')
    hw=plan['hardware']
    require(str(row['allocation_id'])==str(hw['allocation_id']) and row['hostname'].split('.')[0]==hw['hostname'],'Recorded allocation/host')
    for key in ('physical_gpu_uuids','rank_gpu_uuids'):
        require(len(row[key])==2 and sorted(row[key])==sorted(hw['gpu_uuids']),'Recorded physical rank pair')
    clock=row['arrival_clock_contract']
    require(clock['monotonic'] is True and clock['adjustable'] is False and 'CLOCK_MONOTONIC' in clock['implementation'],'Shared monotonic clock contract')
    for key in ('per_rank_conditioning_setup_ns_excluded','per_rank_input_preparation_ns_excluded'):
        require(len(row[key])==2 and all(type(x) in (int,float) and math.isfinite(x) and x>0 for x in row[key]),'Excluded setup duration '+key)
    fix.guard_counts(dict(row,arm='native' if row['arm']=='native' else 'fast'))
    identity=('config_hash','phase','arrival_block_id','block_id','regime','batch_tokens','arm','repetition','schedule_name')
    require(row['record_id']==digest({k:row[k] for k in identity}),'Record ID')
    metrics=old.timing_metrics(row)
    compare(row['queue_metrics'],metrics,'Metrics recomputed from raw absolute clocks')
    compare(row['observed_window_spans'],fix.spans(row),'Measured window clock spans')
    if plan['phase']=='calibration':compare(row['calibration_capacity'],old.capacity(row),'Full256-job capacity interval')
    return metrics


def invariants(rows):
    fields=('source_map','final_map','source_weights_sha256','source_weight_rank_sha256','capture_sha256','replay_sha256',
        'source_prompt_ids','logical_routing_sha256','decision_trajectory_sha256','resolved_kernel_config',
        'arrival_clock_contract','rank_gpu_uuids','job_reuse','newinputmeta')
    for name in fields:
        require(len({canonical(r[name]) for r in rows})==1,'Across arm/calendar invariant '+name)
    for mode in ('native','online'):
        subset=[r for r in rows if r['dispatch_mode']==mode]
        if not subset:continue
        for name in ('assignment_sha256','assignment_metrics','terminal_physical_window'):
            require(len({canonical(r[name]) for r in subset})==1,'Within-dispatch invariant '+name)
        require(len({canonical([d['physical_slot_assignments_sha256'] for d in r['decision_log']]) for r in subset})==1,'Within-dispatch physical observation trajectory')


def summary_check(summary, plan, rows, computed):
    count=len(rows);cal=plan['phase']=='calibration'
    compare(summary,dict(valid=True,phase=plan['phase'],expected_records=count,observed_records=count,valid_records=count,
        matched_triplets=0 if cal else count//3,matched_pairs=0 if cal else count,errors=[],missing_keys=[],
        plan_sha256=plan['freeze']['sha256'],execution_source_sha256=plan['execution_source_sha256']),'Analysis summary identity')
    actual={tuple(t[k] for k in KEY):t for t in summary['trial_summaries']}
    require(len(actual)==len(summary['trial_summaries'])==count,'Summary trial matrix')
    for row in rows:
        key=tuple(row[k] for k in KEY)
        compare(actual[key],dict(dataset=row['dataset'],arrival_date=row['arrival_date'],record_id=row['record_id'],
            allocation_id=row['allocation_id'],metrics=fix.flat(row,computed[key]),observed_window_spans=fix.spans(row)),'Independent summary arithmetic')
    pairs={(tuple(p[k] for k in KEY[:-1]),p['contrast']):p for p in summary['paired_cases']}
    expected=set() if cal else set(itertools.product({tuple(r[k] for k in KEY[:-1]) for r in rows},CONTRASTS))
    require(len(pairs)==len(summary['paired_cases']) and set(pairs)==expected,'Complete summary paired-contrast inventory')
    flat={tuple(r[k] for k in KEY):fix.flat(r,computed[tuple(r[k] for k in KEY)]) for r in rows}
    for (key,contrast),p in pairs.items():
        control,treatment=CONTRASTS[contrast];a,b=flat[(*key,control)],flat[(*key,treatment)]
        compare(p,dict(control_arm=control,treatment_arm=treatment,control=a,treatment=b,
            log_ratios={m:math.log(b[m])-math.log(a[m]) for m in a if a[m]>0 and b[m]>0},
            ratios={m:b[m]/a[m] if a[m]>0 else None for m in a},differences={m:b[m]-a[m] for m in a}),
            'Independent paired-ratio arithmetic')


def audit(plan, rows, repo, capture_root, input_roots, summary=None, source_revision=None):
    base,public,external=identities(plan,repo,source_revision)
    groups,banks,cases,stream,public_blocks=matrix(plan,public,external)
    keys=[tuple(r[k] for k in KEY) for r in rows]
    require(keys==stream,'Complete record matrix and actual frozen arm execution order')
    computed,artifacts,states={}, {}, []
    by_group=defaultdict(list)
    for r in rows:by_group[r['content_group_id']].append(r)
    for gid,members in by_group.items():
        c=route_state(plan,base,banks[gid],public_blocks[gid],capture_root,repo)
        for row in members:
            key=tuple(row[k] for k in KEY)
            computed[key]=row_contract(row,plan,base,groups[gid],cases.get(key[:-1]))
            routing(row,c,base)
            aid=(row['assignment_artifact']['sha256'],Path(row['assignment_artifact']['path']).name)
            if aid not in artifacts:artifacts[aid]=artifact(row,c,input_roots)
        invariants(members)
        states.append(dict(block_id=gid,capture_sha256=banks[gid]['capture_sha256'],replay_sha256=c['provenance']['replay_sha256'],
            logical_routing_sha256=array_hash(c['logical']),assignments_sha256=c['hashes'],unique_inputs=272,measured_jobs=256))
    for field in ('source_weights_sha256','source_weight_rank_sha256','resolved_kernel_config','rank_gpu_uuids'):
        require(len({canonical(r[field]) for r in rows})==1,'Fixed checkpoint/kernel/ranks across content groups: '+field)
    if summary is not None:summary_check(summary,plan,rows,computed)
    return dict(schema_version=1,passed=True,phase=plan['phase'],observed_records=len(rows),expected_records=len(stream),
        content_groups=len(states),reconstructed_assignments=len(states)*272*2,measured_jobs_checked=len(rows)*256,
        proposal_windows_checked=len(rows)*3,terminal_windows_checked=len(rows),plan_sha256=plan['freeze']['sha256'],
        execution_source_sha256=plan['execution_source_sha256'],source_revision_checked=source_revision,
        artifacts=list(artifacts.values()),states=states,
        summary_raw_arithmetic_checked=summary is not None,statistical_inference_checked=False,
        limitations=['CPU reconstruction verifies authenticated logical routing, eligibility, counts and saved GPU route hashes/samples.',
            'GPU floating-point arithmetic and current resident weight bytes are not independently rerun; their recorded attestations, coverage and hashes are checked.',
            'Raw external downloads and prefix near-deduplication have separate independent source audits; this verifier authenticates their public manifests and selected records.',
            'This execution audit does not verify bootstrap inference or establish production/full-model generality.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('plan','records','capture-root','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--summary',type=Path);p.add_argument('--inputs-root',type=Path,nargs='+',default=[])
    p.add_argument('--source-revision',help='Explicit archived git commit for immutable older records; all declared file hashes must match')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    raw=args.records.read_bytes();payload=gzip.decompress(raw) if args.records.suffix=='.gz' else raw
    rows=[json.loads(line) for line in payload.decode().splitlines() if line.strip()]
    summary=json.loads(args.summary.read_text()) if args.summary else None
    record_sha, payload_sha=hashlib.sha256(raw).hexdigest(),hashlib.sha256(payload).hexdigest()
    if summary is not None:
        require(summary.get('records_file_sha256') in (record_sha,payload_sha),'Summary original/compressed record identity')
        if 'records_jsonl_payload_sha256' in summary:require(summary['records_jsonl_payload_sha256']==payload_sha,'Summary uncompressed payload identity')
    roots=list(dict.fromkeys([args.records.parent/'inputs',args.records.parent.parent/'inputs',*args.inputs_root]))
    result=audit(json.loads(args.plan.read_text()),rows,ROOT,args.capture_root,roots,summary,args.source_revision)
    result.update(verifier_sha256=file_hash(Path(__file__)),independent_helper_sha256={p.name:file_hash(p) for p in
        [ROOT/'scripts'/n for n in ('verify_arrival_fix.py','verify_instructor_arrivals.py','verify_resident_online_records.py')]},
        records_file_sha256=record_sha,records_jsonl_payload_sha256=payload_sha)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('passed','phase','observed_records','content_groups','reconstructed_assignments')}))


if __name__=='__main__':main()

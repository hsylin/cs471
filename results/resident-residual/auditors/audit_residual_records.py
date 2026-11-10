#!/usr/bin/env python3
"""Independent fresh residual-study execution audit; no production imports.

Uses earlier independent clock/calendar primitives, an independently expressed
assignment oracle, authenticated captured inputs, and compiled binary bytes.
Inference/statistical acceptance is a separate audit.
"""
import argparse
from collections import Counter
import gzip
import itertools
import json
import math
from pathlib import Path
import tarfile
import numpy as np

import residual_stock_primitives as stock
import audit_residual_development as development
import residual_preflight_primitives as preflight_checks

require,digest=stock.require,stock.sha
ARMS=('stockoff','previous','fix')
PREVIOUS='08d4b09f110b0ca225eb95073723b4c93a3ab799a0cf52b9c04ed0322ceba1ff'
CONFIG=development.base.CONFIG
NATIVE_PINS={
 'model_executor/layers/fused_moe/experts/triton_moe.py':'ee6b83b108a0460e6308e35447434165d0778a053aa39f0c6c8ed7f36a3ddbab',
 'model_executor/layers/fused_moe/fused_moe.py':'21860340e4b409b01666e2ba0f0edae66b0f97b859756b174b573bab895cfaf7',
 'model_executor/layers/fused_moe/moe_align_block_size.py':'2addb33632b611b4c0df21f2b874dc1124d2fc867785f19f05d4ce33d4b872c6'}


def file_sha(path):return digest(Path(path).read_bytes())


def read_records(path):
    path=Path(path)
    files=sorted(path.glob('*.jsonl*')) if path.is_dir() else [path]
    if not files and path.is_dir():files=sorted((path/'trials').glob('*.jsonl*'))
    require(bool(files),'No raw records')
    rows=[];hashes={}
    for p in files:
        data=p.read_bytes();raw=gzip.decompress(data) if p.suffix=='.gz' else data
        key=p.name.removesuffix('.gz');require(key not in hashes,'Duplicate compressed/plain raw file')
        hashes[key]=dict(container_sha256=digest(data),decompressed_sha256=digest(raw))
        rows.extend(json.loads(x) for x in raw.splitlines() if x.strip())
    return rows,hashes


def oracle(ids):
    counts=np.bincount(ids.ravel(),minlength=128)
    require(counts.shape==(128,) and counts.sum()==1024,'Invalid logical IDs')
    forced=[sum(counts[32:80]>0),sum(counts[80:128]>0)]
    active=list(forced);loads=[int(counts[32:80].sum()),int(counts[80:128].sum())]
    shared=[int(x) for x in np.flatnonzero(counts[:32])]
    limit=max(*active,(sum(active)+len(shared)+1)//2)
    home=np.arange(128,dtype=np.int64)
    for e in sorted(shared,key=lambda x:(-int(counts[x]),x)):
        allowed=[r for r in (0,1) if active[r]<limit]
        owner=min(allowed,key=lambda r:(loads[r],r));home[e]=e+128*owner
        active[owner]+=1;loads[owner]+=int(counts[e])
    physical=home[ids];slots=np.bincount(physical.ravel(),minlength=160)
    require(max(active)<=64 and np.array_equal(np.r_[np.arange(128),np.arange(32)][physical],ids),'Eligibility/conservation')
    return physical,dict(physical_counts=slots,per_rank_tokens=slots.reshape(2,80).sum(1),
        per_rank_active_copies=np.count_nonzero(slots.reshape(2,80),axis=1))


def bank_check(path,entry,meta):
    require(file_sha(path)==entry['capture_sha256']==meta['capture_sha256'],'Actual captured bank file bytes')
    with np.load(path,allow_pickle=False) as z:
        m=json.loads(str(z['metadata']));names=('hidden_states_bf16','expert_ids','combining_weights','token_ids','prompt_index','token_position','domain_ids')
        arrays={name:z[name] for name in names}
        import hashlib
        h=hashlib.sha256(json.dumps(m,sort_keys=True,separators=(',',':')).encode())
        for name,a in sorted(arrays.items()):
            a=np.ascontiguousarray(a);h.update(json.dumps([name,a.dtype.str,list(a.shape)],sort_keys=True,separators=(',',':')).encode());h.update(a.tobytes())
        require(h.hexdigest()==str(z['content_sha256']),'Capture internal content hash')
        hidden=arrays['hidden_states_bf16']
        require(hidden.dtype==np.dtype('uint16') and hidden.shape==(meta['source_records'],2048) and not np.any((hidden&0x7f80)==0x7f80),'Captured BF16 input geometry/finiteness')
        require(m['study_id']=='resident_residual_v1' and m['block_id']==entry['block_id'] and m['dataset']==entry['dataset'] and m['split']==entry['split'],'Fresh capture identity')
        require(m['selection_content_sha256']==entry['selection_content_sha256'] and m['correctness_pass'],'Selection/capture correctness provenance')
        require(m['source_model']=='Qwen/Qwen3-30B-A3B' and m['source_revision']=='ad44e777bcd18fa416d9da3bd8f70d33ebb85d39' and m['layer']==0,'Wrong model layer')
        require(len(m['prompts'])==len(m['source_prompt_ids'])==32 and len(set(m['source_prompt_ids']))==32,'Expected32 distinct source documents')
        require(all(p['native_boundary_exact'] and p['bf16_roundtrip_exact'] for p in m['prompts']),'Native capture boundary attestation')
        require(meta['source_prompt_ids']==m['source_prompt_ids'] and meta['gate_weight_sha256']==m['source_tensor_sha256']['model.layers.0.mlp.gate.weight'],'Authenticated source/gate identity')
    return m


def assignments(inputs,gid,meta,entry,capture_root,treatments):
    artifact=stock.input_artifacts(inputs,gid,meta)
    require(meta['schema']=='resident_residual_hidden_jobs_v1' and meta['seed']==471803 and meta['warmup_jobs']==16 and meta['measured_jobs']==4096,'Fresh composition contract')
    m=bank_check(capture_root/entry['capture_path'],entry,meta)
    with np.load(inputs/(gid+'.npz'),allow_pickle=False) as z:
        indices=z['indices'];stored=json.loads(str(z['metadata']))
    require(all(meta.get(k)==v for k,v in stored.items()),'Stored composition metadata differs')
    seen=set()
    for raw,row in enumerate(indices):
        for retry in range(1024):
            seed=int(digest([471803,gid,128,raw,retry])[:16],16)
            wanted=np.random.Generator(np.random.PCG64(seed)).permutation(meta['source_records'])[:128].astype(np.int64)
            identity=np.sort(wanted).tobytes()
            if identity not in seen:break
        require(np.array_equal(row,wanted),'Independent input-index generation failed')
        seen.add(identity)
    with np.load(inputs/(gid+'-stock-routes.npz'),allow_pickle=False) as z:ids=z['logical_ids'];weights=z['combining_weights']
    arms=ARMS if treatments else ('stockoff',)
    physical=np.empty(ids.shape,np.int64) if treatments else None
    hashes=[];diagnostics={}
    if treatments:
        diagnostics={arm:json.loads((inputs/(gid+'-'+arm+'-resident-diagnostics.json')).read_text()) for arm in ('previous','fix')}
        require(all(len(x)==4112 for x in diagnostics.values()),'All4112 actual assignment checks required')
        for raw,x in enumerate(ids):
            p,counts=oracle(x);physical[raw]=p;ph=development.base.integer_hash(p);hashes.append(ph)
            for arm in ('previous','fix'):
                d=diagnostics[arm][raw]
                require(d['physical_sha256']==ph,'Exact policy/physical-ID bytes')
                require(d['logical_sha256']==development.base.integer_hash(x),'Exact actual logical route bytes')
                require(d['weights_sha256']==digest(np.ascontiguousarray(weights[raw],dtype='<f4').tobytes()),'Exact float32 combining weights')
                require(all(np.array_equal(d[k],v) for k,v in counts.items()),'Actual physical assignment diagnostics')
    geometry={};probe_count=0
    for arm in arms:
        path=inputs/(gid+'-'+arm+'-alignment.json');proof=json.loads(path.read_text());require(len(proof)==2,'Two alignment ranks required')
        routes=ids if arm=='stockoff' else physical
        for rank,p in enumerate(proof):
            require(p['native_source_sha256']==NATIVE_PINS,'Reviewed native alignment source hashes')
            probe_count+=development.geometry_proof(p,routes,rank,arm,False)
            slots=64 if arm=='stockoff' else 80
            for actual,r in zip(routes,p['probe_records']):
                local=actual[(actual>=rank*slots)&(actual<(rank+1)*slots)]-rank*slots
                counts=np.bincount(local.ravel(),minlength=slots)
                experts=np.repeat(np.arange(slots),(counts+63)//64)
                require(r['valid_expert_ids_sha256']==development.base.integer_hash(experts),'Independent aligned expert-bucket sequence')
        geometry[arm]=dict(file_sha256=file_sha(path),summaries=[{k:v for k,v in p.items() if k!='probe_records'} for p in proof])
    return artifact,dict(prepared_batches=4112,physical_oracle_batches=4112 if treatments else 0,
        physical_assignment_digest=digest(hashes) if treatments else None,geometry=geometry,
        native_geometry_batches=probe_count,source_prompt_ids=m['source_prompt_ids'],
        authenticated_bank_sha256=meta['capture_sha256'])


def static_check(row):
    arm=row['arm'];mapping=list(range(128))+list(range(32))
    for rank,peer in enumerate(row['per_rank_state']):
        for scope in ('start','end','audit'):
            s=peer[scope];calls=0 if scope=='start' else 4096
            wanted=dict(arm=arm,enable_eplb=True,replicas=32,physical_slots=160,local_slots=80,
                static_resident_mode=True,upstream_eplb_disabled=False,controller_active=False,selector_kind='fused',
                source_map=mapping,final_map=mapping,static_step_calls=calls,mapper_calls=calls,controller_step_calls=0,
                native_counter=0,native_valid_steps=0,recorded_step_indices=[],deferred_boundary_steps=[],decisions=[],async_events=[],
                physical_window_next_write=0,terminal_physical_window_totals_local=[0]*160,pending_result=False,rebalanced=False,
                migration_sent_bytes_local=0,migration_sent_messages_local=0,guard=None,terminal_proposals_suppressed=0,
                completion_disposition='normal_commit_no_discard',logical_router_computation_included=True,native_output_reduction=True,
                stock_router_unmodified=False,stock_policy_unmodified=False,alignment_trim=False,mapper_input_checks=True,
                selector_launch='prebound_compiled' if arm=='fix' else 'jit')
            require(all(s.get(k)==v for k,v in wanted.items()),'Static resident contract '+scope)
            require(s['resolved_kernel_config']==CONFIG and s['kernel_override_applied'] is False,'Unchanged native tile config')
            require(s['gate_weight_sha256']==row['input_meta']['gate_weight_sha256'],'Actual gate identity')
        require(all(peer['audit'][k] is True for k in ('resident_state_verified','rank_map_agreement','normal_completion_verified','static_state_unchanged','observation_and_migration_inactive')),'Full slot-weight/static-state audit')
        require(peer['shortcut_guard'] is None,'Hidden controller guard')
        context=peer['audit']['fused_stream_context']
        require(context['device']==f'cuda:{rank}' and context['sequential_single_stream_required'] and context['reused_output_storage'],'Sequential reused-buffer context')
    return development.compiled_check(row,n=4096,arm=arm)


def order_check(plan):
    if plan['phase']=='calibration':return
    groups={b['block_id']:b['dataset'] for b in plan['banks']};cases=plan['cases']
    index={(c['block_id'],c['arrival_type'],c['rho']):c['arm_order'] for c in cases}
    require(set(index)==set(itertools.product(groups,('code','conversation'),(.5,.85))) and len(index)==len(cases),'Complete case matrix')
    for gid,rho in itertools.product(groups,(.5,.85)):
        require(index[gid,'code',rho]==index[gid,'conversation',rho][::-1],'Content/load reversal')
    perms=set(itertools.permutations(ARMS))
    for rho in (.5,.85):
        for datasets in (('gsm8k_train','wildchat'),('gsm8k_train',),('wildchat',)):
            orders=[tuple(v) for (g,t,r),v in index.items() if r==rho and groups[g] in datasets]
            require(all(o in perms for o in orders),'Illegal arm order')
            counts=Counter(orders);require(max(counts[p] for p in perms)-min(counts[p] for p in perms)<=1,'Permutation imbalance')
            for a,b in itertools.combinations(ARMS,2):require(sum(o.index(a)<o.index(b) for o in orders)*2==len(orders),'Precedence imbalance')
            if plan['phase']=='final':
                for typ in ('code','conversation'):
                    counts=Counter(tuple(v) for (g,t,r),v in index.items() if r==rho and t==typ and groups[g] in datasets)
                    require(max(counts[p] for p in perms)-min(counts[p] for p in perms)<=1,'Arrival-type imbalance')


def audit(plan_path,records_path,inputs,repo,capture_root,calibration=None,binary_root=None,source_archive=None):
    plan=json.loads(plan_path.read_text());phase=plan['phase'];treatments=phase!='calibration'
    require(plan['schema']=='resident_residual_plan_v1' and plan['content_sha256']==digest(stock.unsealed(plan)),'Plan seal')
    require(plan['source_sha256']==digest(plan['source_files']) and plan['previous_closure_sha256']==PREVIOUS,'Source identity')
    if source_archive:
        with tarfile.open(source_archive) as t:
            for p,h in plan['source_files'].items():require(digest(t.extractfile(p).read())==h,'Source archive changed '+p)
    else:
        for p,h in plan['source_files'].items():require(file_sha(repo/p)==h,'Frozen source changed '+p)
    require(file_sha(repo/'docs/resident-residual/protocol-v1.md')==plan['protocol_sha256'],'Protocol changed')
    chosen=(repo/'configs/resident-residual-selected-v1.json').read_bytes()
    require(digest(chosen)==plan['selected_fix_sha256'] and json.loads(chosen)==plan['selected_fix'] and digest(plan['selected_fix'])==plan['selected_fix_content_sha256'],'Selected fix changed')
    require(plan['selected_fix']['selector_launch']=='prebound_compiled' and not plan['selected_fix']['alignment_trim'] and plan['selected_fix']['mapper_input_checks'],'Unexpected selected intervention')
    groups={b['block_id']:b for b in plan['banks']};n=2 if plan['split']=='pilot' else 8
    require(len(groups)==2*n and Counter(b['dataset'] for b in groups.values())==Counter({'gsm8k_train':n,'wildchat':n}),'Wrong fresh content matrix')
    order_check(plan);cases={(c['block_id'],c['arrival_type'],c['rho']):c for c in plan['cases']}
    for c in cases.values():stock.calendar_check(c)
    expected={(gid,None,None,'stockoff') for gid in groups} if not treatments else{(*key,a) for key in cases for a in ARMS}
    if treatments:
        require(calibration is not None,'Actual calibration records required')
        calrows,calfiles=read_records(calibration);by={r['content_group_id']:r for r in calrows}
        require(len(by)==len(calrows)==len(groups) and set(by)==set(groups),'Calibration matrix')
        require({k:v['decompressed_sha256'] for k,v in calfiles.items()}==plan['calibration_files'],'Original calibration bytes')
        for gid,r in by.items():
            require(r['record_id']==digest({k:v for k,v in r.items() if k!='record_id'}) and r['phase']=='calibration' and r['arm']=='stockoff','Calibration record identity')
            require(r['input_meta']['capture_sha256']==groups[gid]['capture_sha256'] and r['dataset']==groups[gid]['dataset'],'Calibration captured content mismatch')
            stock.raw_clock_check(r);elapsed=r['coordinator_finish_ns'][-1]-r['coordinator_start_ns'][0];p=plan['calibration_provenance'][gid]
            require(p['raw_record_sha256']==digest(r) and p['record_id']==r['record_id'] and p['calibration_elapsed_ns']==elapsed==r['calibration_capacity']['calibration_elapsed_ns'],'Actual baseline capacity')
            require(p['calibration_plan_sha256']==r['plan_sha256'] and plan['calibration_record_ids'][gid]==r['record_id'],'Calibration plan/record provenance link')
            require(p['source_sha256']==r['source_sha256']==plan['source_sha256'] and str(r['allocation_id'])==str(plan['hardware']['allocation_id']),'Calibration source/hardware')
            for key,c in cases.items():
                if key[0]==gid:require(c['calendar']['stockoff_calibration_elapsed_ns']==elapsed,'Calendar calibrated from different throughput')
    rows,files=read_records(records_path);require(len(rows)==len(expected),'Require complete frozen matrix')
    seen=set();orders={k:[] for k in cases};artifact={};checks={};identities={};kernel_ids=set();kernels=None;all_sources=set();adversarial={};weight_ids=set();gate_ids=set()
    for r in rows:
        key=tuple(r.get(k) for k in ('content_group_id','arrival_type','rho','arm'));require(key in expected and key not in seen,'Duplicate/unexpected row');seen.add(key)
        gid,_,_,arm=key;entry=groups[gid]
        require(r['record_id']==digest({k:v for k,v in r.items() if k!='record_id'}),'Raw record hash')
        require(r['schema']=='resident_residual_trial_v1' and r['status']=='complete' and r['phase']==phase and r['evidence_eligible']==(phase=='final') and r['inference_role']=='main','Record phase/disposition')
        require(r['dataset']==entry['dataset'] and r['block_id']==gid and r['plan_sha256']==plan['content_sha256'] and r['source_sha256']==plan['source_sha256'] and r['previous_closure_sha256']==PREVIOUS,'Frozen record identity')
        hw=plan['hardware'];require(r['hostname']==hw['hostname'] and sorted(r['gpu_uuids'])==sorted(hw['gpu_uuids']) and str(r['allocation_id'])==str(hw['allocation_id']),'Reserved hardware')
        require(r['clock']['monotonic'] and not r['clock']['adjustable'] and 'CLOCK_MONOTONIC' in r['clock']['implementation'],'Shared clock')
        if treatments:
            c=cases[key[:3]];cal=c['calendar'];orders[key[:3]].append(arm)
            require(r['calendar']==cal and r['arrival_offsets_ns']==cal['arrival_offsets_ns'] and r['horizon_ns']==cal['horizon_ns'] and r['arm_order']==c['arm_order'],'Frozen calendar/order')
        else:require(r['calendar'] is None and r['arrival_offsets_ns']==[0]*4096 and r['horizon_ns']==1 and r['arm_order']==['stockoff'],'Baseline calibration changed')
        stock.raw_clock_check(r)
        if not treatments:
            elapsed=r['coordinator_finish_ns'][-1]-r['coordinator_start_ns'][0]
            require(r['calibration_capacity']['calibration_elapsed_ns']==elapsed and r['calibration_capacity']['calibration_jobs']==4096,'Calibration declaration')
        rt=r['runtime_metadata']
        require(rt['study_id']=='resident_residual_v1' and rt['selected_fix']==plan['selected_fix'] and rt['native_vllm_commit']==plan['vllm_commit'] and rt['model_revision']==plan['checkpoint_revision'] and rt['world_size']==2 and rt['native_output_reduction'],'Native runtime identity')
        require(rt['tf32_allowed'] is False and rt['float32_matmul_precision']=='highest' and rt['precision_per_rank']==[dict(allow_tf32=False,matmul_precision='highest')]*2,'Both-rank FP32 reference flags')
        if treatments:
            proof_id=digest([rt['fused_adversarial'],rt['alignment_adversarial']])
            if proof_id not in adversarial:
                adversarial[proof_id]=preflight_checks.check(rt['fused_adversarial'],rt['alignment_adversarial'])
        require(r['source_weights']['source_revision']==plan['checkpoint_revision'] and r['source_weights']['layer']==0,'Expert checkpoint')
        weight_ids.add(digest(r['source_weights']))
        gate_ids.add(r['input_meta']['gate_weight_sha256'])
        for label,count in (('preflight',4112),('postflight',16)):
            p=r[label];require(p['correctness_pass'] and p['full_batch_inputs_checked']==count and p['atol']==.02 and p['rtol']==.03 and math.isfinite(p['full_batch_max_abs_difference']),'Numerical check coverage')
        pre=r['preflight'];require(all(pre[k] for k in ('logical_ids_equal','combining_weights_equal','rank_routing_agreement','input_unchanged')) and pre['fp32_job_indices']==[16,765,3765,4111] and pre['fp32_tokens_per_job']==16,'Exact routing/reference scope')
        require(r['postflight']['raw_indices']==np.linspace(16,4111,16,dtype=np.int64).tolist(),'Postflight indices')
        meta=r['input_meta'];identity=digest({k:meta[k] for k in ('capture_sha256','indices_sha256','gate_weight_sha256','source_prompt_ids','source_records','index_artifact_sha256')})
        require(gid not in identities or identities[gid]==identity,'Cross-arm/rate inputs changed');identities[gid]=identity
        if gid not in artifact:
            artifact[gid],checks[gid]=assignments(inputs,gid,meta,entry,capture_root,treatments)
            require(not all_sources.intersection(checks[gid]['source_prompt_ids']),'Source document overlaps content groups')
            all_sources.update(checks[gid]['source_prompt_ids'])
        require(pre['logical_routing_sha256']==artifact[gid]['logical_routing_sha256'] and pre['float32_combining_weights_sha256']==artifact[gid]['float32_combining_weights_sha256'],'Actual route artifact identity')
        geometry=checks[gid]['geometry'][arm]
        require(pre['alignment_artifact_sha256']==geometry['file_sha256'] and pre['alignment_probe']==geometry['summaries'],'Exact geometry artifact identity')
        if arm=='stockoff':
            stock.native_check(r,artifact[gid]);development.compiled_check(r,n=4096,arm=arm)
        else:
            require(pre['online_assignment_inputs_checked']==4112 and pre['physical_assignment_digest']==checks[gid]['physical_assignment_digest'],'Every actual assignment checked')
            k=static_check(r)
            if k is not None:kernels=k;kernel_ids.add(digest(k))
        for peer in r['per_rank_state']:
            a=peer['alignment'];require(a['enabled'] is False and a['probe_only'] is False and a['alignment_calls']==0 and not a['runtime_gpu_readback'] and a['native_allocation_preserved'] and a['native_alignment_algorithm_preserved'] and a['native_source_sha256']==NATIVE_PINS and a['config']==CONFIG,'No timed geometry intervention')
            for scope in ('start','end','audit'):
                s=peer[scope];require(s['resolved_kernel_config']==CONFIG and not s['kernel_override_applied'],'Actual native kernel config')
        require(r['native_metadata']==r['per_rank_state'][0]['end'],'Rank0 end metadata mismatch')
        probes=rt['output_reduction_probe_per_rank'];require(len(probes)==2 and probes[0]['observed_successful_backends']==probes[1]['observed_successful_backends'],'Native reduction rank agreement')
    require(seen==expected,'Missing records')
    require(len(weight_ids)==len(gate_ids)==1,'Model weights/gate identity changed between trials')
    for key,observed in orders.items():require(observed==cases[key]['arm_order'],'Executed local arm order differs')
    if treatments:
        for gid in groups:
            observed=[(r['arrival_type'],r['rho'],r['arm']) for r in rows if r['content_group_id']==gid]
            wanted=[(c['arrival_type'],c['rho'],a) for c in plan['cases'] if c['block_id']==gid for a in c['arm_order']]
            require(observed==wanted,'Within-content case execution sequence differs')
    if treatments:require(len(kernel_ids)==1,'Compiled artifact changed across trials')
    return dict(schema='independent_residual_records_v1',passed=True,phase=phase,records=len(rows),
        plan_sha256=file_sha(plan_path),plan_content_sha256=plan['content_sha256'],source_sha256=plan['source_sha256'],
        source_files_verified=len(plan['source_files']),raw_files=files,unique_content_groups=len(groups),
        actual_prepared_batches=sum(c['prepared_batches'] for c in checks.values()),
        physical_oracle_batches=sum(c['physical_oracle_batches'] for c in checks.values()),
        native_geometry_batches=sum(c['native_geometry_batches'] for c in checks.values()),
        raw_jobs=len(rows)*4096,compiled_kernels=kernels,
        compiled_binary_verification=development.check_binary_files(kernels,binary_root) if treatments else None,
        adversarial_checks=adversarial,
        input_checks=checks,auditor_sha256=file_sha(__file__),
        helper_sha256={Path(m.__file__).name:file_sha(m.__file__) for m in (stock,development,development.base,preflight_checks)},
        limits=['No GPU rerun: numerical comparisons and full slot-weight checks are authenticated recorded attestations.',
            'Native gate outputs are authenticated actual float32 route artifacts; CPU does not recompute the native gate/kernel.',
            'Every4112 input-index row, exact resident assignment and native alignment count/grid is independently checked.',
            'Statistical acceptance is handled by a separate independent inference audit.'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('plan','records','inputs-root','repo-root','capture-root','output'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('calibration','binary-root','source-archive'):p.add_argument('--'+key,type=Path)
    a=p.parse_args();require(not a.output.exists(),'Fresh output required')
    result=audit(a.plan,a.records,a.inputs_root,a.repo_root,a.capture_root,a.calibration,a.binary_root,a.source_archive)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('passed','phase','records','actual_prepared_batches','native_geometry_batches','raw_jobs')}))

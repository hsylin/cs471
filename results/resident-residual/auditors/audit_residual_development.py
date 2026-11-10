#!/usr/bin/env python3
"""Independent24-row development audit; no production imports or inference."""
import argparse
import gzip
import json
from pathlib import Path
import re
import tarfile
import numpy as np

import audit_residual_diagnostic as base

require,digest,sha=base.require,base.digest,base.sha
ARMS=('stockoff','previous','fix')
COMPILER_PINS={
 'compiler/compiler.py':'021a45ca590052ab6ca6d850eb0a102abf542165836ad92bfb3bb573218fc5d3',
 'runtime/jit.py':'5fe46fbfac257a2b8645b7a2cc709232273d42dd5e1d55de9770b412a9476d77',
 'backends/nvidia/driver.py':'992ba21e12ee170a904f3a1caca24a78ce3dd66b661087faea5973b995480f63'}
GPU_SOURCE='c0cd944f6f2c9ee0daac8b2f9891e57bec816435b27a8692020fd0f94592c5ae'


def compiled_check(row,n=512,arm=None):
    arm=arm or row['variant'];identities=[];kernels_seen=None
    for rank,peer in enumerate(row['per_rank_state']):
        stream=[]
        for scope in ('start','end','audit'):
            state=peer[scope];meta=state.get('compiled_launch')
            if arm!='fix':
                require(meta is None,'Unexpected compiled launcher in control')
                continue
            expected=dict(schema='resident_prebound_compiled_launch_v1',mode='prebound_compiled',
                triton_version='3.7.1',gpu_algorithm_changed=False,input_checks_retained=True,
                pointer_alignment_guard=True,device=f'cuda:{rank}',fixed_argument_storage_verified=True,
                hook_contract_verified=True,dynamic_launch_metadata_hooks_preserved=True,
                jit_prerun_hooks_required_empty=True,grid=[1,1,1],constexpr_args=[1024,8],
                calls=0 if scope=='start' else n,compiler_source_sha256=COMPILER_PINS,gpu_source_sha256=GPU_SOURCE)
            require(isinstance(meta,dict) and all(meta.get(k)==v for k,v in expected.items()),'Compiled launch contract')
            require(type(meta['stream_handle']) is int and meta['stream_handle']>=0,'Invalid fixed stream')
            stream.append(meta['stream_handle'])
            require(set(meta['kernels'])=={'int32','int64'},'Missing compiled dtype')
            for dtype,k in meta['kernels'].items():
                signature=dict(ids='*i32' if dtype=='int32' else '*i64',real_tokens='*i32',
                    output='*i32' if dtype=='int32' else '*i64',homes='*i32',logical_counts='*i32',
                    summary='*i32',N='constexpr',TOPK='constexpr')
                encoded=[[repr(a),repr(b)] for a,b in sorted(signature.items(),key=lambda x:repr(x[0]))]
                require(k['signature']==encoded and k['constants']==[['(6,)','1024'],['(7,)','8']], 'Actual pointer/constexpr specialization')
                require(k['baseline_jit_same_compiled_object'] is True and k['num_warps']==4 and k['num_ctas']==1 and k['arch']==90,'Wrong/different compiled GPU object')
                require(k['shared']==512 and k['pointer_specialization_attributes'],'Missing specialization metadata')
                require(all(re.fullmatch('[0-9a-f]{64}',k[name]) for name in ('compiled_hash','cubin_sha256','ptx_sha256')),'Invalid binary digest')
            identities.append(digest(meta['kernels']));kernels_seen=meta['kernels']
        if arm=='fix':
            require(len(set(stream))==1 and stream[-1]==peer['audit']['fused_stream_context']['stream_handle'],'Stream context changed')
    if arm=='fix':require(len(set(identities))==1,'Binary specialization differs across rank/scope')
    return kernels_seen


def geometry_proof(proof,routes,rank,arm,trim=False):
    n=len(routes);slots=64 if arm=='stockoff' else 80
    extent=5056 if trim else 1024+slots*2*63
    require(proof['probe_only'] is True and proof['actual_padding_verified'] is True and proof['alignment_calls']==n and len(proof['probe_records'])==n,'Incomplete alignment proof')
    require(proof['enabled']==trim and proof['config']==base.CONFIG and proof['native_allocation_preserved'] and proof['native_alignment_algorithm_preserved'],'Changed alignment path')
    for physical,r in zip(routes,proof['probe_records']):
        local=physical[(physical>=rank*slots)&(physical<(rank+1)*slots)]-rank*slots
        counts=np.bincount(local.ravel(),minlength=slots);pad=int(np.sum((counts+63)//64*64))
        require(r['physical_ids_sha256']==base.integer_hash(physical),'Actual geometry assignment differs')
        require(r['independent_local_counts']==dict(tokens=int(counts.sum()),active=int(np.count_nonzero(counts)),padded_rows=pad,capacity_bound=5056),'Independent padded work differs')
        require(r['actual_padded_rows']==pad<=5056 and r['actual_m_tiles']==pad//64,'Padding bound')
        require(r['backing_pointer_preserved'] and r['valid_prefix_equal'] and r['token_conservation_verified'] and r['scope']=='actual_native_gemm_pair','Missing prefix/GEMM proof')
        expected=[dict(n=N,top_k=k,sorted_extent=extent,m_programs=(extent+63)//64,grid_programs=(extent+63)//64*((N+127)//128)) for N,k in ((1536,8),(2048,1))]
        require(r['gemms']==expected,'Observed GEMM launch extent differs')
    return n


def check_binary_files(kernels,binary_root):
    if binary_root is None:return dict(independently_hashed=False,scope='Recorded runtime artifact attestation only')
    files={sha(p):str(p) for p in Path(binary_root).rglob('*') if p.is_file()}
    found={}
    for dtype,k in kernels.items():
        found[dtype]={}
        for key in ('cubin_sha256','ptx_sha256'):
            require(k[key] in files,'Missing compiled artifact bytes '+dtype+'/'+key)
            found[dtype][key]=dict(sha256=k[key],file=files[k[key]])
    return dict(independently_hashed=True,artifacts=found)


def audit(plan_path,records_path,inputs_root,archive,binary_root=None):
    plan=json.loads(plan_path.read_text())
    require(plan['content_sha256']==digest({k:v for k,v in plan.items() if k!='content_sha256'}),'Plan seal')
    require(plan['source_sha256']==digest(plan['source_files']),'Source map identity')
    with tarfile.open(archive) as t:
        for name,wanted in plan['source_files'].items():require(digest(t.extractfile(name).read())==wanted,'Frozen source '+name)
    require(plan['arms']==list(ARMS) and plan['measured_executions']==24,'Wrong development allocation')
    require(plan['candidate']['selector_launch']=='prebound_compiled' and plan['candidate']['alignment_trim'] is False and plan['candidate']['mapper_input_checks'] is True,'Changed single candidate')
    groups={g['block_id']:g for g in plan['groups']};require(len(groups)==4,'Four reused groups required')
    expected=[]
    for g in plan['groups']:
        require(sorted(g['orders'][0])==sorted(ARMS) and g['orders'][1]==g['orders'][0][::-1],'Not reverse technical repeats')
        expected.extend((g['block_id'],r,a) for r,order in enumerate(g['orders']) for a in order)
    raw=records_path.read_bytes();raw=gzip.decompress(raw) if records_path.suffix=='.gz' else raw
    rows=[json.loads(x) for x in raw.splitlines() if x.strip()]
    require([(r['group'],r['repeat'],r['variant']) for r in rows]==expected,'Complete24rowmatrix/order required')
    arrays={};geometry_count=0;artifacts={}
    for gid in groups:
        p=inputs_root/(gid+'-inputs.npz')
        with np.load(p,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        require(a['logical_ids'].shape==a['physical_ids'].shape==a['combining_weights'].shape==(64,128,8),'Input geometry')
        require(a['indices'].shape==(64,128) and a['combining_weights'].dtype==np.dtype('float32'),'Input dtype/shape')
        for x,y in zip(a['logical_ids'],a['physical_ids']):require(np.array_equal(base.physical_oracle(x),y),'Independent exact physical assignment')
        proof=json.loads((inputs_root/(gid+'-geometry.json')).read_text());require(len(proof)==2,'Missing rank proof')
        for rank,per_arm in enumerate(proof):
            require(set(per_arm)==set(ARMS),'Missing geometry arm')
            for arm,value in per_arm.items():geometry_count+=geometry_proof(value,a['logical_ids'] if arm=='stockoff' else a['physical_ids'],rank,arm)
        arrays[gid]=a;artifacts[gid]=dict(inputs_sha256=sha(p),geometry_sha256=sha(inputs_root/(gid+'-geometry.json')))
    metrics={};kernels=None;kernel_ids=set()
    for r in rows:
        gid,rep,arm=r['group'],r['repeat'],r['variant'];g=groups[gid];a=arrays[gid]
        require(r['record_id']==digest({k:v for k,v in r.items() if k!='record_id'}),'Record hash')
        require(r['schema']=='resident_residual_development_v1' and r['candidate']==plan['candidate'] and r['arm_order']==g['orders'][rep],'Changed arm/candidate/order')
        require(r['source_sha256']==plan['source_sha256'] and r['plan_sha256']==sha(plan_path),'Wrong executed identity')
        require(r['bank_sha256']==g['capture_sha256'] and r['dataset']==g['dataset'],'Different development bank')
        for field,key in [('indices_sha256','indices'),('logical_routes_sha256','logical_ids'),('combining_weights_sha256','combining_weights'),('exact_physical_assignment_sha256','physical_ids')]:require(r[field]==base.array_digest(a[key]),'Changed input bytes '+key)
        require(r['hostname']=='kh014' and sorted(r['gpu_uuids'])==base.UUIDS and r['allocation_id']=='4872248','Wrong reserved hardware')
        require(r['unique_batches']==64 and r['jobs']==512 and r['atol']==.02 and r['rtol']==.03 and np.isfinite(r['fp32_max_abs_error']),'Correctness scope')
        base.check_state(r)
        k=compiled_check(r)
        if k is not None:kernels=k;kernel_ids.add(digest(k))
        metrics[gid,rep,arm]=base.check_timing(r)
    require(len(kernel_ids)==1,'Compiled artifact changed between trials')
    summary={}
    for gid in groups:
        values={arm:{key:float(np.mean([metrics[gid,rep,arm][key] for rep in (0,1)])) for key in ('mean_service_us','job_completion_per_job_us','completed_per_job_us')} for arm in ARMS}
        summary[gid]=dict(dataset=groups[gid]['dataset'],variants=values,
            paired_effects={key:dict(fix_minus_previous=values['fix'][key]-values['previous'][key],fix_minus_stockoff=values['fix'][key]-values['stockoff'][key],fix_over_previous=values['fix'][key]/values['previous'][key],fix_over_stockoff=values['fix'][key]/values['stockoff'][key]) for key in values['fix']})
    submission_path=inputs_root/'submission-diagnostic.json';submission=json.loads(submission_path.read_text())
    submitted={}
    for rank,values in enumerate(submission['per_rank']):
        expected_keys={(arm,rep,i) for arm in ('previous','fix') for rep in (0,1) for i in range(32)}
        require({(x['arm'],x['repeat'],x['input']) for x in values}==expected_keys and len(values)==128,'Incomplete submission diagnostic')
        submitted[str(rank)]={arm:dict(mean_us=float(np.mean([x['submission_ns'] for x in values if x['arm']==arm])/1000),samples=64) for arm in ('previous','fix')}
    return dict(schema='independent_residual_development_v1',passed=True,records=24,
        source_sha256=plan['source_sha256'],source_archive_sha256=sha(archive),plan_sha256=sha(plan_path),
        records_container_sha256=sha(records_path),records_decompressed_sha256=digest(raw),
        auditor_sha256=sha(__file__),primitives_sha256=sha(base.__file__),artifacts=artifacts,
        unique_batch_oracles=256,native_geometry_batches=geometry_count,raw_jobs=24*512,
        per_group=summary,variant_arithmetic_means={arm:{key:float(np.mean([g['variants'][arm][key] for g in summary.values()])) for key in ('mean_service_us','job_completion_per_job_us','completed_per_job_us')} for arm in ARMS},
        compiled_kernels=kernels,compiled_binary_verification=check_binary_files(kernels,binary_root),
        submission_diagnostic=dict(sha256=sha(submission_path),scope=submission['scope'],per_rank=submitted),
        limits=['Descriptive reused-content results; technical repetitions averaged within four groups; no CI or p-value.',
            'Host-submission measurements exclude completion wait; they are not exclusive GPU time or layer-service components.',
            'GPU functional checks are authenticated recorded evidence, not a new GPU execution by this CPU verifier.'])


def markdown(report):
    lines=['# Independent compiled-launch development audit','',f"PASS: {report['records']} complete records; unchanged physical assignment and native geometry verified.",'',
        '| Arm | Mean service µs/job | Completed µs/job |','|---|---:|---:|']
    for arm,v in report['variant_arithmetic_means'].items():lines.append(f"| {arm} | {v['mean_service_us']:.3f} | {v['completed_per_job_us']:.3f} |")
    lines+=['','| Group | Fix/previous service | Fix/stockoff service |','|---|---:|---:|']
    for gid,v in report['per_group'].items():
        e=v['paired_effects']['mean_service_us'];lines.append(f"| {gid} | {e['fix_over_previous']:.6f} | {e['fix_over_stockoff']:.6f} |")
    lines+=['']+['- '+x for x in report['limits']]
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('plan','records','inputs-root','source-archive','output','markdown'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--binary-root',type=Path);a=p.parse_args()
    require(not a.output.exists() and not a.markdown.exists(),'Fresh audit outputs required')
    report=audit(a.plan,a.records,a.inputs_root,a.source_archive,a.binary_root)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');a.markdown.write_text(markdown(report))
    print(json.dumps(dict(passed=True,records=24,output=str(a.output))))

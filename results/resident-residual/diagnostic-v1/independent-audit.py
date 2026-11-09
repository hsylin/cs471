#!/usr/bin/env python3
"""Independent descriptive audit of the fixed seven-arm residual diagnostic.

No production analysis, timing, alignment, or assignment code is imported.
Recorded numerical checks are authenticated attestations, not a new GPU rerun.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import tarfile
import numpy as np

VARIANTS = ('stockoff','stockoff_trim','previous','previous_trim',
            'replay_checked','replay_checked_trim','replay_unchecked')
CONFIG = dict(BLOCK_SIZE_M=64,BLOCK_SIZE_N=128,BLOCK_SIZE_K=64,
              GROUP_SIZE_M=1,SPLIT_K=1,num_warps=4,num_stages=3)
PREVIOUS = '08d4b09f110b0ca225eb95073723b4c93a3ab799a0cf52b9c04ed0322ceba1ff'
UUIDS = sorted(['GPU-204281f9-9c87-3c25-adfb-9edd6c972ec3',
                'GPU-7d5c9009-305b-9341-957f-1deb96611ed5'])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    raw = value if isinstance(value, bytes) else json.dumps(value,sort_keys=True,
        separators=(',',':'),allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    return digest(Path(path).read_bytes())


def array_digest(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256(digest([a.dtype.str,list(a.shape)]).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def integer_hash(value):
    return digest(np.ascontiguousarray(value,dtype='<i8').tobytes())


def physical_oracle(ids):
    """Separate scalar LPT implementation, preserving the frozen tie rules."""
    require(ids.shape==(128,8) and ids.dtype.kind in 'iu' and
            np.all((ids>=0)&(ids<128)), 'Invalid actual logical routes')
    require(all(len(set(r))==8 for r in ids.tolist()), 'Top8 routes contain duplicates')
    frequencies = [int(np.count_nonzero(ids==e)) for e in range(128)]
    load = [sum(frequencies[32:80]),sum(frequencies[80:128])]
    active = [sum(n>0 for n in frequencies[32:80]),sum(n>0 for n in frequencies[80:128])]
    queue = [(frequencies[e],e) for e in range(32) if frequencies[e]]
    cap = max(*active,(sum(active)+len(queue)+1)//2)
    homes = list(range(128))
    for count,expert in sorted(queue,key=lambda item:(-item[0],item[1])):
        allowed = [r for r in range(2) if active[r]<cap]
        require(bool(allowed),'No eligible resident rank')
        selected = min(allowed,key=lambda r:(load[r],r))
        homes[expert] = expert+128*selected
        load[selected] += count
        active[selected] += 1
    out = np.array(homes,dtype=np.int64)[ids]
    require(max(active)<=64,'One-home active bound violated')
    mapping=np.r_[np.arange(128),np.arange(32)]
    require(np.array_equal(mapping[out],ids),'Logical conservation failed')
    return out


def close(a,b,message):
    require(np.isclose(a,b,rtol=2e-12,atol=1e-6),message)


def check_timing(row):
    n=512;epoch=row['epoch_monotonic_ns'];end=row['observation_end_ns']
    require(row['n_jobs']==n and row['horizon_ns']==1 and row['arrival_offsets_ns']==[0]*n,
            'Not the declared saturated diagnostic')
    fields=['coordinator_admit_ns','coordinator_start_ns','coordinator_finish_ns']
    arrays=[]
    for field in fields:
        value=row[field]
        require(len(value)==n and all(type(x) is int and x>0 for x in value),'Missing/noninteger clock '+field)
        arrays.append(np.array(value,dtype=np.int64))
    admit,start,finish=arrays
    require(np.all(admit>=epoch) and np.all(start>=admit) and np.all(finish>start), 'Invalid request clocks')
    require(np.all(admit[1:]>=finish[:-1]) and end>=finish[-1], 'Overlapping/incomplete FCFS jobs')
    require(row['per_job_completion_scope']=='current_inference_stream_after_native_step','Wrong completion scope')
    for rank in range(2):
        rs=np.asarray(row['per_rank_start_ns'][rank],dtype=np.int64)
        rf=np.asarray(row['per_rank_gpu_complete_ns'][rank],dtype=np.int64)
        require(rs.shape==rf.shape==(n,) and np.all(rs>=start) and np.all(rf>=rs) and np.all(rf<=finish), 'Rank clocks outside service')
        require(row['per_rank_controller_quiescent_ns'][rank]>=row['per_rank_controller_tail_start_ns'][rank]>=rf[-1], 'Invalid normal tail clocks')
    require(row['controller_quiescent_ns']>=finish[-1] and end>=row['controller_quiescent_ns'],'Incomplete coordinator tail')
    exact={'job_completed_elapsed_ns':int(finish[-1])-epoch,
        'completed_elapsed_ns':end-epoch,
        'controller_tail_elapsed_ns':row['controller_quiescent_ns']-int(finish[-1]),
        'quiescence_call_elapsed_ns':row['controller_quiescent_ns']-row['controller_quiescence_start_ns'],
        'drain_including_controller_ns':max(0,row['controller_quiescent_ns']-epoch-1)}
    require(all(row[k]==v for k,v in exact.items()),'Elapsed time does not match raw clocks')
    vectors=dict(response=finish-epoch,queue_wait=start-epoch,service=finish-start,
                 admission_delay=admit-epoch,dispatch_delay=start-admit)
    q=row['queue_metrics']
    require(q['complete'] and q['scheduled_jobs']==q['completed_jobs']==n and q['unfinished_jobs']==0,'Missing jobs')
    for key,a in vectors.items():
        require(q['vectors_ns'][key]==a.tolist(),'Stored latency vector differs: '+key)
        expected=dict(mean=float(a.mean()),p50=float(np.quantile(a,.5)),
            p95=float(np.quantile(a,.95)),p99=float(np.quantile(a,.99)),max=int(a.max()))
        for stat,value in expected.items():close(q['summary_ns'][key][stat],value,'Summary differs: '+key+'/'+stat)
    for key,a in [('queue_occupancy',vectors['queue_wait']),('system_occupancy',vectors['response'])]:
        require(q[key]['integral_job_ns']==int(a.sum()) and q[key]['jobs_at_end']==0, 'Occupancy integral differs')
        close(q[key]['mean_jobs'],a.sum()/(end-epoch),'Occupancy mean differs')
    return dict(mean_service_us=float(vectors['service'].mean()/1000),
        job_completion_per_job_us=(finish[-1]-epoch)/n/1000,
        completed_per_job_us=(end-epoch)/n/1000,
        mean_interjob_gap_us=float(np.mean(start[1:]-finish[:-1])/1000),
        per_rank_mean_execution_us=[float(np.mean(np.array(row['per_rank_gpu_complete_ns'][r])-np.array(row['per_rank_start_ns'][r]))/1000) for r in range(2)])


def check_state(row):
    off=row['variant'].startswith('stockoff');trim=row['variant'].endswith('_trim')
    slots=128 if off else 160;mapping=list(range(128))+([] if off else list(range(32)))
    require(len(row['per_rank_state'])==2,'Missing rank states')
    gate_hashes=[]
    for rank,state in enumerate(row['per_rank_state']):
        require(state['postflight_full_inputs']==64,'Incomplete numerical postflight')
        for scope in ['start','end','audit']:
            s=state[scope];gate_hashes.append(s['gate_weight_sha256'])
            require(s['physical_slots']==slots and s['local_slots']==slots//2 and
                    s['source_map']==s['final_map']==mapping,'Changed resident map')
            require(s['resolved_kernel_config']==CONFIG and s['kernel_override_applied'] is False, 'Wrong native kernel config')
            require(s['logical_router_computation_included'] and s['native_output_reduction'], 'Dropped computation')
            require(s['native_experts_class'].endswith('.TritonExperts'),'Unexpected backend')
            require(s['source_expert_parameter_bytes']==slots//2*9437184,'Incorrect resident weight bytes')
            require(not s['pending_result'] and not s['rebalanced'] and s['async_events']==[] and
                    s['decisions']==[] and s['recorded_step_indices']==[], 'Controller advanced in static diagnostic')
            require(s['migration_sent_bytes_local']==s['migration_sent_messages_local']==s['controller_step_calls']==0, 'Unexpected transfer/control')
            require(s['completion_disposition']=='normal_commit_no_discard', 'Wrong completion')
            expected_calls=0 if off or scope=='start' else 512
            require(s['mapper_calls']==expected_calls,'Wrong selector invocation count')
            if off:
                require(not s['enable_eplb'] and s['stock_router_unmodified'] and s['replicas']==0,'Not native EPLB-off')
            else:
                require(s['enable_eplb'] and s['replicas']==32 and not s['stock_router_unmodified'] and
                        s['static_resident_mode'] and not s['controller_active'], 'Not frozen static resident mode')
                require(s['native_counter']==0 and s['physical_window_next_write']==0 and
                        sum(s['terminal_physical_window_totals_local'])==0,'Recording state advanced')
                require(s['static_step_calls']==(0 if scope=='start' else 512),'Wrong step count')
        a=state['audit']
        require(a['resident_state_verified'] and a['rank_map_agreement'] and a['normal_completion_verified'], 'Missing state audit')
        if not off: require(a['static_state_unchanged'] and a['observation_and_migration_inactive'],'Static state not verified')
        alignment=state['alignment']
        require(alignment['enabled']==trim and alignment['alignment_calls']==(512 if trim else 0), 'Incorrect alignment hook calls')
        require(alignment['probe_only'] is False and not alignment['runtime_gpu_readback'] and
                alignment['native_allocation_preserved'] and alignment['native_alignment_algorithm_preserved'], 'Wrong timed adapter')
        require(alignment['capacity_bound']==5056 and alignment['native_capacity']==1024+slots*63,'Incorrect capacity')
    require(len(set(gate_hashes))==1,'Gate identity differs')
    reductions=row['output_reduction_probes']
    require(len(reductions)==2 and reductions[0]['observed_successful_backends']==reductions[1]['observed_successful_backends'], 'Rank reduction backend differs')
    return gate_hashes[0]


def check_geometry(path,arrays):
    data=json.loads(path.read_text());require(len(data)==2,'Geometry missing rank')
    count=0;stats={}
    for rank,by_variant in enumerate(data):
        require(set(by_variant)==set(VARIANTS),'Geometry missing arm')
        for variant,proof in by_variant.items():
            off=variant.startswith('stockoff');trim=variant.endswith('_trim')
            slots=64 if off else 80;global_slots=slots*2
            extent=5056 if trim else 1024+global_slots*63
            routes=arrays['logical_ids'] if off else arrays['physical_ids']
            require(proof['probe_only'] and proof['actual_padding_verified'] and proof['alignment_calls']==64 and len(proof['probe_records'])==64,'Incomplete geometry proof')
            require(proof['enabled']==trim and proof['config']==CONFIG,'Incorrect probe geometry')
            padded_values=[]
            for i,r in enumerate(proof['probe_records']):
                physical=routes[i];local=physical[(physical>=rank*slots)&(physical<(rank+1)*slots)]-rank*slots
                counts=np.bincount(local.ravel(),minlength=slots)
                pad=int(np.sum(((counts+63)//64)*64))
                expected=dict(tokens=int(counts.sum()),active=int(np.count_nonzero(counts)),padded_rows=pad,capacity_bound=5056)
                require(r['independent_local_counts']==expected and r['actual_padded_rows']==pad<=5056 and r['actual_m_tiles']==pad//64,'Incorrect native local padding')
                require(r['physical_ids_sha256']==integer_hash(physical),'Geometry used different assignments')
                require(r['backing_pointer_preserved'] and r['valid_prefix_equal'] and r['token_conservation_verified'] and r['scope']=='actual_native_gemm_pair','Missing native prefix/GEMM proof')
                expected_gemms=[dict(n=n,top_k=k,sorted_extent=extent,m_programs=(extent+63)//64,grid_programs=(extent+63)//64*((n+127)//128)) for n,k in [(1536,8),(2048,1)]]
                require(r['gemms']==expected_gemms,'Wrong actual observed GEMM grids')
                count+=1;padded_values.append(pad)
            stats[f'rank{rank}/{variant}']=dict(mean_padded_rows=float(np.mean(padded_values)),min_padded_rows=min(padded_values),max_padded_rows=max(padded_values),visible_extent=extent)
    return dict(probe_batches_verified=count,per_rank_variant=stats,sha256=sha(path))


def effect_table(means):
    p,t,r,u=means['previous'],means['previous_trim'],means['replay_checked'],means['replay_checked_trim']
    return dict(selector_invocation_package_full_us=p-r,selector_invocation_package_trim_us=t-u,
        launch_extent_package_online_us=p-t,launch_extent_package_replay_us=r-u,
        launch_extent_package_stockoff_us=means['stockoff']-means['stockoff_trim'],
        selector_by_extent_interaction_us=(p-t)-(r-u),
        input_validation_only_us=r-means['replay_unchecked'],
        online_full_minus_stockoff_us=p-means['stockoff'],
        online_trim_minus_stockoff_us=t-means['stockoff'],
        online_trim_minus_trimmed_stockoff_us=t-means['stockoff_trim'])


def audit(plan_path,records_path,inputs_root,source_archive):
    plan=json.loads(plan_path.read_text())
    require(plan['content_sha256']==digest({k:v for k,v in plan.items() if k!='content_sha256'}),'Plan seal failed')
    require(plan['source_sha256']==digest(plan['source_files']),'Source map identity failed')
    require(plan['previous_closure_sha256']==PREVIOUS and plan['variants']==list(VARIANTS), 'Wrong previous/arm identity')
    with tarfile.open(source_archive) as tf:
        for rel,expected in plan['source_files'].items():
            require(digest(tf.extractfile(rel).read())==expected,'Archived source changed: '+rel)
    groups={g['block_id']:g for g in plan['groups']}
    require(len(groups)==4 and plan['measured_executions']==56,'Wrong diagnostic size')
    expected_sequence=[]
    for g in plan['groups']:
        require(sorted(g['orders'][0])==sorted(VARIANTS) and g['orders'][1]==g['orders'][0][::-1],'Not exact reverse orders')
        expected_sequence.extend((g['block_id'],rep,v) for rep,order in enumerate(g['orders']) for v in order)
    raw=records_path.read_bytes()
    if records_path.suffix=='.gz':raw=gzip.decompress(raw)
    rows=[json.loads(line) for line in raw.splitlines() if line.strip()]
    require(len(rows)==56,'Require all56 completed records, not partial outcomes')
    actual_sequence=[(r['group'],r['repeat'],r['variant']) for r in rows]
    require(actual_sequence==expected_sequence,'Recorded execution order differs')
    data={};geometry={};artifacts={};assignments=0
    for gid,g in groups.items():
        p=inputs_root/(gid+'-inputs.npz')
        with np.load(p,allow_pickle=False) as f:data[gid]={k:f[k].copy() for k in f.files}
        a=data[gid]
        require(set(a)=={'indices','logical_ids','combining_weights','physical_ids'},'Unexpected input arrays')
        require(a['indices'].shape==(64,128) and len({tuple(sorted(row)) for row in a['indices'].tolist()})==64,'Not64unique actual batches')
        require(all(len(set(row))==128 for row in a['indices'].tolist()),'Repeated source record within batch')
        require(a['logical_ids'].shape==a['physical_ids'].shape==a['combining_weights'].shape==(64,128,8),'Wrong input geometry')
        require(a['combining_weights'].dtype==np.dtype('float32') and np.all(np.isfinite(a['combining_weights'])) and np.all(a['combining_weights']>=0) and np.allclose(a['combining_weights'].sum(-1),1,rtol=1e-5,atol=1e-5),'Invalid authenticated native route weights')
        for i in range(64):
            require(np.array_equal(physical_oracle(a['logical_ids'][i]),a['physical_ids'][i]),'Physical assignment does not match independent policy')
            assignments+=1
        geometry[gid]=check_geometry(inputs_root/(gid+'-geometry.json'),a)
        artifacts[gid]=dict(input_file=str(p),sha256=sha(p),geometry_sha256=geometry[gid]['sha256'])
    metrics={};gate=set();record_ids=set()
    for r in rows:
        require(r['record_id']==digest({k:v for k,v in r.items() if k!='record_id'}),'Record hash mismatch')
        require(r['record_id'] not in record_ids,'Duplicate record');record_ids.add(r['record_id'])
        gid,rep,v=r['group'],r['repeat'],r['variant'];g=groups[gid];a=data[gid]
        require(r['schema']=='resident_residual_diagnostic_v1' and r['arm_order']==g['orders'][rep],'Schema/order mismatch')
        require(r['source_sha256']==plan['source_sha256'] and r['previous_closure_sha256']==PREVIOUS and r['plan_sha256']==sha(plan_path),'Wrong frozen execution identity')
        require(r['dataset']==g['dataset'] and r['bank_sha256']==g['capture_sha256'],'Changed inputbank identity')
        for field,key in [('indices_sha256','indices'),('logical_routes_sha256','logical_ids'),('combining_weights_sha256','combining_weights'),('exact_physical_assignment_sha256','physical_ids')]:
            require(r[field]==array_digest(a[key]),'Changed input bytes: '+field)
        require(r['jobs']==512 and r['unique_batches']==64 and r['fp32_jobs']==[0,21,42,63] and r['fp32_tokens_per_job']==16 and r['atol']==.02 and r['rtol']==.03 and np.isfinite(r['fp32_max_abs_error']),'Numerical coverage metadata changed')
        require(r['hostname']=='kh014' and sorted(r['gpu_uuids'])==UUIDS and r['allocation_id']=='4872248','Wrong hardware')
        gate.add(check_state(r));metrics[gid,rep,v]=check_timing(r)
    require(len(gate)==1,'Gate weights differ across comparisons')
    table={}
    for gid in groups:
        values={v:{key:float(np.mean([metrics[gid,rep,v][key] for rep in (0,1)])) for key in ('mean_service_us','job_completion_per_job_us','completed_per_job_us','mean_interjob_gap_us')} for v in VARIANTS}
        table[gid]=dict(dataset=groups[gid]['dataset'],variants=values,
            service_effects=effect_table({v:x['mean_service_us'] for v,x in values.items()}),
            completed_effects=effect_table({v:x['completed_per_job_us'] for v,x in values.items()}))
    aggregate={}
    for kind in ('service_effects','completed_effects'):
        aggregate[kind]={key:dict(mean=float(np.mean([g[kind][key] for g in table.values()])),
            min=min(g[kind][key] for g in table.values()),max=max(g[kind][key] for g in table.values())) for key in next(iter(table.values()))[kind]}
    variants={v:{key:float(np.mean([g['variants'][v][key] for g in table.values()])) for key in ('mean_service_us','job_completion_per_job_us','completed_per_job_us')} for v in VARIANTS}
    return dict(schema='independent_residual_diagnostic_audit_v1',passed=True,records=56,
        unique_content_groups=4,technical_repeats_per_group=2,jobs_per_record=512,
        unique_batches_verified=assignments,total_jobs_verified=56*512,
        plan_sha256=sha(plan_path),source_sha256=plan['source_sha256'],
        source_archive_sha256=sha(source_archive),source_files_verified=len(plan['source_files']),
        records_container_sha256=sha(records_path),records_decompressed_sha256=digest(raw),
        auditor_sha256=sha(__file__),artifact_inventory=artifacts,geometry=geometry,
        variant_arithmetic_means=variants,per_group=table,paired_package_effects=aggregate,
        checks=['Exact source archive and record hashes','Complete frozen56rowmatrix and exact reverseorders',
            'All actual float32 input bytes authenticated','Independent perbatch LPTphysical assignments',
            'Bothrank static maps/weights and zero controller state attestations',
            'BothnativeGEMM observed view/grid and actual padded local counts',
            'Raw coordinator/rank timing and every queue vector/summary reconstructed'],
        limits=['Descriptive reuseddevelopmentinputs; no CI,pvalue,or heldoutclaim.',
            'Technical repeats averaged within group;512jobs and64cycledinputs are not independent units.',
            'Packageeffects include chargedPythonchecks/viewcreation and GPU effects; not exclusive component timing.',
            'Offline physical-ID replay is not deployable and not a runtime lowerbound.',
            'Recorded numerical and slot-weight checks are authenticated; this CPUaudit is not a newGPU rerun.',
            'Native atomic scatter order may differ within an expert; no crosscall rawsorted-prefix equality required.',
            'The v2executed source has64actual-layer probes pervariant; separate72adversarial alignment calls belong to unexecutedv3.'])


def markdown(report):
    lines=['# Independent residual diagnostic audit','',f"PASS: {report['records']} records; four reused content groups, two reversed technical repetitions. No inferential claims.",'',
        '| Variant | Mean service µs/job | Completed µs/job |','|---|---:|---:|']
    for v,m in report['variant_arithmetic_means'].items():lines.append(f"| {v} | {m['mean_service_us']:.3f} | {m['completed_per_job_us']:.3f} |")
    lines+=['','Positive package effects below mean the named package costs time; the extent contrast is full minus trim.','',
            '| Paired effect | Mean µs/job | Group range |','|---|---:|---:|']
    for key,m in report['paired_package_effects']['service_effects'].items():lines.append(f"| {key} | {m['mean']:.3f} | [{m['min']:.3f}, {m['max']:.3f}] |")
    lines+=['','Per-group paired differences and completed-time counterparts are in the JSON.','']
    lines += ['- '+x for x in report['limits']]
    return '\n'.join(lines)+'\n'


def self_test():
    import unittest
    class Tests(unittest.TestCase):
        def test_float_bytes(self):
            self.assertNotEqual(array_digest(np.array([.1,.2],np.float32)),array_digest(np.array([.2,.1],np.float32)))
        def test_oracle_coverage(self):
            ids=(np.arange(1024).reshape(128,8)%128).astype(np.int32)
            out=physical_oracle(ids);mapping=np.r_[np.arange(128),np.arange(32)]
            np.testing.assert_array_equal(mapping[out],ids)
            self.assertLessEqual(max(np.count_nonzero(np.bincount(out.ravel(),minlength=160).reshape(2,80),axis=1)),64)
        def test_wrong_assignment_detectable(self):
            ids=np.tile(np.arange(8),(128,1));expected=physical_oracle(ids)
            wrong=expected.copy();wrong[0,0]=32
            self.assertFalse(np.array_equal(expected,wrong))
        def test_interaction_sign(self):
            e=effect_table(dict(stockoff=10,stockoff_trim=9,previous=16,previous_trim=12,replay_checked=11,replay_checked_trim=10,replay_unchecked=10.5))
            self.assertEqual(e['selector_by_extent_interaction_us'],3)
            self.assertEqual(e['input_validation_only_us'],.5)
    result=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['plan','records','inputs-root','source-archive','output','markdown']:p.add_argument('--'+key,type=Path)
    p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:self_test()
    else:
        require(all(getattr(a,key.replace('-','_')) is not None for key in ['plan','records','inputs-root','source-archive','output','markdown']), 'All path arguments required')
        require(not a.output.exists() and not a.markdown.exists(),'Fresh output paths required')
        report=audit(a.plan,a.records,a.inputs_root,a.source_archive)
        a.output.parent.mkdir(parents=True,exist_ok=True);a.markdown.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');a.markdown.write_text(markdown(report))
        print(json.dumps(dict(passed=True,records=report['records'],output=str(a.output))))

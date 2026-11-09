"""Portable independent audit of the frozen optimization tuning pilot.

Reconstructs the selection rule without importing measured-package helpers.
NumPy is used only to inspect/regenerate saved capture input indices. Source
pins, provenance, and pilot banks are required; no GPU execution is performed.
Gzip records are checked against the selection's original decompressed SHA256.
"""
from pathlib import Path
import argparse
import collections
import gzip
import hashlib
import itertools
import json
import math
import random
import struct
import numpy as np

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def average(x):return math.fsum(x)/len(x)
def require(ok,message):
    if not ok:raise AssertionError(message)

def audit(manifest_path, records_path, selection_path, tuning_dir, repo_root, capture_dir=None):
    REPO=Path(repo_root); ROOT=Path(tuning_dir)
    M_PATH=Path(manifest_path); R_PATH=Path(records_path); S_PATH=Path(selection_path)
    CAPTURE=Path(capture_dir) if capture_dir is not None else REPO/'results/optimization/capture'
    m=json.loads(M_PATH.read_text());selection=json.loads(S_PATH.read_text())
    with (gzip.open(R_PATH, 'rb') if R_PATH.suffix == '.gz' else R_PATH.open('rb')) as handle:raw_bytes=handle.read()
    raw_sha256=hashlib.sha256(raw_bytes).hexdigest()
    rows=[json.loads(x) for x in raw_bytes.splitlines() if x.strip()]
    digest=canonical({k:v for k,v in m.items() if k!='freeze'})
    require(digest==m['freeze']['sha256']==selection['config_hash'],'manifest/selection hash')
    require(json.loads((ROOT/'provenance/executed-manifest.json').read_text())==m,'executed manifest differs')
    require(len(selection['record_files_sha256'])==1 and raw_sha256 in selection['record_files_sha256'].values(),'selection original decompressed raw hash')
    mat=m['matrix']; fields=('block_id','domain','batch_tokens','repetition','candidate_id')
    expected=set(itertools.product(mat['block_ids'],mat['domains'],mat['batch_tokens'],range(mat['repetitions']),mat['candidate_ids']))
    index={tuple(r[f] for f in fields):r for r in rows};require(len(index)==len(rows)==192 and set(index)==expected,'matrix/duplicates')
    docs={b['block_id']:b['source_document_ids'] for b in m['blocks']};banks={b['block_id']:b for b in m['banks']}
    require(len({r['record_id'] for r in rows})==len(rows),'record IDs duplicate')
    file_pins=m['pinning']['optimization_file_sha256'];require(canonical(file_pins)==m['environment']['optimization_sha256'],'file pin digest')
    for path,h in file_pins.items():require(sha(REPO/path)==h,'current frozen source differs '+path)
    configs=m['tuning']['candidate_configs'];differences=[];rankdifferences=[]
    def compare(a,b,label):
        difference=abs(a-b);require(difference<=1e-12,label);differences.append(difference)
    # Build the randomized, position-balanced schedule from the frozen seed.
    groups=list(itertools.product(mat['block_ids'],mat['domains'],mat['batch_tokens'],range(mat['repetitions'])))
    orders=[mat['candidate_ids'][i%8:]+mat['candidate_ids'][:i%8] for i in range(len(groups))]
    random.Random(m['execution']['order_seed']).shuffle(orders);order_index=dict(zip(groups,orders))
    position_counts={c:collections.Counter() for c in mat['candidate_ids']};correctness={};identities={}
    for key,r in index.items():
        block,domain,batch,rep,candidate=key
        expected_meta={'config_hash':digest,'optimization_schema_version':1,'phase':'tuning','split':'pilot','measurement_kind':'gpu_kernel_tuning',
           'status':'complete','evidence_eligible':True,'correctness_pass':True,'steps_completed':32,'warmup_forwards':16,
           'raw_input_positions':48,'timed_range_half_open':[16,48],'policy':'SKIP','native_observation_enabled':False,
           'native_rebalance_steps_called':0,'migration_bytes':0,'kernel_config_rank_agreement':True,**m['required_record_metadata']}
        for f,v in expected_meta.items():require(r.get(f)==v,'metadata '+f)
        require(r['optimization_file_sha256']==file_pins,'record source file pins')
        require(r['weights_source']['weights_sha256']==m['environment']['weights_sha256'],'checkpoint hash')
        require(r['source_prompt_ids']==docs[block] and r['capture_sha256']==banks[block]['capture_sha256'],'bank/source IDs')
        require(r['candidate_order']==order_index[key[:4]],'candidate schedule')
        require(r['candidate_order_position']==r['candidate_order'].index(candidate),'candidate position')
        position_counts[candidate][r['candidate_order_position']]+=1
        identity={'config_hash':digest,**{f:r[f] for f in fields[:4]},'candidate_order':r['candidate_order'],'candidate_id':candidate}
        require(canonical(identity)==r['record_id'],'record ID content hash')
        cfg=configs[candidate]
        require(r['kernel_config_id']==candidate and r['kernel_override_config']==cfg and r['kernel_config_sha256']==canonical(cfg),'override config hash')
        require(r['default_resolved_kernel_config']==m['default_configs'][str(batch)],'resolved default')
        require(r['resolved_kernel_config']==(cfg if cfg is not None else m['default_configs'][str(batch)]),'resolved candidate')
        require(r['source_map']==list(range(128))+list(range(32)),'initial mapping')
        require(hashlib.sha256(struct.pack('<160q',*r['source_map'])).hexdigest()==r['source_map_sha256'],'map hash')
        require(hashlib.sha256(json.dumps(r['source_weight_rank_sha256']).encode()).hexdigest()==r['source_weights_sha256'],'rank weight hash combination')
        times=r['per_rank_elapsed_seconds'];require(len(times)==2 and all(type(t) in (int,float) and math.isfinite(t) and t>0 for t in times),'rank times')
        compare(max(times),r['elapsed_seconds'],'max rank elapsed');compare(max(times)/32,r['elapsed_seconds_per_forward'],'time per forward')
        rankdifferences.append(abs(times[0]-times[1]))
        state=key[:3];ids=tuple(r[f] for f in ('capture_sha256','replay_sha256','source_map_sha256','source_weights_sha256','allocation_id','hostname','process_run_id'))
        require(state not in identities or identities[state]==ids,'source identity across candidates/reps');identities[state]=ids
        checks=r['correctness_checks'];require([c['raw_step'] for c in checks]==[16,47],'reference endpoints')
        require(r['correctness_atol']==.02 and r['correctness_rtol']==.03,'numeric tolerance')
        require(r['correctness_checked_this_trial']==(rep==0),'numeric check reuse label')
        for c in checks:
            require(c['full_batch_default_comparison'] is True and c['reference_tokens']==16 and len(set(c['reference_token_positions']))==16
               and c['native_forward_batch_tokens']==batch,'numeric reference coverage')
            for field in ('fp32_max_abs_error','full_batch_default_max_abs_difference'):
                require(math.isfinite(c[field]) and c[field]>=0,'invalid error summary')
        compare(max(c['fp32_max_abs_error'] for c in checks),r['correctness_max_abs_error'],'maximum error summary')
        ckey=(*state,candidate)
        require(ckey not in correctness or correctness[ckey]==checks,'cached correctness changed');correctness[ckey]=checks
    require(all(set(c.values())=={3} and len(c)==8 for c in position_counts.values()),'global order balance')
    # Verify actual saved pilot banks and eight selection artifacts, only pilot data.
    artifact_checks=[]
    for block,entry in banks.items():require(sha(CAPTURE/entry['capture_path'])==entry['capture_sha256'],'actual bank bytes')
    for block,domain,batch in itertools.product(mat['block_ids'],mat['domains'],mat['batch_tokens']):
        r=index[block,domain,batch,0,'DEFAULT'];artifact=r['input_artifact'];path=ROOT/'tuning-inputs'/Path(artifact['path']).name
        require(path.stat().st_size==artifact['bytes'] and sha(path)==artifact['sha256']==r['replay_sha256'],'input artifact bytes')
        with np.load(path,allow_pickle=False) as f:indices=f['indices'];meta=json.loads(str(f['metadata']))
        require(meta==artifact['metadata'] and canonical(meta)==path.stem,'saved metadata identity')
        require(indices.shape==(48,batch) and hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest()==artifact['indices_sha256'],'actual indices hash/shape')
        with np.load(CAPTURE/banks[block]['capture_path'],allow_pickle=False) as bank:
            domain_ids=bank['domain_ids'];bank_meta=json.loads(str(bank['metadata']))
            require(bank_meta['source_prompt_ids']==docs[block] and bank_meta['split']=='pilot','bank source/split')
            require(np.all(domain_ids[indices]==domain) and all(len(set(i))==batch for i in indices),'sampling domain/no replacement')
        seed=(10471+int(hashlib.sha256(json.dumps([block,domain,batch],separators=(',',':')).encode()).hexdigest()[:16],16))%(2**64)
        require(seed==meta['seed'],'input seed')
        rng=np.random.default_rng(seed);pool=np.flatnonzero(domain_ids==domain)
        regenerated=np.stack([rng.choice(pool,size=batch,replace=False) for _ in range(48)])
        require(np.array_equal(indices,regenerated),'deterministic saved input selection')
        artifact_checks.append({'block_id':block,'domain':domain,'batch_tokens':batch,'file':str(path),'sha256':artifact['sha256'],
            'unique_records_across_48_inputs':int(len(set(indices.ravel()))),'timed_forwards':32,'conditioning_forwards':16})
    reported={(r['batch_tokens'],r['candidate_id']):r for r in selection['candidate_results']}
    require(len(reported)==16,'reported candidate inventory')
    results=[];chosen={}
    for batch in mat['batch_tokens']:
        options=[]
        for candidate in mat['candidate_ids']:
            cells=[];paired_ratios=[]
            for block,domain in itertools.product(mat['block_ids'],mat['domains']):
                values=[]
                for rep in range(mat['repetitions']):
                    value=math.log(index[block,domain,batch,rep,candidate]['elapsed_seconds'])-math.log(index[block,domain,batch,rep,'DEFAULT']['elapsed_seconds'])
                    values.append(value);paired_ratios.append(math.exp(value))
                log=average(values);ratio=math.exp(log)
                cells.append({'block_id':block,'domain':domain,'ratio':ratio,'log_ratio':log})
            ratio=math.exp(average([c['log_ratio'] for c in cells]));eligible=all(c['ratio']<=m['tuning']['max_pilot_cell_ratio'] for c in cells)
            old=reported[batch,candidate];require(old['eligible']==eligible and old['issues']==[],'candidate eligibility')
            compare(ratio,old['ratio'],'candidate aggregate')
            require(len(old['cells'])==4,'candidate cells')
            for cell,previous in zip(cells,old['cells']):
                require(cell['block_id']==previous['block_id'] and cell['domain']==previous['domain'],'cell identity')
                compare(cell['log_ratio'],previous['log_ratio'],'cell log ratio');compare(cell['ratio'],previous['ratio'],'cell ratio')
            result={'batch_tokens':batch,'candidate_id':candidate,'ratio':ratio,'eligible':eligible,'cells':cells,
                'technical_paired_ratio_range':[min(paired_ratios),max(paired_ratios)],'technical_repeats_below_one':sum(v<1 for v in paired_ratios)}
            results.append(result)
            if eligible:options.append(result)
        fastest=min(options,key=lambda r:(r['ratio'],mat['candidate_ids'].index(r['candidate_id'])))
        choice='DEFAULT' if 1<=fastest['ratio']*m['tuning']['tie_ratio'] else fastest['candidate_id']
        chosen[str(batch)]={'candidate_id':choice,'config':configs[choice]}
    require(chosen==selection['selected_configs'],'independent selection differs')
    allchecks=[c for checks in correctness.values() for c in checks]
    error_by_candidate=[]
    for batch,candidate in itertools.product(mat['batch_tokens'],mat['candidate_ids']):
        cs=[c for (block,domain,b,cand),checks in correctness.items() if b==batch and cand==candidate for c in checks]
        error_by_candidate.append({'batch_tokens':batch,'candidate_id':candidate,'fp32_max_abs_error':max(c['fp32_max_abs_error'] for c in cs),
            'full_batch_default_max_abs_difference':max(c['full_batch_default_max_abs_difference'] for c in cs)})
    audit={'status':'verified','manifest_sha256':digest,'optimization_sha256':m['environment']['optimization_sha256'],
     'input_files_sha256':{str(p):sha(p) for p in (M_PATH,R_PATH,S_PATH)},'records_decompressed_sha256':raw_sha256,
     'record_input_compression':'gzip' if R_PATH.suffix=='.gz' else 'none','audit_script_sha256':sha(__file__),
     'records':len(rows),'unique_record_ids':len({r['record_id'] for r in rows}),'source_blocks':2,'block_domain_batch_cells':8,
     'candidate_options':8,'technical_repetitions':3,'unique_correctness_candidate_cells':len(correctness),'endpoint_correctness_checks':len(allchecks),
     'allocation_ids':sorted({r['allocation_id'] for r in rows}),'hostnames':sorted({r['hostname'] for r in rows}),
     'process_run_ids':sorted({r['process_run_id'] for r in rows}),'source_commit':(ROOT/'provenance/source-commit.txt').read_text().strip(),
     'selection':chosen,'candidate_results':results,'maximum_reconstruction_difference':max(differences),
     'maximum_rank_elapsed_difference_seconds':max(rankdifferences),'candidate_position_counts':position_counts,
     'actual_saved_input_artifacts':artifact_checks,'correctness':{'maximum_fp32_absolute_error':max(c['fp32_max_abs_error'] for c in allchecks),
     'maximum_full_batch_default_absolute_difference':max(c['full_batch_default_max_abs_difference'] for c in allchecks),
     'atol':.02,'rtol':.03,'by_candidate':error_by_candidate,
     'scope':'64 candidate/cell checks, two timed endpoints each; full DEFAULT comparison plus 16 FP32 reference tokens per endpoint, cached across three repetitions. This audit checks recorded errors and flags, not fresh numerical execution.'},
     'warnings':['The weights_source.replay_scope string is inherited legacy text saying synthetic activations/routes; tuning input metadata and verified saved capture indices identify actual teacher-forced captured inputs.',
     'Pilot has two hand-selected source-document pairs, not enough for final eight-block inference; no significance or population claim.',
     'Selection is on the same pilot observations and has selection optimism; final and validation remain independent of the tuning criterion.',
     'B1024 TUNED is DEFAULT, so its final kernel-label contrast is a planned identical-configuration control.',
     'Selected B128 changes N128 to N64 while retaining M64; it is not evidence that smaller M tiles reduce padding and improve this measured workload.'],
     'scope':'Independent arithmetic and provenance audit by the same team, no measured-package imports. Standard library statistical calculation; NumPy only reads and regenerates saved pilot input indices. No final data inspected and no GPU rerun.'}
    return audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--records',type=Path,required=True,help='One complete .jsonl or .jsonl.gz file')
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--tuning-dir',type=Path,required=True,help='Directory containing provenance/ and tuning-inputs/')
    parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--capture-dir',type=Path,help='Defaults to REPO/results/optimization/capture')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Refusing to overwrite an audit output')
    result=audit(args.manifest,args.records,args.selection,args.tuning_dir,args.repo_root,args.capture_dir)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result,handle,indent=2,allow_nan=False);handle.write('\n')
    print(json.dumps({key:result[key] for key in ('status','records','selection','maximum_reconstruction_difference','records_decompressed_sha256')}))


if __name__=='__main__':main()

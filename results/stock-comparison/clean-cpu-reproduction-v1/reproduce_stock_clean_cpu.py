#!/usr/bin/env python3
"""Complete final-only CPU reconstruction in an already extracted clean source.

Run only after the parent confirms all final data and production analyses are
complete. This script never imports production code or launches GPU work itself;
it invokes pinned CPU analysis/auditors as separate processes in the clean tree.
"""
import argparse,collections,datetime,gzip,hashlib,json,math,os,shutil,subprocess,sys,time
from pathlib import Path

COMMIT='dfd419f7b4e62fdeec408581c572bf4d3a7a3065'
FINAL_HASH='96b7cce1bb268e038991f13740d426405708bf3962e76fcacfef7aad4bcaa46e'
SOURCE_HASH='9c6d6a7e1e94f4be769d435fbf7652e0cbb86501637c046ae078a0ccef1015d4'
STAT_PATH='results/stock-comparison/reproduction/verify_stock_final_statistics.py'

def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(x):return hashlib.sha256(x if isinstance(x,bytes) else canon(x)).hexdigest()
def file_sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def require(c,m):
    if not c:raise ValueError(m)
def load(p):return json.loads(p.read_text())
def plan_check(p):
    require(p['content_sha256']==sha({k:v for k,v in p.items() if k!='content_sha256'}),'Plan hash mismatch')
    require(p['source_sha256']==SOURCE_HASH==sha(p['source_files']),'Wrong source inventory')

def record_inventory(root,expected,phase):
    """Completeness/identity only; no latency summaries from partial input."""
    files=sorted((root/'trials').glob('*.jsonl*'));require(bool(files),'Missing trial directory '+str(root))
    keys=[]
    for file in files:
        raw=file.read_bytes();raw=gzip.decompress(raw) if file.suffix=='.gz' else raw
        require(raw.endswith(b'\n'),'Incomplete trailing record '+str(file))
        for line in raw.splitlines():
            if not line.strip():continue
            row=json.loads(line)
            require(row['phase']==phase and row['status']=='complete','Incomplete/wrong-phase row')
            keys.append((row['content_group_id'],row['arrival_type'],row['rho'],row['arm']))
    require(len(keys)==len(set(keys))==expected,'Incomplete/duplicate matrix '+str(root))
    return len(keys)

def compare_analysis(original,reproduced):
    a=load(original);b=load(reproduced)
    fa=a.pop('record_files');fb=b.pop('record_files')
    require(sorted(x['uncompressed_bytes_sha256'] for x in fa)==sorted(x['uncompressed_bytes_sha256'] for x in fb),'Raw decompressed record digests differ')
    count=0;max_abs=0.;max_relative=0.
    def walk(x,y,path):
        nonlocal count,max_abs,max_relative
        if isinstance(x,dict):
            require(isinstance(y,dict) and x.keys()==y.keys(),path+' object keys')
            for k in x:walk(x[k],y[k],path+'.'+k)
        elif isinstance(x,list):
            require(isinstance(y,list) and len(x)==len(y),path+' list length')
            for i,(u,v) in enumerate(zip(x,y)):walk(u,v,path+f'[{i}]')
        elif isinstance(x,float):
            require(not isinstance(y,bool) and isinstance(y,(int,float)) and math.isfinite(x) and math.isfinite(y),path+' invalid numeric value')
            delta=abs(x-y);max_abs=max(max_abs,delta);max_relative=max(max_relative,delta/max(abs(x),abs(y),1.))
            require(math.isclose(x,y,rel_tol=2e-12,abs_tol=2e-10),path+' numeric mismatch');count+=1
        else:require(type(x) is type(y) and x==y,path+' exact identity mismatch');count+=1
    walk(a,b,'analysis')
    return {'passed':True,'compared_scalar_fields':count,'maximum_absolute_float_difference':max_abs,'maximum_relative_float_difference':max_relative,'normalization':'Only top-level record_files path/compression metadata removed, after exact decompressed-byte digest multiset comparison. Other identities/counts exact; float tolerance2e-12 relative,2e-10 absolute.'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('source-root','repo-root','artifact-root','python','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--final-complete-confirmed',action='store_true',help='Required explicit execution guard; never use for a partial final matrix')
    a=p.parse_args();require(a.final_complete_confirmed,'Await explicit final-data completion notification')
    clean=a.source_root.resolve();repo=a.repo_root.resolve();art=a.artifact_root.resolve();output=a.output.resolve();python=Path(os.path.abspath(a.python))
    require(not output.exists(),'Fresh reproduction output required')
    final=art/'final-v1';cal=art/'calibration-final-v1'
    final_plan=load(final/'provenance/plan.json');cal_plan=load(cal/'provenance/plan.json')
    plan_check(final_plan);plan_check(cal_plan);require(final_plan['content_sha256']==FINAL_HASH,'Unexpected final plan')
    counts={'calibration':record_inventory(cal,18,'calibration'),'final':record_inventory(final,216,'final')}
    require(load(final/'analysis.json')['valid'] is True and load(cal/'analysis.json')['valid'] is True,'Missing/invalid complete production analyses')
    for rel,pin in final_plan['source_files'].items():require(file_sha(clean/rel)==pin,'Clean source differs: '+rel)
    require(cal_plan['source_files']==final_plan['source_files'],'Calibration and final source inventories differ')
    protocol=clean/'docs/stock-comparison/protocol-v1.md'
    require(file_sha(protocol)==final_plan['protocol_sha256'],'Clean protocol differs')
    full_commit=subprocess.check_output(['git','rev-parse',COMMIT],cwd=repo,text=True).strip();require(full_commit==COMMIT,'Archive commit missing')
    stat_bytes=subprocess.check_output(['git','show',COMMIT+':'+STAT_PATH],cwd=repo)
    output.mkdir(parents=True);logs=output/'logs';logs.mkdir();reports=output/'reports';reports.mkdir();data=output/'data';data.mkdir();auditors=output/'auditors';auditors.mkdir()
    stat=auditors/'verify_stock_final_statistics.py';stat.write_bytes(stat_bytes)
    staged={};original={};methods=collections.Counter()
    for stage in ('calibration-final-v1','final-v1'):
        source=art/stage;dest=data/stage
        selected=[p for child in ('trials','inputs','provenance') for p in (source/child).rglob('*') if p.is_file()]
        selected+=[source/'analysis.json']
        for path in sorted(selected):
            rel=path.relative_to(source);target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True)
            try:os.link(path,target);method='hardlink'
            except OSError:shutil.copy2(path,target);method='copy'
            key=str(Path(stage)/rel);h=file_sha(path)
            require(file_sha(target)==h,'Staged artifact byte mismatch')
            staged[key]={'sha256':h,'bytes':path.stat().st_size,'method':method};original[key]=path;methods[method]+=1
    (output/'staged-artifact-manifest.json').write_text(json.dumps(staged,indent=2)+'\n')
    env=os.environ.copy();env['PYTHONPATH']='';env['PYTHONNOUSERSITE']='1';env['PYTHONDONTWRITEBYTECODE']='1'
    commands=[]
    def execute(label,argv):
        print(json.dumps({'phase':label,'status':'running'}),flush=True)
        t=time.monotonic();run=subprocess.run([str(x) for x in argv],cwd=clean,env=env,text=True,capture_output=True,timeout=3600)
        (logs/(label+'.stdout.txt')).write_text(run.stdout);(logs/(label+'.stderr.txt')).write_text(run.stderr)
        tokens=[str(x) for x in argv]
        normalized=[]
        for token in tokens:
            for prefix,name in ((str(clean),'$CLEAN_SOURCE'),(str(output),'$REPRO_OUTPUT'),(str(python),'$CPU_PYTHON')):token=token.replace(prefix,name)
            normalized.append(token)
        commands.append({'label':label,'argv':normalized,'argv_sha256':sha(tokens),'working_directory':'$CLEAN_SOURCE','returncode':run.returncode,'elapsed_seconds':time.monotonic()-t,'stdout_sha256':file_sha(logs/(label+'.stdout.txt')),'stderr_sha256':file_sha(logs/(label+'.stderr.txt'))})
        require(run.returncode==0,label+' failed; inspect saved logs')
        print(json.dumps({'phase':label,'status':'complete','elapsed_seconds':commands[-1]['elapsed_seconds']}),flush=True)
    for stage,short in (('calibration-final-v1','calibration'),('final-v1','final')):
        run=data/stage
        execute(short+'-analysis',[python,clean/'scripts/analyze_stock_comparison.py','--plan',run/'provenance/plan.json','--records',run,'--output',reports/(short+'-analysis.json')])
        args=[python,clean/'scripts/audit_stock_comparison_records.py','--plan',run/'provenance/plan.json','--records',run,'--inputs-dir',run/'inputs','--repo-root',clean,'--output',reports/(short+'-execution-audit.json')]
        if short=='final':args+=['--calibration',data/'calibration-final-v1']
        execute(short+'-execution-audit',args)
    execute('final-independent-statistics',[python,stat,'--plan',data/'final-v1/provenance/plan.json','--records',data/'final-v1','--summary',reports/'final-analysis.json','--protocol',protocol,'--output',reports/'final-statistics-audit.json'])
    comparisons={short:compare_analysis(data/stage/'analysis.json',reports/(short+'-analysis.json')) for stage,short in (('calibration-final-v1','calibration'),('final-v1','final'))}
    audits={short:load(reports/(short+'-execution-audit.json')) for short in ('calibration','final')}
    require(all(r['valid'] for r in audits.values()),'Independent execution audit invalid')
    stats=load(reports/'final-statistics-audit.json');require(stats['passed'],'Independent statistics audit invalid')
    for key,meta in staged.items():require(file_sha(data/key)==meta['sha256']==file_sha(original[key]),'Source/staged artifact changed during reproduction')
    for rel,pin in final_plan['source_files'].items():require(file_sha(clean/rel)==pin,'Clean execution source changed during reconstruction')
    outputs={str(p.relative_to(output)):file_sha(p) for p in sorted(reports.iterdir()) if p.is_file()}
    result={'schema':'stock_clean_cpu_reproduction_v1','passed':True,'source_archive_commit':COMMIT,'source_manifest_sha256':SOURCE_HASH,'frozen_source_files_verified':len(final_plan['source_files']),'final_plan_sha256':FINAL_HASH,'protocol_sha256':final_plan['protocol_sha256'],'complete_trial_counts':counts,'staged_artifact_methods':dict(methods),'staged_artifact_manifest_sha256':file_sha(output/'staged-artifact-manifest.json'),'artifact_files':len(staged),'commands':commands,'analysis_comparisons':comparisons,'independent_execution_audits':{k:{field:r[field] for field in ('valid','expected_records','observed_records','raw_clock_records_checked','groups_with_authentic_stock_routes')} for k,r in audits.items()},'independent_statistics':{'passed':stats['passed'],'program_sha256':stats['program_sha256'],'comparison':stats['comparison'],'draw_indices_sha256':stats['recomputed']['confirmatory']['bootstrap']['draw_indices_sha256']},'output_sha256':outputs,'reproduction_harness_sha256':file_sha(Path(__file__)),'provenance':'Source was independently extracted from pinned Git commit; final/calibration records and input artifacts were copied or hardlinked from the completed local archive, with byte hashes verified before and after. No independent data acquisition or GPU execution.','limitations':['Same-team CPU reconstruction, not external-laboratory replication.','Copied/hardlinked input evidence is shared with the original archive; independent arithmetic does not establish an independent physical measurement.','No new GPU timings, workload selection, parameter changes or statistical endpoint changes.','Final scientific interpretation remains conditional on the frozen workload, hardware and protocol.']}
    (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'trial_counts':counts,'analysis_comparisons':comparisons,'output':'verification.json'}),flush=True)

if __name__=='__main__':main()

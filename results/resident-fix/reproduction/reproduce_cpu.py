#!/usr/bin/env python3
"""Replay frozen resident-fix analyses and independent audits without GPU work.

Writes only to a new output directory. Archived summaries remain unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
PHASES={
    'calibration-pilot':('pilot-calibration-v1.json','calibration-pilot-v1','capture-pilot-v1',None),
    'pilot':('pilot-v1.json','pilot-v1','capture-pilot-v1','calibration-pilot-v1'),
    'calibration-final':('final-calibration-v1.json','calibration-final-v1','capture-final-v1',None),
    'final':('final-v1.json','final-v1','capture-final-v1','calibration-final-v1'),
}


def digest(value):
    raw=value if isinstance(value,bytes) else json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def require(condition,message):
    if not condition:raise ValueError(message)


def check_tools():
    path=HERE/'tools-manifest.json';raw=path.read_bytes();manifest=json.loads(raw)
    require(manifest['schema']=='resident_fix_independent_tools_v1','Wrong independent-tools manifest')
    require(Path(__file__).name in manifest['files'],'Replay helper is not pinned')
    for name,expected in manifest['files'].items():
        require(Path(name).name==name,'Tool manifest paths must stay local')
        require(digest((HERE/name).read_bytes())==expected,'Changed independent tool: '+name)
    return dict(manifest_sha256=digest(raw),files=manifest['files'])


def check_plan(repo,path,run):
    raw=path.read_bytes();plan=json.loads(raw)
    require(plan['content_sha256']==digest({k:v for k,v in plan.items() if k!='content_sha256'}),'Changed plan seal')
    require(plan['source_sha256']==digest(plan['source_files']),'Changed source inventory seal')
    for name,expected in plan['source_files'].items():
        file=(repo/name).resolve();require(file.is_relative_to(repo),'Source path escapes checkout')
        require(digest(file.read_bytes())==expected,'Changed execution source: '+name)
    require(digest((repo/'docs/resident-fix/protocol-v1.md').read_bytes())==plan['protocol_sha256'],'Changed protocol')
    selected=(repo/'configs/resident-fix-selected-v1.json').read_bytes()
    require(digest(selected)==plan['selected_fix_sha256'] and json.loads(selected)==plan['selected_fix']
        and digest(plan['selected_fix'])==plan['selected_fix_content_sha256'],'Changed selected implementation')
    executed=json.loads((run/'provenance/plan.json').read_text())
    require(executed==plan,'Archived executed plan differs from supplied plan')
    require((run/'analysis.json').is_file(),'Archived validated summary missing: '+str(run))
    return dict(plan_sha256=plan['content_sha256'],plan_file_sha256=digest(raw),source_sha256=plan['source_sha256'],source_files=len(plan['source_files']),old_closure_sha256=plan['old_closure_sha256'])


def canonical_analysis(summary):
    """Ignore relocation/compression only; retain all outcomes and original bytes."""
    result=dict(summary);files=[]
    for entry in result.pop('record_files'):
        name=Path(entry['path']).name.removesuffix('.gz')
        files.append(dict(name=name,uncompressed_bytes_sha256=entry['uncompressed_bytes_sha256']))
    require(len({f['name'] for f in files})==len(files),'Duplicate archived trial file identity')
    result['record_files']=sorted(files,key=lambda x:x['name'])
    return result


def compare_archived(generated,archived):
    new_raw=generated.read_bytes();old_raw=archived.read_bytes()
    new,old=canonical_analysis(json.loads(new_raw)),canonical_analysis(json.loads(old_raw))
    require(new.get('valid') is True and old.get('valid') is True,'Cannot reproduce an invalid/incomplete summary')
    require(new==old,'Recomputed summary differs beyond file relocation or gzip packaging')
    return dict(passed=True,comparison='Exact JSON values after normalizing record paths and stored compression hashes only',
        archived_file_sha256=digest(old_raw),regenerated_file_sha256=digest(new_raw),semantic_sha256=digest(new))


def run_step(argv,cwd,log):
    completed=subprocess.run([sys.executable,*map(str,argv)],cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    log.write_text(completed.stdout)
    if completed.returncode:raise RuntimeError(f'CPU step failed ({completed.returncode}); see {log}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,default=HERE.parents[2])
    parser.add_argument('--results-root',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--phases',nargs='+',choices=tuple(PHASES),default=list(PHASES))
    args=parser.parse_args();repo=args.repo_root.resolve();results=(args.results_root or repo/'results/resident-fix').resolve();out=args.output_dir.resolve()
    require(len(set(args.phases))==len(args.phases),'Duplicate replay phase')
    require(not out.exists(),'Output directory must be new; archived outputs are never overwritten')
    # Refuse placement inside either input trial tree or the authenticated sources.
    for _,run_name,_,_ in PHASES.values():require(not out.is_relative_to(results/run_name),'Output is inside an archived input run')
    for protected in ('resident_fastpath','stock_comparison','scripts','configs','docs'):
        require(not out.is_relative_to(repo/protected),'Output is inside source/protocol/config files')
    tool_identity=check_tools();planned={}
    for phase in args.phases:
        plan_name,run_name,_,_=PHASES[phase]
        planned[phase]=check_plan(repo,results/'plans'/plan_name,results/run_name)
    out.mkdir(parents=True);report=dict(schema='resident_fix_cpu_reproduction_v1',passed=False,tools=tool_identity,phases={},python=sys.version,scope='CPU-only replay of frozen observations; no GPU measurements or new inference specification')
    try:
        import numpy
        report['numpy_version']=numpy.__version__
        for phase in args.phases:
            plan_name,run_name,capture_name,calibration_name=PHASES[phase]
            plan=results/'plans'/plan_name;run=results/run_name;target=out/phase;target.mkdir()
            generated=target/'analysis.json'
            run_step([repo/'scripts/analyze_resident_fix.py','--plan',plan,'--records',run,'--output',generated],repo,target/'production-analysis.log')
            comparison=compare_archived(generated,run/'analysis.json')
            raw_args=[HERE/'audit_resident_fix_records.py','--plan',plan,'--records',run,'--inputs-dir',run/'inputs',
                '--banks-dir',results/capture_name/'banks','--repo-root',repo,'--output',target/'independent-record-audit.json']
            if calibration_name:raw_args.extend(['--calibration',results/calibration_name])
            run_step(raw_args,repo,target/'independent-record-audit.log')
            run_step([HERE/'verify_resident_fix_statistics.py','--plan',plan,'--records',run,'--analysis',generated,
                '--repo-root',repo,'--output',target/'independent-statistics-audit.json'],repo,target/'independent-statistics-audit.log')
            audit=json.loads((target/'independent-record-audit.json').read_text());stats=json.loads((target/'independent-statistics-audit.json').read_text())
            require(audit['valid'] and stats['passed'],'Independent audit did not pass')
            report['phases'][phase]=dict(**planned[phase],archived_analysis_comparison=comparison,
                raw_records=audit['observed_records'],prepared_batches=audit['prepared_batches_reconstructed'],
                assignment_batches=audit['exact_greedy_assignment_batches'],statistics_comparison=stats['comparison'],passed=True)
            print(json.dumps(dict(phase=phase,passed=True,records=audit['observed_records'])),flush=True)
        report['passed']=True
    except Exception as error:
        report['error']=str(error)
        raise
    finally:(out/'reproduction-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()

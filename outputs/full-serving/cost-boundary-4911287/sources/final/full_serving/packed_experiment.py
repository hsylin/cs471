"""Packed small-graph dispatch with unchanged native/gate/resident references.

Diagnostic sessions instrument the same requests separately from final timing.
New algorithms must pass pilot admission before the final protocol is frozen.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from full_serving import orchestrate as native, custom_compare as parent

ROOT=Path(__file__).resolve().parents[1]
ARMS=('off','off_packed','b3_light','b3_light_packed','b4_light','b4_light_packed','resident_v2','resident_v2_packed')


def load(path):
    p=json.loads(Path(path).read_bytes())
    assert p['schema']=='cs471_packed_followup_v1'
    assert p['seal']==parent.canonical_hash({k:v for k,v in p.items() if k!='seal'})
    assert p['arms']==list(ARMS) and p['datasets']==['longbench','numinamath']
    assert p['gate']=={'window':1000,'gain':.05,'tv':.05,'unit':'per_layer','native_interval':3000}
    for dataset,order in p['orders'].items():assert len(order)==len(ARMS) and set(order)==set(ARMS)
    for name,digest in p['files'].items():assert native.sha(ROOT/name)==digest,name
    data_path=ROOT/p['data_protocol'];assert native.sha(data_path)==p['data_protocol_sha256']
    data=parent.load_admitted_protocol(data_path)
    assert data['equal_output_length']
    if p['status']=='frozen':
        admission=ROOT/p['pilot_admission']
        assert native.sha(admission)==p['pilot_admission_sha256']
        evidence=json.loads(admission.read_bytes())
        assert evidence['status']=='passed' and evidence['source_files']==p['files']
    return p,data,data_path


def clean(environment,for_srun=False):
    env=native.server_environment(environment) if for_srun else dict(environment)
    for k in list(env):
        if k.startswith(('FULL_SERVING_CUSTOM_','CS471_')) or k in ('PYTHONPATH','VLLM_PLUGINS'):env.pop(k,None)
    return env


def server_arguments(arm,snapshot,ip,role):
    assert arm in ARMS
    arm=arm.removesuffix('_packed')
    args=parent.native_on_arguments(snapshot,ip,role)
    index=args.index('--enable-eplb')
    if arm=='off':del args[index:index+3]
    elif arm=='b1':args[index+2]='{"num_redundant_experts":0}'
    elif arm not in ARMS:raise ValueError('Unknown arm')
    return args


def node(args):
    spec,data,_=load(args.protocol);assert os.environ.get('SLURM_JOB_ID')==native.JOB
    env=clean(os.environ);directory=Path(env['FULL_SERVING_RUN_DIR']);pilot=env.get('FULL_SERVING_PROPOSAL_PILOT')=='1'
    native.command([sys.executable,'-m','full_serving.preflight','runtime','--snapshot',str(args.snapshot),'--checkpoint-manifest',env['FULL_SERVING_CHECKPOINT_MANIFEST'],'--role',args.role,'--primary-ip',args.ip,'--output',str(directory/f'preflight-{args.role}.json')],directory/f'preflight-{args.role}.log',timeout=3600)
    command=['vllm',*server_arguments(args.arm,args.snapshot,args.ip,args.role)]
    base_arm=args.arm.removesuffix('_packed')
    if base_arm in ('resident_v2','b3_light','b4_light'):
        env.update(CS471_EXPERIMENT_ARM=base_arm,VLLM_PLUGINS='cs471_proposal',PYTHONPATH=os.pathsep.join((str(ROOT),str(ROOT/'full_serving/proposal_bootstrap'))))
        if args.arm.startswith('resident'):
            cfg=ROOT/spec['resident_config_pilot' if pilot else 'resident_config'];sources=ROOT/spec['resident_sources']
            from full_serving.resident_plugin import verify_source_manifest
            verify_source_manifest(json.loads(sources.read_bytes()))
            env.update(FULL_SERVING_CUSTOM_MODE='resident_v1',FULL_SERVING_CUSTOM_CONFIG=str(cfg),FULL_SERVING_CUSTOM_SOURCE_MANIFEST=str(sources),FULL_SERVING_CUSTOM_CONFIG_SHA256=native.sha(cfg),FULL_SERVING_CUSTOM_SOURCE_MANIFEST_SHA256=native.sha(sources))
        else:env.update(CS471_GATE_MODE=base_arm.removesuffix('_light'),CS471_GATE_NATIVE_PINS=str(ROOT/spec['native_pins']))
    if base_arm.endswith('_light'):
        env.update(VLLM_PLUGINS='cs471_monitoring',PYTHONPATH=os.pathsep.join((str(ROOT),str(ROOT/'full_serving/monitoring_bootstrap'))),CS471_GATE_HISTORY=spec['history_implementation'])
        if pilot:env['CS471_GATE_SHADOW']='1'
    if args.arm.endswith('_packed'):
        env.update(CS471_EXPERIMENT_ARM=args.arm,VLLM_PLUGINS='cs471_packed',PYTHONPATH=os.pathsep.join((str(ROOT),str(ROOT/'full_serving/packed_bootstrap'))),CS471_PACKED_NATIVE_PINS=str(ROOT/spec['packed_native_pins']))
        if pilot:env['CS471_PACKED_SHADOW']='1'
    diagnostic=env.get('FULL_SERVING_LATENCY_DIAGNOSTIC')=='1'
    if diagnostic:
        env['PYTHONPATH']=env.get('PYTHONPATH',str(ROOT))
        command+=['--middleware','full_serving.latency_probe.LatencyMiddleware']
    if pilot:
        command+=['--profiler-config',json.dumps({'profiler':'torch','torch_profiler_dir':str(directory/'profiles'),'torch_profiler_with_stack':False,'torch_profiler_activities':['CPU','CUDA'],'ignore_frontend':True,'max_iterations':16})]
    native.write_json(directory/f'effective-launch-{args.role}.json',{'arm':args.arm,'pilot':pilot,'diagnostic':diagnostic,'command':command,'plugin':env.get('VLLM_PLUGINS'),'protocol_sha256':native.sha(args.protocol)})
    os.execvpe(command[0],command,env)


def copy_inputs(data,path,private_root,private_out,out,pilot):
    if not pilot:return parent.admit_and_copy_inputs(data,path,private_root,private_out,out,parent.load_source_metadata(data,path))
    # Pilot must not open final prompt files, even to verify their hashes.
    metadata=parent.load_source_metadata(data,path);inputs={}
    from full_serving.output_lengths import validate_targets
    validate_targets(data,path)
    for dataset in data['datasets']:
        d=dataset['dataset'];item=dataset['warmup'];source=parent.private_path(private_root,item['requests_file'])
        assert native.sha(source)==item['requests_sha256'];rows=native.read_jsonl(source)
        assert len(rows)==8 and [r['id'] for r in rows]==item['request_ids']
        for row in rows:parent.verify_private_request(row,metadata['by_id'][row['id']])
        target=private_out/d/'warmup.jsonl';target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes());target.chmod(0o600)
        lengths=parent.artifact_path(path,item['output_lengths_file']);copy=out/'output-lengths'/f'{d}-warmup.json';copy.parent.mkdir(parents=True,exist_ok=True);copy.write_bytes(lengths.read_bytes())
        inputs[d]={'warmup':{'requests':target,'requests_sha256':item['requests_sha256'],'output_lengths':copy}}
    return inputs


def run(args):
    spec,data,data_path=load(args.protocol);assert os.environ.get('SLURM_JOB_ID')==native.JOB
    assert args.source_revision and len(args.source_revision)==40
    if args.pilot:assert args.arm in ARMS
    elif not args.diagnostic and spec['status']!='frozen':raise RuntimeError('Final protocol not frozen')
    out=args.output.resolve();private=args.private_output.resolve();out.mkdir(parents=True,exist_ok=False);private.mkdir(parents=True,exist_ok=False,mode=0o700)
    jobs=[(d,a) for d in spec['datasets'] for a in ([args.arm] if args.pilot else (spec['diagnostic_arms'] if args.diagnostic else spec['orders'][d]))]
    # One pilot server spans both pilot datasets; final uses fresh server per arm/dataset.
    if args.pilot:jobs=[('pilot',args.arm)]
    state={'schema':'cs471_proposal_status_v1','started_utc':native.stamp(),'pilot_only':args.pilot,'completed_sessions':[],'expected_sessions':len(jobs)}
    def update(**kw):state.update(kw,updated_utc=native.stamp());native.write_json(out/'status.json',state)
    update(state='admission');server=None;directory=None
    try:
        (out/'protocol.json').write_bytes(args.protocol.read_bytes());(out/'data-protocol.json').write_bytes(data_path.read_bytes())
        native.write_json(out/'controller-source.json',{'declared_git_revision':args.source_revision,'files':{str(p.relative_to(ROOT)):native.sha(p) for p in sorted((ROOT/'full_serving').rglob('*')) if p.is_file() and p.suffix in ('.py','.sh','.txt') and '__pycache__' not in p.parts}})
        inputs=copy_inputs(data,data_path,args.private_data,private,out,args.pilot)
        allocation=json.loads(native.command(['scontrol','show','job',native.JOB,'--json'],out/'allocation.json'))['jobs'][0];assert 'RUNNING' in allocation['job_state']
        nodes=native.command(['scontrol','show','hostnames',allocation['nodes']]).split();assert nodes==['kh074','kh077']
        ip=socket.gethostbyname(nodes[0]);url=f'http://{ip}:8000'
        native.write_json(out/'topology.json',{'nodes':nodes,'dp':8,'tp':1,'ep':8,'activation_backend':'allgather_reducescatter'})
        native.command([sys.executable,'-m','full_serving.preflight','checkpoint','--snapshot',str(args.snapshot),'--output',str(out/'checkpoint-verified.json')],out/'checkpoint-preflight.log',timeout=3600)
        for dataset,arm in jobs:
            directory=out/dataset/arm;directory.mkdir(parents=True);priv=private/dataset/arm;priv.mkdir(parents=True,mode=0o700)
            update(state='starting_server',active_session=f'{dataset}/{arm}',active_phase=None)
            env=clean(os.environ,True);env.update(SLURM_JOB_NODELIST=allocation['nodes'],SLURM_JOB_NUM_NODES='2',FULL_SERVING_RUN_DIR=str(directory),FULL_SERVING_CHECKPOINT_MANIFEST=str(out/'checkpoint-verified.json'),FULL_SERVING_PROPOSAL_PILOT=str(int(args.pilot)),FULL_SERVING_LATENCY_DIAGNOSTIC=str(int(args.diagnostic)))
            cmd=['srun','--jobid='+native.JOB,'--mpi=none','--exact','--nodes=2','--ntasks=2','--ntasks-per-node=1','--nodelist='+','.join(nodes),'--cpus-per-task=56','--gpus-per-node=4','--mem=300G','--kill-on-bad-exit=1','--input=none','--output='+str(directory/'server-%t.log'),'--error='+str(directory/'server-%t.err'),'bash',str(ROOT/'full_serving/packed_node.sh'),arm,str(args.snapshot),ip,str(args.protocol.resolve())]
            native.write_json(directory/'launch.json',{'command':cmd,'arm':arm,'dataset':dataset,'started_utc':native.stamp()})
            with (directory/'srun.log').open('w') as log:
                begin=time.monotonic();server=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                native.wait_server(server,url,2400,directory);native.write_json(directory/'startup.json',{'elapsed_s':time.monotonic()-begin,'ready_utc':native.stamp()})
                if arm.startswith('resident'):
                    for rank in range(8):
                        binding=json.loads((directory/f'resident-bindings-rank{rank}.json').read_bytes());assert binding['layer_count']==48 and binding['ep_rank']==rank
                        if args.pilot:assert json.loads((directory/f'resident-startup-diagnostic-rank{rank}.json').read_bytes())['status']=='passed'
                for source in (spec['datasets'] if args.pilot else [dataset]):
                    meta=next(d for d in data['datasets'] if d['dataset']==source);block=meta['blocks'][0]
                    for phase in ('warmup-00','warmup-01','profile') if args.pilot else ('warmup-00','warmup-01','measured'):
                        measured=phase=='measured';label=f'{source}-{phase}' if args.pilot else phase;item=inputs[source][block['block_id'] if measured else 'warmup']
                        assert native.sha(item['requests'])==item['requests_sha256']
                        update(state='running_requests',active_phase=label)
                        profiling=args.pilot and phase=='profile'
                        if profiling:
                            req=urllib.request.Request(url+'/start_profile',method='POST');urllib.request.urlopen(req,timeout=120).read()
                        (directory/f'{label}-metrics-before.txt').write_bytes(native.get_url(url+'/metrics'))
                        client=[sys.executable,'-m','full_serving.client','--requests',str(item['requests']),'--base-url',url+'/v1','--output',str(directory/f'{label}.jsonl'),'--private-output',str(priv/f'{label}-outputs.jsonl'),'--run-id',f'{source}-{arm}-{phase}','--arm',arm,'--block-id',source+'-'+block['block_id'],'--load-id',phase,'--pair-id',source+'-'+block['block_id']+'-'+phase,'--max-inflight','64','--timeout-s','1800','--max-tokens',str(parent.generation_for(data,source)['max_tokens']),'--output-lengths',str(item['output_lengths'])]
                        client+=['--calendar',str(item['calendar'])] if measured else ['--rate','0']
                        if phase=='warmup-00':client+=['--logprobs']
                        native.command(client,directory/f'{label}-client.log',timeout=7200)
                        s=native.summary(directory/f'{label}.jsonl');assert s['completed_requests']==(16 if measured else 8) and not s['failed_requests']
                        (directory/f'{label}-metrics-after.txt').write_bytes(native.get_url(url+'/metrics'))
                        if profiling:urllib.request.urlopen(urllib.request.Request(url+'/stop_profile',method='POST'),timeout=300).read()
                update(state='stopping_server');native.stop_server(server,directory);server=None
            native.write_json(directory/'status.json',{'state':'session_completed','arm':arm,'dataset':dataset,'pilot_only':args.pilot})
            state['completed_sessions'].append(f'{dataset}/{arm}');update(state='session_completed')
        update(state='comparison_completed',completed_utc=native.stamp())
    except BaseException as exc:
        update(state='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        if server is not None:native.stop_server(server,directory)


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    n=sub.add_parser('node');n.add_argument('arm',choices=ARMS);n.add_argument('role',choices=('primary','worker'));n.add_argument('snapshot',type=Path);n.add_argument('ip');n.add_argument('--protocol',type=Path,required=True)
    r=sub.add_parser('run')
    for name in ('protocol','private-data','private-output','snapshot','output'):r.add_argument('--'+name,type=Path,required=True)
    r.add_argument('--source-revision',required=True);r.add_argument('--diagnostic',action='store_true');r.add_argument('--pilot',action='store_true');r.add_argument('--arm',choices=ARMS)
    args=p.parse_args()
    def stop(signum,frame):raise InterruptedError(f'Controller signal {signum}')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    (node if args.action=='node' else run)(args)
if __name__=='__main__':main()

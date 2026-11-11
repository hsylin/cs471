"""Sustained natural-EOS and matched-length sessions; earlier runners unchanged."""
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
ARMS=('off','b1','b2','b3','b4','resident_v2')


def load(path):
    p=json.loads(Path(path).read_bytes())
    assert p['schema']=='cs471_sustained_v1'
    assert p['seal']==parent.canonical_hash({k:v for k,v in p.items() if k!='seal'})
    assert p['arms']==list(ARMS)
    for name,digest in p['files'].items():assert native.sha(ROOT/name)==digest,name
    data_path=ROOT/p['data_manifest'];assert native.sha(data_path)==p['data_manifest_sha256']
    data=json.loads(data_path.read_bytes())
    from full_serving.data import digest
    assert data['seal']==digest({k:v for k,v in data.items() if k!='seal'})
    assert data['max_tokens']==8192
    for dataset in ('longbench','numinamath'):
        assert data['datasets'][dataset]['items']['measured']['request_count']==512
        assert set(p['orders'][dataset])==set(ARMS) and len(p['orders'][dataset])==6
    return p,data,data_path


def clean(environment,for_srun=False):
    env=native.server_environment(environment) if for_srun else dict(environment)
    for k in list(env):
        if k.startswith(('FULL_SERVING_CUSTOM_','CS471_')) or k in ('PYTHONPATH','VLLM_PLUGINS'):env.pop(k,None)
    return env


def server_arguments(arm,snapshot,ip,role):
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
    if args.arm!='off':
        env.update(CS471_SUSTAINED_ARM=args.arm,CS471_EXPERIMENT_ARM=args.arm,
            VLLM_PLUGINS='cs471_sustained',PYTHONPATH=os.pathsep.join((str(ROOT),str(ROOT/'full_serving/sustained_bootstrap'))),
            CS471_GATE_NATIVE_PINS=str(ROOT/spec['native_pins']))
        if args.arm=='resident_v2':
            cfg=ROOT/spec['resident_config'];sources=ROOT/spec['resident_sources']
            from full_serving.resident_plugin import verify_source_manifest
            verify_source_manifest(json.loads(sources.read_bytes()))
            env.update(FULL_SERVING_CUSTOM_MODE='resident_v1',FULL_SERVING_CUSTOM_CONFIG=str(cfg),FULL_SERVING_CUSTOM_SOURCE_MANIFEST=str(sources),FULL_SERVING_CUSTOM_CONFIG_SHA256=native.sha(cfg),FULL_SERVING_CUSTOM_SOURCE_MANIFEST_SHA256=native.sha(sources))
        if args.arm in ('b3','b4'):
            env.update(CS471_GATE_MODE=args.arm,CS471_GATE_HISTORY='reduction')
            if pilot:env['CS471_GATE_SHADOW']='1'
        branch=os.environ.get('FULL_SERVING_PAYBACK_BRANCH')
        if branch:env.update(CS471_PAYBACK_BRANCH=branch,CS471_PAYBACK_STEP='3750')
    native.write_json(directory/f'effective-launch-{args.role}.json',{'arm':args.arm,'pilot':pilot,'command':command,'plugin':env.get('VLLM_PLUGINS'),'protocol_sha256':native.sha(args.protocol)})
    os.execvpe(command[0],command,env)


def copy_inputs(data,path,private_root,private_out,out,pilot):
    inputs={}
    for dataset,meta in data['datasets'].items():
        inputs[dataset]={}
        by_id={r['id']:r for r in meta['requests']}
        for label in (('warmup',) if pilot else ('warmup','measured')):
            item=meta['items'][label];source=private_root/item['requests_file']
            assert native.sha(source)==item['requests_sha256']
            rows=native.read_jsonl(source)
            assert [r['id'] for r in rows]==item['request_ids']
            from full_serving.data import digest
            for row in rows:
                assert digest(row['messages'])==by_id[row['id']]['messages_sha256']
                assert row['input_tokens']+8192<=32768
            target=private_out/dataset/(label+'.jsonl');target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(source.read_bytes());target.chmod(0o600)
            entry={'requests':target,'requests_sha256':item['requests_sha256']}
            if label=='measured':
                calendar=path.parent/item['calendar_file'];assert native.sha(calendar)==item['calendar_sha256']
                dest=out/(dataset+'-calendar.json');dest.write_bytes(calendar.read_bytes());entry['calendar']=dest
            inputs[dataset][label]=entry
    return inputs


def matched_targets(reference,dataset,label,item,output):
    from full_serving.output_lengths import reference_targets
    source=reference/dataset/'off'/(label+'.jsonl')
    targets=reference_targets(source.read_bytes(),item,8192)
    target=output/(dataset+'-'+label+'-lengths.json')
    native.write_json(target,targets)
    native.write_json(output/(dataset+'-'+label+'-length-reference.json'),
        {'reference_sha256':native.sha(source),'targets_sha256':native.sha(target),
         'reference':str(source),'total_tokens':sum(targets.values()),'request_count':len(targets)})
    return target


def run(args):
    spec,data,data_path=load(args.protocol);assert os.environ.get('SLURM_JOB_ID')==native.JOB
    assert args.source_revision and len(args.source_revision)==40
    if args.pilot:assert args.arm in ARMS
    elif spec['status']!='frozen':raise RuntimeError('Final protocol not frozen')
    if not args.pilot:
        assert args.pilot_admission is not None
        admission=json.loads(args.pilot_admission.read_text())
        assert admission['status']=='passed' and admission['protocol_sha256']==native.sha(args.protocol)
        assert admission['source_revision']==args.source_revision and admission['arms']==['off','b2','b4']
    out=args.output.resolve();private=args.private_output.resolve();out.mkdir(parents=True,exist_ok=False);private.mkdir(parents=True,exist_ok=False,mode=0o700)
    jobs=[(d,a) for d in spec['datasets'] for a in ([args.arm] if args.pilot else spec['orders'][d])]
    # One pilot server spans both pilot datasets; final uses fresh server per arm/dataset.
    if args.pilot:jobs=[('pilot',args.arm)]
    if args.branch:assert args.pilot and args.arm=='b2'
    state={'schema':'cs471_proposal_status_v1','started_utc':native.stamp(),'pilot_only':args.pilot,'mode':args.mode,'completed_sessions':[],'expected_sessions':len(jobs)}
    def update(**kw):state.update(kw,updated_utc=native.stamp());native.write_json(out/'status.json',state)
    update(state='admission');server=None;directory=None
    try:
        (out/'protocol.json').write_bytes(args.protocol.read_bytes());(out/'data-manifest.json').write_bytes(data_path.read_bytes())
        if not args.pilot:(out/'pilot-admission.json').write_bytes(args.pilot_admission.read_bytes())
        native.write_json(out/'controller-source.json',{'declared_git_revision':args.source_revision,'files':{str(p.relative_to(ROOT)):native.sha(p) for p in sorted((ROOT/'full_serving').rglob('*')) if p.is_file() and p.suffix in ('.py','.sh','.txt') and '__pycache__' not in p.parts}})
        inputs=copy_inputs(data,data_path,args.private_data,private,out,args.pilot)
        if args.branch:
            from full_serving.output_lengths import reference_targets
            assert args.reference is not None
            for ds,meta in data['datasets'].items():
                for phase in ('warmup-00','warmup-01','pilot'):
                    label=ds+'-'+phase
                    ref=args.reference/'pilot'/'off'/(label+'.jsonl')
                    target=out/(label+'-lengths.json')
                    native.write_json(target,reference_targets(ref.read_bytes(),meta['items']['warmup'],8192))
                    inputs[ds][phase+'-lengths']=target
            native.write_json(out/'branch-contract.json',{'reference':str(args.reference),
                'single_inflight':True,'decision_step':3750,'prior_and_later_migrations':'forced no-op',
                'length_files':{p.name:native.sha(p) for p in out.glob('*-lengths.json')},
                'warning':'Separate full-model replays, not cloned KV state. Require predecision demand/map/prefix checks; otherwise causal payback is inconclusive.'})
        if args.mode=='matched':
            assert not args.pilot and args.reference is not None
            for ds,meta in data['datasets'].items():
                for label in ('warmup-00','warmup-01','measured'):
                    item=meta['items']['measured' if label=='measured' else 'warmup']
                    inputs[ds][label+'-lengths']=matched_targets(args.reference,ds,label,item,out)
            native.write_json(out/'matched-targets-seal.json',{'frozen_before_server_start':True,
                'files':{p.name:native.sha(p) for p in out.glob('*-lengths.json')}})
        allocation=json.loads(native.command(['scontrol','show','job',native.JOB,'--json'],out/'allocation.json'))['jobs'][0];assert 'RUNNING' in allocation['job_state']
        nodes=native.command(['scontrol','show','hostnames',allocation['nodes']]).split();assert nodes==['kh074','kh077']
        ip=socket.gethostbyname(nodes[0]);url=f'http://{ip}:8000'
        native.write_json(out/'topology.json',{'nodes':nodes,'dp':8,'tp':1,'ep':8,'activation_backend':'allgather_reducescatter'})
        native.command([sys.executable,'-m','full_serving.preflight','checkpoint','--snapshot',str(args.snapshot),'--output',str(out/'checkpoint-verified.json')],out/'checkpoint-preflight.log',timeout=3600)
        for dataset,arm in jobs:
            directory=out/dataset/arm;directory.mkdir(parents=True);priv=private/dataset/arm;priv.mkdir(parents=True,mode=0o700)
            update(state='starting_server',active_session=f'{dataset}/{arm}',active_phase=None)
            env=clean(os.environ,True);env.update(SLURM_JOB_NODELIST=allocation['nodes'],SLURM_JOB_NUM_NODES='2',FULL_SERVING_RUN_DIR=str(directory),FULL_SERVING_CHECKPOINT_MANIFEST=str(out/'checkpoint-verified.json'),FULL_SERVING_PROPOSAL_PILOT=str(int(args.pilot)))
            if args.branch:env['FULL_SERVING_PAYBACK_BRANCH']=args.branch
            cmd=['srun','--jobid='+native.JOB,'--mpi=none','--exact','--nodes=2','--ntasks=2','--ntasks-per-node=1','--nodelist='+','.join(nodes),'--cpus-per-task=56','--gpus-per-node=4','--mem=300G','--kill-on-bad-exit=1','--input=none','--output='+str(directory/'server-%t.log'),'--error='+str(directory/'server-%t.err'),'bash',str(ROOT/'full_serving/sustained_node.sh'),arm,str(args.snapshot),ip,str(args.protocol.resolve())]
            native.write_json(directory/'launch.json',{'command':cmd,'arm':arm,'dataset':dataset,'started_utc':native.stamp()})
            with (directory/'srun.log').open('w') as log:
                begin=time.monotonic();server=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                native.wait_server(server,url,2400,directory);native.write_json(directory/'startup.json',{'elapsed_s':time.monotonic()-begin,'ready_utc':native.stamp()})
                if arm.startswith('resident'):
                    for rank in range(8):
                        binding=json.loads((directory/f'resident-bindings-rank{rank}.json').read_bytes());assert binding['layer_count']==48 and binding['ep_rank']==rank
                        # Existing all-rank resident numerical admission remains pinned.
                for source in (spec['datasets'] if args.pilot else [dataset]):
                    meta=data['datasets'][source];block={'block_id':source+'-sustained-512'}
                    for phase in ('warmup-00','warmup-01','pilot') if args.pilot else ('warmup-00','warmup-01','measured'):
                        measured=phase=='measured';label=f'{source}-{phase}' if args.pilot else phase;item=inputs[source]['measured' if measured else 'warmup']
                        assert native.sha(item['requests'])==item['requests_sha256']
                        update(state='running_requests',active_phase=label)
                        profiling=False
                        if profiling:
                            req=urllib.request.Request(url+'/start_profile',method='POST');urllib.request.urlopen(req,timeout=120).read()
                        (directory/f'{label}-metrics-before.txt').write_bytes(native.get_url(url+'/metrics'))
                        client=[sys.executable,'-m','full_serving.client','--requests',str(item['requests']),'--base-url',url+'/v1','--output',str(directory/f'{label}.jsonl'),'--private-output',str(priv/f'{label}-outputs.jsonl'),'--run-id',f'{source}-{arm}-{phase}','--arm',arm,'--block-id',source+'-'+block['block_id'],'--load-id',phase,'--pair-id',source+'-'+block['block_id']+'-'+phase,'--max-inflight',str(1 if args.branch else 512),'--timeout-s','7200','--max-tokens',str(data['max_tokens'])]
                        client+=['--calendar',str(item['calendar'])] if measured else ['--rate','0']
                        if args.mode=='matched' or args.branch:client+=['--output-lengths',str(inputs[source][phase+'-lengths'])]
                        if phase=='warmup-00' or args.branch:client+=['--logprobs']
                        native.command(client,directory/f'{label}-client.log',timeout=14400)
                        s=native.summary(directory/f'{label}.jsonl');assert s['completed_requests']==(512 if measured else 8) and not s['failed_requests']
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
    r.add_argument('--pilot-admission',type=Path)
    r.add_argument('--mode',choices=('natural','matched'),default='natural');r.add_argument('--reference',type=Path);r.add_argument('--branch',choices=('allow','skip'));r.add_argument('--source-revision',required=True);r.add_argument('--pilot',action='store_true');r.add_argument('--arm',choices=ARMS)
    args=p.parse_args()
    def stop(signum,frame):raise InterruptedError(f'Controller signal {signum}')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    (node if args.action=='node' else run)(args)
if __name__=='__main__':main()

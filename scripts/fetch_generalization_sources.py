#!/usr/bin/env python3
"""Download the exact public source files in the study lock, verifying full hashes."""
import argparse, hashlib, json, shutil, urllib.request
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lock',type=Path,default=Path('results/generalization/content-v1/download-lock.json'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--content-only',action='store_true')
    args=p.parse_args();lock=json.loads(args.lock.read_text());args.output.mkdir(parents=True,exist_ok=True)
    for r in lock['files']:
        if args.content_only and r['name'].lower().startswith(('azure','azurellm')):continue
        if Path(r['name']).name!=r['name']:raise ValueError('Flat output filenames required')
        target=args.output/r['name']
        if not target.exists():
            tmp=target.with_suffix(target.suffix+'.tmp')
            if 'local_source' in r:shutil.copyfile(r['local_source'],tmp)
            else:
                with urllib.request.urlopen(r['url'],timeout=120) as source,tmp.open('wb') as dest:
                    shutil.copyfileobj(source,dest,length=8*1024*1024)
            if tmp.stat().st_size!=r['bytes'] or sha(tmp)!=r['sha256']:raise ValueError('Downloaded bytes differ: '+r['name'])
            tmp.replace(target)
        elif target.stat().st_size!=r['bytes'] or sha(target)!=r['sha256']:
            raise ValueError('Existing bytes differ; refusing overwrite: '+r['name'])
        print(json.dumps({'file':r['name'],'verified':True}),flush=True)

if __name__=='__main__':main()

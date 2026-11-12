"""Lossless public-record archive; originals retain their exact SHA256 bytes.

python -m full_serving.sustained_archive create
python -m full_serving.sustained_archive restore
Only run create after a separate privacy review of all source records.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'results/full-serving/sustained-512-v1'
ARCHIVE = ROOT / 'public-archive-v3'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['create','restore']);args=p.parse_args()
    if args.mode=='create':
        files={}
        for run in ('run-v1','run-v2','run-v3'):
            for source in sorted((ROOT/run).rglob('*')):
                if not source.is_file():continue
                raw=source.read_bytes();name=str(source.relative_to(ROOT))
                compressed=len(raw)>=1_000_000
                payload=gzip.compress(raw,compresslevel=6,mtime=0) if compressed else raw
                dest=ARCHIVE/(name+'.gz' if compressed else name)
                dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(payload)
                assert (gzip.decompress(payload) if compressed else payload)==raw
                files[name]={'bytes':len(raw),'sha256':digest(raw),'archive_file':str(dest.relative_to(ARCHIVE)),
                    'archive_bytes':len(payload),'archive_sha256':digest(payload),'gzip':compressed}
        (ARCHIVE/'manifest.json').write_text(json.dumps({'schema':'cs471_lossless_public_archive_v1','files':files},indent=2)+'\n')
        print(json.dumps({'files':len(files),'raw_bytes':sum(v['bytes'] for v in files.values()),'archive_bytes':sum(v['archive_bytes'] for v in files.values())}))
    else:
        manifest=json.loads((ARCHIVE/'manifest.json').read_text())
        for name,meta in manifest['files'].items():
            source=ARCHIVE/meta['archive_file'];dest=ROOT/name
            assert source.resolve().is_relative_to(ARCHIVE.resolve()) and dest.resolve().is_relative_to(ROOT.resolve())
            payload=source.read_bytes();assert digest(payload)==meta['archive_sha256'] and len(payload)==meta['archive_bytes']
            raw=gzip.decompress(payload) if meta['gzip'] else payload
            assert digest(raw)==meta['sha256'] and len(raw)==meta['bytes']
            if dest.exists():assert dest.read_bytes()==raw,str(dest)
            else:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
        print(json.dumps({'restored_and_verified_files':len(manifest['files'])}))


if __name__=='__main__':main()

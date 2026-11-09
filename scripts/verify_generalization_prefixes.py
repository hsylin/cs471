#!/usr/bin/env python3
"""Independent audit of prepared full documents and actual captured prefixes.

No imports from generalization.data or the source-preparation implementation.
Requires the private prepared sources, public manifest and pinned tokenizer.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata


def canonical_sha(value):
    return hashlib.sha256(value if isinstance(value,bytes) else json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def normalized(value):return ' '.join(unicodedata.normalize('NFKC',value).casefold().split())


def shingles(value):
    words=re.findall(r'\w+|[^\w\s]',normalized(value))
    return {tuple(words[i:i+5]) for i in range(max(1,len(words)-4))}


def verify_public_identity(private,public):
    assert private['content_sha256']==canonical_sha({k:v for k,v in private.items() if k!='content_sha256'}),'Private canonical hash differs'
    assert public['public_manifest_sha256']==canonical_sha({k:v for k,v in public.items() if k!='public_manifest_sha256'}),'Public canonical hash differs'
    expected=json.loads(json.dumps(private))
    for block in expected['blocks']:
        for record in block['sources']:
            del record['text'];del record['token_ids']
    expected['public_manifest_sha256']=canonical_sha(expected)
    assert public==expected,'Public manifest does not describe exact private selection'


def audit_documents(manifest,encode,decode):
    """Pure audit; injectable tokenizer functions permit independent unit tests."""
    seen_ids=set();exact={};tokenexact=set();index=defaultdict(set)
    sets=[];records=[];near=[];pair_checks=0
    for block in manifest['blocks']:
        for record in block['sources']:
            key=(record['dataset'],record['source_id'])
            assert key not in seen_ids,'Source ID reused';seen_ids.add(key)
            ids=encode(record['text'])
            assert ids[:512]==record['token_ids'],'Retokenized prefix differs'
            assert len(ids)==record['full_token_count'],'Full token count differs'
            assert record['captured_token_count']==len(record['token_ids']),'Captured token count differs'
            assert record['token_ids_sha256']==canonical_sha(record['token_ids']),'Token list hash differs'
            assert canonical_sha(record['text'].encode())==record['text_sha256'],'Full text hash differs'
            assert canonical_sha(normalized(record['text']).encode())==record['normalized_text_sha256'],'Normalized full text hash differs'
            prefix=decode(record['token_ids']);h=canonical_sha(normalized(prefix).encode())
            assert h==record['captured_prefix_normalized_sha256'],'Captured prefix hash differs'
            assert h not in exact,'Normalized prefix reused';exact[h]=key
            assert tuple(record['token_ids']) not in tokenexact,'Token prefix reused';tokenexact.add(tuple(record['token_ids']))
            current=shingles(prefix);possible=set()
            for gram in current:possible.update(index[gram])
            for i in possible:
                pair_checks+=1
                similarity=len(current&sets[i])/len(current|sets[i])
                if similarity>=0.8:
                    near.append({'left':records[i],'right':[block['block_id'],*key],'jaccard':similarity})
            i=len(sets);sets.append(current);records.append([block['block_id'],*key])
            for gram in current:index[gram].add(i)
    return {'status':'pass' if not near else 'failed','blocks':len(manifest['blocks']),
        'documents':len(records),'dataset_documents':dict(Counter(r[1] for r in records)),
        'exact_token_prefix_duplicates':0,'normalized_prefix_duplicates':0,
        'near_prefix_pairs':near,'candidate_pair_comparisons':pair_checks,
        'method':'Independent NFKC/casefold/whitespace normalization and word/punctuation 5-shingle Jaccard >=0.8 on exact decoded captured prefixes. Pairs sharing no shingle have zero similarity. Stored text, token and prefix hashes and token counts are recomputed.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',type=Path,required=True)
    parser.add_argument('--public-manifest',type=Path,required=True)
    parser.add_argument('--tokenizer',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Fresh audit output required; preserve prior audits')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    provenance={'source_sha256':canonical_sha(args.sources.read_bytes()),
        'public_manifest_sha256':canonical_sha(args.public_manifest.read_bytes()),
        'tokenizer_sha256':canonical_sha(args.tokenizer.read_bytes()),
        'verifier_sha256':canonical_sha(Path(__file__).read_bytes())}
    try:
        manifest=json.loads(args.sources.read_text());public=json.loads(args.public_manifest.read_text())
        verify_public_identity(manifest,public)
        assert provenance['tokenizer_sha256']==manifest['raw_files']['qwen-tokenizer.json']['sha256'],'Wrong tokenizer artifact'
        from tokenizers import Tokenizer
        tokenizer=Tokenizer.from_file(str(args.tokenizer))
        result=audit_documents(manifest,lambda text:tokenizer.encode(text,add_special_tokens=False).ids,tokenizer.decode)
        result.update(provenance);result['source_content_sha256']=manifest['content_sha256']
    except Exception as error:
        result={**provenance,'status':'failed','error_type':type(error).__name__,'error':str(error)}
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        raise
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'documents':result['documents'],'near_prefix_pairs':len(result['near_prefix_pairs'])}))
    if result['status']!='pass':raise SystemExit(1)

if __name__=='__main__':main()

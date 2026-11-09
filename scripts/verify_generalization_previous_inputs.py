#!/usr/bin/env python3
"""Descriptive overlap check against actual earlier captured model inputs.

This does not establish absence from checkpoint pretraining or every possible
partial overlap. It uses the prespecified normalized5-shingle Jaccard threshold.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def helper(name,filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/filename)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


prefix=helper('previous_input_prefix_helpers','verify_generalization_prefixes.py')
raw=helper('previous_input_npz_helpers','verify_resident_online_records.py')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('sources','public-manifest','tokenizer','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--previous-capture',type=Path,nargs='+',required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    private=json.loads(a.sources.read_text());public=json.loads(a.public_manifest.read_text())
    prefix.verify_public_identity(private,public)
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(a.tokenizer))
    assert raw.file_digest(a.tokenizer)==private['raw_files']['qwen-tokenizer.json']['sha256']
    current=[]
    for block in private['blocks']:
        for record in block['sources']:
            ids=tokenizer.encode(record['text'],add_special_tokens=False).ids[:512]
            assert ids==record['token_ids']
            text=tokenizer.decode(ids,skip_special_tokens=False)
            current.append((block['block_id'],record['dataset'],record['source_id'],ids,prefix.normalized(text),prefix.shingles(text)))
    previous=[];files=[];seen=set()
    for directory in a.previous_capture:
        entries=json.loads((directory/'capture-index.json').read_text())
        for entry in entries:
            path=directory/'banks'/(entry['block_id']+'.npz')
            meta,bank=raw.authenticated_npz(path,entry['capture_sha256'])
            assert meta['source_revision']==private['source_revision']
            files.append(dict(path=str(path),sha256=entry['capture_sha256'],block_id=entry['block_id']))
            for i,record in enumerate(meta['prompts']):
                mask=bank['prompt_index']==i;ids=bank['token_ids'][mask]
                assert raw.array_digest(ids)==record['token_ids_sha256']
                identity=(meta['source_prompt_ids'][i],raw.array_digest(ids))
                if identity in seen:continue
                seen.add(identity)
                for scope,tokens in [('first512_captured_tokens',ids[:512]),('all_captured_tokens',ids)]:
                    text=tokenizer.decode(tokens.tolist(),skip_special_tokens=False)
                    previous.append((identity[0],scope,tokens.tolist(),prefix.normalized(text),prefix.shingles(text)))
    matches=[];maximum={};pairs=0
    for block,dataset,sid,tokens,text,grams in current:
        for old_id,scope,old_tokens,old_text,old_grams in previous:
            pairs+=1;score=len(grams&old_grams)/len(grams|old_grams)
            finding=dict(block_id=block,dataset=dataset,source_id=sid,previous_source_id=old_id,previous_scope=scope,
                jaccard=score,exact_token_sequence=tokens==old_tokens,exact_normalized_text=text==old_text)
            if score>maximum.get(scope,{}).get('jaccard',-1):maximum[scope]=finding
            if score>=.8 or finding['exact_token_sequence'] or finding['exact_normalized_text']:matches.append(finding)
    result=dict(schema_version=1,passed=not matches,current_documents=len(current),previous_documents=len(seen),
        previous_banks=len(files),pair_comparisons=pairs,threshold=.8,matching_pairs=matches,maximum_similarity_by_scope=maximum,
        previous_banks_inventory=files,source_content_sha256=private['content_sha256'],
        source_file_sha256=raw.file_digest(a.sources),public_manifest_file_sha256=raw.file_digest(a.public_manifest),
        tokenizer_sha256=raw.file_digest(a.tokenizer),verifier_sha256=raw.file_digest(Path(__file__)),
        method='Independent NFKC/casefold/whitespace normalization; word/punctuation5-shingle set Jaccard. Compare all new model-visible prefixes against each prior captured first512 and full captured sequence.',
        limitations=['Checks archived prior model-visible captures only, not full original documents or checkpoint training.',
            'Whole-sequence exact/near comparison does not exclude short shared fragments or semantic similarities.',
            'This is an audit after input selection, not a performance-dependent selection or new optimization rule.'])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('passed','current_documents','previous_documents','previous_banks','pair_comparisons')}))
    if matches:raise SystemExit(1)


if __name__=='__main__':main()

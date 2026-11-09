#!/usr/bin/env python3
"""Prepare bounded, content-disjoint source and contiguous external-arrival manifests.

Downloads are explicit inputs. Public manifests contain identities/hashes;
selected prompt text stays in the requested private output directory.
"""
from pathlib import Path
import argparse, csv, json, sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from generalization.data import digest, deduplicate, deduplicate_captured_prefixes, split_records, validate_splits, select_arrival_blocks
from followup.router_tapes import MODEL, REVISION, file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--downloads',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--public-output',type=Path,required=True)
    args=p.parse_args();root=args.downloads
    if args.output.exists() or args.public_output.exists():raise FileExistsError('Fresh directories required')
    import pyarrow.parquet as pq
    import tokenizers
    tokenizer=tokenizers.Tokenizer.from_file(str(root/'qwen-tokenizer.json'))
    candidates=[]
    # Bounded frame, fixed before any routing or performance observation.
    wild=[]
    for batch in pq.ParquetFile(root/'wildchat-first-shard.parquet').iter_batches(batch_size=1024):
        wild.extend(batch.to_pylist())
        if len(wild)>=8192:break
    for r in wild[:8192]:
        if r['language']!='English' or r['toxic']:continue
        turns=[t for t in r['conversation'] if t['role']=='user']
        if not turns:continue
        t=turns[0]
        if t.get('toxic') or not t['content'].strip():continue
        candidates.append(dict(dataset='wildchat',source_id=r['conversation_id'],text=t['content'],source_row_kind='first_user_turn',original_split='train'))
    for r in map(json.loads,(root/'mbpp-full.jsonl').read_text().splitlines()):
        candidates.append(dict(dataset='mbpp',source_id=str(r['task_id']),text=r['text'],source_row_kind='task_description_only',original_split='test' if 11<=r['task_id']<=510 else 'other_original_split'))
    for i,r in enumerate(map(json.loads,(root/'gsm8k.jsonl').read_text().splitlines())):
        candidates.append(dict(dataset='gsm8k',source_id=f'test-{i}',text=r['question'],source_row_kind='question_only',original_split='test'))
    # No reference answers, solutions, tests, assistant text, or chat template.
    eligible=[];short=[]
    for r in candidates:
        ids=tokenizer.encode(r['text'],add_special_tokens=False).ids
        if len(ids)<8:
            short.append([r['dataset'],r['source_id']]);continue
        eligible.append({**r,'full_token_count':len(ids),'captured_token_count':min(512,len(ids)),
                         'token_ids_sha256':digest(ids[:512]),'token_ids':ids[:512],
                         'captured_prefix_text':tokenizer.decode(ids[:512],skip_special_tokens=False)})
    unique,dropped=deduplicate(eligible)
    unique,prefix_dropped=deduplicate_captured_prefixes(unique)
    blocks=split_records(unique,pilot_documents=64,final_documents=64)
    counts={d:sum(r['dataset']==d for r in unique) for d in ('wildchat','mbpp','gsm8k')}
    names=['wildchat-metadata.json','mbpp-metadata.json','gsm8k-metadata.json','wildchat-first-shard.parquet','wildchat-readme.md','wildchat-license.md','mbpp-full.jsonl','mbpp-readme.md','google-readme.md','gsm8k.jsonl','gsm8k-license.txt','qwen-tokenizer.json']
    files={name:dict(bytes=(root/name).stat().st_size,sha256=file_sha256(root/name)) for name in names}
    manifest={'schema':'generalization_sources_v1','seed':471501,'arrival_seed':471502,'source_model':MODEL,'source_revision':REVISION,
        'formatting':{'chat_template':None,'add_special_tokens':False,'role':'plain first user/question/task text','phase':'causal_prefill_layer0','generated_outputs':False,'teacher_forced_answer':False,'truncation':'first 512 tokens; no concatenation across prompts in attention'},
        'selection':{'wildchat_frame':'first 8192 rows of pinned first shard; English, non-toxic, nonempty first user turn','minimum_tokens':8,'near_duplicate_jaccard':0.8,'shingle_width':5,'pilot_blocks_per_dataset':2,'final_blocks_per_dataset':8,'documents_per_block':64,'eligible_counts':counts,'deduplicated_before_split':True,'captured_prefix_deduplicated':True,'prefix_decoder_tokenizers_version':tokenizers.__version__},
        'capture_limit_note':'Captured prefixes are bounded; inference runs do not generate responses. Unseen means new to this experiment, not proven absent from checkpoint training.',
        'preparation_source_sha256':file_sha256(__file__),'data_contract_sha256':file_sha256(Path(__file__).resolve().parents[1]/'generalization/data.py'),
        'raw_files':files,'blocks':blocks,'duplicate_rejections':dropped,'prefix_duplicate_rejections':prefix_dropped,'short_rejections':short}
    manifest['content_sha256']=digest(manifest)
    validate_splits(blocks)
    args.output.mkdir(parents=True);args.public_output.mkdir(parents=True)
    (args.output/'sources.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    public=json.loads(json.dumps(manifest))
    for b in public['blocks']:
        for r in b['sources']:
            del r['text'];del r['token_ids']
    public['public_manifest_sha256']=digest(public)
    (args.public_output/'source-manifest.json').write_text(json.dumps(public,indent=2)+'\n')
    print(json.dumps({'blocks':len(blocks),'documents':sum(len(b['sources']) for b in blocks),'eligible':counts,'content_sha256':manifest['content_sha256'],'tokens_by_block':{b['block_id']:sum(r['captured_token_count'] for r in b['sources']) for b in blocks}},indent=2))

if __name__=='__main__':main()

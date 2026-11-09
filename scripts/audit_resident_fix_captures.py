#!/usr/bin/env python3
"""Independent CPU capture audit; imports no production capture/runtime modules."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import tokenizers
ROOT=Path(__file__).resolve().parents[1]
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha_file(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def sha(x):return hashlib.sha256(x if isinstance(x,bytes) else canonical(x)).hexdigest()

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 for key in ('sources','public','data-audit','tokenizer','output'):
  parser.add_argument('--'+key,type=Path,required=True)
 for key in ('pilot-capture','final-capture'):
  parser.add_argument('--'+key,type=Path)
 parser.add_argument('--checkpoint-reference',type=Path,default=ROOT/'results/stock-comparison/capture-audit-v1.json')
 args=parser.parse_args()
 assert args.pilot_capture or args.final_capture,'at least one complete split required'
 assert not args.output.exists(),'fresh output required'
 private=args.sources;sources=json.loads(private.read_text())
 assert sources['schema']=='resident_fix_sources_v1'
 assert sources['content_sha256']=='483864373cef39ff7050e463f135fccb8941988b3ed6b75e8d880e319e332df6'
 assert sources['content_sha256']==sha({k:v for k,v in sources.items() if k!='content_sha256'})
 public=json.loads(args.public.read_text());assert public['public_manifest_sha256']==sha({k:v for k,v in public.items() if k!='public_manifest_sha256'})
 expected=json.loads(json.dumps(sources))
 for block in expected['blocks']:
  for row in block['sources']:
   del row['text'];del row['token_ids']
 assert expected=={k:v for k,v in public.items() if k!='public_manifest_sha256'}
 data_audit=json.loads(args.data_audit.read_text())
 assert data_audit['passed'] is True and data_audit['source_sha256']==sources['content_sha256']
 assert sha_file(args.tokenizer)==sources['raw_files']['qwen-tokenizer.json']['sha256']
 blocks={b['block_id']:b for b in sources['blocks']};assert len(blocks)==22
 capture_script=ROOT/'scripts/capture_resident_fix_inputs.py'
 expected_scripts={'capture_script_sha256':sha_file(capture_script),'modeling_sha256':'b2666af64b7b31a02a152e9187ed2c5671262543b61510d0359c26fa491105ca','masking_sha256':'50a737f63d8c778a5597fa34ac139049af921f44958205e3e1c29fe2bae77254'}
 tokenizer=tokenizers.Tokenizer.from_file(str(args.tokenizer))
 keys={'hidden_states_bf16','expert_ids','combining_weights','token_ids','prompt_index','token_position','domain_ids'}
 seen=set();results=[];identities=[];common=None;index_hashes={};source_commits={};provenance_hashes={}
 selected_splits=[]
 for split,expected in [('pilot',5),('final',17)]:
  root=getattr(args,split+'_capture')
  if root is None:continue
  selected_splits.append(split)
  index=json.loads((root/'capture-index.json').read_text());assert len(index)==expected
  assert len(list((root/'banks').glob('*.npz')))==expected
  index_hashes[split]=sha_file(root/'capture-index.json');source_commits[split]=(root/'source-commit.txt').read_text().strip() if (root/'source-commit.txt').exists() else None
  provenance_path=root/'provenance/capture-provenance.json'
  provenance=json.loads(provenance_path.read_text())
  assert provenance['capture_only'] is True and provenance['allocation_id']=='4872248'
  assert provenance['hostname'].split('.')[0]=='kh014' and provenance['step_id'] is not None
  assert provenance['private_manifest_file_sha256']==sha_file(private)
  assert provenance['source_selection_sha256']==sources['content_sha256']
  assert provenance['base_source_commit']==source_commits[split]
  for name,pin in sources['preparation_source_files'].items():assert provenance['file_sha256'][name]==pin
  for name in ('capture_resident_fix_inputs.py','run-resident-fix-capture-kuma.sh','cluster-env.sh'):
   assert sha_file(root/'provenance'/name)==provenance['file_sha256']['scripts/'+name]
  assert sha_file(root/'provenance/public-source-manifest.json')==sha_file(args.public)
  provenance_hashes[split]=sha_file(provenance_path)
  for entry in index:
   name=entry['block_id'];assert name not in seen;seen.add(name)
   b=blocks[name];assert b['split']==split==entry['split'] and entry['dataset']==b['dataset']
   assert entry['documents']==32 and entry['correctness_pass'] is True
   path=root/'banks'/Path(entry['bank_file']).name
   assert sha_file(path)==entry['capture_sha256']
   with np.load(path,allow_pickle=False) as z:
    assert set(z.files)==keys|{'metadata','content_sha256'}
    metadata=json.loads(str(z['metadata']));arrays={k:z[k] for k in keys}
    h=hashlib.sha256(canonical(metadata))
    for k in sorted(keys):
     a=np.ascontiguousarray(arrays[k]);h.update(canonical([k,a.dtype.str,list(a.shape)]));h.update(a.tobytes(order='C'))
    assert h.hexdigest()==str(z['content_sha256'])
    content_hash=h.hexdigest()
   m=metadata
   for k,v in expected_scripts.items():assert m[k]==v,(name,k)
   for k,v in {'source_model':'Qwen/Qwen3-30B-A3B','source_revision':'ad44e777bcd18fa416d9da3bd8f70d33ebb85d39','study_id':'resident_fix_v1','schema':'qwen_router_bank_v1','origin':'prompt_causal_prefill_layer0','layer':0,'num_experts':128,'hidden_size':2048,'top_k':8,'attention_backend':'eager','transformers_version':'5.17.0','correctness_pass':True,'native_serving_trace':False,'timing_evidence':False,'special_tokens_added':False,'chat_template':None,'selection_content_sha256':sources['content_sha256'],'source_manifest_sha256':sha_file(private)}.items():assert m[k]==v,(name,k)
   assert m['block_id']==name and m['dataset']==b['dataset'] and m['split']==split
   assert m['inference_role']==b['inference_role']==entry['inference_role']
   assert m['source_prompt_ids']==[r['dataset']+':'+r['source_id'] for r in b['sources']]
   pins={k:m[k] for k in ('source_shards','source_tensor_sha256','config_sha256','index_sha256','tokenizer_files','modeling_sha256','masking_sha256')}
   if common is None:common=pins
   assert pins==common
   assert {r['file']:r['sha256'] for r in m['tokenizer_files']}['tokenizer.json']==sources['raw_files']['qwen-tokenizer.json']['sha256']
   n=sum(r['captured_token_count'] for r in b['sources']);assert n==entry['tokens']
   expected_dtypes={'hidden_states_bf16':np.dtype('uint16'),'expert_ids':np.dtype('int64'),'combining_weights':np.dtype('float32'),'token_ids':np.dtype('int64'),'prompt_index':np.dtype('int32'),'token_position':np.dtype('int32'),'domain_ids':np.dtype('int8')}
   for k,a in arrays.items():
    assert a.dtype==expected_dtypes[k]
    assert a.shape==((n,2048) if k=='hidden_states_bf16' else (n,8) if k in ('expert_ids','combining_weights') else (n,))
   assert not np.any((arrays['hidden_states_bf16']&0x7f80)==0x7f80)
   ids=arrays['expert_ids'];w=arrays['combining_weights'];assert ((ids>=0)&(ids<128)).all() and (np.diff(np.sort(ids,axis=1),axis=1)>0).all()
   assert np.isfinite(w).all() and (w>=0).all() and np.all((w.view(np.uint32)&0xffff)==0)
   norm_error=float(np.max(np.abs(w.sum(axis=1)-1)));assert norm_error<=.004
   cursor=0;assert len(m['prompts'])==32
   for j,(r,p) in enumerate(zip(b['sources'],m['prompts'])):
    tokens=tokenizer.encode(r['text'],add_special_tokens=False).ids
    assert tokens[:512]==r['token_ids'] and len(tokens)==r['full_token_count']
    take=r['captured_token_count'];assert take==min(512,len(tokens))
    for k,v in r.items():
     if k not in ('text','token_ids','token_ids_sha256'):assert p[k]==v,(name,j,k)
    assert p['prompt_id']==r['source_id'] and p['cleaned_text_sha256']==sha(r['text'].encode())
    assert p['captured_tokens']==take and p['total_source_tokens']==len(tokens) and p['position_start']==p['token_offset']==0
    assert all(p[k] is True for k in ('native_boundary_exact','native_router_exact','manual_router_exact','bf16_roundtrip_exact'))
    token_array=arrays['token_ids'][cursor:cursor+take]
    assert token_array.tolist()==tokens[:512] and p['token_ids_sha256']==sha(token_array.astype('<i8').tobytes())
    assert (arrays['prompt_index'][cursor:cursor+take]==j).all()
    assert (arrays['domain_ids'][cursor:cursor+take]==j%2).all()
    assert np.array_equal(arrays['token_position'][cursor:cursor+take],np.arange(take))
    identities.append((r['dataset'],r['source_id']));cursor+=take
   assert cursor==n
   results.append(dict(block_id=name,split=split,dataset=b['dataset'],documents=32,tokens=n,file_sha256=entry['capture_sha256'],content_sha256=content_hash,all_prompt_tokens_positions_and_hashes_verified=True,all_four_capture_correctness_attestations_per_prompt=True,hidden_bits_finite=True,combining_weights_native_bf16_representable=True,maximum_combining_weight_normalization_error=norm_error))
 expected_ids={k for k,b in blocks.items() if b['split'] in selected_splits}
 assert seen==expected_ids and len(identities)==len(set(identities))==32*len(expected_ids)

 reference=json.loads(args.checkpoint_reference.read_text())
 assert reference['passed'] is True
 assert common==reference['checkpoint_and_implementation_pins'],'changed checkpoint/implementation bytes'
 gate=common['source_tensor_sha256']['model.layers.0.mlp.gate.weight']
 assert gate=='5a165372095204aaa0343a67d2d2afde1086b730a825998b59264fb9c49a216d'
 out=dict(schema='resident_fix_capture_independent_audit_v1',passed=True,banks=len(results),splits=selected_splits,source_documents=len(identities),captured_token_records=sum(r['tokens'] for r in results),source_selection_sha256=sources['content_sha256'],private_manifest_file_sha256=sha_file(private),public_manifest_file_sha256=sha_file(args.public),source_data_audit_sha256=sha_file(args.data_audit),capture_index_sha256=index_hashes,capture_provenance_sha256=provenance_hashes,recorded_base_source_commits=source_commits,checkpoint_and_implementation_pins=common,checkpoint_reference_file_sha256=sha_file(args.checkpoint_reference),capture_script_sha256=expected_scripts['capture_script_sha256'],gate_weight_sha256=gate,banks_audited=results,limitations=['CPU audit verifies file/content digests, document/token identities, positions, tensor encodings and recorded capture attestations; it does not independently rerun attention or GPU capture.','Checkpoint gate matches a previously independently hashed exact safetensors byte range; this audit does not claim a new full-checkpoint CPU hash.','Measured hidden-only input consumption, runtime gate/routing and numerical validity require their own execution audit.','No demand summary, final performance outcome or capture-based tuning is included.','Independent implementation within the same team, not external laboratory reproduction.'],audit_program_sha256=sha_file(__file__))
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({k:out[k] for k in ('passed','banks','source_documents','captured_token_records','gate_weight_sha256')}))

if __name__=="__main__":main()

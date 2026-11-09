"""Independent read-only audit of raw candidate-payback records.

No import from either measured harness or its analyzer. Writes only audit outputs.
Assumes the frozen two-rank E128/K8/W16/R64 protocol; rejects missing/duplicate data.

Examples (run from any directory):
  python scripts/audit_payback_records.py --input-dir results/payback/pilot-native \
      --inputs-dir results/payback/inputs --output-prefix /tmp/native-audit
  python scripts/audit_payback_records.py --manifest /path/executed-manifest.json \
      --records /path/records.jsonl.gz --inputs-dir /path/inputs \
      --analysis-summary /path/analysis/summary.json --output-prefix /tmp/final-audit

The optional raw-input audit independently reconstructs the pinned native replica
mapping in NumPy. It does not rerun pretrained attention/router computation or
GPU GEMMs. Numerical results and weight-integrity checks remain GPU execution
attestations. It reports descriptive ratios; it does not independently validate
bootstrap intervals. All output is separate from the supplied measurements.
"""
from pathlib import Path
import argparse,gzip,hashlib,itertools,json,math,random,statistics
from collections import Counter,defaultdict
import numpy as np
if not __debug__:raise SystemExit('Run without -O: this audit requires assertions enabled.')
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input-dir',type=Path,help='Convenience: directory with records.jsonl[.gz] and provenance/executed-manifest.json.')
parser.add_argument('--manifest',type=Path,help='Frozen executed manifest; overrides --input-dir default.')
parser.add_argument('--records',type=Path,help='One complete raw JSONL or JSONL.GZ file; overrides --input-dir default.')
parser.add_argument('--analysis-summary',type=Path,help='Optional saved analyzer summary for independent point-estimate comparison.')
parser.add_argument('--provenance-dir',type=Path,help='Optional source-files.json directory; default is the manifest directory.')
parser.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[1],help='Matching source checkout; defaults to this script repository.')
parser.add_argument('--output-prefix',type=Path,required=True)
parser.add_argument('--inputs-dir',type=Path,help='Optional archived capture root containing banks/ and tapes-v2/.')
args=parser.parse_args();base=args.input_dir
manifest_path=args.manifest or (base/'provenance/executed-manifest.json' if base else None)
records_path=args.records or (base/'records.jsonl' if base else None)
if manifest_path is None or records_path is None:parser.error('Provide --input-dir or both --manifest and --records.')
if args.records is None and not records_path.exists() and records_path.with_suffix('.jsonl.gz').exists():records_path=records_path.with_suffix('.jsonl.gz')
provenance_dir=args.provenance_dir or manifest_path.parent
analysis_path=args.analysis_summary or (base/'analysis/summary.json' if base else None)
manifest=json.loads(manifest_path.read_text())
with (gzip.open(records_path,'rt') if records_path.suffix=='.gz' else records_path.open()) as handle:
 rows=[json.loads(line) for line in handle if line.strip()]
def canon(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def ah(x,dtype=np.int64):return hashlib.sha256(np.asarray(x,dtype=dtype).tobytes()).hexdigest()
def loads(c,m):
 m=np.asarray(m);multiplicity=np.bincount(m,minlength=128)
 return (np.asarray(c)[m]/multiplicity[m]).reshape(2,80).sum(axis=1)
def gain(c,o,n):
 old=loads(c,o).max();return float((old-loads(c,n).max())/old) if old else 0.
def equal(rs,fields):
 for f in fields:assert all(r[f]==rs[0][f] for r in rs),(f,rs[0]['block_id'],rs[0]['regime'],rs[0]['batch_tokens'])
digest=canon({k:v for k,v in manifest.items() if k!='freeze'})
assert digest==manifest['freeze']['sha256']
assert manifest.get('study_role','payback_study')=='payback_study'
KEY_FIELDS=('block_id','regime','batch_tokens','candidate_id','horizon_steps','repetition')
matrix=manifest['matrix'];selection=rows[0]['execution_selection'];stage=selection['stage']
assert stage in ('screening',None), 'Use the complete screening or final matrix, not a partial confirmation file.'
assert manifest['split'] in ('pilot','final')
assert (manifest['split']=='final')==(stage is None)
ordered=[]
for k in itertools.product(matrix['block_ids'],matrix['regimes'],matrix['batch_tokens'],matrix['candidate_ids'],matrix['horizons'],range(matrix['repetitions'])):
 if stage=='screening' and k[-1]!=0:continue
 if any(selection.get(f) is not None and k[i] not in selection[f] for i,f in enumerate(('blocks','regimes','batch_tokens','candidate_families','horizons','repetitions'))):continue
 ordered.append(k)
expected={k+(a,) for k in ordered for a in matrix['arms']}
assert len(rows)==len(expected) and len(rows)>0
seen=set();groups=defaultdict(list);by_state=defaultdict(list)
by_tape={(t['block_id'],t['regime'],t['batch_tokens']):t for t in manifest['tapes']}
block_documents={b['block_id']:b['source_document_ids'] for b in manifest['blocks']}
assert len(set(d for ds in block_documents.values() for d in ds))==sum(map(len,block_documents.values()))
# Independent NumPy reproduction of the pinned native token-index replica hash.
# The two local source hashes below pin the reviewed algorithm, not our results.
raw_banks={};raw_tapes={};raw_dispatch={}
def raw_npz(path,expected_sha):
 assert hashlib.sha256(path.read_bytes()).hexdigest()==expected_sha
 with np.load(path,allow_pickle=False) as z:
  metadata=json.loads(str(z['metadata']));arrays={k:z[k].copy() for k in z.files if k not in ('metadata','content_sha256')}
  h=hashlib.sha256(json.dumps(metadata,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
  for name,a in sorted(arrays.items()):
   a=np.ascontiguousarray(a);h.update(json.dumps([name,a.dtype.str,list(a.shape)],sort_keys=True,separators=(',',':')).encode());h.update(a.tobytes())
  assert h.hexdigest()==str(z['content_sha256'])
 return metadata,arrays

def raw_tape(entry):
 key=entry['replay_sha256']
 if key not in raw_tapes:
  if entry['capture_sha256'] not in raw_banks:
   bank_meta,bank=raw_npz(args.inputs_dir/entry['capture_path'],entry['capture_sha256'])
   assert bank['expert_ids'].shape==(4096,8) and bank['expert_ids'].dtype==np.int64
   assert np.all((bank['expert_ids']>=0)&(bank['expert_ids']<128))
   assert np.all(np.diff(np.sort(bank['expert_ids'],axis=1),axis=1)>0)
   assert bank['hidden_states_bf16'].shape==(4096,2048) and bank['hidden_states_bf16'].dtype==np.uint16
   assert np.all((bank['hidden_states_bf16']&0x7f80)!=0x7f80)
   assert bank['combining_weights'].shape==(4096,8) and bank['combining_weights'].dtype==np.float32
   assert np.all(np.isfinite(bank['combining_weights'])) and np.all(bank['combining_weights']>=0)
   assert np.all(np.abs(bank['combining_weights'].sum(axis=1)-1)<.004)
   assert bank_meta['source_model']==manifest['environment']['model_id'] and bank_meta['source_revision']==manifest['environment']['model_revision'] and bank_meta['layer']==0
   raw_banks[entry['capture_sha256']]=(bank_meta,bank)
  bank_meta,bank=raw_banks[entry['capture_sha256']]
  meta,tape=raw_npz(args.inputs_dir/entry['replay_path'],key);indices=tape['indices']
  assert indices.shape==(672,entry['batch_tokens']) and indices.dtype==np.int64 and indices.min()>=0 and indices.max()<4096
  assert meta['capture_sha256']==entry['capture_sha256'] and meta['source_prompt_ids']==block_documents[entry['block_id']]
  assert meta['natural_temporal_demand'] is False and meta['origin']=='constructed_repeated_prefill_replay'
  assert np.array_equal(np.bincount(indices.ravel(),minlength=4096),meta['token_use_counts'])
  assert np.all(np.diff(np.sort(indices,axis=1),axis=1)>0)
  assert np.array_equal(bank['domain_ids'][indices],np.broadcast_to(np.asarray(meta['domain_schedule'])[:,None],indices.shape))
  raw_tapes[key]=(meta,bank,indices)
 return raw_tapes[key]

def cpu_physical(ids,mapping):
 mapping=np.asarray(mapping);counts=np.bincount(mapping,minlength=128)
 lut=np.zeros((128,int(counts.max())),np.int64)
 for e in range(128):lut[e,:counts[e]]=np.flatnonzero(mapping==e)
 hashed=(np.arange(ids.shape[1],dtype=np.int64)*2654435769)&0xffffffff
 replica=hashed[None,:,None]%counts[ids]
 physical=lut[ids,replica]
 offsets=np.arange(len(ids),dtype=np.int64)[:,None,None]*160
 return np.bincount((physical+offsets).ravel(),minlength=len(ids)*160).reshape(len(ids),160).astype(np.int32)

if args.inputs_dir:
 # Source reviewed at vLLM db9527a46873454610df6dbedf79a36d6bf1a7f6:
 # base_router.py:45-57 (token-index hash); eplb_state.py:1308-1326
 # (ascending physical-slot order). The exact source is pinned by the harness.
 source_pins=json.loads((args.source_root/'skewbench/native_sources.json').read_text())
 assert source_pins['commit']==manifest['environment']['vllm_commit']=='db9527a46873454610df6dbedf79a36d6bf1a7f6'
 assert source_pins['sha256']['vllm/model_executor/layers/fused_moe/router/base_router.py']=='c909924fcfa461ccdaa6290d215d4d35bd4062fb02feeb23619b234da993f6bf'
 assert source_pins['sha256']['vllm/distributed/eplb/eplb_state.py']=='eaac132acd438074343e5ee56c52e1df46f37a9bf7a303bd3b9373c69b513f17'
for r in rows:
 key=tuple(r[k] for k in KEY_FIELDS)+(r['arm'],)
 assert key in expected and key not in seen;seen.add(key)
 assert r['status']=='complete' and r['correctness_pass'] is True
 assert r['study_role']=='payback_study' and r['evidence_eligible'] is True and r['split']==manifest['split']
 assert r['execution_selection']==selection
 assert np.isfinite(r['elapsed_seconds']) and r['elapsed_seconds']>0
 assert r['elapsed_seconds']==max(r['per_rank_elapsed_seconds'])
 assert len(r['per_rank_elapsed_seconds'])==2
 for field,value in manifest['required_record_metadata'].items():assert r[field]==value
 assert r['source_prompt_ids']==block_documents[r['block_id']]
 assert r['config_hash']==digest and r['followup_sha256']==manifest['environment']['followup_sha256']
 assert canon(r['followup_file_sha256'])==r['followup_sha256']
 assert r['legacy_harness_sha256']==manifest['environment']['legacy_harness_sha256']
 assert canon(r['harness_file_sha256'])==r['legacy_harness_sha256']
 assert r['world_size']==2 and r['dispatch_backend']=='replicated_input_allreduce'
 assert r['eplb_mode']=='synchronous' and r['weight_communicator']=='torch_nccl'
 assert all(r[k] for k in ('source_restore_verified','source_state_verified','candidate_weights_verified','expected_arm_weights_verified','commit_completed'))
 assert r['future_rebalancing_enabled'] is False
 entry=by_tape[(r['block_id'],r['regime'],r['batch_tokens'])]
 assert r['capture_sha256']==entry['capture_sha256'] and r['replay_sha256']==entry['replay_sha256']==r['trace_sha256']
 assert r['replay_metadata']['repeat_steps']==1024 and r['replay_metadata']['num_steps']==672
 if args.inputs_dir:
  raw_metadata,raw_bank,raw_indices=raw_tape(entry)
  assert raw_metadata==r['replay_metadata']
  assert r['state_id']==canon({'block_id':r['block_id'],'regime':r['regime'],'batch_tokens':r['batch_tokens'],'replay_sha256':entry['replay_sha256'],'source_map_sha256':r['source_map_sha256']})
 assert r['decision_step']==127 and r['decision_trace_step']==159
 assert r['future_start_raw_step']==160 and r['future_stop_raw_step_exclusive']==160+r['horizon_steps']
 assert r['steps_completed']==r['horizon_steps']
 assert r['cache_conditioning']['raw_indices']==[158,159]
 assert r['cache_conditioning']['map']==('candidate' if r['arm']=='PREINSTALLED' else 'source')
 for name in ('source_map','candidate_map','final_map'):
  m=r[name];assert len(m)==160 and set(m)==set(range(128))
 assert r['source_map_sha256']==ah(r['source_map']) and r['candidate_map_sha256']==ah(r['candidate_map'])
 assert r['committed_map_sha256']==ah(r['final_map'])
 assert r['final_map']==(r['source_map'] if r['arm']=='KEEP' else r['candidate_map'])
 assert r['candidate_changed_slots']==sum(a!=b for a,b in zip(r['source_map'],r['candidate_map']))
 assert len(r['source_weight_rank_sha256'])==2 and r['source_weights_sha256']==hashlib.sha256(json.dumps(r['source_weight_rank_sha256']).encode()).hexdigest()
 assert np.isclose(r['predicted_gain'],gain(r['current_counts'],r['source_map'],r['candidate_map']),rtol=0,atol=1e-14)
 assert sum(r['current_counts'])==16*r['batch_tokens']*8*2
 assert [d['step'] for d in r['prefix_decisions']]==[63,127]
 previous=list(range(128))+list(range(32))
 for d in r['prefix_decisions']:
  assert d['current_map']==previous and d['committed_map_sha256']==ah(d['committed_map'])
  assert d['commit_completed'] is True
  assert d['committed_map']==(d['proposed_map'] if d['allowed'] else d['current_map'])
  assert d['transfer_sent_bytes']==sum(d['transfer_sent_bytes_by_rank'])
  physical=np.asarray(d['physical_slot_assignments_window'],np.int32)
  assert physical.shape==(16,160) and ah(physical,np.int32)==d['physical_slot_assignments_sha256']
  assert np.all(physical.sum(axis=1)==r['batch_tokens']*8)
  logical=np.bincount(d['current_map'],weights=physical.sum(axis=0),minlength=128).astype(np.int64)*2
  assert np.array_equal(logical,d['current_counts'])
  if args.inputs_dir:
   raw_start=32+d['step']-15;raw_stop=32+d['step']+1
   raw_ids=raw_bank['expert_ids'][raw_indices[raw_start:raw_stop]]
   assert np.array_equal(np.bincount(raw_ids.ravel(),minlength=128)*2,d['current_counts'])
   assert np.array_equal(cpu_physical(raw_ids,d['current_map']),physical)
  assert np.isclose(d['predicted_gain'],gain(d['current_counts'],d['current_map'],d['proposed_map']),rtol=0,atol=1e-14)
  eligible=d['predicted_gain']>=manifest['policy']['epsilon']
  assert (d.get('would_allow') if d['step']==127 else d['allowed'])==eligible
  if d['step']==127:assert d['intercepted_for_branch'] and not d['allowed']
  old_prefix=np.asarray(d['current_map']).reshape(2,80)
  new_prefix=np.asarray(d['committed_map']).reshape(2,80)
  expected_prefix=[len(set(new_prefix[1])-set(old_prefix[1]))*9437184,len(set(new_prefix[0])-set(old_prefix[0]))*9437184]
  assert d['transfer_sent_bytes_by_rank']==expected_prefix
  assert d['physical_window_measured_step_range']==[d['step']-15,d['step']]
  assert d['physical_counter_input_multiplicity']==2
  assert np.array_equal(physical.reshape(16,2,80).sum(axis=2),d['physical_owner_assignments_per_step'])
  previous=d['committed_map']
 assert previous==r['source_map']
 assert r['current_counts']==r['prefix_decisions'][1]['current_counts']
 assert r['native_proposal']==r['prefix_decisions'][1]['proposed_map']
 assert r['b3_would_allow_native']==r['prefix_decisions'][1]['would_allow']
 assert r['replica_budget_actual']==32
 assert r['prefix_migration_bytes_excluded']==sum(d['transfer_sent_bytes'] for d in r['prefix_decisions'])
 bytes_per_expert=3*2048*768*2
 old=np.asarray(r['source_map']).reshape(2,80);new=np.asarray(r['candidate_map']).reshape(2,80)
 expected_senders=[len(set(new[1])-set(old[1]))*bytes_per_expert,len(set(new[0])-set(old[0]))*bytes_per_expert]
 assert r['candidate_preflight_payload_by_rank']==expected_senders
 expected_payload=expected_senders if r['arm']=='CHARGED' else [0,0]
 assert r['migration_sent_bytes_by_rank']==expected_payload and r['migration_bytes']==sum(expected_payload)
 assert r['untimed_install_bytes']==(sum(expected_senders) if r['arm']=='PREINSTALLED' else 0)
 if r['candidate_id']=='native':assert r['candidate_map']==r['native_proposal']
 else:
  assert r['candidate_changed_slots']==1
  slot=r['replacement_slot'];incoming=r['incoming_logical_expert'];removed=r['removed_logical_expert']
  rank=slot//80;assert r['destination_rank']==rank
  assert r['source_map'][slot]==removed and r['candidate_map'][slot]==incoming
  assert r['source_map'].count(removed)>=2 and incoming not in old[rank] and incoming in old[1-rank]
  assert sum(expected_senders)==bytes_per_expert
 for c in r['correctness_checks']:
  assert c['correctness_reference_tokens']==4 and c['correctness_forward_batch_tokens']==r['batch_tokens']
  assert c['correctness_atol']==.02 and c['correctness_rtol']==.03
 assert [c['raw_step'] for c in r['correctness_checks']]==[160,159+r['horizon_steps']]
 assert r['correctness_max_abs_error']==max(c['correctness_max_abs_error'] for c in r['correctness_checks'])
 p=r['physical_dispatch_diagnostic']
 assert p['physical_map_sha256']==r['committed_map_sha256'] and p['logical_conservation_checks']==r['horizon_steps']
 assert p['raw_step_range_half_open']==[160,160+r['horizon_steps']]
 owner=np.asarray(p['owner_assignments_per_step']);active=np.asarray(p['active_physical_experts_per_rank_per_step'])
 assert owner.shape==active.shape==(r['horizon_steps'],2) and np.all(owner.sum(axis=1)==r['batch_tokens']*8)
 assert np.all((active>=0)&(active<=80)&(owner>=active))
 if args.inputs_dir:
  physical_key=(r['replay_sha256'],r['committed_map_sha256'])
  if physical_key not in raw_dispatch:
   raw_dispatch[physical_key]=cpu_physical(raw_bank['expert_ids'][raw_indices[160:672]],r['final_map'])
  expected_physical=raw_dispatch[physical_key][:r['horizon_steps']]
  assert ah(expected_physical,np.int32)==p['physical_counts_sha256']
  expected_owner=expected_physical.reshape(r['horizon_steps'],2,80)
  assert np.array_equal(owner,expected_owner.sum(axis=2))
  assert np.array_equal(active,(expected_owner>0).sum(axis=2))
 assert all(r[k] for k in ('allocation_id','hostname','process_run_id'))
 group={k:r[k] for k in ('block_id','regime','batch_tokens','candidate_id','horizon_steps','repetition','arm_order')}
 assert r['record_id']==canon({'config_hash':digest,**group,'arm':r['arm']})
 assert r['arm_order'][r['arm_index']]==r['arm']
 groups[key[:-1]].append(r);by_state[(r['block_id'],r['regime'],r['batch_tokens'])].append(r)
assert seen==expected
for rs in groups.values():
 assert len(rs)==3
 equal(rs,('state_id','capture_sha256','replay_sha256','source_map_sha256','source_weights_sha256','source_weight_rank_sha256','candidate_map_sha256','current_counts','prefix_decisions','arm_order','candidate_preflight_payload_by_rank','allocation_id','hostname','process_run_id'))
 arms={r['arm']:r for r in rs}
 assert arms['PREINSTALLED']['physical_dispatch_diagnostic']==arms['CHARGED']['physical_dispatch_diagnostic']
for state,rs in by_state.items():
 equal(rs,('state_id','capture_sha256','replay_sha256','source_map_sha256','source_weights_sha256','source_weight_rank_sha256','current_counts','prefix_decisions','native_proposal'))
 for family in matrix['candidate_ids']:
  same=[r for r in rs if r['candidate_id']==family]
  if not same:continue
  equal(same,('candidate_map_sha256','candidate_preflight_payload_by_rank'))
  for arm in matrix['arms']:
   for rep in sorted({r['repetition'] for r in same}):
    pair=sorted([r for r in same if r['arm']==arm and r['repetition']==rep],key=lambda r:r['horizon_steps'])
    assert [r['horizon_steps'] for r in pair]==[128,512]
    for f in ('owner_assignments_per_step','active_physical_experts_per_rank_per_step'):
     assert pair[0]['physical_dispatch_diagnostic'][f]==pair[1]['physical_dispatch_diagnostic'][f][:128]
  if family=='one_copy':
   first=same[0];old=np.array(first['source_map']);counts=first['current_counts'];best=None
   for slot,removed in enumerate(old):
    if (old==removed).sum()<2:continue
    for incoming in range(128):
     if incoming in old[(slot//80)*80:(slot//80+1)*80]:continue
     candidate=old.copy();candidate[slot]=incoming;g=gain(counts,old,candidate)
     if best is None or g>best[0]:best=(g,slot,incoming,candidate)
   assert best[1]==first['replacement_slot'] and best[2]==first['incoming_logical_expert'] and best[3].tolist()==first['candidate_map']
# Prefix at step63 predates all constructed regime differences. Check it exactly.
for block,batch in itertools.product(matrix['block_ids'],matrix['batch_tokens']):
 representatives=[by_state[(block,regime,batch)][0] for regime in matrix['regimes']]
 assert all(r['prefix_decisions'][0]==representatives[0]['prefix_decisions'][0] for r in representatives)
 equal(representatives,('source_map_sha256','source_weights_sha256','source_weight_rank_sha256'))
 transient=by_state[(block,'transient',batch)][0];persistent=by_state[(block,'persistent',batch)][0]
 equal([transient,persistent],('current_counts','prefix_decisions','native_proposal','source_map_sha256','source_weights_sha256'))
 if args.inputs_dir:
  _,_,ti=raw_tape(by_tape[(block,'transient',batch)]);_,_,pi=raw_tape(by_tape[(block,'persistent',batch)])
  assert np.array_equal(ti[:160],pi[:160])
  assert np.array_equal(ti[160:168],pi[160:168]) and not np.array_equal(ti[168:],pi[168:])
cycle=[['KEEP','PREINSTALLED','CHARGED'],['PREINSTALLED','CHARGED','KEEP'],['CHARGED','KEEP','PREINSTALLED']]
cycle += [list(reversed(x)) for x in cycle.copy()]
orders=[cycle[i%6] for i in range(len(ordered))];random.Random(selection['order_seed']).shuffle(orders)
for key,order in zip(ordered,orders):assert groups[key][0]['arm_order']==order
# Verify actual line order, not just assigned-order labels.
assert [tuple(r[k] for k in KEY_FIELDS)+(r['arm'],) for r in rows]==[k+(a,) for k,o in zip(ordered,orders) for a in o]
positions={a:[sum(order[i]==a for order in orders) for i in range(3)] for a in matrix['arms']}
assert all(max(ns)-min(ns)<=1 for ns in positions.values())
executed=rows[0]['followup_file_sha256'];local=args.source_root
source_agreement={p:(local/p).exists() and hashlib.sha256((local/p).read_bytes()).hexdigest()==sha for p,sha in executed.items()}
source_index_path=provenance_dir/'source-files.json'
archived=json.loads(source_index_path.read_text()) if source_index_path.exists() else {}
assert all(archived[p]==sha for p,sha in executed.items() if p in archived)
archive_missing=sorted(set(executed)-set(archived))
assert all(source_agreement[p] for p in archive_missing)
for r in rows:
 assert r['followup_file_sha256']==executed
 assert r['harness_file_sha256']==rows[0]['harness_file_sha256']
assert all(source_agreement.values()), 'Local source differs from the executed snapshot; provide matching --source-root.'
pairs=[]
for key,rs in sorted(groups.items()):
 arms={r['arm']:r for r in rs};keep=arms['KEEP']['elapsed_seconds'];pre=arms['PREINSTALLED']['elapsed_seconds'];charged=arms['CHARGED']['elapsed_seconds']
 pairs.append({**dict(zip(KEY_FIELDS,key)),'headroom_ms':1000*(keep-pre),'net_payback_ms':1000*(keep-charged),'transition_ms':1000*(charged-pre),'preinstalled_over_keep':pre/keep,'charged_over_keep':charged/keep,'charged_over_preinstalled':charged/pre})
qualified=[p for p in pairs if p['regime']!='stable' and p['horizon_steps']==512 and p['net_payback_ms']>=1]
gates=[{'candidate_id':family,'qualified_states':len([p for p in qualified if p['candidate_id']==family]),'distinct_qualified_blocks':len({p['block_id'] for p in qualified if p['candidate_id']==family}), 'confirmation_trigger':len({p['block_id'] for p in qualified if p['candidate_id']==family})>=2} for family in matrix['candidate_ids']] if stage=='screening' else None
cells=[]
for family,regime,batch,horizon in itertools.product(matrix['candidate_ids'],matrix['regimes'],matrix['batch_tokens'],matrix['horizons']):
 ps=[p for p in pairs if (p['candidate_id'],p['regime'],p['batch_tokens'],p['horizon_steps'])==(family,regime,batch,horizon)]
 if not ps:continue
 cells.append({'candidate_id':family,'regime':regime,'batch_tokens':batch,'horizon_steps':horizon,'blocks':len({p['block_id'] for p in ps}),'triples':len(ps),
 'mean_net_payback_ms':statistics.mean(p['net_payback_ms'] for p in ps),'mean_headroom_ms':statistics.mean(p['headroom_ms'] for p in ps),'mean_transition_ms':statistics.mean(p['transition_ms'] for p in ps),
 'preinstalled_over_keep_geomean':math.exp(statistics.mean(math.log(p['preinstalled_over_keep']) for p in ps)),
 'charged_over_keep_geomean':math.exp(statistics.mean(math.log(p['charged_over_keep']) for p in ps))})
states=[]
for key,rs in sorted(by_state.items()):
 first=rs[0];d,e=first['prefix_decisions']
 states.append({'block_id':key[0],'regime':key[1],'batch_tokens':key[2],'first_b3_step63_committed':d['allowed'],'first_b3_gain':d['predicted_gain'],'first_b3_reason':d['reason'],'prefix_sender_bytes':first['prefix_migration_bytes_excluded'],'target_b3_step127_would_allow':e['would_allow'],'target_b3_gain':e['predicted_gain'],'target_b3_reason':first['b3_native_reason'],'source_is_initial_map':first['source_map']==list(range(128))+list(range(32)),
 'candidates':[{'candidate_id':family,'changed_slots':next(r for r in rs if r['candidate_id']==family)['candidate_changed_slots'],'native_sender_bytes':sum(next(r for r in rs if r['candidate_id']==family)['candidate_preflight_payload_by_rank'])} for family in matrix['candidate_ids'] if any(r['candidate_id']==family for r in rs)]})
# Independently compare saved analyzer point estimates and gate, without importing it.
analysis_agreement=None
if analysis_path is not None and analysis_path.exists():
 summary=json.loads(analysis_path.read_text());assert summary['status']=='complete' and summary['eligible_pairs']==len(pairs) and summary['expected_arm_records']==len(rows)
 analyzed={tuple(p[f] for f in KEY_FIELDS):p for p in summary['pairs']}
 for p in pairs:
  q=analyzed[tuple(p[f] for f in KEY_FIELDS)]
  for left,right in [('headroom_ms','headroom_seconds'),('net_payback_ms','payback_seconds'),('transition_ms','transition_seconds')]:assert math.isclose(p[left],1000*q[right],rel_tol=0,abs_tol=1e-12)
 for c in cells:
  for label,field in [('headroom','preinstalled_over_keep_geomean'),('payback','charged_over_keep_geomean')]:
   a=next(x for x in summary['cells'] if all(x[f]==c[f] for f in ('candidate_id','regime','batch_tokens','horizon_steps')) and x['contrast']==label)
   assert math.isclose(a['ratio'],c[field],rel_tol=0,abs_tol=1e-12)
 if gates:
  for g in gates:
   a=next(x for x in summary['pilot_gate']['candidates'] if x['candidate_id']==g['candidate_id']);assert a['continue']==g['confirmation_trigger']
 analysis_agreement=True
result={'status':'verified','scope':'independent raw-record and source/provenance audit; no performance-based edits','split':manifest['split'],'stage':stage,'records':len(rows),'matched_triples':len(groups),'source_states':len(by_state),'manifest_sha256':digest,'followup_sha256':rows[0]['followup_sha256'],'record_file_sha256':hashlib.sha256(records_path.read_bytes()).hexdigest(),
 'checks':{'exact_declared_matrix_and_unique_record_ids':True,'source_and_candidate_pairing_exact':True,'cross_horizon_repetition_family_source_pairing_exact':True,'cross_regime_step63_state_exact':True,'transient_persistent_step127_history_exact':True,'arm_order_assignment_and_execution_exact':True,'native_payload_matches_two_rank_sender_set_difference':True,'prefix_physical_windows_reconstruct_global_logical_counts':True,'sampled_numerical_checks_pass':True,'final_map_and_weight_attestations_pass':True,'full_future_dispatch_conservation_attestations_pass':True,'archived_execution_source_hashes_match_records':True if source_index_path.exists() else None,'saved_analyzer_point_estimates_and_gate_agree':analysis_agreement,'one_copy_exhaustive_search_independently_reproduced': 'one_copy' in matrix['candidate_ids']},
 'raw_input_reconstruction':{'enabled':bool(args.inputs_dir),'banks_verified':len(raw_banks),'tapes_verified':len(raw_tapes),'unique_future_physical_arrays_reproduced':len(raw_dispatch),'prefix_logical_and_physical_exact':bool(args.inputs_dir),'future_physical_full_hash_owner_counts_and_activity_exact':bool(args.inputs_dir),'transient_persistent_input_indices_identical_through_raw167_then_different':bool(args.inputs_dir)},
 'arm_position_counts':positions,'allocations':sorted({r['allocation_id'] for r in rows}),'hostnames':sorted({r['hostname'] for r in rows}),'process_run_ids':sorted({r['process_run_id'] for r in rows}),
 'source_index_path':str(source_index_path) if source_index_path.exists() else None,'records_path':str(records_path),'manifest_path':str(manifest_path),'local_executed_source_agreement':source_agreement,'execution_hashes_not_in_optional_source_index_but_verified_locally':archive_missing,'numerical_checks':{'checked_steps_per_arm':2,'reference_tokens_per_step':4,'native_forward_uses_full_batch':True,'raw_steps':{'128':[160,287],'512':[160,671]},'maximum_observed_absolute_error':max(r['correctness_max_abs_error'] for r in rows),'atol':.02,'rtol':.03},'screening_gate':gates,'qualified_screening_states':qualified if gates else None,'source_state_details':states,'descriptive_cells':cells,'paired_outcomes':pairs,
 'limits':['Screening uses one repetition per state; descriptive differences are not confirmatory estimates or a robust-feasibility pass.', 'Common prefix/planner and candidate selection costs are excluded in all arms. This diagnoses a fixed candidate conditional on its state; it does not measure a full online controller.', 'Raw NPZ banks/tapes and checkpoint tensors are not reexecuted by this local record audit. Recorded hashes and runtime attestations are checked.', 'Numerical references cover four tokens at two future inputs per arm, not every output element.', 'Future physical diagnostics preserve owner counts/activity and hashes; full per-slot future arrays are runtime-attested rather than independently reconstructed here.', 'Constructed replay of fixed captured prefill tokens, replicated-input/allreduce execution, one layer and allocation do not establish natural-serving or production-population effects.']}
if args.inputs_dir:
 result['limits'][2]='Raw authenticated bank/replay NPZ contents are independently verified; actual prefix logical/physical windows and complete future physical-array hashes are reproduced in NumPy using the pinned native mapping algorithm. The pretrained attention/router computation itself is not rerun.'
 result['limits'][4]='Full future physical counters are independently reconstructed from original logical IDs, replay indices and each arm map; this validates mapping diagnostics but does not profile GPU expert execution.'
if manifest['split']=='final':result['limits'][0]='Technical repetitions share source blocks. This structural audit reports point estimates; confirmatory uncertainty must resample the eight full source-document blocks jointly and use the frozen endpoint families.'
args.output_prefix.parent.mkdir(parents=True,exist_ok=True)
args.output_prefix.with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
lines=[f'# Independent {manifest["split"]} payback audit', '', f'Verified {len(rows)} complete GPU arm records in {len(groups)} matched triples and {len(by_state)} source states. Exact manifest SHA: `{digest}`. No frozen source or records were modified. This checker supports a complete pilot screening matrix or complete final matrix.', '', f'All structural, source/candidate pairing, native sender-payload, prefix histogram, arm-order, map/weight attestation, and sampled numerical checks passed. Maximum numerical absolute error: {result["numerical_checks"]["maximum_observed_absolute_error"]:.9g}; tolerance is atol 0.02 + rtol 0.03 × reference magnitude. Each native check uses the complete batch, with a four-token FP32 reference at each of the first and last future inputs.', '']
if gates:
 for g in gates:lines.append(f'{g["candidate_id"]} screening: {g["qualified_states"]} changing H512 states qualify at ≥1 ms net payback, spanning {g["distinct_qualified_blocks"]} source blocks. Confirmation trigger: {g["confirmation_trigger"]}. The trigger needs two distinct blocks; it is a feasibility screen, not a statistical test.')
lines += ['', 'Source decisions at steps 63 and 127 are reproduced from the recorded physical windows and the frozen 2% equal-sharing peak-load threshold. Step 127 is always intercepted for the forced branch; “would allow” reports the original B3 decision.', '', '| Block | Regime | B | Step63 committed | Step63 gain | Step127 would allow | Step127 gain | Prefix bytes |', '|---|---|---:|---|---:|---|---:|---:|']
for s in states:lines.append(f'| {s["block_id"]} | {s["regime"]} | {s["batch_tokens"]} | {s["first_b3_step63_committed"]} | {100*s["first_b3_gain"]:.3f}% | {s["target_b3_step127_would_allow"]} | {100*s["target_b3_gain"]:.3f}% | {s["prefix_sender_bytes"]:,} |')
lines += ['', 'Descriptive cell means over source blocks (positive payback/headroom means saved time). These descriptive means are not confidence intervals.', '', '| Candidate | Regime | B | H | Headroom ms | Net payback ms | Preinstalled/keep | Charged/keep |', '|---|---|---:|---:|---:|---:|---:|---:|']
for c in cells:lines.append(f'| {c["candidate_id"]} | {c["regime"]} | {c["batch_tokens"]} | {c["horizon_steps"]} | {c["mean_headroom_ms"]:.3f} | {c["mean_net_payback_ms"]:.3f} | {c["preinstalled_over_keep_geomean"]:.6f} | {c["charged_over_keep_geomean"]:.6f} |')
lines += ['', f'Raw archived input verification: {bool(args.inputs_dir)}. Verified {len(raw_banks)} bank files and {len(raw_tapes)} replay files, and reproduced {len(raw_dispatch)} complete future physical-counter arrays. Prefix logical and physical counts, future SHA-256 values, owner counts, and active-expert counts match exactly.', '', 'Interpretation and audit limits:', '']+[f'- {s}' for s in result['limits']]
lines += ['', 'Transient and persistent records have identical observed routing histograms, physical windows, source weights/map, and native proposal at step127 within each block and batch. The correspondence is about the common observed history; later branch outcomes remain distinct. Recorded old/new maps determine sender payloads exactly on the two-rank transport. PREINSTALLED and CHARGED diagnostics match exactly, while their timing contrasts can include cache/transition interference.', '', f'Saved analyzer point estimates and screening gate agreement: {analysis_agreement}. The detailed JSON retains every paired outcome and source-state decision.', '']
args.output_prefix.with_suffix('.md').write_text('\n'.join(lines))
print(json.dumps({k:result[k] for k in ('status','records','matched_triples','source_states','screening_gate','numerical_checks')},indent=2))

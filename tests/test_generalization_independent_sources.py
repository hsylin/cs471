import copy
import pytest
from scripts.verify_generalization_prefixes import canonical_sha, normalized, audit_documents, verify_public_identity
from scripts.verify_external_arrival_sources import verify_raw_block, verify_rank_selection, ns


def source(text,source_id):
    ids=[int(w[1:]) for w in text.split()]
    return {'dataset':'fixture','source_id':source_id,'text':text,'token_ids':ids[:512],
        'full_token_count':len(ids),'captured_token_count':min(512,len(ids)),
        'token_ids_sha256':canonical_sha(ids[:512]),'text_sha256':canonical_sha(text.encode()),
        'normalized_text_sha256':canonical_sha(normalized(text).encode()),
        'captured_prefix_normalized_sha256':canonical_sha(normalized(' '.join(f'w{i}' for i in ids[:512])).encode())}


def encode(text):return [int(w[1:]) for w in text.split()]
def decode(ids):return ' '.join(f'w{i}' for i in ids)


def test_independent_prefix_audit_detects_near_overlap_after_truncation():
    first=list(range(600));second=first[:];second[255]=999;second[512:]=range(2000,2088)
    manifest={'blocks':[{'block_id':'pilot','sources':[source(decode(first),'a')]},
                        {'block_id':'final','sources':[source(decode(second),'b')]}]}
    result=audit_documents(manifest,encode,decode)
    assert result['status']=='failed' and len(result['near_prefix_pairs'])==1
    assert result['near_prefix_pairs'][0]['jaccard']>.95


def test_independent_prefix_audit_rejects_changed_tokens():
    record=source('w1 w2 w3 w4 w5 w6 w7 w8','a');record['token_ids'][0]=77
    with pytest.raises(AssertionError,match='Retokenized prefix'):
        audit_documents({'blocks':[{'block_id':'a','sources':[record]}]},encode,decode)


def test_public_identity_cannot_be_rehashed_into_different_selection():
    private={'blocks':[{'sources':[source('w1 w2 w3 w4 w5 w6 w7 w8','a')]}]}
    private['content_sha256']=canonical_sha(private)
    public=copy.deepcopy(private)
    del public['blocks'][0]['sources'][0]['text'];del public['blocks'][0]['sources'][0]['token_ids']
    public['public_manifest_sha256']=canonical_sha(public)
    verify_public_identity(private,public)
    public['blocks'][0]['sources'][0]['source_id']='different'
    public['public_manifest_sha256']=canonical_sha({k:v for k,v in public.items() if k!='public_manifest_sha256'})
    with pytest.raises(AssertionError,match='exact private selection'):verify_public_identity(private,public)


def raw_fixture():
    rows=[[f'2024-05-12 00:00:00.00000{i}+00:00',10+i,20+i] for i in range(3)]
    block={'jobs':2,'source_date_utc':'2024-05-12','source_rows_sha256':canonical_sha(rows),
        'raw_timestamps':[r[0] for r in rows[:2]],'context_tokens':[10,11],'generated_tokens':[20,21],
        'boundary_timestamp':rows[-1][0],'boundary_context_tokens':12,'boundary_generated_tokens':22,
        'source_origin_ns':ns(rows[0][0]),'boundary_timestamp_ns':ns(rows[-1][0]),
        'original_offsets_ns':[0,1000],'original_horizon_ns':2000,'terminal_gap_ns':1000}
    return block,rows


def test_raw_arrival_boundary_off_by_one_rejected():
    block,rows=raw_fixture();verify_raw_block(block,rows)
    block['original_horizon_ns']=1000
    with pytest.raises(AssertionError,match='Horizon differs'):verify_raw_block(block,rows)


def test_raw_metadata_mutation_rejected_even_with_same_timestamps():
    block,rows=raw_fixture();rows[0][1]=999
    with pytest.raises(AssertionError,match='source-row hash'):verify_raw_block(block,rows)


def test_hash_rank_mutation_rejected():
    candidates=[{'row_start':i*257,'hash_rank_score':str(i)*64} for i in range(3)]
    selected=[{**c,'day_hash_rank':i+1} for i,c in enumerate(candidates[:2])]
    verify_rank_selection(selected,candidates)
    selected[0]={**candidates[2],'day_hash_rank':1}
    with pytest.raises(AssertionError,match='hash winners'):verify_rank_selection(selected,candidates)

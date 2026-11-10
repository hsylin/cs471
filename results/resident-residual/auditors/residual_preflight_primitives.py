"""Independent reconstruction of the nine fixed selector/alignment probes."""
import json
from pathlib import Path
import numpy as np
from audit_residual_diagnostic import digest,integer_hash,require


def cases():
    shared=np.tile(np.arange(8),(128,1))
    single=np.tile([32,33,34,35,80,81,82,83],(128,1))
    asymmetric=np.array([[r%32]+[32+(r*7+c)%48 for c in range(7)] for r in range(128)])
    boundary=np.tile(np.arange(32,40),(128,1))
    for c,n in enumerate((63,64,65)):boundary[:n,c]=c
    mixed=shared.copy();mixed[1::2]=single[1::2]
    partial=shared.copy();partial[64:]=-1
    return [('shared_count_ties',shared,128),('singleton_only',single,128),
        ('asymmetric_singleton_owners',asymmetric,128),('tile_boundary_counts',boundary,128),
        ('hot_shared_with_forced_owners',np.tile([0,32,33,34,80,81,82,83],(128,1)),128),
        ('shared_and_singleton_ties',mixed,128),('all_logical_experts',np.arange(1024).reshape(128,8)%128,128),
        ('partial_real_rows',partial,64),('zero_real_rows',np.full((128,8),-1),0)]


def expected(ids,real):
    counts=np.bincount(ids[:real].ravel(),minlength=128)
    forced=[int(np.count_nonzero(counts[32:80])),int(np.count_nonzero(counts[80:]))]
    active=forced.copy();loads=[int(counts[32:80].sum()),int(counts[80:].sum())]
    shared=np.flatnonzero(counts[:32]).tolist();cap=max(*forced,(sum(forced)+len(shared)+1)//2)
    homes=np.arange(128,dtype=np.int64)
    for e in sorted(shared,key=lambda e:(-int(counts[e]),e)):
        owner=min((r for r in (0,1) if active[r]<cap),key=lambda r:(loads[r],r))
        homes[e]=e+128*owner;active[owner]+=1;loads[owner]+=int(counts[e])
    physical=np.full(ids.shape,-1,dtype=np.int64);valid=(ids>=0)&(ids<128);physical[valid]=homes[ids[valid]]
    return dict(physical_ids=physical,homes=homes,logical_counts=counts,
        summary=np.array([cap,*forced,len(shared),*loads,*active],dtype=np.int64))


def check(fused,alignment):
    require(fused['schema']=='resident_fused_adversarial_proof_v1' and fused['passed'] and fused['cases_checked']==9 and fused['gpu_invocations_checked']==18 and fused['dtypes_checked']==['int32','int64'],'Fused probe allocation')
    require(all(fused[k] is True for k in ('rank_agreement','stream_context_unchanged','internal_real_tokens_unchanged','input_ids_unchanged','external_real_tokens_unchanged')) and fused['internal_real_tokens']==128 and fused['reference_count']==2,'Fused invariant proof')
    require(fused['evidence_sha256']==digest(fused['cases']),'Fused evidence hash')
    expected_keys={(name,dtype) for name,_,_ in cases() for dtype in ('int32','int64')}
    seen=set()
    for row in fused['cases']:
        key=row['case'],row['dtype'];require(key in expected_keys and key not in seen,'Fused duplicate/missing case');seen.add(key)
        _,ids,real=next(c for c in cases() if c[0]==key[0]);values=expected(ids,real)
        require(row['real_tokens']==real and row['input_ids_sha256']==integer_hash(ids),'Adversarial input changed')
        require(all(row[field+'_sha256']==integer_hash(value) for field,value in values.items()),'Actual GPU output field differs from independent scalar policy')
    require(seen==expected_keys,'Fused missing probe')
    require(alignment['schema']=='resident_residual_alignment_adversarial_v1' and alignment['passed'] and alignment['cases']==9 and alignment['native_alignment_calls_per_rank']==72 and alignment['capacity_bound']==5056,'Alignment probe allocation')
    require(alignment['real_token_default_preserved'] and alignment['stream_context_preserved'] and len(alignment['per_rank'])==2,'Alignment context proof')
    calls=0
    for rank,rows in enumerate(alignment['per_rank']):
        seen=set()
        for row in rows:
            key=row['case'],row['dtype'];require(key in expected_keys and key not in seen,'Alignment duplicate/missing case');seen.add(key)
            _,ids,real=next(c for c in cases() if c[0]==key[0]);values=expected(ids,real);physical=values['physical_ids']
            require(row['logical_ids_sha256']==integer_hash(ids) and row['physical_ids_sha256']==integer_hash(physical) and row['real_tokens']==real and row['rank_assignment_agreement'],'Alignment input/policy identity')
            require(set(row['layouts'])=={'stockoff','previous'},'Alignment layout set')
            for layout,nslot,routes in [('stockoff',64,ids),('previous',80,physical)]:
                pair=row['layouts'][layout];require(pair['canonical_assignment_and_padding_equal'],'Full/trim different semantics')
                local=routes[(routes>=rank*nslot)&(routes<(rank+1)*nslot)]-rank*nslot
                counts=np.bincount(local.ravel(),minlength=nslot);pad=int(np.sum((counts+63)//64*64))
                experts=np.repeat(np.arange(nslot),(counts+63)//64)
                for view in ('full','trimmed'):
                    record=pair[view];calls+=1
                    require(record['scope']=='alignment_only' and record['gemms']==[],'Adversarial scope overclaim')
                    require(record['actual_padded_rows']==pad<=5056 and record['actual_m_tiles']==pad//64 and record['physical_ids_sha256']==integer_hash(routes),'Actual alignment padding differs')
                    require(record['independent_local_counts']==dict(tokens=int(counts.sum()),active=int(np.count_nonzero(counts)),padded_rows=pad,capacity_bound=5056),'Actual local assignment counts differ')
                    require(record['valid_expert_ids_sha256']==integer_hash(experts),'Actual aligned expert bucket order differs')
                    require(record['backing_pointer_preserved'] and record['valid_prefix_equal'] and record['token_conservation_verified'],'Recorded native prefix check failed')
        require(seen==expected_keys,'Missing alignment input')
    return dict(passed=True,selector_cases=18,fields_per_selector_case=4,native_alignment_calls= calls,
        scope='Independent CPU reconstruction of all selector field hashes and alignment expert-bucket counts/order; sorted-token within-bucket order remains native runtime attestation')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('fused','alignment','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),'Fresh output required')
    result=check(json.loads(a.fused.read_text()),json.loads(a.alignment.read_text()))
    result.update(fused_file_sha256=digest(a.fused.read_bytes()),alignment_file_sha256=digest(a.alignment.read_bytes()),auditor_sha256=digest(Path(__file__).read_bytes()))
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

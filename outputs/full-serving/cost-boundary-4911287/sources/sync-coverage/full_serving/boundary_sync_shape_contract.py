"""Pure contracts for aligned native-DP diagnostic samples."""
def selected(ordinal,profile=False,capturing=False):
    assert type(ordinal) is int and ordinal>0
    return not profile and not capturing and (ordinal<=8 or ordinal%64==0)

def empty_summary(padded_tokens,padded_requests):
    assert type(padded_tokens) is int and padded_tokens>=0
    return dict(real_requests=0,real_tokens=0,prefill_tokens=0,decode_tokens=0,
                padded_tokens=padded_tokens,padded_requests=padded_requests,
                sequence_cpu_upper_max=0,attention_pairs_cpu_upper_proxy=0,
                no_actual_request_batch=True,empty_evidence="native execute_model dummy_run=True,is_profile=False; pinned branch states no actual tokens")

def aligned_summary(peers):
    """Only common, explicitly observed samples support simultaneous comparisons."""
    assert len(peers)==8
    common=set.intersection(*(set(p) for p in peers))
    samples=[]
    for ordinal in sorted(common):
        records=[p[ordinal] for p in peers]
        assert all(r["sync_call"]==ordinal and r["sample_policy"]=="native_dp_sync_ordinal" for r in records)
        vectors=[r["padded_tokens_across_dp"] for r in records]
        assert all(v is not None and len(v)==8 and v==vectors[0] for v in vectors)
        real=[r["real_tokens"] for r in records]
        assert all(type(x) is int and x>=0 for x in real)
        for r in records:
            assert r["profile"] is False
            if r["dummy"]:assert r["no_actual_request_batch"] and r["real_tokens"]==0
            else:assert not r.get("no_actual_request_batch",False)
        total=sum(real)
        samples.append(dict(sync_call=ordinal,real_tokens_per_rank=real,
                            dummy_ranks=[i for i,r in enumerate(records) if r["dummy"]],
                            prefill_tokens_per_rank=[r["prefill_tokens"] for r in records],
                            decode_tokens_per_rank=[r["decode_tokens"] for r in records],
                            attention_pairs_upper_per_rank=[r["attention_pairs_cpu_upper_proxy"] for r in records],
                            padded_tokens_across_dp=vectors[0],
                            real_token_max_over_mean=max(real)/(total/8) if total else None,
                            total_real_tokens=total))
    return dict(all_rank_observed_samples=len(samples),samples=samples,
                simultaneous_evidence_available=bool(samples),
                limitation="Systematic native DP-ordinal samples, not unbiased frequencies. Padded sizes are not actual work; CPU attention proxies are not exact FLOPs/KV. No GPU critical-path inference.")

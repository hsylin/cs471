"""Synthetic CPU fixtures test provenance/index rules, never performance evidence."""
import hashlib
from types import SimpleNamespace
import numpy as np
import pytest
from followup.router_tapes import MODEL, REVISION
from generalization.inputs import INPUT_SEED, build_job_indices, reuse_summary, validate_input_bank


def fixture_bank():
    n, length = 320, 80
    token_ids = np.arange(n, dtype=np.int64)
    prompt_index = np.repeat(np.arange(4, dtype=np.int32), length)
    prompts = []
    for i in range(4):
        prompts.append(dict(dataset='test', source_id=f'doc{i}', prompt_id=f'doc{i}',
            captured_tokens=length, total_source_tokens=length, position_start=0, token_offset=0,
            token_ids_sha256=hashlib.sha256(token_ids[prompt_index == i].tobytes()).hexdigest(),
            native_boundary_exact=True, native_router_exact=True,
            manual_router_exact=True, bf16_roundtrip_exact=True))
    config = dict(source_model=MODEL, source_revision=REVISION, layer=0,
                  num_experts=128, top_k=8, hidden_size=2048)
    m = dict(config, block_id='test-pilot-01', split='pilot', dataset='test',
        selection_content_sha256='newselection', study_id='generalization_v1', correctness_pass=True,
        native_serving_trace=False, timing_evidence=False, origin='prompt_causal_prefill_layer0',
        prompts=prompts, source_prompt_ids=[f'test:doc{i}' for i in range(4)])
    bank = SimpleNamespace(metadata=m, sha256='filehash', token_ids=token_ids,
        prompt_index=prompt_index, token_position=np.tile(np.arange(length, dtype=np.int32), 4),
        domain_ids=(prompt_index % 2).astype(np.int8),
        hidden_states_bf16=np.zeros((n, 2048), dtype=np.uint16),
        expert_ids=np.tile(np.arange(8, dtype=np.int64), (n, 1)),
        combining_weights=np.full((n, 8), .125, dtype=np.float32))
    entry = dict(block_id=m['block_id'], capture_sha256=bank.sha256, split='pilot')
    plan = dict(input_seed=INPUT_SEED, content_selection_sha256='newselection', phase='pilot')
    group = dict(block_id=m['block_id'], dataset='test', regime='newcontent', batch_tokens=128)
    return bank, entry, config, plan, group


def test_jobs_are_distinct_combinations_of_authentic_record_indices():
    a, m = build_job_indices(1093, 'test-pilot-01')
    assert a.dtype == np.int64 and a.shape == (272, 128)
    assert ((a >= 0) & (a < 1093)).all()
    assert all(len(set(row)) == 128 for row in a)
    assert len({tuple(sorted(row)) for row in a}) == 272
    assert len({tuple(sorted(row)) for row in a[16:]}) == 256
    assert set((a % 2).ravel()) == {0, 1}
    assert m['natural_request_or_batch_trace'] is False
    assert m['selection_independent_of_arm_calendar_repetition'] is True


def test_reproducible_without_global_rng_state_or_arm():
    a, ma = build_job_indices(320, 'test-pilot-01')
    np.random.seed(9182); np.random.random(500)
    b, mb = build_job_indices(320, 'test-pilot-01')
    np.testing.assert_array_equal(a, b)
    assert ma == mb
    c, _ = build_job_indices(320, 'test-pilot-02')
    assert not np.array_equal(a, c)


def test_reuse_records_conserve_every_occurrence_and_separate_warmup():
    a, m = build_job_indices(320, 'test-pilot-01')
    for name, rows in [('all_jobs', a), ('warmup_jobs', a[:16]), ('measured_jobs', a[16:])]:
        counts = np.bincount(rows.ravel(), minlength=320); s = m['job_reuse'][name]
        assert s['token_record_occurrences'] == counts.sum() == rows.size
        assert s['maximum_uses'] == counts.max()
        assert s['distinct_captured_records_used'] == np.count_nonzero(counts)
    assert m['job_reuse']['measured_jobs']['token_record_occurrences'] == 32768
    assert m['job_reuse']['all_jobs']['distinct_captured_records_used'] < 34816


@pytest.mark.parametrize('count', [0, 127, 128, 129, True])
def test_impossible_bank_rejected(count):
    with pytest.raises(ValueError):build_job_indices(count, 'test-pilot-01')


def test_duplicate_record_and_duplicate_unordered_job_rejected():
    a, _ = build_job_indices(320, 'test-pilot-01')
    b = a.copy(); b[0, 1] = b[0, 0]
    with pytest.raises(ValueError, match='twice within'):reuse_summary(b, 320)
    b = a.copy(); b[17] = b[16][::-1]
    with pytest.raises(ValueError, match='Repeated job'):reuse_summary(b, 320)


def test_record_selection_preserves_joint_fields():
    bank, entry, config, plan, group = fixture_bank()
    validation = validate_input_bank(bank, entry, config, plan, group)
    a, _ = build_job_indices(len(bank.expert_ids), group['block_id'])
    bank.hidden_states_bf16[:, 0] = np.arange(320, dtype=np.uint16)
    selected_hidden = bank.hidden_states_bf16[a]
    for raw, pos in [(0, 0), (16, 127), (271, 45)]:
        source = a[raw, pos]
        assert selected_hidden[raw, pos, 0] == source
        assert bank.token_ids[a][raw, pos] == source
        assert bank.prompt_index[a][raw, pos] * 80 + bank.token_position[a][raw, pos] == source
        np.testing.assert_array_equal(bank.expert_ids[a][raw, pos], bank.expert_ids[source])
        np.testing.assert_array_equal(bank.combining_weights[a][raw, pos], bank.combining_weights[source])
    assert validation['source_documents'] == 4


@pytest.mark.parametrize('field,value', [('source_revision', 'different'), ('layer', 1),
                                        ('num_experts', 64), ('top_k', 2), ('hidden_size', 1024)])
def test_wrong_checkpoint_or_geometry_rejected(field, value):
    args = fixture_bank(); args[0].metadata[field] = value
    with pytest.raises(ValueError, match='identity|geometry'):validate_input_bank(*args)


def test_old_selection_cannot_enter_new_plan():
    args = fixture_bank(); args[3]['content_selection_sha256'] = 'v3-after-prefix-dedup'
    with pytest.raises(ValueError, match='content selection'):validate_input_bank(*args)


def test_final_rejects_pilot_and_pilot_rejects_final():
    args = fixture_bank(); args[3]['phase'] = 'final'
    with pytest.raises(ValueError, match='reuse pilot'):validate_input_bank(*args)
    args = fixture_bank(); args[0].metadata['split'] = args[1]['split'] = 'final'
    with pytest.raises(ValueError, match='held-out final'):validate_input_bank(*args)


@pytest.mark.parametrize('corruption', ['token', 'position', 'parity', 'attestation', 'source', 'missing_prompt', 'nonfinite', 'hash'])
def test_corrupted_record_provenance_rejected(corruption):
    args = fixture_bank(); bank, entry, _, _, _ = args
    if corruption == 'token':bank.token_ids[0] += 1
    elif corruption == 'position':bank.token_position[0] = 1
    elif corruption == 'parity':bank.domain_ids[0] = 1
    elif corruption == 'attestation':bank.metadata['prompts'][0]['native_router_exact'] = False
    elif corruption == 'source':bank.metadata['source_prompt_ids'][0] = 'test:other'
    elif corruption == 'missing_prompt':bank.prompt_index[0] = 99
    elif corruption == 'nonfinite':bank.hidden_states_bf16[0, 0] = 0x7f80
    elif corruption == 'hash':entry['capture_sha256'] = 'wrong'
    with pytest.raises(ValueError):validate_input_bank(*args)

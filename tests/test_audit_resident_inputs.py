"""Synthetic CPU fixtures only; no held-out demand or performance inspection."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from followup.router_tapes import MODEL, REVISION, _save, compose_router_tape, save_router_bank

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('audit_resident_inputs', ROOT/'scripts/audit_resident_inputs.py')
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def valid_arrays():
    return {
        'hidden_states_bf16': np.zeros((4096, 2048), np.uint16),
        'expert_ids': np.tile(np.arange(8, dtype=np.int64), (4096, 1)),
        'combining_weights': np.full((4096, 8), .125, np.float32),
        'token_ids': np.arange(4096, dtype=np.int64),
        'domain_ids': np.repeat(np.array([0, 1], np.int8), 2048),
        'prompt_index': np.repeat(np.array([0, 1], np.int32), 2048),
        'token_position': np.tile(np.arange(2048, dtype=np.int32), 2),
    }


def test_array_contract_accepts_exact_bf16_fixture():
    AUDIT.validate_bank_arrays(valid_arrays())


@pytest.mark.parametrize('damage', ['duplicate_expert', 'invalid_expert', 'ids_dtype',
                                    'nonfinite_hidden', 'unrounded_weight', 'wrong_position',
                                    'wrong_domain_dtype', 'extra_array'])
def test_array_contract_rejects_non_native_or_misaligned_records(damage):
    arrays = valid_arrays()
    if damage == 'duplicate_expert': arrays['expert_ids'][0, 1] = arrays['expert_ids'][0, 0]
    elif damage == 'invalid_expert': arrays['expert_ids'][0, 0] = 128
    elif damage == 'ids_dtype': arrays['expert_ids'] = arrays['expert_ids'].astype(np.int32)
    elif damage == 'nonfinite_hidden': arrays['hidden_states_bf16'][0, 0] = 0x7f80
    elif damage == 'unrounded_weight': arrays['combining_weights'][0, 0] = np.float32(.12500001)
    elif damage == 'wrong_position': arrays['token_position'][2048] = 2048
    elif damage == 'wrong_domain_dtype': arrays['domain_ids'] = arrays['domain_ids'].astype(np.int64)
    elif damage == 'extra_array': arrays['replacement_hidden'] = np.zeros(1)
    with pytest.raises(AssertionError):
        AUDIT.validate_bank_arrays(arrays)


def test_file_hash_cannot_mask_internal_npz_content_tampering(tmp_path):
    path = tmp_path/'fixture.npz'
    _save(path, {'fixture': 'synthetic-test-only'}, {'indices': np.array([1, 2], np.int64)})
    AUDIT.npz_contents(path, AUDIT.file_hash(path))
    with np.load(path, allow_pickle=False) as saved:
        parts = {name: saved[name].copy() for name in saved.files}
    parts['indices'][0] = 3
    with path.open('wb') as handle:
        np.savez_compressed(handle, **parts)
    with pytest.raises(AssertionError, match='Content checksum'):
        AUDIT.npz_contents(path, AUDIT.file_hash(path))


@pytest.mark.parametrize('regime', AUDIT.REGIMES)
def test_independent_schedule_matches_producer_and_rejects_wrong_seed(tmp_path, regime):
    arrays = valid_arrays()
    # Small hidden dimension only for this composition fixture; it never passes
    # the production bank geometry check and is not a native capture artifact.
    arrays['hidden_states_bf16'] = arrays['hidden_states_bf16'][:, :4].copy()
    metadata = {'schema': 'qwen_router_bank_v1', 'source_model': MODEL,
        'source_revision': REVISION, 'layer': 0, 'num_experts': 128, 'top_k': 8,
        'hidden_size': 4, 'block_id': 'synthetic-cpu-fixture', 'split': 'test',
        'source_prompt_ids': ['test-prose', 'test-code']}
    bank = save_router_bank(tmp_path/'fixture.npz', metadata, arrays)
    tape = compose_router_tape(bank.path, tmp_path/'tape.npz', regime=regime,
        batch_tokens=128, num_steps=544, warmup_steps=32,
        phase_steps=112 if regime == 'switching' else 120,
        duration_steps=16, repeat_steps=1024, seed=20471)
    AUDIT.verify_replay_arrays(tape.metadata, {'indices': tape.indices}, 128, AUDIT.expected_positions(128))
    with pytest.raises(AssertionError, match='independently regenerated'):
        AUDIT.verify_replay_arrays(tape.metadata, {'indices': tape.indices}, 128,
                                   AUDIT.expected_positions(128, seed=10471))


def test_missing_capture_is_incomplete_and_does_not_claim_verification(tmp_path):
    result = AUDIT.audit(tmp_path, ROOT/'resident_dispatch/corpus/sources.json')
    assert result['status'] == 'incomplete'
    assert result['expected_bank_files'] == 8 and result['expected_replay_files'] == 64
    assert result['routing_distributions_inspected'] is False
    assert result['runtime_or_tuning_outcomes_inspected'] is False

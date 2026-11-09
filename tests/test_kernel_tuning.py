"""CPU input/order contracts for the separate tuning experiment."""
from collections import Counter
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from optimization.kernel_config import CANDIDATES
from optimization.tuning import input_indices, tuning_groups, _persist_indices, kernel_event_summary


def test_saved_tuning_inputs_are_deterministic_complete_records_without_intrabatch_replacement():
    domains = np.repeat([0, 1], 2048)
    indices, seed = input_indices(domains, "pilot-a", 1, 1024)
    again, again_seed = input_indices(domains, "pilot-a", 1, 1024)
    assert indices.shape == (48, 1024) and seed == again_seed
    assert np.array_equal(indices, again) and np.all(domains[indices] == 1)
    assert all(len(np.unique(row)) == 1024 for row in indices)
    other, _ = input_indices(domains, "pilot-b", 1, 1024)
    assert not np.array_equal(indices, other)
    assert len(indices[:16]) == 16 and len(indices[16:48]) == 32
    with pytest.raises(ValueError, match="Insufficient"):
        input_indices(domains[:2], "a", 1, 128)


def test_all192_trials_retained_and_candidate_position_balanced():
    manifest = {"matrix": {"block_ids": ["a", "b"], "domains": [0, 1],
                           "batch_tokens": [128, 1024], "repetitions": 3,
                           "candidate_ids": list(CANDIDATES)},
                "tuning": {"forwards_per_trial": 32, "candidate_configs": CANDIDATES},
                "execution": {"order_seed": 10471}}
    groups = tuning_groups(manifest)
    assert groups == tuning_groups(manifest) and len(groups) == 24
    assert sum(len(g["candidate_order"]) for g in groups) == 192
    for position in range(8):
        assert Counter(g["candidate_order"][position] for g in groups) == {c: 3 for c in CANDIDATES}
    assert all(set(g["candidate_order"]) == set(CANDIDATES) for g in groups)
    manifest["tuning"]["forwards_per_trial"] = 31
    with pytest.raises(ValueError, match="32 timed"):
        tuning_groups(manifest)


def test_saved_input_sha_is_actual_file_and_reuse_checks_exact_indices(tmp_path):
    rt = SimpleNamespace(rank=0, ep_group=SimpleNamespace(cpu_group=None),
                         torch=SimpleNamespace(distributed=SimpleNamespace(
                             broadcast_object_list=lambda *a, **kw: None)))
    bank = SimpleNamespace(sha256="a" * 64, metadata={"source_prompt_ids": ["prose", "code"]})
    group = {"block_id": "pilot-a", "domain": 0, "batch_tokens": 128}
    indices, seed = input_indices(np.repeat([0, 1], 2048), "pilot-a", 0, 128)
    first = _persist_indices(rt, tmp_path / "records.jsonl", bank, indices, seed, group)
    from pathlib import Path
    assert first["sha256"] == hashlib.sha256(Path(first["path"]).read_bytes()).hexdigest()
    assert _persist_indices(rt, tmp_path / "records.jsonl", bank, indices, seed, group) == first
    assert not list(tmp_path.rglob("*.tmp"))
    changed = indices.copy(); changed[0, 0] += 1
    with pytest.raises(AssertionError, match="Saved tuning inputs"):
        _persist_indices(rt, tmp_path / "records.jsonl", bank, changed, seed, group)


def test_profile_summary_counts_only_actual_kernels_not_duplicate_annotations():
    trace = {"traceEvents": [
        {"cat": "kernel", "ph": "X", "name": "fused_moe_kernel", "dur": 100},
        {"cat": "kernel", "ph": "X", "name": "fused_moe_kernel", "dur": 200},
        {"cat": "cpu_op", "ph": "X", "name": "vllm::fused_experts", "dur": 999},
        {"cat": "user_annotation", "ph": "X", "name": "captured_fixed_map_moe", "dur": 999},
        {"cat": "kernel", "ph": "X", "name": "ncclDevKernel_AllReduce_Sum", "dur": 10},
    ]}
    result = kernel_event_summary(trace, 1)
    assert result["fused_expert_gemm"] == {"calls": 2, "total_us": 300, "us_per_forward": 300}
    assert result["nccl_allreduce"]["total_us"] == 10
    assert "other_kernels" not in result
    with pytest.raises(AssertionError, match="two real"):
        kernel_event_summary(trace, 2)

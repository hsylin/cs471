"""CPU-checkable boundaries of the GPU harness; these are not timing evidence."""
import json
from pathlib import Path

import numpy as np
import pytest

from skewbench.native import (
    CountingCommunicator, initial_mapping, validate_layer_shapes, verify_native_sources,
)
from skewbench.runner import build_jobs, _trace_config
from types import SimpleNamespace


def test_physical_slots_cover_every_logical_expert_with_fixed_extra_budget():
    mapping = initial_mapping(128, 32, 2)
    assert mapping.shape == (160,)
    assert np.array_equal(np.unique(mapping), np.arange(128))
    assert np.array_equal(np.bincount(mapping), np.r_[np.full(32, 2), np.ones(96)])
    with pytest.raises(ValueError, match="divide evenly"):
        initial_mapping(128, 31, 2)


def test_export_rejects_transposed_or_missing_expert_shapes():
    cfg = {"num_experts": 128, "hidden_size": 2048, "intermediate_size": 768}
    validate_layer_shapes((128, 1536, 2048), (128, 2048, 768), cfg)
    with pytest.raises(ValueError, match="shapes"):
        validate_layer_shapes((128, 2048, 1536), (128, 2048, 768), cfg)


def test_counter_counts_only_posted_sender_payload_and_forwards_calls():
    class Tensor:
        def __init__(self, n, size):
            self.n, self.size = n, size
        def numel(self):
            return self.n
        def element_size(self):
            return self.size
    class Inner:
        def __init__(self):
            self.calls = []
        def add_send(self, tensors, dst, **kw):
            self.calls.append((tensors, dst, kw))
        def execute(self):
            return "done"
    inner = Inner()
    counter = CountingCommunicator(inner)
    tensors = [Tensor(12, 2), Tensor(6, 4)]
    counter.add_send(tensors, 1, expert_id=5)
    assert counter.sent_bytes == 48
    assert counter.sent_messages == 1
    assert inner.calls == [(tensors, 1, {"expert_id": 5})]
    assert counter.execute() == "done"
    assert counter.sent_bytes == 48


def test_source_attestation_fails_closed_on_unpinned_install(tmp_path):
    with pytest.raises(RuntimeError, match="does not match"):
        verify_native_sources(tmp_path)


def test_randomized_jobs_preserve_declared_paired_cells_and_phase_variants():
    manifest = {"matrix": {"methods": ["B2", "B3", "B4", "B5", "SKIP"],
                 "cases": [{"regime": "burst", "trace_variant": "core", "phase_steps": 0},
                           {"regime": "burst", "trace_variant": "phase_shift", "phase_steps": 4}],
                 "batch_tokens": [128], "seeds": [11, 12], "repetitions": 2}}
    jobs = build_jobs(manifest)
    assert jobs == build_jobs(manifest)
    assert len(jobs) == 40
    assert len({json.dumps(job, sort_keys=True) for job in jobs}) == 40
    for seed in (11, 12):
        for rep in (0, 1):
            for variant in ("core", "phase_shift"):
                subset = [j for j in jobs if (j["seed"], j["repetition"], j["trace_variant"])
                          == (seed, rep, variant)]
                assert {j["method"] for j in subset} == set(manifest["matrix"]["methods"])


def test_measured_horizon_excludes_two_real_history_windows_and_case_phase_wins():
    manifest = {"trace": {"num_experts": 128, "top_k": 8, "num_steps": 512,
                          "window_steps": 16, "phase_steps": 999,
                          "burst_repeat_steps": 128, "hotspot_lifetime_steps": 16}}
    job = {"seed": 11, "regime": "burst", "batch_tokens": 128, "phase_steps": 56}
    weights = SimpleNamespace(config={"num_experts": 128, "top_k": 8})
    cfg = _trace_config(manifest, job, weights)
    assert cfg.num_steps == 544 and cfg.warmup_steps == 32
    assert cfg.phase_steps == 56 and cfg.burst_repeat_steps == 128

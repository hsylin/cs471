import types

import numpy as np
import pytest

from resident_residual.alignment import (
    AlignmentTrim, CAPACITY, EXPECTED_CONFIG, one_home_active_cap,
    padded_capacity_bound, verify_count_bound,
)


class FakeTensor:
    def __init__(self, array, *, dtype="torch.int32", pointer=None):
        self.a = np.asarray(array)
        self.shape = self.a.shape
        self.dtype = dtype
        self.is_cuda = True
        self._version = 0
        self.pointer = id(self) if pointer is None else pointer
    def is_contiguous(self):
        return True
    def data_ptr(self):
        return self.pointer
    def __getitem__(self, key):
        return FakeTensor(self.a[key], dtype=self.dtype, pointer=self.pointer)
    def numel(self):
        return self.a.size
    def clone(self):
        return FakeTensor(self.a.copy(), dtype=self.dtype)
    def detach(self):
        return self
    def cpu(self):
        return self
    def numpy(self):
        return self.a
    def item(self):
        return self.a.item()


def fake_guard(*, enabled=True, probe=False, global_experts=160):
    guard = AlignmentTrim.__new__(AlignmentTrim)
    guard.enabled, guard.probe = enabled, probe
    guard._installed = False
    guard.calls = 0
    guard.records = []
    guard.global_experts = global_experts
    guard.expert_map = FakeTensor(np.arange(global_experts))
    guard.map_version = 0
    guard.map_pointer = guard.expert_map.data_ptr()
    guard.native_calls = []
    def original(*args, **kwargs):
        guard.native_calls.append((args, kwargs))
        return (FakeTensor(np.arange(1024 + global_experts * 63)),
                FakeTensor(np.arange(174)), FakeTensor([4096]))
    guard.original = original
    guard.original_invoke = lambda *args, **kwargs: "native GEMM"
    guard.module = types.SimpleNamespace(_prepare_expert_assignment=original,
        invoke_fused_moe_triton_kernel=guard.original_invoke)
    return guard


def call(guard, **overrides):
    kwargs = dict(topk_ids=FakeTensor(np.zeros((128, 8), dtype=np.int32)),
        config=dict(EXPECTED_CONFIG), num_tokens=128, top_k_num=8,
        global_num_experts=guard.global_experts, expert_map=guard.expert_map,
        ignore_invalid_experts=True)
    kwargs.update(overrides)
    return guard.module._prepare_expert_assignment(**kwargs)


def test_policy_cap_exhaustive_geometry():
    caps = [one_home_active_cap(a, b, s)
            for a in range(49) for b in range(49) for s in range(33)]
    assert max(caps) == 64
    assert padded_capacity_bound() == CAPACITY == 79 * 64


def test_bound_attainable_with_valid_unique_top8_rows():
    # A legitimate degree sequence: 15 experts have65 tokens,49 have1.
    # Greedily realize128 rows of8 distinct IDs. This reaches79 M tiles.
    remaining = np.array([65] * 15 + [1] * 49)
    rows = []
    for _ in range(128):
        selected = np.argsort(-remaining, kind="stable")[:8]
        assert np.all(remaining[selected] > 0)
        remaining[selected] -= 1
        rows.append(selected)
    ids = np.array(rows)
    assert not np.any(remaining)
    assert all(len(set(row)) == 8 for row in ids)
    counts = np.bincount(ids.ravel(), minlength=64)
    assert counts.max() <= 128
    assert verify_count_bound(counts)["padded_rows"] == CAPACITY


def test_padding_edges_and_empty_input():
    for counts, expected in [([0], 0), ([1, 63, 64, 65, 128], 448), ([1024], 1024)]:
        assert verify_count_bound(np.array(counts))["padded_rows"] == expected
    assert padded_capacity_bound(0, 64, 64) == 0


@pytest.mark.parametrize("counts", [np.ones(65, dtype=int), np.array([1025]),
    np.array([-1]), np.array([1.5]), np.zeros((2, 2), dtype=int)])
def test_count_assumption_violations_rejected(counts):
    with pytest.raises(ValueError):
        verify_count_bound(counts)


@pytest.mark.parametrize("global_experts", [128, 160])
def test_trim_preserves_native_call_and_backing(global_experts):
    g = fake_guard(global_experts=global_experts)
    with g.active():
        view, experts, padded = call(g)
        assert view.shape == (CAPACITY,)
        args, kw = g.native_calls[0]
        assert args[4] == global_experts
        assert args[5] is g.expert_map
        assert kw == dict(ignore_invalid_experts=True, use_int8_w8a16=False,
                          use_int4_w4a16=False, block_shape=None)
        assert experts.shape == (174,)
        assert padded.a.tolist() == [4096]
        assert g.calls == 1
    assert g.module._prepare_expert_assignment is g.original
    assert g.module.invoke_fused_moe_triton_kernel is g.original_invoke


def test_disabled_installs_nothing():
    g = fake_guard(enabled=False)
    with g.active():
        assert g.module._prepare_expert_assignment is g.original
        assert call(g)[0].shape == (11104,)
    assert g.calls == 0


def test_probe_full_geometry_delegates_unchanged():
    g = fake_guard(enabled=False, probe=True)
    with g.active():
        full, _, _ = call(g)
        assert full is g.records[0]["full"]
        assert full is g.records[0]["view"]
        assert g.module.invoke_fused_moe_triton_kernel != g.original_invoke


def test_restoration_after_error_and_nesting_rejection():
    g = fake_guard()
    with pytest.raises(RuntimeError, match="deliberate"):
        with g.active():
            with pytest.raises(RuntimeError, match="nonnested"):
                with fake_guard().active():
                    pass
            raise RuntimeError("deliberate")
    assert g.module._prepare_expert_assignment is g.original
    assert g.module.invoke_fused_moe_triton_kernel is g.original_invoke
    with g.active():
        assert call(g)[0].shape == (CAPACITY,)


@pytest.mark.parametrize("mutation", ["shape", "filter", "config", "map", "version", "quant"])
def test_fail_closed_before_native_call(mutation):
    g = fake_guard()
    kwargs = {}
    if mutation == "shape":
        kwargs["topk_ids"] = FakeTensor(np.zeros((64, 8), dtype=int))
    if mutation == "filter":
        kwargs["ignore_invalid_experts"] = False
    if mutation == "config":
        kwargs["config"] = {**EXPECTED_CONFIG, "BLOCK_SIZE_M": 32}
    if mutation == "map":
        kwargs["expert_map"] = FakeTensor(np.arange(160))
    if mutation == "version":
        g.expert_map._version += 1
    if mutation == "quant":
        kwargs["use_int8_w8a16"] = True
    with g.active(), pytest.raises(RuntimeError, match="unsafe trim"):
        call(g, **kwargs)
    assert not g.native_calls


def test_wrong_original_buffer_rejected():
    g = fake_guard()
    g.original = lambda *args, **kwargs: (FakeTensor(np.arange(1024)), None, None)
    g.module._prepare_expert_assignment = g.original
    with g.active(), pytest.raises(RuntimeError, match="alignment buffer"):
        call(g)


def complete_probe():
    g = fake_guard(probe=True, global_experts=128)
    g.expert_map.a[64:] = -1
    g.layer = types.SimpleNamespace(slots=64,
        torch=types.SimpleNamespace(equal=lambda a, b: np.array_equal(a.a, b.a)))
    g.policy, g.config, g.source_hashes = "stockoff", dict(EXPECTED_CONFIG), {}
    ids = np.tile(np.arange(8), (128, 1))  #128 occurrences of each unique top8 expert
    full = np.full(9088, 1024, dtype=np.int32)
    full[:1024] = np.concatenate([np.flatnonzero(ids.ravel() == e) for e in range(8)])
    experts = np.full(142, -1, dtype=np.int32)
    experts[:16] = np.repeat(np.arange(8), 2)
    def original(*args, **kwargs):
        return FakeTensor(full), FakeTensor(experts), FakeTensor([1024])
    def invoke(B, sorted_token_ids, expert_ids, num_tokens_post_padded, top_k, config):
        return "native"
    g.original, g.original_invoke = original, invoke
    g.module._prepare_expert_assignment = original
    g.module.invoke_fused_moe_triton_kernel = invoke
    with g.active():
        view, exp, pad = call(g, topk_ids=FakeTensor(ids))
        for n, k in [(1536, 8), (2048, 1)]:
            g.module.invoke_fused_moe_triton_kernel(types.SimpleNamespace(shape=(64, n, 2048)),
                view, exp, pad, k, dict(EXPECTED_CONFIG))
    return g


def test_probe_both_gemms_and_max128_expert_conservation():
    report = complete_probe().verify()
    row = report["probe_records"][0]
    assert row["token_conservation_verified"]
    assert row["actual_padded_rows"] == 1024
    assert [g["grid_programs"] for g in row["gemms"]] == [948, 1264]


@pytest.mark.parametrize("mutation", ["dropped", "wrong_expert", "padding", "missing_gemm"])
def test_probe_catches_incorrect_alignment_or_invocation(mutation):
    g = complete_probe()
    r = g.records[0]
    if mutation == "dropped":
        r["full"].a[0] = 1024
    if mutation == "wrong_expert":
        r["experts"].a[0] = 7
    if mutation == "padding":
        r["padded"].a[0] = 5120
    if mutation == "missing_gemm":
        r["gemms"].pop()
    with pytest.raises(AssertionError):
        g.verify()

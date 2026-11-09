"""CPU tests for native override lifetime; not GPU correctness or timing evidence."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from optimization.kernel_config import (
    CANDIDATES, config_fingerprint, kernel_override, verify_kernel_sources, resolve_kernel_config,
)


def fake_native(previous):
    module = SimpleNamespace(value=previous)
    module.get_config = lambda: module.value
    @contextmanager
    def unsafe_override(config):
        old = module.value
        module.value = config
        yield
        module.value = old  # Deliberately reproduce native's missing finally.
    module.override_config = unsafe_override
    return module


def test_declared_search_is_eight_choices_and_preserves_fixed_dimensions():
    assert len(CANDIDATES) == 8 and CANDIDATES["DEFAULT"] is None
    explicit = [c for c in CANDIDATES.values() if c is not None]
    assert len({config_fingerprint(c) for c in explicit}) == 7
    assert all(c["BLOCK_SIZE_K"] == 64 and c["GROUP_SIZE_M"] == c["SPLIT_K"] == 1
               and c["num_stages"] == 3 for c in explicit)


@pytest.mark.parametrize("candidate", list(CANDIDATES))
def test_override_restores_prior_exact_object_on_success_and_failure(candidate):
    previous = {"sentinel": 17}
    native = fake_native(previous)
    config = CANDIDATES[candidate]
    with kernel_override(config, native_module=native):
        assert native.get_config() == config
        if config is not None:
            assert native.get_config() is not config
    assert native.get_config() is previous
    with pytest.raises(RuntimeError, match="candidate failed"):
        with kernel_override(config, native_module=native):
            raise RuntimeError("candidate failed")
    assert native.get_config() is previous


def test_nested_default_scope_restores_outer_candidate_after_exception():
    native = fake_native(None)
    config = CANDIDATES["M16_N64_K64_W4_S3"]
    with kernel_override(config, native_module=native):
        outer = native.get_config()
        with pytest.raises(ValueError):
            with kernel_override(None, native_module=native):
                assert native.get_config() is None
                raise ValueError("inner failed")
        assert native.get_config() is outer
    assert native.get_config() is None


def test_unregistered_tuning_choice_rejected_before_native_mutation():
    native = fake_native(None)
    wrong = dict(CANDIDATES["M16_N64_K64_W4_S3"], num_stages=8)
    with pytest.raises(ValueError, match="bounded"):
        with kernel_override(wrong, native_module=native):
            pass
    assert native.get_config() is None


def test_native_source_attestation_fails_closed(tmp_path):
    directory = tmp_path / "model_executor/layers/fused_moe"
    directory.mkdir(parents=True)
    (directory / "__init__.py").write_text("changed")
    (directory / "fused_moe.py").write_text("changed")
    with pytest.raises(RuntimeError, match="pinned"):
        verify_kernel_sources(tmp_path)


def test_resolution_uses_native_exact_shape_dtype_selector_and_rejects_changed_fallback():
    layer = SimpleNamespace(dtype="torch.bfloat16", top_k=8,
                            w13=SimpleNamespace(shape=(80, 1536, 2048)),
                            w2=SimpleNamespace(shape=(80, 2048, 768)))
    native = fake_native(None)
    observed = {}
    def selector(**kwargs):
        observed.update(kwargs)
        return native.get_config() or CANDIDATES["M64_N128_K64_W4_S3"]
    implementation = SimpleNamespace(_get_config_dtype_str=lambda **kwargs: None,
                                     try_get_optimal_moe_config=selector)
    assert resolve_kernel_config(layer, 128, native_module=native,
                                 implementation_module=implementation) == CANDIDATES["M64_N128_K64_W4_S3"]
    assert observed == {"w1_shape": (80, 1536, 2048), "w2_shape": (80, 2048, 768),
                        "top_k": 8, "dtype": None, "M": 128, "block_shape": None}
    with kernel_override(CANDIDATES["M16_N64_K64_W4_S3"], native_module=native):
        assert resolve_kernel_config(layer, 1024, native_module=native,
                                     implementation_module=implementation) == CANDIDATES["M16_N64_K64_W4_S3"]
    with pytest.raises(AssertionError, match="Resolved"):
        resolve_kernel_config(layer, 1024, native_module=native, implementation_module=implementation)

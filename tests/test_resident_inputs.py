"""CPU-only source isolation, capture attestation, and constructed-replay checks."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from resident_dispatch.corpus.validation import (
    ROOT, PRIOR_MANIFESTS, sha256, source_text, validate_captured_provenance,
    validate_source_manifest,
)
from followup.router_tapes import MODEL, REVISION, compose_router_tape, save_router_bank

SOURCES = ROOT / "resident_dispatch/corpus/sources.json"


def manifest():
    return json.loads(SOURCES.read_text())


def test_all_sixteen_documents_exclude_both_prior_corpora():
    result = validate_source_manifest(manifest())
    assert result["source_blocks"] == 8
    assert result["source_documents"] == 16
    assert result["prior_documents_excluded"] == 40
    assert result["source_document_counts"] == {"pilot": 0, "final": 8}
    assert result["overlap_by_id_url_raw_or_cleaned_content"] is False


@pytest.mark.parametrize("prior_path", PRIOR_MANIFESTS)
@pytest.mark.parametrize("field", ["source_id", "url", "sha256", "cleaned_text_sha256"])
def test_renaming_cannot_hide_document_reuse(prior_path, field):
    data = manifest()
    old = json.loads((ROOT / prior_path).read_text())["blocks"][0]["sources"][0]
    value = old[field] if field != "cleaned_text_sha256" else hashlib.sha256(
        source_text(old, (ROOT / prior_path).parent).encode()).hexdigest()
    data["blocks"][0]["sources"][0][field] = value
    with pytest.raises(ValueError, match="overlap"):
        validate_source_manifest(data)


def test_missing_final_block_and_changed_prior_pin_are_rejected():
    data = manifest()
    data["blocks"].pop()
    with pytest.raises(ValueError, match="eight final"):
        validate_source_manifest(data)
    data = manifest()
    data["excluded_source_manifests"][1]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="pins changed"):
        validate_source_manifest(data)


def test_archived_source_tampering_is_detected(tmp_path):
    directory = tmp_path / "corpus"
    shutil.copytree(SOURCES.parent, directory)
    data = manifest()
    source = directory / data["blocks"][0]["sources"][0]["file"]
    source.write_bytes(source.read_bytes() + b"\nchanged")
    with pytest.raises(ValueError, match="Source checksum"):
        validate_source_manifest(data, directory)


def test_native_capture_computation_is_exact_predecessor():
    old = (ROOT / "scripts/capture_optimization_tapes.py").read_text()
    new = (ROOT / "scripts/capture_resident_tapes.py").read_text()
    # These blocks cover all checkpoint tensors, loading, decoder construction,
    # actual forward/router calls, and independent boundary/serialization checks.
    for start, end in [("    config_path, index_path =", "    index_file ="),
                       ("        for domain, source in enumerate", "        arrays =")]:
        old_code = old[old.index(start):old.index(end)]
        new_code = new[new.index(start):new.index(end)]
        assert old_code == new_code
    old_tree, new_tree = ast.parse(old), ast.parse(new)
    pins = ("TRANSFORMERS_VERSION", "MODELING_SHA256", "MASKING_SHA256")
    def constants(tree):
        return {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id in pins}
    assert constants(old_tree) == constants(new_tree)


def attested_metadata():
    data = manifest()
    block = data["blocks"][0]
    return {
        "block_id": block["block_id"], "study_id": "resident_dispatch_v1",
        "source_manifest_sha256": sha256(SOURCES),
        "source_prompt_ids": [s["source_id"] for s in block["sources"]],
        "split": "final", "origin": "teacher_forced_causal_prefill_layer0",
        "native_serving_trace": False, "correctness_pass": True,
        "capture_script_sha256": sha256(ROOT / "scripts/capture_resident_tapes.py"),
        "source_validation_module_sha256": sha256(ROOT / "resident_dispatch/corpus/validation.py"),
        "source_text_module_sha256": sha256(ROOT / "optimization/input_sources.py"),
        "router_tapes_module_sha256": sha256(ROOT / "followup/router_tapes.py"),
        "capture_computation_predecessor_sha256": sha256(ROOT / "scripts/capture_optimization_tapes.py"),
        "excluded_source_manifests": data["excluded_source_manifests"],
        "prompts": [{"prompt_id": s["source_id"], "cleaned_text_sha256": s["cleaned_text_sha256"],
                     "native_boundary_exact": True, "native_router_exact": True,
                     "manual_router_exact": True, "bf16_roundtrip_exact": True}
                    for s in block["sources"]],
    }


@pytest.mark.parametrize("field", ["capture_script_sha256", "source_validation_module_sha256",
                                  "router_tapes_module_sha256", "source_manifest_sha256"])
def test_old_or_changed_capture_implementation_is_rejected(field):
    metadata = attested_metadata()
    validate_captured_provenance(metadata, manifest(), SOURCES)
    metadata[field] = "0" * 64
    with pytest.raises(ValueError, match="provenance differs"):
        validate_captured_provenance(metadata, manifest(), SOURCES)


@pytest.mark.parametrize("flag", ["native_boundary_exact", "native_router_exact", "manual_router_exact",
                                 "bf16_roundtrip_exact"])
def test_missing_native_capture_attestation_is_rejected(flag):
    metadata = attested_metadata()
    metadata["prompts"][0][flag] = False
    with pytest.raises(ValueError, match="native attestation"):
        validate_captured_provenance(metadata, manifest(), SOURCES)


def test_fresh_schedule_keeps_exact_bank_records_and_single_transient(tmp_path):
    # Synthetic CPU fixture only. It is never eligible as a production capture.
    n = 4096
    arrays = {
        "expert_ids": np.tile(np.array([[0, 1]], np.int64), (n, 1)),
        "combining_weights": np.tile(np.array([[.25, .75]], np.float32), (n, 1)),
        "hidden_states_bf16": np.arange(n * 4, dtype=np.uint16).reshape(n, 4),
        "token_ids": np.arange(n, dtype=np.int64), "prompt_index": np.repeat([0, 1], n // 2).astype(np.int32),
        "token_position": np.tile(np.arange(n // 2, dtype=np.int32), 2),
        "domain_ids": np.repeat([0, 1], n // 2).astype(np.int8),
    }
    metadata = {"schema": "qwen_router_bank_v1", "source_model": MODEL, "source_revision": REVISION,
                "layer": 0, "num_experts": 8, "top_k": 2, "hidden_size": 4,
                "block_id": "synthetic-cpu-test-only", "split": "test",
                "source_prompt_ids": ["test-prose", "test-code"]}
    bank = save_router_bank(tmp_path / "fixture.npz", metadata, arrays)
    original_hash = sha256(bank.path)
    tapes = {}
    for regime in ("stable", "transient", "persistent", "switching"):
        tapes[regime] = compose_router_tape(bank.path, tmp_path / f"{regime}.npz", regime=regime,
            batch_tokens=128, num_steps=544, warmup_steps=32,
            phase_steps=112 if regime == "switching" else 120,
            duration_steps=16, repeat_steps=1024, seed=20471)
    transient = tapes["transient"]
    assert transient.indices.shape == (544, 128)
    assert sum(transient.metadata["domain_schedule"]) == 16
    assert transient.metadata["domain_schedule"][152:168] == [1] * 16
    np.testing.assert_array_equal(transient.indices[:152], tapes["persistent"].indices[:152])
    np.testing.assert_array_equal(transient.indices[:32], tapes["stable"].indices[:32])
    np.testing.assert_array_equal(transient.expert_ids, arrays["expert_ids"][transient.indices])
    np.testing.assert_array_equal(transient.combining_weights, arrays["combining_weights"][transient.indices])
    assert all(len(set(row)) == 128 for row in transient.indices)
    assert sha256(bank.path) == original_hash
    other = compose_router_tape(bank.path, tmp_path / "old-seed.npz", regime="transient",
        batch_tokens=128, num_steps=544, warmup_steps=32, phase_steps=120,
        duration_steps=16, repeat_steps=1024, seed=10471)
    assert not np.array_equal(transient.indices, other.indices)


def test_composer_refuses_existing_output_without_mutation(tmp_path):
    spec = importlib.util.spec_from_file_location("compose_resident_tapes", ROOT / "scripts/compose_resident_tapes.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "existing"
    output.mkdir()
    (output / "marker").write_text("unchanged")
    with pytest.raises(FileExistsError):
        module.compose_directory(tmp_path / "banks", output)
    assert (output / "marker").read_text() == "unchanged"

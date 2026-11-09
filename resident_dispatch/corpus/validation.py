"""CPU provenance checks excluding all documents in both previous corpora."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from optimization.input_sources import source_text

ROOT = Path(__file__).resolve().parents[2]
PRIOR_MANIFESTS = ("followup/corpus/sources.json", "optimization/corpus/sources.json")
STUDY_ID = "resident_dispatch_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_source_manifest(manifest: dict, directory: Path | None = None) -> dict:
    directory = directory or ROOT / "resident_dispatch/corpus"
    if manifest.get("study_id") != STUDY_ID:
        raise ValueError("Expected distinct resident-dispatch study identity")
    required_pins = [{"path": p, "sha256": sha256(ROOT / p)} for p in PRIOR_MANIFESTS]
    if manifest.get("excluded_source_manifests") != required_pins:
        raise ValueError("Prior source-manifest pins changed")
    previous = {key: set() for key in ("source_id", "url", "sha256", "cleaned_text_sha256")}
    prior_documents = 0
    for name in PRIOR_MANIFESTS:
        path = ROOT / name
        for block in json.loads(path.read_text())["blocks"]:
            for source in block["sources"]:
                prior_documents += 1
                for key in ("source_id", "url", "sha256"):
                    previous[key].add(source[key])
                cleaned = source_text(source, path.parent)
                previous["cleaned_text_sha256"].add(hashlib.sha256(cleaned.encode()).hexdigest())
    if prior_documents != 40:
        raise ValueError("Expected exactly forty prior source documents")
    seen = {key: set() for key in previous}
    blocks = set()
    for block in manifest["blocks"]:
        block_id = block["block_id"]
        if block_id in blocks:
            raise ValueError("Repeated source block")
        blocks.add(block_id)
        if block["split"] != "final" or block_id not in {f"rd-final-{n:02d}" for n in range(1, 9)}:
            raise ValueError("Expected final resident-dispatch block identity")
        if [s["domain"] for s in block["sources"]] != ["prose", "code"]:
            raise ValueError("Each block must contain prose then code")
        for source in block["sources"]:
            for key in seen:
                value = source[key]
                if value in seen[key] or value in previous[key]:
                    raise ValueError(f"Source document overlap detected by {key}")
                seen[key].add(value)
            text = source_text(source, directory)
            if len(text) < 5000:
                raise ValueError("Source too short; require a genuine full document")
            if source["domain"] == "prose":
                metadata = directory / source["metadata_file"]
                if sha256(metadata) != source["metadata_sha256"]:
                    raise ValueError("Gutenberg metadata checksum mismatch")
                if "Public domain in the USA" not in metadata.read_text():
                    raise ValueError("Archived catalog lacks declared copyright status")
            elif source.get("source_revision") != manifest["cpython_revision"]:
                raise ValueError("CPython revision differs from manifest")
    if blocks != {f"rd-final-{n:02d}" for n in range(1, 9)}:
        raise ValueError("Expected exactly eight final source pairs")
    if sha256(directory / "raw/cpython_LICENSE") != manifest["cpython_license_sha256"]:
        raise ValueError("CPython license checksum mismatch")
    return {
        "study_id": STUDY_ID,
        "source_blocks": len(blocks),
        "source_documents": len(seen["source_id"]),
        "prior_documents_excluded": prior_documents,
        "source_document_counts": {"pilot": 0, "final": 8},
        "overlap_by_id_url_raw_or_cleaned_content": False,
        "all_source_hashes_verified": True,
    }


def validate_captured_provenance(metadata: dict, manifest: dict, sources: Path) -> None:
    """Reject old or unattested banks before constructing a new replay."""
    blocks = {b["block_id"]: b for b in manifest["blocks"]}
    block = blocks.get(metadata.get("block_id"))
    if block is None:
        raise ValueError("Captured bank is not in the fresh source inventory")
    expected = {
        "study_id": STUDY_ID,
        "source_manifest_sha256": sha256(sources),
        "source_prompt_ids": [s["source_id"] for s in block["sources"]],
        "split": "final",
        "origin": "teacher_forced_causal_prefill_layer0",
        "native_serving_trace": False,
        "correctness_pass": True,
        "capture_script_sha256": sha256(ROOT / "scripts/capture_resident_tapes.py"),
        "source_validation_module_sha256": sha256(Path(__file__)),
        "source_text_module_sha256": sha256(ROOT / "optimization/input_sources.py"),
        "router_tapes_module_sha256": sha256(ROOT / "followup/router_tapes.py"),
        "capture_computation_predecessor_sha256": sha256(ROOT / "scripts/capture_optimization_tapes.py"),
        "excluded_source_manifests": manifest["excluded_source_manifests"],
    }
    for key, value in expected.items():
        actual = metadata.get(key)
        if (actual is not value) if isinstance(value, bool) else (actual != value):
            raise ValueError(f"Fresh capture provenance differs: {key}")
    prompts = metadata.get("prompts", [])
    if len(prompts) != 2 or [p.get("prompt_id") for p in prompts] != expected["source_prompt_ids"]:
        raise ValueError("Captured prompt provenance differs from inventory")
    for prompt, source in zip(prompts, block["sources"]):
        for key in ("native_boundary_exact", "native_router_exact", "manual_router_exact", "bf16_roundtrip_exact"):
            if prompt.get(key) is not True:
                raise ValueError(f"Captured prompt lacks native attestation: {key}")
        if prompt.get("cleaned_text_sha256") != source["cleaned_text_sha256"]:
            raise ValueError("Captured prompt cleaned-content pin differs")

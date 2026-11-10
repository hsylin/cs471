"""Public-only protocol checks and fail-closed private materialization gates.

These tests use the repository's admitted public metadata. No real private
payload is read; private serialization tests use synthetic Unicode messages.
"""
import json
from pathlib import Path

import pytest

from full_serving import freeze_custom_compare as freezer
from full_serving.data import MODEL_REVISION, digest, file_digest
from full_serving.resident_plugin import PIN, balanced_map

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT/"results/full-serving/custom-compare-v1/protocol-draft.json"


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+"\n")
    return file_digest(path)


def public_gate_files(tmp_path, monkeypatch, *, passed):
    config = tmp_path/"config.json"
    sources = tmp_path/"sources.json"
    admission = tmp_path/"admission.json"
    write(config, {"schema": "cs471_full_resident_v1", "mode": "resident_v1", "vllm_commit": PIN,
                   "model_revision": MODEL_REVISION, "physical_to_logical_map": balanced_map(), "startup_diagnostic": False})
    write(sources, {"test_only": "No model implementation is executed by these gate tests"})
    # Scope the filesystem test to gate ordering, independent of live custom
    # source edits by another agent; production verifies the complete manifest.
    monkeypatch.setattr(freezer, "verify_source_manifest", lambda value: None)
    write(admission, {"status": "passed", "pilot_only": True, "correctness_gate_passed": True,
                      "performance_gate_passed": passed, "custom_config_sha256": file_digest(config),
                      "custom_source_manifest_sha256": file_digest(sources)})
    return config, sources, admission


def test_actual_twenty_public_calendars_verify_without_private_access(monkeypatch):
    monkeypatch.setattr(freezer, "_source_rows", lambda *args: pytest.fail("private data read"))
    _, metadata, report = freezer.check_draft(DRAFT)
    assert report["status"] == "passed" and len(report["calendars"]) == 20
    assert report["private_payloads_read"] is False and len(metadata["by_id"]) == 640


def test_changed_calendar_is_rejected_even_after_rehashing_its_file_and_draft(tmp_path):
    draft = json.loads(DRAFT.read_text())
    for key, value in list(draft["source_admission"].items()):
        if key.endswith("_file"):
            draft["source_admission"][key] = str(freezer.compare.artifact_path(DRAFT, value))
    for dataset in draft["datasets"]:
        dataset["pilot_calendar_file"] = str(freezer.compare.artifact_path(DRAFT, dataset["pilot_calendar_file"]))
        for block in dataset["blocks"]:
            block["calendar_file"] = str(freezer.compare.artifact_path(DRAFT, block["calendar_file"]))
    block = draft["datasets"][0]["blocks"][0]
    calendar = json.loads(Path(block["calendar_file"]).read_text())
    calendar["arrival_offsets_ns"][1] += 1
    block["calendar_file"] = str(tmp_path/"altered-calendar.json")
    block["calendar_sha256"] = write(Path(block["calendar_file"]), calendar)
    draft["protocol_sha256"] = digest({k: v for k, v in draft.items() if k != "protocol_sha256"})
    path = tmp_path/"draft.json"; write(path, draft)
    with pytest.raises(ValueError, match="exact rational scaling"):
        freezer.check_draft(path)


def test_failed_pilot_gate_precedes_any_private_read_or_output(tmp_path, monkeypatch):
    config, sources, admission = public_gate_files(tmp_path, monkeypatch, passed=False)
    monkeypatch.setattr(freezer, "_source_rows", lambda *args: pytest.fail("private data read before admission"))
    output, private = tmp_path/"frozen/protocol.json", tmp_path/"private-output"
    with pytest.raises(ValueError, match="both custom pilot gates"):
        freezer.freeze(DRAFT, config, sources, admission, tmp_path/"nonexistent-input", private, output, execute=True)
    assert not output.parent.exists() and not private.exists()


def test_admitted_dry_freeze_still_never_opens_or_materializes_private_inputs(tmp_path, monkeypatch):
    config, sources, admission = public_gate_files(tmp_path, monkeypatch, passed=True)
    monkeypatch.setattr(freezer, "_source_rows", lambda *args: pytest.fail("private data read during dry freeze"))
    output, private = tmp_path/"frozen/protocol.json", tmp_path/"private-output"
    result = freezer.freeze(DRAFT, config, sources, admission, tmp_path/"nonexistent-input", private, output)
    assert result["pilot_admission"] == "passed" and result["execute"] is False
    assert not output.parent.exists() and not private.exists()


def test_source_bytes_unicode_and_private_permissions_are_preserved(tmp_path):
    row = {"id": "synthetic", "messages": [{"role": "user", "content": "甲\u2028乙"*1000}]}
    raw = (json.dumps(row, ensure_ascii=True, sort_keys=True)+"\n").encode()
    path = tmp_path/"private"/"requests.jsonl"
    freezer._private_write(path, raw)
    copied, rows = freezer._source_rows(path, digest(raw))
    assert copied == raw and rows == [row] and path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    with pytest.raises(FileExistsError):
        freezer._private_write(path, raw)
    with pytest.raises(ValueError, match="source payload bytes changed"):
        freezer._source_rows(path, "0"*64)

"""Admission rejects plausible corruption even when outer hashes are resealed."""
import json

import pytest

from full_serving.data import PriorExclusions, digest, prepare_dataset
from full_serving.prepare_data import serialize_requests
from full_serving.verify_data import read_jsonl, verify_bundle


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(range(sum(len(m["content"]) for m in messages) + 3))


@pytest.fixture
def bundle(tmp_path):
    raw = [{"conversation_id": str(i), "language": "English", "toxic": False,
            "conversation": [{"role": "user", "content": f"Prompt {i} \u2028 separator \n genuine newline"},
                             {"role": "assistant", "content": "Excluded target"}]} for i in range(8)]
    rows, manifest = prepare_dataset(raw, Tokenizer(), PriorExclusions(), pilot_requests=4,
                                    final_bundles=2, requests_per_bundle=2, smoke_requests=2)
    private = tmp_path / "private"
    private.mkdir()
    by_id = {r["id"]: r for r in rows}
    files = {"requests": rows, "pilot": rows[:4], "pilot-00": rows[:4],
             "smoke": [by_id[i] for i in manifest["smoke_request_ids"]],
             "final-00": rows[4:6], "final-01": rows[6:]}
    for name, records in files.items():
        (private / (name + ".jsonl")).write_text(serialize_requests(records))
    manifest["private_payload_sha256"] = digest((private / "requests.jsonl").read_bytes())
    manifest["public_manifest_sha256"] = digest(manifest)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return private, path


def reseal_payload(private, path):
    manifest = json.loads(path.read_text())
    manifest["private_payload_sha256"] = digest((private / "requests.jsonl").read_bytes())
    manifest["public_manifest_sha256"] = digest({k: v for k, v in manifest.items() if k != "public_manifest_sha256"})
    path.write_text(json.dumps(manifest))


def test_complete_bundle_matches_manifest_and_retokenizes_optionally(bundle):
    private, manifest = bundle
    report = verify_bundle(private, manifest)
    assert report["requests"] == 8 and report["status"] == "passed"
    assert report["recomputed_tokenization"] is False
    assert len(report["verified_files"]) == 6
    assert verify_bundle(private, manifest, tokenizer=Tokenizer())["recomputed_tokenization"] is True


def test_public_seal_tampering_is_detected(bundle):
    private, path = bundle
    manifest = json.loads(path.read_text())
    manifest["seed"] += 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest seal"):
        verify_bundle(private, path)


def test_complete_payload_tampering_is_detected_before_content_parse(bundle):
    private, manifest = bundle
    path = private / "requests.jsonl"
    path.write_text(path.read_text().replace("Prompt", "Edited", 1))
    with pytest.raises(ValueError, match="payload hash"):
        verify_bundle(private, manifest)


def test_message_tampering_still_fails_when_outer_payload_hash_is_resealed(bundle):
    private, manifest = bundle
    _, rows = read_jsonl(private / "requests.jsonl")
    rows[0]["messages"][0]["content"] += " changed"
    (private / "requests.jsonl").write_text(serialize_requests(rows))
    reseal_payload(private, manifest)
    with pytest.raises(ValueError, match="Message content hash"):
        verify_bundle(private, manifest)


def test_input_token_metadata_cannot_drift_from_public_manifest(bundle):
    private, manifest = bundle
    _, rows = read_jsonl(private / "requests.jsonl")
    rows[0]["input_tokens"] += 1
    (private / "requests.jsonl").write_text(serialize_requests(rows))
    reseal_payload(private, manifest)
    with pytest.raises(ValueError, match="metadata mismatch"):
        verify_bundle(private, manifest)


@pytest.mark.parametrize("name", ["smoke.jsonl", "pilot.jsonl", "pilot-00.jsonl", "final-00.jsonl"])
def test_subset_order_is_part_of_the_contract(bundle, name):
    private, manifest = bundle
    _, rows = read_jsonl(private / name)
    (private / name).write_text(serialize_requests(rows[::-1]))
    with pytest.raises(ValueError, match="Ordered subset"):
        verify_bundle(private, manifest)


def test_retokenization_mismatch_is_explicit(bundle):
    class ChangedTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return [999]
    with pytest.raises(ValueError, match="Retokenized prompt differs"):
        verify_bundle(*bundle, tokenizer=ChangedTokenizer())


def test_parser_preserves_literal_unicode_line_separators(tmp_path):
    path = tmp_path / "input.jsonl"
    row = {"messages": [{"role": "user", "content": "hello\u2028world\u2029again"}]}
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n")
    assert read_jsonl(path)[1] == [row]

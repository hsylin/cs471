"""Source admission rejects previously measured data relabeled as held out."""
import copy
import json

import pytest

from full_serving.custom_inputs import DATASETS, load_source_metadata, verify_private_request
from full_serving.data import MODEL_REVISION, digest, normalized


def sealed_write(path, value, seal=None):
    if seal:
        value[seal] = digest({k: v for k, v in value.items() if k != seal})
    raw = (json.dumps(value, indent=2) + "\n").encode()
    path.write_bytes(raw)
    return digest(raw)


def request(dataset, index):
    rid = f"{dataset}-{index}"
    messages = [{"role": "user", "content": rid + " 中文\u2028authentic input"}]
    row = {"id": rid, "dataset": dataset, "messages": messages,
           "input_tokens": 900, "max_tokens": 2048, "prompt_token_ids_sha256": "a"*64,
           "messages_sha256": digest(messages),
           "normalized_messages_sha256": digest([{**m, "content": normalized(m["content"])} for m in messages]),
           "first_user_normalized_sha256": digest(normalized(messages[0]["content"]).encode()),
           "split": "pilot" if index < 64 else "final",
           "content_block": "pilot-00" if index < 64 else f"final-{(index-64)//64:02d}",
           "conversation_id" if dataset == "wildchat" else "group_id": rid}
    return row


def fixture(tmp_path):
    rows = {name: [{k: v for k, v in request(name, i).items() if k != "messages"}
                   for i in range(448 if name == "wildchat" else 128)] for name in DATASETS}
    wild = {"schema": "full_serving_requests_v1", "source_revision": MODEL_REVISION, "requests": rows["wildchat"]}
    wild_sha = sealed_write(tmp_path/"wild.json", wild, "public_manifest_sha256")
    suite = {"schema": "full_serving_separate_suite_v1", "source_model_revision": MODEL_REVISION,
             "selection_status": "pilot_ready_final_candidates_not_frozen", "cohorts": [],
             "wildchat_admission": {"manifest_file_sha256": wild_sha}}
    for name in DATASETS:
        cohort = {"dataset": name, "private_files": {"final.jsonl": "b"*64, "smoke.jsonl": "c"*64},
                  "smoke_request_ids": [r["id"] for r in rows[name][:8]]}
        cohort.update({"parent_manifest_sha256": wild_sha} if name == "wildchat" else {"requests": rows[name]})
        suite["cohorts"].append(cohort)
    suite_sha = sealed_write(tmp_path/"suite.json", suite, "manifest_sha256")
    admission = {"schema": "full_serving_suite_admission_v1", "status": "passed",
                 "suite_manifest_file_sha256": suite_sha, "suite_manifest_sha256": suite["manifest_sha256"],
                 "selected_requests": 640, "pilot_requests": 320, "reserved_final_candidates": 320,
                 "recomputed_tokenization": True, "tokenizer": {"model_revision": MODEL_REVISION}}
    admitted_sha = sealed_write(tmp_path/"admission.json", admission)
    protocol = {"source_admission": {
        "suite_manifest_file": "suite.json", "suite_manifest_sha256": suite_sha,
        "wildchat_manifest_file": "wild.json", "wildchat_manifest_sha256": wild_sha,
        "suite_admission_file": "admission.json", "suite_admission_sha256": admitted_sha}, "datasets": []}
    for name in DATASETS:
        dataset = {"dataset": name, "reserved_source_final_file_sha256": "b"*64,
                   "source_public_manifest_sha256": wild_sha if name == "wildchat" else suite_sha,
                   "warmup": {"split": "pilot", "request_count": 8, "requests_sha256": "c"*64,
                              "request_ids": [r["id"] for r in rows[name][:8]]}, "blocks": []}
        for index in range(4):
            ids = [r["id"] for r in rows[name][64+index*16:64+(index+1)*16]]
            dataset["blocks"].append({"split": "final", "request_count": 16,
                                      "request_ids": ids, "request_ids_sha256": digest(ids)})
        protocol["datasets"].append(dataset)
    return protocol, tmp_path/"protocol.json"


def test_public_admission_needs_no_private_payload_and_includes_all_pilots(tmp_path):
    protocol, path = fixture(tmp_path)
    result = load_source_metadata(protocol, path)
    assert result["status"] == "passed" and result["private_payloads_read"] is False
    assert len(result["by_id"]) == 640
    assert all(len(c["pilot_ids"]) == len(c["final_ids"]) == 64 for c in result["datasets"].values())
    assert len(list(tmp_path.iterdir())) == 3  # Only public JSON manifests exist.


@pytest.mark.parametrize("damage", ["prior_nonwarmup_pilot", "wrong_final_order"])
def test_self_consistent_block_hash_cannot_hide_wrong_heldout_membership(tmp_path, damage):
    protocol, path = fixture(tmp_path)
    block = protocol["datasets"][0]["blocks"][0]
    if damage == "prior_nonwarmup_pilot":
        block["request_ids"][0] = "wildchat-10"
    else:
        block["request_ids"][:2] = block["request_ids"][:2][::-1]
    block["request_ids_sha256"] = digest(block["request_ids"])
    with pytest.raises(ValueError, match="reserved final membership/order"):
        load_source_metadata(protocol, path)


def test_missing_or_changed_source_manifest_never_bypasses_gate(tmp_path):
    protocol, path = fixture(tmp_path)
    (tmp_path/"suite.json").write_text("{}")
    with pytest.raises(ValueError, match="file hash differs"):
        load_source_metadata(protocol, path)
    (tmp_path/"suite.json").unlink()
    with pytest.raises(FileNotFoundError):
        load_source_metadata(protocol, path)


def test_prior_admission_must_bind_the_same_suite_even_if_its_file_hash_is_updated(tmp_path):
    protocol, path = fixture(tmp_path)
    value = json.loads((tmp_path/"admission.json").read_text())
    value["suite_manifest_file_sha256"] = "f"*64
    protocol["source_admission"]["suite_admission_sha256"] = sealed_write(tmp_path/"admission.json", value)
    with pytest.raises(ValueError, match="different suite"):
        load_source_metadata(protocol, path)


def test_private_row_preserves_full_unicode_input_and_distinct_hash_encodings():
    row = request("sharegpt", 64)
    public = {k: v for k, v in row.items() if k != "messages"}
    result = verify_private_request(row, public)
    assert result["status"] == "passed" and row["input_tokens"] > 512
    assert result["client_input_sha256"] != result["messages_sha256"]
    assert result["tokenization_recomputed"] is False


@pytest.mark.parametrize("damage", ["message", "group", "token_count"])
def test_private_content_or_metadata_cannot_depart_from_public_admission(damage):
    row = request("sharegpt", 64)
    public = copy.deepcopy({k: v for k, v in row.items() if k != "messages"})
    if damage == "message": row["messages"][0]["content"] += " changed"
    elif damage == "group": row["group_id"] = "another-conversation"
    else: row["input_tokens"] += 1
    with pytest.raises(ValueError):
        verify_private_request(row, public)

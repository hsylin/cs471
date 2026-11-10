"""Bind the three-arm comparison to previously admitted held-out identities.

The metadata loader reads public manifests only. The row verifier is called
later, after custom-pilot admission, when the controller opens private inputs.
Neither helper selects new data, changes splits, or retokenizes held-out text.
"""
from __future__ import annotations

import json
from pathlib import Path

from full_serving.data import MODEL_REVISION, digest
from full_serving.verify_suite import validate_request

DATASETS = ("wildchat", "sharegpt", "instructcoder", "numinamath", "longbench")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _public(protocol, protocol_path, stem, seal_key=None):
    declaration = protocol["source_admission"]
    value = declaration.get(stem + "_file")
    require(isinstance(value, str) and bool(value), f"missing public {stem} path")
    path = Path(value)
    path = path if path.is_absolute() else Path(protocol_path).resolve().parent / path
    raw = path.read_bytes()
    require(digest(raw) == declaration.get(stem + "_sha256"), f"public {stem} file hash differs")
    document = json.loads(raw)
    if seal_key is not None:
        require(document.get(seal_key) == digest({k: v for k, v in document.items() if k != seal_key}),
                f"public {stem} seal differs")
    return document, digest(raw)


def load_source_metadata(protocol, protocol_path):
    """Return public ``by_id`` rows after verifying all admitted split identities."""
    if protocol.get("schema") == "resident_compare_two_workloads_v1":
        from full_serving.custom_compare import validate_protocol
        from full_serving.output_lengths import LENGTH_FIELDS
        parent, _ = _public(protocol, protocol_path, "parent_protocol")
        validate_protocol(parent)
        require(parent.get("schema") == "resident_compare_protocol_v1", "amendment must retain the original full protocol")
        for key, value in parent["source_admission"].items():
            if key.endswith("_sha256"):
                require(protocol["source_admission"].get(key) == value, "amendment changed source admission")
        for planned in protocol["datasets"]:
            original = next(d for d in parent["datasets"] if d["dataset"] == planned["dataset"])
            for key in original:
                if key not in ("blocks", "arm_position_counts"):
                    actual_value = planned.get(key)
                    if key == "warmup" and isinstance(actual_value, dict):
                        actual_value = {k: v for k, v in actual_value.items() if k not in LENGTH_FIELDS}
                    require(actual_value == original[key], "amendment changed source, warmup or arrival-rate binding")
            require(len(planned["blocks"]) == 1, "amendment requires one group")
            expected, actual = dict(original["blocks"][0]), dict(planned["blocks"][0])
            actual = {k: v for k, v in actual.items() if k not in LENGTH_FIELDS}
            expected.pop("arm_order"); actual.pop("arm_order")
            require(actual == expected, "amendment must use the first reserved group without reselection")
        # Rebase the parent's original source references to the archived paths.
        # The public source digests above remain unchanged, and all640 original
        # pilot/final identities and duplicate checks are still reconstructed.
        parent["source_admission"] = {k: protocol["source_admission"][k] for k in parent["source_admission"]}
        return load_source_metadata(parent, protocol_path)
    suite, suite_hash = _public(protocol, protocol_path, "suite_manifest", "manifest_sha256")
    wildchat, wild_hash = _public(protocol, protocol_path, "wildchat_manifest", "public_manifest_sha256")
    admission, admission_hash = _public(protocol, protocol_path, "suite_admission")
    require(suite.get("schema") == "full_serving_separate_suite_v1"
            and suite.get("source_model_revision") == MODEL_REVISION
            and suite.get("selection_status") == "pilot_ready_final_candidates_not_frozen",
            "unrecognized original suite/model/split admission")
    require(wildchat.get("schema") == "full_serving_requests_v1"
            and wildchat.get("source_revision") == MODEL_REVISION, "unrecognized WildChat parent")
    require(admission.get("schema") == "full_serving_suite_admission_v1" and admission.get("status") == "passed"
            and admission.get("suite_manifest_file_sha256") == suite_hash
            and admission.get("suite_manifest_sha256") == suite["manifest_sha256"]
            and admission.get("selected_requests") == 640 and admission.get("pilot_requests") == 320
            and admission.get("reserved_final_candidates") == 320
            and admission.get("recomputed_tokenization") is True
            and admission.get("tokenizer", {}).get("model_revision") == MODEL_REVISION,
            "prior source/tokenization admission is incomplete or bound to a different suite")
    require([c["dataset"] for c in suite["cohorts"]] == list(DATASETS)
            and [c["dataset"] for c in protocol["datasets"]] == list(DATASETS), "source cohort set/order differs")
    require(suite["cohorts"][0]["parent_manifest_sha256"] == wild_hash
            and suite["wildchat_admission"]["manifest_file_sha256"] == wild_hash,
            "WildChat parent/admission link differs")
    require(len(wildchat["requests"]) == 448
            and len({r["id"] for r in wildchat["requests"]}) == 448,
            "original WildChat admission does not cover all 448 requests")

    by_id, datasets, global_pilots, global_finals = {}, {}, set(), set()
    final_message_hashes, pilot_message_hashes = set(), set()
    for planned, cohort in zip(protocol["datasets"], suite["cohorts"]):
        name = cohort["dataset"]
        rows = ([r for r in wildchat["requests"] if r["split"] == "pilot" or r["content_block"] == "final-00"]
                if name == "wildchat" else cohort["requests"])
        group_key = "conversation_id" if name == "wildchat" else "group_id"
        require(len(rows) == 128 and len({r["id"] for r in rows}) == 128
                and all(r.get("dataset") == name for r in rows), f"{name}: invalid admitted cohort")
        groups = [str(r.get(group_key)) for r in rows]
        require("None" not in groups and len(set(groups)) == 128, f"{name}: source group overlaps pilot/final or requests")
        pilots = [r for r in rows if r["split"] == "pilot"]
        finals = [r for r in rows if r["split"] == "final"]
        require(len(pilots) == len(finals) == 64, f"{name}: expected all 64 pilot and 64 final identities")
        pilot_ids, final_ids = [r["id"] for r in pilots], [r["id"] for r in finals]
        require(not set(pilot_ids).intersection(final_ids), f"{name}: pilot/final ID overlap")
        require(planned.get("reserved_source_final_file_sha256") == cohort["private_files"]["final.jsonl"]
                and planned.get("source_public_manifest_sha256") == (wild_hash if name == "wildchat" else suite_hash),
                f"{name}: reserved final/source manifest binding differs")
        warmup = planned["warmup"]
        require(warmup.get("split") == "pilot" and warmup.get("request_count") == 8
                and warmup.get("request_ids") == cohort["smoke_request_ids"]
                and len(set(warmup["request_ids"])) == 8 and set(warmup["request_ids"]) <= set(pilot_ids)
                and warmup.get("requests_sha256") == cohort["private_files"]["smoke.jsonl"],
                f"{name}: warmup differs from admitted pilot subset")
        require(len(planned["blocks"]) == 4, f"{name}: expected four final blocks")
        for index, block in enumerate(planned["blocks"]):
            expected = finals[index*16:(index+1)*16]
            ids = [r["id"] for r in expected]
            require(block.get("split") == "final" and block.get("request_count") == 16
                    and block.get("request_ids") == ids
                    and block.get("request_ids_sha256") == digest(ids),
                    f"{name}: final block differs from reserved final membership/order (including prior pilot exclusions)")
            if "source_final_index_range_half_open" in block:
                require(block["source_final_index_range_half_open"] == [index*16, (index+1)*16],
                        f"{name}: final index range differs")
            if "source_group_ids_sha256" in block:
                require(block["source_group_ids_sha256"] == digest([r[group_key] for r in expected]),
                        f"{name}: final source group binding differs")
        for row in rows:
            require(row["id"] not in by_id, "source identity overlaps datasets")
            by_id[row["id"]] = dict(row)
        global_pilots.update(pilot_ids)
        global_finals.update(final_ids)
        pilot_message_hashes.update(r["normalized_messages_sha256"] for r in pilots)
        final_message_hashes.update(r["normalized_messages_sha256"] for r in finals)
        datasets[name] = {"pilot_ids": pilot_ids, "final_ids": final_ids, "warmup_ids": warmup["request_ids"]}
    require(not global_pilots.intersection(global_finals)
            and not pilot_message_hashes.intersection(final_message_hashes), "cross-source pilot/final identity or normalized content overlaps")
    require(len(final_message_hashes) == len(global_finals) == len(global_pilots) == 320,
            "final content or source identity is duplicated")
    return {"schema": "resident_compare_source_metadata_v1", "status": "passed", "by_id": by_id,
            "datasets": datasets, "manifests": {"suite_manifest_sha256": suite_hash,
                "wildchat_manifest_sha256": wild_hash, "suite_admission_sha256": admission_hash},
            "private_payloads_read": False,
            "tokenization_scope": "Bound to the prior all-640 retokenized admission; this function does not retokenize inputs"}


def verify_private_request(row, metadata):
    """Check a subsequently opened private row against one admitted public row."""
    require(isinstance(row, dict) and isinstance(metadata, dict) and row.get("id") == metadata.get("id"),
            "private/public request identity differs")
    require(all(key in row and row[key] == value for key, value in metadata.items()),
            "private metadata/group/tokenization differs from admitted public source")
    validate_request(row, metadata["dataset"], 32768, 2048)
    return {"status": "passed", "id": row["id"], "messages_sha256": row["messages_sha256"],
            "client_input_sha256": digest(json.dumps(row["messages"], ensure_ascii=False, sort_keys=True,
                                                     separators=(",", ":")).encode()),
            "tokenization_recomputed": False}

"""Post-selection integrity admission for the five separate serving cohorts.

This does not select or change requests, freeze a final protocol, execute a
model, or establish performance. Candidate bytes must agree with the suite and
their separate upstream preparation declaration before any GPU run is admitted.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from full_serving.data import digest, file_digest, MODEL_REVISION, normalized, verify_pinned_file
from full_serving.prepare_suite import PUBLIC_FIELDS
from full_serving.verify_data import verify_bundle

DATASETS = ("wildchat", "sharegpt", "instructcoder", "numinamath", "longbench")
DERIVED_FIELDS = {"group_id", "input_tokens", "messages_sha256", "normalized_messages_sha256",
                  "first_user_normalized_sha256", "prompt_token_ids_sha256", "max_tokens", "split", "content_block"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def read_private(path):
    raw = Path(path).read_bytes()
    require(raw.endswith(b"\n"), f"Missing JSONL final newline: {Path(path).name}")
    lines = raw.split(b"\n")[:-1]
    require(all(lines), f"Blank private JSONL record: {Path(path).name}")
    rows = [json.loads(line) for line in lines]
    require(all(isinstance(row, dict) for row in rows), "Private request must be an object")
    canonical = "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows).encode()
    require(raw == canonical, f"Noncanonical private JSONL: {Path(path).name}")
    return rows, {"sha256": digest(raw), "bytes": len(raw), "requests": len(rows)}


def source_declaration(dataset, cohort, repo_root):
    """Resolve the source-specific candidate declaration, recording every link."""
    root = Path(repo_root).resolve()
    path = (root / cohort["source_manifest_path"]).resolve()
    require(path.is_relative_to(root), "Source manifest escapes repository")
    actual = file_digest(path)
    require(actual == cohort["source_manifest_sha256"], f"Source manifest changed: {dataset}")
    source = json.loads(path.read_text())
    require(source.get("dataset") == dataset, "Source manifest dataset mismatch")
    report = {"source_manifest_path": str(path.relative_to(root)), "source_manifest_sha256": actual}
    if dataset in ("instructcoder", "numinamath"):
        expected = source["private_candidates"]["sha256"]
        report["declaration_field"] = "private_candidates.sha256"
    elif dataset == "longbench":
        expected = source["candidate_payload_sha256"]
        report["declaration_field"] = "candidate_payload_sha256"
    elif dataset == "sharegpt":
        name = source["preparation_manifest"]
        require(isinstance(name, str) and Path(name).name == name, "Unsafe ShareGPT preparation path")
        preparation_path = path.parent / name
        preparation_sha = file_digest(preparation_path)
        # v1 source metadata names this file but does not seal its digest.
        # The independently sealed suite candidate digest supplies the binding.
        declared_preparation_sha = source.get("preparation_manifest_sha256")
        if declared_preparation_sha is not None:
            require(preparation_sha == declared_preparation_sha, "ShareGPT preparation manifest changed")
        preparation = json.loads(preparation_path.read_text())
        expected = preparation["candidate_file_sha256"]
        report.update({"declaration_field": "named preparation_manifest -> candidate_file_sha256",
                       "preparation_manifest_path": str(preparation_path.relative_to(root)),
                       "preparation_manifest_sha256": preparation_sha,
                       "preparation_digest_previously_declared": declared_preparation_sha is not None,
                       "preparation_digest_scope": "Recorded during this admission; candidate digest must independently match sealed suite metadata"})
    else:
        raise ValueError("Unsupported candidate source")
    require(valid_hash(expected), "Malformed declared candidate SHA256")
    require(expected == cohort["candidate_file_sha256"], f"Candidate declaration differs from suite: {dataset}")
    report["declared_candidate_sha256"] = expected
    return expected, report


def verify_candidate_file(path, dataset, expected_sha, selected):
    """Hash complete raw JSONL and bind selected records to authentic messages."""
    hasher = hashlib.sha256()
    selected_by_id = {row["id"]: row for row in selected}
    require(len(selected_by_id) == len(selected), "Duplicate selected request identity")
    found, all_ids = set(), set()
    byte_count = 0
    with Path(path).open("rb") as source:
        for line in source:
            hasher.update(line)
            byte_count += len(line)
            require(line.endswith(b"\n") and line.strip(), "Invalid candidate JSONL boundary")
            row = json.loads(line)
            identity = row["id"]
            require(row.get("dataset") == dataset, "Mixed candidate dataset")
            require(identity not in all_ids, "Duplicate candidate identity")
            all_ids.add(identity)
            if identity not in selected_by_id:
                continue
            chosen = selected_by_id[identity]
            require(chosen["messages"] == row["messages"], f"Selected original messages changed: {dataset}")
            original = {key: value for key, value in row.items() if key not in DERIVED_FIELDS}
            retained = {key: value for key, value in chosen.items() if key not in DERIVED_FIELDS}
            require(original == retained, f"Selected original metadata changed: {dataset}")
            expected_group = str(row.get("group_id", row.get("source_id", identity)))
            require(chosen["group_id"] == expected_group, "Selected source group changed")
            found.add(identity)
    actual = hasher.hexdigest()
    require(actual == expected_sha, f"Candidate file SHA256 mismatch: {dataset}")
    require(found == set(selected_by_id), f"Selected request absent from candidate source: {dataset}")
    return {"candidate_file_basename": Path(path).name, "candidate_file_sha256": actual,
            "candidate_file_bytes": byte_count, "candidate_rows": len(all_ids),
            "selected_source_rows_verified": len(found)}


def validate_request(row, dataset, max_context, max_output, tokenizer=None):
    require(row.get("dataset") == dataset, "Mixed private cohort")
    messages = row.get("messages")
    require(isinstance(messages, list) and messages and messages[-1].get("role") == "user",
            "Request must end in an authentic user turn")
    for message in messages:
        require(isinstance(message, dict) and set(message) == {"role", "content"}, "Unexpected message fields")
        require(message["role"] in {"system", "user", "assistant"}
                and isinstance(message["content"], str) and message["content"].strip(), "Invalid message content")
    content_key = digest([{**message, "content": normalized(message["content"])} for message in messages])
    first_key = digest(normalized(next(message["content"] for message in messages if message["role"] == "user")).encode())
    require(row.get("messages_sha256") == digest(messages), "Message content hash mismatch")
    require(row.get("normalized_messages_sha256") == content_key, "Normalized message hash mismatch")
    require(row.get("first_user_normalized_sha256") == first_key, "First-user content hash mismatch")
    require(type(row.get("input_tokens")) is int and row["input_tokens"] > 0
            and row.get("max_tokens") == max_output and row["input_tokens"] + max_output <= max_context,
            "Invalid input/output context allowance")
    require(valid_hash(row.get("prompt_token_ids_sha256")), "Invalid prompt token digest")
    require(row.get("split") in ("pilot", "final"), "Invalid cohort split")
    if tokenizer is not None:
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                                               truncation=False, add_generation_prompt=True, enable_thinking=False)
        require(len(tokens) == row["input_tokens"] and digest(tokens) == row["prompt_token_ids_sha256"],
                "Recomputed prompt token count/digest mismatch")
    return content_key, first_key


def verify_suite(private_dir, manifest_path, candidate_paths, *, wildchat_private,
                 wildchat_manifest, repo_root=None, tokenizer=None):
    root = Path(repo_root or Path(__file__).resolve().parents[1])
    manifest_path, private_dir = Path(manifest_path), Path(private_dir)
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("schema") == "full_serving_separate_suite_v1", "Unexpected suite schema")
    seal = digest({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    require(manifest.get("manifest_sha256") == seal, "Suite manifest seal mismatch")
    require(manifest.get("source_model_revision") == MODEL_REVISION, "Wrong pinned model/tokenizer")
    require(manifest.get("selection_status") == "pilot_ready_final_candidates_not_frozen", "Unexpected final-freeze status")
    require(manifest.get("datasets_kept_separate") is True and manifest.get("pooled_speedup_claim") is False,
            "Dataset separation contract changed")
    require(manifest.get("selection_order") == list(DATASETS)
            and [cohort["dataset"] for cohort in manifest["cohorts"]] == list(DATASETS), "Unexpected cohort set/order")
    require(set(candidate_paths) == set(DATASETS[1:]), "All four candidate files are required")
    formatting, generation = manifest["formatting"], manifest["generation"]
    require(formatting.get("truncation") is False and formatting.get("enable_thinking") is False
            and formatting.get("add_generation_prompt") is True and formatting.get("max_model_len") == 32768,
            "Prompt-format contract changed")
    require(generation == {"max_tokens": 2048, "temperature": 0, "ignore_eos": False}, "Generation contract changed")
    for relative, expected in manifest["preparation_sources"].items():
        path = (root / relative).resolve()
        require(path.is_relative_to(root.resolve()) and file_digest(path) == expected, "Preparation source changed")
    wild_admission = verify_bundle(wildchat_private, wildchat_manifest)
    require(file_digest(wildchat_manifest) == manifest["cohorts"][0]["parent_manifest_sha256"],
            "WildChat parent manifest changed")
    require(wild_admission["manifest_file_sha256"] == manifest["wildchat_admission"]["manifest_file_sha256"],
            "WildChat prior admission does not match")
    parent_rows = [json.loads(line) for line in (Path(wildchat_private) / "requests.jsonl").read_text().split("\n") if line]
    require(len(parent_rows) == 448, "Expected all448 prior WildChat requests")
    expected_wild = [row for row in parent_rows if row["split"] == "pilot" or row["content_block"] == "final-00"]
    seen_content = {row["normalized_messages_sha256"] for row in parent_rows}
    seen_first = {row["first_user_normalized_sha256"] for row in parent_rows}
    all_ids, reports = set(), []
    for cohort in manifest["cohorts"]:
        dataset = cohort["dataset"]
        require(set(cohort["private_files"]) == {"requests.jsonl", "pilot.jsonl", "final.jsonl", "smoke.jsonl"},
                "Unexpected private file contract")
        loaded, verified_files = {}, {}
        for name, expected in cohort["private_files"].items():
            rows, info = read_private(private_dir / dataset / name)
            require(info["sha256"] == expected, f"Private file SHA256 mismatch: {dataset}/{name}")
            loaded[name], verified_files[name] = rows, info
        rows = loaded["requests.jsonl"]
        require(len(rows) == 128, f"Expected128 selected requests per dataset: {dataset}")
        require(len({row["id"] for row in rows}) == 128 and not all_ids.intersection(row["id"] for row in rows),
                "Duplicate selected identity within/across datasets")
        all_ids.update(row["id"] for row in rows)
        groups = [str(row.get("group_id", row.get("conversation_id", row.get("source_id")))) for row in rows]
        require(len(set(groups)) == 128 and "None" not in groups, "Source group crosses selected requests/splits")
        for row in rows:
            content, first = validate_request(row, dataset, formatting["max_model_len"], generation["max_tokens"], tokenizer)
            if dataset != "wildchat":
                require(content not in seen_content and first not in seen_first,
                        "Selected content duplicates another source or any prior WildChat row")
                seen_content.add(content)
                seen_first.add(first)
        pilots = [row for row in rows if row["split"] == "pilot"]
        finals = [row for row in rows if row["split"] == "final"]
        require(len(pilots) == len(finals) == 64, "Pilot/final count mismatch")
        require(loaded["pilot.jsonl"] == pilots and loaded["final.jsonl"] == finals, "Ordered pilot/final subset mismatch")
        smoke_ids = cohort["smoke_request_ids"]
        require(len(smoke_ids) == len(set(smoke_ids)) == 8 and set(smoke_ids) <= {row["id"] for row in pilots},
                "Smoke must contain eight distinct pilot requests")
        by_id = {row["id"]: row for row in rows}
        require(loaded["smoke.jsonl"] == [by_id[identity] for identity in smoke_ids], "Ordered smoke subset mismatch")
        report = {"dataset": dataset, "requests": 128, "pilot": 64, "reserved_final_candidates": 64,
                  "unique_groups": len(set(groups)), "verified_private_files": verified_files,
                  "input_tokens_min": min(row["input_tokens"] for row in rows),
                  "input_tokens_max": max(row["input_tokens"] for row in rows)}
        if dataset == "wildchat":
            require(rows == expected_wild, "WildChat membership/content changed from parent pilot+final00")
            require(cohort["other_final_bundles"] == "reserved, not discarded or measured", "WildChat reservation status changed")
        else:
            public = [{key: value for key, value in row.items() if key in PUBLIC_FIELDS} for row in rows]
            require(all(all(isinstance(value, (str, int, bool, type(None))) for value in record.values()) for record in public),
                    "Public metadata must remain scalar")
            require(cohort["requests"] == public, "Selected public/private metadata mismatch")
            require(cohort["selected_count"] == 128 and cohort["pilot_count"] == 64
                    and cohort["final_candidate_count"] == 64, "Public allocation differs")
            expected, provenance = source_declaration(dataset, cohort, root)
            candidate_audit = verify_candidate_file(candidate_paths[dataset], dataset, expected, rows)
            require(candidate_audit["candidate_rows"] == cohort["streaming_frame"]["input_candidate_rows"],
                    "Candidate source count differs from selection")
            report["source_binding"] = {**provenance, **candidate_audit}
        reports.append(report)
    return {"schema": "full_serving_suite_admission_v1", "status": "passed", "suite_manifest_sha256": seal,
            "suite_manifest_file_sha256": file_digest(manifest_path), "selected_requests": len(all_ids),
            "pilot_requests": 320, "reserved_final_candidates": 320, "additional_wildchat_finals_preserved": 320,
            "wildchat_prior_requests_checked": 448, "cohorts": reports,
            "recomputed_tokenization": tokenizer is not None,
            "tokenization_scope": "All640 selected prompts retokenized" if tokenizer is not None
                else "Token counts and digests match sealed public metadata; tokenization not recomputed",
            "final_protocol_frozen": False, "verifier_sha256": file_digest(__file__),
            "scope": "CPU data/source admission only; no generated outputs, GPU correctness, EPLB commit or performance claim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidates", action="append", required=True, help="DATASET=private_candidate_jsonl")
    parser.add_argument("--wildchat-private", type=Path, required=True)
    parser.add_argument("--wildchat-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = {}
    for item in args.candidates:
        name, path = item.split("=", 1)
        require(name not in candidates, "Duplicate candidate argument")
        candidates[name] = Path(path)
    tokenizer, tokenizer_record = None, None
    if args.tokenizer_dir:
        files = {name: verify_pinned_file(args.tokenizer_dir / name, name)
                 for name in ("tokenizer.json", "tokenizer_config.json")}
        import transformers
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True, trust_remote_code=False)
        tokenizer_record = {"model_revision": MODEL_REVISION, "pinned_files": files,
                            "transformers_version": transformers.__version__}
    report = verify_suite(args.private_dir, args.manifest, candidates,
                          wildchat_private=args.wildchat_private, wildchat_manifest=args.wildchat_manifest,
                          tokenizer=tokenizer)
    report["tokenizer"] = tokenizer_record
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"status": report["status"], "selected_requests": report["selected_requests"],
                      "recomputed_tokenization": report["recomputed_tokenization"], "output": str(args.output)}))


if __name__ == "__main__":
    main()

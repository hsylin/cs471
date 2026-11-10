"""CPU-only admission checks for the private full-model request bundle."""
import argparse
import json
from pathlib import Path

from full_serving.data import (DATASET_REVISION, MODEL_REVISION, conversation_prefix,
                               digest, file_digest, normalized, text_digest,
                               verify_pinned_file)
from full_serving.prepare_data import serialize_requests


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read_jsonl(path):
    # JSONL is delimited by ASCII LF, not Unicode paragraph/line separators
    # which may legitimately occur inside a decoded conversation string.
    raw = Path(path).read_bytes()
    require(raw.endswith(b"\n"), f"Missing final JSONL newline: {Path(path).name}")
    lines = raw.decode("utf-8").split("\n")[:-1]
    require(all(lines), f"Blank JSONL record: {Path(path).name}")
    rows = [json.loads(line) for line in lines]
    require(all(isinstance(row, dict) for row in rows), "Request records must be JSON objects")
    return raw, rows


def verify_bundle(private_dir, manifest_path, *, tokenizer=None):
    """Check bytes, request contracts and every expected ordered subset.

    ``tokenizer`` is optional so admission can run without Transformers/PyTorch.
    Without it, token counts and token-ID hashes are checked against the public
    manifest, but are explicitly *not* claimed to have been recomputed.
    """
    private_dir, manifest_path = Path(private_dir), Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("schema") == "full_serving_requests_v1", "Unexpected request manifest schema")
    seal = digest({k: v for k, v in manifest.items() if k != "public_manifest_sha256"})
    require(manifest.get("public_manifest_sha256") == seal, "Public manifest seal mismatch")
    require(manifest.get("dataset_revision") == DATASET_REVISION, "Unpinned dataset revision")
    require(manifest.get("source_revision") == MODEL_REVISION, "Unpinned model/tokenizer revision")
    formatting, generation = manifest["formatting"], manifest["generation"]
    require(formatting.get("enable_thinking") is False and formatting.get("truncation") is False
            and formatting.get("add_generation_prompt") is True, "Unexpected prompt-format contract")
    require(generation.get("temperature") == 0 and generation.get("ignore_eos") is False,
            "Unexpected generation contract")
    raw, rows = read_jsonl(private_dir / "requests.jsonl")
    require(digest(raw) == manifest.get("private_payload_sha256"), "Complete private payload hash mismatch")
    require(raw == serialize_requests(rows).encode(), "Complete payload is not canonical one-line-per-request JSONL")
    allocation = manifest["allocation"]
    expected_count = allocation["pilot_requests"] + allocation["final_bundles"] * allocation["requests_per_final_bundle"]
    require(len(rows) == len(manifest["requests"]) == manifest["counts"]["selected"] == expected_count,
            "Request count differs from public allocation")
    require(len({r["id"] for r in rows}) == len(rows), "Duplicate request identity")
    require(len({r["conversation_id"] for r in rows}) == len(rows), "Conversation identity crosses splits")
    require(len({r["first_user_normalized_sha256"] for r in rows}) == len(rows), "Duplicate first-user content")
    max_context, max_output = formatting["max_model_len"], generation["max_tokens"]
    for row, public in zip(rows, manifest["requests"]):
        identity = row["id"]
        require({k: v for k, v in row.items() if k != "messages"} == public,
                f"Private/public request metadata mismatch: {identity}")
        messages = row["messages"]
        prefix, last_user = conversation_prefix({"conversation": messages})
        require(prefix == messages and last_user == len(messages)-1, f"Prompt does not end in user turn: {identity}")
        require(all(set(m) == {"role", "content"} for m in messages), f"Unexpected message fields: {identity}")
        require(digest(messages) == row["messages_sha256"], f"Message content hash mismatch: {identity}")
        require(digest([{"role": m["role"], "content": normalized(m["content"])} for m in messages])
                == row["normalized_messages_sha256"], f"Normalized message hash mismatch: {identity}")
        first_user = next(m["content"] for m in messages if m["role"] == "user")
        require(text_digest(first_user) == row["first_user_normalized_sha256"], f"First-user hash mismatch: {identity}")
        require(row["last_user_turn_index"] == last_user
                and row["recorded_assistant_history_turns"] == sum(m["role"] == "assistant" for m in messages),
                f"Conversation history metadata mismatch: {identity}")
        require(type(row["input_tokens"]) is int and row["input_tokens"] > 0
                and row["max_tokens"] == max_output and row["input_tokens"] + max_output <= max_context,
                f"Invalid prompt/output context allowance: {identity}")
        require(isinstance(row["prompt_token_ids_sha256"], str) and len(row["prompt_token_ids_sha256"]) == 64,
                f"Missing prompt token digest: {identity}")
        if tokenizer is not None:
            tokens = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                                                   truncation=False, add_generation_prompt=True,
                                                   enable_thinking=False)
            require(len(tokens) == row["input_tokens"] and digest(tokens) == row["prompt_token_ids_sha256"],
                    f"Retokenized prompt differs: {identity}")
    pilot = [r for r in rows if r["split"] == "pilot"]
    require(len(pilot) == allocation["pilot_requests"], "Pilot request count mismatch")
    require(all(r["content_block"] == "pilot-00" for r in pilot), "Unexpected pilot block")
    blocks = {f"final-{i:02d}" for i in range(allocation["final_bundles"])}
    require({r["content_block"] for r in rows if r["split"] == "final"} == blocks,
            "Final block identity mismatch")
    require(all(r["split"] in {"pilot", "final"} for r in rows), "Unknown content split")
    smoke_ids = manifest["smoke_request_ids"]
    require(len(smoke_ids) == len(set(smoke_ids)) == allocation["smoke_requests"], "Smoke identity/count mismatch")
    require(set(smoke_ids) <= {r["id"] for r in pilot}, "Smoke request outside pilot")
    by_id = {r["id"]: r for r in rows}
    expected_files = {"requests.jsonl": rows, "pilot.jsonl": pilot, "pilot-00.jsonl": pilot,
                      "smoke.jsonl": [by_id[identity] for identity in smoke_ids]}
    for block in sorted(blocks):
        selected = [r for r in rows if r["content_block"] == block]
        require(len(selected) == allocation["requests_per_final_bundle"], f"Final bundle size mismatch: {block}")
        expected_files[block + ".jsonl"] = selected
    verified = {}
    for name, expected_rows in sorted(expected_files.items()):
        subset_raw, subset_rows = read_jsonl(private_dir / name)
        expected_raw = serialize_requests(expected_rows).encode()
        require(subset_rows == expected_rows, f"Ordered subset identities/content mismatch: {name}")
        require(subset_raw == expected_raw, f"Subset exact-byte hash mismatch: {name}")
        verified[name] = {"requests": len(subset_rows), "bytes": len(subset_raw),
                          "sha256": digest(subset_raw), "ordered_ids_sha256": digest([r["id"] for r in subset_rows])}
    return {"schema": "full_serving_data_validation_v1", "status": "passed",
            "public_manifest_sha256": seal, "manifest_file_sha256": file_digest(manifest_path),
            "private_payload_sha256": digest(raw), "requests": len(rows), "verified_files": verified,
            "selection_status": manifest["selection_status"], "final_protocol_frozen": manifest["final_protocol_frozen"],
            "recomputed_tokenization": tokenizer is not None,
            "tokenization_check": "Recomputed every prompt token count and token-ID hash" if tokenizer is not None
                else "Token counts/token-ID hashes match public manifest; not retokenized in this invocation",
            "unverified_extra_jsonl_files": sorted(p.name for p in private_dir.glob("*.jsonl") if p.name not in expected_files),
            "scope": "CPU data admission only; no model execution, numerical-output or GPU-performance claim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retokenize", action="store_true")
    parser.add_argument("--tokenizer-dir", type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "Validation output already exists; use a fresh path")
    tokenizer, environment = None, None
    if args.retokenize:
        require(args.tokenizer_dir is not None, "--retokenize requires --tokenizer-dir")
        for name in ("tokenizer.json", "tokenizer_config.json"):
            verify_pinned_file(args.tokenizer_dir / name, name)
        import transformers
        import tokenizers
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_dir), local_files_only=True,
                                                   trust_remote_code=False)
        environment = {"transformers_version": transformers.__version__, "tokenizers_version": tokenizers.__version__,
                       "chat_template_sha256": digest(tokenizer.chat_template.encode())}
    elif args.tokenizer_dir:
        raise ValueError("--tokenizer-dir requires --retokenize")
    report = verify_bundle(args.private_dir, args.manifest, tokenizer=tokenizer)
    if environment is not None:
        report["tokenizer_environment"] = environment
    report["verifier_source_sha256"] = file_digest(__file__)
    report["report_sha256"] = digest(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "requests": report["requests"],
                      "retokenized": report["recomputed_tokenization"], "output": str(args.output),
                      "report_sha256": report["report_sha256"]}), flush=True)


if __name__ == "__main__":
    main()

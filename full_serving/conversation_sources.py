"""Pinned ShareGPT released segments and BurstGPT metadata-only diagnostics.

Raw files and candidate text remain outside Git. No original response is used
as a generation target or a forced output length. BurstGPT does not contain
prompt text and this module never constructs a text request from its lengths.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re


SHAREGPT_REVISION = "192ab2185289094fc556ec8ce5ce1e8e587154ca"
SHAREGPT_SHA256 = "35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4"
BURSTGPT_REVISION = "7eb2c4f8350f8a6985272386f5c14af1f678b299"
BURSTGPT_SHA256 = "4bb3783693d0a435686fbfc885615d2349bd067239079fa4b749f2e679e12122"


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_json_array(path, chunk_size=1 << 20):
    """Read the released JSON array with bounded buffering (not all 673 MB)."""
    decoder = json.JSONDecoder()
    with Path(path).open(encoding="utf-8") as handle:
        buffer, position, ended = "", 0, False

        def refill():
            nonlocal buffer, position, ended
            chunk = handle.read(chunk_size)
            buffer = buffer[position:] + chunk
            position = 0
            ended = not chunk

        def next_character():
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position] in " \r\n\t":
                    position += 1
                if position < len(buffer) or ended:
                    return buffer[position:position + 1]
                refill()

        refill()
        if next_character() != "[":
            raise ValueError("Expected one JSON array")
        position += 1
        first = True
        while True:
            char = next_character()
            if char == "]":
                position += 1
                if next_character():
                    raise ValueError("Trailing data after array")
                return
            if not first:
                if char != ",":
                    raise ValueError("Missing array separator")
                position += 1
                if next_character() == "]":
                    raise ValueError("Trailing array comma")
            while True:
                try:
                    row, end = decoder.raw_decode(buffer, position)
                    position = end
                    break
                except json.JSONDecodeError:
                    if ended:
                        raise
                    refill()
                    next_character()
            yield row
            first = False


def sharegpt_candidate(row, index):
    source_id = row.get("id")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("missing_source_id")
    conversation = row.get("conversations")
    if not isinstance(conversation, list) or not conversation:
        raise ValueError("missing_conversation")
    roles = {"human": "user", "gpt": "assistant", "system": "system"}
    messages = []
    for entry in conversation:
        if not isinstance(entry, dict) or entry.get("from") not in roles:
            raise ValueError("unsupported_source_role")
        if not isinstance(entry.get("value"), str) or not entry["value"].strip():
            raise ValueError("empty_or_nontext_message")
        messages.append({"role": roles[entry["from"]], "content": entry["value"]})
    user_indices = [i for i, entry in enumerate(messages) if entry["role"] == "user"]
    if not user_indices:
        raise ValueError("no_user_message")
    last_user = user_indices[-1]
    prefix = messages[:last_user + 1]
    # This grouping rule must be checked against the pinned splitter and IDs;
    # unknown IDs stay whole rather than guessing a different grouping scheme.
    match = re.fullmatch(r"(.+)_([0-9]+)", source_id)
    group = match.group(1) if match else source_id
    return {"id": f"sharegpt:{source_id}:{index}", "dataset": "sharegpt",
            "source_id": source_id, "source_row_index": index,
            "source_split": "released_cleaned_segment_no_official_train_test_split",
            "group_id": f"sharegpt:{group}", "messages": prefix,
            "source_revision": SHAREGPT_REVISION,
            "source_file_sha256": SHAREGPT_SHA256,
            "released_segment_messages": len(messages), "input_messages": len(prefix),
            "excluded_target_messages": len(messages) - len(prefix),
            "released_segment_starts_with_assistant": prefix[0]["role"] == "assistant",
            "recorded_assistant_history_turns": sum(m["role"] == "assistant" for m in prefix),
            "transformation": "full released prefix through last user; final recorded assistant reply excluded; no text truncation"}


def prepare_sharegpt(source, output):
    actual = sha_file(source)
    if actual != SHAREGPT_SHA256:
        raise ValueError("ShareGPT source checksum mismatch")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    rejected, ids, groups = Counter(), Counter(), Counter()
    total, accepted, multi_turn, max_characters, suffix_ids, assistant_initial = 0, 0, 0, 0, 0, 0
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        for index, row in enumerate(iter_json_array(source)):
            total += 1
            if isinstance(row.get("id"), str):
                ids[row["id"]] += 1
                suffix_ids += bool(re.fullmatch(r"(.+)_([0-9]+)", row["id"]))
            try:
                candidate = sharegpt_candidate(row, index)
            except ValueError as exc:
                rejected[str(exc)] += 1
                continue
            groups[candidate["group_id"]] += 1
            accepted += 1
            multi_turn += candidate["recorded_assistant_history_turns"] > 0
            assistant_initial += candidate["released_segment_starts_with_assistant"]
            max_characters = max(max_characters, sum(len(m["content"]) for m in candidate["messages"]))
            handle.write(json.dumps(candidate, ensure_ascii=True) + "\n")
    return {"source_sha256": actual, "released_rows": total, "candidate_rows": accepted,
            "rejected": dict(rejected), "multi_turn_candidate_rows": multi_turn,
            "max_input_characters": max_characters, "ids_with_numeric_suffix": suffix_ids,
            "assistant_initial_segments_preserved": assistant_initial,
            "duplicate_release_id_count": sum(v - 1 for v in ids.values()),
            "underlying_groups": len(groups), "groups_with_multiple_segments": sum(v > 1 for v in groups.values()),
            "candidate_file_sha256": sha_file(output), "candidate_file_bytes": output.stat().st_size,
            "tokenization": "not performed; root cohort selection uses pinned model tokenizer",
            "grouping_rule": "strip final underscore plus numeric starting-message index; verified against pinned upstream splitter"}


def ns(timestamp):
    value = Decimal(timestamp) * 1_000_000_000
    if value != value.to_integral_value():
        raise ValueError("timestamp finer than integer nanoseconds")
    return int(value)


def prepare_burstgpt(source, count=24, seed="cs471-burstgpt-diagnostic-v1"):
    actual = sha_file(source)
    if actual != BURSTGPT_SHA256:
        raise ValueError("BurstGPT source checksum mismatch")
    with Path(source).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = sum(1 for _ in reader)
    if count < 1 or rows < count + 1:
        raise ValueError("invalid diagnostic row count")
    start = int(hashlib.sha256(seed.encode()).hexdigest(), 16) % (rows - count)
    chosen = []
    with Path(source).open(newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if start <= index <= start + count:
                chosen.append({"source_row_index": index, "raw": row})
            if index == start + count:
                break
    next_record = chosen.pop()
    timestamps = [ns(row["raw"]["Timestamp"]) for row in chosen]
    if timestamps != sorted(timestamps):
        raise ValueError("chosen contiguous source rows are not time ordered")
    offsets = [value - timestamps[0] for value in timestamps]
    next_offset = ns(next_record["raw"]["Timestamp"]) - timestamps[0]
    for row in chosen:
        raw = row["raw"]
        row.update(source_failed_zero_response=int(raw["Response tokens"]) == 0,
                   source_request_tokens=int(raw["Request tokens"]),
                   source_response_tokens=int(raw["Response tokens"]))
        if int(raw["Total tokens"]) != row["source_request_tokens"] + row["source_response_tokens"]:
            raise ValueError("source token total is inconsistent")
    return {"schema": "burstgpt_metadata_diagnostic_v1", "dataset": "burstgpt",
            "kind": "metadata_only_no_prompt_text", "source_release": "v2.0",
            "source_revision": BURSTGPT_REVISION, "source_asset_id": 341026710,
            "source_sha256": actual, "source_total_rows": rows, "source_columns": fields,
            "selection": f"one contiguous {count}-row block; SHA256 fixed seed modulo valid start rows; no model, length, failure, or performance filtering",
            "seed": seed, "source_start_row_index": start, "count": count,
            "source_start_timestamp_ns": timestamps[0], "original_offsets_ns": offsets,
            "original_span_ns": offsets[-1], "next_source_offset_ns": next_offset,
            "original_horizon_ns": max(next_offset, offsets[-1] + 1_000_000_000),
            "horizon_rule": "next source row offset, or one source-resolution second beyond final included timestamp if boundary ties",
            "source_failure_rows_retained": sum(row["source_failed_zero_response"] for row in chosen),
            "source_lengths_are_qwen_token_counts": False,
            "source_failures_are_injected_into_new_server": False,
            "rows": chosen,
            "limitation": "Published GPT-service token lengths are metadata; pairing with independently acquired text creates a hybrid replay, not recovered BurstGPT prompts or identical token lengths."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="dataset", required=True)
    share = subs.add_parser("sharegpt")
    share.add_argument("--source", type=Path, required=True)
    share.add_argument("--output-private", type=Path, required=True)
    share.add_argument("--manifest", type=Path, required=True)
    burst = subs.add_parser("burstgpt")
    burst.add_argument("--source", type=Path, required=True)
    burst.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_sharegpt(args.source, args.output_private) if args.dataset == "sharegpt" else prepare_burstgpt(args.source)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"dataset": args.dataset, "manifest": str(args.manifest),
                      "candidate_rows": result.get("candidate_rows"), "diagnostic_count": result.get("count")}))


if __name__ == "__main__":
    main()

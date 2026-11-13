"""Conversation-disjoint, full-message inputs for actual model serving.

No GPU, routing, or performance observations enter selection.  The main sample
is hash ordered; length is used only for rejecting an explicit context overflow
and for choosing a diagnostic smoke subset *within* the pilot.
"""
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import unicodedata

MODEL = "Qwen/Qwen3-30B-A3B"
MODEL_REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
DATASET_REVISION = "f66566ceaaeb619dd98ffb0f3bf3ce1f86775ac4"
SEED = 471901
MAX_CONTEXT = 32768
MAX_NEW_TOKENS = 2048
PINNED_FILES = {
    "wildchat-first-shard.parquet": (290754400, "d42159f2cc33b6c90aa5ca429aa625cff5ed7c72d6c66a64993957fa9444b48e"),
    "wildchat-license.md": (19947, "a7c7f6bdb20d261b8726c7594afb5832b36926e0cabfa116c41f58625bbc776c"),
    "tokenizer.json": (11422654, "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
    "tokenizer_config.json": (9732, "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"),
}


def digest(value):
    raw = value if isinstance(value, bytes) else json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def text_digest(text):
    return digest(normalized(text).encode())


def verify_pinned_file(path, logical_name):
    path = Path(path)
    size, sha = PINNED_FILES[logical_name]
    actual = {"bytes": path.stat().st_size, "sha256": file_digest(path)}
    if (actual["bytes"], actual["sha256"]) != (size, sha):
        raise ValueError(f"Pinned source bytes changed: {logical_name}")
    return actual


@dataclass
class PriorExclusions:
    conversation_ids: set = field(default_factory=set)
    normalized_hashes: set = field(default_factory=set)
    manifests: list = field(default_factory=list)


def load_prior_manifests(paths, *, old_corpus_paths=(), repo_root=None):
    """Read public identities/hashes, never require old private conversations."""
    prior = PriorExclusions()
    for path in paths:
        path = Path(path)
        value = json.loads(path.read_text())
        if "public_manifest_sha256" not in value or value["public_manifest_sha256"] != digest(
                {k: v for k, v in value.items() if k != "public_manifest_sha256"}):
            raise ValueError(f"Invalid public manifest seal: {path}")
        records = [r for b in value["blocks"] for r in b["sources"]]
        if not records:
            raise ValueError(f"Empty prior source manifest: {path}")
        for row in records:
            if row["dataset"] == "wildchat":
                prior.conversation_ids.add(str(row["source_id"]))
            if "normalized_text_sha256" not in row:
                raise ValueError(f"Missing normalized source hash: {path}")
            prior.normalized_hashes.add(row["normalized_text_sha256"])
            if row.get("captured_prefix_normalized_sha256"):
                prior.normalized_hashes.add(row["captured_prefix_normalized_sha256"])
        prior.manifests.append({"path": _display_path(path, repo_root),
                                "sha256": file_digest(path), "records": len(records)})
    # Earliest experiments used public-domain books/code. Their public manifests
    # retain raw source paths, allowing exact normalized whole-source exclusions.
    for path in old_corpus_paths:
        path = Path(path)
        value = json.loads(path.read_text())
        records = [r for b in value["blocks"] for r in b["sources"]]
        for row in records:
            raw_path = path.parent / row["file"]
            if file_digest(raw_path) != row["sha256"]:
                raise ValueError(f"Historical source bytes changed: {raw_path}")
            text = raw_path.read_text(encoding="utf-8-sig")
            if row["cleaning"] == "content_between_gutenberg_start_end_markers":
                start = re.search(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*\*\*\*", text)
                end = re.search(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*\*\*\*", text)
                if not start or not end or start.end() >= end.start():
                    raise ValueError("Invalid historical cleaning boundaries")
                text = text[start.end():end.start()].strip()
            elif row["cleaning"] != "none":
                raise ValueError("Unknown historical source cleaning")
            prior.normalized_hashes.add(text_digest(text))
        prior.manifests.append({"path": _display_path(path, repo_root),
                                "sha256": file_digest(path), "records": len(records)})
    return prior


def _display_path(path, repo_root):
    return str(path.resolve().relative_to(Path(repo_root).resolve())) if repo_root else path.name


def conversation_prefix(row):
    """Keep all recorded history through the final user turn, without targets."""
    turns = row.get("conversation")
    if not isinstance(turns, list) or not turns:
        raise ValueError("missing_conversation")
    roles = [t.get("role") if isinstance(t, dict) else None for t in turns]
    start = 1 if roles[0] == "system" else 0
    if start == len(roles) or any(role != ("user" if (i-start) % 2 == 0 else "assistant")
                                  for i, role in enumerate(roles[start:], start)):
        raise ValueError("invalid_role_order")
    last_user = max(i for i, role in enumerate(roles) if role == "user")
    prefix = turns[:last_user + 1]
    if any(not isinstance(t.get("content"), str) or not t["content"].strip() for t in prefix):
        raise ValueError("empty_or_nontext_message")
    if any(t.get("toxic", False) for t in prefix):
        raise ValueError("toxic_message")
    return [{"role": t["role"], "content": t["content"]} for t in prefix], last_user


def length_distribution(lengths):
    ordered = sorted(lengths)
    if not ordered:
        return {"count": 0}
    return {"count": len(ordered), "minimum": ordered[0], "maximum": ordered[-1],
            "mean": sum(ordered) / len(ordered),
            **{f"p{p}": ordered[round((len(ordered)-1)*p/100)] for p in (25, 50, 75, 90, 95, 99)},
            "at_most_512": sum(x <= 512 for x in ordered),
            "513_to_2048": sum(512 < x <= 2048 for x in ordered),
            "2049_to_8192": sum(2048 < x <= 8192 for x in ordered),
            "over_8192": sum(x > 8192 for x in ordered)}


def prepare_dataset(rows, tokenizer, prior, *, seed=SEED, pilot_requests=64,
                    final_bundles=6, requests_per_bundle=64, max_context=MAX_CONTEXT,
                    max_new_tokens=MAX_NEW_TOKENS, smoke_requests=8):
    """Return private JSONL-ready requests and a public, explicitly draft manifest.

    Iterate the complete pinned source frame before sampling. Tokenize the hash
    ordered, deduplicated candidates only until the requested sample is full;
    statistics distinguish unexamined candidates from accepted requests.
    """
    if min(pilot_requests, final_bundles, requests_per_bundle, smoke_requests, max_new_tokens) <= 0:
        raise ValueError("Sample counts and output limit must be positive")
    if max_new_tokens >= max_context:
        raise ValueError("Output allowance leaves no prompt context")
    if smoke_requests > pilot_requests:
        raise ValueError("Smoke subset must fit inside pilot")
    candidates, rejected, source_ids = [], [], {}
    scanned = 0
    for index, row in enumerate(rows):
        scanned += 1
        identity = row.get("conversation_id")
        reject = {"source_row": index, "conversation_id": identity}
        reason = None
        if not isinstance(identity, str) or not identity:
            reason = "missing_conversation_id"
        elif identity in prior.conversation_ids:
            reason = "prior_conversation_id"
        elif row.get("language") != "English":
            reason = "non_english"
        elif row.get("toxic", False):
            reason = "toxic_conversation"
        if reason:
            rejected.append({**reject, "reason": reason})
            continue
        try:
            messages, last_user = conversation_prefix(row)
        except ValueError as exc:
            rejected.append({**reject, "reason": str(exc)})
            continue
        message_hash = digest(messages)
        if identity in source_ids:
            if source_ids[identity] != message_hash:
                raise ValueError(f"Ambiguous conversation identity: {identity}")
            rejected.append({**reject, "reason": "duplicate_conversation_id"})
            continue
        source_ids[identity] = message_hash
        normalized_messages = [{"role": m["role"], "content": normalized(m["content"])} for m in messages]
        first_user = next(m["content"] for m in messages if m["role"] == "user")
        hashes = {text_digest(m["content"]) for m in messages}
        hashes.add(text_digest("\n".join(m["content"] for m in messages)))
        if hashes & prior.normalized_hashes:
            rejected.append({**reject, "reason": "prior_normalized_source_content"})
            continue
        candidates.append({"id": f"wc-{identity}-{last_user}", "dataset": "wildchat",
                           "conversation_id": identity, "source_row": index,
                           "messages": messages, "messages_sha256": message_hash,
                           "normalized_messages_sha256": digest(normalized_messages),
                           "first_user_normalized_sha256": text_digest(first_user),
                           "last_user_turn_index": last_user,
                           "recorded_assistant_history_turns": sum(m["role"] == "assistant" for m in messages)})
    candidates.sort(key=lambda r: (digest([seed, r["conversation_id"], r["messages_sha256"]]), r["id"]))
    unique, seen_content = [], set()
    for row in candidates:
        keys = {row["normalized_messages_sha256"], row["first_user_normalized_sha256"]}
        if keys & seen_content:
            rejected.append({"source_row": row["source_row"], "conversation_id": row["conversation_id"],
                             "reason": "duplicate_candidate_content"})
        else:
            unique.append(row)
        # Index every candidate, including duplicate continuations, before split.
        seen_content.update(keys)
    needed = pilot_requests + final_bundles * requests_per_bundle
    requests, examined_lengths = [], []
    for row in unique:
        tokens = tokenizer.apply_chat_template(row["messages"], tokenize=True,
                                               add_generation_prompt=True, enable_thinking=False,
                                               truncation=False, return_dict=False)
        if not isinstance(tokens, list) or any(not isinstance(t, int) for t in tokens):
            raise ValueError("Expected a flat chat-template token ID list")
        length = len(tokens)
        examined_lengths.append(length)
        if length < 1 or length + max_new_tokens > max_context:
            rejected.append({"source_row": row["source_row"], "conversation_id": row["conversation_id"],
                             "reason": "context_overflow" if length else "empty_template",
                             "input_tokens": length, "max_tokens": max_new_tokens,
                             "max_model_len": max_context})
            continue
        position = len(requests)
        block = "pilot-00" if position < pilot_requests else f"final-{(position-pilot_requests)//requests_per_bundle:02d}"
        requests.append({**row, "input_tokens": length, "prompt_token_ids_sha256": digest(tokens),
                         "split": "pilot" if position < pilot_requests else "final",
                         "content_block": block, "max_tokens": max_new_tokens})
        if len(requests) == needed:
            break
    if len(requests) < needed:
        raise ValueError(f"Insufficient fresh eligible conversations: {len(requests)} < {needed}; no reuse allowed")
    pilot_by_length = sorted(requests[:pilot_requests], key=lambda r: (r["input_tokens"], r["id"]))
    positions = [round(i * (pilot_requests-1) / (smoke_requests-1)) for i in range(smoke_requests)] if smoke_requests > 1 else [pilot_requests//2]
    smoke_ids = [pilot_by_length[i]["id"] for i in positions]
    public_rows = [{k: v for k, v in r.items() if k != "messages"} for r in requests]
    manifest = {
        "schema": "full_serving_requests_v1", "study_id": "full_serving_native_v1",
        "selection_status": "draft_until_pilot_freeze", "final_protocol_frozen": False,
        "dataset": "allenai/WildChat", "dataset_revision": DATASET_REVISION,
        "source_frame": "All rows of pinned train-00000-of-00006.parquet; English, non-toxic; not all WildChat shards",
        "source_model": MODEL, "source_revision": MODEL_REVISION,
        "seed": seed, "selection": "SHA256(seed, conversation_id, messages_sha256), ascending; no performance or routing selection",
        "deduplication": "Exclude prior conversation IDs and exact normalized message/source text; deduplicate first-user text and full prompt before split. Near duplicates are not guaranteed absent.",
        "formatting": {"prefix": "Complete recorded history through last user turn; following recorded assistant answer excluded",
                       "template": "Pinned Qwen tokenizer chat_template", "add_generation_prompt": True,
                       "enable_thinking": False, "truncation": False, "max_model_len": max_context},
        "generation": {"temperature": 0, "max_tokens": max_new_tokens, "ignore_eos": False,
                       "output": "Actual autoregressive next-assistant generation; history is conditioning only"},
        "allocation": {"pilot_requests": pilot_requests, "final_bundles": final_bundles,
                       "requests_per_final_bundle": requests_per_bundle, "smoke_requests": smoke_requests},
        "prior_manifests": prior.manifests,
        "prior_exclusions": {"unique_conversation_ids": len(prior.conversation_ids),
                             "unique_normalized_hashes": len(prior.normalized_hashes)},
        "counts": {"source_rows": scanned, "source_filtered_candidates": len(candidates),
                   "content_unique_candidates": len(unique), "tokenized_candidates": len(examined_lengths),
                   "unexamined_context_eligibility": len(unique)-len(examined_lengths), "selected": len(requests)},
        "exclusion_counts": dict(sorted(Counter(r["reason"] for r in rejected).items())),
        "exclusions": rejected,
        "length_distributions": {"examined_before_context_filter": length_distribution(examined_lengths),
                                 "selected": length_distribution([r["input_tokens"] for r in requests]),
                                 **{block: length_distribution([r["input_tokens"] for r in requests if r["content_block"] == block])
                                    for block in sorted({r["content_block"] for r in requests})}},
        "long_prompt_check": {"selected_over_512": sum(r["input_tokens"] > 512 for r in requests),
                              "reweighting_performed": False},
        "smoke_request_ids": smoke_ids, "smoke_selection": "Evenly spaced order statistics of pilot prompt length; diagnostic only",
        "requests": public_rows,
        "scope": "Fresh conversation-disjoint inputs relative to archived selections; no pretraining-disjointness claim. Final held-out selection is draft until pilot feasibility and arrival protocol are frozen.",
    }
    return requests, manifest

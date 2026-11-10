"""Compare already generated pilot answers without exposing their private text.

This is a descriptive token-prefix/logprob diagnostic, not a numerical gate.
It runs no model and never compares logprobs after the generated paths diverge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from full_serving.audit_pilots import client_hash, require
from full_serving.data import file_digest


def read_rows(path):
    raw = Path(path).read_bytes()
    require(raw.endswith(b"\n"), "incomplete JSONL tail")
    rows = [json.loads(line) for line in raw.split(b"\n") if line]
    require(all(isinstance(row, dict) for row in rows), "JSONL rows must be objects")
    return rows


def token_ids(value):
    return isinstance(value, list) and all(type(i) is int and i >= 0 for i in value)


def describe(values):
    return {"n": len(values), "mean": statistics.fmean(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def bind_outputs(public_path, private_path):
    public = read_rows(public_path)
    headers = [r for r in public if r.get("type") == "run_header"]
    summaries = [r for r in public if r.get("type") == "run_summary"]
    rows = sorted((r for r in public if r.get("type") == "request"), key=lambda r: r["request_index"])
    require(len(headers) == len(summaries) == 1 and len(public) == len(rows)+2 and bool(rows),
            "one complete public diagnostic replay is required")
    header, summary = headers[0], summaries[0]
    require(header.get("logprobs") is True and header.get("diagnostic_return_token_ids") is True,
            "token-ID/logprob diagnostic replay required")
    ids = [r["request_id"] for r in rows]
    require(all(isinstance(rid, str) and rid for rid in ids) and len(set(ids)) == len(ids)
            and [r["request_index"] for r in rows] == list(range(len(rows)))
            and header.get("request_count") == len(rows)
            and header.get("request_ids_sha256") == client_hash(ids), "public request membership differs")
    require(summary.get("valid") is True and summary.get("completed_requests") == len(rows)
            and summary.get("failed_requests") == 0, "public pilot is incomplete or failed")
    private = read_rows(private_path)
    private_ids = [r.get("request_id") for r in private]
    require(len(private) == len(rows) and len(set(private_ids)) == len(private_ids)
            and set(private_ids) == set(ids), "private/public request membership differs")
    indexed = {r["request_id"]: r for r in private}
    bound = []
    for row in rows:
        value = indexed[row["request_id"]]
        require(row.get("success") is True and row.get("diagnostic_requested") is True
                and row.get("diagnostic_passed") is True, "request diagnostic did not pass")
        output = value.get("output")
        require(isinstance(output, str), "private output must be text")
        output_sha = hashlib.sha256(output.encode()).hexdigest()
        require(output_sha == row.get("output_sha256") == value.get("output_sha256")
                and len(output) == row.get("output_chars")
                and len(output.encode()) == row.get("output_utf8_bytes"), "private output hash or size differs")
        prompt, completion = value.get("prompt_token_ids"), value.get("completion_token_ids")
        require(token_ids(prompt) and token_ids(completion), "private token IDs are invalid")
        usage = row.get("usage", {})
        require(len(prompt) == row.get("prompt_token_ids_count") == usage.get("prompt_tokens") == row.get("expected_input_tokens")
                and client_hash(prompt) == row.get("prompt_token_ids_sha256") == row.get("expected_prompt_token_ids_sha256"),
                "private prompt IDs differ from admitted public hash/count")
        require(len(completion) == row.get("completion_token_ids_count") == usage.get("completion_tokens")
                and client_hash(completion) == row.get("completion_token_ids_sha256"),
                "private completion IDs differ from public hash/count")
        require(usage.get("total_tokens") == len(prompt)+len(completion), "public token usage total differs")
        events = value.get("token_logprobs")
        require(isinstance(events, list), "private logprob events are missing")
        logprobs = []
        for event in events:
            require(isinstance(event, dict) and isinstance(event.get("logprobs"), dict)
                    and isinstance(event["logprobs"].get("content"), list), "invalid private logprob event")
            for entry in event["logprobs"]["content"]:
                number = entry.get("logprob") if isinstance(entry, dict) else None
                require(type(number) in (int, float) and math.isfinite(number), "invalid private selected-token logprob")
                logprobs.append(number)
        require(len(logprobs) == len(completion) == row.get("diagnostic_logprob_count"),
                "private logprob/token count differs")
        require(sum(number == -9999.0 for number in logprobs)
                == row.get("diagnostic_clamped_or_placeholder_logprobs"), "placeholder count differs")
        bound.append({"public": row, "prompt": prompt, "completion": completion, "logprobs": logprobs})
    return header, bound


def compare_outputs(left_public, left_private, right_public, right_private):
    left_header, left = bind_outputs(left_public, left_private)
    right_header, right = bind_outputs(right_public, right_private)
    require([r["public"]["request_id"] for r in left] == [r["public"]["request_id"] for r in right],
            "paired diagnostic request membership/order differs")
    require(all(left_header.get(key) == right_header.get(key) for key in (
        "model", "max_tokens", "max_model_len", "temperature", "enable_thinking", "ignore_eos")),
        "paired diagnostic generation configuration differs")
    pairs, all_differences = [], []
    for a, b in zip(left, right):
        x, y = a["public"], b["public"]
        require(a["prompt"] == b["prompt"] and x.get("input_sha256") == y.get("input_sha256")
                and x.get("payload_sha256") == y.get("payload_sha256"), "paired prompts or HTTP payloads differ")
        u, v = a["completion"], b["completion"]
        prefix = 0
        for token_a, token_b in zip(u, v):
            if token_a != token_b:
                break
            prefix += 1
        eligible = [(p, q) for p, q in zip(a["logprobs"][:prefix], b["logprobs"][:prefix])
                    if p != -9999.0 and q != -9999.0]
        differences = [abs(p-q) for p, q in eligible]
        all_differences.extend(differences)
        pairs.append({"request_id": x["request_id"], "first_generated_token_match": u[0] == v[0] if u and v else None,
            "common_generated_prefix_tokens": prefix, "generated_token_sequences_identical": u == v,
            "left_completion_tokens": len(u), "right_completion_tokens": len(v),
            "completion_token_count_difference_right_minus_left": len(v)-len(u),
            "output_hash_matches": x["output_sha256"] == y["output_sha256"],
            "left_finish_reason": x.get("finish_reason"), "right_finish_reason": y.get("finish_reason"),
            "selected_token_abs_logprob_difference": describe(differences),
            "shared_prefix_placeholder_pairs_excluded": prefix-len(differences),
            "logprob_positions_outside_shared_prefix_excluded": {"left": len(u)-prefix, "right": len(v)-prefix}})
    return {"schema": "cs471_pilot_output_diagnostic_v1", "integrity_status": "verified",
        "left_arm": left_header.get("arm"), "right_arm": right_header.get("arm"),
        "evidence_sha256": {"left_public": file_digest(left_public), "left_private": file_digest(left_private),
                            "right_public": file_digest(right_public), "right_private": file_digest(right_private)},
        "scope": "Existing diagnostic pilot outputs only. No model execution, quality claim, numerical-equivalence verdict or acceptance threshold.",
        "conditioning": "Logprobs are compared only where prompt token IDs, all earlier generated token IDs and the selected token ID match. Later positions remain excluded even if tokens reconverge.",
        "limitations": "Selected-token logprobs are not full logits. Server event order aligns logprob entries with completion token IDs; token bytes/text are not independently retokenized. Token sequences may include EOS absent from output text. Identical token paths need not imply identical dynamic batches or floating-point reductions. -9999 placeholders are excluded.",
        "private_content_published": False, "requests": pairs,
        "aggregate": {"paired_requests": len(pairs),
            "first_generated_token_comparable_requests": sum(p["first_generated_token_match"] is not None for p in pairs),
            "first_generated_token_matches": sum(p["first_generated_token_match"] is True for p in pairs),
            "identical_generated_token_sequences": sum(p["generated_token_sequences_identical"] for p in pairs),
            "matching_output_hashes": sum(p["output_hash_matches"] for p in pairs),
            "matching_completion_token_counts": sum(p["left_completion_tokens"] == p["right_completion_tokens"] for p in pairs),
            "common_generated_prefix_tokens": describe([p["common_generated_prefix_tokens"] for p in pairs]),
            "selected_token_abs_logprob_difference": describe(all_differences),
            "logprob_aggregation_unit": "Eligible generated-token positions; descriptive only, not independent statistical samples.",
            "shared_prefix_placeholder_pairs_excluded": sum(p["shared_prefix_placeholder_pairs_excluded"] for p in pairs)}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("left-public", "left-private", "right-public", "right-private", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = vars(parser.parse_args(argv))
    output = args.pop("output")
    report = compare_outputs(**args)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps({"integrity_status": report["integrity_status"], "paired_requests": report["aggregate"]["paired_requests"]}))


if __name__ == "__main__":
    main()

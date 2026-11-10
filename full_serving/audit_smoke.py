"""Independently audit closed OFF/ON smoke records without exposing private text.

python -m full_serving.audit_smoke CLOSED_COHORT --private-dir PRIVATE_DIR --output AUDIT.json
PRIVATE_DIR contains off-smoke-outputs.jsonl and on-smoke-outputs.jsonl.
Use --private-prefix wildchat- for older dataset-prefixed filenames.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics


IDENTITIES = ("requests_sha256", "calendar_sha256", "request_ids_sha256", "block_id", "pair_id", "load_id",
              "model", "max_tokens", "max_model_len", "temperature", "enable_thinking", "ignore_eos", "logprobs",
              "diagnostic_return_token_ids", "client_source_sha256", "max_inflight", "http_timeout_s")


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(value):
    return _hash(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _read_jsonl(path):
    raw = path.read_bytes()
    # Literal U+2028/U+2029 are legitimate JSON string content, not record delimiters.
    rows = [json.loads(line) for line in raw.decode("utf-8").split("\n") if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("JSONL records must be objects")
    return rows, _hash(raw)


def _index(rows):
    ids = [row.get("request_id") for row in rows]
    if not all(isinstance(rid, str) and rid for rid in ids) or len(set(ids)) != len(ids):
        raise ValueError("missing or duplicate request IDs")
    return dict(zip(ids, rows))


def _differences(values):
    return {"count": len(values), "mean_absolute": statistics.mean(abs(v) for v in values) if values else None,
            "max_absolute": max((abs(v) for v in values), default=None),
            "signed_mean": statistics.mean(values) if values else None}


def _validate_request(public, private, header, decoder=None):
    """Return public issue codes and private comparison state; never error text."""
    issues = []
    if (public.get("success") is not True or public.get("saw_done") is not True
            or public.get("http_status") != 200 or public.get("usage_events") != 1
            or public.get("error_type") is not None or public.get("error_message") is not None
            or private.get("error_detail") is not None):
        issues.append("request success/HTTP/DONE/error gate failed")
    if public.get("diagnostic_requested") is not True or public.get("diagnostic_passed") is not True:
        issues.append("diagnostic admission did not pass")
    usage = public.get("usage", {})
    if (not isinstance(usage, dict) or not all(_integer(usage.get(k)) for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
            or usage.get("total_tokens") != usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)):
        issues.append("usage counts invalid")
        return issues, None
    prompt, generated = private.get("prompt_token_ids"), private.get("completion_token_ids")
    if not all(isinstance(ids, list) and all(_integer(x) for x in ids) for ids in (prompt, generated)):
        issues.append("private token IDs invalid or unavailable")
        return issues, None
    if (len(prompt) != usage["prompt_tokens"] or public.get("prompt_token_ids_count") != len(prompt)
            or public.get("expected_input_tokens") != len(prompt)):
        issues.append("prompt token count mismatch")
    if (len(generated) != usage["completion_tokens"] or public.get("completion_token_ids_count") != len(generated)):
        issues.append("completion token count mismatch")
    prompt_hash = _canonical(prompt)
    if (public.get("prompt_token_ids_sha256") != prompt_hash
            or public.get("expected_prompt_token_ids_sha256") != prompt_hash):
        issues.append("prompt token hash or expected prompt hash mismatch")
    if public.get("completion_token_ids_sha256") != _canonical(generated):
        issues.append("completion token hash mismatch")
    output = private.get("output")
    if not isinstance(output, str):
        issues.append("private generated text unavailable")
    else:
        output_hash = _hash(output.encode())
        if output_hash != private.get("output_sha256") or output_hash != public.get("output_sha256"):
            issues.append("generated text hash mismatch")
        if public.get("output_chars") != len(output) or public.get("output_utf8_bytes") != len(output.encode()):
            issues.append("generated text length mismatch")
        if decoder is not None:
            try:
                if decoder(generated) != output:
                    issues.append("local tokenizer decode differs from streamed text")
            except Exception:
                issues.append("local tokenizer decode failed")
    if (len(prompt) + len(generated) > header["max_model_len"] or len(generated) > header["max_tokens"]
            or public.get("finish_reason") not in ("stop", "length")
            or (public.get("finish_reason") == "length" and len(generated) != header["max_tokens"])):
        issues.append("context/output cap or finish-reason gate failed")
    logprobs = []
    try:
        events = private["token_logprobs"]
        if not isinstance(events, list):
            raise ValueError
        for event in events:
            content = event["logprobs"]["content"]
            if not isinstance(content, list):
                raise ValueError
            for token in content:
                value = token["logprob"]
                if not _finite(value):
                    raise ValueError
                logprobs.append(value)
    except (KeyError, TypeError, ValueError):
        issues.append("selected-token logprobs missing, malformed or nonfinite")
    if len(logprobs) != len(generated) or public.get("diagnostic_logprob_count") != len(logprobs):
        issues.append("selected-token logprob count mismatch")
    clamped = sum(value == -9999 for value in logprobs)
    if public.get("diagnostic_clamped_or_placeholder_logprobs") != clamped:
        issues.append("clamped/placeholder logprob count mismatch")
    return issues, {"prompt": prompt, "generated": generated, "logprobs": logprobs, "clamped": clamped,
                    "output_hash": public.get("output_sha256")}


def audit_smoke(cohort, private_dir, *, private_prefix="", expected_requests=8, decoder=None, tokenizer_description=None):
    cohort, private_dir = Path(cohort), Path(private_dir)
    result = {"schema": "full_serving_smoke_paired_audit_v2", "result_directory": str(cohort.resolve()),
              "expected_request_pairs": expected_requests, "valid": False, "issues": [], "arms": {},
              "source_sha256": {}, "requests": [], "comparison_available": False, "request_pairs": None,
              "equal_generated_token_sequences": None, "equal_generated_texts": None,
              "shared_prefix_selected_logprob_difference_on_minus_off": None,
              "logprob_comparison_available": False,
              "decoder": {"requested": decoder is not None, "tokenizer": tokenizer_description,
                          "verified": False if decoder is not None else None},
              "comparison_boundary": "Only same selected token under exactly identical prompt and previously generated prefix. Divergent positions and all later positions are excluded.",
              "tolerance": "No numerical tolerance is invented. Differences are descriptive, not a numerical pass/fail threshold.",
              "interpretation": "Greedy generations can differ with batching/reduction order. These checks cannot identify an algorithmic cause or distinguish near-tie selection from mapping/kernel errors. Different outputs change subsequent workloads.",
              "full_model_numerical_correctness_proved": False, "algorithmic_cause_established": False, "speedup_claim": False}
    if not _integer(expected_requests) or expected_requests == 0:
        raise ValueError("expected_requests must be positive")
    try:
        status_raw, splits_raw = (cohort / "status.json").read_bytes(), (cohort / "pilot-splits.json").read_bytes()
        status, splits = json.loads(status_raw), json.loads(splits_raw)
        if not isinstance(status, dict) or not isinstance(splits, dict):
            raise ValueError
        expected_ids = splits["smoke"]["ids"]
        result.update(dataset=status.get("dataset"), controller_state=status.get("state"))
        result["source_sha256"].update(status=_hash(status_raw), pilot_splits=_hash(splits_raw))
        if status.get("state") != "pilot_completed":
            result["issues"].append("cohort is not closed with pilot_completed status")
        if (not isinstance(expected_ids, list) or len(expected_ids) != expected_requests
                or not all(isinstance(x, str) and x for x in expected_ids) or len(set(expected_ids)) != len(expected_ids)):
            raise ValueError
    except (OSError, ValueError, TypeError, KeyError):
        result["issues"].append("closed-cohort status or frozen smoke membership missing/invalid")
        return result
    states, headers, public_rows = {}, {}, {}
    for arm in ("off", "on"):
        arm_result = {"valid": False, "issues": [], "public_requests": None, "private_requests": None}
        result["arms"][arm] = arm_result
        try:
            public, pubhash = _read_jsonl(cohort / arm / "smoke.jsonl")
            private, privhash = _read_jsonl(private_dir / f"{private_prefix}{arm}-smoke-outputs.jsonl")
            result["source_sha256"][arm] = {"public": pubhash, "private": privhash}
            hs = [r for r in public if r.get("type") == "run_header"]
            ss = [r for r in public if r.get("type") == "run_summary"]
            rows = [r for r in public if r.get("type") == "request"]
            arm_result.update(public_requests=len(rows), private_requests=len(private))
            if (len(hs) != 1 or len(ss) != 1 or len(public) != len(rows) + 2
                    or len(rows) != expected_requests or len(private) != expected_requests
                    or any(r.get("type") != "private_request" for r in private)):
                raise ValueError
            h, summary = hs[0], ss[0]
            headers[arm] = h
            if (h.get("arm") != arm or h.get("request_count") != expected_requests
                    or not all(key in h for key in IDENTITIES)
                    or h.get("logprobs") is not True or h.get("diagnostic_return_token_ids") is not True
                    or h.get("enable_thinking") is not False or h.get("ignore_eos") is not False or h.get("temperature") != 0
                    or not _integer(h.get("max_tokens")) or not _integer(h.get("max_model_len"))
                    or h.get("max_tokens", 0) == 0 or h.get("max_model_len", 0) == 0
                    or h.get("request_ids_sha256") != _canonical(expected_ids)
                    or h.get("requests_sha256") != splits["smoke"]["request_file_sha256"]):
                raise ValueError
            if (summary.get("valid") is not True or summary.get("offered_requests") != expected_requests
                    or summary.get("completed_requests") != expected_requests or summary.get("failed_requests") != 0):
                arm_result["issues"].append("run summary incomplete or failed")
            indexed, secrets = _index(rows), _index(private)
            if set(indexed) != set(expected_ids) or set(secrets) != set(expected_ids):
                raise ValueError
            states[arm], public_rows[arm] = {}, indexed
            for index, rid in enumerate(expected_ids):
                if indexed[rid].get("request_index") != index:
                    arm_result["issues"].append(f"request index {index}: frozen index mismatch")
                issues, state = _validate_request(indexed[rid], secrets[rid], h, decoder)
                arm_result["issues"].extend(f"request index {index}: {issue}" for issue in issues)
                states[arm][rid] = state
            if not arm_result["issues"]:
                arm_result.update(valid=True,
                    completion_tokens=sum(len(states[arm][rid]["generated"]) for rid in expected_ids),
                    prompt_tokens=sum(len(states[arm][rid]["prompt"]) for rid in expected_ids),
                    finish_reason_counts=dict(Counter(indexed[rid]["finish_reason"] for rid in expected_ids)),
                    output_cap_completions=sum(indexed[rid]["finish_reason"] == "length" for rid in expected_ids),
                    clamped_or_placeholder_logprobs=sum(states[arm][rid]["clamped"] for rid in expected_ids),
                    token_ids_decode_matches_streamed_text=True if decoder is not None else None)
                if (summary.get("completed_output_tokens") != arm_result["completion_tokens"]
                        or summary.get("completed_input_tokens") != arm_result["prompt_tokens"]
                        or summary.get("output_cap_completions") != arm_result["output_cap_completions"]
                        or summary.get("finish_reason_counts") != arm_result["finish_reason_counts"]):
                    arm_result["issues"].append("run summary token/finish totals mismatch")
                    arm_result["valid"] = False
        except (OSError, ValueError, TypeError, KeyError):
            arm_result["issues"].append("missing/malformed files, duplicate/missing requests, membership or header mismatch")
        if not arm_result["valid"]:
            result["issues"].append(f"{arm} smoke evidence failed; no successful-subset comparison")
    if result["issues"]:
        return result
    mismatches = [key for key in IDENTITIES if headers["off"][key] != headers["on"][key]]
    for rid in expected_ids:
        b, t = public_rows["off"][rid], public_rows["on"][rid]
        for key in ("input_sha256", "payload_sha256", "expected_input_tokens", "expected_prompt_token_ids_sha256", "arrival_offset_ns"):
            if key not in b or key not in t or b[key] != t[key]:
                mismatches.append(key)
        if states["off"][rid]["prompt"] != states["on"][rid]["prompt"]:
            mismatches.append("actual_prompt_token_ids")
    if mismatches:
        result["issues"].append("OFF/ON input/configuration identity mismatch: " + ", ".join(sorted(set(mismatches))))
        return result
    result.update(valid=True, comparison_available=True, request_pairs=expected_requests, input_token_sequences_identical=True,
                  equal_generated_token_sequences=0, equal_generated_texts=0,
                  off_completion_tokens=result["arms"]["off"]["completion_tokens"],
                  on_completion_tokens=result["arms"]["on"]["completion_tokens"])
    has_clamps = any(result["arms"][arm]["clamped_or_placeholder_logprobs"] for arm in ("off", "on"))
    result["logprob_comparison_available"] = not has_clamps
    result["logprob_comparison_note"] = ("Withheld: clamped/placeholder selected logprobs are not genuine probability measurements."
                                          if has_clamps else "All selected logprobs finite, with no -9999 placeholders; compare common-prefix positions only.")
    differences = []
    for index, rid in enumerate(expected_ids):
        off, on = states["off"][rid], states["on"][rid]
        left, right = off["generated"], on["generated"]
        shared = 0
        for a, b in zip(left, right):
            if a != b:
                break
            shared += 1
        equal = left == right
        local = [on["logprobs"][i] - off["logprobs"][i] for i in range(shared)] if not has_clamps else []
        differences.extend(local)
        text_equal = off["output_hash"] == on["output_hash"]
        result["equal_generated_token_sequences"] += equal
        result["equal_generated_texts"] += text_equal
        result["requests"].append({"request_id": rid, "request_index": index,
            "off_generated_tokens": len(left), "on_generated_tokens": len(right),
            "off_finish_reason": public_rows["off"][rid]["finish_reason"],
            "on_finish_reason": public_rows["on"][rid]["finish_reason"],
            "generated_token_ids_equal": equal, "generated_text_equal": text_equal,
            "shared_generated_prefix_tokens": shared, "first_divergence_zero_based_index": None if equal else shared,
            "divergence_kind": "none" if equal else "sequence_ended" if shared == min(len(left), len(right)) else "different_next_token",
            "shared_prefix_selected_logprob_difference_on_minus_off": _differences(local) if not has_clamps else None,
            "post_divergence_logprobs_compared": False})
    result["shared_generated_prefix_tokens"] = sum(r["shared_generated_prefix_tokens"] for r in result["requests"])
    result["shared_prefix_selected_logprob_difference_on_minus_off"] = _differences(differences) if not has_clamps else None
    result["decoder"]["verified"] = True if decoder is not None else None
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("--private-dir", type=Path, required=True)
    parser.add_argument("--private-prefix", default="")
    parser.add_argument("--expected-requests", type=int, default=8)
    parser.add_argument("--tokenizer-dir", type=Path, help="Optional pinned local tokenizer; never downloaded")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    decoder = None
    if args.tokenizer_dir:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True, trust_remote_code=False)
        decoder = lambda ids: tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    result = audit_smoke(args.cohort, args.private_dir, private_prefix=args.private_prefix,
                         expected_requests=args.expected_requests, decoder=decoder,
                         tokenizer_description=str(args.tokenizer_dir) if args.tokenizer_dir else None)
    evidence_files = {args.private_dir / f"{args.private_prefix}{arm}-smoke-outputs.jsonl" for arm in ("off", "on")}
    evidence_files.update(args.cohort / arm / "smoke.jsonl" for arm in ("off", "on"))
    evidence_files.update(args.cohort / name for name in ("status.json", "pilot-splits.json"))
    if args.output.resolve() in {p.resolve() for p in evidence_files}:
        parser.error("audit output must not overwrite source evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "valid": result["valid"], "request_pairs": result["request_pairs"],
                      "equal_generations": result["equal_generated_token_sequences"],
                      "logprob_comparison_available": result["logprob_comparison_available"]}))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

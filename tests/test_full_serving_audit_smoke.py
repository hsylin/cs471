import hashlib
import json
from pathlib import Path

import pytest

from full_serving.audit_smoke import audit_smoke, main


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def text_digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")


def read_rows(path):
    return [json.loads(line) for line in path.read_text().split("\n") if line]


def make_cohort(tmp_path):
    cohort, private = tmp_path / "closed", tmp_path / "private"
    cohort.mkdir()
    ids = ["r0", "r1", "r2"]
    (cohort / "status.json").write_text(json.dumps({"dataset": "test", "state": "pilot_completed"}))
    (cohort / "pilot-splits.json").write_text(json.dumps({"smoke": {"ids": ids, "request_file_sha256": "requests"}}))
    for arm in ("off", "on"):
        header = {"type": "run_header", "arm": arm, "request_count": 3, "requests_sha256": "requests",
                  "calendar_sha256": "calendar", "request_ids_sha256": digest(ids), "block_id": "block",
                  "pair_id": "pair", "load_id": "smoke", "model": "qwen", "max_tokens": 3,
                  "max_model_len": 32768, "temperature": 0, "enable_thinking": False, "ignore_eos": False,
                  "logprobs": True, "diagnostic_return_token_ids": True, "client_source_sha256": "source",
                  "max_inflight": 64, "http_timeout_s": 1800}
        records, secrets = [], []
        for index, rid in enumerate(ids):
            prompt = [1, 2]
            generated = [[10, 11, 12], [20, 21] if arm == "off" else [20, 22], [30] if arm == "off" else [30, 31]][index]
            output = "secret text\u2028kept\u2029" + str(generated)
            values = [-.1, -.2, -.3][:len(generated)]
            if arm == "on":
                values = [v + .01 for v in values]
                if index == 1:
                    values[-1] = -1000  # Different selected token: never compared.
            finish = "length" if len(generated) == 3 else "stop"
            records.append({"type": "request", "request_id": rid, "request_index": index,
                "input_sha256": "input" + rid, "payload_sha256": "payload" + rid, "expected_input_tokens": 2,
                "expected_prompt_token_ids_sha256": digest(prompt), "arrival_offset_ns": index * 100,
                "success": True, "saw_done": True, "http_status": 200, "usage_events": 1,
                "error_type": None, "error_message": None, "diagnostic_requested": True, "diagnostic_passed": True,
                "usage": {"prompt_tokens": 2, "completion_tokens": len(generated), "total_tokens": 2 + len(generated)},
                "prompt_token_ids_count": 2, "completion_token_ids_count": len(generated),
                "prompt_token_ids_sha256": digest(prompt), "completion_token_ids_sha256": digest(generated),
                "output_sha256": text_digest(output), "output_chars": len(output), "output_utf8_bytes": len(output.encode()),
                "finish_reason": finish, "diagnostic_logprob_count": len(generated),
                "diagnostic_clamped_or_placeholder_logprobs": 0})
            secrets.append({"type": "private_request", "request_id": rid, "error_detail": None,
                "prompt_token_ids": prompt, "completion_token_ids": generated, "output": output,
                "output_sha256": text_digest(output), "stop_reason": None,
                "token_logprobs": [{"received_ns": 123, "logprobs": {"content": [
                    {"token": "private-token", "logprob": value, "bytes": None, "top_logprobs": []} for value in values]}}]})
        summary = {"type": "run_summary", "valid": True, "offered_requests": 3, "completed_requests": 3,
                   "failed_requests": 0, "completed_input_tokens": 6, "completed_output_tokens": 6 if arm == "off" else 7,
                   "output_cap_completions": 1, "finish_reason_counts": {"length": 1, "stop": 2}}
        write_rows(cohort / arm / "smoke.jsonl", [header, *reversed(records), summary])
        write_rows(private / f"{arm}-smoke-outputs.jsonl", list(reversed(secrets)))
    return cohort, private


def test_hash_binding_unicode_and_common_prefix_excludes_divergent_logprobs(tmp_path):
    cohort, private = make_cohort(tmp_path)
    result = audit_smoke(cohort, private, expected_requests=3)
    assert result["valid"] and result["comparison_available"]
    assert result["request_pairs"] == 3
    assert result["equal_generated_token_sequences"] == result["equal_generated_texts"] == 1
    assert result["shared_generated_prefix_tokens"] == 5
    stats = result["shared_prefix_selected_logprob_difference_on_minus_off"]
    assert stats["count"] == 5 and stats["mean_absolute"] == pytest.approx(.01)
    assert stats["max_absolute"] == pytest.approx(.01)
    assert [r["first_divergence_zero_based_index"] for r in result["requests"]] == [None, 1, 1]
    assert [r["divergence_kind"] for r in result["requests"]] == ["none", "different_next_token", "sequence_ended"]
    assert not result["full_model_numerical_correctness_proved"] and not result["algorithmic_cause_established"]
    assert "secret text" not in json.dumps(result) and "private-token" not in json.dumps(result)
    assert len(result["source_sha256"]["off"]["private"]) == 64


@pytest.mark.parametrize("damage", ["duplicate_private", "missing_private", "duplicate_public", "wrong_text_hash", "wrong_prompt_hash",
                                  "wrong_count", "failed", "nonfinite", "wrong_frozen_membership", "wrong_calendar"])
def test_damage_invalidates_whole_pair_not_successful_subset(tmp_path, damage):
    cohort, private = make_cohort(tmp_path)
    pub, sec = cohort / "on/smoke.jsonl", private / "on-smoke-outputs.jsonl"
    public, secrets = read_rows(pub), read_rows(sec)
    if damage == "duplicate_private":
        secrets[0] = secrets[1]
    elif damage == "missing_private":
        secrets.pop()
    elif damage == "duplicate_public":
        public[1] = public[2]
    elif damage == "wrong_text_hash":
        secrets[0]["output"] += "tamper"
    elif damage == "wrong_prompt_hash":
        public[1]["expected_prompt_token_ids_sha256"] = "wrong"
    elif damage == "wrong_count":
        public[1]["usage"]["completion_tokens"] += 1
    elif damage == "failed":
        public[1]["success"] = False
    elif damage == "nonfinite":
        secrets[0]["token_logprobs"][0]["logprobs"]["content"][0]["logprob"] = float("nan")
    elif damage == "wrong_frozen_membership":
        splits = json.loads((cohort / "pilot-splits.json").read_text())
        splits["smoke"]["ids"][0] = "different"
        (cohort / "pilot-splits.json").write_text(json.dumps(splits))
    elif damage == "wrong_calendar":
        public[0]["calendar_sha256"] = "different"
    write_rows(pub, public)
    write_rows(sec, secrets)
    result = audit_smoke(cohort, private, expected_requests=3)
    assert not result["valid"] and not result["comparison_available"]
    assert result["issues"] and result["request_pairs"] is None
    assert result["shared_prefix_selected_logprob_difference_on_minus_off"] is None
    assert not result["requests"]


def test_clamped_values_are_exposed_and_numerical_statistics_withheld(tmp_path):
    cohort, private = make_cohort(tmp_path)
    pub, sec = cohort / "on/smoke.jsonl", private / "on-smoke-outputs.jsonl"
    public, secrets = read_rows(pub), read_rows(sec)
    secrets[0]["token_logprobs"][0]["logprobs"]["content"][0]["logprob"] = -9999
    public[1]["diagnostic_clamped_or_placeholder_logprobs"] = 1
    write_rows(pub, public)
    write_rows(sec, secrets)
    result = audit_smoke(cohort, private, expected_requests=3)
    assert result["valid"] and result["comparison_available"]
    assert not result["logprob_comparison_available"]
    assert result["arms"]["on"]["clamped_or_placeholder_logprobs"] == 1
    assert result["shared_prefix_selected_logprob_difference_on_minus_off"] is None
    assert all(r["shared_prefix_selected_logprob_difference_on_minus_off"] is None for r in result["requests"])


def test_optional_decoder_is_independent_and_mismatch_invalidates_audit(tmp_path):
    cohort, private = make_cohort(tmp_path)
    outputs = {tuple(r["completion_token_ids"]): r["output"] for arm in ("off", "on")
               for r in read_rows(private / f"{arm}-smoke-outputs.jsonl")}
    result = audit_smoke(cohort, private, expected_requests=3, decoder=lambda ids: outputs[tuple(ids)])
    assert result["valid"] and result["decoder"]["verified"] is True
    bad = audit_smoke(cohort, private, expected_requests=3, decoder=lambda ids: "wrong")
    assert not bad["valid"] and not bad["decoder"]["verified"]


def test_cli_emits_public_audit_and_protects_input_evidence(tmp_path):
    cohort, private = make_cohort(tmp_path)
    output = tmp_path / "audit.json"
    args = [str(cohort), "--private-dir", str(private), "--expected-requests", "3"]
    before = {p: p.read_bytes() for p in private.glob("*.jsonl")}
    assert main([*args, "--output", str(output)]) == 0
    assert json.loads(output.read_text())["valid"]
    for path, raw in before.items():
        assert path.read_bytes() == raw
    for protected in [private / "off-smoke-outputs.jsonl", cohort / "off/smoke.jsonl", cohort / "status.json"]:
        with pytest.raises(SystemExit):
            main([*args, "--output", str(protected)])

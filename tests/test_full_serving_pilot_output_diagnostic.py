import copy
import hashlib
import json

import pytest

from full_serving.audit_pilots import AuditError, client_hash
from full_serving.pilot_output_diagnostic import compare_outputs, main


def dump(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows))


def fixture(tmp_path):
    paths = {}
    for arm in ("left", "right"):
        public, private = [], []
        specs = [([10, 11, 12], [-.1, -.2, -.3]), ([20, 21, 22], [-.2, -9999., -.4]), ([], [])]
        if arm == "right":
            # The final token reconverges, but its conditioning path differs.
            specs[0] = ([10, 99, 12], [-.3, -5., -20.])
            specs[1] = ([20, 21, 22], [-.3, -.8, -.6])
        for index, (completion, logprobs) in enumerate(specs):
            rid = f"pilot-{index}"
            prompt = [1, 2, 3, 4]
            output = "私密 generated answer " + (arm if index == 0 else str(index))
            output_sha = hashlib.sha256(output.encode()).hexdigest()
            record = {"type": "request", "request_id": rid, "request_index": index,
                "success": True, "diagnostic_requested": True, "diagnostic_passed": True,
                "output_sha256": output_sha, "output_chars": len(output), "output_utf8_bytes": len(output.encode()),
                "prompt_token_ids_count": len(prompt), "expected_input_tokens": len(prompt),
                "prompt_token_ids_sha256": client_hash(prompt), "expected_prompt_token_ids_sha256": client_hash(prompt),
                "completion_token_ids_count": len(completion), "completion_token_ids_sha256": client_hash(completion),
                "diagnostic_logprob_count": len(logprobs),
                "diagnostic_clamped_or_placeholder_logprobs": sum(p == -9999. for p in logprobs),
                "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(completion), "total_tokens": len(prompt)+len(completion)},
                "input_sha256": client_hash(prompt), "payload_sha256": client_hash({"prompt": prompt}), "finish_reason": "stop"}
            public.append(record)
            private.append({"request_id": rid, "output": output, "output_sha256": output_sha,
                "prompt_token_ids": prompt, "completion_token_ids": completion,
                "token_logprobs": [{"received_ns": 10+j, "logprobs": {"content": [
                    {"token": "PRIVATE_TOKEN_TEXT", "logprob": p, "bytes": None, "top_logprobs": []}]}}
                    for j, p in enumerate(logprobs)]})
        header = {"type": "run_header", "arm": arm, "logprobs": True, "diagnostic_return_token_ids": True,
            "request_count": len(public), "request_ids_sha256": client_hash([r["request_id"] for r in public]),
            "model": "qwen3-full", "max_tokens": 2048, "max_model_len": 32768, "temperature": 0,
            "enable_thinking": False, "ignore_eos": False}
        summary = {"type": "run_summary", "valid": True, "completed_requests": len(public), "failed_requests": 0}
        paths[arm+"_public"] = tmp_path/(arm+"-public.jsonl")
        paths[arm+"_private"] = tmp_path/(arm+"-private.jsonl")
        dump(paths[arm+"_public"], [header, *public, summary])
        # Private HTTP completions need not finish in request order.
        dump(paths[arm+"_private"], private[::-1])
    return paths


def change(path, mutate):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutate(rows)
    dump(path, rows)


def test_only_identical_conditioning_prefix_logprobs_are_compared(tmp_path):
    report = compare_outputs(**fixture(tmp_path))
    first, second, empty = report["requests"]
    assert first["common_generated_prefix_tokens"] == 1
    assert first["first_generated_token_match"] is True
    assert first["selected_token_abs_logprob_difference"]["n"] == 1
    assert first["selected_token_abs_logprob_difference"]["mean"] == pytest.approx(.2)
    assert first["logprob_positions_outside_shared_prefix_excluded"] == {"left": 2, "right": 2}
    assert second["generated_token_sequences_identical"] is True
    assert second["shared_prefix_placeholder_pairs_excluded"] == 1
    assert second["selected_token_abs_logprob_difference"]["n"] == 2
    assert empty["first_generated_token_match"] is None
    assert empty["selected_token_abs_logprob_difference"]["mean"] is None
    aggregate = report["aggregate"]
    assert aggregate["selected_token_abs_logprob_difference"]["n"] == 3
    assert aggregate["selected_token_abs_logprob_difference"]["mean"] == pytest.approx(1/6)
    assert aggregate["matching_output_hashes"] == 2
    rendered = json.dumps(report, ensure_ascii=False)
    assert "私密" not in rendered and "PRIVATE_TOKEN_TEXT" not in rendered
    assert "prompt_token_ids" not in rendered and "completion_token_ids" not in rendered
    assert "numerical-equivalence verdict" in report["scope"]


@pytest.mark.parametrize("damage", ["text", "prompt_ids", "completion_ids", "missing_member", "duplicate_member", "logprob_count", "nonfinite", "placeholder"])
def test_private_corruption_cannot_pass_public_binding(tmp_path, damage):
    paths = fixture(tmp_path)
    def corrupt(rows):
        row = next(row for row in rows if row["request_id"] == "pilot-0")
        if damage == "text": row["output"] += "changed"
        if damage == "prompt_ids": row["prompt_token_ids"][0] = 999
        if damage == "completion_ids": row["completion_token_ids"][0] = 999
        if damage == "missing_member": rows.pop()
        if damage == "duplicate_member": rows[0] = copy.deepcopy(rows[1])
        if damage == "logprob_count": row["token_logprobs"].pop()
        if damage == "nonfinite": row["token_logprobs"][0]["logprobs"]["content"][0]["logprob"] = float("nan")
        if damage == "placeholder": row["token_logprobs"][0]["logprobs"]["content"][0]["logprob"] = -9999.
    change(paths["right_private"], corrupt)
    with pytest.raises(AuditError):
        compare_outputs(**paths)


def test_different_prompts_reject_comparison_even_with_consistent_individual_hashes(tmp_path):
    paths = fixture(tmp_path)
    prompt = [8, 8, 8, 8]
    change(paths["right_private"], lambda rows: rows[-1].update(prompt_token_ids=prompt))
    change(paths["right_public"], lambda rows: rows[1].update(prompt_token_ids_sha256=client_hash(prompt),
                                                            expected_prompt_token_ids_sha256=client_hash(prompt)))
    with pytest.raises(AuditError, match="paired prompts"):
        compare_outputs(**paths)


def test_changed_first_token_excludes_all_later_positions(tmp_path):
    paths = fixture(tmp_path)
    completion = [99, 11, 12]
    change(paths["right_private"], lambda rows: rows[-1].update(completion_token_ids=completion))
    change(paths["right_public"], lambda rows: rows[1].update(completion_token_ids_sha256=client_hash(completion)))
    result = compare_outputs(**paths)["requests"][0]
    assert result["first_generated_token_match"] is False
    assert result["common_generated_prefix_tokens"] == 0
    assert result["selected_token_abs_logprob_difference"]["n"] == 0


def test_cli_writes_only_new_public_artifact(tmp_path):
    paths = fixture(tmp_path)
    output = tmp_path/"diagnostic.json"
    argv = [part for key, value in paths.items() for part in ("--"+key.replace("_", "-"), str(value))]
    main([*argv, "--output", str(output)])
    first = output.read_bytes()
    with pytest.raises(FileExistsError):
        main([*argv, "--output", str(output)])
    assert output.read_bytes() == first

"""Corruption checks that stored summary validity flags cannot bypass."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from full_serving.analyze import summarize_run
from full_serving.audit_pilots import (AuditError, COUNT_METRICS, CUDA_CRITICAL_FILES, FLASHINFER_SOURCES,
                                      NATIVE_FINISH_REASONS, audit_cohort, audit_counters, audit_request,
                                      audit_run, client_hash, paired_build_evidence, paired_configuration)


def request_fixture(index=0):
    private = {"id": f"request-{index}", "messages": [{"role": "user", "content": "PRIVATE 中文\u2028request"}],
               "input_tokens": 4, "prompt_token_ids_sha256": client_hash([1, 2, 3, 4])}
    header = {"model": "qwen3-full", "max_tokens": 2048, "max_model_len": 32768, "logprobs": False}
    payload = {"model": "qwen3-full", "messages": private["messages"], "stream": True,
               "stream_options": {"include_usage": True}, "temperature": 0, "max_tokens": 2048,
               "n": 1, "chat_template_kwargs": {"enable_thinking": False}}
    row = {"type": "request", "request_id": private["id"], "request_index": index,
           "input_sha256": client_hash(private["messages"]), "payload_sha256": client_hash(payload),
           "output_sha256": hashlib.sha256(b"ab").hexdigest(), "expected_input_tokens": 4,
           "expected_prompt_token_ids_sha256": private["prompt_token_ids_sha256"],
           "scheduled_ns": 100, "arrival_offset_ns": 0, "arrival_wakeup_ns": 110,
           "semaphore_acquired_ns": 120, "send_ns": 130, "headers_ns": 140,
           "first_content_ns": 200, "last_content_ns": 230, "done_ns": 250, "end_ns": 260,
           "success": True, "http_status": 200, "saw_done": True, "error_type": None, "error_message": None,
           "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
           "finish_reason": "stop", "server_model": "qwen3-full", "diagnostic_requested": False,
           "client_schedule_lag_ns": 10, "client_semaphore_wait_ns": 10, "client_send_delay_ns": 30,
           "response_from_schedule_ns": 160, "response_from_send_ns": 130,
           "ttft_from_schedule_ns": 100, "ttft_from_send_ns": 70,
           "tpot_stream_boundary_estimate_ns": 15.0, "output_chars": 2, "output_utf8_bytes": 2,
           "chunks": [{"received_ns": 200, "content_chars": 1, "content_utf8_bytes": 1},
                      {"received_ns": 230, "content_chars": 1, "content_utf8_bytes": 1}]}
    return row, header, private


def test_raw_timestamp_equations_and_private_unicode_payload_pass():
    row, header, private = request_fixture()
    audit_request(row, header, 100, 270, private)


@pytest.mark.parametrize("key,value,reason", [
    ("response_from_send_ns", 129, "timing equation"),
    ("arrival_offset_ns", 1, "absolute arrival"),
    ("first_content_ns", 201, "first/last"),
    ("output_chars", 3, "output size"),
    ("tpot_stream_boundary_estimate_ns", 16.0, "TPOT"),
    ("end_ns", 139, "timestamp order"),
])
def test_inconsistent_raw_record_cannot_pass_on_success_flag(key, value, reason):
    row, header, private = request_fixture()
    row[key] = value
    with pytest.raises(AuditError, match=reason):
        audit_request(row, header, 100, 270, private)


def test_same_token_count_does_not_hide_changed_private_text():
    row, header, private = request_fixture()
    private["messages"][0]["content"] = "different request of same declared length"
    with pytest.raises(AuditError, match="full messages hash"):
        audit_request(row, header, 100, 270, private)


def calibration_fixture(tmp_path):
    records = [request_fixture(i)[0] for i in range(32)]
    ids = [r["request_id"] for r in records]
    calendar = {"kind": "synthetic_pilot_burst", "rate_requests_per_s": 0.0,
                "arrival_offsets_ns": [0]*32, "request_ids": ids}
    header = {**request_fixture()[1], "type": "run_header", "arm": "off", "request_count": 32,
              "request_ids_sha256": client_hash(ids), "requests_sha256": "a"*64,
              "calendar_sha256": hashlib.sha256(json.dumps(calendar, sort_keys=True).encode()).hexdigest(),
              "temperature": 0, "enable_thinking": False, "ignore_eos": False,
              "max_inflight": 64, "diagnostic_return_token_ids": False}
    timing = {"epoch_monotonic_ns": 100, "client_finished_ns": 270, "last_request_end_ns": 260,
              "offered_horizon_ns": 0, "completion_ns": 260, "completed_elapsed_ns": 160, "drain_ns": 160}
    summary = {"type": "run_summary", **timing, **summarize_run(records, timing)}
    path = tmp_path / "calibration.jsonl"
    data = [header, *records, summary]
    path.write_text("".join(json.dumps(r)+"\n" for r in data))
    return path, {"ids": ids, "request_file_sha256": "a"*64}, data


def test_completed_run_recomputes_summary_and_does_not_publish_private_content(tmp_path):
    path, split, _ = calibration_fixture(tmp_path)
    report, _ = audit_run(path, "calibration", "off", split)
    assert report["status"] == "passed"
    assert report["client_tail_after_completion_ns"] == 10
    assert "PRIVATE" not in json.dumps(report)


def test_tampered_summary_mean_rejected_even_with_valid_true(tmp_path):
    path, split, data = calibration_fixture(tmp_path)
    data[-1]["response_from_schedule_ms"]["mean"] += 1
    path.write_text("".join(json.dumps(r)+"\n" for r in data))
    report, _ = audit_run(path, "calibration", "off", split)
    assert report["status"] == "failed" and "stored summary differs" in report["reason"]


def test_incomplete_transfer_is_pending_and_wrong_calendar_is_failed(tmp_path):
    path, split, data = calibration_fixture(tmp_path)
    path.write_text(json.dumps(data[0]))
    assert audit_run(path, "calibration", "off", split)[0]["status"] == "pending"
    data[0]["calendar_sha256"] = "b"*64
    path.write_text("".join(json.dumps(r)+"\n" for r in data))
    report, _ = audit_run(path, "calibration", "off", split)
    assert report["status"] == "failed" and "calendar hash" in report["reason"]


def counter_text(completed):
    lines = []
    for metric in COUNT_METRICS:
        for engine in range(8):
            for reason in NATIVE_FINISH_REASONS if metric.endswith("request_success_total") else (None,):
                extra = f',finished_reason="{reason}"' if reason else ""
                value = completed if engine == 0 and reason in (None, "stop") else 0
                lines.append(f'{metric}{{engine="{engine}",model_name="qwen3-full"{extra}}} {value}')
    for metric in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
        for engine in range(8):
            lines.append(f'{metric}{{engine="{engine}",model_name="qwen3-full"}} 0')
    return "\n".join(lines)+"\n"


def test_native_completion_counts_match_without_pooling_timings(tmp_path):
    before, after = tmp_path / "before.txt", tmp_path / "after.txt"
    before.write_text(counter_text(0)); after.write_text(counter_text(2))
    result = audit_counters(before, after, 2, {"stop": 2})
    assert result["status"] == "matched"
    assert len(result["series"]) == 80
    assert all("labels" in row for row in result["series"])


def test_missing_or_reset_native_counters_never_assumed_zero(tmp_path):
    before, after = tmp_path / "before.txt", tmp_path / "after.txt"
    after.write_text(counter_text(2))
    assert audit_counters(before, after, 2, {"stop": 2})["status"] == "unavailable"
    before.write_text(counter_text(3))
    result = audit_counters(before, after, 2, {"stop": 2})
    assert result["status"] == "unreconciled" and any("reset" in x for x in result["issues"])


@pytest.mark.parametrize("old,new", [('engine="7"', 'engine="9"'), ('model_name="qwen3-full"', 'model_name="wrong-model"')])
def test_equal_native_totals_cannot_hide_wrong_engine_or_model_scope(tmp_path, old, new):
    before, after = tmp_path / "before.txt", tmp_path / "after.txt"
    before.write_text(counter_text(0).replace(old, new))
    after.write_text(counter_text(2).replace(old, new))
    result = audit_counters(before, after, 2, {"stop": 2})
    assert result["status"] == "unreconciled"
    assert all(v == 2 for v in result["completion_totals"].values())
    assert any("expected eight-engine scope" in issue for issue in result["issues"])


def test_overlap_between_stages_fails_partition(tmp_path):
    splits = {"smoke": {"ids": [str(i) for i in range(8)]},
              "calibration": {"ids": [str(i) for i in range(8, 40)]},
              "external": {"ids": [str(i) for i in range(39, 63)]}}
    (tmp_path / "pilot-splits.json").write_text(json.dumps(splits))
    report = audit_cohort(tmp_path, {"dataset": "wildchat"})
    assert report["status"] == "failed" and "disjoint" in report["reason"]


def test_native_log_constructor_is_never_executed(tmp_path):
    (tmp_path / "off").mkdir()
    (tmp_path / "off/server-0.log").write_text("non-default args: {'evil': __import__('os').system('false')}\n")
    report = paired_configuration(tmp_path)
    assert report["status"] == "failed" and "unsupported constructor" in report["reason"]


def build_evidence_fixture(*, include_cache=True):
    result = {}
    for arm in ("off", "on"):
        result[arm] = {}
        for role, host, module_hash in (("primary", "node-a", "a"*64), ("worker", "node-b", "b"*64)):
            value = {"hostname": host, "cuda_compiler_toolchain": {
                "cuda_home": "/tools/cuda", "install_manifest_sha256": "c"*64,
                "critical_file_sha256": {name: "d"*64 for name in CUDA_CRITICAL_FILES},
                "nvcc_version": "V12.9.86", "runtime_library_environment_changed": False}}
            if include_cache:
                value["flashinfer_prebuild"] = {
                    "workspace_base": f"/cache/{host}", "jit_dir": f"/cache/{host}/ops",
                    "source_sha256": {name: "e"*64 for name in FLASHINFER_SOURCES},
                    "sampling_module_sha256": {"sampling/sampling.so": module_hash},
                    "checks": [{"exact_mask_match": True, "top_k": [5]},
                               {"exact_mask_match": True, "top_k": [1, 5]},
                               {"exact_sample_match": True, "operation": "top_k_top_p_sampling_from_logits"},
                               {"exact_sample_match": True, "operation": "top_p_sampling_from_probs"}],
                    "report_path": f"/results/{arm}/{host}.json", "report_sha256": ("f" if arm == "off" else "0")*64,
                    "get_sampling_module_elapsed_s": 0.1 if arm == "off" else 0.2,
                    "step_id": "1" if arm == "off" else "2"}
            result[arm][role] = value
    return result


def test_native_build_pair_allows_node_specific_modules_and_arm_specific_bookkeeping():
    result = paired_build_evidence(build_evidence_fixture())
    assert result["cuda_compiler_toolchain"]["status"] == "matched"
    assert result["flashinfer_prebuild"]["status"] == "matched"
    nodes = result["flashinfer_prebuild"]["nodes"]
    assert nodes[0]["sampling_module_sha256"] != nodes[1]["sampling_module_sha256"]


def test_legacy_missing_cache_evidence_is_explicit_and_never_inferred():
    result = paired_build_evidence(build_evidence_fixture(include_cache=False))
    assert result["cuda_compiler_toolchain"]["status"] == "matched"
    assert result["flashinfer_prebuild"]["status"] == "not_recorded_in_either_arm"


@pytest.mark.parametrize("corruption,reason", [("cache_hash", "FlashInfer cache evidence differs"),
                                               ("compiler_hash", "CUDA compiler/install-manifest evidence differs"),
                                               ("missing_cache", "missing from one arm")])
def test_native_build_pair_rejects_same_node_cache_or_compiler_corruption(corruption, reason):
    evidence = build_evidence_fixture()
    changed = evidence["on"]["primary"]
    if corruption == "cache_hash":
        changed["flashinfer_prebuild"]["sampling_module_sha256"]["sampling/sampling.so"] = "9"*64
    elif corruption == "compiler_hash":
        changed["cuda_compiler_toolchain"]["critical_file_sha256"]["bin/ptxas"] = "9"*64
    else:
        del changed["flashinfer_prebuild"]
    with pytest.raises(AuditError, match=reason):
        paired_build_evidence(evidence)

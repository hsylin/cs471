from datetime import datetime, timezone
import json

from full_serving.audit_runtime import audit_cohort, parse_logs, read_phase


def test_profiles_and_launches_never_become_commits(tmp_path):
    p = tmp_path / "server-0.log"
    p.write_text("INFO 11-10 15:00:00 Rearranging experts (async mode) (profile)...\n"
                 "INFO 11-10 15:00:01 Rearranged experts (profile) in 1.8 s.\n"
                 "INFO 11-10 15:01:00 Rearranging experts (async mode) ...\n")
    d = parse_logs([p], {})
    assert d["ordinary_async_launch_log_entries"] == 1
    assert len(d["profile_rearrangement_entries"]) == 1
    assert d["normal_async_commits_observed"] is None
    assert d["committed_state_consistency_verified"] is False


def test_shutdown_error_is_separate_from_prior_inference_error(tmp_path):
    p = tmp_path / "server-0.log"
    p.write_text("ERROR 11-10 15:00:00 worker inference failed\n"
                 "INFO 11-10 15:01:00 [shutdown] shutdown triggered\n"
                 "INFO 11-10 15:01:00 [shutdown] mode=abort timeout=0s\n"
                 "WARNING 11-10 15:01:00 force killing remaining process EngineCore_DP0\n"
                 "ERROR 11-10 15:01:00 EngineDeadError\n")
    d = parse_logs([p], {})
    assert d["errors_before_shutdown_marker"] == 1
    assert [e["after_shutdown_marker_in_same_file"] for e in d["errors"]] == [False, True]
    assert d["shutdown_abort_zero_timeout"] and d["forced_engine_shutdown"]


def test_jit_phase_uses_wall_time_not_monotonic_or_timezone_guess(tmp_path):
    p = tmp_path / "server-0.log"
    p.write_text("WARNING 11-10 15:00:05 Triton kernel JIT compilation during inference: fused_moe_kernel. This causes a latency spike\n")
    start = datetime(2026, 11, 10, 14, 0, 0, tzinfo=timezone.utc).timestamp()
    d = parse_logs([p], {"smoke": (start, start + 10), "external": (start + 30, start + 40)})
    assert d["jit_during_inference"][0]["compatible_phases"] == ["smoke"]
    assert d["jit_during_inference"][0]["utc_second"] == "2026-11-10T14:00:05+00:00"


def test_partial_files_and_missing_arm_do_not_complete_cohort(tmp_path):
    path = tmp_path / "off" / "smoke.jsonl"
    path.parent.mkdir()
    path.write_text('{"type": "request"')
    assert read_phase(path)[0]["available"] is False
    (tmp_path / "status.json").write_text(json.dumps({"state": "pilot_completed"}))
    d = audit_cohort(tmp_path)
    assert d["completed_pair"] is False
    assert d["pairs"]["external"]["input_calendar_client_hashes_match"] is None


def test_jsonl_embedded_unicode_separators_are_not_record_boundaries(tmp_path):
    path = tmp_path / "smoke.jsonl"
    rows = [{"type": "run_header", "note": "line\u2028paragraph\u2029separator"},
            {"type": "run_summary", "valid": True}]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    result, _ = read_phase(path)
    assert result["available"] is True
    assert result["complete"] is True
    assert result["valid"] is True


def test_failed_request_with_null_usage_is_reported_without_false_completion(tmp_path):
    path = tmp_path / "smoke.jsonl"
    rows = [{"type": "run_header", "request_count": 1},
            {"type": "request", "success": False, "saw_done": False, "usage": None,
             "expected_input_tokens": 20},
            {"type": "run_summary", "valid": True, "offered_requests": 1,
             "completed_requests": 1, "failed_requests": 0}]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    result, _ = read_phase(path)
    assert result["valid"] is True  # Preserve the contradictory reported flag.
    assert result["observed_request_completion_coherent"] is False
    assert result["failures"] == 1
    assert result["generated_tokens"] is None
    assert result["all_prompt_counts_match"] is False


def test_per_rank_capacity_and_weight_backend_preserved(tmp_path):
    p = tmp_path / "server-0.log"
    p.write_text("(EngineCore_DP0 pid=1) INFO 11-10 15:00:00 GPU KV cache size: 744,464 tokens\n"
                 "(EngineCore_DP1 pid=2) INFO 11-10 15:00:00 GPU KV cache size: 713,312 tokens\n"
                 "INFO 11-10 15:00:00 Using ['PYNCCL'] all-reduce backends for group 'eplb:0'\n"
                 "INFO 11-10 15:00:01 Initialized EPLB communicator: TorchDistGlooStagedEplbCommunicator.\n")
    d = parse_logs([p], {})
    assert d["kv_tokens_per_dp_rank"] == {"0": 744464, "1": 713312}
    assert d["selected_weight_communicators"] == ["TorchDistGlooStagedEplbCommunicator"]

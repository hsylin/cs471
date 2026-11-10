"""Pilot admission must be complete and correct, without filtering slow methods."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from full_serving import custom_admission as admission
from full_serving.audit_pilots import AuditError, client_hash
from full_serving.analyze import summarize_run
from full_serving.data import digest, file_digest
from full_serving.resident_plugin import CONFIG_SCHEMA, MODEL_REVISION, PIN, ROOT, balanced_map
from test_full_serving_audit_pilots import request_fixture, counter_text


@pytest.mark.parametrize("body,expected", [
    ("ERROR 11-10 19:08:59 inference failure\nINFO 11-10 19:08:59 [shutdown] stopping\n", 1),
    ("INFO 11-10 19:08:59 [shutdown] stopping\nERROR 11-10 19:08:59 teardown failure\n", 0),
    ("ERROR 11-10 19:08:59 inference failure\nTraceback\nINFO 11-10 19:08:59 [shutdown] stopping\n", 2),
    ("INFO 11-10 19:08:59 [shutdown] stopping\nTraceback\n", 0),
])
def test_shutdown_same_second_uses_per_file_order_not_timestamp_equality(body, expected):
    assert admission.errors_before_shutdown({"server-0.log": body}) == expected


def test_shutdown_in_other_file_does_not_clear_ambiguous_same_second_error():
    text = {"server-0.log": "INFO 11-10 19:08:59 [shutdown] stopping\n",
            "server-1.err": "ERROR 11-10 19:08:59 worker failure\n"}
    assert admission.errors_before_shutdown(text) == 1
    text["server-1.err"] = "ERROR 11-10 19:09:00 later teardown failure\n"
    assert admission.errors_before_shutdown(text) == 0


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def http_fixture(tmp_path, phase=0, latency_factor=1):
    records, private = [], []
    for i in range(8):
        row, _, prompt = request_fixture(i)
        offset = i*500_000_000
        for key in ("scheduled_ns", "arrival_wakeup_ns", "semaphore_acquired_ns", "send_ns", "headers_ns",
                    "first_content_ns", "last_content_ns", "done_ns", "end_ns"):
            row[key] += offset
        for chunk in row["chunks"]:
            chunk["received_ns"] += offset
        row["type"] = "request"
        row["arrival_offset_ns"] = offset
        row["diagnostic_requested"] = phase == 0
        if phase == 0:
            row.update(diagnostic_passed=True, prompt_token_ids_count=4,
                       prompt_token_ids_sha256=row["expected_prompt_token_ids_sha256"], completion_token_ids_count=3,
                       diagnostic_logprob_count=3)
        payload = {"model": "qwen3-full", "messages": prompt["messages"], "stream": True,
                   "stream_options": {"include_usage": True}, "temperature": 0, "max_tokens": 2048,
                   "n": 1, "chat_template_kwargs": {"enable_thinking": False}}
        if phase == 0:
            payload.update(logprobs=True, top_logprobs=0, return_token_ids=True)
        row["payload_sha256"] = client_hash(payload)
        records.append(row); private.append(prompt)
    ids = [p["id"] for p in private]
    cohort = {"smoke_request_ids": ids, "private_files": {"smoke.jsonl": "a"*64}}
    calendar = {"kind": "synthetic_regular_pilot", "rate_requests_per_s": 2.0,
                "arrival_offsets_ns": [i*500_000_000 for i in range(8)], "request_ids": ids}
    header = {"type": "run_header", "arm": "resident", "request_count": 8, "max_tokens": 2048,
              "max_model_len": 32768, "temperature": 0, "enable_thinking": False, "ignore_eos": False,
              "max_inflight": 64, "model": "qwen3-full", "logprobs": phase == 0,
              "diagnostic_return_token_ids": phase == 0, "requests_sha256": "a"*64,
              "request_ids_sha256": client_hash(ids), "client_source_sha256": "b"*64,
              "calendar_sha256": hashlib.sha256(json.dumps(calendar, sort_keys=True).encode()).hexdigest()}
    last = records[-1]["end_ns"]
    timing = {"epoch_monotonic_ns": 100, "client_finished_ns": last+10, "last_request_end_ns": last,
              "offered_horizon_ns": 3_500_000_000, "completion_ns": last, "completed_elapsed_ns": last-100,
              "drain_ns": 160}
    summary = {"type": "run_summary", **timing, **summarize_run(records, timing)}
    path = tmp_path/f"wildchat-pass{phase}.jsonl"
    rows = [header, *records, summary]
    path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows))
    (tmp_path/f"wildchat-pass{phase}-metrics-before.txt").write_text(counter_text(0))
    (tmp_path/f"wildchat-pass{phase}-metrics-after.txt").write_text(counter_text(8))
    return path, rows, cohort, private


@pytest.mark.parametrize("phase", [0, 1])
def test_http_raw_admission_preserves_private_unicode_and_both_instrumentation_modes(tmp_path, phase):
    _, _, cohort, private = http_fixture(tmp_path, phase)
    report = admission.http_phase(tmp_path, "wildchat", phase, "resident", cohort, private, "b"*64)["public"]
    assert report["requests"] == 8 and report["private_pilot_payloads_verified"]
    assert "PRIVATE" not in json.dumps(report)


@pytest.mark.parametrize("corruption", ["summary", "arrival", "private", "diagnostic"])
def test_http_success_flags_cannot_hide_raw_corruption(tmp_path, corruption):
    path, rows, cohort, private = http_fixture(tmp_path)
    if corruption == "summary": rows[-1]["response_from_schedule_ms"]["mean"] += 1
    elif corruption == "arrival": rows[1]["arrival_offset_ns"] = 1
    elif corruption == "private": private[0]["messages"][0]["content"] = "changed private input"
    else: rows[1]["diagnostic_passed"] = False
    path.write_text("".join(json.dumps(r)+"\n" for r in rows))
    with pytest.raises(AuditError):
        admission.http_phase(tmp_path, "wildchat", 0, "resident", cohort, private, "b"*64)


def kernel_fixture(tmp_path):
    value = {"schema": "full_resident_kernel_gate_v1", "passed": True, "native_recording_unchanged_zero": True,
             "gpu": "NVIDIA H100", "torch": "2.13.0+cu129",
             "source_sha256": {name: file_digest(ROOT/"full_serving"/name)
                               for name in ("resident_policy.py", "resident_gpu.py", "check_resident_kernel.py")},
             "cases": [{"rows": n, "dtype": dtype, "mode": mode, "cpu_oracle_exact": True,
                        "padding_and_conservation": True, "changed_input_graph_replays": 2 if n else 0,
                        "native_hash_exact": True if mode == "native_hash" else None}
                       for n in admission.SHAPES for dtype in ("torch.int32", "torch.int64")
                       for mode in ("resident_v1", "native_hash")]}
    sources = {"files": {"full_serving/"+name: sha for name, sha in value["source_sha256"].items()}}
    return write(tmp_path/"kernel.json", value), value, sources


@pytest.mark.parametrize("corruption", [None, "missing_shape", "graph", "source"])
def test_kernel_gate_requires_complete_current_source_graph_evidence(tmp_path, corruption):
    path, value, sources = kernel_fixture(tmp_path)
    if corruption == "missing_shape": value["cases"].pop()
    elif corruption == "graph": value["cases"][-1]["changed_input_graph_replays"] = 0
    elif corruption == "source": value["source_sha256"]["resident_gpu.py"] = "0"*64
    write(path, value)
    if corruption:
        with pytest.raises(AuditError): admission.kernel_gate(path, sources)
    else:
        assert admission.kernel_gate(path, sources)["cases"] == 44


def configuration_fixture(tmp_path):
    paths = {name: tmp_path/name for name in ("kernel", "diagnostic", "off", "resident", "placement_only",
                  "diagnostic_config", "resident_config", "placement_config", "sources", "suite_manifest", "private_pilot_root")}
    config = {"schema": CONFIG_SCHEMA, "mode": "resident_v1", "vllm_commit": PIN, "model_revision": MODEL_REVISION,
              "physical_to_logical_map": balanced_map(), "startup_diagnostic": False}
    write(paths["resident_config"], config)
    write(paths["diagnostic_config"], {**config, "startup_diagnostic": True})
    write(paths["placement_config"], {**config, "mode": "native_hash"})
    names = ["full_serving/"+name for name in ("resident_plugin.py", "resident_policy.py", "resident_gpu.py",
             "resident_diagnostic.py", "custom_entrypoint.py")]
    names += ["full_serving/bootstrap/cs471_resident-0.1.dist-info/"+name for name in ("METADATA", "entry_points.txt")]
    write(paths["sources"], {"schema": "cs471_full_resident_sources_v1", "files": {name: file_digest(ROOT/name) for name in names}})
    cohorts = []
    for dataset in admission.DATASETS:
        rows = [{"id": f"{dataset}-{i}", "messages": [{"role": "user", "content": "private pilot only"}]} for i in range(8)]
        p = paths["private_pilot_root"]/dataset/"smoke.jsonl"
        p.parent.mkdir(parents=True);p.write_text("".join(json.dumps(r)+"\n" for r in rows))
        cohorts.append({"dataset": dataset, "smoke_request_ids": [r["id"] for r in rows], "private_files": {"smoke.jsonl": file_digest(p)}})
    suite = {"cohorts": cohorts};suite["manifest_sha256"] = digest(suite);write(paths["suite_manifest"], suite)
    return paths


def mock_completed_evidence(monkeypatch, paths, *, absent_timed=False, slow_resident=True):
    monkeypatch.setattr(admission, "kernel_gate", lambda *args: {"cases": 44})
    monkeypatch.setattr(admission, "loaded_weights", lambda *args: {"full_replica_weights": "unit-test separate checker"})
    def runtime(path, arm, *args):
        if absent_timed and path != paths["diagnostic"]:
            raise FileNotFoundError()
        flags = {} if arm == "off" else {"enable_eplb": True, "eplb_config": {"num_redundant_experts": 32}}
        return {"args": flags, "engine": "same native engine", "identities": ["same eight GPUs and checkpoint"],
                "preflights": {}, "client_sha256": "same client", "public": {"checked": True}}
    monkeypatch.setattr(admission, "pilot_runtime", runtime)
    monkeypatch.setattr(admission, "paired_build_evidence", lambda *args: {"flashinfer_prebuild": {"status": "matched"}})
    def phase(path, dataset, phase, arm, *args):
        return {"header": {}, "records": [{"request_id": f"{dataset}-{i}"} for i in range(8)],
                "public": {"mean_response_ms": 100 if arm == "resident" and slow_resident else 1,
                           "completed_elapsed_s": 10 if arm == "resident" and slow_resident else 1}}
    monkeypatch.setattr(admission, "http_phase", phase)


def test_completed_correct_but_slower_method_passes_admission_without_win_filter(tmp_path, monkeypatch):
    paths = configuration_fixture(tmp_path)
    mock_completed_evidence(monkeypatch, paths)
    result = admission.build_admission(paths)
    assert result["status"] == "passed" and result["performance_gate_passed"]
    assert result["gates"]["timed_comparison"]["win_filter_applied"] is False
    assert result["custom_config_sha256"] == file_digest(paths["resident_config"])
    assert "private pilot only" not in json.dumps(result)


def test_completed_correctness_alone_does_not_admit_missing_timed_pilots(tmp_path, monkeypatch):
    paths = configuration_fixture(tmp_path)
    mock_completed_evidence(monkeypatch, paths, absent_timed=True)
    result = admission.build_admission(paths)
    assert result["status"] == "pending" and result["correctness_gate_passed"] and not result["performance_gate_passed"]


def test_completed_off_progress_is_retained_while_resident_is_pending(tmp_path, monkeypatch):
    paths = configuration_fixture(tmp_path)
    mock_completed_evidence(monkeypatch, paths)
    complete_runtime = admission.pilot_runtime
    def pending_resident(path, *args):
        if path == paths["resident"]:
            raise admission.Pending("pilot has not completed")
        return complete_runtime(path, *args)
    monkeypatch.setattr(admission, "pilot_runtime", pending_resident)
    result = admission.build_admission(paths)
    assert result["status"] == "pending" and not result["performance_gate_passed"]
    assert result["timed_pilot_progress"]["off"]["status"] == "passed"
    assert result["timed_pilot_progress"]["off"]["completed_requests"] == 80
    assert result["timed_pilot_progress"]["resident"]["status"] == "pending"
    assert result["timed_pilot_progress"]["placement_only"]["status"] == "pending"


def test_kernel_failure_cannot_be_overridden_by_successful_http(tmp_path, monkeypatch):
    paths = configuration_fixture(tmp_path)
    mock_completed_evidence(monkeypatch, paths)
    def failure(*args): raise AuditError("kernel mismatch")
    monkeypatch.setattr(admission, "kernel_gate", failure)
    result = admission.build_admission(paths)
    assert result["status"] == "failed" and not result["correctness_gate_passed"] and not result["performance_gate_passed"]


def test_timed_startup_instrumentation_is_rejected_before_any_evidence_read(tmp_path):
    paths = configuration_fixture(tmp_path)
    value = json.loads(paths["resident_config"].read_bytes());value["startup_diagnostic"] = True
    write(paths["resident_config"], value)
    result = admission.build_admission(paths)
    assert result["status"] == "failed" and "instrumentation" in result["reason"]


@pytest.mark.parametrize("corruption", [None, "replica", "tolerance", "failed_case"])
def test_loaded_weights_reconcile_all_copies_and_preserve_fixed_numerical_checks(tmp_path, corruption):
    mapping = balanced_map()
    flags = {k: True for k in ("logical_identity_exact", "physical_cpu_oracle_exact",
             "one_home_policy_checked_by_cpu_oracle", "routing_weights_unchanged", "token_conservation_exact",
             "gathered_rank_agreement", "native_expert_outputs_close", "direct_fp32_outputs_close")}
    for rank in range(8):
        weights = []
        for layer in range(48):
            for slot in range(rank*20, (rank+1)*20):
                logical = mapping[slot]
                h = hashlib.sha256(f"{layer}:{logical}".encode()).hexdigest()
                weights.append({"layer_index": layer, "layer": f"model.layers.{layer}.mlp.experts",
                                "physical_slot": slot, "logical_expert": logical, "w13_sha256": h, "w2_sha256": h})
        cases = [{"layer_index": layer, "case": name, "rows_per_rank": rows, "real_rows_per_rank": real,
                  "atol": .02, "rtol": .03, **flags}
                 for layer in (0, 23, 47) for name, (rows, real) in admission.CASES.items()]
        value = {"status": "passed", "ep_rank": rank, "config_sha256": "a"*64,
                 "source_manifest_sha256": "b"*64, "source_sha256": file_digest(ROOT/"full_serving/resident_diagnostic.py"),
                 "reference_matmul_precision": "highest", "local_weight_records": weights,
                 "native_expert_numerical_checks": cases}
        if rank == 0:
            if corruption == "replica": weights[16]["w13_sha256"] = "0"*64
            elif corruption == "tolerance": cases[0]["rtol"] = 1
            elif corruption == "failed_case": cases[0]["direct_fp32_outputs_close"] = False
        write(tmp_path/f"resident-startup-diagnostic-rank{rank}.json", value)
    if corruption:
        with pytest.raises((AuditError, AssertionError)):
            admission.loaded_weights(tmp_path, "a"*64, "b"*64)
    else:
        result = admission.loaded_weights(tmp_path, "a"*64, "b"*64)
        assert result["full_replica_weights"]["physical_layer_experts"] == 7680
        assert result["full_replica_weights"]["duplicated_logical_layer_experts"] == 1536

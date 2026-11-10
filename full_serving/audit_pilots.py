"""Read-only integrity audit of completed full-model pilot records.

This audits recorded identities, timing arithmetic, pairing and metric counts.
It neither runs a model nor interprets an integrity pass as a speedup claim.
Private input checking is opt-in; private text/token arrays never enter output.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

from full_serving.analyze import summarize_run
from full_serving.data import MODEL_REVISION, digest, file_digest

DATASETS = ("wildchat", "sharegpt", "instructcoder", "numinamath", "longbench")
STAGES = {"smoke": 8, "calibration": 32, "external": 24}
PAIR_FIELDS = ("requests_sha256", "calendar_sha256", "request_ids_sha256", "request_count",
               "model", "max_tokens", "max_model_len", "temperature", "enable_thinking",
               "ignore_eos", "max_inflight", "http_timeout_s", "logprobs", "diagnostic_return_token_ids",
               "client_source_sha256", "python_version", "aiohttp_version", "endpoint",
               "block_id", "load_id", "pair_id", "arrival_kind", "synthetic_rate_requests_per_s")
COUNT_METRICS = ("vllm:request_success_total", "vllm:e2e_request_latency_seconds_count",
                 "vllm:request_queue_time_seconds_count", "vllm:request_inference_time_seconds_count",
                 "vllm:request_prefill_time_seconds_count", "vllm:request_decode_time_seconds_count")
NATIVE_FINISH_REASONS = ("stop", "length", "abort", "error", "repetition")


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def integer(value):
    return type(value) is int


def client_hash(value):
    # Client input/payload hashes preserve Unicode; preparation's digest uses
    # ASCII escapes. Token-ID hashes agree because integer arrays are ASCII.
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def same(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(actual) == set(expected) and all(same(actual[k], v) for k, v in expected.items())
    if isinstance(expected, float):
        return isinstance(actual, (int, float)) and not isinstance(actual, bool) and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9)
    return actual == expected


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    raw = Path(path).read_bytes()
    require(raw.endswith(b"\n"), "JSONL snapshot lacks a final newline")
    return raw, [json.loads(line) for line in raw.split(b"\n")[:-1] if line]


def audit_request(row, header, epoch, finished, private=None):
    rid = row["request_id"]
    def check(condition, message):
        require(condition, f"{rid}: {message}")
    check(row.get("success") is True and row.get("http_status") == 200 and row.get("saw_done") is True,
          "unsuccessful HTTP/SSE completion")
    check(row.get("error_type") is None and row.get("error_message") is None, "completion retains an error")
    ordered = [row.get(k) for k in ("scheduled_ns", "arrival_wakeup_ns", "semaphore_acquired_ns", "send_ns", "headers_ns", "done_ns", "end_ns")]
    check(all(integer(v) and v >= 0 for v in ordered) and ordered == sorted(ordered), "invalid request timestamp order")
    check(integer(row.get("arrival_offset_ns")) and row["arrival_offset_ns"] >= 0
          and row["scheduled_ns"] == epoch + row["arrival_offset_ns"], "absolute arrival identity differs")
    check(row["end_ns"] <= finished, "request ended after client completion")
    equations = {"client_schedule_lag_ns": row["arrival_wakeup_ns"]-row["scheduled_ns"],
                 "client_semaphore_wait_ns": row["semaphore_acquired_ns"]-row["arrival_wakeup_ns"],
                 "client_send_delay_ns": row["send_ns"]-row["scheduled_ns"],
                 "response_from_schedule_ns": row["end_ns"]-row["scheduled_ns"],
                 "response_from_send_ns": row["end_ns"]-row["send_ns"]}
    for key, expected in equations.items():
        check(integer(row.get(key)) and row[key] == expected, f"timing equation differs: {key}")
    usage = row.get("usage")
    check(isinstance(usage, dict) and all(integer(usage.get(k)) and usage[k] >= 0
          for k in ("prompt_tokens", "completion_tokens", "total_tokens")), "invalid token usage")
    check(usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"], "token total differs")
    check(usage["prompt_tokens"] == row["expected_input_tokens"] and row["expected_input_tokens"] > 0,
          "prompt token count differs")
    check(row["expected_input_tokens"] + header["max_tokens"] <= header["max_model_len"], "prompt/output context overflow")
    check(usage["completion_tokens"] <= header["max_tokens"], "generated output exceeds cap")
    check(row.get("finish_reason") in {"stop", "length"}, "unknown finish reason")
    if row["finish_reason"] == "length":
        check(usage["completion_tokens"] == header["max_tokens"], "length stop differs from declared output cap")
    chunks = row["chunks"]
    check(isinstance(chunks, list) and all(isinstance(c, dict) for c in chunks), "invalid chunk records")
    receipts = [c.get("received_ns") for c in chunks]
    check(all(integer(v) and row["headers_ns"] <= v <= row["done_ns"] for v in receipts)
          and receipts == sorted(receipts), "chunk receipt time outside response interval")
    check(all(integer(c.get("content_chars")) and integer(c.get("content_utf8_bytes"))
              and 0 <= c["content_chars"] <= c["content_utf8_bytes"] for c in chunks), "invalid content size")
    check(sum(c["content_chars"] for c in chunks) == row["output_chars"]
          and sum(c["content_utf8_bytes"] for c in chunks) == row["output_utf8_bytes"], "output size/chunk totals differ")
    nonempty = [c["received_ns"] for c in chunks if c["content_chars"]]
    first, last = (nonempty[0], nonempty[-1]) if nonempty else (None, None)
    check(row["first_content_ns"] == first and row["last_content_ns"] == last, "first/last content receipt differs")
    check(row["ttft_from_schedule_ns"] == (None if first is None else first-row["scheduled_ns"])
          and row["ttft_from_send_ns"] == (None if first is None else first-row["send_ns"]), "TTFT equation differs")
    tpot = (last-first)/(usage["completion_tokens"]-1) if first is not None and usage["completion_tokens"] > 1 else None
    check(same(row["tpot_stream_boundary_estimate_ns"], tpot), "stream-boundary TPOT equation differs")
    check(row.get("server_model") == header["model"], "served model identity differs")
    for key in ("input_sha256", "payload_sha256", "output_sha256", "expected_prompt_token_ids_sha256"):
        check(isinstance(row.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", row[key]), f"invalid hash: {key}")
    check(row.get("diagnostic_requested") is header["logprobs"], "diagnostic mode differs")
    if header["logprobs"]:
        check(row.get("diagnostic_passed") is True
              and row.get("prompt_token_ids_count") == usage["prompt_tokens"]
              and row.get("prompt_token_ids_sha256") == row["expected_prompt_token_ids_sha256"]
              and row.get("completion_token_ids_count") == usage["completion_tokens"]
              and row.get("diagnostic_logprob_count") == usage["completion_tokens"], "diagnostic count/hash checks differ")
    if private is not None:
        check(private["id"] == rid and private["input_tokens"] == row["expected_input_tokens"]
              and private["prompt_token_ids_sha256"] == row["expected_prompt_token_ids_sha256"], "admitted private metadata differs")
        check(client_hash(private["messages"]) == row["input_sha256"], "admitted full messages hash differs")
        payload = {"model": header["model"], "messages": private["messages"], "stream": True,
                   "stream_options": {"include_usage": True}, "temperature": 0, "max_tokens": header["max_tokens"],
                   "n": 1, "chat_template_kwargs": {"enable_thinking": False}}
        if header["logprobs"]:
            payload.update(logprobs=True, top_logprobs=0, return_token_ids=True)
        check(client_hash(payload) == row["payload_sha256"], "actual HTTP payload differs from admitted request")


def audit_run(path, stage, arm, split, calendar=None, private_rows=None):
    path = Path(path)
    if not path.exists():
        return {"status": "pending", "reason": "request records not available locally"}, None
    try:
        raw = path.read_bytes()
        if not raw.endswith(b"\n"):
            return {"status": "pending", "reason": "incomplete JSONL tail in local snapshot"}, None
        rows = [json.loads(line) for line in raw.split(b"\n")[:-1] if line]
        summaries = [r for r in rows if r.get("type") == "run_summary"]
        if not summaries:
            return {"status": "pending", "reason": "completion summary not available locally"}, None
        headers = [r for r in rows if r.get("type") == "run_header"]
        records = sorted((r for r in rows if r.get("type") == "request"), key=lambda r: r["request_index"])
        require(len(headers) == len(summaries) == 1 and len(rows) == len(records)+2, "unexpected record type/header/summary count")
        header, summary = headers[0], summaries[0]
        count = STAGES[stage]
        require(header["arm"] == arm and header["request_count"] == count and len(records) == count, "arm or request count differs")
        require([r["request_index"] for r in records] == list(range(count)), "missing/duplicate request indices")
        require([r["request_id"] for r in records] == split["ids"], "request order/membership differs from pilot partition")
        require(header["request_ids_sha256"] == client_hash(split["ids"])
                and header["requests_sha256"] == split["request_file_sha256"], "request file/identity hash differs")
        require(header["max_tokens"] == 2048 and header["max_model_len"] == 32768
                and header["temperature"] == 0 and header["enable_thinking"] is False
                and header["ignore_eos"] is False and header["max_inflight"] == 64
                and header["model"] == "qwen3-full", "generation/context/model contract differs")
        require(header["logprobs"] is (stage == "smoke") and header["diagnostic_return_token_ids"] is (stage == "smoke"),
                "incorrect smoke diagnostic configuration")
        if stage == "external":
            require(calendar is not None, "external calendar unavailable")
            values = calendar["value"]
            require(header["calendar_sha256"] == calendar["sha256"] and values["request_ids"] == split["ids"],
                    "external calendar file/identity differs")
        else:
            rate = 2.0 if stage == "smoke" else 0.0
            values = {"kind": "synthetic_regular_pilot" if rate else "synthetic_pilot_burst",
                      "rate_requests_per_s": rate, "arrival_offsets_ns": [round(i/rate*1e9) if rate else 0 for i in range(count)],
                      "request_ids": split["ids"]}
            require(header["calendar_sha256"] == hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest(),
                    "synthetic calendar hash differs")
        offsets = values["arrival_offsets_ns"]
        horizon = values.get("horizon_ns", offsets[-1])
        require([r["arrival_offset_ns"] for r in records] == offsets, "absolute arrivals differ from declared calendar")
        require(offsets == sorted(offsets) and all(integer(v) and v >= 0 for v in offsets)
                and integer(horizon) and horizon >= offsets[-1], "invalid arrival calendar")
        epoch, finished = summary["epoch_monotonic_ns"], summary["client_finished_ns"]
        require(integer(epoch) and integer(finished), "invalid run timing origin")
        by_id = None if private_rows is None else {r["id"]: r for r in private_rows}
        for row in records:
            audit_request(row, header, epoch, finished, None if by_id is None else by_id[row["request_id"]])
        last_end = max(r["end_ns"] for r in records)
        completion = max(last_end, epoch+horizon)
        equations = {"last_request_end_ns": last_end, "offered_horizon_ns": horizon,
                     "completion_ns": completion, "completed_elapsed_ns": completion-epoch,
                     "drain_ns": max(0, last_end-epoch-horizon)}
        require(finished >= completion, "client did not wait through drain/offered horizon")
        for key, expected in equations.items():
            require(summary.get(key) == expected, f"run timing equation differs: {key}")
        recomputed = summarize_run(records, summary)
        require(recomputed["valid"] is True, "raw run is not a valid completed replay")
        for key, expected in recomputed.items():
            require(same(summary.get(key), expected), f"stored summary differs from raw records: {key}")
        report = {"status": "passed", "record_file_sha256": digest(raw), "requests": count,
                  "input_tokens": {"minimum": min(r["expected_input_tokens"] for r in records),
                                   "maximum": max(r["expected_input_tokens"] for r in records)},
                  "output_cap": 2048, "output_cap_completions": summary["output_cap_completions"],
                  "completed_elapsed_ns": summary["completed_elapsed_ns"],
                  "client_finished_after_epoch_ns": finished-epoch,
                  "client_tail_after_completion_ns": finished-completion,
                  "admitted_private_messages_checked": private_rows is not None,
                  "raw_timing_identities_checked": True, "stored_summary_recomputed": True}
        internal = {"header": header, "finish_counts": dict(Counter(r["finish_reason"] for r in records)),
                    "rows": [{k: r[k] for k in ("request_id", "input_sha256", "payload_sha256", "expected_input_tokens",
                               "expected_prompt_token_ids_sha256", "arrival_offset_ns")} for r in records]}
        return report, internal
    except (KeyError, TypeError, ValueError, OSError) as exc:
        return {"status": "failed", "reason": str(exc) if isinstance(exc, AuditError) else type(exc).__name__}, None


def parse_counts(text):
    result = {}
    names = set(COUNT_METRICS) | {"vllm:num_requests_running", "vllm:num_requests_waiting"}
    pattern = re.compile(r'^(\S+?)(\{.*\})?\s+([-+\d.eE]+)(?:\s+.*)?$')
    labels_pattern = re.compile(r'\s*([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"\s*(,|$)')
    for line in text.split("\n"):
        match = pattern.match(line)
        if not match or match[1] not in names:
            continue
        labels, position = {}, 0
        body = (match[2] or "{}")[1:-1]
        while position < len(body):
            item = labels_pattern.match(body, position)
            require(item is not None and item[1] not in labels, "invalid/duplicate metric labels")
            labels[item[1]] = json.loads('"'+item[2]+'"')
            position = item.end()
        value = float(match[3])
        require(math.isfinite(value) and value >= 0 and value.is_integer(), "invalid native completion/gauge count")
        key = (match[1], tuple(sorted(labels.items())))
        require(key not in result, "duplicate native count series")
        result[key] = int(value)
    return result


def audit_counters(before, after, expected, finish_counts):
    before, after = Path(before), Path(after)
    if not before.exists() or not after.exists():
        return {"status": "unavailable", "reason": "one or both native snapshots unavailable; no zero baseline assumed"}
    try:
        b, a = parse_counts(before.read_text()), parse_counts(after.read_text())
        series, totals, reasons, issues = [], {}, Counter(), []
        for name in (*COUNT_METRICS, "vllm:num_requests_running", "vllm:num_requests_waiting"):
            label_sets = []
            for engine in range(8):
                for reason in NATIVE_FINISH_REASONS if name == "vllm:request_success_total" else (None,):
                    labels = {"engine": str(engine), "model_name": "qwen3-full"}
                    if reason is not None:
                        labels["finished_reason"] = reason
                    label_sets.append((name, tuple(sorted(labels.items()))))
            expected_scope = set(label_sets)
            if {key for key in b if key[0] == name} != expected_scope or {key for key in a if key[0] == name} != expected_scope:
                issues.append(f"native labels differ from expected eight-engine scope: {name}")
        for key in sorted(set(b) | set(a)):
            name, labels = key
            if name not in COUNT_METRICS:
                continue
            if key not in b or key not in a:
                issues.append(f"unmatched native counter labels: {name}")
                continue
            delta = a[key]-b[key]
            if delta < 0:
                issues.append(f"native counter reset: {name}")
                continue
            series.append({"metric": name, "labels": dict(labels), "before": b[key], "after": a[key], "delta": delta})
            totals[name] = totals.get(name, 0)+delta
            if name == "vllm:request_success_total":
                reasons[dict(labels).get("finished_reason", "missing")] += delta
        reasons = {k: v for k, v in reasons.items() if v}
        for name in COUNT_METRICS:
            if totals.get(name) != expected:
                issues.append(f"{name}: observed {totals.get(name)}, expected {expected}")
        if reasons != finish_counts:
            issues.append("native finish-reason totals differ from client records")
        active = [{"metric": k[0], "labels": dict(k[1]), "value": value} for k, value in a.items()
                  if k[0] in {"vllm:num_requests_running", "vllm:num_requests_waiting"}]
        if len(active) != 16 or any(r["value"] for r in active):
            issues.append("after-snapshot running/waiting gauges do not show eight drained engines")
        return {"status": "matched" if not issues else "unreconciled", "issues": issues,
                "series": series, "completion_totals": totals, "finish_reason_totals": reasons,
                "after_active_gauges": active, "before_sha256": file_digest(before), "after_sha256": file_digest(after),
                "scope": "Counts summed across distinct engine labels; no pooled timing mean. Snapshot publication lag can cause an unreconciled count; no commit completion inference."}
    except (AuditError, OSError, ValueError) as exc:
        return {"status": "unreconciled", "reason": str(exc) if isinstance(exc, AuditError) else type(exc).__name__}


def paired_configuration(directory):
    """Compare logged native arguments without executing any log content."""
    try:
        configs, fingerprints = {}, {}
        for arm in ("off", "on"):
            path = directory / arm / "server-0.log"
            if not path.exists():
                return {"status": "pending", "reason": "server configuration log unavailable"}
            entries = [line.split("non-default args: ", 1)[1] for line in path.read_text().split("\n") if "non-default args: " in line]
            require(len(entries) == 1, "ambiguous or absent native argument record")
            tree = ast.parse(entries[0], mode="eval")
            class SafeConfig(ast.NodeTransformer):
                def visit_Call(self, node):
                    require(isinstance(node.func, ast.Name) and node.func.id == "EPLBConfig" and not node.args
                            and all(k.arg is not None for k in node.keywords), "unsupported constructor in native configuration")
                    return ast.Dict(keys=[ast.Constant(k.arg) for k in node.keywords], values=[k.value for k in node.keywords])
            configs[arm] = ast.literal_eval(SafeConfig().visit(tree))
            require(configs[arm].get("data_parallel_size") == 8
                    and configs[arm].get("data_parallel_size_local") == 4
                    and configs[arm].get("tensor_parallel_size", 1) == 1
                    and configs[arm].get("enable_expert_parallel") is True
                    and configs[arm].get("dtype") == "bfloat16"
                    and configs[arm].get("max_model_len") == 32768
                    and str(configs[arm].get("model", "")).endswith(MODEL_REVISION),
                    "native model/parallel/context configuration differs from scope")
            runtime = []
            for role in ("primary", "worker"):
                p = directory / arm / f"preflight-{role}.json"
                if not p.exists():
                    return {"status": "pending", "reason": "runtime preflight unavailable"}
                value = read_json(p)
                require(value.get("model_revision") == MODEL_REVISION
                        and value.get("vllm_commit") == "db9527a46873454610df6dbedf79a36d6bf1a7f6"
                        and len(value.get("devices", [])) == 4
                        and all(d.get("name") == "NVIDIA H100" for d in value["devices"]),
                        "runtime checkpoint/vLLM/H100 configuration differs from scope")
                runtime.append({k: value[k] for k in ("hostname", "model_revision", "devices", "torch_version", "vllm_commit", "selected_vllm_source_sha256")})
            fingerprints[arm] = digest(runtime)
            require(len({device["uuid"] for node in runtime for device in node["devices"]}) == 8,
                    "runtime does not identify eight distinct H100s")
        differences = sorted(k for k in set(configs["off"]) | set(configs["on"]) if configs["off"].get(k) != configs["on"].get(k))
        require(differences == ["enable_eplb", "eplb_config"], "native config differs outside EPLB treatment")
        require(not configs["off"].get("enable_eplb", False) and configs["on"].get("enable_eplb") is True,
                "wrong native EPLB arms")
        require(configs["on"]["eplb_config"].get("num_redundant_experts") == 32
                and configs["on"]["eplb_config"].get("use_async") is True, "unexpected native EPLB configuration")
        require(fingerprints["off"] == fingerprints["on"], "paired hardware/checkpoint/source fingerprints differ")
        return {"status": "passed", "allowed_differences": differences,
                "off_eplb": False, "on_eplb_config": configs["on"]["eplb_config"],
                "common_runtime_fingerprint_sha256": fingerprints["off"],
                "scope": "Logged arguments and selected runtime fingerprints; not proof of asynchronous migration commits"}
    except (KeyError, TypeError, ValueError, OSError, SyntaxError) as exc:
        return {"status": "failed", "reason": str(exc) if isinstance(exc, AuditError) else type(exc).__name__}


def audit_cohort(directory, cohort, private_dir=None):
    directory = Path(directory)
    report = {"dataset": cohort["dataset"], "stages": {}, "status": "pending",
              "admitted_private_binding": "not_checked" if private_dir is None else "pending"}
    split_path = directory / "pilot-splits.json"
    if not split_path.exists():
        report["reason"] = "pilot partition not available locally"
        return report
    try:
        controller_path = directory / "status.json"
        controller = read_json(controller_path) if controller_path.exists() else {}
        report["controller_state"] = controller.get("state", "not_available")
        if controller.get("state") == "failed":
            report["controller_failure"] = {"arm": controller.get("arm"), "error_type": controller.get("error_type"),
                "note": "Retained stage field may describe the preceding phase; it does not establish execution of the failed arm."}
        splits = read_json(split_path)
        require(set(splits) == set(STAGES), "unexpected pilot stage set")
        all_ids = [rid for stage in STAGES for rid in splits[stage]["ids"]]
        require(all(len(splits[s]["ids"]) == n for s, n in STAGES.items())
                and len(set(all_ids)) == 64, "pilot stages do not form a disjoint8/32/24 partition")
        expected = None
        if private_dir is not None:
            private_dir = Path(private_dir)
            for name in ("pilot.jsonl", "smoke.jsonl"):
                require(file_digest(private_dir / name) == cohort["private_files"][name], "admitted private pilot/smoke file changed")
            _, pilot = read_jsonl(private_dir / "pilot.jsonl")
            _, smoke = read_jsonl(private_dir / "smoke.jsonl")
            require(len(pilot) == 64 and len(smoke) == 8 and [r["id"] for r in smoke] == cohort["smoke_request_ids"],
                    "admitted pilot/smoke allocation differs")
            by_id = {r["id"]: r for r in pilot}
            require(len(by_id) == 64 and set(all_ids) == set(by_id), "observed partition differs from admitted pilot")
            remainder = [r for r in pilot if r["id"] not in {x["id"] for x in smoke}]
            expected = {"smoke": smoke, "calibration": remainder[:32], "external": remainder[32:]}
            for stage, rows in expected.items():
                payload = "".join(json.dumps(row, ensure_ascii=True)+"\n" for row in rows).encode()
                require(splits[stage]["ids"] == [r["id"] for r in rows]
                        and splits[stage]["request_file_sha256"] == digest(payload), "controller partition bytes/order differ from admitted private inputs")
            report["admitted_private_binding"] = "passed"
            lengths = [r["input_tokens"] for r in pilot]
            report["pilot_input_tokens"] = {"count": 64, "minimum": min(lengths), "maximum": max(lengths),
                                            "above_512": sum(x > 512 for x in lengths)}
        report["partition"] = {"status": "passed", "counts": STAGES, "unique_requests": 64,
                               "pilot_splits_sha256": file_digest(split_path)}
        calendar_path = directory / "pilot-calendar.json"
        calendar = {"value": read_json(calendar_path), "sha256": file_digest(calendar_path)} if calendar_path.exists() else None
        internal = {}
        for stage in STAGES:
            stage_report, paired = {"arms": {}}, {}
            for arm in ("off", "on"):
                run, value = audit_run(directory / arm / f"{stage}.jsonl", stage, arm, splits[stage],
                                       calendar, None if expected is None else expected[stage])
                if value is not None:
                    run["native_completion_counters"] = audit_counters(directory / arm / f"{stage}-metrics-before.txt",
                        directory / arm / f"{stage}-metrics-after.txt", STAGES[stage], value["finish_counts"])
                stage_report["arms"][arm], paired[arm] = run, value
            if all(paired.values()):
                issues = [key for key in PAIR_FIELDS if paired["off"]["header"].get(key) != paired["on"]["header"].get(key)]
                if paired["off"]["rows"] != paired["on"]["rows"]:
                    issues.append("request input/payload/tokenization/arrival identities")
                stage_report["pair"] = {"status": "failed" if issues else "passed", "mismatches": issues,
                                        "scope": "paired content, absolute arrival calendar and client configuration identity only"}
            else:
                stage_report["pair"] = {"status": "pending", "reason": "both valid completed arms required"}
            report["stages"][stage] = stage_report
            internal[stage] = paired
        report["native_configuration_pair"] = paired_configuration(directory)
        states = [a["status"] for s in report["stages"].values() for a in s["arms"].values()]
        states += [s["pair"]["status"] for s in report["stages"].values()] + [report["native_configuration_pair"]["status"]]
        report["status"] = "failed" if "failed" in states else "pending" if "pending" in states else "passed"
        if controller.get("state") == "failed":
            report["status"] = "failed"
            report["reason"] = "controller attempt failed; passing completed phases do not form a completed cohort"
        metric_states = [a.get("native_completion_counters", {}).get("status", "unavailable")
                         for s in report["stages"].values() for a in s["arms"].values()]
        report["native_counter_reconciliation"] = "matched" if all(x == "matched" for x in metric_states) else "unreconciled_or_incomplete"
        report["scope"] = "Pilot integrity only. Fixed order, one block, no final allocation freeze. Integrity does not establish superiority, all-layer numerical equivalence, or EPLB committed state."
    except (KeyError, TypeError, ValueError, OSError) as exc:
        report.update(status="failed", reason=str(exc) if isinstance(exc, AuditError) else type(exc).__name__)
    return report


def audit_pilots(root, manifest_path, *, private_dir=None, datasets=DATASETS):
    manifest = read_json(manifest_path)
    require(manifest["manifest_sha256"] == digest({k: v for k, v in manifest.items() if k != "manifest_sha256"}), "suite manifest seal differs")
    require(manifest.get("selection_status") == "pilot_ready_final_candidates_not_frozen", "unexpected final-freeze status")
    cohorts = {c["dataset"]: c for c in manifest["cohorts"]}
    require(set(datasets) <= set(cohorts) and len(set(datasets)) == len(datasets), "unknown/duplicate cohort")
    results = [audit_cohort(Path(root)/name, cohorts[name], None if private_dir is None else Path(private_dir)/name) for name in datasets]
    states = [r["status"] for r in results]
    return {"schema": "full_serving_pilot_integrity_audit_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "failed" if "failed" in states else "incomplete" if "pending" in states else "passed",
            "suite_manifest_sha256": manifest["manifest_sha256"], "suite_manifest_file_sha256": file_digest(manifest_path),
            "auditor_source_sha256": file_digest(__file__), "cohorts": results,
            "inference_claims": False, "raw_text_or_token_arrays_published": False,
            "notes": ["A stored speed_claim_allowed flag is an integrity gate, not causal superiority evidence.",
                      "Completed elapsed excludes the separately reported client tail after last completion/offered horizon; initialization is outside replay timing.",
                      "HTTP timestamps cannot isolate GPU service or server queueing; native timing histograms are not recomputed here.",
                      "Missing or unreconciled metrics never become zero, and asynchronous drain does not establish committed weights."]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--suite-manifest", type=Path, required=True)
    p.add_argument("--private-dir", type=Path, help="Optional explicit admitted private suite directory")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--require-complete", action="store_true")
    args = p.parse_args()
    require(not args.output.exists(), "Audit output exists; choose a fresh snapshot path")
    report = audit_pilots(args.root, args.suite_manifest, private_dir=args.private_dir, datasets=args.datasets)
    report["report_sha256"] = digest(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": report["status"], "cohorts": {r["dataset"]: r["status"] for r in report["cohorts"]},
                      "output": str(args.output), "report_sha256": report["report_sha256"]}))
    return 1 if report["status"] == "failed" else 2 if args.require_complete and report["status"] != "passed" else 0


if __name__ == "__main__":
    raise SystemExit(main())

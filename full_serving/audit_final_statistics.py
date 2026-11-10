"""Independent, stdlib-only reconstruction of the frozen 60-session results.

Reads public records only; never imports the primary analysis/report code.
An arithmetic audit is not hardware, correctness, independence or speedup proof.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

DATASETS = ("wildchat", "sharegpt", "instructcoder", "numinamath", "longbench")
ARMS = ("off", "on", "custom")
METRICS = ("mean_response_ms", "p95_response_ms", "completed_elapsed_s")
T_DF3 = 3.182446305284263


class Incomplete(Exception):
    pass


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


def interval(ratios):
    require(len(ratios) == 4 and all(math.isfinite(x) and x > 0 for x in ratios),
            "four positive block ratios required")
    logs = [math.log(x) for x in ratios]
    center = math.fsum(logs)/4
    sample_variance = math.fsum((x-center)**2 for x in logs)/3
    half_width = T_DF3*math.sqrt(sample_variance/4)
    return {"block_ratios": ratios, "independent_block_count": 4,
            "equal_block_geometric_mean_ratio": math.exp(center),
            "pointwise_95_percent_interval": [math.exp(center-half_width), math.exp(center+half_width)]}


def reconstruct(path, arm, block, calendar, max_tokens=2048):
    raw = Path(path).read_bytes()
    if not raw.endswith(b"\n"):
        raise Incomplete("public JSONL tail is incomplete")
    lines = [json.loads(line) for line in raw.split(b"\n") if line]
    headers = [r for r in lines if r.get("type") == "run_header"]
    summaries = [r for r in lines if r.get("type") == "run_summary"]
    if not summaries:
        raise Incomplete("run completion summary is missing")
    rows = sorted((r for r in lines if r.get("type") == "request"), key=lambda r: r["request_index"])
    require(len(headers) == len(summaries) == 1 and len(lines) == 18 and len(rows) == 16,
            "expected exactly sixteen requests and one header/summary")
    header, summary = headers[0], summaries[0]
    ids = [r["request_id"] for r in rows]
    require(ids == block["request_ids"] == calendar["request_ids"] and len(set(ids)) == 16
            and [r["request_index"] for r in rows] == list(range(16)), "request membership/order differs")
    require(header.get("arm") == arm and header.get("request_count") == 16
            and header.get("requests_sha256") == block["requests_sha256"]
            and header.get("calendar_sha256") == block["calendar_sha256"]
            and header.get("request_ids_sha256") == canonical(ids), "frozen input/calendar/header identity differs")
    require(header.get("max_tokens") == max_tokens and header.get("max_model_len") == 32768
            and header.get("logprobs") is False and header.get("temperature") == 0
            and header.get("enable_thinking") is False and header.get("ignore_eos") is False,
            "measured generation contract differs")
    epoch, horizon = summary["epoch_monotonic_ns"], calendar["horizon_ns"]
    offsets = calendar["arrival_offsets_ns"]
    require(type(epoch) is type(horizon) is int and epoch >= 0 and horizon >= 0
            and len(offsets) == 16 and all(type(x) is int and x >= 0 for x in offsets)
            and offsets == sorted(offsets) and horizon >= offsets[-1], "invalid epoch or frozen arrival calendar")
    responses, signatures, output_tokens, cap_count = [], [], 0, 0
    for row, offset in zip(rows, offsets):
        require(row.get("success") is True and row.get("http_status") == 200 and row.get("saw_done") is True,
                "failed or incomplete request")
        scheduled, send, end = row["scheduled_ns"], row["send_ns"], row["end_ns"]
        require(all(type(x) is int for x in (scheduled, send, end, row["arrival_offset_ns"]))
                and scheduled == epoch+offset and row["arrival_offset_ns"] == offset
                and scheduled <= send <= end, "raw request schedule/timing identity differs")
        require(row.get("response_from_schedule_ns") == end-scheduled
                and row.get("response_from_send_ns") == end-send, "stored response differs from timestamps")
        usage = row["usage"]
        require(all(type(usage.get(k)) is int and usage[k] >= 0 for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
                and usage["total_tokens"] == usage["prompt_tokens"]+usage["completion_tokens"]
                and usage["prompt_tokens"] == row["expected_input_tokens"] > 0
                and usage["prompt_tokens"]+max_tokens <= 32768 and usage["completion_tokens"] <= max_tokens,
                "token accounting differs from the generation contract")
        require(row["finish_reason"] in ("stop", "length")
                and (row["finish_reason"] != "length" or usage["completion_tokens"] == max_tokens), "invalid finish reason/output cap")
        responses.append((end-scheduled)/1_000_000)
        output_tokens += usage["completion_tokens"]
        cap_count += row["finish_reason"] == "length"
        signatures.append((row["request_id"], row["input_sha256"], row["payload_sha256"], offset, usage["prompt_tokens"]))
    ordered = sorted(responses)
    # Linear interpolation at (n-1)*0.95, explicitly 14.25 for n=16.
    p95 = .75*ordered[14]+.25*ordered[15]
    last_end = max(r["end_ns"] for r in rows)
    completion = max(last_end, epoch+horizon)
    elapsed, drain = completion-epoch, max(0, last_end-epoch-horizon)
    require(elapsed > 0 and summary.get("last_request_end_ns") == last_end
            and summary.get("completion_ns") == completion and summary.get("completed_elapsed_ns") == elapsed
            and summary.get("offered_horizon_ns") == horizon and summary.get("drain_ns") == drain
            and type(summary.get("client_finished_ns")) is int and summary["client_finished_ns"] >= completion,
            "stored elapsed/drain differs from raw completion and frozen horizon")
    result = {"mean_response_ms": math.fsum(responses)/16, "p95_response_ms": p95,
              "completed_elapsed_s": elapsed/1_000_000_000, "drain_s": drain/1_000_000_000,
              "completed_requests": 16, "completed_output_tokens": output_tokens, "output_cap_completions": cap_count}
    return result, signatures


def compare_value(differences, path, actual, expected):
    if expected is None:
        if actual is not None:
            differences.append({"path": path, "reported": actual, "recomputed": None})
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, list) or len(actual) != len(expected):
            differences.append({"path": path, "reported": actual, "recomputed": expected})
        else:
            for i, value in enumerate(expected):
                compare_value(differences, path+f"/{i}", actual[i], value)
    elif type(expected) is int:
        if type(actual) is not int or actual != expected:
            differences.append({"path": path, "reported": actual, "recomputed": expected})
    elif not (type(actual) in (int, float) and math.isfinite(actual)
              and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9)):
        differences.append({"path": path, "reported": actual, "recomputed": expected})


def audit(root, report_path):
    root, report_path = Path(root), Path(report_path)
    result = {"schema": "cs471_independent_final_statistics_v1", "status": "incomplete",
        "scope": "Independent public-record arithmetic and paired block statistics only; no new performance, numerical-correctness or quality claim.",
        "method": "Response=(end-scheduled)/1e6 ms; p95 interpolates sorted positions14/15 at0.75/0.25; elapsed includes frozen arrival horizon and drain. Four block log ratios use sample variance/3 and t(0.975,3)=3.182446305284263.",
        "uncertainty_limit": "Pointwise exploratory intervals reproduce the declared four-block assumptions. No request bootstrap or cross-source pooling; this audit does not establish statistical independence.",
        "script_sha256": sha(__file__), "differences": [], "incomplete": [], "datasets": [], "reconstructed_sessions": 0,
        "private_payloads_read": False, "evidence_sha256": {}}
    differences, incomplete = result["differences"], result["incomplete"]
    try:
        protocol_path = root/"frozen-protocol.json"
        protocol, report = load(protocol_path), load(report_path)
        result["evidence_sha256"].update(protocol=sha(protocol_path), primary_report=sha(report_path))
        compact = protocol.get("schema") == "resident_compare_two_workloads_v1"
        names = ["longbench", "numinamath"] if compact else list(DATASETS)
        block_count, session_count = (1, 6) if compact else (4, 60)
        require((compact or protocol.get("schema") == "resident_compare_protocol_v1") and protocol.get("status") == "frozen"
                and protocol.get("final_execution_allowed") is True, "frozen final protocol required")
        require([d["dataset"] for d in protocol["datasets"]] == names
                and all(len(d["blocks"]) == block_count for d in protocol["datasets"]), "declared dataset/group design differs")
        if compact:
            require(protocol.get("generation_by_dataset") == {"longbench": {"max_tokens": 16}, "numinamath": {"max_tokens": 2048}}, "amended output budgets differ")
            result["method"] = "Raw-timestamp reconstruction of one paired16-request group per workload; six sessions. Descriptive ratios only."
            result["uncertainty_limit"] = "No confidence intervals or between-group/run variability estimate is possible with one group."
        require(report.get("schema") == "resident_compare_report_v1"
                and report.get("protocol_sha256") == sha(protocol_path), "primary report/protocol identity differs")
        require([d["dataset"] for d in report["datasets"]] == names, "primary report source membership differs")
    except FileNotFoundError:
        incomplete.append("frozen protocol or primary report is missing")
        return result
    except (KeyError, TypeError, ValueError) as exc:
        differences.append({"path": "protocol/report", "reason": str(exc)})
        result["status"] = "failed"
        return result
    for dataset, reported_dataset in zip(protocol["datasets"], report["datasets"]):
        name = dataset["dataset"]
        out = {"dataset": name, "blocks": [], "summary": {}}
        reported_blocks = {b["block_id"]: b for b in reported_dataset["blocks"]}
        expected_blocks = [b["block_id"] for b in dataset["blocks"]]
        if list(reported_blocks) != expected_blocks or len(reported_dataset["blocks"]) != block_count:
            differences.append({"path": name, "reason": "primary report block membership/order differs"})
        for block in dataset["blocks"]:
            bid = block["block_id"]
            entry = {"block_id": bid, "arms": {}, "contrasts": {}, "raw_pair_complete": False}
            out["blocks"].append(entry)
            reported_block = reported_blocks.get(bid, {})
            signatures = {}
            try:
                calendar_path = root/"calendars"/name/(bid+".json")
                require(sha(calendar_path) == block["calendar_sha256"], "frozen calendar bytes changed")
                calendar = load(calendar_path)
                result["evidence_sha256"][str(calendar_path.relative_to(root))] = sha(calendar_path)
                for arm in ARMS:
                    path = root/name/bid/arm/"measured.jsonl"
                    cap = protocol.get("generation_by_dataset", {}).get(name, {}).get("max_tokens", 2048)
                    values, signatures[arm] = reconstruct(path, arm, block, calendar, cap)
                    entry["arms"][arm] = values
                    result["reconstructed_sessions"] += 1
                    result["evidence_sha256"][str(path.relative_to(root))] = sha(path)
                    actual = reported_block.get("arms", {}).get(arm, {}).get("summary")
                    if not isinstance(actual, dict):
                        incomplete.append(f"{name}/{bid}/{arm}: primary arm summary unavailable")
                        continue
                    comparisons = {"mean_response_ms": actual.get("response_from_schedule_ms", {}).get("mean"),
                                   "p95_response_ms": actual.get("response_from_schedule_ms", {}).get("p95")}
                    comparisons.update({k: actual.get(k) for k in values if k not in comparisons})
                    for metric, value in values.items():
                        compare_value(differences, f"{name}/{bid}/{arm}/{metric}", comparisons[metric], value)
                require(signatures["off"] == signatures["on"] == signatures["custom"], "paired input/payload/absolute-arrival identity differs")
                if reported_block.get("valid") is not True:
                    incomplete.append(f"{name}/{bid}: primary report withholds block on its broader admission checks")
                for baseline in ("off", "on"):
                    contrast = "custom/"+baseline
                    require(all(entry["arms"][a][m] > 0 for a in ("custom", baseline) for m in METRICS),
                            "positive completed timing metrics required for log ratios")
                    ratios = {metric: entry["arms"]["custom"][metric]/entry["arms"][baseline][metric] for metric in METRICS}
                    entry["contrasts"][contrast] = ratios
                    if reported_block.get("valid") is True:
                        for metric, value in ratios.items():
                            actual = reported_block.get("contrasts", {}).get(contrast, {}).get("ratios", {}).get(metric)
                            compare_value(differences, f"{name}/{bid}/{contrast}/{metric}", actual, value)
                entry["raw_pair_complete"] = True
            except (FileNotFoundError, Incomplete) as exc:
                incomplete.append(f"{name}/{bid}: "+(str(exc) if isinstance(exc, Incomplete) else "public calendar or measured record missing"))
            except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
                differences.append({"path": f"{name}/{bid}", "reason": str(exc)})
        if all(b["raw_pair_complete"] for b in out["blocks"]):
            for contrast in ("custom/off", "custom/on"):
                out["summary"][contrast] = {}
                for metric in METRICS:
                    ratios = [b["contrasts"][contrast][metric] for b in out["blocks"]]
                    estimate = ({"block_ratios": ratios, "independent_block_count": 1,
                                 "equal_block_geometric_mean_ratio": ratios[0], "pointwise_95_percent_interval": None}
                                if compact else interval(ratios))
                    out["summary"][contrast][metric] = estimate
                    actual = reported_dataset.get("summary", {}).get(contrast, {}).get(metric, {})
                    if actual.get("available") is not True:
                        incomplete.append(f"{name}/{contrast}/{metric}: primary report interval unavailable")
                    else:
                        for key, value in estimate.items():
                            compare_value(differences, f"{name}/{contrast}/{metric}/{key}", actual.get(key), value)
        result["datasets"].append(out)
    if report.get("complete") is not True:
        incomplete.append("primary report is not complete")
    result["status"] = "failed" if differences else ("passed" if not incomplete and result["reconstructed_sessions"] == session_count else "incomplete")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(args.directory, args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps({"status": result["status"], "reconstructed_sessions": result["reconstructed_sessions"],
                      "differences": len(result["differences"]), "incomplete": len(result["incomplete"])}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

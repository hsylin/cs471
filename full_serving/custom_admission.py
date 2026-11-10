"""Fail-closed admission of pilot-only full-model resident evidence.

Only the five already executed smoke files may be read. Passing permits a
frozen final experiment; it does not require or establish a speedup.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

from full_serving.analyze import summarize_run
from full_serving.audit_pilots import (AuditError, PAIR_FIELDS, audit_counters, audit_request,
                                      client_hash, paired_build_evidence, require, same)
from full_serving.audit_runtime import log_time, parse_logs
from full_serving.custom_compare import verify_custom_bindings
from full_serving.custom_runtime import engine_configuration, logged_arguments, runtime_identity
from full_serving.data import digest, file_digest
from full_serving.resident_diagnostic import validate_weight_records
from full_serving.resident_plugin import (EXPECTED_NATIVE_SOURCES, MODEL_REVISION, ROOT,
                                          balanced_map, validate_config, verify_source_manifest)

DATASETS = ("wildchat", "sharegpt", "instructcoder", "numinamath", "longbench")
TIMED_ARMS = ("off", "resident", "placement_only")
SHAPES = (0, 1, 3, 8, 17, 128, 512, 513, 2048, 8192, 65536)
CASES = {"heterogeneous_padding": (8, [3, 5, 0, 2, 1, 7, 4, 6]),
         "full_distinct_inputs": (17, [17]*8), "all_padding_capture_case": (8, [0]*8)}


class Pending(AuditError):
    pass


def read_json(path):
    return json.loads(Path(path).read_bytes())


def stage(function):
    try:
        return {"status": "passed", **function()}
    except (FileNotFoundError, Pending) as exc:
        return {"status": "pending", "reason": str(exc) if isinstance(exc, Pending) else "required evidence file unavailable"}
    except (AttributeError, AssertionError, KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
        return {"status": "failed", "reason": str(exc) if isinstance(exc, (AuditError, AssertionError)) else type(exc).__name__}


def kernel_gate(path, sources):
    value = read_json(path)
    require(value.get("schema") == "full_resident_kernel_gate_v1" and value.get("passed") is True
            and value.get("native_recording_unchanged_zero") is True
            and value.get("gpu") == "NVIDIA H100" and value.get("torch") == "2.13.0+cu129",
            "kernel correctness gate did not pass on the declared hardware/software")
    expected = {(n, dtype, mode) for n in SHAPES for dtype in ("torch.int32", "torch.int64")
                for mode in ("resident_v1", "native_hash")}
    rows = value.get("cases", [])
    require(len(rows) == 44 and {(r["rows"], r["dtype"], r["mode"]) for r in rows} == expected,
            "kernel gate shape/dtype/mode coverage differs")
    require(all(r.get("cpu_oracle_exact") is True and r.get("padding_and_conservation") is True
                and r.get("changed_input_graph_replays") == (2 if r["rows"] else 0)
                and (r["mode"] != "native_hash" or r.get("native_hash_exact") is True) for r in rows),
            "kernel oracle/native-hash/graph replay checks incomplete")
    for name in ("resident_policy.py", "resident_gpu.py", "check_resident_kernel.py"):
        observed = value.get("source_sha256", {}).get(name)
        require(observed == file_digest(ROOT / "full_serving" / name), "kernel diagnostic source changed")
        if name != "check_resident_kernel.py":
            require(observed == sources["files"]["full_serving/"+name], "kernel differs from admitted implementation")
    return {"artifact_sha256": file_digest(path), "cases": 44,
            "changed_input_graph_replays": sum(r["changed_input_graph_replays"] for r in rows)}


def checkpoint_identity(path):
    value = read_json(path)
    require(value.get("complete") is True and value.get("kind") == "full_checkpoint_sha256"
            and value.get("model_revision") == MODEL_REVISION
            and value.get("expert_layers") == list(range(48)) and value.get("experts_per_layer") == 128
            and value.get("expert_projections") == ["down", "gate", "up"]
            and value.get("tensor_count") == 18867 and value.get("tensor_bytes") == 61064245248,
            "full checkpoint verification scope incomplete")
    require(set(value["shards"]) == {f"model-{i:05d}-of-00016.safetensors" for i in range(1, 17)}
            and set(value["assets"]) == {"config.json", "generation_config.json", "model.safetensors.index.json",
                                        "tokenizer.json", "tokenizer_config.json"}, "checkpoint file set differs")
    content = {kind: {name: row["sha256"] for name, row in value[kind].items()}
               for kind in ("assets", "shards")}
    require(all(re.fullmatch(r"[0-9a-f]{64}", h) for rows in content.values() for h in rows.values()),
            "invalid checkpoint content hash")
    return digest(content)


def errors_before_shutdown(text):
    """Use file order first; a one-second timestamp cannot prove later order."""
    stop_times = [log_time(line) for body in text.values() for line in body.split("\n")
                  if "[shutdown]" in line or "signal=SIGTERM" in line]
    cutoff = min((t for t in stop_times if t is not None), default=math.inf)
    before_stop = 0
    for body in text.values():
        last_time, shutdown = None, False
        for line in body.split("\n"):
            last_time = log_time(line) or last_time
            shutdown |= "[shutdown]" in line or "signal=SIGTERM" in line
            if re.search(r"\bERROR\b|Traceback|EngineDeadError|async loop error", line):
                # An earlier marker in this file establishes teardown order.
                # Without one, the first shutdown second is ambiguous and must
                # fail closed; only a strictly later second proves later timing.
                before_stop += not shutdown and (last_time is None or last_time <= cutoff)
    return before_stop


def pilot_runtime(directory, arm, config_path, sources_path, suite_sha):
    directory = Path(directory)
    status = read_json(directory / "status.json")
    require(status.get("state") != "failed", "pilot controller failed")
    if status.get("state") != "pilot_completed":
        raise Pending("pilot has not completed")
    require(status.get("arm") == arm and status.get("pilot_only") is True
            and status.get("final_inputs_opened") is False, "wrong pilot arm or final data opened")
    require(status.get("config_sha256") == file_digest(config_path)
            and status.get("source_manifest_sha256") == file_digest(sources_path), "pilot source/config identity differs")
    controller = read_json(directory / "controller-source.json")
    require(controller.get("suite_manifest_sha256") == suite_sha, "pilot suite manifest differs")
    require(controller.get("files", {}).get("full_serving/client.py") == file_digest(ROOT / "full_serving/client.py"),
            "pilot HTTP client source differs")
    logs = [directory / f"server-{i}.{suffix}" for i in range(2) for suffix in ("log", "err")]
    text = {p.name: p.read_text() for p in logs}
    args = logged_arguments(text["server-0.log"])
    engines = [engine_configuration(text[f"server-{i}.log"]) for i in range(2)]
    require(engines[0] == engines[1], "node engine configurations disagree")
    for node, rank in enumerate((0, 4)):
        require((f"rank {rank} in world size 8 is assigned as DP rank {rank}, PP rank 0, "
                 f"PCP rank 0, TP rank 0, ETP rank 0, EP rank {rank}, EPLB rank ") in text[f"server-{node}.log"],
                "actual DP/EP rank topology evidence missing")
    observations = parse_logs(logs, {})
    require(observations["runner"] == ["Using V2 Model Runner"]
            and observations["activation_managers"] == ["Using AgRsAll2AllManager"]
            and observations["moe"] == ["Using TRITON Unquantized MoE"]
            and observations["attention"] == ["Using FlashAttention version 3"], "actual pilot backend differs")
    require(set(observations["graph_capture_rank_records"]) == {str(i) for i in range(8)}
            and all(len(v) == 2 for v in observations["graph_capture_rank_records"].values()),
            "all eight ranks must complete both graph capture paths")
    require(not observations["selected_weight_communicators"] and not observations["ordinary_async_launches"]
            and not observations["profile_rearrangement_entries"], "native control unexpectedly active in fixed/OFF pilot")
    # Intentional teardown can emit exceptions after the drained request phase.
    before_stop = errors_before_shutdown(text)
    require(before_stop == 0, "runtime error preceded intentional shutdown")
    checkpoint = directory / "checkpoint-verified.json"
    cp_identity = checkpoint_identity(checkpoint)
    preflights, identities = {}, []
    for role in ("primary", "worker"):
        value = read_json(directory / f"preflight-{role}.json")
        identity = runtime_identity(value, role)
        require(identity.pop("checkpoint_manifest_sha256") == file_digest(checkpoint),
                "runtime checkpoint record differs from archived verification")
        identity["checkpoint_content_sha256"] = cp_identity
        identities.append(identity)
        preflights[role] = value
    require(len({n["hostname"] for n in identities}) == 2
            and len({d["uuid"] for n in identities for d in n["devices"]}) == 8, "eight distinct H100s required")
    if arm != "off":
        config = read_json(config_path)
        protocol = {"custom": {"config_file": str(Path(config_path).resolve()), "mode": config["mode"],
                                "config_sha256": file_digest(config_path), "source_manifest_sha256": file_digest(sources_path)}}
        binding = verify_custom_bindings(directory, protocol, ROOT / "unused-public-protocol.json")
        for rank in range(8):
            value = read_json(directory / f"resident-bindings-rank{rank}.json")
            require(value["native_source_sha256"] == EXPECTED_NATIVE_SOURCES
                    and value["hostname"] == identities[rank // 4]["hostname"], "custom source/host binding differs")
    else:
        require(not args.get("enable_eplb", False) and "eplb_config" not in args, "OFF pilot enabled EPLB")
        require(not list(directory.glob("resident-bindings-rank*.json")), "custom bindings leaked into native OFF")
        binding = None
    return {"args": args, "engine": engines[0], "identities": identities, "preflights": preflights,
            "client_sha256": controller["files"]["full_serving/client.py"],
            "public": {"status_sha256": file_digest(directory / "status.json"),
                       "runtime_fingerprint_sha256": digest(identities), "custom_bindings": binding,
                       "graph_capture_rank_count": 8, "error_entries_before_shutdown": before_stop,
                       "error_entries_total": len(observations["errors"]),
                       "evidence_sha256": {p.name: file_digest(p) for p in logs} | {
                           f"preflight-{role}.json": file_digest(directory / f"preflight-{role}.json") for role in preflights}}}


def http_phase(directory, dataset, phase, arm, cohort, private_rows, client_sha):
    label = f"{dataset}-pass{phase}"
    path = Path(directory) / (label+".jsonl")
    raw = path.read_bytes()
    require(raw.endswith(b"\n"), "incomplete HTTP record tail")
    rows = [json.loads(line) for line in raw.split(b"\n") if line]
    headers = [r for r in rows if r.get("type") == "run_header"]
    summaries = [r for r in rows if r.get("type") == "run_summary"]
    records = sorted((r for r in rows if r.get("type") == "request"), key=lambda r: r["request_index"])
    require(len(headers) == len(summaries) == 1 and len(records) == 8 and len(rows) == 10,
            "HTTP pilot must contain eight completed records and one header/summary")
    header, summary = headers[0], summaries[0]
    ids = cohort["smoke_request_ids"]
    require([r["request_index"] for r in records] == list(range(8))
            and [r["request_id"] for r in records] == ids and header["request_count"] == 8,
            "HTTP pilot indices/membership differ")
    require(header["arm"] == arm and header["requests_sha256"] == cohort["private_files"]["smoke.jsonl"]
            and header["request_ids_sha256"] == client_hash(ids) and header["client_source_sha256"] == client_sha,
            "HTTP pilot arm/input/source identity differs")
    require(header["max_tokens"] == 2048 and header["max_model_len"] == 32768 and header["temperature"] == 0
            and header["enable_thinking"] is False and header["ignore_eos"] is False
            and header["max_inflight"] == 64 and header["model"] == "qwen3-full"
            and header["logprobs"] is (phase == 0) and header["diagnostic_return_token_ids"] is (phase == 0),
            "HTTP generation/diagnostic contract differs")
    offsets = [round(i/2*1e9) for i in range(8)]
    calendar = {"kind": "synthetic_regular_pilot", "rate_requests_per_s": 2.0,
                "arrival_offsets_ns": offsets, "request_ids": ids}
    require(header["calendar_sha256"] == hashlib.sha256(json.dumps(calendar, sort_keys=True).encode()).hexdigest()
            and [r["arrival_offset_ns"] for r in records] == offsets, "pilot absolute arrivals differ")
    epoch, finished = summary["epoch_monotonic_ns"], summary["client_finished_ns"]
    require(type(epoch) is type(finished) is int, "invalid HTTP time origin")
    for row, private in zip(records, private_rows):
        audit_request(row, header, epoch, finished, private)
    last = max(r["end_ns"] for r in records)
    end = max(last, epoch+offsets[-1])
    equations = {"last_request_end_ns": last, "offered_horizon_ns": offsets[-1], "completion_ns": end,
                 "completed_elapsed_ns": end-epoch, "drain_ns": max(0, last-epoch-offsets[-1])}
    require(finished >= end and all(summary.get(k) == v for k, v in equations.items()),
            "HTTP completion/drain timing identity differs")
    recomputed = summarize_run(records, summary)
    require(recomputed["valid"] is True and all(same(summary.get(k), v) for k, v in recomputed.items()),
            "stored HTTP summary differs from valid raw records")
    require(math.isfinite(summary["completed_elapsed_s"]) and summary["completed_elapsed_s"] > 0,
            "pilot performance is not finite and completed")
    metrics = audit_counters(Path(directory)/(label+"-metrics-before.txt"), Path(directory)/(label+"-metrics-after.txt"),
                             8, dict(Counter(r["finish_reason"] for r in records)))
    require(metrics["status"] == "matched", "native HTTP completion counters are unreconciled")
    return {"header": header, "records": records, "public": {"record_sha256": file_digest(path), "requests": 8,
            "private_pilot_payloads_verified": True, "raw_timing_and_summary_verified": True,
            "completed_elapsed_s": summary["completed_elapsed_s"], "completed_output_tokens": summary["completed_output_tokens"],
            "mean_response_ms": summary["response_from_schedule_ms"]["mean"], "native_metrics": metrics}}


def loaded_weights(directory, config_sha, sources_sha):
    peers, checks = [], []
    for rank in range(8):
        path = Path(directory) / f"resident-startup-diagnostic-rank{rank}.json"
        value = read_json(path)
        require(value.get("status") == "passed" and value.get("ep_rank") == rank
                and value.get("config_sha256") == config_sha and value.get("source_manifest_sha256") == sources_sha
                and value.get("source_sha256") == file_digest(ROOT/"full_serving/resident_diagnostic.py")
                and value.get("reference_matmul_precision") == "highest", "loaded-weight diagnostic identity differs")
        peers.append({"rank": rank, "weights": value["local_weight_records"]})
        rows = value["native_expert_numerical_checks"]
        require(len(rows) == 9 and {(r["layer_index"], r["case"]) for r in rows}
                == {(layer, case) for layer in (0, 23, 47) for case in CASES}, "numerical probe coverage differs")
        for row in rows:
            require((row["rows_per_rank"], row["real_rows_per_rank"]) == CASES[row["case"]]
                    and row["atol"] == .02 and row["rtol"] == .03, "numerical tolerance or padding case changed")
            require(all(row.get(k) is True for k in ("logical_identity_exact", "physical_cpu_oracle_exact",
                    "one_home_policy_checked_by_cpu_oracle", "routing_weights_unchanged", "token_conservation_exact",
                    "gathered_rank_agreement", "native_expert_outputs_close", "direct_fp32_outputs_close")),
                    "loaded-weight numerical diagnostic did not pass")
        checks.append({"rank": rank, "sha256": file_digest(path), "cases": 9})
    weights = validate_weight_records(peers, balanced_map())
    return {"full_replica_weights": weights, "rank_reports": checks,
            "scope": "All48 layer weight identities; native arithmetic at layers0/23/47. Full-model logits were not independently recomputed."}


def build_admission(paths):
    paths = {k: Path(v) for k, v in paths.items()}
    result = {"schema": "cs471_full_resident_pilot_admission_v1", "status": "pending", "pilot_only": True,
              "created_utc": datetime.now(timezone.utc).isoformat(), "correctness_gate_passed": False,
              "performance_gate_passed": False, "final_payloads_read": False,
              "performance_gate_meaning": "Finite, complete, identically configured pilot replays; no requirement to beat OFF and no final speedup claim."}
    def configuration():
        resident = validate_config(read_json(paths["resident_config"]))
        diagnostic = validate_config(read_json(paths["diagnostic_config"]))
        placement = validate_config(read_json(paths["placement_config"]))
        require(resident["mode"] == "resident_v1" and resident.get("startup_diagnostic") is False,
                "timed resident configuration must exclude startup instrumentation")
        require(diagnostic == {**resident, "startup_diagnostic": True}
                and placement == {**resident, "mode": "native_hash"}, "pilot configurations differ outside declared ablations")
        sources = read_json(paths["sources"])
        verify_source_manifest(sources)
        suite = read_json(paths["suite_manifest"])
        require(suite.get("manifest_sha256") == digest({k: v for k, v in suite.items() if k != "manifest_sha256"}),
                "suite manifest seal differs")
        cohorts = {c["dataset"]: c for c in suite["cohorts"]}
        require(set(cohorts) == set(DATASETS), "five separate source cohorts required")
        private = {}
        for dataset in DATASETS:
            path = paths["private_pilot_root"] / dataset / "smoke.jsonl"
            require(file_digest(path) == cohorts[dataset]["private_files"]["smoke.jsonl"], "private pilot file changed")
            rows = [json.loads(line) for line in path.read_bytes().split(b"\n") if line]
            require(len(rows) == 8 and [r["id"] for r in rows] == cohorts[dataset]["smoke_request_ids"],
                    "private pilot membership differs")
            private[dataset] = rows
        result.update(custom_config_sha256=file_digest(paths["resident_config"]),
                      custom_source_manifest_sha256=file_digest(paths["sources"]), suite_manifest_sha256=file_digest(paths["suite_manifest"]))
        return sources, cohorts, private
    try:
        sources, cohorts, private = configuration()
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
        result.update(status="failed", reason=str(exc) if isinstance(exc, AuditError) else type(exc).__name__)
        return result
    gates = result["gates"] = {}
    gates["kernel"] = stage(lambda: kernel_gate(paths["kernel"], sources))
    diagnostic_internal = {}
    def diagnostic_gate():
        runtime = pilot_runtime(paths["diagnostic"], "resident", paths["diagnostic_config"], paths["sources"], result["suite_manifest_sha256"])
        diagnostic_internal.update(runtime)
        weights = loaded_weights(paths["diagnostic"], file_digest(paths["diagnostic_config"]), result["custom_source_manifest_sha256"])
        phases = {ds: http_phase(paths["diagnostic"], ds, 0, "resident", cohorts[ds], private[ds], runtime["client_sha256"])["public"]
                  for ds in DATASETS}
        return {"runtime": runtime["public"], "loaded_weights": weights, "http_pilots": phases, "completed_requests": 40}
    gates["loaded_weights_and_full_model_http"] = stage(diagnostic_gate)
    result["correctness_gate_passed"] = all(gates[k]["status"] == "passed" for k in gates)
    progress = result["timed_pilot_progress"] = {
        arm: {"status": "pending", "reason": "not yet inspected"} for arm in TIMED_ARMS}
    def timed_gate():
        if not result["correctness_gate_passed"]:
            raise Pending("kernel and full-model correctness admission required first")
        runtimes, phases = {}, {}
        for arm in TIMED_ARMS:
            def audit_arm():
                config_path = paths["placement_config"] if arm == "placement_only" else paths["resident_config"]
                runtimes[arm] = pilot_runtime(paths[arm], arm, config_path, paths["sources"], result["suite_manifest_sha256"])
                require(runtimes[arm]["engine"] == diagnostic_internal["engine"]
                        and runtimes[arm]["identities"] == diagnostic_internal["identities"],
                        "timed pilot GPUs/checkpoint/source/backend/environment differ from admitted diagnostic")
                build = paired_build_evidence({"off": diagnostic_internal["preflights"], "on": runtimes[arm]["preflights"]})
                require(build["flashinfer_prebuild"]["status"] == "matched", "timed pilot cache/compiler evidence incomplete")
                phases[arm] = {f"{ds}-pass{phase}": http_phase(paths[arm], ds, phase, arm, cohorts[ds], private[ds], runtimes[arm]["client_sha256"])
                               for ds in DATASETS for phase in (0, 1)}
                return {"completed_requests": 80, "runtime": runtimes[arm]["public"],
                        "compiler_cache_pair_with_diagnostic": build,
                        "phases": {name: row["public"] for name, row in phases[arm].items()},
                        "scope": "This arm passed its individual checks; the complete three-arm comparison is admitted separately."}
            progress[arm] = stage(audit_arm)
            if progress[arm]["status"] == "pending":
                raise Pending(f"{arm}: {progress[arm]['reason']}")
            require(progress[arm]["status"] == "passed", f"{arm}: {progress[arm].get('reason', 'pilot audit failed')}")
        off = runtimes["off"]
        builds = {}
        for arm in ("resident", "placement_only"):
            other = runtimes[arm]
            differences = sorted(k for k in set(off["args"]) | set(other["args"]) if off["args"].get(k) != other["args"].get(k))
            require(differences == ["enable_eplb", "eplb_config"] and other["args"].get("enable_eplb") is True
                    and other["args"]["eplb_config"].get("num_redundant_experts") == 32,
                    "pilot native flags differ outside intended treatment")
            require(other["engine"] == off["engine"] == diagnostic_internal["engine"]
                    and other["identities"] == off["identities"] == diagnostic_internal["identities"],
                    "pilot GPUs/checkpoint/source/backend/environment differ")
            builds[arm] = paired_build_evidence({"off": off["preflights"], "on": other["preflights"]})
            require(builds[arm]["flashinfer_prebuild"]["status"] == "matched", "pilot cache/compiler evidence incomplete")
            for name in phases[arm]:
                a, b = phases["off"][name], phases[arm][name]
                require(all(a["header"].get(k) == b["header"].get(k) for k in PAIR_FIELDS), "paired pilot HTTP headers differ")
                require(all(all(x.get(k) == y.get(k) for k in ("request_id", "input_sha256", "payload_sha256",
                            "expected_input_tokens", "expected_prompt_token_ids_sha256", "arrival_offset_ns"))
                            for x, y in zip(a["records"], b["records"])), "paired pilot payloads/arrivals differ")
        require(runtimes["resident"]["args"] == runtimes["placement_only"]["args"], "placement-only native config differs")
        return {"required_arms": list(TIMED_ARMS), "completed_requests": 240,
                "runtimes": {arm: r["public"] for arm, r in runtimes.items()}, "compiler_cache_pairs": builds,
                "phases": {arm: {name: r["public"] for name, r in rows.items()} for arm, rows in phases.items()},
                "win_filter_applied": False, "timing_caveat": "Fixed-order pilot sessions; pass0 uses logprobs, pass1 excludes that HTTP instrumentation. Neither is final comparative evidence."}
    gates["timed_comparison"] = stage(timed_gate)
    result["performance_gate_passed"] = gates["timed_comparison"]["status"] == "passed"
    result["status"] = "failed" if any(g["status"] == "failed" for g in gates.values()) else (
        "passed" if result["correctness_gate_passed"] and result["performance_gate_passed"] else "pending")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("kernel", "diagnostic", "off", "resident", "placement-only", "diagnostic-config",
                 "resident-config", "placement-config", "sources", "suite-manifest", "private-pilot-root", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = vars(parser.parse_args())
    output = args.pop("output")
    report = build_admission(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps({k: report[k] for k in ("status", "correctness_gate_passed", "performance_gate_passed")}))


if __name__ == "__main__":
    main()

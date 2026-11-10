"""Separate three-arm held-out comparison; never alters completed native pilots.

The default run command only prints the frozen schedule. --execute is required
to use the already authorized allocation. Final payloads are opened only after
the frozen protocol and matching custom-pilot admission have passed validation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

from full_serving import orchestrate as native


ARMS = ("off", "on", "custom")
DATASETS = ("wildchat", "sharegpt", "instructcoder", "numinamath", "longbench")
ROOT = Path(__file__).resolve().parents[1]
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
HASH = re.compile(r"[0-9a-f]{64}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def artifact_path(protocol_path, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(protocol_path).resolve().parent / path).resolve()


def private_path(private_root, value):
    path = Path(value)
    require(not path.is_absolute() and ".." not in path.parts, "private request path must be relative and confined")
    resolved = (Path(private_root).resolve() / path).resolve()
    require(resolved.is_relative_to(Path(private_root).resolve()), "private request path escapes admitted root")
    return resolved


def validate_protocol(protocol, *, allow_draft=False):
    """Metadata-only checks: no private files, GPUs or network are touched."""
    require(protocol.get("schema") == "resident_compare_protocol_v1", "unsupported comparison protocol")
    frozen = protocol.get("status") == "frozen"
    require(frozen or (allow_draft and protocol.get("status") == "draft"), "protocol must be frozen after the custom pilot gate")
    datasets = protocol.get("datasets", [])
    require([d.get("dataset") for d in datasets] == list(DATASETS), "freeze the five main datasets in declared order")
    generation = protocol.get("generation", {})
    require(all(generation.get(key) == value for key, value in {
        "max_tokens": 2048, "max_model_len": 32768, "temperature": 0, "enable_thinking": False,
        "ignore_eos": False, "truncate_prompts": False}.items()), "generation contract differs from the fixed client/server configuration")
    custom = protocol.get("custom", {})
    if frozen:
        require(protocol.get("final_execution_allowed") is True, "final execution remains disabled")
        require(re.fullmatch(r"full_serving\.[a-zA-Z_][a-zA-Z0-9_.]*", custom.get("entrypoint") or ""), "invalid custom entrypoint")
        require(custom.get("mode") == "resident_v1", "unsupported custom mode")
        require(custom.get("bootstrap_path") == "full_serving/bootstrap", "unexpected worker bootstrap path")
        for key in ("config_sha256", "source_manifest_sha256"):
            require(isinstance(custom.get(key), str) and HASH.fullmatch(custom[key]), f"missing {key}")
        for key in ("config_file", "source_manifest_file"):
            require(isinstance(custom.get(key), str) and custom[key], f"missing {key}")
    admission = protocol.get("pilot_admission", {})
    if frozen:
        require(isinstance(admission.get("sha256"), str) and HASH.fullmatch(admission["sha256"]) and admission.get("file"),
                "a hash-bound custom pilot admission is required")
    execution = protocol.get("execution", {})
    require(execution.get("repeats", 1) == 1, "this bounded protocol has one session per content/arrival block")
    if frozen:
        require(execution.get("warmup_passes") == 2, "freeze two identical pilot-only warmup passes")
    require(execution.get("max_inflight", 64) == 64, "freeze max_inflight=64 across arms")
    for key, default in (("startup_timeout_s", 2400), ("http_timeout_s", 1800)):
        value = execution.get(key, default)
        require(isinstance(value, (int, float)) and not isinstance(value, bool) and 0 < value <= 7200,
                f"finite bounded {key} required")
    require(isinstance(protocol.get("acceptance"), dict) and protocol["acceptance"], "freeze acceptance criteria before final measurements")
    global_positions = [Counter() for _ in ARMS]
    seen_blocks = set()
    for dataset in datasets:
        warmup = dataset.get("warmup", {})
        require(warmup.get("request_count") == 8 and warmup.get("requests_file"), "eight pilot-only warmup requests required")
        require(HASH.fullmatch(str(warmup.get("requests_sha256", ""))), "warmup request file hash missing")
        require(warmup.get("split") == "pilot", "warmup must be explicitly pilot-only")
        blocks = dataset.get("blocks", [])
        require(len(blocks) == 4, "four held-out blocks per dataset are required")
        local_positions = [Counter() for _ in ARMS]
        for block in blocks:
            bid = block.get("block_id", "")
            require(NAME.fullmatch(bid) and (dataset["dataset"], bid) not in seen_blocks, "unsafe/duplicate block ID")
            seen_blocks.add((dataset["dataset"], bid))
            require(block.get("request_count") == 16 and block.get("split") == "final", "each block must contain sixteen held-out requests")
            require(block.get("requests_file") and block.get("calendar_file"), "block input/calendar paths missing")
            for key in ("requests_sha256", "calendar_sha256"):
                require((not frozen and key == "requests_sha256" and block.get(key) is None)
                        or HASH.fullmatch(str(block.get(key, ""))), f"block {key} missing")
            order = block.get("arm_order", [])
            require(len(order) == 3 and set(order) == set(ARMS), "each block must contain each arm exactly once")
            for position, arm in enumerate(order):
                local_positions[position][arm] += 1
                global_positions[position][arm] += 1
        require(all(1 <= counts[arm] <= 2 for counts in local_positions for arm in ARMS),
                "every dataset must expose every arm at every order position")
    require(all(max(counts[arm] for arm in ARMS) - min(counts[arm] for arm in ARMS) <= 1 for counts in global_positions),
            "global arm-position counts must differ by at most one")
    return protocol


def load_admitted_protocol(path):
    """Read public admission first; never opens a final request payload."""
    path = Path(path)
    protocol = validate_protocol(json.loads(path.read_bytes()))
    return validate_admission_artifacts(protocol, path)


def validate_admission_artifacts(protocol, path):
    """Also usable with path-rebased copies of the archived public evidence."""
    admission_record = protocol["pilot_admission"]
    admission_path = artifact_path(path, admission_record["file"])
    require(native.sha(admission_path) == admission_record["sha256"], "custom pilot admission changed")
    admission = json.loads(admission_path.read_bytes())
    require(admission.get("status") == "passed" and admission.get("pilot_only") is True,
            "custom pilot correctness/performance admission has not passed")
    custom = protocol["custom"]
    for field in ("config_sha256", "source_manifest_sha256"):
        require(admission.get("custom_" + field) == custom[field], "custom implementation differs from admitted pilot")
    require(admission.get("correctness_gate_passed") is True and admission.get("performance_gate_passed") is True,
            "both custom pilot gates must pass before final payload admission")
    # Feasible, finite completed timing is sufficient. This does not require
    # beating OFF: a correct method with a negative result must still be tested.
    for stem in ("config", "source_manifest"):
        require(native.sha(artifact_path(path, custom[stem + "_file"])) == custom[stem + "_sha256"],
                "frozen custom configuration/source manifest changed")
    config = json.loads(artifact_path(path, custom["config_file"]).read_bytes())
    require(config.get("startup_diagnostic", False) is False, "timed comparison must disable correctness-only startup probes")
    return protocol


def load_source_metadata(protocol, path):
    from full_serving.custom_inputs import load_source_metadata as load
    return load(protocol, path)


def verify_private_request(row, metadata):
    from full_serving.custom_inputs import verify_private_request as verify
    return verify(row, metadata)


def schedule(protocol, *, allow_draft=False):
    validate_protocol(protocol, allow_draft=allow_draft)
    return [{"ordinal": ordinal, "dataset": dataset["dataset"], "block_id": block["block_id"],
             "arm": arm, "position": position, "session_id": f'{dataset["dataset"]}/{block["block_id"]}/{arm}'}
            for ordinal, (dataset, block, position, arm) in enumerate(
                (dataset, block, position, arm)
                for dataset in protocol["datasets"] for block in dataset["blocks"]
                for position, arm in enumerate(block["arm_order"]))]


def admit_and_copy_inputs(protocol, protocol_path, private_root, private_out, public_out, source_metadata=None):
    """Called only after load_admitted_protocol; source files remain read-only."""
    validate_admission_artifacts(validate_protocol(protocol), protocol_path)
    source_metadata = source_metadata or load_source_metadata(protocol, protocol_path)
    inputs, warmup_ids, final_ids, warmup_hashes, final_hashes = {}, set(), set(), set(), set()
    request_evidence = {}
    for dataset in protocol["datasets"]:
        name = dataset["dataset"]
        inputs[name] = {}
        for label, item in [("warmup", dataset["warmup"]), *[(b["block_id"], b) for b in dataset["blocks"]]]:
            source = private_path(private_root, item["requests_file"])
            raw = source.read_bytes()
            require(hashlib.sha256(raw).hexdigest() == item["requests_sha256"], "frozen request file changed")
            rows = [json.loads(line) for line in raw.decode().split("\n") if line.strip()]
            require(len(rows) == item["request_count"], "request count differs from frozen metadata")
            ids = [r["id"] for r in rows]
            require(len(set(ids)) == len(ids), "duplicate requests in a block")
            require(item.get("request_ids") == ids, "private request order differs from frozen public identities")
            if item.get("request_ids_sha256") is not None:
                require(canonical_hash(ids) == item["request_ids_sha256"], "frozen request identity digest mismatch")
            hashes = [canonical_hash(r["messages"]) for r in rows]
            for row in rows:
                metadata = source_metadata["by_id"][row["id"]]
                verified = verify_private_request(row, metadata)
                request_evidence[row["id"]] = {**verified, "dataset": name, "block_id": label,
                    "request_id": row["id"], "source_metadata_sha256": canonical_hash(metadata),
                    "client_input_sha256": canonical_hash(row["messages"])}
            target_ids, target_hashes = (warmup_ids, warmup_hashes) if label == "warmup" else (final_ids, final_hashes)
            require(not target_ids.intersection(ids), "request identity reused across independent blocks")
            require(not target_hashes.intersection(hashes) and len(set(hashes)) == len(hashes), "duplicate content across blocks")
            target_ids.update(ids)
            target_hashes.update(hashes)
            target = private_out / "inputs" / name / (label + ".jsonl")
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with target.open("xb") as handle:
                handle.write(raw)
            target.chmod(0o600)
            inputs[name][label] = {"requests": target, "requests_sha256": item["requests_sha256"], "request_ids": ids}
            if label != "warmup":
                calendar_source = artifact_path(protocol_path, item["calendar_file"])
                cal_raw = calendar_source.read_bytes()
                require(hashlib.sha256(cal_raw).hexdigest() == item["calendar_sha256"], "frozen calendar changed")
                cal = json.loads(cal_raw)
                offsets = cal.get("arrival_offsets_ns", [])
                require(cal.get("request_ids") == ids and len(offsets) == len(ids) and offsets[0] == 0,
                        "calendar/request identity mismatch")
                require(all(type(x) is int and x >= 0 for x in offsets) and offsets == sorted(offsets)
                        and type(cal.get("horizon_ns")) is int and cal["horizon_ns"] > offsets[-1], "invalid absolute calendar")
                target_cal = public_out / "calendars" / name / (label + ".json")
                target_cal.parent.mkdir(parents=True, exist_ok=True)
                target_cal.write_bytes(cal_raw)
                inputs[name][label].update(calendar=target_cal, calendar_sha256=item["calendar_sha256"])
    require(not warmup_ids.intersection(final_ids) and not warmup_hashes.intersection(final_hashes), "pilot/final content overlap")
    native.write_json(public_out / "request-admission.json", {"schema": "resident_compare_request_admission_v1",
        "status": "passed", "requests": request_evidence,
        "source_manifest_hashes": {k: v for k, v in protocol.get("source_admission", {}).items() if k.endswith("_sha256")},
        "scope": "Frozen private bytes verified against previously admitted public source metadata; only hashes and identities retained."})
    return inputs


def clean_server_environment(environment, protocol):
    env = native.server_environment(environment)
    for key in list(env):
        if key.startswith("FULL_SERVING_CUSTOM_"):
            env.pop(key)
    bootstrap = str((ROOT / protocol["custom"]["bootstrap_path"]).resolve())
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = os.pathsep.join(p for p in env["PYTHONPATH"].split(os.pathsep)
                                          if p and str(Path(p).resolve()) != bootstrap)
    if "VLLM_PLUGINS" in env:
        plugins = [name for name in env["VLLM_PLUGINS"].split(",") if name.strip() != "cs471_resident"]
        if plugins:
            env["VLLM_PLUGINS"] = ",".join(plugins)
        else:
            env.pop("VLLM_PLUGINS")
    return env


def verify_custom_bindings(arm_dir, protocol, protocol_path):
    """Bind all eight rank reports; this is installation evidence, not numerical proof."""
    config = json.loads(artifact_path(protocol_path, protocol["custom"]["config_file"]).read_bytes())
    expected_map = config["physical_to_logical_map"]
    require(len(expected_map) == 160, "resident map must have160 physical slots")
    steps = {p.read_text().strip() for p in Path(arm_dir).glob("server-step-*.txt")}
    require(len(steps) == 1, "missing or inconsistent serving step records")
    step = next(iter(steps))
    require(re.fullmatch(native.JOB + r"\.\d+", step), "unexpected serving step")
    reference_layers, reference_sources, reports = None, None, []
    inactive = ("native_eplb_state_created", "native_observation_history_active", "native_profile_migration_active",
                "native_async_worker_active", "native_control_step_active", "normal_commit_applicable")
    for rank in range(8):
        path = Path(arm_dir) / f"resident-bindings-rank{rank}.json"
        record = json.loads(path.read_bytes())
        require(record.get("schema") == "cs471_full_resident_bindings_v1" and record.get("status") == "bound",
                "custom binding record missing/incomplete")
        require(record.get("ep_rank") == rank and record.get("mode") == protocol["custom"]["mode"], "wrong custom rank or mode")
        require(record.get("job_id") == native.JOB and record.get("step_id") == step.split(".")[1], "stale custom binding step")
        require(record.get("config_sha256") == protocol["custom"]["config_sha256"]
                and record.get("source_manifest_sha256") == protocol["custom"]["source_manifest_sha256"], "custom binding source/config mismatch")
        require(record.get("physical_to_logical_map") == expected_map, "rank map disagreement")
        require(record.get("native_vllm_commit") == config["vllm_commit"] and record.get("model_revision") == config["model_revision"],
                "custom binding model/source pin mismatch")
        require(all(record.get(key) is False for key in inactive), "native recurring control remains active in custom arm")
        require("startup_diagnostic" in record, "startup diagnostic mode not recorded")
        diagnostic = record["startup_diagnostic"]
        if config.get("startup_diagnostic", False):
            require(isinstance(diagnostic, dict) and diagnostic.get("status") == "passed"
                    and diagnostic.get("timed_serving_measurement") is False, "startup correctness probe did not pass")
            diagnostic_path = Path(arm_dir) / f"resident-startup-diagnostic-rank{rank}.json"
            require(Path(diagnostic.get("path", "")).name == diagnostic_path.name
                    and native.sha(diagnostic_path) == diagnostic.get("sha256"), "startup diagnostic reference changed")
            probe = json.loads(diagnostic_path.read_bytes())
            require(probe.get("schema") == "cs471_full_resident_startup_diagnostic_v1" and probe.get("status") == "passed"
                    and probe.get("ep_rank") == rank and probe.get("job_id") == native.JOB
                    and probe.get("step_id") == step.split(".")[1]
                    and probe.get("config_sha256") == protocol["custom"]["config_sha256"]
                    and probe.get("source_manifest_sha256") == protocol["custom"]["source_manifest_sha256"], "startup probe identity mismatch")
            require(probe.get("weight_check") == {"physical_layer_experts": 7680, "logical_layer_experts": 6144,
                "duplicated_logical_layer_experts": 1536, "full_bf16_replica_hashes_equal": True}
                and len(probe.get("local_weight_records", [])) == 960, "incomplete replica-weight diagnostic")
            checks = probe.get("native_expert_numerical_checks", [])
            flags = ("logical_identity_exact", "physical_cpu_oracle_exact", "one_home_policy_checked_by_cpu_oracle",
                     "routing_weights_unchanged", "token_conservation_exact", "gathered_rank_agreement",
                     "native_expert_outputs_close", "direct_fp32_outputs_close")
            require(len(checks) == 9 and all(all(check.get(key) is True for key in flags) for check in checks),
                    "incomplete/failed native expert correctness probes")
        else:
            require(diagnostic is None, "correctness-only probes leaked into a timing configuration")
        layers = record.get("layers", [])
        names = [layer["layer"] for layer in layers]
        require(record.get("layer_count") == len(layers) == len(set(names)) == 48, "all48 layer bindings are required")
        indices = [re.search(r"(?:^|\.)layers\.(\d+)\.", name) for name in names]
        require(all(indices) and {int(m[1]) for m in indices} == set(range(48)), "wrong layer coverage")
        for layer in layers:
            require(layer.get("physical_slots") == list(range(rank * 20, (rank + 1) * 20))
                    and layer.get("resident_logical_experts") == expected_map[rank * 20:(rank + 1) * 20]
                    and layer.get("w13_shape") == [20, 1536, 2048] and layer.get("w2_shape") == [20, 2048, 768],
                    "wrong resident weight geometry or local mapping")
        sources = record.get("native_source_sha256")
        require(isinstance(sources, dict) and sources, "native source record missing")
        if reference_layers is None:
            reference_layers, reference_sources = sorted(names), sources
        require(sorted(names) == reference_layers and sources == reference_sources, "rank binding/source disagreement")
        reports.append({"rank": rank, "sha256": native.sha(path), "layer_count": 48})
    return {"status": "passed", "reports": reports, "all_layer_numerical_equivalence_proved": False,
            "meaning": "Eight rank manifests bind the intended hook, sources and fixed map; numerical correctness relies on the separate pilot admission."}


def native_on_arguments(snapshot, primary_ip, role):
    args = ["serve", str(snapshot), "--served-model-name", "qwen3-full", "--dtype", "bfloat16",
            "--max-model-len", "32768", "--generation-config", "vllm", "--tensor-parallel-size", "1",
            "--data-parallel-size", "8", "--data-parallel-size-local", "4", "--data-parallel-backend", "mp",
            "--enable-expert-parallel", "--all2all-backend", "allgather_reducescatter",
            "--data-parallel-address", primary_ip, "--data-parallel-rpc-port", "13345",
            "--enable-eplb", "--eplb-config", '{"num_redundant_experts":32}']
    if role == "primary":
        args += ["--host", "0.0.0.0", "--port", "8000", "--api-server-count", "1"]
    else:
        require(role == "worker", "unknown node role")
        args += ["--data-parallel-start-rank", "4", "--headless"]
    return args


def launch_node(args):
    require(os.environ.get("SLURM_JOB_ID") == native.JOB, "wrong allocation")
    require(native.sha(args.protocol) == os.environ.get("FULL_SERVING_PROTOCOL_SHA256"), "protocol changed before node launch")
    protocol = load_admitted_protocol(args.protocol)
    env = clean_server_environment(os.environ, protocol)
    if args.arm != "custom":
        command = ["bash", str(ROOT / "full_serving/launch_native_draft.sh"), args.arm, args.role,
                   str(args.snapshot), args.primary_ip, "--execute"]
    else:
        # Installed package checks run before the opt-in custom hook is enabled.
        preflight = [sys.executable, "-m", "full_serving.preflight", "runtime", "--snapshot", str(args.snapshot),
                     "--checkpoint-manifest", env["FULL_SERVING_CHECKPOINT_MANIFEST"], "--role", args.role,
                     "--primary-ip", args.primary_ip, "--output", str(Path(env["FULL_SERVING_RUN_DIR"]) / f"preflight-{args.role}.json")]
        checked = subprocess.run(preflight, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, timeout=3600)
        (Path(env["FULL_SERVING_RUN_DIR"]) / f"preflight-{args.role}.log").write_text(checked.stdout)
        require(checked.returncode == 0, "native runtime preflight failed before custom hook activation")
        custom = protocol["custom"]
        env.update(FULL_SERVING_CUSTOM_MODE=custom["mode"],
                   FULL_SERVING_CUSTOM_CONFIG=str(artifact_path(args.protocol, custom["config_file"])),
                   FULL_SERVING_CUSTOM_SOURCE_MANIFEST=str(artifact_path(args.protocol, custom["source_manifest_file"])),
                   FULL_SERVING_CUSTOM_CONFIG_SHA256=custom["config_sha256"],
                   FULL_SERVING_CUSTOM_SOURCE_MANIFEST_SHA256=custom["source_manifest_sha256"],
                   OMP_NUM_THREADS=env.get("OMP_NUM_THREADS", "4"), TOKENIZERS_PARALLELISM="false")
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT / custom["bootstrap_path"]), str(ROOT), env.get("PYTHONPATH")]))
        command = [sys.executable, "-m", custom["entrypoint"], *native_on_arguments(args.snapshot, args.primary_ip, args.role)]
    native.write_json(Path(env["FULL_SERVING_RUN_DIR"]) / f"effective-launch-{args.role}.json",
                      {"arm": args.arm, "command": command, "custom_mode": env.get("FULL_SERVING_CUSTOM_MODE"),
                       "protocol_sha256": native.sha(args.protocol)})
    os.execvpe(command[0], command, env)


def run(args):
    protocol = load_admitted_protocol(args.protocol)
    source_metadata = load_source_metadata(protocol, args.protocol)
    jobs = schedule(protocol)
    if not args.execute:
        print(json.dumps({"execute": False, "servers": len(jobs), "measured_requests": len(jobs) * 16, "schedule": jobs}, indent=2))
        return
    require(os.environ.get("SLURM_JOB_ID") == native.JOB, "run only in the authorized allocation")
    require(re.fullmatch(r"[0-9a-f]{40}", args.source_revision or ""), "full deployed Git revision required")
    out, private = args.output.resolve(), args.private_output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    private.mkdir(parents=True, exist_ok=False, mode=0o700)
    status = {"schema": "resident_compare_status_v1", "state": "admission", "started_utc": native.stamp(),
              "job_id": native.JOB, "protocol_sha256": native.sha(args.protocol), "completed_sessions": [],
              "held_out_comparison": True, "superiority_claim": False}
    def update(**values):
        status.update(values, updated_utc=native.stamp())
        native.write_json(out / "status.json", status)
    server, arm_dir = None, None
    update()
    try:
        (out / "frozen-protocol.json").write_bytes(args.protocol.read_bytes())
        native.write_json(out / "schedule.json", jobs)
        (out / "implementation").mkdir()
        for name, source in (("config.json", protocol["custom"]["config_file"]),
                             ("source-manifest.json", protocol["custom"]["source_manifest_file"]),
                             ("pilot-admission.json", protocol["pilot_admission"]["file"])):
            (out / "implementation" / name).write_bytes(artifact_path(args.protocol, source).read_bytes())
        (out / "source-admission").mkdir()
        for key, source in protocol["source_admission"].items():
            if key.endswith("_file"):
                (out / "source-admission" / (key.removesuffix("_file") + ".json")).write_bytes(artifact_path(args.protocol, source).read_bytes())
        inputs = admit_and_copy_inputs(protocol, args.protocol, args.private_data, private, out, source_metadata)
        update(state="inputs_admitted", request_admission_sha256=native.sha(out / "request-admission.json"))
        source_files = sorted((ROOT / "full_serving").rglob("*.py")) + sorted((ROOT / "full_serving").glob("*.sh"))
        native.write_json(out / "controller-source.json", {"declared_git_revision": args.source_revision,
            "files": {str(p.relative_to(ROOT)): native.sha(p) for p in source_files}})
        allocation = json.loads(native.command(["scontrol", "show", "job", native.JOB, "--json"], out / "allocation.json"))["jobs"][0]
        require("RUNNING" in allocation["job_state"], "allocation is not running")
        allocated_nodes = allocation["nodes"]
        nodes = native.command(["scontrol", "show", "hostnames", allocated_nodes]).split()
        require(len(nodes) == 2, "two nodes required")
        primary_ip = socket.gethostbyname(nodes[0])
        base_url = f"http://{primary_ip}:8000"
        native.write_json(out / "topology.json", {"nodes": nodes, "primary_ip": primary_ip, "dp": 8, "tp": 1, "ep": 8,
                                                   "activation_backend": "allgather_reducescatter"})
        native.command([sys.executable, "-m", "full_serving.preflight", "checkpoint", "--snapshot", str(args.snapshot),
                        "--output", str(out / "checkpoint-verified.json")], out / "checkpoint-preflight.log", timeout=3600)
        execution = protocol.get("execution", {})
        for job in jobs:
            arm_dir = out / job["session_id"]
            arm_dir.mkdir(parents=True, exist_ok=False)
            private_session = private / job["session_id"]
            private_session.mkdir(parents=True, exist_ok=False, mode=0o700)
            update(state="starting_server", active_session=job["session_id"])
            env = clean_server_environment(os.environ, protocol)
            env.update(SLURM_JOB_NODELIST=allocated_nodes, SLURM_JOB_NUM_NODES="2",
                       FULL_SERVING_CHECKPOINT_MANIFEST=str(out / "checkpoint-verified.json"), FULL_SERVING_RUN_DIR=str(arm_dir),
                       FULL_SERVING_PROTOCOL_SHA256=status["protocol_sha256"])
            cmd = ["srun", "--jobid=" + native.JOB, "--mpi=none", "--exact", "--nodes=2", "--ntasks=2", "--ntasks-per-node=1",
                   "--nodelist=" + ",".join(nodes), "--cpus-per-task=56", "--gpus-per-node=4", "--mem=300G",
                   "--kill-on-bad-exit=1", "--input=none", "--output=" + str(arm_dir / "server-%t.log"),
                   "--error=" + str(arm_dir / "server-%t.err"), "bash", str(ROOT / "full_serving/custom_compare_node.sh"),
                   job["arm"], str(args.snapshot), primary_ip, str(args.protocol.resolve())]
            native.write_json(arm_dir / "launch.json", {"command": cmd, "started_utc": native.stamp(), **job})
            with (arm_dir / "srun.log").open("w") as log:
                started = time.monotonic()
                server = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                models = native.wait_server(server, base_url, execution.get("startup_timeout_s", 2400), arm_dir)
                if job["arm"] == "custom":
                    native.write_json(arm_dir / "custom-startup-admission.json", verify_custom_bindings(arm_dir, protocol, args.protocol))
                native.write_json(arm_dir / "startup.json", {"startup_elapsed_s": time.monotonic() - started,
                                                           "ready_utc": native.stamp(), "models": models})
                native.write_json(arm_dir / "warmup-plan.json", {"requests": 8, "passes": 2,
                    "first_pass_diagnostics": True, "second_pass_diagnostics": False, "rate": "simultaneous pilot burst",
                    "final_payload_used_for_warmup": False,
                    "native_on_state": "Native ON may record demand and adapt its map during warmup; post-warmup state is policy-dependent. HTTP drain is not proof of committed EPLB weights.",
                    "timing_boundary": "Startup and warmup are separate from the measured request episode; no deployment-total speedup claim."})
                for phase in ("warmup-00", "warmup-01", "measured"):
                    measured = phase == "measured"
                    item = inputs[job["dataset"]][job["block_id"] if measured else "warmup"]
                    require(native.sha(item["requests"]) == item["requests_sha256"], "admitted request copy changed")
                    if measured:
                        require(native.sha(item["calendar"]) == item["calendar_sha256"], "frozen common calendar changed")
                    update(state="running_requests", active_phase=phase)
                    (arm_dir / f"{phase}-metrics-before.txt").write_bytes(native.get_url(base_url + "/metrics"))
                    client = [sys.executable, "-m", "full_serving.client", "--requests", str(item["requests"]),
                              "--base-url", base_url + "/v1", "--output", str(arm_dir / (phase + ".jsonl")),
                              "--private-output", str(private_session / (phase + "-outputs.jsonl")),
                              "--run-id", job["session_id"].replace("/", "-") + "-" + phase, "--arm", job["arm"],
                              "--block-id", job["dataset"] + "-" + job["block_id"], "--load-id", phase,
                              "--pair-id", job["dataset"] + "-" + job["block_id"] + "-" + phase,
                              "--max-inflight", "64", "--timeout-s", str(execution.get("http_timeout_s", 1800))]
                    client += ["--calendar", str(item["calendar"])] if measured else ["--rate", "0"]
                    if phase == "warmup-00":
                        client += ["--logprobs"]
                    native.command(client, arm_dir / f"{phase}-client.log", timeout=7200)
                    result = native.summary(arm_dir / (phase + ".jsonl"))
                    require(result["completed_requests"] == (16 if measured else 8), "incomplete phase; do not select successful requests")
                    (arm_dir / f"{phase}-metrics-after.txt").write_bytes(native.get_url(base_url + "/metrics"))
                update(state="stopping_server")
                native.stop_server(server, arm_dir)
                server = None
            native.write_json(arm_dir / "status.json", {"state": "session_completed", **job, "completed_utc": native.stamp(),
                "normal_eplb_commits_verified": False, "all_layer_numerical_equivalence_proved": False})
            status["completed_sessions"].append(job["session_id"])
            update(state="session_completed")
        update(state="comparison_completed", completed_utc=native.stamp())
    except BaseException as exc:
        update(state="failed", error_type=type(exc).__name__, error="Comparison stopped; preserve all completed and partial records for audit.")
        raise
    finally:
        if server is not None:
            native.stop_server(server, arm_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("plan", help="metadata-only schedule; no private input reads")
    plan.add_argument("--protocol", type=Path, required=True)
    run_parser = sub.add_parser("run", help="dry schedule unless --execute is explicit")
    for key in ("protocol", "private-data", "snapshot", "output", "private-output"):
        run_parser.add_argument("--" + key, type=Path, required=True)
    run_parser.add_argument("--source-revision")
    run_parser.add_argument("--execute", action="store_true")
    node = sub.add_parser("launch-node")
    node.add_argument("arm", choices=ARMS)
    node.add_argument("role", choices=("primary", "worker"))
    node.add_argument("snapshot", type=Path)
    node.add_argument("primary_ip")
    node.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "plan":
        print(json.dumps({"execution_allowed": False, "schedule": schedule(json.loads(args.protocol.read_bytes()), allow_draft=True)}, indent=2))
    elif args.action == "launch-node":
        launch_node(args)
    else:
        def terminate(signum, _frame):
            raise InterruptedError(f"Comparison controller received signal {signum}")
        signal.signal(signal.SIGTERM, terminate)
        signal.signal(signal.SIGINT, terminate)
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Pilot-only full-model admission before the held-out three-arm protocol.

Only the already executed eight smoke inputs per dataset are admitted. This
controller cannot open final data, and these fixed-order runs are not final
performance evidence. Configurations and original failures remain archived.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from full_serving import orchestrate as native
from full_serving.custom_compare import native_on_arguments
from full_serving.resident_plugin import validate_config, verify_source_manifest

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("wildchat", "sharegpt", "instructcoder", "numinamath", "longbench")


def launch_node(args):
    assert os.environ.get("SLURM_JOB_ID") == native.JOB
    env = dict(os.environ)
    for key in ("FULL_SERVING_CUSTOM_MODE", "VLLM_PLUGINS", "PYTHONPATH"):
        env.pop(key, None)
    if args.arm == "off":
        command = ["bash", str(ROOT/"full_serving/launch_native_draft.sh"), "off", args.role, str(args.snapshot), args.ip, "--execute"]
    else:
        cfg = validate_config(json.loads(args.config.read_bytes()))
        verify_source_manifest(json.loads(args.sources.read_bytes()))
        native.command([sys.executable, "-m", "full_serving.preflight", "runtime", "--snapshot", str(args.snapshot),
                        "--checkpoint-manifest", env["FULL_SERVING_CHECKPOINT_MANIFEST"], "--role", args.role,
                        "--primary-ip", args.ip, "--output", str(Path(env["FULL_SERVING_RUN_DIR"])/f"preflight-{args.role}.json")],
                       Path(env["FULL_SERVING_RUN_DIR"])/f"preflight-{args.role}.log", timeout=3600)
        env.update(FULL_SERVING_CUSTOM_MODE=cfg["mode"], FULL_SERVING_CUSTOM_CONFIG=str(args.config.resolve()),
                   FULL_SERVING_CUSTOM_SOURCE_MANIFEST=str(args.sources.resolve()),
                   FULL_SERVING_CUSTOM_CONFIG_SHA256=native.sha(args.config),
                   FULL_SERVING_CUSTOM_SOURCE_MANIFEST_SHA256=native.sha(args.sources))
        command = [sys.executable, "-m", "full_serving.custom_entrypoint", *native_on_arguments(args.snapshot, args.ip, args.role)]
    native.write_json(Path(env["FULL_SERVING_RUN_DIR"])/f"effective-launch-{args.role}.json",
                      {"pilot_only": True, "arm": args.arm, "command": command, "config_sha256": native.sha(args.config),
                       "source_manifest_sha256": native.sha(args.sources)})
    os.execvpe(command[0], command, env)


def run(args):
    assert os.environ.get("SLURM_JOB_ID") == native.JOB
    validate_config(json.loads(args.config.read_bytes()))
    verify_source_manifest(json.loads(args.sources.read_bytes()))
    args.output.mkdir(parents=True, exist_ok=False)
    args.private_output.mkdir(parents=True, exist_ok=False, mode=0o700)
    manifest = json.loads(args.suite_manifest.read_bytes())
    source_cohorts = {x["dataset"]: x for x in manifest["cohorts"]}
    for dataset in DATASETS:
        path = args.private_data/dataset/"smoke.jsonl"
        assert native.sha(path) == source_cohorts[dataset]["private_files"]["smoke.jsonl"]
        rows = native.read_jsonl(path)
        assert len(rows) == 8 and [x["id"] for x in rows] == source_cohorts[dataset]["smoke_request_ids"]
    status = {"schema": "full_resident_pilot_status_v1", "started_utc": native.stamp(), "pilot_only": True,
              "arm": args.arm, "config_sha256": native.sha(args.config), "source_manifest_sha256": native.sha(args.sources),
              "completed_phases": [], "final_inputs_opened": False}
    def update(**changes):
        status.update(changes, updated_utc=native.stamp())
        native.write_json(args.output/"status.json", status)
    server = None
    update(state="preflight")
    try:
        allocation = json.loads(native.command(["scontrol", "show", "job", native.JOB, "--json"], args.output/"allocation.json"))["jobs"][0]
        assert "RUNNING" in allocation["job_state"]
        nodes = native.command(["scontrol", "show", "hostnames", allocation["nodes"]]).split()
        assert len(nodes) == 2
        ip = socket.gethostbyname(nodes[0])
        url = f"http://{ip}:8000"
        native.write_json(args.output/"controller-source.json", {"git_revision": args.source_revision,
                          "files": {str(p.relative_to(ROOT)): native.sha(p) for p in sorted((ROOT/"full_serving").glob("*.py"))},
                          "suite_manifest_sha256": native.sha(args.suite_manifest)})
        native.command([sys.executable, "-m", "full_serving.preflight", "checkpoint", "--snapshot", str(args.snapshot),
                        "--output", str(args.output/"checkpoint-verified.json")], args.output/"checkpoint-preflight.log", timeout=3600)
        env = native.server_environment(os.environ)
        for key in ("FULL_SERVING_CUSTOM_MODE", "VLLM_PLUGINS", "PYTHONPATH"):
            env.pop(key, None)
        env.update(SLURM_JOB_NODELIST=allocation["nodes"], SLURM_JOB_NUM_NODES="2",
                   FULL_SERVING_CHECKPOINT_MANIFEST=str(args.output/"checkpoint-verified.json"), FULL_SERVING_RUN_DIR=str(args.output))
        command = ["srun", "--jobid="+native.JOB, "--mpi=none", "--exact", "--nodes=2", "--ntasks=2", "--ntasks-per-node=1",
                   "--nodelist="+",".join(nodes), "--cpus-per-task=56", "--gpus-per-node=4", "--mem=300G", "--kill-on-bad-exit=1", "--input=none",
                   "--output="+str(args.output/"server-%t.log"), "--error="+str(args.output/"server-%t.err"),
                   "bash", str(ROOT/"full_serving/custom_pilot_node.sh"), args.arm, str(args.snapshot), ip, str(args.config), str(args.sources)]
        native.write_json(args.output/"launch.json", {"command": command, "started_utc": native.stamp()})
        update(state="starting_server")
        started = time.monotonic()
        with (args.output/"srun.log").open("w") as log:
            server = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            native.wait_server(server, url, 2400, args.output)
            native.write_json(args.output/"startup.json", {"startup_elapsed_s": time.monotonic()-started})
            if args.arm != "off":
                cfg = json.loads(args.config.read_bytes())
                for rank in range(8):
                    binding = json.loads((args.output/f"resident-bindings-rank{rank}.json").read_bytes())
                    assert binding["layer_count"] == 48 and binding["ep_rank"] == rank
                    assert binding["config_sha256"] == native.sha(args.config)
                    assert binding["source_manifest_sha256"] == native.sha(args.sources)
                    if cfg.get("startup_diagnostic"):
                        diagnostic = json.loads((args.output/f"resident-startup-diagnostic-rank{rank}.json").read_bytes())
                        assert diagnostic["status"] == "passed"
            for dataset in DATASETS:
                for phase in range(args.passes):
                    label = f"{dataset}-pass{phase}"
                    update(state="running_requests", dataset=dataset, phase=phase)
                    (args.output/f"{label}-metrics-before.txt").write_bytes(native.get_url(url+"/metrics"))
                    cmd = [sys.executable, "-m", "full_serving.client", "--requests", str(args.private_data/dataset/"smoke.jsonl"),
                           "--base-url", url+"/v1", "--output", str(args.output/f"{label}.jsonl"),
                           "--private-output", str(args.private_output/f"{label}-outputs.jsonl"),
                           "--run-id", f"custom-pilot-{args.arm}-{label}", "--arm", args.arm,
                           "--block-id", dataset+"-resident-pilot", "--load-id", f"pass{phase}",
                           "--max-inflight", "64", "--timeout-s", "1800", "--rate", "2"]
                    if phase == 0:
                        cmd.append("--logprobs")
                    native.command(cmd, args.output/f"{label}-client.log", timeout=5400)
                    result = native.summary(args.output/f"{label}.jsonl")
                    assert result["completed_requests"] == 8 and result["failed_requests"] == 0
                    (args.output/f"{label}-metrics-after.txt").write_bytes(native.get_url(url+"/metrics"))
                    status["completed_phases"].append({"dataset": dataset, "phase": phase, "summary": result})
                    update()
            update(state="pilot_completed", completed_utc=native.stamp())
    except BaseException as exc:
        update(state="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if server is not None:
            native.stop_server(server, args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    node = sub.add_parser("launch-node")
    for name in ("arm", "role", "snapshot", "ip"):
        node.add_argument(name, type=Path if name == "snapshot" else str)
    node.add_argument("--config", type=Path, required=True)
    node.add_argument("--sources", type=Path, required=True)
    control = sub.add_parser("run")
    control.add_argument("--arm", choices=("off", "resident", "placement_only"), required=True)
    control.add_argument("--passes", type=int, choices=(1, 2), default=2)
    for name in ("snapshot", "config", "sources", "suite-manifest", "private-data", "output", "private-output"):
        control.add_argument("--"+name, type=Path, required=True)
    control.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    def stop(signum, frame):
        raise InterruptedError(f"Controller signal {signum}")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    (launch_node if args.command == "launch-node" else run)(args)


if __name__ == "__main__":
    main()

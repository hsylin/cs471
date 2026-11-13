"""Run a bounded, real full-model pilot in the authorized eight-H100 allocation.

This is a feasibility/calibration pilot, not the final inferential experiment.
It never changes native EPLB intervals or imports the layer-only dispatcher.
The off arm chooses the pilot arrival scale before on-arm outcomes exist.
Every stage uses disjoint conversations and all requests generate real outputs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
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
import urllib.error
import urllib.request

JOB = "4911287"


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().split("\n") if line.strip()]


def split_pilot(pilot, smoke):
    """Keep warmup, capacity calibration and external-arrival content disjoint."""
    if len(pilot) != 64 or len(smoke) != 8:
        raise ValueError("This pilot requires 64 pilot conversations and 8 smoke requests")
    ids = [r["id"] for r in pilot]
    smoke_ids = {r["id"] for r in smoke}
    if len(set(ids)) != 64 or len(smoke_ids) != 8 or not smoke_ids <= set(ids):
        raise ValueError("Invalid pilot/smoke identities")
    remaining = [r for r in pilot if r["id"] not in smoke_ids]
    return {"smoke": smoke, "calibration": remaining[:32], "external": remaining[32:]}


def scale_arrivals(block, request_ids, rate):
    """Preserve source order and relative burstiness using exact rational scaling."""
    offsets = block["original_offsets_ns"]
    horizon = block["original_horizon_ns"]
    if len(offsets) != len(request_ids) or not offsets or offsets[0] != 0:
        raise ValueError("External block and request count disagree")
    if (horizon <= offsets[-1] or offsets != sorted(offsets)
            or any(not isinstance(x, int) or x < 0 for x in offsets)):
        raise ValueError("Invalid external arrival timestamps")
    target = Fraction(len(offsets) * 10**9, 1) / Fraction(str(rate))
    scale = target / horizon
    return {"schema": "full_serving_calendar_v1", "source_block": block,
            "request_ids": request_ids,
            "arrival_offsets_ns": [int(x * scale) for x in offsets],
            "horizon_ns": int(target), "absolute_arrivals": True,
            "offered_requests_per_s": rate,
            "scale_numerator": scale.numerator, "scale_denominator": scale.denominator,
            "mapping": "Independently selected input requests assigned in frozen order to recorded external timestamps; not a joint production trace",
            "source_request_sizes_executed": False}


def summary(path):
    rows = read_jsonl(path)
    summaries = [r for r in rows if r.get("type") == "run_summary"]
    if len(summaries) != 1 or not summaries[0]["valid"]:
        raise RuntimeError(f"Incomplete/invalid request run: {path}")
    return summaries[0]


def command(args, output=None, timeout=120):
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=timeout)
    if output:
        Path(output).write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {args[0]}; see archived log")
    return result.stdout


def get_url(url, timeout=5):
    # Never route private cluster requests through environment HTTP proxies.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=timeout) as response:
        return response.read()


def startup_failure(arm_dir):
    """Detect native worker failure even while the API parent is still alive."""
    if arm_dir is None:
        return None
    fatal = re.compile(r"\bERROR\b.*(?:WorkerProc hit an exception|EngineCore failed to start)")
    for path in sorted(Path(arm_dir).glob("server-*.*")):
        if path.suffix not in (".log", ".err"):
            continue
        for line in path.read_text(errors="replace").split("\n"):
            if fatal.search(line):
                return f"Native worker failed during startup; inspect {path.name}"
    return None


def wait_server(process, base_url, timeout, arm_dir=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server step exited during startup: {process.returncode}")
        failure = startup_failure(arm_dir)
        if failure:
            raise RuntimeError(failure)
        try:
            get_url(base_url + "/health")
            models = json.loads(get_url(base_url + "/v1/models"))
            if any(m["id"] == "qwen3-full" for m in models.get("data", [])):
                return models
        except (OSError, ValueError, urllib.error.URLError):
            pass
        time.sleep(5)
    raise TimeoutError("Full-model startup exceeded its declared limit")


def server_environment(environment):
    """Do not inherit zero-GPU controller step resource options into srun."""
    options = {"SLURM_CPUS_PER_TASK", "SRUN_CPUS_PER_TASK", "SLURM_NTASKS", "SLURM_NTASKS_PER_NODE",
               "SLURM_NNODES", "SLURM_NPROCS", "SLURM_MEM_PER_CPU", "SLURM_MEM_PER_NODE",
               "SLURM_MEM_PER_GPU", "SLURM_GPUS", "SLURM_GPUS_PER_NODE", "SLURM_GPUS_PER_TASK",
               "SLURM_GPUS_PER_SOCKET", "SLURM_CPUS_PER_GPU", "SLURM_TRES_PER_TASK", "SLURM_TRES_BIND",
               "SLURM_TRES_FREQ", "SLURM_GPU_BIND", "SLURM_GPU_FREQ", "SLURM_EXACT", "CUDA_VISIBLE_DEVICES",
               "SLURM_GRES", "SLURM_NTASKS_PER_GPU"}
    return {k: v for k, v in environment.items()
            if k not in options and not k.startswith(("SLURM_STEP_", "SLURM_CPU_BIND"))}


def stop_server(process, arm_dir):
    """Signal the recorded serving step; never cancel the holding reservation."""
    if process.poll() is None:
        records = list(Path(arm_dir).glob("server-step-*.txt"))
        steps = {p.read_text().strip() for p in records}
        if any(not re.fullmatch(JOB + r"\.\d+", step) for step in steps) or len(steps) > 1:
            raise RuntimeError("Invalid server step identity; refusing broad cancellation")
        if steps:
            step = steps.pop()
            try:
                subprocess.run(["scancel", "--signal=TERM", step], check=True, timeout=30)
            except subprocess.CalledProcessError:
                if process.poll() is None:
                    raise
        else:
            # No node task has started. Cancel only this srun launcher.
            process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=90)
        except subprocess.TimeoutExpired:
            if records:
                try:
                    subprocess.run(["scancel", "--signal=KILL", step], check=True, timeout=30)
                except subprocess.CalledProcessError:
                    if process.poll() is None:
                        raise
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=30)


def paired_smoke_diagnostic(off_file, on_file):
    def requests(path):
        return {r["request_id"]: r for r in read_jsonl(path) if r.get("type") == "request"}
    off, on = requests(off_file), requests(on_file)
    if set(off) != set(on):
        raise ValueError("Smoke identities differ")
    pairs = [{"request_id": key, "same_generated_text_hash": off[key]["output_sha256"] == on[key]["output_sha256"],
              "off_usage": off[key]["usage"], "on_usage": on[key]["usage"],
              "off_finish_reason": off[key]["finish_reason"], "on_finish_reason": on[key]["finish_reason"]}
             for key in sorted(off)]
    return {"pairs": pairs,
            "all_generated_text_hashes_equal": all(p["same_generated_text_hash"] for p in pairs),
            "interpretation": "Equality is a smoke check, not an all-layer numerical/routing proof. Divergent deterministic generations require logprob/numerical investigation before a final correctness claim.",
            "routing_numerical_and_commit_consistency_verified": False}


def run(args):
    if os.environ.get("SLURM_JOB_ID") != JOB:
        raise RuntimeError("Run only inside authorized allocation 4882284")
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    private = args.private_output.resolve()
    private.mkdir(parents=True, mode=0o700, exist_ok=False)
    status = {"schema": "full_serving_controller_v1", "job_id": JOB, "started_utc": stamp(),
              "state": "preflight", "kind": "full_model_functional_calibration_pilot",
              "final_measurements": False, "custom_dispatcher_installed": False,
              "dataset": getattr(args, "dataset_id", "wildchat")}
    dataset_id = status["dataset"]
    def update(**values):
        status.update(values, updated_utc=stamp())
        write_json(out / "status.json", status)
    update()
    server = None
    arm_dir = None
    try:
        allocation_text = command(["scontrol", "show", "job", JOB, "--json"], out / "allocation.json")
        job = json.loads(allocation_text)["jobs"][0]
        states = job["job_state"]
        if not (states == "RUNNING" or "RUNNING" in states):
            raise RuntimeError("Allocation is not running")
        # A one-node controller step can rewrite SLURM_JOB_NODELIST. Query the
        # allocation record rather than treating the current step as the job.
        allocated_nodes = job["nodes"]
        host_text = command(["scontrol", "show", "hostnames", allocated_nodes])
        nodes = host_text.split()
        if len(nodes) != 2:
            raise RuntimeError("Expected two allocated nodes")
        primary_ip = socket.gethostbyname(nodes[0])
        base_url = f"http://{primary_ip}:8000"
        write_json(out / "topology.json", {"nodes": nodes, "primary_ip": primary_ip,
                   "dp": 8, "tp": 1, "ep": 8, "activation_backend": "allgather_reducescatter"})
        command([sys.executable, "-m", "full_serving.preflight", "checkpoint", "--snapshot", str(args.snapshot),
                 "--output", str(out / "checkpoint-verified.json")], out / "checkpoint-preflight.log", timeout=3600)
        source_files = sorted((root / "full_serving").glob("*.py")) + sorted((root / "full_serving").glob("*.sh"))
        revision = getattr(args, "source_revision", None) or os.environ.get("FULL_SERVING_SOURCE_REVISION")
        if revision is None:
            revision = command(["git", "rev-parse", "HEAD"]).strip()
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise RuntimeError("A full deployed Git revision is required")
        write_json(out / "controller-source.json", {"declared_git_revision": revision,
                   "deployment": "Exact source files are hashed below; cluster copy need not contain Git metadata",
                   "files": {str(p.relative_to(root)): sha(p) for p in source_files},
                   "data_manifest_sha256": sha(args.data_manifest), "arrival_manifest_sha256": sha(args.arrival_manifest)})
        subsets = split_pilot(read_jsonl(args.pilot), read_jsonl(args.smoke))
        for stage, rows in subsets.items():
            path = private / f"{stage}-requests.jsonl"
            path.write_text("".join(json.dumps(r, ensure_ascii=True) + "\n" for r in rows))
            path.chmod(0o600)
        write_json(out / "pilot-splits.json", {stage: {"ids": [r["id"] for r in rows],
                    "request_file_sha256": sha(private / f"{stage}-requests.jsonl")} for stage, rows in subsets.items()})
        arrivals = json.loads(args.arrival_manifest.read_text())
        pilot_blocks = [b for b in arrivals["blocks"] if b["split"] == "pilot"]
        if len(pilot_blocks) != 1 or len(pilot_blocks[0]["original_offsets_ns"]) != 24:
            raise ValueError("One 24-request external pilot block is required")
        phases = []
        for arm in ("off", "on"):
            arm_dir = out / arm
            arm_dir.mkdir()
            env = dict(server_environment(os.environ),
                       SLURM_JOB_NODELIST=allocated_nodes, SLURM_JOB_NUM_NODES="2",
                       FULL_SERVING_CHECKPOINT_MANIFEST=str(out / "checkpoint-verified.json"), FULL_SERVING_RUN_DIR=str(arm_dir))
            cmd = ["srun", "--jobid=" + JOB, "--mpi=none", "--exact", "--nodes=2", "--ntasks=2", "--ntasks-per-node=1",
                   "--nodelist=" + ",".join(nodes),
                   "--cpus-per-task=56", "--gpus-per-node=4", "--mem=300G", "--kill-on-bad-exit=1", "--input=none",
                   "--output=" + str(arm_dir / "server-%t.log"), "--error=" + str(arm_dir / "server-%t.err"),
                   "bash", str(root / "full_serving/run_node.sh"), arm, str(args.snapshot), primary_ip]
            write_json(arm_dir / "launch.json", {"command": cmd, "started_utc": stamp()})
            update(state="starting_server", arm=arm)
            started = time.monotonic()
            with (arm_dir / "srun.log").open("w") as log:
                server = subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                          start_new_session=True)
                models = wait_server(server, base_url, args.startup_timeout, arm_dir)
                write_json(arm_dir / "startup.json", {"startup_elapsed_s": time.monotonic() - started,
                           "ready_utc": stamp(), "models": models})
                for stage in ("smoke", "calibration", "external"):
                    update(state="running_requests", arm=arm, stage=stage)
                    (arm_dir / f"{stage}-metrics-before.txt").write_bytes(get_url(base_url + "/metrics"))
                    client = [sys.executable, "-m", "full_serving.client", "--requests", str(private / f"{stage}-requests.jsonl"),
                              "--base-url", base_url + "/v1", "--output", str(arm_dir / f"{stage}.jsonl"),
                              "--private-output", str(private / f"{arm}-{stage}-outputs.jsonl"),
                              "--run-id", f"full-pilot-{dataset_id}-{arm}-{stage}", "--arm", arm,
                              "--block-id", dataset_id + "-pilot-" + stage,
                              "--load-id", stage, "--max-inflight", "64", "--timeout-s", "1800"]
                    if stage == "external":
                        client += ["--calendar", str(out / "pilot-calendar.json")]
                    else:
                        client += ["--rate", "2" if stage == "smoke" else "0"]
                    if stage == "smoke":
                        client += ["--logprobs"]
                    command(client, arm_dir / f"{stage}-client.log", timeout=5400)
                    result = summary(arm_dir / f"{stage}.jsonl")
                    phases.append({"arm": arm, "stage": stage, "summary": result})
                    write_json(out / "completed-phases.json", phases)
                    (arm_dir / f"{stage}-metrics-after.txt").write_bytes(get_url(base_url + "/metrics"))
                    if arm == "off" and stage == "calibration":
                        # Pilot scaling only; do not infer final sustainable capacity from one burst.
                        rate = 0.6 * result["completed_requests"] / result["completed_elapsed_s"]
                        cal = scale_arrivals(pilot_blocks[0], [r["id"] for r in subsets["external"]], rate)
                        cal["rate_selection"] = {"rule": "0.6 * off pilot burst completed requests / completed elapsed", 
                                                 "calibration_file_sha256": sha(arm_dir / "calibration.jsonl"),
                                                 "not_a_final_capacity_estimate": True}
                        write_json(out / "pilot-calendar.json", cal)
                update(state="stopping_server", arm=arm)
                stop_server(server, arm_dir)
                server = None
        write_json(out / "smoke-output-comparison.json", paired_smoke_diagnostic(out / "off/smoke.jsonl", out / "on/smoke.jsonl"))
        update(state="pilot_completed", completed_utc=stamp(),
               next_step="Inspect generated replies, logprobs, source/backend/hardware and capacity before freezing independent final runs. No final superiority claim is authorized by this fixed-order pilot.")
    except BaseException as exc:
        update(state="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if server is not None:
            stop_server(server, arm_dir)


def main():
    def terminate(signum, _frame):
        raise InterruptedError(f"Controller received signal {signum}")
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--pilot", type=Path, required=True)
    p.add_argument("--smoke", type=Path, required=True)
    p.add_argument("--data-manifest", type=Path, required=True)
    p.add_argument("--arrival-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--private-output", type=Path, required=True)
    p.add_argument("--startup-timeout", type=float, default=2400)
    p.add_argument("--source-revision", help="Full Git SHA for a deployment without .git metadata")
    p.add_argument("--dataset-id", default="wildchat")
    run(p.parse_args())


if __name__ == "__main__":
    main()

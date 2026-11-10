"""Run separate dataset pilots sequentially on the same eight H100s.

Each dataset starts fresh native off/on servers. No content is pooled, and no
final inference is made from these single-block calibration pilots. BurstGPT
is a separate, explicitly repeated-content timing diagnostic using ShareGPT.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import time
from types import SimpleNamespace

from full_serving import orchestrate


def verify_private_files(cohort, directory):
    for filename, expected in cohort["private_files"].items():
        if Path(filename).name != filename or orchestrate.sha(directory / filename) != expected:
            raise ValueError(f"Private suite file mismatch: {cohort['dataset']}/{filename}")


def wait_for_pilot(path, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            status = json.loads(path.read_text())
            if status["state"] == "pilot_completed":
                return status
            if status["state"] == "failed":
                raise RuntimeError("Prior full-model pilot failed; inspect its preserved logs before continuing")
        time.sleep(30)
    raise TimeoutError("Prior pilot did not complete within the declared waiting period")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--private-data", type=Path, required=True)
    p.add_argument("--suite-manifest", type=Path, required=True)
    p.add_argument("--admission", type=Path, required=True)
    p.add_argument("--arrivals", type=Path, required=True)
    p.add_argument("--burstgpt-block", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--private-output", type=Path, required=True)
    p.add_argument("--source-revision", required=True)
    p.add_argument("--wait-for", type=Path)
    p.add_argument("--wait-timeout", type=float, default=14400)
    p.add_argument("--datasets", nargs="+", default=["wildchat", "sharegpt", "instructcoder", "numinamath", "longbench"])
    p.add_argument("--startup-timeout", type=float, default=2400)
    args = p.parse_args()
    if os.environ.get("SLURM_JOB_ID") != orchestrate.JOB:
        raise RuntimeError("Suite runs only in allocation 4882284")
    args.output.mkdir(parents=True, exist_ok=False)
    args.private_output.mkdir(parents=True, mode=0o700, exist_ok=False)
    def terminate(signum, _frame):
        raise InterruptedError(f"Suite received signal {signum}")
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    status = {"state": "admission", "started_utc": orchestrate.stamp(), "completed_datasets": [],
              "datasets_kept_separate": True, "final_measurements": False, "source_revision": args.source_revision}
    def update(**fields):
        status.update(fields, updated_utc=orchestrate.stamp())
        orchestrate.write_json(args.output / "status.json", status)
    try:
        admission = json.loads(args.admission.read_text())
        if (admission.get("status") != "passed"
                or admission.get("suite_manifest_file_sha256") != orchestrate.sha(args.suite_manifest)):
            raise ValueError("Suite needs a passed private/source admission report")
        suite = json.loads(args.suite_manifest.read_text())
        cohorts = {c["dataset"]: c for c in suite["cohorts"]}
        if len(set(args.datasets)) != len(args.datasets) or not set(args.datasets) <= set(cohorts):
            raise ValueError("Invalid or duplicate dataset list")
        for dataset in args.datasets:
            verify_private_files(cohorts[dataset], args.private_data / dataset)
        orchestrate.write_json(args.output / "inputs.json", {
            "suite_manifest_sha256": orchestrate.sha(args.suite_manifest),
            "admission_sha256": orchestrate.sha(args.admission),
            "arrivals_sha256": orchestrate.sha(args.arrivals), "burstgpt_sha256": orchestrate.sha(args.burstgpt_block),
            "dataset_order": args.datasets,
            "order_limit": "Fixed dataset/arm order is for feasibility and calibration; not final superiority evidence"})
        if args.wait_for:
            update(state="waiting_for_initial_pilot", prior_status=str(args.wait_for))
            prior = wait_for_pilot(args.wait_for, args.wait_timeout)
            orchestrate.write_json(args.output / "prior-pilot.json", prior)
        for dataset in args.datasets:
            update(state="running_dataset", dataset=dataset)
            child = SimpleNamespace(snapshot=args.snapshot,
                pilot=args.private_data / dataset / "pilot.jsonl", smoke=args.private_data / dataset / "smoke.jsonl",
                data_manifest=args.suite_manifest, arrival_manifest=args.arrivals,
                output=args.output / dataset, private_output=args.private_output / dataset,
                startup_timeout=args.startup_timeout, source_revision=args.source_revision, dataset_id=dataset)
            orchestrate.run(child)
            status["completed_datasets"].append(dataset)
            update(state="dataset_completed")
        if "sharegpt" in args.datasets:
            block = json.loads(args.burstgpt_block.read_text())
            if block["kind"] != "metadata_only_no_prompt_text":
                raise ValueError("Expected a trace-only BurstGPT diagnostic")
            arrival_path = args.output / "burstgpt-arrival-manifest.json"
            orchestrate.write_json(arrival_path, {"blocks": [{**block, "split": "pilot"}],
                "content": "Repeated ShareGPT pilot; not BurstGPT prompts or fresh payloads",
                "original_request_lengths_executed": False})
            update(state="running_burstgpt_diagnostic", dataset="sharegpt-burstgpt-diagnostic")
            child = SimpleNamespace(snapshot=args.snapshot,
                pilot=args.private_data / "sharegpt/pilot.jsonl", smoke=args.private_data / "sharegpt/smoke.jsonl",
                data_manifest=args.suite_manifest, arrival_manifest=arrival_path,
                output=args.output / "sharegpt-burstgpt-diagnostic",
                private_output=args.private_output / "sharegpt-burstgpt-diagnostic",
                startup_timeout=args.startup_timeout, source_revision=args.source_revision,
                dataset_id="sharegpt-burstgpt-diagnostic")
            orchestrate.run(child)
        update(state="suite_pilots_completed", completed_utc=orchestrate.stamp())
    except BaseException as exc:
        update(state="failed", error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == "__main__":
    main()

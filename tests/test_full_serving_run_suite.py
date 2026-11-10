"""CPU tests of cohort orchestration; no server or Slurm commands execute."""
import json
import sys

import pytest

from full_serving import run_suite
from full_serving.data import file_digest


DATASETS = ["wildchat", "sharegpt", "instructcoder", "numinamath", "longbench"]


def configure(tmp_path, monkeypatch, datasets=None):
    private = tmp_path / "private-input"
    cohorts = []
    for dataset in DATASETS:
        directory = private / dataset
        directory.mkdir(parents=True)
        files = {}
        for filename in ("requests.jsonl", "pilot.jsonl", "final.jsonl", "smoke.jsonl"):
            path = directory / filename
            path.write_text(json.dumps({"dataset": dataset, "subset": filename}) + "\n")
            files[filename] = file_digest(path)
        cohorts.append({"dataset": dataset, "private_files": files})
    manifest = tmp_path / "suite.json"
    manifest.write_text(json.dumps({"cohorts": cohorts, "selection_status": "pilot_ready_final_candidates_not_frozen"}))
    admission = tmp_path / "admission.json"
    admission.write_text(json.dumps({"status": "passed", "suite_manifest_file_sha256": file_digest(manifest)}))
    arrivals = tmp_path / "arrivals.json"
    arrivals.write_text(json.dumps({"blocks": [{"split": "pilot", "original_offsets_ns": list(range(24))}]}))
    burst = tmp_path / "burst.json"
    burst.write_text(json.dumps({"kind": "metadata_only_no_prompt_text", "original_offsets_ns": list(range(24)),
                                 "original_horizon_ns": 30, "source_lengths_are_qwen_token_counts": False}))
    public_output, private_output = tmp_path / "public-results", tmp_path / "private-results"
    argv = ["run_suite", "--snapshot", str(tmp_path / "model"), "--private-data", str(private),
            "--suite-manifest", str(manifest), "--admission", str(admission), "--arrivals", str(arrivals),
            "--burstgpt-block", str(burst), "--output", str(public_output), "--private-output", str(private_output),
            "--source-revision", "a" * 40]
    if datasets is not None:
        argv += ["--datasets", *datasets]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setenv("SLURM_JOB_ID", run_suite.orchestrate.JOB)
    monkeypatch.setattr(run_suite.signal, "signal", lambda *args: None)
    calls = []
    monkeypatch.setattr(run_suite.orchestrate, "run", lambda args: calls.append(args))
    return {"calls": calls, "manifest": manifest, "admission": admission, "private": private,
            "output": public_output, "private_output": private_output, "burst": burst}


def test_each_dataset_has_separate_pilot_and_fresh_child_output_without_finals(tmp_path, monkeypatch):
    setup = configure(tmp_path, monkeypatch)
    run_suite.main()
    assert [args.dataset_id for args in setup["calls"]] == DATASETS + ["sharegpt-burstgpt-diagnostic"]
    for args, dataset in zip(setup["calls"], DATASETS):
        assert args.pilot == setup["private"] / dataset / "pilot.jsonl"
        assert args.smoke == setup["private"] / dataset / "smoke.jsonl"
        assert args.output == setup["output"] / dataset
        assert args.private_output == setup["private_output"] / dataset
        assert not any("final.jsonl" in str(value) for value in vars(args).values())
    status = json.loads((setup["output"] / "status.json").read_text())
    assert status["state"] == "suite_pilots_completed"
    assert status["completed_datasets"] == DATASETS
    assert status["datasets_kept_separate"] is True and status["final_measurements"] is False


def test_burstgpt_is_explicit_repeated_sharegpt_content_diagnostic(tmp_path, monkeypatch):
    setup = configure(tmp_path, monkeypatch, ["sharegpt"])
    run_suite.main()
    regular, diagnostic = setup["calls"]
    assert regular.pilot == diagnostic.pilot and regular.smoke == diagnostic.smoke
    assert diagnostic.dataset_id == "sharegpt-burstgpt-diagnostic"
    block = json.loads(diagnostic.arrival_manifest.read_text())
    assert "Repeated ShareGPT pilot" in block["content"]
    assert "not BurstGPT prompts or fresh payloads" in block["content"]
    assert block["original_request_lengths_executed"] is False
    assert block["blocks"][0]["kind"] == "metadata_only_no_prompt_text"


def test_child_failure_stops_later_datasets_and_diagnostic(tmp_path, monkeypatch):
    setup = configure(tmp_path, monkeypatch)
    def run(args):
        setup["calls"].append(args)
        if args.dataset_id == "instructcoder":
            raise RuntimeError("injected child failure")
    monkeypatch.setattr(run_suite.orchestrate, "run", run)
    with pytest.raises(RuntimeError, match="injected"):
        run_suite.main()
    assert [args.dataset_id for args in setup["calls"]] == DATASETS[:3]
    status = json.loads((setup["output"] / "status.json").read_text())
    assert status["state"] == "failed" and status["completed_datasets"] == DATASETS[:2]
    assert status["final_measurements"] is False


@pytest.mark.parametrize("mutation", ["admission", "manifest", "private_payload"])
def test_invalid_admission_or_payload_prevents_every_child_launch(tmp_path, monkeypatch, mutation):
    setup = configure(tmp_path, monkeypatch)
    if mutation == "admission":
        setup["admission"].write_text(json.dumps({"status": "failed"}))
    elif mutation == "manifest":
        with setup["manifest"].open("a") as handle:
            handle.write("\n")  # Same JSON meaning, different exact admitted bytes.
    else:
        (setup["private"] / "longbench/pilot.jsonl").write_text("changed")
    with pytest.raises(ValueError):
        run_suite.main()
    assert setup["calls"] == []
    assert json.loads((setup["output"] / "status.json").read_text())["state"] == "failed"


def test_subset_without_sharegpt_has_no_burstgpt_run(tmp_path, monkeypatch):
    setup = configure(tmp_path, monkeypatch, ["numinamath", "longbench"])
    run_suite.main()
    assert [args.dataset_id for args in setup["calls"]] == ["numinamath", "longbench"]
    assert not (setup["output"] / "burstgpt-arrival-manifest.json").exists()


def test_failed_prior_pilot_does_not_wait_or_continue(tmp_path, monkeypatch):
    path = tmp_path / "prior-status.json"
    path.write_text(json.dumps({"state": "failed"}))
    monkeypatch.setattr(run_suite.time, "sleep", lambda _: pytest.fail("Failed prior pilot should stop immediately"))
    with pytest.raises(RuntimeError, match="Prior full-model pilot failed"):
        run_suite.wait_for_pilot(path, 30)

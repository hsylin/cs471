"""Controller checks use fake Slurm/server processes; no cluster allocation."""
import json
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest

from full_serving import orchestrate


def make_requests(count=64):
    return [{"id": str(i), "messages": [{"role": "user", "content": f"conversation {i}"}],
             "input_tokens": 4} for i in range(count)]


def test_pilot_content_is_disjoint_and_unicode_jsonl_preserves_input(tmp_path):
    pilot = make_requests()
    pilot[0]["messages"][0]["content"] = "preserve\u2028this\u2029conversation"
    path = tmp_path / "pilot.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in pilot) + "\n")
    loaded = orchestrate.read_jsonl(path)
    assert loaded == pilot
    smoke = pilot[::8]
    stages = orchestrate.split_pilot(loaded, smoke)
    assert {stage: len(rows) for stage, rows in stages.items()} == {"smoke": 8, "calibration": 32, "external": 24}
    groups = [{r["id"] for r in rows} for rows in stages.values()]
    assert len(set.union(*groups)) == 64
    assert all(not left & right for index, left in enumerate(groups) for right in groups[index + 1:])
    with pytest.raises(ValueError):
        orchestrate.split_pilot(pilot, [pilot[0]] * 8)


def test_external_arrivals_preserve_bursts_and_are_scaled_once():
    source = {"original_offsets_ns": [0, 0, 10, 30], "original_horizon_ns": 40}
    result = orchestrate.scale_arrivals(source, ["a", "b", "c", "d"], rate=2)
    assert result["arrival_offsets_ns"] == [0, 0, 500_000_000, 1_500_000_000]
    assert result["horizon_ns"] == 2_000_000_000
    assert result["source_block"] == source
    assert source["original_offsets_ns"] == [0, 0, 10, 30]
    assert result["absolute_arrivals"] is True
    assert result["source_request_sizes_executed"] is False
    with pytest.raises(ValueError):
        orchestrate.scale_arrivals(source, ["a"], rate=2)
    with pytest.raises(ValueError):
        orchestrate.scale_arrivals({**source, "original_horizon_ns": 30}, ["a", "b", "c", "d"], rate=2)


def test_startup_worker_failure_stops_wait_even_when_parent_is_alive(tmp_path, monkeypatch):
    (tmp_path / "server-1.log").write_text(
        "(Worker_DP4) ERROR 11-10 16:02:06 WorkerProc hit an exception.\n"
        "ninja: error: opening build log: No such file or directory\n")
    monkeypatch.setattr(orchestrate, "get_url", lambda *a: pytest.fail("failed startup cannot be ready"))
    monkeypatch.setattr(orchestrate.time, "sleep", lambda *a: pytest.fail("do not wait for full timeout"))
    with pytest.raises(RuntimeError, match="server-1.log"):
        orchestrate.wait_server(SimpleNamespace(poll=lambda: None), "http://localhost:8000", 2400, tmp_path)


def test_startup_wait_does_not_reject_warning_or_normal_warmup(tmp_path, monkeypatch):
    (tmp_path / "server-0.log").write_text(
        "(Worker_DP0) WARNING 11-10 16:02:06 Using default MoE config.\n"
        "(EngineCore_DP0) INFO 11-10 16:02:07 No available shared memory broadcast block found in 60 seconds.\n")
    monkeypatch.setattr(orchestrate, "get_url", lambda url: b'{"data":[{"id":"qwen3-full"}]}')
    assert orchestrate.wait_server(SimpleNamespace(poll=lambda: None), "http://localhost:8000", 5, tmp_path)["data"][0]["id"] == "qwen3-full"


def test_nested_server_environment_drops_controller_resource_limits_preserves_allocation():
    inherited = {"SLURM_JOB_ID": orchestrate.JOB, "SLURM_JOB_NODELIST": "mock[1-2]",
                 "SLURM_JOB_NUM_NODES": "2", "PATH": "/test/bin", "HF_HOME": "/cache",
                 "SLURM_GPUS": "0", "SLURM_GPUS_PER_NODE": "0", "CUDA_VISIBLE_DEVICES": "",
                 "SLURM_CPUS_PER_TASK": "4", "SLURM_MEM_PER_NODE": "16384",
                 "SLURM_TRES_PER_TASK": "cpu=4", "SLURM_STEP_ID": "0",
                 "SLURM_STEP_GPUS": "", "SLURM_CPU_BIND_LIST": "0x0f"}
    clean = orchestrate.server_environment(inherited)
    assert clean == {"SLURM_JOB_ID": orchestrate.JOB, "SLURM_JOB_NODELIST": "mock[1-2]",
                     "SLURM_JOB_NUM_NODES": "2", "PATH": "/test/bin", "HF_HOME": "/cache"}
    assert inherited["SLURM_GPUS"] == "0", "caller environment must not be mutated"


@pytest.mark.parametrize("identity", [orchestrate.JOB, "999999.2", "4882284.batch", "4882284.2;touch forbidden"])
def test_cleanup_rejects_wrong_or_broad_step_identity(tmp_path, monkeypatch, identity):
    (tmp_path / "server-step-primary.txt").write_text(identity)
    monkeypatch.setattr(orchestrate.subprocess, "run", lambda *a, **k: pytest.fail("must not invoke scancel"))
    process = SimpleNamespace(poll=lambda: None)
    with pytest.raises(RuntimeError, match="identity"):
        orchestrate.stop_server(process, tmp_path)


def test_cleanup_signals_exact_step_then_waits_without_killing_launcher(tmp_path, monkeypatch):
    for role in ("primary", "worker"):
        (tmp_path / f"server-step-{role}.txt").write_text("4882284.7\n")
    calls = []
    monkeypatch.setattr(orchestrate.subprocess, "run", lambda cmd, **kwargs: calls.append(("run", cmd)))
    monkeypatch.setattr(orchestrate.os, "killpg", lambda *a: pytest.fail("graceful completion needs no launcher kill"))
    process = SimpleNamespace(poll=lambda: None, wait=lambda timeout: calls.append(("wait", timeout)))
    orchestrate.stop_server(process, tmp_path)
    assert calls == [("run", ["scancel", "--signal=TERM", "4882284.7"]), ("wait", 90)]


def test_cleanup_without_started_node_cancels_only_its_launcher(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(orchestrate.subprocess, "run", lambda *a, **k: pytest.fail("no recorded remote step"))
    process = SimpleNamespace(poll=lambda: None,
        send_signal=lambda sig: calls.append(("signal", sig)),
        wait=lambda timeout: calls.append(("wait", timeout)))
    orchestrate.stop_server(process, tmp_path)
    assert calls == [("signal", signal.SIGINT), ("wait", 90)]


def test_mocked_controller_runs_fresh_native_arms_and_reuses_off_selected_calendar(tmp_path, monkeypatch):
    pilot = make_requests()
    args = SimpleNamespace(snapshot=tmp_path / "snapshot", pilot=tmp_path / "pilot.jsonl",
        smoke=tmp_path / "smoke.jsonl", data_manifest=tmp_path / "data.json",
        arrival_manifest=tmp_path / "arrivals.json", output=tmp_path / "output",
        private_output=tmp_path / "private", startup_timeout=20)
    args.pilot.write_text("\n".join(json.dumps(r) for r in pilot))
    args.smoke.write_text("\n".join(json.dumps(r) for r in pilot[::8]))
    args.data_manifest.write_text("{}")
    args.arrival_manifest.write_text(json.dumps({"blocks": [{"split": "pilot",
        "original_offsets_ns": list(range(24)), "original_horizon_ns": 24}]}))
    monkeypatch.setenv("SLURM_JOB_ID", orchestrate.JOB)
    monkeypatch.setenv("SLURM_JOB_NODELIST", "mock1")  # controller step has only one node
    monkeypatch.setattr(orchestrate.socket, "gethostbyname", lambda node: "127.0.0.1")
    launches, clients, stopped = [], [], []

    class Process:
        returncode = None
        pid = 987654

        def __init__(self, cmd, **kwargs):
            launches.append((cmd, kwargs))

        def poll(self):
            return None

    monkeypatch.setattr(orchestrate.subprocess, "Popen", Process)
    monkeypatch.setattr(orchestrate, "wait_server", lambda *args: {"data": [{"id": "qwen3-full"}]})
    monkeypatch.setattr(orchestrate, "stop_server", lambda *args, **kwargs: stopped.append(args[0]))
    monkeypatch.setattr(orchestrate, "get_url", lambda url: b"# mocked native metrics\n")

    def command(cmd, output=None, timeout=120):
        if cmd[:3] == ["scontrol", "show", "job"]:
            result = json.dumps({"jobs": [{"job_state": "RUNNING", "nodes": "mock[1-2]"}]})
        elif cmd[:3] == ["scontrol", "show", "hostnames"]:
            assert cmd[3] == "mock[1-2]", "use scheduler allocation, not controller step nodelist"
            result = "mock1\nmock2\n"
        elif cmd[:2] == ["git", "rev-parse"]:
            result = "a" * 40 + "\n"
        elif "full_serving.preflight" in cmd:
            Path(cmd[cmd.index("--output") + 1]).write_text("{}")
            result = "mock checkpoint verified\n"
        elif "full_serving.client" in cmd:
            clients.append(cmd)
            value = lambda flag: cmd[cmd.index(flag) + 1]
            rows = orchestrate.read_jsonl(value("--requests"))
            records = [{"type": "request", "request_id": row["id"], "output_sha256": row["id"],
                        "usage": {"prompt_tokens": 4, "completion_tokens": 3}, "finish_reason": "stop"}
                       for row in rows]
            records.append({"type": "run_summary", "valid": True, "completed_requests": len(rows),
                            "completed_elapsed_s": 10})
            Path(value("--output")).write_text("\n".join(json.dumps(r) for r in records))
            result = "mock client completed\n"
        else:
            raise AssertionError(f"Unexpected real command: {cmd}")
        if output:
            Path(output).write_text(result)
        return result

    monkeypatch.setattr(orchestrate, "command", command)
    orchestrate.run(args)
    assert len(launches) == len(stopped) == 2
    assert len(clients) == 6
    assert [cmd[cmd.index("--arm") + 1] for cmd in clients] == ["off"] * 3 + ["on"] * 3
    assert all("--gpus-per-node=4" in cmd and "--nodes=2" in cmd for cmd, _ in launches)
    assert all("--nodelist=mock1,mock2" in cmd for cmd, _ in launches)
    assert all(kwargs["start_new_session"] for _, kwargs in launches)
    assert all("--logprobs" in clients[i] for i in (0, 3))
    assert all(clients[i][clients[i].index("--rate") + 1] == "0" for i in (1, 4))
    external = [clients[i][clients[i].index("--calendar") + 1] for i in (2, 5)]
    assert external[0] == external[1]
    calendar = json.loads(Path(external[0]).read_text())
    assert calendar["offered_requests_per_s"] == pytest.approx(.6 * 32 / 10)
    assert calendar["rate_selection"]["not_a_final_capacity_estimate"] is True
    state = json.loads((args.output / "status.json").read_text())
    assert state["state"] == "pilot_completed"
    assert state["final_measurements"] is False and state["custom_dispatcher_installed"] is False
    assert json.loads((args.output / "smoke-output-comparison.json").read_text())["all_generated_text_hashes_equal"]

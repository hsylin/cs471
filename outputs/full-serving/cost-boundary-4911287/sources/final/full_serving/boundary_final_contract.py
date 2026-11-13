"""Fail-closed contract for a later, reviewed OFF/B4/no-op matched comparison.

This prepares execution only. A source/protocol freeze and evidence review are
still required; preparing this module does not select or admit a winning fix.
"""
import hashlib
import json
from pathlib import Path

TREATMENTS = {"off": ("off", "native"), "b4": ("b4", "native"),
              "b4_noop": ("b4", "noop_maps")}
TOTALS = {"longbench": 75246, "numinamath": 741753}
EVIDENCE = ("noop_correctness", "cost_controls", "sustained_shapes", "transition_controls", "selection_review")

# Explicit diagnostic settings only. Native model-memory profiling during startup
# remains unchanged; it is not an externally enabled performance trace.
DIAGNOSTIC_ENV = frozenset({
    "VLLM_TRACE_FUNCTION", "VLLM_CUSTOM_SCOPES_FOR_PROFILING",
    "VLLM_NVTX_SCOPES_FOR_PROFILING", "CUDA_LAUNCH_BLOCKING",
})
DIAGNOSTIC_PREFIXES = ("VLLM_TORCH_PROFILER_", "VLLM_PROTON_PROFILER_",
                       "VLLM_PROFILER_")


def diagnostic_environment_keys(environment):
    return sorted(k for k in environment
                  if k in DIAGNOSTIC_ENV or k.startswith(DIAGNOSTIC_PREFIXES))


def verify_unprofiled(command, environment):
    assert isinstance(command, list) and command and command[0] == "vllm"
    assert all(isinstance(arg, str) for arg in command)
    # Config-file indirection is absent from the admitted CLI and must not
    # introduce profiler options outside this recorded command.
    options = [arg.split("=", 1)[0].replace("_", "-") for arg in command
               if arg.startswith("--")]
    bad = [arg for arg in options if arg == "--config"
           or arg == "--profiler-config" or arg.startswith("--profiler-config.")]
    keys = diagnostic_environment_keys(environment)
    assert not bad, bad
    assert not keys, keys
    return {"schema": "cs471_unprofiled_launch_v1", "checked": True,
            "profiler_options": [], "diagnostic_environment_keys": []}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_admission(path, protocol_path, source, treatment, runtime_root):
    receipt = json.loads(Path(path).read_text())
    protocol = json.loads(Path(protocol_path).read_text())
    assert receipt["status"] == "passed"
    assert protocol["status"] == "frozen_final"
    assert protocol["schema"] == "cs471_boundary_final_v1"
    assert receipt["source_revision"] == source and len(source) == 40
    assert receipt["protocol_sha256"] == digest(protocol_path)
    assert protocol["selected_fix"] == receipt["selected_fix"] == "exact_unchanged_map_commit_v1"
    assert protocol["groups"] == 1 and protocol["requests_per_dataset"] == 512
    assert protocol["output_totals"] == TOTALS
    assert protocol["treatments"] == list(TREATMENTS)
    assert treatment in TREATMENTS
    required = {"full_serving/boundary_noop_commit.py", "full_serving/monitoring_gate_plugin.py",
                "full_serving/logical_history_gpu.py", "full_serving/client.py"}
    assert required <= receipt["admitted_runtime_files"].keys()
    for name, expected in receipt["admitted_runtime_files"].items():
        assert not Path(name).is_absolute() and ".." not in Path(name).parts
        assert digest(Path(runtime_root)/name) == expected, name
    assert set(receipt["evidence"]) == set(EVIDENCE)
    for key in EVIDENCE:
        item = receipt["evidence"][key]
        assert digest(item["path"]) == item["sha256"], key
        evidence = json.loads(Path(item["path"]).read_text())
        assert evidence["status"] == "passed", key
        if key == "selection_review":
            assert evidence["selected_fix"] == protocol["selected_fix"]
            assert evidence["policy_preserving"] is True
            assert evidence["shared_transport_changed"] is False
            assert evidence["final_measurements_used_for_selection"] is False
    return receipt


def performance_environment(environment, treatment):
    arm, fix = TREATMENTS[treatment]
    diagnostic_keys = set(diagnostic_environment_keys(environment))
    env = {k: v for k, v in environment.items()
           if not k.startswith(("FULL_SERVING_BOUNDARY_", "FULL_SERVING_PAYBACK_", "CS471_"))
           and k not in diagnostic_keys}
    env.update(FULL_SERVING_BOUNDARY_PROBE="off",
               FULL_SERVING_BOUNDARY_FIX=fix,
               FULL_SERVING_BOUNDARY_FIX_SHADOW="0",
               FULL_SERVING_PROPOSAL_PILOT="0")
    return env


def validate_targets(root):
    root = Path(root)
    for dataset, total in TOTALS.items():
        targets = json.loads((root / (dataset + "-measured-lengths.json")).read_text())
        assert len(targets) == 512 and sum(targets.values()) == total
        assert all(type(x) is int and 1 <= x <= 8192 for x in targets.values())


def verify_launch(launch, treatment):
    arm, fix = TREATMENTS[treatment]
    assert launch["profiling_isolation"] == verify_unprofiled(launch["command"], {})
    assert launch["arm"] == arm and launch["pilot"] is False
    assert launch["map_commit"] == fix and launch["map_shadow"] == "0"
    assert launch["host_probe"] == "off" and launch["history_shadow"] is False
    assert launch["payback_branch"] is None
    assert launch["plugin"] == (None if arm == "off" else "cs471_sustained")

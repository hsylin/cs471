"""Separate pilot-only timeline diagnosis; never a primary timing record.

One pinned pilot bank/domain, two batch sizes and four physical-dispatch arms.
Every arm has16 warmup inputs followed by32 distinct captured input batches.
Instance wrappers and a restored process-local collective wrapper add NVTX and
CUDA-event ranges without changing measured or legacy modules. Nsight Systems
captures only cudaProfilerStart/Stop regions. The explicit torch backend exports
CPU+CUDA Chrome traces when external collection is unavailable.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from optimization.common import array_hash, make_layer, manifest_hash
from optimization.kernel_config import kernel_override
from resident_dispatch.common import load_manifest, numerical_check, runtime_metadata
from resident_dispatch.pilot import prepare_cell
from skewbench.native import Runtime, Weights


ARMS = ("native", "pre_native", "one_copy_home", "token_balanced")
WARMUP_STEPS = 16
PROFILE_STEPS = 32


def source_identity():
    root = Path(__file__).resolve().parents[1]
    paths = (Path(__file__), root / "scripts/profile-resident-kuma.sh")
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def profile_groups(manifest, block_id=None, domain=0):
    """Keep the profile bounded and limited to previously authorized pilot banks."""
    if manifest.get("phase") != "pilot" or manifest.get("split") != "pilot":
        raise ValueError("Profiling is restricted to a frozen pilot manifest")
    allowed = list(manifest["matrix"]["block_ids"])
    block_id = block_id or allowed[0]
    if block_id not in allowed or domain not in manifest["matrix"]["domains"]:
        raise ValueError("Selected bank/domain is absent from the frozen pilot")
    if not {128, 1024}.issubset(manifest["matrix"]["batch_tokens"]):
        raise ValueError("Expected the existing128/1024 pilot batch sizes")
    for batch in (128, 1024):
        yield {"block_id": block_id, "domain": domain, "regime": f"domain{domain}",
               "batch_tokens": batch, "repetition": 0}


class PhaseRecorder:
    """Rank-local events add overhead and are explicitly diagnostic only."""

    def __init__(self, torch, rank, backend):
        self.torch, self.rank, self.backend = torch, rank, backend
        self.rows = []
        self.enabled = False
        self.step = self.raw = None
        self.anchor = None
        self.in_forward = False

    @contextmanager
    def phase(self, name):
        if not self.enabled:
            yield
            return
        torch = self.torch
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        label = f"resident/rank{self.rank}/step{self.step:02d}/raw{self.raw:02d}/{name}"
        entry_wall = time.time_ns()
        entry_monotonic = time.monotonic_ns()
        torch.cuda.nvtx.range_push(label)
        context = torch.profiler.record_function(label) if self.backend == "torch" else nullcontext()
        start.record()
        try:
            with context:
                yield
        finally:
            end.record()
            torch.cuda.nvtx.range_pop()
            self.rows.append({"phase": name, "profile_step": self.step, "raw_index": self.raw,
                              "host_entry_wall_ns": entry_wall,
                              "host_entry_monotonic_ns": entry_monotonic,
                              "host_exit_wall_ns": time.time_ns(),
                              "host_exit_monotonic_ns": time.monotonic_ns(),
                              "start_event": start, "end_event": end})

    def resolved(self):
        result = []
        for row in self.rows:
            start, end = row["start_event"], row["end_event"]
            result.append({**{k: v for k, v in row.items() if not k.endswith("_event")},
                           "cuda_elapsed_ms": start.elapsed_time(end),
                           "cuda_start_ms_from_rank_anchor": self.anchor.elapsed_time(start),
                           "cuda_end_ms_from_rank_anchor": self.anchor.elapsed_time(end)})
        return result


@contextmanager
def instrument_layer(layer, recorder):
    """Restore every wrapped callable, including the process-wide torch function."""
    mapping = layer.mapping_kernel
    experts = layer.fused_experts
    distributed = layer.torch.distributed
    all_reduce = distributed.all_reduce

    def wrapped_mapping(*args, **kwargs):
        with recorder.phase("mapping"):
            return mapping(*args, **kwargs)

    def wrapped_experts(*args, **kwargs):
        with recorder.phase("fused_experts"):
            return experts(*args, **kwargs)

    def wrapped_all_reduce(*args, **kwargs):
        name = "output_allreduce" if recorder.in_forward else "other_allreduce"
        with recorder.phase(name):
            return all_reduce(*args, **kwargs)

    layer.mapping_kernel, layer.fused_experts = wrapped_mapping, wrapped_experts
    distributed.all_reduce = wrapped_all_reduce
    try:
        yield
    finally:
        layer.mapping_kernel, layer.fused_experts = mapping, experts
        distributed.all_reduce = all_reduce


def _profile_arm(rt, weights, manifest, cell, group, arm, backend, output_dir):
    from resident_dispatch.gpu import PrecomputedMapper

    torch = rt.torch
    layer = make_layer(rt, weights, manifest, "SKIP")
    layer.assert_maps_and_weights()
    method = "native" if arm in ("native", "pre_native") else arm
    adapter = None
    if arm != "native":
        adapter = PrecomputedMapper(torch, cell["assignments"][method])
        layer.mapping_kernel = adapter

    def select(raw):
        if adapter is not None:
            adapter.raw_index = raw

    tensors = cell["tensors"]
    preflight = numerical_check(rt, layer, tensors, cell["references"], list(range(48)), select)
    layer.warming_up = True
    for raw in range(WARMUP_STEPS):
        select(raw)
        layer.forward(*(tensor[raw] for tensor in tensors), observe=True)
        layer.step(raw)
    layer.warming_up = False
    # Match the pilot's reset.32 profile steps do not reach the64-step planner.
    layer.state.expert_rearrangement_step = 0
    layer.state.expert_load_window_step = 0
    layer.ms.expert_load_window.zero_()
    layer.ms.expert_load_pass.zero_()
    layer.state._update_layer_should_record(log_stats=False)
    layer.planner_seconds = layer.controller_seconds = 0.
    layer.decisions.clear()
    layer.counter.sent_bytes = layer.counter.sent_messages = 0

    recorder = PhaseRecorder(torch, rt.rank, backend)
    stem = f"{group['block_id']}-d{group['domain']}-b{group['batch_tokens']}-{arm}-rank{rt.rank}"
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    context = (torch.profiler.profile(activities=activities, record_shapes=True,
                                      with_stack=False, profile_memory=False)
               if backend == "torch" else nullcontext(None))
    rt.sync()
    rt.barrier()
    rt.sync()
    capture_started = False
    capture_wall_ns = time.time_ns()
    try:
        if backend == "nsys":
            torch.cuda.profiler.start()
            capture_started = True
        with context as prof, instrument_layer(layer, recorder):
            recorder.anchor = torch.cuda.Event(enable_timing=True)
            recorder.anchor.record()
            recorder.enabled = True
            torch.cuda.nvtx.range_push(f"resident_profile/{stem}")
            try:
                for step in range(PROFILE_STEPS):
                    raw = WARMUP_STEPS + step
                    select(raw)
                    recorder.step, recorder.raw = step, raw
                    with recorder.phase("forward"):
                        recorder.in_forward = True
                        try:
                            layer.forward(*(tensor[raw] for tensor in tensors), observe=True)
                        finally:
                            recorder.in_forward = False
                    with recorder.phase("step"):
                        layer.end_of_input = step == PROFILE_STEPS - 1
                        layer.step(step)
                rt.sync()
            finally:
                recorder.enabled = False
                torch.cuda.nvtx.range_pop()
    finally:
        if capture_started:
            torch.cuda.profiler.stop()
    capture_end_wall_ns = time.time_ns()
    if backend == "torch":
        prof.export_chrome_trace(str(output_dir / f"{stem}.chrome.json"))
        (output_dir / f"{stem}.kernels.txt").write_text(
            prof.key_averages().table(sort_by="self_device_time_total", row_limit=80))
    phase_rows = recorder.resolved()
    layer.close()
    layer.assert_maps_and_weights()
    if (not np.array_equal(layer.map_snapshot(), cell["mapping"])
            or layer.counter.sent_bytes or layer.counter.sent_messages or layer.decisions):
        raise AssertionError("Profiler fixed map changed or unexpected proposal/transfer occurred")
    post = numerical_check(rt, layer, tensors, cell["references"], [16, 47], select)
    row = {
        **rt.metadata, **group, "arm": arm, "rank": rt.rank,
        "resident_dispatch_profile_schema_version": 1,
        "config_hash": manifest_hash(manifest), "phase": "pilot_profile", "split": "pilot",
        "measurement_kind": "instrumented_fixed_placement_dispatch_profile", "evidence_eligible": False,
        "status": "complete", "profile_backend": backend, "profile_stem": stem,
        "profile_source_sha256": source_identity(), "warmup_forwards": WARMUP_STEPS,
        "profile_forwards": PROFILE_STEPS, "raw_indices": list(range(16, 48)),
        "input_feeding_excluded": True, "offline_assignment_construction_excluded": True,
        "online_selector_implemented": False, "policy": "SKIP",
        "planner_opportunities_profiled": 0, "profile_capture_wall_start_ns": capture_wall_ns,
        "profile_capture_wall_end_ns": capture_end_wall_ns,
        "phase_events": phase_rows,
        "phase_event_semantics": "CUDA events are rank-local stream spans; nested phases overlap and must not be summed. Allreduce includes dependencies/wait, not pure transport time. Host wall timestamps share a host clock; GPU anchors are not cross-rank synchronized.",
        "profile_overhead": "NVTX, CUDA events, optional Torch profiler and external collector perturb execution; no primary elapsed-time estimate or inferential comparison is emitted.",
        "assignment_method": method, "assignment_artifact": cell["artifact"],
        "assignment_sha256": cell["assignment_sha256"][method],
        "assignment_metrics": cell["metrics"][method][16:48],
        "capture_sha256": cell["capture_sha256"], "source_prompt_ids": cell["source_prompt_ids"],
        "source_map": cell["mapping"].tolist(), "source_map_sha256": array_hash(cell["mapping"]),
        "source_weights_sha256": cell["source_weights_sha256"],
        "source_weight_rank_sha256": cell["source_weight_rank_sha256"],
        "resolved_kernel_config": cell["resolved_kernel_config"],
        "gpu_mapping_and_record_checks": cell["gpu_mapping_and_record_checks"],
        "source_restore_verified": True, "expected_arm_weights_verified": True,
        "logical_routing_preserved": True, "rank_assignment_agreement": True,
        "migration_bytes": 0, "preflight_correctness": preflight, **post,
    }
    (output_dir / f"{stem}.profile.json").write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
    del layer, recorder
    gc.collect()
    rt.sync()
    rt.barrier()
    return {"arm": arm, "batch_tokens": group["batch_tokens"], "rank": rt.rank,
            "profile_stem": stem, "correctness_pass": post["correctness_pass"],
            "phase_events": len(phase_rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--banks-dir", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("nsys", "torch"), default="torch")
    parser.add_argument("--block-id")
    parser.add_argument("--domain", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    groups = list(profile_groups(manifest, args.block_id, args.domain))
    rt = Runtime()
    try:
        # Rank0 creates the directory and broadcasts refusal before any bank or
        # profile writes, preventing ambiguous partial-group append/resume.
        if rt.rank == 0:
            try:
                args.output_dir.mkdir(parents=True, exist_ok=False)
                error = None
            except OSError as exc:
                error = str(exc)
        else:
            error = None
        message = [error]
        rt.torch.distributed.broadcast_object_list(message, src=0, group=rt.ep_group.cpu_group)
        if message[0] is not None:
            raise FileExistsError(message[0])
        rt.barrier()
        weights = Weights(args.weights_dir)
        runtime_metadata(rt, weights, manifest)
        rows = []
        with kernel_override(None):
            for group in groups:
                # prepare_cell persists exact input indices, logical routes,
                # all assignment tensors, source map and construction metadata.
                cell = prepare_cell(rt, weights, manifest, args.banks_dir,
                                    args.output_dir / "diagnostic-inputs.jsonl", group)
                for arm in ARMS:
                    if rt.rank == 0:
                        print(json.dumps({"event": "profile_start", **group, "arm": arm,
                                          "backend": args.backend}), flush=True)
                    rows.append(_profile_arm(rt, weights, manifest, cell, group, arm,
                                             args.backend, args.output_dir))
                del cell
                gc.collect()
                rt.torch.cuda.empty_cache()
        (args.output_dir / f"profile-index-rank{rt.rank}.json").write_text(json.dumps({
            "evidence_eligible": False, "config_hash": manifest_hash(manifest),
            "profile_source_sha256": source_identity(), "profiles": rows,
            "scope": "one pilot source bank/domain; both ranks; eight arm profiles per rank; no measured-study result",
            "ncu": "Not invoked. Availability is separately logged; counter permissions and one-kernel replay safety remain unverified. Never replay NCCL kernels.",
        }, indent=2) + "\n")
    finally:
        rt.close()


if __name__ == "__main__":
    main()

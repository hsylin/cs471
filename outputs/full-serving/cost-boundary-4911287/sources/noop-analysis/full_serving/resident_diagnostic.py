"""Untimed full-weight and native-expert correctness probes before profiling.

This is opt-in startup work for a correctness-only server. It uses synthetic
hidden states and must never be reported as a real-dataset performance result.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import MethodType


def validate_weight_records(peers, mapping, *, layers=48):
    """Independent CPU check of physical ownership and complete replica hashes."""
    if len(peers) != 8 or sorted(p["rank"] for p in peers) != list(range(8)):
        raise AssertionError("Expected exactly eight distinct weight-record ranks")
    by_key, seen = {}, set()
    for peer in peers:
        if len(peer["weights"]) != layers * 20:
            raise AssertionError("Incomplete local weight fingerprint record")
        for r in peer["weights"]:
            key = r["layer_index"], r["physical_slot"]
            if (key in seen or not 0 <= key[0] < layers
                    or key[1] // 20 != peer["rank"] or not 0 <= key[1] < 160
                    or r["logical_expert"] != mapping[key[1]]):
                raise AssertionError("Weight placement/physical ownership mismatch")
            seen.add(key)
            digests = tuple(r[name] for name in ("w13_sha256", "w2_sha256"))
            if any(len(d) != 64 or set(d) - set("0123456789abcdef") for d in digests):
                raise AssertionError("Invalid weight fingerprint")
            by_key.setdefault((key[0], r["logical_expert"]), []).append(digests)
    if len(seen) != layers * 160 or len(by_key) != layers * 128:
        raise AssertionError("Missing logical or physical expert weights")
    duplicated = 0
    for (layer, expert), hashes in by_key.items():
        if len(hashes) != mapping.count(expert) or len(set(hashes)) != 1:
            raise AssertionError("Duplicate resident expert weights differ")
        duplicated += len(hashes) > 1
    return {"physical_layer_experts": len(seen), "logical_layer_experts": len(by_key),
            "duplicated_logical_layer_experts": duplicated, "full_bf16_replica_hashes_equal": True}


def tensor_sha(tensor):
    import torch
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def run_startup_diagnostics(bound, config, admission):
    import numpy as np
    import torch
    import torch.distributed as dist
    from vllm.config import get_current_vllm_config
    from vllm.distributed import get_ep_group
    from vllm.forward_context import set_forward_context
    from full_serving.resident_policy import ResidentMapper, cpu_reference

    group = get_ep_group()
    rank = group.rank_in_group
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("The independent FP32 reference requires native highest matmul precision")
    vllm_config = get_current_vllm_config()
    mapping = config["physical_to_logical_map"]
    directory = Path(os.environ["FULL_SERVING_RUN_DIR"])

    def gather(value):
        peers = [None] * 8
        dist.all_gather_object(peers, value, group=group.cpu_group)
        return peers

    weights = []
    for layer_index, (name, layer, _) in enumerate(bound):
        routed = layer.routed_experts
        for local in range(20):
            physical = rank * 20 + local
            weights.append({"layer_index": layer_index, "layer": name,
                            "physical_slot": physical, "logical_expert": mapping[physical],
                            "w13_sha256": tensor_sha(routed.w13_weight[local]),
                            "w2_sha256": tensor_sha(routed.w2_weight[local])})
    peers = gather({"rank": rank, "weights": weights})
    weight_check = validate_weight_records(peers, mapping)
    print(f"[cs471-resident-diagnostic] rank={rank} all48_full_weight_hashes_passed=true", flush=True)

    numerical = []
    cases = [("heterogeneous_padding", 8, [3, 5, 0, 2, 1, 7, 4, 6]),
             ("full_distinct_inputs", 17, [17] * 8),
             ("all_padding_capture_case", 8, [0] * 8)]
    for layer_index in (0, 23, 47):
        name, layer, _ = bound[layer_index]
        router = layer.router
        prepare = layer.routed_experts.quant_method.moe_kernel.prepare_finalize
        # A fresh class-bound native method avoids nesting the installed wrapper.
        native_prepare = type(prepare).prepare.__get__(prepare, type(prepare))
        saved_prepare = prepare.prepare
        saved_mapping = router._apply_eplb_mapping
        saved_compute = router._compute_routing
        device = layer.routed_experts.w13_weight.device
        resident = ResidentMapper(device, mapping, mode="resident_v1")
        native = ResidentMapper(device, mapping, mode="native_hash")
        for case_index, (case_name, rows, real) in enumerate(cases):
            generator = torch.Generator(device=device).manual_seed(471110 + rank + 100 * case_index)
            hidden = torch.randn((rows, 2048), generator=generator, device=device, dtype=torch.bfloat16)
            padding = torch.arange(rows, device=device) >= real[rank]
            sizes = torch.tensor([rows] * 8, dtype=torch.int32)
            outputs, captures = {}, {}
            try:
                for mode in ("resident_v1", "native_hash"):
                    capture = {}
                    def compute(self, *args, **kwargs):
                        w, ids = saved_compute(*args, **kwargs)
                        capture["local_logical"] = ids.clone()
                        return w, ids
                    router._compute_routing = MethodType(compute, router)
                    if mode == "resident_v1":
                        router._apply_eplb_mapping = MethodType(lambda self, ids: ids, router)
                    else:
                        router._apply_eplb_mapping = MethodType(lambda self, ids: native(ids), router)
                    def checked_prepare(self, *args, **kwargs):
                        a, scales, metadata, ids, w = native_prepare(*args, **kwargs)
                        capture["gathered_before_mapping"] = ids.clone()
                        capture["global_hidden"] = a.clone()
                        capture["weights_before"] = w.clone()
                        physical = resident(ids) if mode == "resident_v1" else ids
                        capture["physical"] = physical.clone()
                        capture["weights_after"] = w
                        return a, scales, metadata, physical, w
                    prepare.prepare = MethodType(checked_prepare, prepare)
                    with torch.no_grad(), set_forward_context(None, vllm_config, num_tokens=rows,
                                                             num_tokens_across_dp=sizes, is_padding=padding):
                        # Match Qwen3MoeSparseMoeBlock.forward exactly. The
                        # runner owns the gate and replaces this placeholder.
                        outputs[mode] = layer(hidden_states=hidden, router_logits=hidden).clone()
                    torch.cuda.synchronize(device)
                    captures[mode] = capture
                    local = capture["local_logical"].cpu().numpy()
                    if not np.all(local[real[rank]:] == -1):
                        raise AssertionError("Native top-k failed to mask per-rank padding")
                    local_peers = gather(local.tolist())
                    logical = np.concatenate([np.asarray(x, dtype=np.int64) for x in local_peers])
                    if mode == "resident_v1":
                        expected = cpu_reference(logical)["physical_ids"]
                        np.testing.assert_array_equal(capture["gathered_before_mapping"].cpu().numpy(), logical)
                    else:
                        expected = np.concatenate([cpu_reference(x, mode="native_hash")["physical_ids"]
                                                   for x in local_peers])
                    actual = capture["physical"].cpu().numpy()
                    np.testing.assert_array_equal(actual, expected)
                    valid = logical >= 0
                    np.testing.assert_array_equal(np.asarray(mapping)[actual[valid]], logical[valid])
                    assert np.count_nonzero(actual >= 0) == sum(real) * 8
                    assert np.all(actual[~valid] == -1)
                    torch.testing.assert_close(capture["weights_after"], capture["weights_before"], rtol=0, atol=0)
                    signatures = gather({"physical": tensor_sha(capture["physical"]),
                                         "weights": tensor_sha(capture["weights_before"]),
                                         "global_hidden": tensor_sha(capture["global_hidden"])})
                    if any(s != signatures[0] for s in signatures):
                        raise AssertionError("Ranks disagree on gathered inputs, weights or physical assignment")
                r, n = outputs["resident_v1"], outputs["native_hash"]
                torch.testing.assert_close(r[:real[rank]].float(), n[:real[rank]].float(), atol=.02, rtol=.03)
                for key in ("local_logical", "weights_before", "global_hidden"):
                    torch.testing.assert_close(captures["resident_v1"][key], captures["native_hash"][key], atol=0, rtol=0)
                # Independent FP32 matrix arithmetic: canonical original copies
                # are slots 0..15 on each rank. Sum all eight ranks' contributions.
                c = captures["resident_v1"]
                ids = torch.as_tensor(logical, device=device)
                reference = torch.zeros((8 * rows, 2048), device=device, dtype=torch.float32)
                routed = layer.routed_experts
                for local in range(16):
                    token, column = torch.where(ids == (16 * rank + local))
                    if not token.numel():
                        continue
                    gate_up = c["global_hidden"][token].float() @ routed.w13_weight[local].float().T
                    gate, up = gate_up.chunk(2, dim=-1)
                    value = (torch.nn.functional.silu(gate) * up) @ routed.w2_weight[local].float().T
                    value *= c["weights_before"][token, column, None]
                    reference.index_add_(0, token, value)
                dist.all_reduce(reference, group=group.device_group)
                local_reference = reference[rank * rows:rank * rows + real[rank]]
                torch.testing.assert_close(r[:real[rank]].float(), local_reference, atol=.02, rtol=.03)
                delta = (r[:real[rank]].float() - n[:real[rank]].float()).abs()
                fp32_delta = (r[:real[rank]].float() - local_reference).abs()
                numerical.append({"layer_index": layer_index, "layer": name, "case": case_name,
                                  "rows_per_rank": rows, "real_rows_per_rank": real,
                                  "logical_identity_exact": True, "physical_cpu_oracle_exact": True,
                                  "one_home_policy_checked_by_cpu_oracle": True,
                                  "routing_weights_unchanged": True, "token_conservation_exact": True,
                                  "gathered_rank_agreement": True, "native_expert_outputs_close": True,
                                  "direct_fp32_outputs_close": True, "atol": .02, "rtol": .03,
                                  "resident_vs_native_max_abs": delta.max().item() if delta.numel() else 0,
                                  "resident_vs_fp32_max_abs": fp32_delta.max().item() if fp32_delta.numel() else 0})
            finally:
                router._compute_routing = saved_compute
                router._apply_eplb_mapping = saved_mapping
                prepare.prepare = saved_prepare
    report = {"schema": "cs471_full_resident_startup_diagnostic_v1", "status": "passed",
              "ep_rank": rank, "job_id": os.environ.get("SLURM_JOB_ID"),
              "step_id": os.environ.get("SLURM_STEP_ID"), "weight_check": weight_check,
              "reference_matmul_precision": torch.get_float32_matmul_precision(),
              "local_weight_records": weights, "native_expert_numerical_checks": numerical,
              "scope": "Synthetic pre-profile correctness probes on actual loaded full-model weights; not serving measurements or complete-model numerical equivalence.",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), **admission}
    path = directory / f"resident-startup-diagnostic-rank{rank}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    print(f"[cs471-resident-diagnostic] rank={rank} native_expert_numerical_cases={len(numerical)} passed=true", flush=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "status": "passed", "timed_serving_measurement": False}

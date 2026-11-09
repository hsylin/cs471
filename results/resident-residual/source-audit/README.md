# Native alignment extent audit

The three archived Python files were SHA256-checked against the installed vLLM
on Kuma on2026-11-10. `inventory.json` contains the exact hashes and primary
source URLs. The C++ file is retrieved pinned source; its digest does not by
itself attest the compiled CUDA object. No GPU measurements were performed for
this audit.

All links below refer to vLLM commit
`db9527a46873454610df6dbedf79a36d6bf1a7f6`.

* [TritonExperts, lines338–374](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/layers/fused_moe/experts/triton_moe.py#L338-L374)
  drops nonlocal assignments during alignment for this unquantized, non-LoRA
  EP path, while passing the global physical expert count to alignment.
* [Alignment preparation, lines1541–1587](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/layers/fused_moe/fused_moe.py#L1541-L1587)
  delegates to the normal native alignment operation. The naive path requires
  no expert map and is inapplicable here.
* [Python alignment, lines137–172](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/layers/fused_moe/moe_align_block_size.py#L137-L172)
  allocates `1024 + global_experts * 63` sorted entries for B128/top8/M64.
* [GEMM invocation, lines820–839](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/layers/fused_moe/fused_moe.py#L820-L839)
  uses the sorted tensor's visible length for the launch grid. Both matrices
  have M at least64, so the separate small-M cap does not apply.
* [GEMM kernel, lines402–405](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/layers/fused_moe/fused_moe.py#L402-L405)
  returns before reading sorted IDs for programs outside the actual padded
  extent. Removing only those programs preserves all nonempty work.
* [Native local mapping and padding](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/csrc/libtorch_stable/moe/moe_align_sum_kernels.cu#L122-L227)
  validates the global input IDs, maps them to local IDs, computes padded local
  bucket counts, and fills the buffers. [The host entry point](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/csrc/libtorch_stable/moe/moe_align_sum_kernels.cu#L673-L791)
  passes the allocated sorted tensor's size to the kernels. The proposed
  adapter leaves all these allocations and calls unchanged.

## Capacity proof and source-derived grid

For nonnegative local bucket counts summing to at most1024, with at most64
positive buckets, each bucket adds at most63 padded rows. Therefore
`sum(ceil(count/64)*64) <= 1024 +64*63 =5056 =79*64`.
Stockoff has64 resident experts per rank. The frozen one-copy-home policy has
at most48 forced singleton experts per rank and32 shared experts; its capacity
target is `max(forced0,forced1,ceil(total_active/2)) <=64`. Exact replay of those
physical assignments has the same bound. A mere160-slot layout without this
policy does **not** establish the64-active bound.

| Path | Visible sorted extent | M program positions | w13 grid | w2 grid |
|---|---:|---:|---:|---:|
| Native128 global slots |9088|142|1704|2272|
| Native160 global slots |11104|174|2088|2784|
| Proven-cap view, either supported path |5056|79|948|1264|

These are source-derived program counts, not measured useful GEMM operations,
GPU durations, or a prediction of wall-time speedup. Actual populated tiles,
expert assignments, weights, alignment initialization, and numerical work are
unchanged. The view can cause a different Triton specialization because its
extent is a compile-time argument; it is not an upstream source change.

## Adapter contract and verification

`resident_residual.alignment.trim_native_alignment(layer, enabled=True,
probe=False, policy=None)` is a scoped context manager yielding a guard. It
temporarily replaces only the imported preparation function used by the
reviewed TritonExperts path, and restores the original on normal or exceptional
exit. The timed path validates the supported call, calls the original native
alignment, and returns a `[:5056]` view. These Python checks and view creation
are charged. No GPU readback or CPU workload decision enters the timed path.
Disabled, non-probe mode installs no hook.

Create the context after resetting trial state: map identity, version and
pointer are guarded. `snapshot()` is lightweight. `probe=True` is strictly
untimed: it also observes both native GEMM invocations. After leaving the
context, `verify()` checks actual padded length, identical backing/prefix,
independent local assignment conservation, and the two expected GEMM grids.
Full and trimmed numerical outputs must additionally be compared by the
runner on the same actual inputs. Raw sorted-token hashes can differ across
separate native alignment calls because within-expert atomic scatter order is
not guaranteed; the checker validates the bucket multisets instead.
`AlignmentTrim(..., probe=True).probe_assignments(ids)` supports separate
adversarial alignment-only tests. It does not claim numerical GEMM coverage.

The scope excludes concurrent layer calls, graphs, LoRA, quantized kernels,
different batch/top-k/tile geometry, changed placement, different assignment
policy, and any rank layout besides the reviewed two-rank128/160-slot cases.

## Diagnostic interpretation

The fixed seven-arm development diagnostic can identify a launch-extent
package effect, selector-invocation package effect, their interaction, and an
input-validation-only host effect on identical logical/physical assignments.
Precomputed physical-ID replay is an offline diagnostic, not an implementable
online baseline or lower bound. The checked-versus-unchecked replay contrast
does not remove all Python/framework overhead. No measured effect should be
attributed exclusively to GPU arithmetic, bandwidth, or host launch time
without additional evidence.

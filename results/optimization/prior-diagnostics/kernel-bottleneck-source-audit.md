# Kernel bottleneck and bounded optimization audit

The best-supported first target is the **two fused expert GEMMs in each forward**, with excessive per-expert row padding as a concrete optimization hypothesis. This is stronger evidence than choosing migration because its individual call takes about6ms: migration is an occasional action, whereas the fused GEMMs execute on every input. The previous fixed-map profile contains no migration at all. The available evidence identifies the dominant named GPU compute work; it does not yet separate weight bandwidth, tensor-core efficiency, occupancy and launch scheduling into causal fractions.

## Evidence checked without nested profiler attribution

The eight original Chrome traces were reparsed using only events with `cat='kernel'` and `ph='X'`. Each session contains eight forwards and16 `fused_moe_kernel` launches. Dividing the sum of these **kernel events**, rather than operator annotations, by eight gives:

| Batch | Two fused expert kernels/forward | Physical-map kernel/forward | NCCL output-allreduce kernel/forward |
|---:|---:|---:|---:|
| 128 | 324.551–338.335µs | 2.912–3.100µs | 16.820–294.744µs |
| 1024 | 380.167–385.108µs | 2.884–2.920µs | 58.688–85.371µs |

Ranges span domain and rank, not independent-sample confidence intervals. The extreme294.744µs collective belongs to the first profiler session, whose other rank records18.212µs and whose host instrumentation overhead is large. Collective residence includes arrival waiting, so it does not establish that the network is the main bottleneck. Unprofiled median forward durations are roughly393–404µs at128 and528µs at1024. Do not divide profiled component times by these separate unprofiled timings to claim additive percentages.

The pinned native path maps physical experts, executes fused expert computation, then performs NCCL output allreduce. The expert kernel's sorted token groups are padded to `BLOCK_SIZE_M`; non-local expert blocks follow a zero-output path, and excess grid blocks can return early. The existing profile has71–77 active local expert slots per rank. For small batches, each active expert receives about7 assignments, yet the fallback M tile is64. At1024 the fallback M tile is128 and average active-expert occupancy is about53 assignments.

Using the actual recorded per-slot counts, `sum_e ceil(count_e / block_m) * block_m` yields these local padded-row-to-useful-assignment ratios:

| Batch | Native fallback | Ratio at fallback | Ratio at M16 | Ratio at M32 |
|---:|---:|---:|---:|---:|
| 128 | M64 | 8.89–9.38× | 2.25–2.34× | 4.45–4.69× |
| 1024 | M128 | 2.37–2.41× | 1.14–1.16× | 1.29–1.32× |

These are integer tiling-work proxies, not promised speedups or measured hardware instruction counts. Smaller tiles can reduce padded matrix rows but also alter block count, weight reuse, tensor-core utilization and occupancy. Testing is necessary. The source chooses its fallback using whole-input M and local expert count, which does not directly encode the observed routed-token distribution across physical experts.

Primary pinned source: [expert alignment and kernel padding](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/layers/fused_moe/fused_moe.py#L1541), [fallback/override selection](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/layers/fused_moe/fused_moe.py#L1296). Full raw-kernel sums, trace hashes and padding calculations are saved in `work/kernel-bottleneck-source-evidence.json`.

## Implemented bounded control

New `optimization/kernel_config.py` supplies exactly eight declared choices:

- DEFAULT: unchanged native lookup/fallback.
- M16/M32/M64 × N64/N128, each with K64, GROUP_SIZE_M1, SPLIT_K1, four warps and three stages.
- M128/N64/K64, GROUP_SIZE_M1, SPLIT_K1, eight warps and three stages.

At batch128 the explicit M64/N128/W4 choice duplicates the default settings. Retain it as an implementation/control check if desired, but do not count it as independent search diversity. The implementation exposes `CANDIDATES`, `config_fingerprint`, `kernel_override`, `verify_kernel_sources` and `resolve_kernel_config`. The resolver invokes the same native selector with actual weight shapes, top-k, BF16 configuration dtype and input batch size inside the override, and rejects any departure from the declared override or fallback; each run can record the resolved dictionary.

The pinned [native process-global override API](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/layers/fused_moe/__init__.py#L50) lacks exception-safe restoration around its yield. The separate wrapper exits the native context normally inside its own finally block, so a candidate failure propagates while the previous override is restored. It supports nested scopes and DEFAULT without editing installed vLLM. It must not be used concurrently from multiple threads. Thirteen CPU tests cover all candidates, normal/exception restoration, nesting, invalid candidate rejection and source attestation; GPU numerical correctness remains to be measured.

Source pins checked: installed `fused_moe.py` SHA256`21860340e4b409b01666e2ba0f0edae66b0f97b859756b174b573bab895cfaf7`; `__init__.py` SHA256`64a32665be6d5afcfbaf7a0915717267278b66b880b908b0263ea58f08e51b5e`. The latter was also downloaded read-only from the exact public commit and hashed independently. The new helper fails closed if either installed file differs.

## Paired attribution and fresh validation

Use fresh captured source blocks for the new experiment. Select at most one configuration per batch on the declared pilot subset; freeze the selection before held-out outcomes. Materialize identical per-step hidden states/routes/combining weights outside all arm timings and disclose that common input-feeding cost is excluded. Warm/compile every candidate before timing, verify full resident weights/maps and native output agreement, and balance candidate order rather than measuring default first every time.

For attribution, compare DEFAULT and TUNED at an identical fixed map and actual route sequence, with isolated kernel-only trace categories and a complete unprofiled elapsed timer. If GEMM kernel time falls along with whole-forward latency, that supports a kernel-path explanation. Padding counts strengthen the mechanism evidence but cannot alone prove speedup.

For the deployable test, retain the planned2×2 DEFAULT/TUNED × B3/SKIP comparison with unchanged replicas, observation, candidate planner, admission and synchronous transfer. Include native monitor/planning/commit costs in full elapsed time. Validate that equivalent policy arms see the same routing windows, produce the same candidates/commits and account for actual payloads; kernel tuning should not silently alter the controller treatment. This distinguishes a common faster kernel from a benefit specifically due to migration. No new async, CUDA-graph, communication-backend or precision change is needed in this bounded first attempt.

No previous frozen source, thresholds, measurements or results were changed. No GPU job was submitted by this audit.

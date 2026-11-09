# Comparison with genuine stock vLLM layer configurations

The prespecified useful-gain claim against both tested stock configurations was not established. The current method has substantially higher high-load response latency than EPLB-disabled stockoff within this frozen, conditional design. At high offered load, ours/stockoff mean-response ratio was 6.3730, a 537.30% higher paired point estimate; 95% interval [5.5766, 7.2937]. Passing required an upper bound below 0.95 plus all safeguards. A failed safeguard is not automatically proof of a regression beyond 2%.

## What was compared

This is an eager, single Qwen3-30B-A3B MoE layer on the same two H100s, using FCFS jobs and replicated-input TP2+EP. It is not a full engine, full-model generation, chat-request, TTFT or deployment result. All arms include the actual pinned stock gate/top-k and experts. Native untuned Triton resolved to M64; a delegate-only untimed probe observed CustomAllreduce without forcing it. Stockon32 used staged Gloo transport. Only ours changes the policy and physical dispatcher.

| Arm | Resident experts | Controller / mapping |
| --- | --- | --- |
| stockoff | 128 total; 64/rank | EPLB-default-off within this controlled layer harness; no added forced-SKIP recorder/controller. |
| stockon32 | 160 total; 80/rank | Stock enabled EPLB; 32 redundant experts; window 1,000 / interval 3,000; native asynchronous proposals and commits. |
| ours | 160 total; 80/rank | Resident one-home mapper + unchanged-map shortcut; window 16 / interval 64 forced-SKIP policy; all 64 proposals charged. |

Expert parameters occupy 576 MiB per rank for stockoff and 720 MiB for stockon32/ours, a 25% increase; all arms also have the same 0.5 MiB gate. Enabled EPLB adds buffers. Recorded process peak allocation includes all prepared comparator objects, inputs and references, so it is not standalone-engine memory.

Stockoff uses the EPLB default, not the entire shipped engine configuration. Hardware and maximum replica allowance match; actual residency does not. Stockoff cannot legally retain 32 redundant experts with EPLB disabled. This corrects the earlier comparison against an instrumented forced-SKIP baseline: its positive conditional results remain archived, but they did not establish superiority to stockoff. Neither present comparator represents every tunable vLLM configuration.

These are 4,096-job warm-process, reset-state episodes. Untimed conditioning exercises kernels and asynchronous transport; measured recording history and native counters are then reset. Stockon32 starts with a 750-step partial window, followed typically by one full 1,000-step window at its next native opportunity; observed deferrals and normal completion tails remain included. This is not a lifelong-adaptation or steady-state serving result.

## Prespecified paired inference

The final design contains 18 fresh content groups (six each GSM8K, MBPP and WildChat), two external Azure arrival types, two calibrated loads and three arms: 216 trials of 4,096 B128 layer jobs. A source timestamp releases a constructed fixed-token job; original request lengths and prompt contents are not executed together. Ratios first average paired logs over arrival types within content group, then groups within dataset and datasets equally. The 100,000-draw bootstrap resamples whole content bundles within dataset; jobs are not independent replicates.

| High-load mean response | Paired ratio | 95% interval | Decision |
| --- | --- | --- | --- |
| ours / stockoff | 6.3730 | [5.5766, 7.2937] | 5% target not established |
| ours / stockon32 | 0.7213 | not tested: hierarchy closed | descriptive point only |

The stockon32 secondary is confirmatory only after the stockoff primary, 14 regression guards and date-deletion checks pass. Guards use one-sided percentile upper bounds at 1 − 0.05/14 = 99.6429%, each required ≤ 1.02. They cover each dataset/load mean and aggregate high-load p95 against both controls.

| Guard endpoint | ours/off: point / upper; guard | ours/on32: point / upper; guard |
| --- | --- | --- |
| gsm8k; ρ=0.5; mean | 1.2700 / 1.3194; not established | 0.9725 / 0.9882; pass |
| gsm8k; ρ=0.85; mean | 5.7839 / 7.8818; not established | 0.7267 / 0.8459; pass |
| mbpp; ρ=0.5; mean | 1.2694 / 1.2962; not established | 0.9911 / 0.9989; pass |
| mbpp; ρ=0.85; mean | 6.3827 / 8.4675; not established | 0.8306 / 0.9255; pass |
| wildchat; ρ=0.5; mean | 1.2712 / 1.3120; not established | 0.9676 / 0.9774; pass |
| wildchat; ρ=0.85; mean | 7.0114 / 9.8864; not established | 0.6218 / 0.7459; pass |
| all; ρ=0.85; p95 | 5.3498 / 6.3968; not established | 0.7368 / 0.7946; pass |

Deleting each of 7 actual source dates gives high-load point-ratio ranges 5.9298–6.9740 versus stockoff and 0.7042–0.7508 versus stockon32. The ≤1.02 date-sensitivity safeguard was not established. These are finite-design point checks, not new-day confidence intervals.

## Both loads and completed work

| Load ρ | Contrast | Mean-response ratio | p95 ratio | Completed-extent ratio |
| --- | --- | --- | --- | --- |
| 0.5 | ours/stockoff | 1.2702 | 1.3385 | 1.0003 |
| 0.5 | ours/stockon32 | 0.9770 | 0.9970 | 1.0003 |
| 0.85 | ours/stockoff | 6.3730 | 5.3498 | 1.0091 |
| 0.85 | ours/stockon32 | 0.7213 | 0.7368 | 0.9921 |

The table above gives descriptive paired geometric ratios, without additional intervals or tests. Below are arithmetic means of complete trial summaries; they are different estimands and must not replace the primary paired endpoint.

| ρ | Arm | Response ms | Queue ms | Service ms | Completed s | Jobs/s | Tail ms | Drain incl. control ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | stockoff | 2.049 | 1.316 | 0.733 | 5.971 | 686.4 | 0.101 | 1.210 |
| 0.5 | stockon32 | 2.653 | 1.799 | 0.854 | 5.971 | 686.3 | 0.117 | 1.619 |
| 0.5 | ours | 2.593 | 1.754 | 0.839 | 5.973 | 686.2 | 0.089 | 3.185 |
| 0.85 | stockoff | 5.583 | 4.860 | 0.723 | 3.516 | 1165.6 | 0.101 | 4.515 |
| 0.85 | stockon32 | 51.944 | 51.100 | 0.844 | 3.576 | 1146.0 | 0.111 | 64.585 |
| 0.85 | ours | 37.588 | 36.760 | 0.828 | 3.548 | 1155.1 | 0.087 | 36.290 |

Response ends at inference-stream output completion plus CPU ACK. Completed extent includes normal asynchronous controller quiescence and the full offered horizon. Jobs/s is the arithmetic mean of each finite episode’s completed jobs divided by that extent, including horizon/drain; it is not saturated capacity. Tail is post-output wall time, not exclusive controller cost. Device-wide synchronization is absent from each job; native dependencies remain charged.

## Mechanisms and limits

| Structural measure | Mean ours/off ratio across group means |
| --- | --- |
| active_copies_sum | 1.0000 |
| active_copies_maxrank | 0.9462 |
| nonempty_expert_tiles_M64_maxrank | 0.9462 |

Total active-copy and M64 tile counts match for every compared job; their distribution across ranks can still differ. Saved actual stock routes authenticate these counts. M64 nonempty expert tiles are structural work measures, not measured CUDA duration. Raw service includes selection, native control, execution and coordination; no ablation isolates their individual costs.

Relative to stockoff, mean summed service difference was +149.665 ms/trial at our 64 control positions and +283.170 ms across the other 4,032 positions (negative favors ours). This partitions an observed difference, not causal component attribution. Native stockon32 completed 144 asynchronous cycles; 0 committed in the final tail and 0 trials deferred a boundary. Launch/commit neighborhoods are reported separately as temporal associations.

All 216 records passed independent raw-input/timing, native-lifecycle and statistical audits. Per content group, numerical coverage includes all 4,112 prepared batches against stockoff and 64 sampled token positions against dense FP32. Each trial also checks 16 whole postflight batches and every resident expert weight. Uncertainty is conditional on selected content/trace material, reused public weeks, constructed batches and one GPU allocation; it does not generalize to new production days, hardware populations or full-model serving. No final outcomes tuned the method or the 5%/2% margins.

Provenance: frozen plan 96b7cce1bb268e038991f13740d426405708bf3962e76fcacfef7aad4bcaa46e; execution source 9c6d6a7e1e94f4be769d435fbf7652e0cbb86501637c046ae078a0ccef1015d4. Complete raw records, frozen protocol, three audits and descriptive mechanism tables accompany this report.

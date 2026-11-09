# Resident-path repair: complete audited results

**The repair did not satisfy the frozen advance criteria.** Main acceptance: `not_established`; MBPP descriptive sentinel: point limit exceeded; overall decision: `do_not_advance`. These decisions use every prespecified final case.

At offered load 0.85, the main fix/stockoff mean-response ratio was **1.2853**, with conditional 95% interval **[1.2310, 1.3454]**. The useful-gain criterion requires the interval's upper bound to be strictly below 0.95 (at least 5% lower response). Its result was **not established**. A failure of this target alone is not proof of regression.

Here the entire primary interval is above 1, supporting higher mean response than stockoff within this conditional design. The practical recommendation for this tested configuration remains stock EPLB-off.

The fix/current high-load mean-response point ratio was **0.2033**. The inferential gate was closed. This point is descriptive; no secondary confidence interval or significance claim is assigned.

This is one eager BF16/B128 Qwen3-30B-A3B MoE layer on the same two H100s. Stock gate/top-k, native Triton experts and output all-reduce execute in every arm. Inputs are replicated across ranks. The result concerns completed FCFS layer jobs, not full-model chat latency, generated-token throughput or token all-to-all.

## Frozen decisions and uncertainty

| Endpoint | Ratio | 95% interval | Required upper bound | Status |
| --- | --- | --- | --- | --- |
| Fix / stockoff, high-load mean | 1.2853 | [1.2310, 1.3454] | < 0.95 | not established |
| Fix / current, gated high-load mean | 0.2033 | not tested | < 0.95 | not tested |

Main inference uses 16 whole content bundles: eight GSM8K and eight WildChat. Each bundle retains both arrival types, both loads and all arms. Paired log ratios are averaged across types, groups within dataset, and then the two dataset means equally. The 100,000 PCG64 bootstrap draws use seed 471712. Intervals are conditional on these constructed payloads, assigned public-calendar segments and one allocation. They do not quantify new-day, hardware-population or production-request uncertainty.

| Guard | Fix/control ratio | One-sided 99.5% UCB | Limit | Status |
| --- | --- | --- | --- | --- |
| stockoff gsm8k rho0.5 mean response | 1.0363 | 1.0423 | ≤ 1.02 | not established |
| stockoff gsm8k rho0.85 mean response | 1.1831 | 1.2509 | ≤ 1.02 | not established |
| stockoff wildchat rho0.5 mean response | 1.0549 | 1.0642 | ≤ 1.02 | not established |
| stockoff wildchat rho0.85 mean response | 1.3963 | 1.5531 | ≤ 1.02 | not established |
| stockoff high load p95 response | 1.3124 | 1.4124 | ≤ 1.02 | not established |
| current gsm8k rho0.5 mean response | 0.7993 | 0.8203 | ≤ 1.02 | pass |
| current gsm8k rho0.85 mean response | 0.1989 | 0.2600 | ≤ 1.02 | pass |
| current wildchat rho0.5 mean response | 0.8070 | 0.8310 | ≤ 1.02 | pass |
| current wildchat rho0.85 mean response | 0.2079 | 0.2256 | ≤ 1.02 | pass |
| current high load p95 response | 0.2453 | 0.2702 | ≤ 1.02 | pass |

All ten guards belong to the frozen Bonferroni family (0.05/10); the bound must be at most 1.02. The 5% useful-gain and 2% regression margins are carried-forward engineering criteria, separate from numerical tolerances.

| Actual date removed | Fix / stockoff | Fix / current | Both point limits ≤ 1.02 |
| --- | --- | --- | --- |
| 2024-05-12 | 1.2592 | 0.1894 | limit not met |
| 2024-05-13 | 1.2884 | 0.1962 | limit not met |
| 2024-05-14 | 1.3058 | 0.2277 | limit not met |
| 2024-05-15 | 1.2661 | 0.2027 | limit not met |
| 2024-05-16 | 1.2815 | 0.2140 | limit not met |
| 2024-05-17 | 1.3006 | 0.2033 | limit not met |
| 2024-05-18 | 1.2879 | 0.2026 | limit not met |

Date deletion removes each actual source date jointly across services, retains surviving types/groups with the frozen weighting, and fails if a dataset disappears. These are point sensitivity checks, not new confidence intervals or a new-day generalization test.

## Matched main outcomes at both loads

Loads 0.5 and 0.85 normalize each group’s common absolute arrival rate to its stockoff-only saturation calibration. They are not measured GPU utilization. All arms receive the same immutable calendar, retain original relative arrival structure, and drain every admitted job.

| Load | Treatment / control | Response mean | Trial response p95 | Service mean | Queue mean | Completed extent | Finite-episode throughput |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | fix / stockoff | 1.0456 | 1.0491 | 1.0279 | 1.0586 | 1.000026 | 0.999974 |
| 0.5 | fix / current | 0.8031 | 0.7520 | 0.8933 | 0.7490 | 0.999626 | 1.000374 |
| 0.5 | current / stockoff | 1.3019 | 1.3951 | 1.1507 | 1.4133 | 1.000400 | 0.999600 |
| 0.85 | fix / stockoff | 1.2853 | 1.3124 | 1.0271 | 1.3257 | 1.000650 | 0.999351 |
| 0.85 | fix / current | 0.2033 | 0.2453 | 0.8920 | 0.1846 | 0.986753 | 1.013425 |
| 0.85 | current / stockoff | 6.3215 | 5.3509 | 1.1515 | 7.1827 | 1.014083 | 0.986112 |

These are hierarchically aggregated matched geometric ratios over the 16 main bundles, without additional intervals. Lower latency/time ratios favor the numerator; higher throughput ratios favor it. Finite-episode throughput includes the offered horizon and drain and is not saturated capacity. Queue ratios would be N/A if any paired denominator/log ratio were undefined; no pair would be silently removed.

| Load | Arm | Response mean ms | Mean trial p95 ms | Service mean ms | Queue mean ms | Extent s | Drain ms | Tail ms | Jobs/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | stockoff | 2.022 | 4.590 | 0.739 | 1.283 | 6.018 | 1.718 | 0.094 | 680.728 |
| 0.5 | current | 2.615 | 6.335 | 0.850 | 1.765 | 6.021 | 4.122 | 0.087 | 680.455 |
| 0.5 | fix | 2.113 | 4.805 | 0.760 | 1.353 | 6.019 | 1.872 | 0.096 | 680.710 |
| 0.85 | stockoff | 6.033 | 19.120 | 0.728 | 5.304 | 3.548 | 8.345 | 0.100 | 1,154.856 |
| 0.85 | current | 38.355 | 91.022 | 0.839 | 37.516 | 3.598 | 58.549 | 0.088 | 1,138.895 |
| 0.85 | fix | 7.956 | 25.301 | 0.748 | 7.208 | 3.550 | 10.661 | 0.099 | 1,154.109 |

Absolute values are arithmetic means of trial summaries, with the same type/group/dataset hierarchy. “Mean trial p95” averages each completed trial’s p95; it is not a pooled-job p95. Drain is post-horizon extent through normal quiescence; tail is final-output-to-quiescence time. They are not isolated migration cost or GPU utilization.

## Every workload cell, including regressions

| Dataset | Arrival | Load | Bundles | Fix/off mean | Fix/off p95 | Fix/current mean | Fix/current p95 | Current/off mean | Current/off p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gsm8k | code | 0.5 | 8 | 1.0386 | 1.0432 | 0.7875 | 0.7314 | 1.3189 | 1.4263 |
| gsm8k | code | 0.85 | 8 | 1.2217 | 1.1936 | 0.2188 | 0.2996 | 5.5847 | 3.9843 |
| gsm8k | conversation | 0.5 | 8 | 1.0340 | 1.0328 | 0.8112 | 0.7700 | 1.2747 | 1.3413 |
| gsm8k | conversation | 0.85 | 8 | 1.1457 | 1.1772 | 0.1808 | 0.1871 | 6.3384 | 6.2905 |
| wildchat | code | 0.5 | 8 | 1.0596 | 1.0690 | 0.7866 | 0.7155 | 1.3471 | 1.4941 |
| wildchat | code | 0.85 | 8 | 1.4267 | 1.4285 | 0.2398 | 0.3227 | 5.9489 | 4.4262 |
| wildchat | conversation | 0.5 | 8 | 1.0503 | 1.0516 | 0.8280 | 0.7935 | 1.2685 | 1.3252 |
| wildchat | conversation | 0.85 | 8 | 1.3666 | 1.4783 | 0.1802 | 0.2000 | 7.5836 | 7.3902 |
| mbpp (sentinel) | code | 0.5 | 1 | 1.0338 | 1.0290 | 0.8158 | 0.8175 | 1.2672 | 1.2588 |
| mbpp (sentinel) | code | 0.85 | 1 | 1.1223 | 1.0941 | 0.2559 | 0.3706 | 4.3856 | 2.9522 |
| mbpp (sentinel) | conversation | 0.5 | 1 | 1.0410 | 1.0398 | 0.8119 | 0.7664 | 1.2823 | 1.3569 |
| mbpp (sentinel) | conversation | 0.85 | 1 | 1.2059 | 1.2112 | 0.2797 | 0.3613 | 4.3116 | 3.3526 |

All cells are descriptive; ratios above 1 remain visible. No subgroup interval or significance claim is added. MBPP cells each use one held-out bundle and remain outside the main bootstrap.

## MBPP operational sentinel

| Control | Load | Endpoint | Point ratio | Limit | Status |
| --- | --- | --- | --- | --- | --- |
| stockoff | 0.5 | response mean | 1.0374 | ≤ 1.02 | point limit exceeded |
| stockoff | 0.85 | response mean | 1.1634 | ≤ 1.02 | point limit exceeded |
| stockoff | 0.85 | response p95 | 1.1512 | ≤ 1.02 | point limit exceeded |
| current | 0.5 | response mean | 0.8138 | ≤ 1.02 | pass |
| current | 0.85 | response mean | 0.2675 | ≤ 1.02 | pass |
| current | 0.85 | response p95 | 0.3659 | ≤ 1.02 | pass |

These six checks average the two arrival types for one 32-document MBPP bundle. They have no confidence intervals or MBPP-population nonregression guarantee. The sentinel gates overall advance separately and cannot rescue or alter the main inference.

## What changed, and what the measurements identify

The 60-run development factorial held the 160-slot placement and exact physical assignments fixed. Its matched package effects were 51.872 µs/job for recording/control and 35.154 µs/job for online selection. They are coordinator-service contrasts between implementation packages, not exclusive kernel/CPU component times. Offline known-home assignments were diagnostic, not an online competitor or strict runtime lower bound.

The selected repair makes residency explicitly static, fuses histogram/home choice/remapping into one launch, and reuses output storage on one sequential inference stream. It preserves the current method’s exact physical assignments. In a separate 36-trial development check, service ratio was 0.892074 versus current and 1.021542 versus stockoff; every development group still lost to stockoff. This motivated fresh confirmation without establishing a repair advantage over stockoff. The repair changes the controller package and selector implementation together; it is not an isolated fusion ablation.

| Arm | Resident slots total / rank | Recording and control | Physical assignment |
| --- | --- | --- | --- |
| stockoff | 128 / 64 | Genuine upstream EPLB off | Native logical placement |
| current | 160 / 80 | 16-job observation window; 64-job forced-SKIP controller | Current online home policy |
| fix | 160 / 80 | Inactive; enabled-layout buffers remain allocated | Exactly the same policy, fused implementation |

A common maximum allowance of 32 replicas is not equal resident memory. The fix is custom static residency, not supported upstream EPLB-off. Its comparison with stockoff also changes placement and physical geometry; the residual time difference cannot be assigned solely to selector overhead.

| Main arm | All jobs ms | Positions 0–47 ms | Positions 48–62 ms | Position 63 ms |
| --- | --- | --- | --- | --- |
| stockoff | 0.733580 | 0.733729 | 0.733148 | 0.732937 |
| current | 0.844451 | 0.808253 | 0.811257 | 3.079870 |
| fix | 0.753829 | 0.753946 | 0.753525 | 0.752752 |

Actual final coordinator-service means at matched positions modulo 64, averaged over all main groups, loads and arrival types. Each trial contributes 3,072 nonrecording-position jobs, 960 recording-position jobs and 64 boundary-position jobs. The labels describe the current controller; stockoff/fix use the same positions without that controller. Temporal associations are not exclusive causal stage costs.

The repaired path still has 2.71% higher matched mean service than stockoff at high load. With identical absolute arrivals, its mean queueing ratio is 1.3257 and its response ratio is 1.2853. Small per-job delays accumulate under bursty load; improved service relative to the old method therefore does not establish an advantage over stockoff. This is observed FCFS behavior, not an extra GPU-work percentage or a full-serving prediction.

| Arm | Global active copies | Max-rank active copies | Max-rank routes | Global nonempty M tiles | Padded rows | Counted-tile GFLOP proxy | Useful/proxy fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| stockoff | 115.1127 | 60.2013 | 566.8666 | 115.1127 | 7,367.2100 | 69.5257 | 0.1395 |
| current | 115.1127 | 57.8058 | 513.4799 | 115.1127 | 7,367.2100 | 69.5257 | 0.1395 |
| fix | 115.1127 | 57.8058 | 513.4799 | 115.1127 | 7,367.2100 | 69.5257 | 0.1395 |

Actual-route work-proxy equality on every main measured batch (global active copies, nonempty M tiles, padded rows and counted-tile FLOPs) across all three arms: **true**. Current/fix assignment equality is required independently. These are structural proxies, not measured instructions, SM/tensor-core utilization, HBM traffic or critical-path time. Equal global work does not imply equal launch geometry, cache behavior or latency.

The attested common native configuration is `BLOCK_SIZE_K=64, BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, GROUP_SIZE_M=1, SPLIT_K=1, num_stages=3, num_warps=4`. No tile override was applied in this study.

| Arm | Sorted-ID allocation | M-grid positions | GEMM1 programs/rank | GEMM2 programs/rank |
| --- | --- | --- | --- | --- |
| stockoff | 9088 | 142 | 1704 | 2272 |
| current | 11104 | 174 | 2088 | 2784 |
| fix | 11104 | 174 | 2088 | 2784 |

Launch extents above are derived from pinned source and actual configuration, including early-return programs; they are not Nsight-observed grid counts or causal measured overhead. They cannot turn a count difference into an attributed runtime cost.

## Data, correctness and reproducibility

Final inputs comprise 544 fresh documents: 256 GSM8K, 256 WildChat and 32 MBPP, in 16 main groups plus one descriptive sentinel. There are 204 final trials (192 main, 12 sentinel). Across the full bounded study, 96 development executions, 264 fresh treatment trials and 22 baseline calibrations total 382 executions; capture/preparation/correctness work is separate. Pilot and final captures together contain 704 documents and 62,931 source-position records. Freshness excludes the project’s earlier selected identities/prefixes, not model pretraining or every possible semantic overlap.

Final source-document and Azure arrival-row intervals are disjoint from earlier selections. The public weeks and services are reused. Payloads and calendars are paired independently, so original production content–arrival dependence and original request lengths are absent. Every trial reuses a bounded bank in 4,096 constructed B128 jobs. Sixteen distinct conditioning-input combinations are cycled for 64 untimed forwards before state reset. Materialization, host-to-device transfer, reference computation and conditioning are excluded from timing.

The complete final independent record audit reconstructed 69,904 prepared batches and 69,904 greedy-assignment batches across 17 groups. GPU-recorded numerical coverage is all 4,112 prepared batches/group for current/fix against stockoff, 64 sampled stockoff FP32 token positions/group, and 16 whole postflight batches/trial, with resident map/weight checks and adversarial checks. Tolerances are `atol=0.02, rtol=0.03`; this joint criterion is not a universal absolute-error ceiling or downstream answer-quality result.

The selected fused path is limited to BF16/B128/top-8, the exact frozen 160-slot map, and sequential reuse on one constructing CUDA stream. Enabled-layout buffers remain allocated. Reported process peaks include comparator layers, prepared inputs and references and are not standalone engine memory. No claim is made for other model layers, batch sizes, models, hardware, all-to-all backends or concurrent serving.

The clean CPU replay passed all four phases (5 pilot calibrations, 60 pilot treatments, 17 final calibrations and 204 final treatments), with exact archived-analysis agreement after path/compression normalization. Independently implemented statistics compared 47,936 scalars; maximum absolute numerical difference was 5.55e-17. This is same-team replay of the same artifacts, not external-laboratory reproduction or an independent GPU replication.

Earlier negative results remain part of the evidence: [migration/payback evaluation](https://github.com/hsylin/cs471/blob/experiments/resident-static-fastpath/docs/payback-final-evaluation.md), [tile-tuning evaluation](https://github.com/hsylin/cs471/blob/experiments/resident-static-fastpath/docs/optimization-evaluation.md), and [stock-vLLM comparator study](https://github.com/hsylin/cs471/blob/experiments/resident-static-fastpath/docs/stock-comparison/results.md). This experiment does not retroactively change their designs or conclusions.

Source and evidence: [branch source](https://github.com/hsylin/cs471/tree/experiments/resident-static-fastpath/resident_fastpath); [frozen protocol](https://github.com/hsylin/cs471/blob/experiments/resident-static-fastpath/docs/resident-fix/protocol-v1.md); [selected implementation](https://github.com/hsylin/cs471/blob/experiments/resident-static-fastpath/configs/resident-fix-selected-v1.json); [final raw artifacts and audits](https://github.com/hsylin/cs471/tree/experiments/resident-static-fastpath/results/resident-fix/final-v1); [clean CPU proof](https://github.com/hsylin/cs471/tree/experiments/resident-static-fastpath/results/resident-fix/clean-cpu-reproduction-v1); [reproduction guide](https://github.com/hsylin/cs471/blob/experiments/resident-static-fastpath/docs/resident-fix/reproduction.md); [data provenance and terms](https://github.com/hsylin/cs471/blob/experiments/resident-static-fastpath/docs/resident-fix/data-provenance.md).

Frozen source `08d4b09f110b0ca225eb95073723b4c93a3ab799a0cf52b9c04ed0322ceba1ff`; final plan `c51db22a98b6817bc9f27b9c303c16966292bc2646d5c2b73f93f768be40a3cc`; protocol `e24962fedc6983e8050967e82f7800fa77a24baecddb75edb4a6c067d051f95e`; bootstrap draw digest `d146e6a35099d249d0429f283517faefd5e7d1029737dac3d12c68c1d3765621`. The capture ledger retains the archive-only failure after five valid pilot banks; no bank was recaptured or replaced.

| Required evidence artifact | SHA256 |
| --- | --- |
| plan | 270d548782592152b938200ca87b135eca48b9876371131504d70eb1261c01ca |
| analysis | f0342450ab60fdf50b86e6e77825dd8fc7a4ac3e916edb348870bd1662812c79 |
| records | 8c13067e4f674f489ee3e914106bc37717869c606b830a17d3f59407d0a0dfc0 |
| statistics | 60bdaab92e713b200f1c9f04d88ed440903ca9fc5c46455dba53b0dfffc306d6 |
| mechanism | 961ed77cb13639520fcb3cf6ef79b0061399f150e30a28b610d4b5912c559452 |
| reproduction | 2082c6281fd3ecd34577ede4da6c48e5e2a8062e01af3813401196e1fc972452 |

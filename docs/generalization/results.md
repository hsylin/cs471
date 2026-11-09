# Generalization result on new content and external arrivals

The frozen resident dispatcher adds **7.80% lower mean layer-job response latency** beyond the unchanged-map shortcut on the declared held-out workload construction. The conditional 95% interval is **6.82–8.75% lower**. The upper ratio bound is 0.9318, below the prespecified 0.95 useful-gain threshold. All seven 2% regression guards and all seven shared-date deletion checks pass. The gated comparison against native SKIP is **47.55% lower**, with a conditional 95% interval of **45.14–49.85% lower**.

This is an incremental fixed-layer result. It measures constructed B128 jobs using authentic, previously unused experiment content and scaled external timestamps on one reserved pair of H100s. It does not establish full autoregressive request latency, a population of future production days, a different allocation, or an all-to-all serving path. The dispatcher algorithm was frozen, and placement stayed fixed across arms. The earlier synthetic-calendar/reused-payload result and all negative migration/dispatcher results remain intact.

The final comparison completed 216 trials and 55,296 jobs: 18 document-disjoint content groups, six per dataset, crossed with two assigned arrival types and two rate definitions, with three contemporaneous arms. Each group contains 64 documents, so the final timings use 1,152 documents. The 72-trial pilot uses another 384 documents. All 1,920 candidate documents were captured and checked before timing; six unselected final groups were not chosen by performance. The precision-only pilot rule selected six groups per dataset. With 24 corrected diagnostic trials and 24 retained, ineligible first-attempt diagnostics, the treatment/diagnostic count is 336, below 384. Native-only calibration is additional preparation: 24 eligible calibrations plus six retained superseded calibrations.

## Frozen comparisons

The primary estimand is the geometric mean of paired arithmetic-mean response ratios, averaging arrival types within content group and weighting the three datasets equally. It is not a pooled traffic-weighted latency ratio. The 100,000 bootstrap draws resample entire content bundles within dataset, retaining both arrival types, both rates and all three arms. They condition on the selected external timestamp material. They do not estimate variation across new execution processes, machine allocations or production dates. Jobs and bootstrap draws are not independent workload samples. The seven guards use one-sided 99.2857% upper bounds, with a Bonferroni family correction; only the primary and the gated native comparison have the stated two-sided 95% intervals.

| Comparison, calibrated rate | Mean-response ratio | Conditional 95% interval | Interpretation |
| --- | ---: | --- | --- |
| Combined / shortcut | 0.9220 | [0.9125, 0.9318] | Incremental useful-gain test passes |
| Combined / native SKIP | 0.5245 | [0.5015, 0.5486] | Hierarchical secondary passes |
| Shortcut / native SKIP | 0.5689 | Not assigned a confirmatory interval | Descriptive existing-shortcut benefit |

The common absolute rate is 1,059.009 jobs/s, frozen from the geometric mean of the six eligible pilot native calibration cycles at load 0.85. The calibrated rate uses each final group's own native capacity at the same nominal load. Both arms in each pair see identical absolute arrivals. The code source blocks span 2.609–88.731 seconds and conversation blocks 3.600–12.181 seconds; exact rational scaling maps them to approximately 0.231–0.260-second offered intervals. These are externally derived arrival shapes at controlled layer-job rates, not unmodified production request execution.

| Guard / workload | Combined / shortcut mean response | Guard upper bound | Mean service reduction | Completed observed elapsed reduction |
| --- | ---: | ---: | ---: | ---: |
| gsm8k / calibrated | 0.9198 | 0.9356 | 3.26% | 0.026% |
| gsm8k / common | 0.9195 | 0.9295 | 3.29% | 0.081% |
| mbpp / calibrated | 0.9626 | 0.9793 | 1.93% | 0.031% |
| mbpp / common | 0.9573 | 0.9599 | 2.00% | 0.050% |
| wildchat / calibrated | 0.8852 | 0.9138 | 5.13% | 0.119% |
| wildchat / common | 0.8891 | 0.9150 | 5.19% | 0.126% |

Every mean-response guard upper bound is below 1.02. The seventh guard, aggregate calibrated finite-trial p95, has ratio 0.9320 and upper bound 0.9507, also passing. Removing each actual shared calendar date jointly gives primary point ratios 0.9191–0.9270. This is a finite-design sensitivity range, not an interval over unseen dates.

The benefit is not uniformly at least 5%. MBPP's calibrated point reduction is 3.74%, versus 8.02% for GSM8K and 11.48% for WildChat. All 72 matched final cases have lower mean response and mean service, but four finite-trial p95 comparisons regress by 0.32–1.48%: MBPP-final-05/conversation/calibrated, MBPP-final-06/conversation/calibrated, MBPP-final-08/conversation/common, and WildChat-final-07/conversation/common. Twenty completed-elapsed comparisons increase slightly, with the largest increase 0.209%. These observations are retained; no per-cell confidence or universal improvement claim is added.

## Timing and mechanism

Beyond the shortcut, calibrated mean service decreases 3.45%, mean response decreases 7.80%, and completed observed elapsed decreases only 0.059%. Completed jobs per observation second increases 0.059%. The fixed offered interval and complete drain make those last quantities explicit; this is not a 7.8% throughput or capacity improvement. The last-completion elapsed ratio is 0.99933, a 0.067% reduction. Input construction, full references and conditioning are separately recorded outside the service boundary; dispatch selection, command broadcast, expert execution, output allreduce, observation/planning, normal completion and both-rank ACK are charged.

Across calibrated paired cases, the arithmetic mean response difference is −209.35 microseconds per job: −24.17 microseconds of service and −185.18 microseconds of queueing. Their sum is an exact timing identity, not a causal mediation estimate. Small service savings reduce accumulated waiting under these bursty calendars. The much larger shortcut/native difference remains associated with the periodic unchanged-map path: the extra dispatcher makes a smaller contribution after that path has already been shortened.

The maximum per-rank active-copy count falls from 65.47 to 54.46 for GSM8K, 59.21 to 49.25 for MBPP, and 76.09 to 62.24 for WildChat. Every measured active copy occupies exactly one M tile, so padded-row work is 64 times active-copy count. Those are the same structural signal here. They are not two independent explanations, measured bandwidth, hardware utilization, or a proof that the maximum-count rank determines completion. Ordinary-step service falls by about 22.79, 13.07 and 37.85 microseconds respectively. Cadence-boundary service remains roughly 2.4 milliseconds and includes both control work and different input jobs.

Exploratory content-bundle correlations do not identify a robust online threshold. Maximum-rank active reduction and padding reduction have the identical Spearman correlation −0.496 with log response ratio; token-imbalance change is −0.420 and assigned gap CV is 0.110. Dataset, content and assigned-arrival effects are mixed in these 18 observations. These are descriptive associations, not fitted or validated predictors. The pilot showed no repeatable negative regime, so no fallback or new placement was selected.

The routing-only length check uses exactly the first eight captured token positions of every selected document, 9,216 positions in total. Dataset-pair routing TV distances remain substantial: GSM8K/MBPP changes from 0.347 under full captured prefixes to 0.467 under the match; GSM8K/WildChat is 0.226 to 0.227; MBPP/WildChat is 0.355 to 0.357. Prefix-position composition changes as well as count weighting. This establishes differing routing under this bounded match, not a matched-latency effect or a causal effect of prompt length.

The corrected 24-trial diagnostic subset uses one hash-selected pilot group per dataset. Its main sequence has 256 distinct jobs; the two short sequences repeat the same 32 jobs eight times and differ by a 16-job rotation. The rotation retains frequencies but changes input-to-arrival alignment and the 16-job observation windows at control boundaries. It is not a pure intervention on control cost.

| Development state | Combined/shortcut response, 256-job sequence | 32-job cycle, phase 0 | Same cycle, phase 16 | Precomputed online/native service ratio | Charged online minus precomputed-online service |
| --- | ---: | ---: | ---: | ---: | ---: |
| GSM8K | 0.8289 | 0.9270 | 0.9138 | 0.9381 | +28.13 µs/job |
| MBPP | 0.9707 | 0.8070 | 0.9355 | 0.9580 | +11.26 µs/job |
| WildChat | 0.7504 | 0.7142 | 0.7847 | 0.9072 | +22.66 µs/job |

The same-adapter precomputed assignment comparisons save 41.56, 27.70 and 67.71 microseconds of service per job respectively. Their assignments are the actual frozen native/online choices, not a runtime-optimal oracle. Construction is excluded, while lookup/copy/recording is charged. The online/precomputed-online difference includes both removed selection work and changed adapter costs. The corresponding native/precomputed-native differences range from 0.24 to 41.15 microseconds, exposing substantial path/timing sensitivity. In GSM8K the charged main gain is larger than the offline assignment-reference gain; the latter is therefore not a defensible latency ceiling. Three development states and one run per condition do not establish an intrinsic selector cost or a predictive fallback rule.

Sequence choice materially changes response ratios, including MBPP's 0.8070 versus 0.9355 between two equal-frequency cycles. The main conclusion uses the separately frozen 256-distinct-job final experiment, not whichever diagnostic sequence looks best. All diagnostic effects are descriptive, without confidence intervals or inclusion in the acceptance gate. The MBPP main-sequence diagnostic p95 is 0.94% higher, reinforcing the limit on uniform-tail claims.

The first 24 diagnostics are retained but ineligible: their floating combining-weight checksum incorrectly reused an integer-array hashing helper. The strict independent audit rejected every recorded checksum. Version 2 changes only post-timing attestation and its analysis contract, hashing exact little-endian float32 bytes; the same states, calendars, orders, sequences, algorithms and placement were frozen again before the 24-trial rerun. All corrected digests pass. The main study's integer route/index hashes and authenticated capture-byte hashes did not use the defective floating digest, and its scope was independently re-audited.

## Raw timing summaries

These are arithmetic averages of the twelve calibrated-rate trials per dataset/arm, each containing 256 completed jobs. Their ratios need not equal the geometric paired estimands above. P95 is the average of finite-trial p95 values. Observed elapsed extends through the offered horizon and any drain; last-completion elapsed is also retained in the machine-readable summary.

| Dataset | Arm | Mean response ms | P95 response ms | Service µs | Queue ms | Observed elapsed ms | Drain ms | Completed jobs/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gsm8k | native | 4.678 | 11.675 | 787.05 | 3.891 | 248.050 | 8.364 | 1032.24 |
| gsm8k | shortcut | 2.577 | 6.140 | 681.80 | 1.895 | 241.618 | 1.932 | 1059.65 |
| gsm8k | combined | 2.370 | 5.710 | 659.60 | 1.710 | 241.555 | 1.869 | 1059.92 |
| mbpp | native | 4.308 | 11.063 | 758.05 | 3.550 | 238.527 | 6.721 | 1073.49 |
| mbpp | shortcut | 2.469 | 5.710 | 652.12 | 1.816 | 233.163 | 1.356 | 1098.02 |
| mbpp | combined | 2.380 | 5.582 | 639.54 | 1.740 | 233.090 | 1.283 | 1098.36 |
| wildchat | native | 4.849 | 12.108 | 838.43 | 4.010 | 263.623 | 8.355 | 971.33 |
| wildchat | shortcut | 2.844 | 6.900 | 734.79 | 2.109 | 257.374 | 2.108 | 994.85 |
| wildchat | combined | 2.512 | 6.078 | 697.07 | 1.815 | 257.067 | 1.799 | 996.02 |

## Evidence and operating limits

The raw sources are filtered WildChat first-user text, MBPP task descriptions and GSM8K questions. Capture uses the pinned Qwen3-30B-A3B checkpoint's actual layer-0 embedding/attention/norm/router path, BF16, hidden 2,048/intermediate 768, 128 logical experts and top eight. There are no answer continuations, teacher-forced answers or substituted expert IDs. Each main job is a deterministic new combination of 128 captured token records; records recur across jobs. New documents and 256 distinct job combinations do not turn these into naturally batched requests or 256 independent sessions.

A two-pass full-text and model-visible prefix duplicate filter was frozen before accepted capture. The independent prefix audit found no qualifying pairs among the 1,920 accepted documents, and the separate check found no qualifying overlap against 56 earlier archived captured documents. These checks do not exclude semantic overlap, short shared fragments or model pretraining. WildChat comes from one bounded English/non-toxic shard prefix; MBPP and GSM8K are task datasets. Disjoint constructed bundles support the declared conditional analysis, not random-user or organization-population sampling.

Azure code dates are May 10–16, 2024 and conversation dates May 12–18. The first two dates within each service are pilot material. Final uses the remaining five per service, a seven-calendar-date union with overlap. This is not a global future-date holdout. Azure contains timestamps and lengths, not the original prompt text; this experiment cannot recover production content–arrival correlation. Original lengths are metadata, while execution remains one B128 layer job per timestamp.

The resident map is the existing content-independent `[0..127,0..31]`, 160 physical slots and 32 extra replicas, unchanged across datasets and arms. The backend replicates logical inputs and allreduces outputs. No partitioned-input all-to-all communication, KV cache, autoregressive decoder or production batcher is tested. Native vLLM remains pinned at `db9527a46873454610df6dbedf79a36d6bf1a7f6`, with the synchronous EPLB path. No weights migrate. Before/after map and weight checks, both-rank agreement, the guarded movement/normal-commit pairing and complete terminal state are checked; merely draining asynchronous work is not used as proof of commitment.

All 216 final records pass both the production validator and independent execution/statistical implementations. The independent audit reconstructs 9,792 native/online assignments from authenticated captures and checks 55,296 completed job clocks, 648 proposal windows, 216 terminal windows, physical/logical conservation and normal-state attestations. The independently reproduced 100,000-draw digest and all inference fields match. Corrected diagnostics pass a separate strict audit of 24 float-weight digests, 1,632 reconstructed assignments, 6,144 job clocks and 30 paired calculations. The complete CPU suite passes 696 tests and 112 subtests.

A clean extraction of the source/data archive reproduces all four main stages with zero numerical differences, including pilot allocation and calibration/common-rate provenance. A second clean extraction reproduces corrected diagnostic-v2: all 4,344 compared numeric values match, and the regenerated summary is byte-identical to the archived summary. These are independently implemented same-team audits and clean-directory CPU reproductions; no external laboratory has rerun the GPU experiment, and the CPU auditors do not independently recompute CUDA floating-point outputs. The archive and commands make that external test possible without changing the frozen acceptance criteria.

The source-selection mistakes were corrected before treatment timing and retained in the attempt ledger: full-text-only dedup missed truncated-prefix leakage; the Azure assets had a narrower date range than their README; and the first arm-order freezer was balanced globally but not within rate. Its six native calibrations and unexecuted treatment plan are superseded. Two shell-launch failures occurred before any trial, one from an unset working-directory variable and one from a launcher permission bit. Neither was replaced by an unreported performance rerun.

A valid mean-response upper ratio bound at or above 0.95 would have failed to establish the useful incremental gain. A guard upper bound above 1.02 would have left the safeguard unestablished; it would not itself prove harm. A date-deletion point above 1.02 would have failed the prespecified robustness criterion. Any routing, combining-weight, conservation, numerical, source, rank, state or completion failure would have invalidated acceptance. Earlier synthetic stress tests remain separate historical evidence and are not pooled into this external-workload estimate. The present aggregate passes these finite-design tests. It still does not establish at least 5% benefit on every dataset, every tail, every load or future arrivals.

The scientific contribution is a controlled incremental composition result with charged overhead and documented workload limits. Recognizing activated-expert cost is established prior work, and three arms do not estimate a factorial shortcut-by-dispatch interaction. The strongest course-project presentation preserves the earlier failed TV-gate, migration-payback and small-batch results, then explains this narrower positive result. Full-model integration remains optional future work.

The [literature synthesis](literature-review.md), [50-paper matrix](evidence-matrix.html), [closest-work appendix](closest-work-appendix.md), [frozen protocols](protocol-v1.md) with [v2](protocol-v2.md) and [v3](protocol-v3.md), [pilot decision](pilot-decision.md), [reproduction procedure](reproduction.md), and [raw final evidence](../../results/generalization/generalization-final-v1/) preserve the distinctions between literature, measured results, deductions and unresolved hypotheses.

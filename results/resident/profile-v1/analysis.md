The resident-copy assignment changes expert computation, but these instrumented timelines cannot establish an end-to-end speedup. Home assignment reduces both actual expert-GEMM durations and active/padded work. Large all-reduce spans primarily expose mismatched rank arrivals in this profile, including first-step delays of milliseconds.

All 5,632 of 5,632 recorded CUDA kernels in 16 reports were linked to their CUDA runtime launch and the innermost same-thread NVTX phase. All 32 inputs are retained. Exact bank, weight, source-map, kernel-configuration, and input-artifact invariants pass; native and precomputed-native physical assignment hashes agree.

| Batch | Arm | Rank | Active copies | Padded rows | Two GEMMs (µs) | Mapper kernel (µs) | NCCL median / max (µs) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 128 | native | 0 | 64.16 | 4106 | 290.76 | 3.32 | 17.09 / 101.73 |
| 128 | native | 1 | 64.12 | 4104 | 291.43 | 3.34 | 22.11 / 94.94 |
| 128 | pre_native | 0 | 64.16 | 4106 | 290.42 | 2.15 | 18.08 / 94.40 |
| 128 | pre_native | 1 | 64.12 | 4104 | 291.17 | 2.15 | 70.80 / 105.60 |
| 128 | one_copy_home | 0 | 52.44 | 3356 | 244.97 | 2.19 | 18.03 / 18.94 |
| 128 | one_copy_home | 1 | 53.03 | 3394 | 246.71 | 2.21 | 261.74 / 14575.28 |
| 128 | token_balanced | 0 | 62.34 | 3990 | 281.62 | 2.19 | 229.02 / 794.98 |
| 128 | token_balanced | 1 | 64.81 | 4148 | 295.71 | 2.20 | 18.13 / 18.91 |
| 1024 | native | 0 | 72.19 | 10676 | 372.13 | 3.10 | 103.76 / 141.41 |
| 1024 | native | 1 | 71.62 | 9780 | 364.83 | 3.12 | 58.32 / 131.58 |
| 1024 | pre_native | 0 | 72.19 | 10676 | 371.44 | 2.09 | 58.26 / 59.55 |
| 1024 | pre_native | 1 | 71.62 | 9780 | 365.48 | 2.09 | 108.85 / 10237.73 |
| 1024 | one_copy_home | 0 | 58.31 | 8780 | 317.52 | 2.10 | 153.76 / 2727.40 |
| 1024 | one_copy_home | 1 | 59.03 | 8480 | 313.66 | 2.08 | 58.16 / 59.81 |
| 1024 | token_balanced | 0 | 71.94 | 10584 | 368.33 | 2.05 | 58.32 / 132.13 |
| 1024 | token_balanced | 1 | 71.94 | 9856 | 368.20 | 2.08 | 62.37 / 131.39 |

Active/padded counts and GEMM/mapper durations in the table are 32-input means per rank. Padded rows use the resolved native M tile and count each owner route once; they are not measured FLOPs or memory bytes. Two-GEMM duration is a component statistic, not an additive wall-time share.

At batch 128, home/precomputed-native two-GEMM ratios are 0.8435 and 0.8473 on ranks 0/1. The mean of the per-input maximum-rank GEMM sums has ratio 0.8384. This is a consistent compute-headroom signal within this instrumented source block; it does not include online assignment construction or establish a 5% completed-time benefit.

At batch 1024, home/precomputed-native two-GEMM ratios are 0.8548 and 0.8582 on ranks 0/1. The mean of the per-input maximum-rank GEMM sums has ratio 0.8589. This is a consistent compute-headroom signal within this instrumented source block; it does not include online assignment construction or establish a 5% completed-time benefit.


The decisive waiting examples come from the paired timelines:

- Batch 128, one_copy_home, raw input 16: rank NCCL durations 18.82/14575.28 µs; GPU arrival skew 14558.80 µs; completion skew 2.34 µs; tail after both arrivals 18.82 µs. The host forward-entry skew is 14701.40 µs.
- Batch 1024, pre_native, raw input 16: rank NCCL durations 59.07/10237.73 µs; GPU arrival skew 10180.27 µs; completion skew 1.62 µs; tail after both arrivals 59.07 µs. The host forward-entry skew is 10284.64 µs.
- Batch 1024, one_copy_home, raw input 16: rank NCCL durations 2727.40/58.78 µs; GPU arrival skew 2670.89 µs; completion skew 2.28 µs; tail after both arrivals 58.78 µs. The host forward-entry skew is 2758.14 µs.

A long early-rank NCCL kernel ending almost simultaneously with the late rank's short NCCL kernel is direct evidence of a peer-arrival dependency. The traces locate the delay before the late rank's first forward or launch sequence; they do not uniquely attribute its host cause to Nsight, OS scheduling, or the Python/event instrumentation. cudaProfilerStart occurs after the pre-capture barrier, so collection startup is one plausible contributor. The short tail after both GPU arrivals is descriptive and is still not a transport-only latency measurement.

Precomputed-native and native retain the same assignments and similar GEMM times; replacing the mapper changes its actual GPU kernel by about 1 µs. Home and token-balanced assignments both produce exactly 512 or 4,096 owner-executed routes per rank at the two batch sizes. Yet home reduces both GEMMs while token balance alone leaves the maximum-rank GEMM statistic nearly unchanged. This localizes the observed component improvement to changed physical assignments rather than the mapper bypass or token balance alone. It does not separately identify active-expert count, padding, or memory-traffic effects. Reduced expert weight traffic is plausible, but no DRAM/L2/occupancy counters were collected to establish it.

Collection quality limits the timeline claims. Nsight reports that CUDA driver 12.6 is unsupported by this build and that tracing uses 12.5 libraries; it also warns that some CUDA/NVTX events might be absent. Every report nevertheless contains the expected 352 kernels and 160 application phase ranges, all linked, so the observed sequence is internally consistent. This check does not establish exhaustive collection of every runtime event. Scheduler/context-switch tracing was disabled, preventing a direct scheduling-cause diagnosis.

The appropriate causal test remains the separately timed apparatus-controlled pilot/fresh evaluation. These profiles were collected in a fixed arm order for one pilot source block/domain, with no repetitions or planner boundaries. They support testing one bounded online assignment mechanism only if the clean headroom gate passes; they cannot resolve the separate A/A outlier or waive that guard.

Limitations:

- No GPU counters were collected: weight-traffic, occupancy and bandwidth explanations remain hypotheses.
- Profile startup and per-phase events perturb rank arrivals; this is not the clean headroom pilot or a 5% efficacy test.
- Nsight warns that driver 12.6 is unsupported by this build and uses 12.5 collection libraries, with possible incomplete CUDA/NVTX events. Expected kernel/range counts pass, but exhaustive event coverage is not established.
- NCCL kernels can wait for peer ranks. Their durations are not pure transfer time.
- Nested event/CPU phase spans overlap. Kernel durations are not exclusive shares of completed wall time.
- All 32 steps retained. Median and identified first-step delays are descriptive; no outlier filter or inferential CI.
- Offline assignment construction is excluded; no demonstrated online selector cost or full-serving benefit.
- No planner opportunity occurs during these 32 profile steps; no migration or scheduling conclusion follows.

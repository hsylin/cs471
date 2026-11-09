The pilot calibration shows strongly sublinear runtime growth between the two batch sizes, while the profiler identifies fused expert kernels as the largest named compute component. It does **not** establish that batch 128 is memory-bound and batch 1024 is compute-bound, or that a placement change will pay for itself.

The captured bank is `pilot-01`: Alice in Wonderland (`gutenberg-11`) and CPython `Lib/heapq.py`. Its SHA-256 matches the pilot smoke provenance. For each domain and batch, the script takes the first 128 or 1,024 captured token records, runs a 64-forward B2 observation prefix to form a resident map, then 16 fixed-map warm-up forwards. Each of five unprofiled trials times 32 identical forwards and reports the maximum rank duration divided by 32. These are repeated measurements of one batch and map, not five independent workload samples. A separate profiler session records eight forwards per rank.

| Batch | Domain | Median unprofiled µs/forward | Mean | Trial range | Rank assignments | Active physical slots/rank |
|---|---|---:|---:|---:|---|---|
| 128 | Prose | 392.875 | 392.849 | 392.455–393.125 | 513 / 511 | 72 / 71 |
| 128 | Code | 403.662 | 412.321 | 403.147–446.730 | 505 / 519 | 74 / 75 |
| 1,024 | Prose | 528.552 | 528.593 | 528.440–528.963 | 4,106 / 4,086 | 77 / 76 |
| 1,024 | Code | 527.575 | 527.642 | 527.296–527.972 | 4,097 / 4,095 | 76 / 77 |

All five trials remain included. The code/batch-128 first trial is visibly slower, which explains its mean–median difference. Eight times as many tokens corresponds to 1.345× the median duration for prose and 1.307× for code. Each cell has a separately formed map, so this is a descriptive comparison of resident workloads, not a controlled batch-scaling curve with mapping held constant.

The actual native physical assignment counts are nearly balanced across ranks: maximum/mean ratios are 1.001953, 1.013672, 1.002441, and 1.000244 in table order. Nevertheless, 71–77 of 80 slots per rank are active. Batch 128 supplies only 6.82–7.20 assignments per active slot on average; batch 1,024 supplies 53.18–53.91. Maximum slot counts are 18, 12, 106, and 83 respectively. The nearly constant number of active expert slots helps explain why token count alone is an inadequate runtime model, but these observations do not measure the causal effect of changing a map or isolate weight traffic, launch cost, and padding.

The following are **profiled kernel durations**, converted to per-forward totals by dividing each eight-forward aggregate by eight. Each range spans the two ranks; it is not a confidence interval. The fused-kernel row includes 16 calls, two per forward.

| Batch/domain | Fused expert kernels, µs/forward | NCCL allreduce kernel, µs/forward | Physical mapping kernel, µs/forward |
|---|---:|---:|---:|
| 128/prose | 324.500–327.875 | 18.212–294.750 | 2.912–3.092 |
| 128/code | 333.625–338.375 | 16.820–43.001 | 2.932–3.100 |
| 1,024/prose | 381.625–385.125 | 59.584–85.371 | 2.892–2.900 |
| 1,024/code | 380.125–384.375 | 58.688–62.336 | 2.884–2.920 |

The profiled activation kernel takes approximately 5.01–5.16 µs per forward at batch 128 and 14.46–14.83 µs at batch 1,024; expert-output summation takes 4.12–4.30 and 17.37–17.64 µs respectively. These are component observations under profiling, not percentages of the unprofiled elapsed time. Host launch, framework work, synchronization, and collectives are present in the completed unprofiled timing.

The first profiler cell requires special caution. Its recorded self-CPU total is 65.562/65.575 ms across eight forwards on ranks 0/1, compared with 4.345–4.847 ms in the later sessions. The first cell's CPU `captured_fixed_map_moe` self time alone is 59.803/61.478 ms. This is consistent with a large first-profile/instrumentation effect; the table cannot uniquely establish its cause. That overhead is absent from the separately collected calibration samples. The first cell also has an extreme rank asymmetry in NCCL duration, 294.750 versus 18.212 µs per forward. Collective kernel duration can contain arrival/synchronization waiting and should not be called isolated network-transfer time.

Do not add the profiler rows into a wall-time breakdown. `vllm::fused_experts` and `fused_moe_kernel` repeat attributed device work; `record_param_comms`, `nccl:all_reduce`, and the NCCL kernel likewise refer to overlapping attribution. The tables also report annotation self-CUDA percentages above 100%, reaching 150.52%. CPU and GPU totals describe overlapping execution domains. The saved text is sufficient to identify kernels and approximate component magnitudes, but not an additive critical-path decomposition.

The hardware snapshot records two H100s, 95,830 MiB device memory each, and an NV6 connection. The driver is 560.35.03; `nvidia-smi` reports CUDA 12.6, while the running PyTorch build reports 2.13.0+cu129/CUDA 12.9 and vLLM 0.31.0. These labels describe different environment components. The snapshot does not measure in-run bandwidth, FLOP throughput, power, clocks, or occupancy. No roofline or hardware-counter evidence supports a hard bottleneck classification. Kernel-configuration fallback logs and Chrome traces were not present among the fetched profile files, so no fallback diagnosis is made here.

Numerical checks compare four reference tokens per batch. Recorded maximum absolute errors are 0.000774, 0.007279, 0.000957, and 0.007279, with tolerances `atol=0.02`, `rtol=0.03`; this is a sampled check, not validation of every token. The measured path remains replicated inputs, native fused MoE, and NCCL output allreduce. It excludes attention, logical router computation, production token all-to-all, and serving scheduling. No migration, admission-policy, three-arm payback, or final-data conclusion follows from this calibration.

Source files: `work/payback-results/profile/calibration.json`, eight `batch*-domain*-rank*.txt` tables, `nvidia-smi.txt`, `topology.txt`, and the read-only implementation `work/cs471/followup/profile_layer.py`. The companion `work/payback-profile-audit.json` retains all calibration trials, selected profiler rows, derived quantities, source-file hashes, and the same limitations.

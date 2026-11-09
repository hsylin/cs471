# Resident-dispatch architectural accounting

Descriptive follow-up to the completed frozen experiment; no new timing, threshold, policy selection or significance test.

Each top-8 assignment performs two GEMMs: H→2I costs 4HI FLOPs and I→H costs 2HI. With H=2048, I=768, the useful total is 9,437,184 FLOPs per assignment (multiply-add counts as two). Activation, reduction and routing arithmetic are excluded. We compute tile-implied work by replacing useful rows with Σ ceil(tokens_in_physical_slot/BM)×BM. N/K dimensions divide the recorded tiles exactly. This is a padded-GEMM work proxy, not a hardware instruction count.

Replicated ranks hold the same global assignment list. Each slot is owned by one rank: global useful work is B×8×6HI, not twice that quantity. Active-expert parameter footprint is a separate residency-access proxy; it is not measured HBM traffic.

## Saved final assignments

| Regime | Batch | Arm | Active copies | M tiles | Useful-row fraction | Useful GFLOPs | Tile-implied GFLOPs |
|---|---:|---|---:|---:|---:|---:|---:|
| persistent | 128 | native | 137.39 | 137.44 | 11.673% | 9.664 | 83.011 |
| persistent | 128 | online | 114.02 | 114.07 | 14.067% | 9.664 | 68.897 |
| persistent | 1024 | native | 148.43 | 160.85 | 39.807% | 77.309 | 194.299 |
| persistent | 1024 | online | 120.95 | 135.45 | 47.278% | 77.309 | 163.615 |
| stable | 128 | native | 126.78 | 127.01 | 12.614% | 9.664 | 76.713 |
| stable | 128 | online | 104.77 | 105.00 | 15.260% | 9.664 | 63.417 |
| stable | 1024 | native | 141.52 | 157.99 | 40.525% | 77.309 | 190.843 |
| stable | 1024 | online | 115.56 | 133.97 | 47.792% | 77.309 | 161.829 |
| switching | 128 | native | 132.34 | 132.48 | 12.121% | 9.664 | 80.013 |
| switching | 128 | online | 109.62 | 109.75 | 14.635% | 9.664 | 66.288 |
| switching | 1024 | native | 145.19 | 159.52 | 40.141% | 77.309 | 192.689 |
| switching | 1024 | online | 118.43 | 134.76 | 47.515% | 77.309 | 162.790 |
| transient | 128 | native | 127.23 | 127.46 | 12.573% | 9.664 | 76.980 |
| transient | 128 | online | 105.16 | 105.38 | 15.209% | 9.664 | 63.648 |
| transient | 1024 | native | 141.81 | 158.10 | 40.496% | 77.309 | 190.980 |
| transient | 1024 | online | 115.78 | 134.02 | 47.773% | 77.309 | 161.894 |

Each row averages source-state means; identical technical-repetition assignment artifacts are verified and counted once. Final summaries use all 512 measured inputs per state. Work proxies come from saved per-slot assignment arrays checked against recorded metrics, not from elapsed-time inference.

## Existing development kernel profiles

| Batch | Arm | Rank | Two GEMMs (µs) | Mapper kernel (µs) | Expert inter-kernel gaps (µs) | NCCL residence (µs) | Useful / tile-proxy TFLOP/s over GEMM residence |
|---:|---|---:|---:|---:|---:|---:|---:|
| 128 | native | 0 | 290.76 | 3.32 | 59.75 | 30.13 | 17.9 / 133.3 |
| 128 | pre_native | 0 | 290.42 | 2.15 | 61.41 | 20.60 | 17.9 / 133.4 |
| 128 | one_copy_home | 0 | 244.97 | 2.19 | 60.64 | 18.11 | 19.7 / 129.3 |
| 128 | token_balanced | 0 | 281.62 | 2.19 | 12.35 | 249.49 | 17.2 / 133.7 |
| 1024 | native | 0 | 372.13 | 3.10 | 18.14 | 107.48 | 112.0 / 270.8 |
| 1024 | pre_native | 0 | 371.44 | 2.09 | 52.13 | 58.28 | 112.2 / 271.3 |
| 1024 | one_copy_home | 0 | 317.52 | 2.10 | 12.71 | 240.03 | 121.8 / 261.0 |
| 1024 | token_balanced | 0 | 368.33 | 2.05 | 46.20 | 70.63 | 105.0 / 271.2 |
| 128 | native | 1 | 291.43 | 3.34 | 63.04 | 28.19 | 15.4 / 132.9 |
| 128 | pre_native | 1 | 291.17 | 2.15 | 60.21 | 67.13 | 15.4 / 133.0 |
| 128 | one_copy_home | 1 | 246.71 | 2.21 | 12.00 | 714.15 | 19.6 / 129.8 |
| 128 | token_balanced | 1 | 295.71 | 2.20 | 59.22 | 18.16 | 16.4 / 132.4 |
| 1024 | native | 1 | 364.83 | 3.12 | 56.21 | 60.62 | 97.7 / 253.0 |
| 1024 | pre_native | 1 | 365.48 | 2.09 | 12.30 | 432.71 | 97.5 / 252.5 |
| 1024 | one_copy_home | 1 | 313.66 | 2.08 | 56.57 | 58.26 | 123.3 / 255.2 |
| 1024 | token_balanced | 1 | 368.20 | 2.08 | 45.49 | 69.31 | 105.0 / 252.6 |

These are kernel-residence and geometry statistics from one development domain, not wall-time fractions or measured hardware utilization. `one_copy_home` here uses the offline DP assignment; it is not the online greedy kernel. The raw linked rows contain two fused GEMMs per input. Largest GEMM-residence rank and later expert-completion rank are both retained in JSON; profiling-induced rank start/arrival skew prevents treating either alone as an uninstrumented critical path. NCCL residence includes possible peer waiting.

## Online selector diagnostic

| Batch | Recording | Native stream µs/call | Online stream µs/call | Difference µs |
|---:|---|---:|---:|---:|
| 128 | False | 21.782 | 33.782 | 12.000 |
| 128 | True | 20.641 | 33.820 | 13.179 |
| 1024 | False | 20.970 | 33.920 | 12.950 |
| 1024 | True | 20.701 | 33.733 | 13.032 |

CUDA-stream intervals include launch gaps and use the existing fixed diagnostic order. They directly show combined selector/launch cost, but cannot isolate atomic-counter overhead or predict a fused implementation’s gain. Recording-on versus recording-off differences are not a randomized estimate of counter cost.

## Hardware-counter gap and bounded diagnostic

The installed absolute-path Nsight Compute is 2024.2.1.0; its CPU-only GH100 catalog query confirms all requested metric names. GPU counter permission and compatibility with the actual job remain unverified. A login-node query is not a successful counter capture. No GPU jobs were launched by this analysis.

A bounded follow-up would use only old development input raw16, both B128/B1024, both ranks, native/online, with identical map/weights and a warmed path. Collect the two expert GEMMs per case, explicitly excluding NCCL replay. Query the local metric catalog, then request DRAM read/write bytes, DRAM/SM throughput percentages, Tensor-pipe active cycles, achieved warp occupancy, issue activity, L2 hit ratio and kernel duration. Keep the counter run separate from primary latency. Use a one-kernel permission/compatibility probe first; if denied, retain the failure and request normal administrator support rather than changing system configuration.

Nsight replay can serialize kernels, change cache state and add passes. The old installed tool must not be assumed to support newer multi-process communicator features. Filter to local expert GEMMs; preserve/declare cache policy, use clock-control none if clocks cannot be set, and log tool/driver/kernel identity. One input is mechanism evidence only, not a new population performance claim.

Required metrics (GH100 names checked against the installed tool):

- `dram__bytes_read.sum`
- `dram__bytes_write.sum`
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`
- `sm__throughput.avg.pct_of_peak_sustained_elapsed`
- `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`
- `sm__warps_active.avg.pct_of_peak_sustained_active`
- `smsp__issue_active.avg.pct_of_peak_sustained_active`
- `gpu__time_duration.sum`
- `lts__t_sector_hit_rate.pct`

Measured DRAM bytes would permit useful-FLOPs/DRAM-byte arithmetic intensity. Tensor-pipe active cycles are not identical to achieved FLOP utilization; SM throughput is a composite peak-normalized metric, not `nvidia-smi` utilization; occupancy is active warps relative to the supported maximum. Report all with kernel identity and duration, not as interchangeable percentages.

NVIDIA explains replay/cache effects in the [profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) and supported metric/replay options in the [CLI reference](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html). Counter access can require administrator configuration; see [NVIDIA’s permission documentation](https://developer.nvidia.com/ERR_NVGPUCTRPERM). The pinned [vLLM kernel source](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/layers/fused_moe/fused_moe.py) defines the expert GEMM path.

The completed study’s acceptance/failure remains unchanged. These descriptive metrics explain candidate mechanisms and identify missing counters; they do not turn a 3.4% result into a passed 5% criterion.

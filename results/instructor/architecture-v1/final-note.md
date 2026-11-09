# Architectural diagnosis after the counter-access probe

The saved assignment arrays show that the online dispatcher reduces redundant active copies and padded expert work. They do **not** establish measured HBM bandwidth, Tensor Core utilization or SM occupancy. The attempted hardware-counter diagnostic was blocked by node permissions; the completed runtime study and its failed overall 5% acceptance criterion remain unchanged.

## Verified counter outcome

Allocation **4870934**, node **kh083**, completed in **41 seconds**. Nsight Compute **2024.2.1.0** returned `ERR_NVGPUCTRPERM` on device 0, and the recorded driver setting was `RmProfilingAdminOnly: 1`. The profiler exited with code 1; the wrapper deliberately stopped cleanly and recorded `counters_unavailable`. A read-only `sacct` check confirmed the batch's `COMPLETED / 0:0` status. That clean batch exit means the failure was handled, not that counters were collected.

The small Torch fill completed, but no `.ncu-rep`, metric CSV, per-rank expert-counter metadata or full-collection log was produced. The native-versus-online expert counter phase therefore did not run. No permissions, driver settings or system configuration were changed. NVIDIA documents this error as restricted GPU-counter access; enabling it requires administrator action. [NVIDIA counter-permission documentation](https://developer.nvidia.com/ERR_NVGPUCTRPERM).

## Architectural quantities that are available

The independent analysis checked 288 completed development/final records, deduplicated repeated assignment artifacts into 144 state-arm trajectories, and recomputed per-slot counts. Final ranges below are the four regime means, each averaged across eight source blocks; they are descriptive ranges, not confidence intervals.

| Batch / arm | Active physical copies | Padded M tiles | Useful-row fraction | Tile-implied two-GEMM work |
|---|---:|---:|---:|---:|
| 128 / native | 126.8–137.4 | 127.0–137.4 | 11.67–12.61% | 76.7–83.0 GFLOPs |
| 128 / online | 104.8–114.0 | 105.0–114.1 | 14.07–15.26% | 63.4–68.9 GFLOPs |
| 1,024 / native | 141.5–148.4 | 158.0–160.8 | 39.81–40.52% | 190.8–194.3 GFLOPs |
| 1,024 / online | 115.6–120.9 | 134.0–135.4 | 47.28–47.79% | 161.8–163.6 GFLOPs |

Useful two-GEMM work remains **9.664 GFLOPs at 128 tokens** and **77.309 GFLOPs at 1,024 tokens**. The calculation is `B × 8 × (4HI + 2HI)`, with H=2048 and I=768, counting a multiply-add as two FLOPs. Padded work substitutes each physical expert's rounded-up M rows, using the recorded BM=64 or 128. N/K dimensions divide the recorded tiles exactly. These are geometry-based work proxies: no measured instruction count, HBM byte count or hardware-utilization percentage is inferred. Replicated input copies do not double useful expert work because each physical slot has one owning rank.

## What the existing profiles establish

The old Nsight Systems profiles cover one development bank/domain, both batch sizes and both ranks. Their `one_copy_home` arm uses the **offline DP assignment**, not the online greedy dispatcher. Against the same precomputed-native adapter, its two fused-GEMM residence times are about **14–16% shorter**. This supports redundant-copy activation and padding as contributors to kernel cost. Rank arrival skew and NCCL waiting prevent treating those durations as fractions of the uninstrumented critical path; the JSON retains largest-residence and later-completion ranks separately.

Separate online development mapper-only stream intervals average roughly **33.8 µs online versus 20.7 µs native** with recording enabled, including host launch gaps. They show about **13 µs of combined added selector/launch cost**, but do not isolate atomic-counter cost or prove a gain from future fusion. Together, lower padded work and higher dispatch overhead provide a plausible explanation for the observed batch dependence. A bandwidth-versus-compute roofline classification remains unsupported.

The original `analysis.json`/`analysis.md` and raw failed probe are retained unchanged. This note supersedes only their pre-probe availability discussion. The archive includes an independent failure-outcome check and hashes of every raw counter-probe file. No new GPU measurement, policy retuning or revision to the original success/failure criteria was performed for this note.

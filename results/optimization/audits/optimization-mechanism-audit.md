# Actual selected-kernel mechanism audit

The verified traces identify the **two expert GEMMs as the largest repeated named GPU cost**, but show only a small local kernel improvement from the selected change. At batch 128, the selected configuration reduces the output-column tile N from 128 to 64; it does **not** reduce the M tile or the observed expert-token padding. At batch 1024, selection retained DEFAULT, so TUNED is a duplicate-control arm. The 64-run complete-policy pilot validation does not establish a useful overall improvement. No held-out final outcomes were inspected for this memo.

The independent verifier was rerun against the actual sixteen profile files and frozen validation manifest. It passed all 16 trace hashes, 16 input-artifact hashes, physical-count conservation and padding recomputations, 256 launch-warp checks and eight same-state rank comparisons. Profile timing uses only actual Chrome events with `cat='kernel'` and `ph='X'`; nested `vllm::fused_experts` and user annotations are excluded. The first profiler session was discarded before recorded comparisons, as declared.

## What changed on the GPU

| Batch/variant | M/N/K tile | Threads/block | First GEMM grid | Second GEMM grid | GEMM launches/input |
|---|---|---:|---:|---:|---:|
| 128 DEFAULT | 64/128/64 | 128 | 2088 | 2784 | 2 |
| 128 TUNED | 64/64/64 | 128 | 4176 | 5568 | 2 |
| 1024 DEFAULT | 128/128/64 | 256 | 2676 | 3568 | 2 |
| 1024 TUNED | 128/128/64 | 256 | 2676 | 3568 | 2 |

Grid entries are the actual first grid dimension; the other two dimensions are 1. These values hold for both domains and ranks. Thus the batch 128 override changed the executed launch geometry, not merely a configuration label. Halving N doubles the number of thread blocks in each GEMM grid; it does **not** double the number of kernel launches.

The gate/up and down output widths are 1536 and 2048, both divisible by 64 and 128. The observed physical assignments, active experts and padded M rows are identical across DEFAULT/TUNED. Consequently, this selected improvement cannot be described as removing token-row padding or N-edge padding. It repartitions the same nominal padded matrix work among smaller output-column tiles. A change in per-block resources, scheduling, concurrency or data reuse is plausible, but register, occupancy, cache and bandwidth counters were not collected to distinguish those explanations. Grid size is not itself executed FLOPs: the pinned kernel contains early-return/non-local-expert paths.

## Actual device-kernel observations

The following sums cover the two real GEMM kernel events per input. Ranges are across the two ranks and domains of one pilot bank, not confidence intervals.

| Batch | DEFAULT GEMM time/input | TUNED GEMM time/input | Paired TUNED/DEFAULT ratio |
|---:|---:|---:|---:|
| 128 | 289.608–331.510µs | 285.479–325.254µs | 0.98113–0.98574 |
| 1024 | 366.020–383.298µs | 366.202–382.983µs | 0.99826–1.00243 |

At batch 128, each rank/domain comparison saves 4.129–6.256µs across the two GEMMs (arithmetic mean 4.957µs, descriptive only). This is a 1.43–1.89% kernel-time reduction; most of the difference is in the first, larger gate/up GEMM. That first GEMM takes 188.804–217.919µs under DEFAULT versus 185.375–212.430µs under TUNED. The second changes from 100.804–113.591µs to 100.104–112.824µs.

At batch 1024 the settings and launch grids are identical, and the roughly ±0.25% GEMM differences are duplicate-control variation, not an optimization effect. The actual native mapping kernel is about 3.1–3.3µs/input. Other non-GEMM/non-NCCL device kernels total roughly 21µs/input at 128 and 55µs at 1024. These magnitudes support prioritizing the expert GEMM implementation as repeated work, without assigning exclusive wall-time percentages.

NCCL output-allreduce residence is 17.568–53.828µs at 128 and 59.612–67.636µs at 1024 across the compared traces. Some paired rank differences are much larger proportionally than the GEMM difference and change sign across ranks. Collectives include arrival waiting, so these observations do not identify network bandwidth, transport saturation or an independent communication speedup. CPU launch and framework costs also remain in complete-forward timings. No roofline, achieved FLOP/s, memory-throughput or occupancy measurement is available to classify the execution as memory-bound or compute-bound.

## Padding hypothesis versus observed selection

The original smaller-M hypothesis was reasonable from the exact per-expert token counts, but the selected configuration does not realize it. Both batch 128 variants keep M64. Their mean padded local rows/input are 4096–4664, with 64.0–72.875 active local slots. Both batch 1024 variants keep M128 and have 9760–10640 padded rows/input, with 71.5–76.375 active local slots. The verifier confirms a padded-row ratio of exactly 1 for every selected profile pair.

The bounded pilot search selected M64/N64 at 128 with a completed-forward candidate/default ratio 0.9861145. At 1024 the tested M16/N64 and M16/N128 combinations have ratios 1.1038639 and 1.0801035, respectively; selection correctly retained DEFAULT. Both M16 configurations also use four warps instead of DEFAULT's eight; the N64 combination additionally changes the N tile from 128 to 64. Thus the slowdown is evidence about these combinations, not a clean one-factor causal estimate that smaller M alone is harmful. Nor does reduced integer padding guarantee lower elapsed time: occupancy, reuse and useful per-block work can offset it.

## Complete-policy interpretation

The pilot validation contains 64 complete runs over the selected kernels crossed with B3/SKIP. Its six-changing-cell aggregate ratios are:

- TUNED+B3 / DEFAULT+B3: **1.0008435**.
- TUNED+SKIP / DEFAULT+SKIP: **0.9957496**.

These are nonconfirmatory pilot numbers with only two source-document blocks and one technical repetition. Half of the batch strata intentionally retain an identical DEFAULT setting. The small fixed-map GEMM improvement at 128 therefore should not be described as an established complete-policy speedup, and profiling alone cannot explain the sign or size of every policy-runtime difference. The final held-out comparison is a separate test of the frozen selection; these pilot results do not authorize another configuration search or post-hoc exclusion.

The scientifically supported mechanism statement is: **the selected batch 128 N-tile override changed the real GPU work partition and reduced expert-GEMM residence by about 1.4–1.9% in these matched pilot profiles, with unchanged routing and padding; its practical whole-policy benefit remains to be assessed independently.** The repeated costly component was correctly targeted, but a costly component is not necessarily an easily removable bottleneck.

## Recommendation if the held-out useful-gain test is negative

Do not present the selected tile as a demonstrated useful full-policy optimization or as removal of the main bottleneck. Preserve the frozen result and keep DEFAULT as the conservative operating choice unless a separately stated deployment requirement values a smaller, reproducible effect. Failure of the declared 5% useful-gain margin is not automatically evidence of slowdown; distinguish a small positive estimate, an inconclusive effect and an actual regression using the final uncertainty intervals.

The bounded engineering conclusion would be that expert GEMMs account for the largest named repeated device cost, yet this eight-choice tile search did not establish the required complete-policy benefit on the held-out corpus. It would not show that every GEMM implementation, H100 workload, or future tuning search lacks headroom. No additional candidate, post-hoc regime exclusion or new experiment is justified as part of interpreting this frozen test. The current intervention also leaves the number of CUDA kernel launches unchanged; it does not independently test whether reducing host launch overhead would help.

## Provenance and limits

The exact runtime remains the pinned Qwen layer 0/BF16, two-H100, fixed 32-replica, synchronous native path with replicated inputs and output allreduce. Profiles use one fresh pilot bank, two constructed domain samples, eight actual varying inputs per variant and 16 conditioning forwards. Ranks participate in the same collective execution and are not independent workload replications. Earlier profile and new profile datasets have different source/maps and must not be compared as an old-versus-new paired speedup.

Inputs: `work/optimization-results/validation/profile/profile-summary.json`, its 16 raw traces and saved index artifacts; `work/optimization-results/tuning/selection.json`; `work/optimization-results/validation-analysis/summary.json`. The new independent recheck is `work/optimization-mechanism-source-recheck.json`. The compact companion `work/optimization-mechanism-audit.json` retains source-file hashes, every matched launch-grid/kernel-time comparison, pilot candidate results and the same limitations.

No frozen source, configuration, threshold, recorded measurement or final outcome was changed or selected during this audit.

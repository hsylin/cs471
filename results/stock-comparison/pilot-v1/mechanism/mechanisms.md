# Descriptive stock-comparator mechanisms

Phase: pilot. 72 complete treatment trials. Descriptive only; no additional inference.

Service includes native control, routing, inference-stream completion and common coordination. Boundary and launch/commit patterns are temporal associations. Structural counts are not runtime predictions.

| Arm | Mean response (ms) | Mean service (ms) | Mean queue (ms) | Mean completed extent (s) | Mean output/controller drain (ms) |
|---|---:|---:|---:|---:|---:|
| stockoff | 3.7386 | 0.7193 | 3.0193 | 4.6949 | 7.2039 / 7.2911 |
| stockon32 | 26.9339 | 0.8429 | 26.0911 | 4.7471 | 59.3617 / 59.4743 |
| ours | 18.4489 | 0.8279 | 17.6210 | 4.7253 | 37.6507 / 37.7459 |

Means above average complete trial outcomes equally; they are not the frozen paired geometric endpoint.

| Arm | Mean service at64 fixed positions (ms) | At other4032 positions (ms) | Observed kernel M |
|---|---:|---:|---|
| stockoff | 0.7192 | 0.7193 | [64] |
| stockon32 | 0.8358 | 0.8430 | [64] |
| ours | 3.0748 | 0.7922 | [64] |

For stockoff/stockon32 these are the same job positions, not asserted native control boundaries.

| Stockon32 temporal neighborhood | Mean per-trial service (ms) | Total sampled jobs |
|---|---:|---:|
| launch_exact | 1.1742 | 48 |
| commit_in_job_exact | 6.0858 | 48 |
| outside_launch_commit_pm1 | 0.8394 | 98016 |

Event labels may overlap; these associations do not isolate the controller or migration cost.

| Contrast | Geometric mean response ratio | Geometric service ratio | Geometric completed-extent ratio | Mean service difference at64 positions (ms/trial) | At other4032 positions (ms/trial) |
|---|---:|---:|---:|---:|---:|
| ours/stockoff | 2.73126 | 1.15121 | 1.00867 | 150.7555 | 294.1453 |
| ours/stockon32 | 0.77855 | 0.98233 | 0.99382 | 143.2925 | -204.6435 |

Negative service differences mean ours consumed less total coordinator service. The two columns partition the difference arithmetically; neither isolates a causal mechanism.

| Structural measure | Mean ours/stockoff ratio of within-group means |
|---|---:|
| active_copies_sum | 1.00000 |
| active_copies_maxrank | 0.95041 |
| tokens_maxrank | 0.91346 |
| nonempty_expert_tiles_M64_sum | 1.00000 |
| nonempty_expert_tiles_M64_maxrank | 0.95041 |

Total active-copy and M64 tile counts match on every job in every group; any structural advantage lies in distribution across ranks.

Limitations:
- No selector-only or controller-only timing ablation; event timing cannot identify exclusive component cost.
- Work-count and max-rank changes do not imply runtime speedup; kernel execution, overlap, routing and coordination remain measured only in raw service.
- Stockoff128slots and ours160slots have different permitted residency; both use one physical home per active logical expert.
- Counts use the same constructed hidden jobs; tokens/jobs are not independent statistical observations.
- No new confidence intervals, p-values, margins, selection or tuning; final confirmatory inference remains the separately frozen analysis.

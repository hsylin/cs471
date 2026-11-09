# Descriptive stock-comparator mechanisms

Phase: final. 216 complete treatment trials. Descriptive only; no additional inference.

Service includes native control, routing, inference-stream completion and common coordination. Boundary and launch/commit patterns are temporal associations. Structural counts are not runtime predictions.

| Arm | Mean response (ms) | Mean service (ms) | Mean queue (ms) | Mean completed extent (s) | Mean output/controller drain (ms) |
|---|---:|---:|---:|---:|---:|
| stockoff | 3.8158 | 0.7277 | 3.0880 | 4.7432 | 2.7791 / 2.8622 |
| stockon32 | 27.2982 | 0.8490 | 26.4492 | 4.7734 | 33.0037 / 33.1019 |
| ours | 20.0905 | 0.8334 | 19.2571 | 4.7601 | 19.6570 / 19.7373 |

Means above average complete trial outcomes equally; they are not the frozen paired geometric endpoint.

| Arm | Mean service at64 fixed positions (ms) | At other4032 positions (ms) | Observed kernel M |
|---|---:|---:|---|
| stockoff | 0.7276 | 0.7277 | [64] |
| stockon32 | 0.8461 | 0.8491 | [64] |
| ours | 3.0661 | 0.7980 | [64] |

For stockoff/stockon32 these are the same job positions, not asserted native control boundaries.

| Stockon32 temporal neighborhood | Mean per-trial service (ms) | Total sampled jobs |
|---|---:|---:|
| launch_exact | 1.1693 | 144 |
| commit_in_job_exact | 5.7658 | 144 |
| outside_launch_commit_pm1 | 0.8457 | 294048 |

Event labels may overlap; these associations do not isolate the controller or migration cost.

| Contrast | Geometric mean response ratio | Geometric service ratio | Geometric completed-extent ratio | Mean service difference at64 positions (ms/trial) | At other4032 positions (ms/trial) |
|---|---:|---:|---:|---:|---:|
| ours/stockoff | 2.84518 | 1.14522 | 1.00469 | 149.6647 | 283.1700 |
| ours/stockon32 | 0.83950 | 0.98169 | 0.99618 | 142.0853 | -206.0503 |

Negative service differences mean ours consumed less total coordinator service. The two columns partition the difference arithmetically; neither isolates a causal mechanism.

| Structural measure | Mean ours/stockoff ratio of within-group means |
|---|---:|
| active_copies_sum | 1.00000 |
| active_copies_maxrank | 0.94622 |
| tokens_maxrank | 0.91738 |
| nonempty_expert_tiles_M64_sum | 1.00000 |
| nonempty_expert_tiles_M64_maxrank | 0.94622 |

Total active-copy and M64 tile counts match on every job in every group; their distribution across ranks can still differ.

Limitations:
- No selector-only or controller-only timing ablation; event timing cannot identify exclusive component cost.
- Work-count and max-rank changes do not imply runtime speedup; kernel execution, overlap, routing and coordination remain measured only in raw service.
- Stockoff128slots and ours160slots have different permitted residency; both use one physical home per active logical expert.
- Counts use the same constructed hidden jobs; tokens/jobs are not independent statistical observations.
- No new confidence intervals, p-values, margins, selection or tuning; final confirmatory inference remains the separately frozen analysis.

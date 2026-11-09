# Fused development audit

Arithmetic average of two technical repetitions within each content group, then geometric mean of six paired group ratios. No confidence intervals or tests.

| Matched contrast | Service ratio | Completed-elapsed ratio | Observed service group range |
|---|---:|---:|---:|
| static_fused_over_current_online | 0.892074 | 0.895295 | [0.882746, 0.899975] |
| static_fused_over_stockoff | 1.021542 | 1.021148 | [1.007921, 1.036421] |

All36 records, both ranks, archived source, input/assignment identities and18 adversarial case/dtype checks passed the independent CPU audit.

Cross-run three-kernel context is not a clean matched comparison:
- stockoff-normalized fused/three-kernel service ratio: 0.942603.
- current_online-normalized fused/three-kernel service ratio: 0.953048.

- Old development inputs only; not a held-out confirmation.
- The current-vs-fused matched contrast changes both controller and selector implementation, while preserving the physical-assignment policy.
- Fused160-slot residency and stock-off128-slot residency have different memory budgets.
- Service/completed intervals include common host/distributed orchestration; they are not GPU-kernel residence.
- Recorded GPU checks are audited, not externally rerun by this CPU script.

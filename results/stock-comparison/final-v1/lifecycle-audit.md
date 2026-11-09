# Native lifecycle audit

Status: PASS; completed-study lifecycle audit. Audited 216/216 trial records.

Native asynchronous arm: 72 trials, 144 launched and normally committed cycles, 144 with nonzero sender payload. 0 commits occurred in the post-output tail. 0 trials deferred a native boundary; 0 cycles sampled history spanning a placement change.

All interval summaries are descriptive. The ACK interval includes other work and scheduling, and is not network latency or exclusive migration cost. No arm speedup or statistical test is computed.

| Quantity | Minimum | Median | p95 | Maximum |
|---|---:|---:|---:|---:|
| Launch-to-commit step distance | 45 | 84 | 114.55 | 125 |
| Sender payload bytes/cycle | 3.11427e+08 | 5.19045e+08 | 6.44088e+08 | 6.7004e+08 |
| Post-launch ACK to commit ACK/quiescence (ns) | 6.32569e+07 | 9.99738e+07 | 1.20776e+08 | 1.31932e+08 |
| Native trial tail extent (ns) | 93139 | 112538 | 138770 | 154410 |

## Limits

- Descriptive counts and intervals only; cycles and repeated content/rates are not independent uncertainty units.
- This complements the separate full record/calendar/statistics auditors; it does not replace their source-file, calendar or inferential checks.
- Launch-to-commit ACK interval includes intervening work, arrivals and coordination; it is not exclusive planner, transfer or bandwidth time.
- Current-stream completion and later normal commits show allowed asynchronous progress, not independently measured GPU overlap.
- Resident-weight and numerical verification flags are recorded GPU assertions; this audit does not rerun CUDA.
- Route reconstruction authenticates saved stock routing bytes, not an independent rerun of gate/top-k.
- Mixed-map physical history is reproduced as native executes it; deviation from true logical demand is reported without excluding the native trial.

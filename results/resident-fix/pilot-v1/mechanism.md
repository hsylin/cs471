# Descriptive service and work proxies

Complete pilot matrix: 60 trials. No confidence intervals or new tests.

| Scope | Arm | All jobs ms | Boundary ms | Recording ordinary ms | Nonrecording ordinary ms |
|---|---|---:|---:|---:|---:|
| main | stockoff | 0.735792 | 0.734992 | 0.735780 | 0.735812 |
| main | current | 0.842376 | 3.075197 | 0.810318 | 0.805877 |
| main | fix | 0.750540 | 0.751450 | 0.750352 | 0.750580 |
| mbpp_sentinel | stockoff | 0.675424 | 0.675458 | 0.675026 | 0.675548 |
| mbpp_sentinel | current | 0.789409 | 3.076190 | 0.756014 | 0.752204 |
| mbpp_sentinel | fix | 0.694535 | 0.694297 | 0.694523 | 0.694544 |

All compared arms have equal per-batch global active-copy/nonempty-tile/padded-row/count-proxy values: True. See JSON for actual configuration checks and rank-imbalance summaries.

- Descriptive phase associations are not exclusive causal stage timings.
- Current/fix same assignment guarantees equal work proxies only with the attested common tile configuration.
- Equal global nonempty expert tiles do not imply equal launched programs, cache traffic, hardware instructions or runtime.
- Proxy FLOPs and useful fraction are not SM/tensor-core utilization, HBM bytes or roofline measurements.
- MBPP sentinel is reported separately and never pooled into the main summary.
- Independent execution/statistics audits remain required; this script adds no decision gate.

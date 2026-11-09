# Descriptive service and work proxies

Complete final matrix: 204 trials. No confidence intervals or new tests.

| Scope | Arm | All jobs ms | Boundary ms | Recording ordinary ms | Nonrecording ordinary ms |
|---|---|---:|---:|---:|---:|
| main | stockoff | 0.733580 | 0.732937 | 0.733148 | 0.733729 |
| main | current | 0.844451 | 3.079870 | 0.811257 | 0.808253 |
| main | fix | 0.753829 | 0.752752 | 0.753525 | 0.753946 |
| mbpp_sentinel | stockoff | 0.700092 | 0.701901 | 0.700842 | 0.699820 |
| mbpp_sentinel | current | 0.805862 | 3.082112 | 0.772587 | 0.768839 |
| mbpp_sentinel | fix | 0.718312 | 0.717015 | 0.718461 | 0.718293 |

All compared arms have equal per-batch global active-copy/nonempty-tile/padded-row/count-proxy values: True. See JSON for actual configuration checks and rank-imbalance summaries.

- Descriptive phase associations are not exclusive causal stage timings.
- Current/fix same assignment guarantees equal work proxies only with the attested common tile configuration.
- Equal global nonempty expert tiles do not imply equal launched programs, cache traffic, hardware instructions or runtime.
- Proxy FLOPs and useful fraction are not SM/tensor-core utilization, HBM bytes or roofline measurements.
- MBPP sentinel is reported separately and never pooled into the main summary.
- Independent execution/statistics audits remain required; this script adds no decision gate.

# Payback records

This archive is a separate follow-up to the original TV-gate experiment.
`results/final/` and its old frozen manifests remain unchanged. The follow-up
compares KEEP, the same candidate PREINSTALLED, and the same candidate CHARGED
with native transfer and normal commit included in its timer.

| Location | Role |
|---|---|
| `final/records.jsonl.gz` | All 1,536 declared held-out arm records, 512 matched triples. |
| `final/analysis/` | Frozen primary/secondary statistics, paired CSVs and figure. |
| `final/provenance/` | Executed manifest, source-file hashes, installed packages, allocation and hardware. |
| `pilot-native/`, `pilot-one-copy/` | Separate 96-arm screens. Neither qualified for confirmation or policy development. |
| `smoke/` | 24 correctness arms; `evidence_eligible=false`, excluded from performance evidence. |
| `inputs/` | Exact 10 captured banks and 80 canonical v2 tapes, plus immutable provenance inventories. |
| `profile/` | Separate pilot fixed-map calibration and CPU/CUDA profiler traces. |
| `audits/` | Independent arithmetic, source, raw-input/dispatch and protocol checks. |
| `execution-logs/`, `attempts/` | Successful and failed/cancelled startup records, retained separately. |
| `SHA256SUMS` | Raw-file hashes for every archived file except the inventory itself. |

The eight final document pairs are disjoint from the two pilot pairs. Technical
repetitions, regimes, batch sizes and horizons are correlated within a document
pair. The main analysis uses H512 and resamples whole blocks after averaging
paired log ratios over repetitions. H128 remains secondary. These intervals
describe the hand-selected corpus under a block-resampling assumption; they are
not production-workload population or cross-allocation confidence intervals.

The scope is one pretrained Qwen3-30B-A3B layer on two H100s, fixed 32 extra
replicas, real teacher-forced inputs arranged into constructed timelines,
replicated input/output-allreduce execution, synchronous `torch_nccl` migration,
and the pinned default fused-MoE kernel configuration. Common proposal/prefix
costs are sunk and excluded from this conditional diagnostic. No new online
controller was selected or timed after the negative pilot gate.

Start with [the measured report](../../docs/payback-final-evaluation.md),
[protocol](../../docs/payback-protocol.md), and
[reproduction instructions](../../docs/payback-reproduction.md).
`scripts/reproduce-payback-reports.sh` rebuilds the analysis and independent
checks without GPUs. Raw GPU timings are measured observations, not generated
examples; profiler and smoke measurements must not be pooled with final data.

# Equal-length routing supplement

The selected frozen final inputs contain 1,152 documents in 18 groups. This CPU-only comparison uses exactly eight captured token positions per document (9,216 tokens). It makes no latency or causal claim.

Each row averages group distributions equally. TV ranges refer to within-group full-prefix versus first-eight differences; entropy is for the dataset mean distribution. Replica demand is the share of logical top-k assignments belonging to replicated experts, not measured physical-copy traffic.

| Dataset | Groups | Full tokens | Mean within-group TV | TV range | Entropy full → first8 (bits) | Replicated demand full → first8 |
|---|---:|---:|---:|---:|---:|---:|
| gsm8k | 6 | 23,469 | 0.1671 | 0.1586–0.1742 | 6.5053 → 6.4529 | 0.2085 → 0.2233 |
| mbpp | 6 | 6,170 | 0.1668 | 0.1508–0.1751 | 6.2061 → 5.8302 | 0.2368 → 0.2552 |
| wildchat | 6 | 52,836 | 0.1900 | 0.1702–0.2096 | 6.8204 → 6.6055 | 0.2246 → 0.2339 |

| Dataset pair | Full-prefix TV | First8 TV |
|---|---:|---:|
| gsm8k / mbpp | 0.3467 | 0.4673 |
| gsm8k / wildchat | 0.2261 | 0.2270 |
| mbpp / wildchat | 0.3555 | 0.3572 |

The first-eight construction controls retained token count per prompt, while changing token-position composition and removing the full-prefix weighting of longer prompts. Remaining dataset differences are descriptive routing differences in these captures. They are not proof that length caused any runtime change. No timing outcomes, confidence intervals or policy decisions enter this report.

Frozen final plan: `84dc07a9e61a0ef2e0f601df51dfd3f187c71020120c676ece2609f3d387dddd`. Full per-group counts, probability vectors and exact source hashes are retained in the adjacent JSON.

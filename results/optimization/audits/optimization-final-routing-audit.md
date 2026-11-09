Independent optimization routing audit: verified.

Verified 512 runs in 128 matched quartets. Reconstructed all 3584 logged windows from actual captured logical IDs, including 9,175,040 nonnegative physical counter entries. Checked 11,776 complete maps of 160 slots, each retaining all 128 logical experts.

Current logical counts, exact token-index replica assignments, physical-to-logical conservation, accepted map chains, native sender bytes, and cross-kernel/repetition trajectories agree. Each run suppresses the step 511 terminal proposal. No elapsed-time values were analyzed.

| Kernel | Policy | Runs | Accepted decisions | Rejected decisions | Sender bytes |
|---|---|---:|---:|---:|---:|
| DEFAULT | B3 | 128 | 218 | 678 | 118,814,146,560 |
| DEFAULT | SKIP | 128 | 0 | 896 | 0 |
| TUNED | B3 | 128 | 218 | 678 | 118,814,146,560 |
| TUNED | SKIP | 128 | 0 | 896 | 0 |

Maximum recorded FP32 absolute error: 0.00392729044. Numerical scope and remaining limits:

- Exact raw routing reconstruction covers every logged decision window: seven 16-step windows per run. It does not add an observation record for the suppressed terminal window or unmonitored intervals.
- Map/weight identities and completed GPU numerical checks are execution attestations; this CPU audit does not rerun expert GEMMs or independently inspect GPU weight tensors.
- Numerical checks use full native batches versus DEFAULT plus 16 FP32 reference tokens at each of two raw endpoints, before and after timing. They do not check every measured output.
- The trace is constructed repeated-prefill replay on the named replicated-input/allreduce path, not production serving traffic. No performance ratios are computed by this auditor.

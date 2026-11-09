Independent optimization routing audit: verified.

Verified 64 runs in 16 matched quartets. Reconstructed all 448 logged windows from actual captured logical IDs, including 1,146,880 nonnegative physical counter entries. Checked 1,472 complete maps of 160 slots, each retaining all 128 logical experts.

Current logical counts, exact token-index replica assignments, physical-to-logical conservation, accepted map chains, native sender bytes, and cross-kernel/repetition trajectories agree. Each run suppresses the step 511 terminal proposal. No elapsed-time values were analyzed.

| Kernel | Policy | Runs | Accepted decisions | Rejected decisions | Sender bytes |
|---|---|---:|---:|---:|---:|
| DEFAULT | B3 | 16 | 23 | 89 | 12,409,896,960 |
| DEFAULT | SKIP | 16 | 0 | 112 | 0 |
| TUNED | B3 | 16 | 23 | 89 | 12,409,896,960 |
| TUNED | SKIP | 16 | 0 | 112 | 0 |

Maximum recorded FP32 absolute error: 0.00228646398. Numerical scope and remaining limits:

- Exact raw routing reconstruction covers every logged decision window: seven 16-step windows per run. It does not add an observation record for the suppressed terminal window or unmonitored intervals.
- Map/weight identities and completed GPU numerical checks are execution attestations; this CPU audit does not rerun expert GEMMs or independently inspect GPU weight tensors.
- Numerical checks use full native batches versus DEFAULT plus 16 FP32 reference tokens at each of two raw endpoints, before and after timing. They do not check every measured output.
- The trace is constructed repeated-prefill replay on the named replicated-input/allreduce path, not production serving traffic. No performance ratios are computed by this auditor.

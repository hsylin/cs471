# Independent pilot payback audit

Verified 96 complete GPU arm records in 32 matched triples and 16 source states. Exact manifest SHA: `96ea1ecb835ba96136242ab422ef67e3e430302510d6e9860e0f202e74fc28a1`. No frozen source or records were modified. This checker supports a complete pilot screening matrix or complete final matrix.

All structural, source/candidate pairing, native sender-payload, prefix histogram, arm-order, map/weight attestation, and sampled numerical checks passed. Maximum numerical absolute error: 0.00170078874; tolerance is atol 0.02 + rtol 0.03 × reference magnitude. Each native check uses the complete batch, with a four-token FP32 reference at each of the first and last future inputs.

one_copy screening: 0 changing H512 states qualify at ≥1 ms net payback, spanning 0 source blocks. Confirmation trigger: False. The trigger needs two distinct blocks; it is a feasibility screen, not a statistical test.

Source decisions at steps 63 and 127 are reproduced from the recorded physical windows and the frozen 2% equal-sharing peak-load threshold. Step 127 is always intercepted for the forced branch; “would allow” reports the original B3 decision.

| Block | Regime | B | Step63 committed | Step63 gain | Step127 would allow | Step127 gain | Prefix bytes |
|---|---|---:|---|---:|---|---:|---:|
| pilot-01 | persistent | 128 | False | 0.444% | False | 0.112% | 0 |
| pilot-01 | persistent | 1024 | False | 1.125% | False | 0.019% | 0 |
| pilot-01 | stable | 128 | False | 0.444% | False | 0.333% | 0 |
| pilot-01 | stable | 1024 | False | 1.125% | False | 1.324% | 0 |
| pilot-01 | switching | 128 | False | 0.444% | False | 1.044% | 0 |
| pilot-01 | switching | 1024 | False | 1.125% | False | 1.803% | 0 |
| pilot-01 | transient | 128 | False | 0.444% | False | 0.112% | 0 |
| pilot-01 | transient | 1024 | False | 1.125% | False | 0.019% | 0 |
| pilot-02 | persistent | 128 | True | 5.448% | True | 2.932% | 452,984,832 |
| pilot-02 | persistent | 1024 | True | 6.707% | False | 1.591% | 519,045,120 |
| pilot-02 | stable | 128 | True | 5.448% | False | 1.353% | 452,984,832 |
| pilot-02 | stable | 1024 | True | 6.707% | False | 0.141% | 519,045,120 |
| pilot-02 | switching | 128 | True | 5.448% | True | 7.426% | 452,984,832 |
| pilot-02 | switching | 1024 | True | 6.707% | True | 3.409% | 519,045,120 |
| pilot-02 | transient | 128 | True | 5.448% | True | 2.932% | 452,984,832 |
| pilot-02 | transient | 1024 | True | 6.707% | False | 1.591% | 519,045,120 |

Descriptive cell means over source blocks (positive payback/headroom means saved time). These descriptive means are not confidence intervals.

| Candidate | Regime | B | H | Headroom ms | Net payback ms | Preinstalled/keep | Charged/keep |
|---|---|---:|---:|---:|---:|---:|---:|
| one_copy | stable | 128 | 128 | 0.294 | -5.828 | 0.993724 | 1.124592 |
| one_copy | stable | 128 | 512 | 0.985 | -5.082 | 0.994653 | 1.027295 |
| one_copy | stable | 1024 | 128 | 0.109 | -5.868 | 0.998344 | 1.089537 |
| one_copy | stable | 1024 | 512 | 0.111 | -5.966 | 0.999573 | 1.022877 |
| one_copy | transient | 128 | 128 | 0.332 | -5.765 | 0.993008 | 1.122311 |
| one_copy | transient | 128 | 512 | 0.561 | -5.620 | 0.997040 | 1.030250 |
| one_copy | transient | 1024 | 128 | 0.196 | -6.062 | 0.997008 | 1.092633 |
| one_copy | transient | 1024 | 512 | 0.984 | -5.572 | 0.996224 | 1.021368 |
| one_copy | persistent | 128 | 128 | 0.550 | -5.570 | 0.989197 | 1.110188 |
| one_copy | persistent | 128 | 512 | 1.334 | -4.772 | 0.993397 | 1.023863 |
| one_copy | persistent | 1024 | 128 | 0.108 | -5.799 | 0.998368 | 1.087231 |
| one_copy | persistent | 1024 | 512 | 0.172 | -6.185 | 0.999351 | 1.023359 |
| one_copy | switching | 128 | 128 | 0.123 | -5.914 | 0.997557 | 1.121394 |
| one_copy | switching | 128 | 512 | -0.288 | -6.394 | 1.001577 | 1.033200 |
| one_copy | switching | 1024 | 128 | 0.119 | -5.755 | 0.998182 | 1.087375 |
| one_copy | switching | 1024 | 512 | 0.784 | -5.238 | 0.997006 | 1.019937 |

Raw archived input verification: True. Verified 2 bank files and 16 replay files, and reproduced 32 complete future physical-counter arrays. Prefix logical and physical counts, future SHA-256 values, owner counts, and active-expert counts match exactly.

Interpretation and audit limits:

- Screening uses one repetition per state; descriptive differences are not confirmatory estimates or a robust-feasibility pass.
- Common prefix/planner and candidate selection costs are excluded in all arms. This diagnoses a fixed candidate conditional on its state; it does not measure a full online controller.
- Raw authenticated bank/replay NPZ contents are independently verified; actual prefix logical/physical windows and complete future physical-array hashes are reproduced in NumPy using the pinned native mapping algorithm. The pretrained attention/router computation itself is not rerun.
- Numerical references cover four tokens at two future inputs per arm, not every output element.
- Full future physical counters are independently reconstructed from original logical IDs, replay indices and each arm map; this validates mapping diagnostics but does not profile GPU expert execution.
- Constructed replay of fixed captured prefill tokens, replicated-input/allreduce execution, one layer and allocation do not establish natural-serving or production-population effects.

Transient and persistent records have identical observed routing histograms, physical windows, source weights/map, and native proposal at step127 within each block and batch. The correspondence is about the common observed history; later branch outcomes remain distinct. Recorded old/new maps determine sender payloads exactly on the two-rank transport. PREINSTALLED and CHARGED diagnostics match exactly, while their timing contrasts can include cache/transition interference.

Saved analyzer point estimates and screening gate agreement: True. The detailed JSON retains every paired outcome and source-state decision.

# Independent pilot payback audit

Verified 96 complete GPU arm records in 32 matched triples and 16 source states. Exact manifest SHA: `2f71314b0564ad391d5d884f0c668923275c5887588017e7dd3062b911783fdf`. No frozen source or records were modified. This checker supports a complete pilot screening matrix or complete final matrix.

All structural, source/candidate pairing, native sender-payload, prefix histogram, arm-order, map/weight attestation, and sampled numerical checks passed. Maximum numerical absolute error: 0.00180661678; tolerance is atol 0.02 + rtol 0.03 × reference magnitude. Each native check uses the complete batch, with a four-token FP32 reference at each of the first and last future inputs.

native screening: 0 changing H512 states qualify at ≥1 ms net payback, spanning 0 source blocks. Confirmation trigger: False. The trigger needs two distinct blocks; it is a feasibility screen, not a statistical test.

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
| native | stable | 128 | 128 | -1.442 | -9.542 | 1.031428 | 1.204238 |
| native | stable | 128 | 512 | -6.205 | -14.873 | 1.033994 | 1.080552 |
| native | stable | 1024 | 128 | -0.702 | -8.890 | 1.010721 | 1.135996 |
| native | stable | 1024 | 512 | -2.868 | -11.201 | 1.010980 | 1.042946 |
| native | transient | 128 | 128 | -1.188 | -9.146 | 1.025678 | 1.194175 |
| native | transient | 128 | 512 | -5.777 | -13.600 | 1.031590 | 1.073560 |
| native | transient | 1024 | 128 | -0.386 | -8.024 | 1.005893 | 1.122608 |
| native | transient | 1024 | 512 | -1.408 | -9.380 | 1.005387 | 1.035945 |
| native | persistent | 128 | 128 | -0.267 | -8.322 | 1.005315 | 1.164279 |
| native | persistent | 128 | 512 | -1.803 | -10.371 | 1.009017 | 1.051678 |
| native | persistent | 1024 | 128 | -0.734 | -8.490 | 1.011086 | 1.128012 |
| native | persistent | 1024 | 512 | -2.982 | -10.537 | 1.011297 | 1.039830 |
| native | switching | 128 | 128 | -0.455 | -8.796 | 1.009472 | 1.180375 |
| native | switching | 128 | 512 | -2.335 | -10.877 | 1.012186 | 1.056289 |
| native | switching | 1024 | 128 | -0.322 | -8.146 | 1.004903 | 1.123766 |
| native | switching | 1024 | 512 | -1.189 | -8.840 | 1.004539 | 1.033658 |

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

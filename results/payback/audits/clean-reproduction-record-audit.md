# Independent final payback audit

Verified 1536 complete GPU arm records in 512 matched triples and 64 source states. Exact manifest SHA: `ee0236aa6d7f466e8bbfb8452b9983df3790d7447fba742d93b82151ab2ac706`. No frozen source or records were modified. This checker supports a complete pilot screening matrix or complete final matrix.

All structural, source/candidate pairing, native sender-payload, prefix histogram, arm-order, map/weight attestation, and sampled numerical checks passed. Maximum numerical absolute error: 0.00419467688; tolerance is atol 0.02 + rtol 0.03 × reference magnitude. Each native check uses the complete batch, with a four-token FP32 reference at each of the first and last future inputs.


Source decisions at steps 63 and 127 are reproduced from the recorded physical windows and the frozen 2% equal-sharing peak-load threshold. Step 127 is always intercepted for the forced branch; “would allow” reports the original B3 decision.

| Block | Regime | B | Step63 committed | Step63 gain | Step127 would allow | Step127 gain | Prefix bytes |
|---|---|---:|---|---:|---|---:|---:|
| final-01 | persistent | 128 | True | 5.670% | True | 2.207% | 500,170,752 |
| final-01 | persistent | 1024 | True | 5.769% | False | 1.179% | 528,482,304 |
| final-01 | stable | 128 | True | 5.670% | False | 0.195% | 500,170,752 |
| final-01 | stable | 1024 | True | 5.769% | False | 0.080% | 528,482,304 |
| final-01 | switching | 128 | True | 5.670% | True | 3.684% | 500,170,752 |
| final-01 | switching | 1024 | True | 5.769% | True | 2.342% | 528,482,304 |
| final-01 | transient | 128 | True | 5.670% | True | 2.207% | 500,170,752 |
| final-01 | transient | 1024 | True | 5.769% | False | 1.179% | 528,482,304 |
| final-02 | persistent | 128 | True | 10.626% | False | 0.748% | 500,170,752 |
| final-02 | persistent | 1024 | True | 11.210% | False | 0.022% | 462,422,016 |
| final-02 | stable | 128 | True | 10.626% | False | 0.335% | 500,170,752 |
| final-02 | stable | 1024 | True | 11.210% | False | 0.080% | 462,422,016 |
| final-02 | switching | 128 | True | 10.626% | False | 0.210% | 500,170,752 |
| final-02 | switching | 1024 | True | 11.210% | False | 0.067% | 462,422,016 |
| final-02 | transient | 128 | True | 10.626% | False | 0.748% | 500,170,752 |
| final-02 | transient | 1024 | True | 11.210% | False | 0.022% | 462,422,016 |
| final-03 | persistent | 128 | True | 5.368% | True | 9.054% | 528,482,304 |
| final-03 | persistent | 1024 | True | 5.492% | True | 4.810% | 556,793,856 |
| final-03 | stable | 128 | True | 5.368% | False | 0.589% | 528,482,304 |
| final-03 | stable | 1024 | True | 5.492% | False | 0.358% | 556,793,856 |
| final-03 | switching | 128 | True | 5.368% | True | 16.608% | 528,482,304 |
| final-03 | switching | 1024 | True | 5.492% | True | 9.950% | 556,793,856 |
| final-03 | transient | 128 | True | 5.368% | True | 9.054% | 528,482,304 |
| final-03 | transient | 1024 | True | 5.492% | True | 4.810% | 556,793,856 |
| final-04 | persistent | 128 | True | 5.134% | True | 5.661% | 509,607,936 |
| final-04 | persistent | 1024 | True | 5.208% | True | 6.045% | 462,422,016 |
| final-04 | stable | 128 | True | 5.134% | False | 1.380% | 509,607,936 |
| final-04 | stable | 1024 | True | 5.208% | False | 0.107% | 462,422,016 |
| final-04 | switching | 128 | True | 5.134% | True | 11.418% | 509,607,936 |
| final-04 | switching | 1024 | True | 5.208% | True | 11.582% | 462,422,016 |
| final-04 | transient | 128 | True | 5.134% | True | 5.661% | 509,607,936 |
| final-04 | transient | 1024 | True | 5.208% | True | 6.045% | 462,422,016 |
| final-05 | persistent | 128 | True | 6.764% | True | 2.292% | 537,919,488 |
| final-05 | persistent | 1024 | True | 6.671% | False | 0.247% | 500,170,752 |
| final-05 | stable | 128 | True | 6.764% | False | 0.524% | 537,919,488 |
| final-05 | stable | 1024 | True | 6.671% | False | 0.020% | 500,170,752 |
| final-05 | switching | 128 | True | 6.764% | True | 7.641% | 537,919,488 |
| final-05 | switching | 1024 | True | 6.671% | False | 0.190% | 500,170,752 |
| final-05 | transient | 128 | True | 6.764% | True | 2.292% | 537,919,488 |
| final-05 | transient | 1024 | True | 6.671% | False | 0.247% | 500,170,752 |
| final-06 | persistent | 128 | True | 7.125% | True | 2.627% | 471,859,200 |
| final-06 | persistent | 1024 | True | 7.206% | True | 7.070% | 537,919,488 |
| final-06 | stable | 128 | True | 7.125% | False | 1.257% | 471,859,200 |
| final-06 | stable | 1024 | True | 7.206% | False | 0.009% | 537,919,488 |
| final-06 | switching | 128 | True | 7.125% | True | 5.818% | 471,859,200 |
| final-06 | switching | 1024 | True | 7.206% | True | 13.208% | 537,919,488 |
| final-06 | transient | 128 | True | 7.125% | True | 2.627% | 471,859,200 |
| final-06 | transient | 1024 | True | 7.206% | True | 7.070% | 537,919,488 |
| final-07 | persistent | 128 | True | 4.558% | True | 6.105% | 481,296,384 |
| final-07 | persistent | 1024 | True | 3.955% | True | 6.807% | 566,231,040 |
| final-07 | stable | 128 | True | 4.558% | True | 2.239% | 481,296,384 |
| final-07 | stable | 1024 | True | 3.955% | False | 0.101% | 566,231,040 |
| final-07 | switching | 128 | True | 4.558% | True | 12.813% | 481,296,384 |
| final-07 | switching | 1024 | True | 3.955% | True | 12.695% | 566,231,040 |
| final-07 | transient | 128 | True | 4.558% | True | 6.105% | 481,296,384 |
| final-07 | transient | 1024 | True | 3.955% | True | 6.807% | 566,231,040 |
| final-08 | persistent | 128 | False | 1.859% | False | 1.128% | 0 |
| final-08 | persistent | 1024 | True | 3.308% | True | 3.830% | 490,733,568 |
| final-08 | stable | 128 | False | 1.859% | True | 2.144% | 0 |
| final-08 | stable | 1024 | True | 3.308% | False | 0.102% | 490,733,568 |
| final-08 | switching | 128 | False | 1.859% | False | 0.454% | 0 |
| final-08 | switching | 1024 | True | 3.308% | True | 7.187% | 490,733,568 |
| final-08 | transient | 128 | False | 1.859% | False | 1.128% | 0 |
| final-08 | transient | 1024 | True | 3.308% | True | 3.830% | 490,733,568 |

Descriptive cell means over source blocks (positive payback/headroom means saved time). These descriptive means are not confidence intervals.

| Candidate | Regime | B | H | Headroom ms | Net payback ms | Preinstalled/keep | Charged/keep |
|---|---|---:|---:|---:|---:|---:|---:|
| native | stable | 128 | 128 | -0.356 | -8.236 | 1.007752 | 1.172927 |
| native | stable | 128 | 512 | -1.600 | -9.554 | 1.008753 | 1.050660 |
| native | stable | 1024 | 128 | -0.019 | -8.031 | 1.000287 | 1.122086 |
| native | stable | 1024 | 512 | -0.338 | -8.330 | 1.001303 | 1.031800 |
| native | transient | 128 | 128 | 0.633 | -7.548 | 0.986703 | 1.158093 |
| native | transient | 128 | 512 | 2.724 | -5.449 | 0.985572 | 1.028688 |
| native | transient | 1024 | 128 | 0.684 | -7.727 | 0.989544 | 1.117278 |
| native | transient | 1024 | 512 | 2.692 | -5.790 | 0.989653 | 1.022012 |
| native | persistent | 128 | 128 | -1.183 | -9.438 | 1.024350 | 1.193068 |
| native | persistent | 128 | 512 | -4.774 | -13.067 | 1.024699 | 1.067324 |
| native | persistent | 1024 | 128 | 0.260 | -8.205 | 0.996163 | 1.121988 |
| native | persistent | 1024 | 512 | 0.775 | -7.656 | 0.997133 | 1.028594 |
| native | switching | 128 | 128 | 1.067 | -7.259 | 0.977876 | 1.150206 |
| native | switching | 128 | 512 | 4.212 | -4.227 | 0.978027 | 1.021953 |
| native | switching | 1024 | 128 | 1.427 | -7.156 | 0.978481 | 1.107431 |
| native | switching | 1024 | 512 | 5.410 | -3.180 | 0.979495 | 1.011914 |
| one_copy | stable | 128 | 128 | 0.165 | -6.002 | 0.996421 | 1.125707 |
| one_copy | stable | 128 | 512 | 0.283 | -5.639 | 0.998494 | 1.029725 |
| one_copy | stable | 1024 | 128 | 0.122 | -5.760 | 0.998153 | 1.087543 |
| one_copy | stable | 1024 | 512 | 0.194 | -5.680 | 0.999257 | 1.021672 |
| one_copy | transient | 128 | 128 | 0.230 | -5.641 | 0.995184 | 1.118136 |
| one_copy | transient | 128 | 512 | 0.663 | -5.263 | 0.996507 | 1.027758 |
| one_copy | transient | 1024 | 128 | 0.147 | -5.782 | 0.997767 | 1.087799 |
| one_copy | transient | 1024 | 512 | 0.480 | -5.407 | 0.998161 | 1.020618 |
| one_copy | persistent | 128 | 128 | 0.015 | -5.880 | 0.999753 | 1.120287 |
| one_copy | persistent | 128 | 512 | 0.007 | -5.889 | 1.000029 | 1.030334 |
| one_copy | persistent | 1024 | 128 | 0.280 | -5.658 | 0.995845 | 1.084154 |
| one_copy | persistent | 1024 | 512 | 0.851 | -5.026 | 0.996835 | 1.018775 |
| one_copy | switching | 128 | 128 | 0.102 | -5.776 | 0.997877 | 1.119449 |
| one_copy | switching | 128 | 512 | 0.234 | -5.648 | 0.998785 | 1.029404 |
| one_copy | switching | 1024 | 128 | 0.199 | -5.686 | 0.997004 | 1.085475 |
| one_copy | switching | 1024 | 512 | 0.617 | -5.337 | 0.997672 | 1.020136 |

Raw archived input verification: True. Verified 8 bank files and 64 replay files, and reproduced 192 complete future physical-counter arrays. Prefix logical and physical counts, future SHA-256 values, owner counts, and active-expert counts match exactly.

Interpretation and audit limits:

- Technical repetitions share source blocks. This structural audit reports point estimates; confirmatory uncertainty must resample the eight full source-document blocks jointly and use the frozen endpoint families.
- Common prefix/planner and candidate selection costs are excluded in all arms. This diagnoses a fixed candidate conditional on its state; it does not measure a full online controller.
- Raw authenticated bank/replay NPZ contents are independently verified; actual prefix logical/physical windows and complete future physical-array hashes are reproduced in NumPy using the pinned native mapping algorithm. The pretrained attention/router computation itself is not rerun.
- Numerical references cover four tokens at two future inputs per arm, not every output element.
- Full future physical counters are independently reconstructed from original logical IDs, replay indices and each arm map; this validates mapping diagnostics but does not profile GPU expert execution.
- Constructed replay of fixed captured prefill tokens, replicated-input/allreduce execution, one layer and allocation do not establish natural-serving or production-population effects.

Transient and persistent records have identical observed routing histograms, physical windows, source weights/map, and native proposal at step127 within each block and batch. The correspondence is about the common observed history; later branch outcomes remain distinct. Recorded old/new maps determine sender payloads exactly on the two-rank transport. PREINSTALLED and CHARGED diagnostics match exactly, while their timing contrasts can include cache/transition interference.

Saved analyzer point estimates and screening gate agreement: True. The detailed JSON retains every paired outcome and source-state decision.

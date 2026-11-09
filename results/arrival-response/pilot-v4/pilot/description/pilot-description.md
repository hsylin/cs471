# Descriptive unchanged-map pilot

All 24 pilot trials and four diagnostic trials are retained. A/A ratios are descriptive: there is no timing eligibility threshold, confidence interval, or performance-based selection. N/A/F means native/native alias/fast.

| B | Arrivals | Repeat | Mean response N/A/F (ms) | A/A response ratio | Fast/native response ratio | Mean service N/A/F (ms) | Total charged service N/A/F (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | burst8 | 0 | 4.426 / 4.168 / 3.239 | 0.9418 | 0.7318 | 0.838 / 0.819 / 0.721 | 214.430 / 209.554 / 184.532 |
| 128 | burst8 | 1 | 4.146 / 4.250 / 3.227 | 1.0249 | 0.7782 | 0.816 / 0.824 / 0.718 | 209.021 / 210.829 / 183.687 |
| 128 | constant | 0 | 2.089 / 1.983 / 0.820 | 0.9493 | 0.3924 | 0.824 / 0.822 / 0.720 | 211.038 / 210.392 / 184.252 |
| 128 | constant | 1 | 2.054 / 2.042 / 0.814 | 0.9945 | 0.3964 | 0.830 / 0.828 / 0.717 | 212.517 / 211.850 / 183.426 |
| 1024 | burst8 | 0 | 4.806 / 4.849 / 3.915 | 1.0091 | 0.8146 | 0.960 / 0.956 / 0.872 | 245.726 / 244.715 / 223.228 |
| 1024 | burst8 | 1 | 4.778 / 4.775 / 3.857 | 0.9994 | 0.8073 | 0.956 / 0.959 / 0.857 | 244.847 / 245.538 / 219.299 |
| 1024 | constant | 0 | 2.101 / 2.077 / 0.955 | 0.9887 | 0.4548 | 0.956 / 0.956 / 0.858 | 244.639 / 244.855 / 219.660 |
| 1024 | constant | 1 | 2.049 / 2.094 / 0.956 | 1.0219 | 0.4666 | 0.952 / 0.962 / 0.860 | 243.806 / 246.241 / 220.175 |

| B | Arm | Ordinary service (ms) | Boundary service (ms) | Terminal service (ms) |
| --- | --- | --- | --- | --- |
| 128 | native | 0.716 | 7.816 | 6.381 |
| 128 | native_alias | 0.712 | 7.820 | 6.386 |
| 128 | fast | 0.691 | 2.444 | 0.950 |
| 1024 | native | 0.844 | 8.042 | 6.855 |
| 1024 | native_alias | 0.845 | 8.069 | 6.761 |
| 1024 | fast | 0.834 | 2.583 | 1.102 |

Ordinary service excludes jobs 63/127/191/255. Boundary service averages those four jobs. These are charged command-to-ACK intervals, not isolated GPU kernels.

| B | Diagnostic arm | Rank | Equality (ms) | Agreement (ms) | Original movement (ms) | Original commit (ms) | Movement wrapper (ms) | Commit wrapper (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | native_observe | 0 | 0.023 | 0.081 | 1.134 | 4.314 | 1.261 | 4.316 |
| 128 | native_observe | 1 | 0.023 | 0.087 | 1.136 | 4.263 | 1.267 | 4.265 |
| 128 | fast | 0 | 0.022 | 0.083 | 0.000 | 0.000 | 0.126 | 0.001 |
| 128 | fast | 1 | 0.023 | 0.093 | 0.000 | 0.000 | 0.137 | 0.001 |
| 1024 | native_observe | 0 | 0.022 | 0.086 | 1.144 | 4.346 | 1.275 | 4.348 |
| 1024 | native_observe | 1 | 0.023 | 0.098 | 1.139 | 4.308 | 1.282 | 4.309 |
| 1024 | fast | 0 | 0.023 | 0.091 | 0.000 | 0.000 | 0.135 | 0.001 |
| 1024 | fast | 1 | 0.023 | 0.096 | 0.000 | 0.000 | 0.141 | 0.001 |

Diagnostic entries average the two measured boundaries in each separate 128-job diagnostic. The CSV retains both stage pairs. Function-wall intervals include natural blocking; rank intervals overlap and must not be summed as serial CPU/GPU time. The observed-native wrapper is diagnostic and is not the unwrapped timing baseline.

| Untimed proof step | Expected skip | Actual sender bytes | Actual messages | Numerical check |
| --- | --- | --- | --- | --- |
| unchanged_before | True | 0 | 0 | True |
| changed_swap_0_80 | False | 9437184 | 1 | True |
| restore_original | False | 18874368 | 2 | True |
| unchanged_after_restore | True | 0 | 0 | True |

Metadata/proof checks: True. This is a structural check, not a timing gate. Raw records and source/proof checksums remain the evidence; this report does not replace the complete independent route/weight audit.

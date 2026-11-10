# Independent compiled-launch development audit

PASS: 24 complete records; unchanged physical assignment and native geometry verified.

| Arm | Mean service µs/job | Completed µs/job |
|---|---:|---:|
| stockoff | 721.681 | 735.261 |
| previous | 741.122 | 754.803 |
| fix | 734.111 | 747.833 |

| Group | Fix/previous service | Fix/stockoff service |
|---|---:|---:|
| fix-gsm8k-pilot-01 | 0.978018 | 1.006786 |
| fix-gsm8k-pilot-02 | 1.001600 | 1.025247 |
| fix-wildchat-pilot-01 | 1.006102 | 1.021795 |
| fix-wildchat-pilot-02 | 0.976739 | 1.014945 |

- Descriptive reused-content results; technical repetitions averaged within four groups; no CI or p-value.
- Host-submission measurements exclude completion wait; they are not exclusive GPU time or layer-service components.
- GPU functional checks are authenticated recorded evidence, not a new GPU execution by this CPU verifier.

# Identical-input factorial diagnostic

Arithmetic mean coordinator service per trial; average technical repetitions within group; equally weight six content groups.

| Matched contrast | Service ratio | Observed group range | Saved µs/job |
|---|---:|---:|---:|
| control_removal_online | 0.936022 | [0.928965, 0.947028] | 52.407 |
| control_removal_known_homes | 0.934481 | [0.917974, 0.945267] | 51.338 |
| selection_removed_current | 0.956330 | [0.948865, 0.962168] | 35.688 |
| selection_removed_static | 0.954757 | [0.932650, 0.967215] | 34.620 |
| static_online_over_stockoff | 1.083746 | [1.064225, 1.098885] | -59.295 |
| static_known_homes_over_stockoff | 1.034713 | [1.018528, 1.053488] | -24.676 |

Control-package main effect: 51.872µs/job. Selection-package main effect: 35.154µs/job. Interaction: 1.068µs/job.

- Only six previously measured development groups; this is not fresh held-out evidence.
- Matched ablations change the declared control or selection package; differences do not identify individual Python/GPU operations.
- Offline homes use prior knowledge and are never an online-production comparison.
- Clock intervals include coordinator protocol and GPU completion; not device-kernel residence.
- Reported ranges are the six observed content-group effects, not uncertainty intervals or advancement tests.
- Floating-point correctness comes from runner assertions; this CPU analysis does not rerun GPU arithmetic.

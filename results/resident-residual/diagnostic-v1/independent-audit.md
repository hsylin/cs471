# Independent residual diagnostic audit

PASS: 56 records; four reused content groups, two reversed technical repetitions. No inferential claims.

| Variant | Mean service µs/job | Completed µs/job |
|---|---:|---:|
| stockoff | 726.946 | 740.902 |
| stockoff_trim | 733.702 | 747.507 |
| previous | 744.478 | 757.887 |
| previous_trim | 751.191 | 765.010 |
| replay_checked | 714.928 | 728.845 |
| replay_checked_trim | 724.834 | 738.640 |
| replay_unchecked | 715.143 | 728.990 |

Positive package effects below mean the named package costs time; the extent contrast is full minus trim.

| Paired effect | Mean µs/job | Group range |
|---|---:|---:|
| selector_invocation_package_full_us | 29.549 | [21.421, 46.504] |
| selector_invocation_package_trim_us | 26.356 | [20.979, 32.848] |
| launch_extent_package_online_us | -6.713 | [-17.384, 1.454] |
| launch_extent_package_replay_us | -9.906 | [-17.000, -5.957] |
| launch_extent_package_stockoff_us | -6.756 | [-9.128, -5.154] |
| selector_by_extent_interaction_us | 3.193 | [-11.428, 18.454] |
| input_validation_only_us | -0.214 | [-6.063, 8.415] |
| online_full_minus_stockoff_us | 17.532 | [10.224, 34.724] |
| online_trim_minus_stockoff_us | 24.245 | [13.154, 33.270] |
| online_trim_minus_trimmed_stockoff_us | 17.488 | [8.000, 26.481] |

Per-group paired differences and completed-time counterparts are in the JSON.

- Descriptive reuseddevelopmentinputs; no CI,pvalue,or heldoutclaim.
- Technical repeats averaged within group;512jobs and64cycledinputs are not independent units.
- Packageeffects include chargedPythonchecks/viewcreation and GPU effects; not exclusive component timing.
- Offline physical-ID replay is not deployable and not a runtime lowerbound.
- Recorded numerical and slot-weight checks are authenticated; this CPUaudit is not a newGPU rerun.
- Native atomic scatter order may differ within an expert; no crosscall rawsorted-prefix equality required.
- The v2executed source has64actual-layer probes pervariant; separate72adversarial alignment calls belong to unexecutedv3.

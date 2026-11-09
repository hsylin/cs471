The tuning selection is consistent with the frozen rule and all 192 raw records. Independent reconstruction found zero numerical difference from the reported 16 candidate aggregates and 64 block/domain cell ratios. This is an audit by the same team, not external reproduction or a new GPU measurement.

The measured run is allocation **4867433**, host **kh077**, source commit `36d4a5d2bf98151f2f607e469dec61877ad447e5`. The frozen manifest is `8469e235d3254921fe7f5b912a6635489ddd0656d2701690dd1f81c808ae61e8`; its source fingerprint is `3784df43528c4224fcb9005c569a5a8e6ecd47ac9b9d4a07ab27635487ad37e7`. The executed manifest and current pinned source files match.

The independent calculation averages paired candidate/default log runtimes over three technical repetitions within each block/domain cell, then weights the four cells equally. It excludes candidates with any cell ratio above 1.02, chooses the fastest remaining candidate per batch, and retains DEFAULT when it is within 1% of that candidate. It reproduces both choices exactly:

| Batch | Selection | Candidate/default ratio | Observed runtime reduction |
|---|---|---:|---:|
| 128 | M64_N64_K64_W4_S3 | 0.986114517 | 1.3885% |
| 1024 | DEFAULT | 1.000000000 | 0% |

For B128, the selected candidate's four cell ratios are 0.988382467, 0.983384113, 0.989947336, and 0.982763678. All 12 paired technical observations are below one, with a range of 0.981724–0.995734. These are repeated observations of two source-document pairs, not 12 independent workloads. The observed gain is below the eventual 5% target. No candidate aggregate at either batch reaches a 5% reduction. This pilot selection does not establish a confidence interval or demonstrate that a 5% gain is impossible.

The B128 choice preserves M=64 and changes N from 128 to 64. It therefore does not support a claim that reducing the M tile improved padding behavior. The smaller-M candidates are not selected. At B1024, DEFAULT is fastest by aggregate time; TUNED consequently resolves to the identical DEFAULT configuration in later validation/final runs. That batch's kernel-label comparison is an identical-configuration control. The disclosed B128 default-equivalent explicit candidate has ratio 0.999664313, which is consistent with a small timing difference between repetitions of the same configuration.

All declared matrix keys and 192 record IDs are unique and complete. Record IDs, candidate order, resolved configurations, rank agreement, source maps/weights, checkpoint pins, source-document IDs, captured bank hashes, and input selection identities agree. Every candidate occurs three times in each of the eight execution positions over the complete matrix. Each recorded elapsed time equals the maximum of its two rank times; the largest between-rank elapsed difference is 34.882 microseconds. The records report zero migration and no native observation/rebalance steps, consistent with fixed-map tuning.

The audit hashes both actual pilot bank files and all eight saved selection artifacts. It regenerates their 48 input batches from the declared seed and verifies exact indices, domain membership, within-batch sampling without replacement, and reuse of the same inputs across candidates and repetitions. It does not inspect final data or repeat GPU tensor comparisons.

The recorded numerical checks cover 64 candidate/cell combinations, each at the first and last timed input: full-batch comparison to DEFAULT plus 16 FP32 reference tokens per endpoint. These checks are cached across the three repetitions. The maximum reported full-batch DEFAULT difference is **0**. The largest FP32 absolute error is **0.008345604** (B1024); B128's maximum is **0.001824647**, both below the absolute tolerance 0.02; the recorded relative tolerance is 0.03. This verifies the reported summaries and coverage, not numerical correctness at every timed input.

One metadata qualification should accompany interpretation: `weights_source.replay_scope` retains legacy text referring to synthetic activations/routes. The tuning's input metadata, saved selection artifacts, and verified captured banks identify actual teacher-forced causal-prefill inputs. This inherited string does not change the measured inputs, but should not be quoted as the scope of this experiment.

No blocker or basis for retuning was found. The preplanned whole-policy validation and fresh final matrix are still needed: these 32-forward trials exclude compilation, input materialization, conditioning, observation, planner work, and migration, and cannot establish full-serving or complete-policy gains.

Machine-readable checks and per-candidate numerical details are in `work/optimization-tuning-audit.json`; the independent audit implementation is `work/audit_optimization_tuning.py` and imports no measured-package helpers.

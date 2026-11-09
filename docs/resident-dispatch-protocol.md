# Resident dispatch protocol

This is a new, bounded study of dispatch among already resident expert copies. It does not revise the completed TV, migration-payback or kernel-tuning analyses. The scope remains one pinned Qwen3 layer, BF16, two H100s, 128 logical experts, top-k 8, 32 additional resident copies, replicated inputs and the existing output all-reduce. The resident physical-to-logical map and expert weights stay fixed. Logical top-k IDs and combining weights remain unchanged.

The causal question is whether assigning the same logical expert invocations to different resident physical copies can reduce completed replay time enough to justify an online selector. Fewer active copies, padded rows or more even token counts are mechanism measurements. They are not runtime bounds. An optimum for active-copy count is not an optimum for elapsed time.

## Pilot contract

Use the two existing development source-document blocks, their two domains (`domain0`, `domain1`), and batches 128 and 1024: eight fixed input states. Final document blocks remain unavailable for policy selection. Each state contains 16 conditioning input batches followed by 32 measurement input batches prepared from genuine captured hidden states, logical IDs and combining weights. Captured token rows may be reused; this is a constructed replay, not a naturally observed arrival stream.

Cycle the 32 measurement inputs eight times, producing 256 measured forwards and 256 native `SKIP.step` calls per trial. Reset the native counter to zero after conditioning, retain the required observation history, and verify proposals at measured steps 63, 127 and 191. Suppress the terminal proposal at step 255 before planning. No migration is accepted. Native observation and nonterminal planning remain charged. This is a cyclic mechanism replay; extending the timing interval does not create 256 independent observations.

Run the following five arms twice per state, using a frozen initial arm permutation and its reverse for the second repetition. These 80 rows are the entire pilot matrix. Warm and restore comparable state before each arm; retain raw execution order and timestamps.

| Arm | Meaning | Main interpretation |
|---|---|---|
| `native` | Unmodified native dispatch, fixed map, native SKIP accounting | Reference runtime |
| `native_aa` | Identical executable path and configuration with a different experimental label | A/A timing calibration |
| `pre_native` | Precomputed native physical assignments through the same recording adapter used for the two candidate arms | Adapter/precomputation contrast, not A/A |
| `one_copy_home` | Precomputed assignment to one existing home copy per logical expert | Optimistic concentration mechanism |
| `token_balanced` | Precomputed assignment balancing token work among existing copies | Optimistic token-load mechanism |

All arms use the same native expert kernels, logical routes, weights, output collective, recording semantics and fixed resident map. The candidate/native contrast measures optimistic net headroom excluding assignment selection. Candidate/pre_native isolates assignment differences within the shared precomputed route path. Report both, along with pre_native/native; do not attribute an adapter-path saving to better assignment.

Precomputed assignment construction is outside the pilot timer and explicitly excluded. Per-step input handling, any copies/adaptation needed to deliver the prepared inputs or assignments, forward execution, recording and native SKIP observation/planning stay inside it. Initialization, compilation and common bank materialization can be excluded only with the same stated boundary for all arms. Completed runtime is the maximum elapsed time of the two ranks, after completion of required GPU work. Profiler runs are diagnostic and do not replace this primary unprofiled measurement.

Native EPLB reconstructs logical loads from recorded physical loads. Moving an invocation between copies of the same logical expert must therefore preserve the logical window counts, candidate planner maps and SKIP decision trajectory under the fixed-map contract. Validate this directly. A dispatcher changing physical counts does not by itself imply changed logical demand or a different planner trajectory.

## Pilot advance and stop rules

### Single calibration revision

The original clean attempt (allocation4869036, headroom-v2) completed all80 rows
but failed the A/A band in one cell: 0.98832085. It remains a complete failed
calibration attempt, with no selected candidate. A first-trial ordinary-step
stall motivated one setup-hardening revision; its cause is not established.
No rows are removed and this attempt is not pooled with its replacement.

Before the revised timer, record every start/end CUDA event once and synchronize
to initialize their handles. After the16 original conditioning inputs, execute
64 additional forwards/SKIP.step calls with `warming_up=False`, cycling raw input
indices16..47 twice. This must execute one real native planner/no-op rearrangement,
preserve maps/weights and send zero bytes. Reset counters, windows/pass buffers,
history validity, decisions, timing totals and EOF suppression afterward. Run
`gc.collect()` before the common sync/barrier boundary in every arm; leave GC
enabled during measurement. Record excluded setup time and the conditioning
schedule. The80 rows, assignments, inputs, order, gates and256-step timer remain
unchanged. This establishes a revised steady-state contract, not a causal test
of event initialization, planner startup or GC separately.

This is the single allowed calibration revision. If its A/A screen fails,
stop performance selection and do not repeatedly rerun until it passes. Freeze
the revised manifest and execution hash before timing. Confirmation data remain
unused. A passing revision cannot retrospectively validate the original attempt.

These are conservative development screens, not confidence statements from two source blocks. The 1% margins and selector reserve are reviewer choices fixed before pilot timing, not empirical guarantees or literature standards.

For each arm, first average its paired log runtime ratios to native over the two repetitions of each source-block/domain/batch cell and exponentiate. Give the four domain/batch cells equal weight in each block's geometric aggregate. Do not treat ranks, cyclic iterations, sampled tokens or repeated timings as independent source blocks.

1. All 80 expected rows, exact input/map/route invariants, completed accounting and declared numerical checks must pass. Missing, duplicate, unexpected or incorrect rows cannot be silently dropped or filled.
2. Every native_aa/native cell mean and the overall geometric mean must fall in [0.99, 1.01]. Otherwise stop performance selection and investigate measurement conditions. Passing this screen is not a formal equivalence test.
3. Let the noise reserve be the larger of 0.01 and the largest absolute native_aa/native cell deviation from one. Reserve another 0.01 of native runtime for the eventual selector/input overhead. A candidate advances only if **each block's four-cell candidate/native aggregate plus both reserves is at most 0.95**. With a calibrated null this requires at least 7% optimistic runtime reduction.
4. Every individual candidate/native cell plus both reserves must also be at most 1.02. A fast aggregate cannot conceal a regressing domain/batch cell.
5. If neither candidate passes, stop this direction. If one passes, implement only that candidate's online form. If both pass, select the lower worst-block aggregate; break an exact tie in favor of `one_copy_home`, the simpler mechanism. This deterministic choice must be made from the pilot only. Publish both candidates' results.

The reserve is a screening allowance, not proof that the real selector will fit. Before confirmation, a bounded development check must charge actual online selection, physical-ID production, any required copies/permutations, native accounting and completed execution. If the implementation cannot meet the practical margin without excluding real work, stop. Do not rescue it by weakening the margin or using held-out demand to tune it. Failing one assignment heuristic does not prove that every possible resident dispatcher is ineffective.

## Conditional confirmation

Confirmation happens only after the pilot advances one mechanism and its fully charged implementation passes correctness checks. Freeze one online policy, its source and configuration, all input hashes, analysis plan, exclusion rules and timing boundaries before inspecting new final demand or performance.

Use eight fresh source-document blocks disjoint from development and all prior studies, four constructed regimes (`stable`, `transient`, `persistent`, `switching`), batches 128 and 1024, and two arms (`native`, `online`). Preserve the approved replay geometry: 32 conditioning steps and 512 measured steps, with all online work and native SKIP observation/planning charged. The two technical repetitions occur in two distinct allocation windows on the same hardware class. Each allocation contains one matched native/online pair for every block/regime/batch cell. This is **256 runs**, reusing the prior repetition budget rather than adding a large grid.

Counterbalance native/online order locally with a frozen schedule, randomize starting labels, and retain restoration, execution order, allocation IDs and process/device metadata. Report each allocation's paired results separately. Two allocation windows are an environmental consistency check; they do not estimate a broad hardware-day population precisely. A matched B3 diagnostic, if separately authorized later, is contextual and does not replace or expand the one primary contrast.

## One primary endpoint

The primary endpoint is the geometric **online/native ratio of completed replay runtime**, equally weighting the six changing regime/batch cells. For each source block and cell, average the two paired log ratios across allocation repetitions. Average the six cell log ratios within that block, then average over the eight blocks and exponentiate. Bootstrap the eight source blocks jointly, 20,000 draws with seed 10471, preserving all paired cells and repetitions within a resampled block.

Useful gain passes only when the primary aggregate 95% interval's upper endpoint is **≤0.95**. It fails when the lower endpoint is **>0.95** and is otherwise inconclusive. This is a 5% runtime-reduction criterion, not a 5% throughput-ratio criterion.

The stable guard requires both stable batch cells' family-adjusted upper endpoints and the stable aggregate 95% upper endpoint to be **≤1.02**. The two stable cells use percentile tails 0.025/2 and 1−0.025/2. Changing-cell diagnostic intervals use the corresponding family of six; they do not create six new primary endpoints. Overall acceptance requires both the primary useful-gain criterion and the stable guard to pass. If either fails, acceptance fails; otherwise an unresolved component makes acceptance inconclusive.

All intervals describe the selected source corpus under the block-resampling assumption, conditional on these two allocation windows and constructed replays. Technical repeats, ranks and cyclic steps are not independent workload units. No production-serving or hardware-population generalization follows automatically. Report an allocation-specific discrepancy even when the pooled primary estimate passes; investigate it before making a portability claim.

## Correctness and raw evidence

Archive source/configuration and input hashes; exact logical routes/weights; fixed maps and resident slot coverage; assignment legality; physical counters, collapsed logical windows and their expected totals; planner candidates and decision reasons; sender bytes (zero); and the native terminal-proposal suppression. All ranks must use consistent routing and collective order. Compare actual numerical outputs with native routing under the declared BF16/FP32 check contract; explicitly describe sampled coverage rather than claiming every output was exhaustively validated.

The statistical analyzer requires row fields `block_id`, `regime`, `batch_tokens`, `repetition`, `allocation_id`, `arm`, `elapsed_seconds`, `status`, `correctness_pass` and `evidence_eligible`. Status must be `complete`, both flags must be literal true, and elapsed time must be finite and positive. It rejects unexpected and duplicate keys, reports missing/invalid matched groups, checks that final repetitions occupy two distinct allocation windows, and compares supplied common state hashes. These checks do not replace raw routing reconstruction, provenance checks or numerical validation; a true flag is an assertion to audit, not independent proof.

The analysis API is `resident_dispatch.analysis.analyze_pilot(records, **plan)` for the five-arm development screen and `analyze(records, **plan)` for the two-arm matrix. The latter accepts `phase`, `expected_blocks`, `expected_regimes`, `expected_batches`, `expected_repetitions`, optional `allocation_by_repetition`, arm names and frozen analysis constants. No final interval or acceptance is available for fewer than eight blocks. Optional state fields compared when supplied include configuration/input/source-map/source-weight hashes, logical routing and logical planner-trajectory hashes, resolved kernel settings, source prompt IDs, completed steps, policy and timing scope. Supplied restoration/routing/commit/agreement flags must be true, migration counters must be zero, and supplied source/final maps must agree.

For an archived JSON plan containing these keyword arguments:

```bash
python -m resident_dispatch.analysis --plan analysis-plan.json --records records.jsonl.gz --output analysis/summary.json
```

Use `phase: "pilot_screen"` in the JSON plan to select the five-arm pilot analyzer. Plain JSONL and gzip JSONL are both supported. The frozen runner manifest, raw-data audit and this analysis plan must remain linked in the archive.

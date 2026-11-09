# One-copy pilot audit

The audit passes: **96 completed arm records, 16 retained source states, 32 paired three-arm groups, and 4,780 offline checks**, with no invariant or source-pin mismatch. This is a read-only audit of the recorded GPU run and executed source. No GPU measurements, model-weight downloads, or frozen-code changes were performed.

The principal result is a **roughly 6 ms synchronized native placement transition**, not a measurement of network transfer time or link bandwidth. Each charged candidate sends exactly **9,437,184 bytes (9 MiB)**, comprising the selected expert's BF16 gate/up and down tensors. All 16 H512 states, and all 16 H128 states, are slower with charged migration than with KEEP. These are pilot observations from two source-document blocks and one technical repetition; they are not final confidence estimates.

## Observed timing

Positive differences below mean slower. Medians summarize the 16 retained states per horizon.

| Horizon | Native transfer+commit call | CHARGED − PREINSTALLED | CHARGED − KEEP | PREINSTALLED − KEEP |
|---|---:|---:|---:|---:|
| 128 inputs | 5.924 ms | 6.025 ms | 5.844 ms | −0.230 ms |
| 512 inputs | 5.940 ms | 6.154 ms | 5.627 ms | −0.443 ms |

At H512 the charged penalty relative to KEEP ranges from 3.678 to 7.234 ms. The preinstalled candidate is faster than KEEP in 14 of 16 states, but its absolute gains are small: the largest observed gain is 2.380 ms. None offsets the charged transition. The prerecorded screening gate therefore correctly returns zero qualifying states and does not request confirmation repeats.

## Exact timing and completion semantics

The frozen [`_run_arm` implementation](/Users/hsing-yulin/Documents/Codex/2026-11-08/us-2/work/cs471/followup/payback.py) restores the source weights and maps, performs full weight/map validation, installs PREINSTALLED's candidate, and conditions each arm using the same last two observed-prefix inputs **before** timing. It resets payload counters, executes a rank barrier and device synchronization, and starts the primary timer at line 437. CHARGED alone calls `force_migration` inside that timer (lines 439–442). All three arms then replay the same complete future input interval, and `layer.close()` synchronizes device completion before the elapsed timestamp (lines 443–446). Reported elapsed time is the maximum of the two rank durations. Cross-rank gathering, weight/map validation, output-reference checks, and physical-count diagnostics occur after timing.

The nested `migration_seconds` timer wraps the complete [`force_migration` call](/Users/hsing-yulin/Documents/Codex/2026-11-08/us-2/work/cs471/skewbench/native.py) at lines 519–529: candidate-to-CPU-tensor conversion, native rearrangement, native map commit, and explicit device synchronization. The pinned native implementation additionally performs CPU-side transfer-plan construction, old-map device-to-host access, NCCL P2P posting and waits, receive-buffer/workspace copies, and inverse-map/replica-count reconstruction. Sources are the exact executed vLLM commit:

- [Native rearrangement and host-side plan preparation](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/distributed/eplb/rebalance_execute.py#L508).
- [Torch NCCL communicator posting and request waits](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/distributed/eplb/eplb_communicator.py#L99).
- [Physical/logical map and replica-count commit](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/distributed/eplb/eplb_state.py#L1380).

Consequently, saying that the observed 6 ms is **the full native transition overhead for a 9 MiB action** is supported. Saying that 9 MiB takes 6 ms on the network, or that a quantified fraction is CPU overhead rather than bandwidth, is unsupported: the run does not separately time transport, host preparation, device copies, and map commit. A payload-size or component-instrumented benchmark would be required for that attribution. No such additional experiment is needed to report this bounded action diagnostic.

CHARGED−PREINSTALLED is also not a pure transport decomposition. PREINSTALLED conditions the candidate layout before timing, whereas CHARGED conditions the source layout and then transitions. Their difference includes real placement/cache-transition effects. The separately recorded native-call duration avoids conflating all later expert execution with migration, but still includes the native synchronization and commit work listed above.

The common B3 prefix and candidate planner/selection work are excluded from all three arms and logged separately. The measurement is conditional on a proposed action; it is not a complete online-controller latency measurement.

## Placement, payload and correctness evidence

All candidates retain 160 physical slots for 128 logical experts, change exactly one slot, remove only a redundant copy, and introduce an expert absent from the destination rank. Exhaustive offline recomputation of the one-copy selector reproduces every selected candidate and its deterministic tie rule. Candidate and source map hashes, rank-ordered source-weight hashes, and frozen source digests match the records.

Each charged move's actual sender payload equals its independently executed untimed candidate preflight: one sender contributes exactly 9 MiB and the receiving rank contributes zero sender bytes. KEEP and PREINSTALLED send no weight payload during measurement; PREINSTALLED records the same 9 MiB install outside timing. The communicator's recorded message count is one expert-tensor bundle, not a count of packets or necessarily individual NCCL tensor sends.

Every arm attests exact local tensor equality with its checkpoint experts, valid inverse maps/replica counts, and cross-rank agreement on the global placement. All expected committed maps agree with the arm definition. These full weight checks are stronger than the numerical output sample, but are runtime attestations recorded by the frozen harness; this audit did not retrieve model tensors and recompare them independently.

Numerical checks run the actual configured batch shape (128 or 1024 tokens), then compare its first four outputs against an independent FP32 reference at the first and last future inputs. The maximum observed absolute error is 0.001701, with the declared tolerances `atol=0.02`, `rtol=0.03`. This does **not** constitute an independent reference check of every output element at every future input.

Every future diagnostic conserves exactly batch × 8 logical assignments per input, and PREINSTALLED/CHARGED share identical native-mapper physical-count hashes and rank-owned assignment traces. Both prefix opportunities' raw physical-window hashes and physical-to-logical recovery were rechecked. The future intervals are exactly `[160, 288)` and `[160, 672)`; input159 belongs to the source prefix. No future rebalancing is enabled.

All records use one warmed process on allocation4866428, nodekh071. Its same-process, same-process-group transport warmup sends real weights in both directions and rolls back before measurement. Thus the known first-use P2P startup confound is explicitly controlled.

## Proxy and scope limitations

All 16 one-copy actions improve the **current-window equal-share token-count proxy**, by 0.026%–1.570% (median1.341%). However, native-mapper diagnostics show a reduction in the mean future per-input peak rank assignment count in only **9 of 16 states** at either horizon. At H512 this actual future proxy change ranges from a 1.330% worsening to a 1.369% improvement, with median improvement0.058%. This is expected to differ from the selection score because the latter describes a past window and equal sharing, whereas the future diagnostics use exact token-hash assignments on the constructed future workload. Assignment count also does not encode active-expert or kernel-shape costs and is not a latency oracle.

The actual B3 prefix accepted its first scheduled proposal in all eight pilot-02 states and rejected it in all eight pilot-01 states. Therefore half of the source states had an earlier real migration and half retained their initial layout. At the intercepted target opportunity, native B3 would admit only four of the 16 proposals. All states remain included by the predeclared design: this is neither an exclusively adapted-layout study nor an exclusively admission-eligible action sample.

The temporal schedules are constructed from captured teacher-forced causal-prefill tokens. Two independent source-document blocks, a single allocation, fixed two-H100 hardware, one model-derived layer, replicated-input/allreduce dispatch and synchronous native migration bound the claim. Technical arms and horizons do not create additional independent workload samples. The final held-out experiment should preserve this scope and both predeclared candidate families; these pilot timings do not justify retuning the frozen protocol.

## Provenance

- Records SHA256: `ad28918a9c8db59ed88447a8eacf6bc6d3fa77879ba2fe416db6cb46578cee68`.
- Frozen manifest: `96ea1ecb835ba96136242ab422ef67e3e430302510d6e9860e0f202e74fc28a1`.
- Follow-up source fingerprint: `ba8c8ab0dc4519a78564d25f3acb301fab5b6e4e5d30631bed99cd9ba9e6fc61`.
- Legacy source fingerprint: `894bf6474c57655c15d41709ffdb9e84f9e9e16aaece6d7de6a4fc88c94594cb`.
- Runtime native-source attestation: all six declared vLLM source files match commit`db9527a46873454610df6dbedf79a36d6bf1a7f6`.

The machine-readable [audit JSON](/Users/hsing-yulin/Documents/Codex/2026-11-08/us-2/work/payback-pilot-one-copy-audit.json) contains all check totals and per-cell diagnostics. This audit rechecked frozen local source, all JSONL records, and read-only native source from the installed environment. It did not reopen the original captured banks or rerun GPU outputs; those limits should accompany any claim of independent validation.

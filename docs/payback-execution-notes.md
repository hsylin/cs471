The payback follow-up preserves the completed TV study. The legacy `skewbench/`
fingerprint remains `894bf6474c57655c15d41709ffdb9e84f9e9e16aaece6d7de6a4fc88c94594cb`.
New execution and analysis live in `followup/`; new data use separate directories.

The capture at source commit `57a917f62da600812f3a7592f99d40fe2156d782`
completed in Slurm job 4865987 on kh040 (exit 0:0, 2m08s). It captured 40,960
token records from ten disjoint document pairs. Each domain uses the first
2,048 tokenizer tokens as one causal prefix. The exact pinned embedding,
attention, normalization, and router execute; only the experts after routing
are replaced by an observer. Native boundary/router/formula checks and BF16
serialization checks passed for all source documents. This is first-layer
teacher-forced prefill capture, not autoregressive generation or serving.

The initially composed `tapes/` used recurring transients every 128 steps.
Before payback measurements, the design was clarified to one transient, with
`repeat_steps=1024`. `scripts/compose_router_tapes.py` created 80 new immutable
`tapes-v2/` files from the same banks. Initial tapes remain retained and unused.
The bank capture-script hash therefore refers to the original capture, while
the separate composition index records the later script and module hashes.
All measured manifests use the v2 file hashes. Stable schedules retain a
phase metadata value of 120, which has no effect because their domain is
always prose. The other schedules use phase 120, except switching at 112.

The source snapshot is at measured decision 127, after a B3 decision opportunity
at 63. The first opportunity may accept or reject; the descriptive configuration
field `prefix_commit_steps=[63]` does not force a commit. Actual `prefix_decisions`
are authoritative. In both smoke states, B3 rejected steps 63 and 127 because
predicted improvement was below its 2% threshold. Those source maps remained
the initial map. Forced moves at these states are diagnostic interventions,
not moves B3 would have made or evidence that B3 missed a profitable action.

Startup attempts are preserved rather than silently replaced:

- Profile job 4866069 failed before GPU execution because this uv-managed
  environment has no `pip` module. The wrappers now record installed package
  versions through `importlib.metadata`; no dependency was installed or changed.
- Profile 4866151 and smoke 4866207 on kh024 stalled during library imports.
  Worker wait states were `d_alloc_parallel`/`cxiWaitEventWait`, with little CPU
  time. They were cancelled before producing experiment records.
- The same configuration on kh074 completed: profile 4866238 in 8m48s and
  smoke 4866237 in 9m12s, both exit 0:0. A CPU import trace also showed slow
  shared-filesystem library access. This supports transient GPFS import latency;
  it does not prove a Python cache lock, runtime deadlock, or permanent node fault.
  The environment stayed unchanged. Later submissions exclude kh024 operationally.

The smoke used source commit `7dffdab40339636af7022849ba3c6bbb0c732dc0`,
follow-up SHA `ba8c8ab0dc4519a78564d25f3acb301fab5b6e4e5d30631bed99cd9ba9e6fc61`,
and manifest SHA `1ef53556253b2b6eb02ec8f1b7aee1bdfdde76011bdbd418acb7d87270814d5d`.
All 24 arms (eight matched triples) passed. These records have
`evidence_eligible=false` and are not pooled with pilot or final timings.
The independent smoke audit reproduces one-copy selection, native sender payload,
logical-count reconstruction, mapping hashes, and complete future boundaries.
Numerical checking runs the full configured native batch, then compares four
tokens to an FP32 reference at the first and last future inputs. It does not
verify every numerical output in the timed horizon.

The separate fixed-map profile uses one pilot document pair and B2-derived
resident maps. Its five unprofiled trials per cell each time 32 complete forwards,
including mapping, expert kernels, and output all-reduce. Profiler runs are
separate from these trials and from all payback timing. The first profiler
invocation contains substantial cold CPU overhead; nested operator and kernel
rows can duplicate attribution. These records cannot be summed into a percentage
pie or treated as a policy speedup. No kernel configuration was changed based
on this profile, and no hardware-counter roofline analysis was performed.

Pilot manifests were frozen at commit
`fc5897b7b9f9dad1ba6fba6f2097a22895ec2723` before pilot timing. Native screening
job 4866303 starts only after successful smoke completion. The native pilot
manifest SHA is `2f71314b0564ad391d5d884f0c668923275c5887588017e7dd3062b911783fdf`;
the predeclared optional one-copy pilot SHA is
`96ea1ecb835ba96136242ab422ef67e3e430302510d6e9860e0f202e74fc28a1`.
Screening is 96 arms per enabled candidate. Confirmation, if the frozen trigger
passes, adds 72 arms across all twelve changing H512 states. Final performance
has not yet been measured at this stage of the execution record.

Native pilot 4866303 completed on kh071 in 11m48s (exit 0:0): 96 arms,
32 valid triples, no missing/invalid pairs. All sixteen H512 charged paybacks
were negative, and no changing state qualified for the frozen 1 ms screen.
Native confirmation was therefore not run.

The predeclared one-copy pilot 4866428 completed on kh071 in 3m50s
(exit 0:0): 96 arms, 32 valid triples, no missing/invalid pairs. Its sixteen
H512 charged paybacks were also all negative. Neither candidate qualified
for confirmation or new policy development. These single-repetition pilot
observations are feasibility evidence, not final confidence intervals.

Before any final timing, the unchanged execution/analysis fingerprint and
64 final replay hashes were frozen in `configs/frozen/payback-final-v1.json`
(SHA ee0236aa6d7f466e8bbfb8452b9983df3790d7447fba742d93b82151ab2ac706).
The final matrix retains both families, eight held-out document pairs, two
technical repetitions and all three arms: 1,536 arm records. No online
controller is developed, following the pilot-only stopping rule.

The final execution log confirms that vLLM uses its default fused-MoE kernel
configuration because the shape-specific `E=80,N=768,device_name=NVIDIA_H100.json`
configuration is absent. This setting is common to every arm; it was not changed
during the pilot or final campaign. Results characterize this pinned default
execution path, not an independently tuned optimum. The separate profiler does
not establish a roofline bound or prove that kernel tuning cannot change payback.

Final job 4866456 completed on kh071 in 29m22s, exit 0:0. All 1,536 arms
form 512 valid triples with no missing or invalid records. The independent
statistics implementation matched all 96 cell and 24 aggregate estimates
(maximum absolute arithmetic difference 2.22e-16). The independent raw-input
audit verified all eight final banks, 64 tapes, 64 single-copy searches and
192 full future physical-counter arrays. Maximum sampled numerical absolute
error was 0.004194676876068115, within the frozen reference tolerance.

The archived-data reproduction command completed from a fresh local output
directory, including archive checksums, frozen analyses, independent interval
reconstruction, raw-dispatch audit and mechanism summaries. The primary plot
was visually inspected. Runtime/source fingerprints stayed unchanged. The
standalone audit and reporting helpers live outside the measured modules.
The report was reviewed against raw results and the literature skill style
pass ran once with no issues; precision edits preserve the measured scope.

The frozen decision remains no new controller. Primary changing aggregate
CHARGED/KEEP is 1.0299333341 for native and 1.0244932189 for one-copy. Small
resident gains remain, and three native H512 technical pairs have positive
payback. These final observations are retained without reopening pilot-only
selection or extending the horizon/candidate set.

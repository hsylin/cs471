# Execution record

## Integration checks

The first environment check failed before execution because a newer PyPI
`torchcodec` wheel required CUDA 13. The environment was repaired using the
official `torchcodec==0.16.0+cu129` wheel, consistent with vLLM and PyTorch's
CUDA 12.9 builds. The vLLM version and source commit were unchanged.

Kuma jobs 4862064 (B4 smoke) and 4862098 (native B2 smoke) completed with the
actual pretrained layer. They validated native expert transfer, normal map
commit, checkpoint weight identity, and numerical output tolerances. They are
explicitly excluded from confirmatory performance evidence. The B2 run also
validated exact current observation windows and terminal-proposal suppression.
The forced unchanged-map helper check is a transport check, not by itself proof
of the admission gate's recovery protocol.

## Histogram implementation correction before final measurements

Pilot job 4862109 was stopped after 96 completed runs. Its original frozen
manifests and raw records are retained. It is an incomplete diagnostic pilot;
it does not provide the final cooldown selection or any confirmatory verdict.

The initial B4 collector used `torch.bincount` with a known minimum histogram
length. PyTorch 2.13's CUDA implementation nevertheless computes input bounds
and retrieves them on the CPU, synchronizing the device in the online loop:
[SummaryOps.cu](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/cuda/SummaryOps.cu#L273-L280).
The diagnostic pilot showed a substantial additional elapsed cost even where
B3 and B4 made the same admission decisions. This led to a source inspection,
not a change in scientific thresholds or trace selection.

The corrected collector clears a preallocated fixed-size integer histogram and
uses integer `scatter_add_` with preallocated ones. It retains continuous
observation, the same consecutive windows, and the same admission formula.
Exact observation-vector checks against the executed trace remain mandatory.
The corrected harness receives a new source hash and a complete new pilot;
no held-out final measurements preceded this correction.

The initial pilot also verified actual scheduled B4 rejection followed by a
later accepted migration on burst and persistent-change cases. Post-run checks
confirmed rank agreement, expected committed maps, exact current and previous
logical windows, and physical dispatch counts interpreted under the old map.

## Transport warm-start correction before final measurements

The second pilot completed all 308 planned runs and eight pilot branch pairs.
The correctness and metadata checks passed, but the first real weight transfer
in each fresh process paid lazy transport initialization inside its timer.
For example, the first B3 run in the 64-step cooldown process took 0.4050 s,
versus 0.2684 s for B5 despite identical decisions and payloads. In the 256-step
process, the first B3 run took 1.2973 s; its first proposal step alone took
1.0428 s. The first branch allow case showed the same startup effect.

The automatically selected cooldown from that pilot is retained as a diagnostic
record and superseded. It is not the final selection. No slow timing observation
is individually deleted or replaced to improve a result.

The warm-start contract now explicitly initializes native weight transport in
the actual measurement process before any run: a throwaway layer transfers a
full cross-rank map, restores the original map, and verifies the restored expert
weights. It uses the same native EPLB device process group as subsequent runs.
Both moves reach normal commit. This startup work is untimed, just like model
loading and compilation; every migration admitted during the measured horizon
remains fully charged. The corrected source receives a new hash and complete
new pilot, with scientific thresholds, traces, and selection rule unchanged.

## Final bounded collection correction

The valid warm-start pilot selected cooldown 256 by its predeclared 1% tie rule;
all three candidates met the pilot stationary constraint. The comparator
execution paths, thresholds, and selection remain fixed.

A final implementation audit identified unused B4 work: with R=64 and W=16,
only phases 32–47 supply the previous window, while native EPLB supplies phases
48–63 for the current window. Continuous collection computed four times as
many histograms as the gate reads. The final collector uses that fixed previous
window schedule, retains continuous warmup, and advances logical timestamps
on every input step. Exact timestamp and vector checks guard against stale
history. This optimization is valid for this synchronous, fixed-cadence harness;
it is not claimed to support arbitrary asynchronous or dummy-step schedules.

The unused continuous-collector final manifest is archived as
`final-unexecuted-continuous.json`; no held-out run used it. Before freezing the
final execution hash, a 28-run B4 replay compares the same pilot tapes with the
valid warm-start pilot. Counts, TV, admission, candidate/committed maps, and
payload accounting must match. All 28 trajectories matched exactly. The report
is `results/history-validation/comparison.json`; elapsed timings are deliberately
not compared. This validation is not final performance evidence. No further
performance optimization follows it.

The pilot branch execution used seed 11, but its original analysis selector
accidentally named final seed 1001. The frozen files and original report remain
unchanged. A supplemental execution-based audit validates all eight declared
pilot states and explicitly records that mismatch. The final execution and
analysis selectors both name seed 1001. The final eight pairs also use a
predeclared balanced order: four allow-first and four skip-first, shuffled with
order seed 471. The original seed-only shuffle gave all pilot strata the same
order; this additional correction removes that diagnostic order confound.

The final numerical checker additionally replays the complete configured token
batch outside timing, then compares its first four outputs with the independent
FP32 reference. Earlier checks reduced the inputs to four tokens before the
native forward, which used a different kernel batch shape. The final tolerance
remains atol 0.02 and rtol 0.03. This last change affects only untimed validation;
the measured paths validated by the 28-trajectory comparison are unchanged.

## Reporting limits

The native fused MoE path uses its fallback kernel configuration for this
E=80, N=768, H100 local shape. No method-specific kernel tuning is performed.
This is a replicated-input distributed layer benchmark with an NCCL output
all-reduce, not a DeepEP all-to-all or full-model serving measurement.

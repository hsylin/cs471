# Native kernel optimization artifacts

The complete held-out kernel comparison fails the predeclared 5% useful-gain
criterion and passes the stable regression guard. TUNED+B3 / DEFAULT+B3 is
0.995753938 (95% block interval 0.994511836–0.997027789). The small estimate also
needs the identical-configuration control qualification in the
[evaluation](../../docs/optimization-evaluation.md).

All stages retain the two-H100, pretrained Qwen layer-0 BF16, fixed-32-replica,
synchronous native execution path. DEFAULT/TUNED changes only a kernel
configuration; batch 1024 TUNED is deliberately identical to DEFAULT.

| Artifact | Contents |
|---|---|
| `capture/` | 10 banks, 80 exact replay tapes, 40,960 captured token records and capture provenance |
| `tuning/` | 192 complete fixed-map trials, eight saved input-index files, exact pilot selection |
| `validation/` | 64 complete policy runs and 16 separately recorded rank-level GPU profiles |
| `validation-analysis/` | Pilot report; two source blocks, no final confidence claim |
| `final/` | 512 complete runs, 128 matched four-arm groups, eight source-document blocks |
| `analysis/` | Frozen final summary, 128 paired rows, 48 cell rows, 18 aggregate rows, PNG/PDF figure |
| `audits/` | Separate statistics, raw routing, input, kernel-profile and clean reproduction checks |
| `prior-diagnostics/` | Original raw-kernel reanalysis that motivated the bounded intervention |
| `provenance/` | Scheduler accounting and complete job logs, including two never-started cancelled requests |

Raw timing files use gzip. Decompression recovers the original JSONL bytes;
the pilot selection pins those original bytes, not the compressed container.
Capture and tape files are exact NPZ artifacts, readable with `allow_pickle=False`.
Metadata retains original execution paths; portable auditors resolve archived
files through the directories supplied on their command lines.

Final manifest: `configs/optimization-final-frozen-v1.json`, canonical SHA256
`2cec4996f000f5f49dae39f3d38059513c2040844259207a0e3d31224d4060ea`.
Measured optimization source fingerprint:
`3784df43528c4224fcb9005c569a5a8e6ecd47ac9b9d4a07ab27635487ad37e7`.
The final GPU job executed source commit
`789391f7a860152b815fbaa44904d75e166e31b7` before results were inspected.
Later commits add records, audits and documentation without changing that
measured implementation. Capture, tuning, validation and final job IDs are
4867142, 4867433, 4867609 and 4867908; all completed with exit code zero.

The main timer covers 512 completed policy steps with observation, planning,
transfers, commits and synchronization. Initialization, JIT, input materialization,
common observation warmup and output audits are excluded and separately logged.
`step_seconds_by_rank` includes shared CUDA-event instrumentation and is not an
additive kernel breakdown. CPU planner/controller subregions can overlap GPU work.

Logical routes, raw native windows, physical assignments, mapping/weight checks,
transfer payloads and normal synchronous completion are retained. The numerical
contract checks two input endpoints before and after timing, with full-batch
DEFAULT comparisons plus 16 FP32 reference samples per endpoint. It does not
validate every timed output. A legacy weight-metadata string mentions synthetic
inputs; the actual capture/tape hashes and array audits establish that these
runs use the declared authentic captured inputs.

The [reproduction guide](../../docs/optimization-reproduction.md) gives commands
for every check. Clean-checkout CPU reproduction from commit
`4e1b1de4d31ecc0c0476534a4d42e92f37856ac8` recovers the final summary byte for byte;
independent numerical differences are at most 2.22e-16. This is reproducible
analysis by the same team, not an external GPU replication. The source corpus
is finite and hand-selected, and the final timing allocation is one hardware-day
sample. The result is not a full-model serving benchmark.

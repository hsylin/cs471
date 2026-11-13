# CS471 cost-boundary experiments, 2026-11-13

[Read the Chinese report](report.html). The selected unchanged-map commit elision
improved mean response versus original B4 by 0.903% on LongBench and 2.207% on
NuminaMath. Neither reached the prespecified 5% useful-gain criterion. It remained
4.597% and 5.608% slower than OFF, exceeding the 2% regression guard on both
datasets. All six matched cells completed: 3,072 successful requests, no failures.
This is one ordered group, descriptive only; realized routing decisions differ.

The exact unchanged inverse-map commit elision preserves the native planner,
worker, transfers, readiness, pending reset and consumed-event retirement.
Correctness admission preceded selection and protocol freeze. Actual elisions
were observed on all eight ranks; operation counts do not imply saved time.
Detailed exposed-critical-path attribution remains unresolved. The frozen
transition experiment did not establish causal payback or useful headroom for
an additional runtime-aware admission policy, so that conditional stage was not
implemented. No final-result tuning or fastest-attempt selection was performed.

## Public artifacts

- `final-summary.json` and CSV: all six aggregate rows, contrasts and completion evidence.
- `diagnostic-summary.json`: all 16 cost cells, six transition cells, sustained shape distributions and retained failure descriptions.
- `final-protocol.json`: protocol frozen before any final measurements.
- `completion.json` and `model-review.json`: reconstruction hashes and review outcome.
- `source-manifest.json` and `sources/`: exact source subsets for nine implementations, deduplicated against the final snapshot.
- `build_report.py`: deterministic HTML renderer using only the two aggregate JSON files.
- `publication-manifest.json`: hashes of the reviewed publication files.

Raw request records, prompts, generated outputs, token arrays, logs and profiles
remain private. The full private reconstruction checked input/calendar/target
identity, source bytes, native runtime and checkpoint, eight GPU identities,
independent raw mean/p95 response and TTFT, and 404 stable final-record files.
The public verifier cannot repeat those private-data checks; it verifies only
the published aggregates, arithmetic and source artifacts.

## Rebuild the public report

Use Python 3 without `-O` (the verifier uses assertions):

```bash
python verify_publication.py
python build_report.py
```

The output HTML contains no JavaScript, external assets, prompts or raw outputs.
All timing is from the unprofiled frozen run; subsequent reporting changes do
not modify the measured source or protocol. The NuminaMath count of 26 targets
at 8192 tokens is inherited from the designated natural-OFF length reference;
it is not new natural-EOS truncation evidence for these matched arms.

## Inspect exact source subsets

```bash
python verify_publication.py --materialize final --destination /tmp/cs471-final-source
cd /tmp/cs471-final-source
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m unittest discover -s tests -p 'test_boundary_final*.py'
```

`final` stores the common source files; other labels store only files with
different hashes. The manifest lists each snapshot's complete exported subset
and removals. Original Git revision IDs are provenance identifiers from private
execution checkouts, not promises that those commits are fetchable from this
public repository. Exports include `full_serving` Python/shell/bootstrap metadata,
focused boundary tests and `scripts/cluster-env.sh`; they are not full Git archives.
The rest of the repository supplies preexisting data/native manifests and dependencies.
Full GPU reproduction additionally requires the pinned vLLM/CUDA environment,
checkpoint, separately authorized private inputs, evidence receipts and fresh
allocation/output paths. The fixed historical launcher must not be rerun against
the original result directories. Public tests do not reproduce GPU correctness
or raw-request audits.

The final source revision is `e72cebb24b62b6eaaa3d929809bbfb59bdf3d697`;
protocol SHA256 is `f80a889f05c7d4861c90e86ca490cba180d09af8edc036882cdaca67fdf0da58`.

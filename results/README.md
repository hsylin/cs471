# SkewBench measured evidence

The final study completed with no missing or invalid runs. Its performance
criterion failed: the two-window TV gate did not establish a 5% improvement over
improvement-only admission or the frozen cooldown. This is distinct from the
successful execution and correctness checks.

| Directory | Measured evidence | Role |
|---|---|---|
| `final/` | 1,400 policy runs, eight same-state allow/skip pairs, preflight | Held-out study |
| `pilot-v3/` | 308 policy runs, eight pilot branch pairs, preflight | Cooldown selection and implementation calibration |
| `history-validation/` | 28 same-tape B4 runs and preflight | Exact semantic equivalence of optimized history collection |
| `provenance/` | Environment lock and wheel/package metadata | Dependency reconstruction |

Raw JSONL files are compressed with gzip. Final summaries, CSV tables, and PNG/PDF
figures are in `final/analysis/` and `final/diagnostics/`. The independent audit
programs and their outputs are in `scripts/` and `final/audits/`. See the root
README for a tested CPU-only command that regenerates the reports. This
recomputes saved observations; independent experimental replication needs a new
GPU run.

`final/provenance/execution.json` records the completed Slurm job, original frozen
commit, harness and manifest hashes, and the scope of the experiment. Hardware
logs, the complete job log, model export metadata, and the raw-tape inventory
are in the same directory. The inventory covers 141 files totaling
2,039,032,324 bytes, all verified against the original archives. Exact logical
IDs, combining weights, and hidden inputs remain on the owner's server at:

```text
/work/cs-471/skewbench-hlin/results/final/traces
```

These large NPZ tapes and pretrained checkpoint tensors are not included in Git
or the compact source/results archive. The pinned generator, model extraction
script, revisions, hashes, and environment lock support rerunning them. The
raw-array branch correspondence auditor requires the original tapes (or a copy
with the recorded file hashes); it accepts `--trace-dir` to relocate them.

The valid pilot uses the older measured harness at source checkout `de304a6`.
`pilot-v3/source-de304a6.tar.gz` contains that source snapshot for replication
without relying on Git history. The original pilot branch-analysis selector
mistakenly named seed 1001 while execution used seed 11; the original empty
analysis is retained in `pilot-v3/original-branch-audit/`. The supplemental
`pilot-v3/audits/execution-branch-audit.json` validates all eight executed pilot
states. Final execution and analysis both use seed 1001 and agree exactly.

Earlier diagnostic pilots, including the cold transport-start attempt, remain
on the server in the separately named `pilot` and `pilot-v2` directories. They
were superseded before final measurements and are not pooled into final
estimates. See `docs/execution-notes.md` for the corrective history.

From the repository root, verify the archived result files with:

```bash
shasum -a 256 -c results/SHA256SUMS
```

The confidence intervals use ten independent seed blocks, not 1,400 independent
workloads. Branch paybacks are eight descriptive single-pair measurements at a
fixed 128-step horizon. No external human replication or full-model serving
measurement has been claimed.

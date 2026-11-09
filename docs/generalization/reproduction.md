# Reproduce the new-content / external-arrival study

This guide distinguishes reconstruction of archived evidence from a new GPU run. The execution source frozen for protocol v3 is commit `9263d53`; later artifact/guide commits may contain the same execution files. The operative protocols are v1 **as amended by v2 and v3**. The experiment has three arms (`native`, `shortcut`, `combined`), B128, a fixed resident map, two GPUs, and 256 completed layer jobs per trial. It does not measure complete generated responses or deploy a production request server.

The archived main result is complete: 72 pilot trials, 216 final trials, and 24 eligible diagnostic-v2 trials. The first 24 diagnostic trials remain ineligible and retained, for 336 total treatment/diagnostic executions. Clean-archive main and diagnostic-v2 verification records are in `results/generalization/clean-reproduction/`; these are same-team CPU reproductions, not new GPU measurements.

Run commands from the repository root, with the complete source-and-results archive. A source tarball has no Git history: extract it into a fresh directory and enter the repository directory. `git worktree` and archived `git show` checks below require a Git clone. Do not replace frozen hashes to make another implementation look identical.

## Identity, environments and retained attempts

The runtime is pinned vLLM `db9527a46873454610df6dbedf79a36d6bf1a7f6` (`0.31.0+cu129`), synchronous EPLB, replicated inputs and output allreduce. The model-derived layer is Qwen/Qwen3-30B-A3B revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, layer0, BF16, hidden2048/intermediate768,128 logical experts, top8. The physical map is `[0..127,0..31]`:160 slots,80 per rank,32 extra copies. Placement, logical routes, combining weights and expert parameters stay fixed across treatments.

The frozen physical allocation is Slurm `4872248`, node `kh014`, with GPUs `GPU-204281f9-9c87-3c25-adfb-9edd6c972ec3` and `GPU-7d5c9009-305b-9341-957f-1deb96611ed5`, two H10094GB devices connected by NV6. The wrappers and plans deliberately reject another allocation/pair. After this reservation ends, the archived GPU command is **not** an immediately portable rerun: declare and freeze a hardware-only reproduction amendment, retain the original plans, and adapt the explicit hardware/allocation checks before measurement. Do not silently edit old evidence.

CPU preparation/audit used Python3.13.7, NumPy2.5.4, tokenizers0.23.3, PyArrow26.0.1 and pytest9.2.0. The GPU environment is separate: Python3.12.14, PyTorch2.13.0+cu129, vLLM0.31.0+cu129, Triton3.7.1, Transformers5.17.0 and tokenizers0.23.3. Use the complete executed package lock at `results/generalization/generalization-pilot-calibration-v2/provenance/environment-lock.txt`, not only this abbreviated list. Driver560.35.03 reports CUDA12.6; the PyTorch build uses CUDA12.9. Preserve both identities and topology.

```bash
python3.13 -m venv .venv-generalization
.venv-generalization/bin/python -m pip install \
  'numpy==2.5.4' 'tokenizers==0.23.3' 'pyarrow==26.0.1' 'pytest==9.2.0'
export GEN_REPRO="$PWD/work/generalization-reproduction-new"
mkdir -p "$GEN_REPRO"
.venv-generalization/bin/python -m pytest -q tests/test_generalization*.py tests/test_external_arrivals.py
```

Run from the root rather than relying on the legacy package installer's package discovery. Do not use Python `-O`: some independent checks use assertions. Choose fresh audit and measurement destinations.

The attempt ledger is `results/generalization/capture-attempts/ledger.json`. It preserves the capture-v1 `digest` shadowing failure before capture; completed content-v1 capture-v2 rejected after the prefix-leakage audit; successful content-v2 capture-pilot-v3 and capture-final-v1; superseded native calibration-v1 and unexecuted treatment pilot-v1; and the pilot-v2 shell-launch failure before `torchrun`. The latter failed because `SLURM_SUBMIT_DIR` was unset. The first diagnostic launch also failed before its shell or `torchrun` because the wrapper lacked an executable bit. Commit `2144233` changed only that mode; the wrapper's SHA256 stayed `50f4332cc1ea7560ec9c8faa9d93d57f2ffd315b57138697687f0e3368e165d4`. Neither launch failure was a measured trial. Commands below invoke wrappers through `bash` to avoid reliance on archive-preserved executable bits. Rejected content-v1 capture banks are retained separately outside the published repository; their capture index, source commit and rejection record remain published. None are selected as valid performance inputs.

To regenerate the archived CPU analyses and independent audits together, use the authenticated bundle. Its explicit study list defines the evidence included in that reproduction:

```bash
.venv-generalization/bin/python scripts/reproduce_generalization_cpu.py \
  --repo-root . --bundle results/generalization/cpu-reproduction-inputs.json \
  --output "$GEN_REPRO/archived-evidence"
```

This recreates the summaries and audits in a fresh directory, compares their numerical fields with the archives, and checks the pilot allocation against the final plan. It does not rerun GPUs or demonstrate external-laboratory replication.

## Public inputs, rights statements and omitted material

`results/generalization/content-v1/download-lock.json` is still the raw-download lock used for content-v2; the directory name does not make the rejected content selection eligible. It pins URLs, revisions, byte sizes and full SHA256 values. The downloader verifies existing files and refuses mismatches. Archived metadata entries use repository-relative `local_source` paths, so run it from the repository root.

| Material | Pinned provider statement and scope |
|---|---|
| Filtered WildChat | [Pinned README](https://huggingface.co/datasets/allenai/WildChat/blob/f66566ceaaeb619dd98ffb0f3bf3ce1f86775ac4/README.md) and archived `licenses/wildchat-license.md`: ODC-BY database license. The license itself distinguishes database rights from rights in individual contents; this is not blanket clearance for every conversation. |
| MBPP task descriptions | [Pinned Google Research README](https://raw.githubusercontent.com/google-research/google-research/0eebfd403ba438bfd14cb70b9b24a7745c997083/README.md): datasets CC-BY4.0; repository source code Apache2.0. These are different statements. |
| GSM8K questions | [Pinned OpenAI release license](https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/LICENSE): MIT. Retain its notice. |
| Azure2024 arrival assets | [Pinned dataset README](https://raw.githubusercontent.com/Azure/AzurePublicDataset/215becdacba1ce682c7368642ce97d5a332de7a6/AzureLLMInferenceDataset2024.md) and archived repository license: CC-BY4.0; attribution requests DynamoLLM, HPCA2025. |
| Qwen checkpoint/tokenizer | [License at the exact model revision](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/LICENSE): Apache2.0. Model weights are fetched separately and are not stored in Git. |

The locked attribution record identifies WildChat/Zhao et al., MBPP/Austin et al., GSM8K/Cobbe et al. and Azure/DynamoLLM/Stojkovic et al. Keep those notices with reused artifacts. This repository currently has no top-level project license; public upstream artifacts do not make this private project's code automatically open source. These are recorded provider statements, not a new license grant.

Raw source downloads and private prepared `sources.json` are omitted from the repository and can be regenerated below. Public source manifests omit prompt text and token lists. **Capture NPZs contain token IDs, however, so their selected prompt prefixes are recoverable.** Do not describe the whole capture archive as hash-only or anonymized. Filtering English/non-toxic WildChat records is not an independent privacy audit. NPZs also retain hidden-state bits, router IDs, combining weights and provenance. Preserve the applicable notices and treat any redistribution decision separately from reproducibility.

## Download and reproduce source selection

```bash
.venv-generalization/bin/python scripts/fetch_generalization_sources.py \
  --lock results/generalization/content-v1/download-lock.json \
  --output "$GEN_REPRO/downloads"

cp results/generalization/arrivals-v1/source-documentation/azure-week-download-manifest.json \
   results/generalization/arrivals-v1/source-documentation/azure-2024-release.json \
   "$GEN_REPRO/downloads/"

.venv-generalization/bin/python scripts/prepare_generalization_data.py \
  --downloads "$GEN_REPRO/downloads" \
  --output "$GEN_REPRO/private" \
  --public-output "$GEN_REPRO/public-content"

.venv-generalization/bin/python scripts/verify_generalization_prefixes.py \
  --sources "$GEN_REPRO/private/sources.json" \
  --public-manifest results/generalization/content-v2/source-manifest.json \
  --tokenizer "$GEN_REPRO/downloads/qwen-tokenizer.json" \
  --output "$GEN_REPRO/prefix-audit.json"
```

The two copied files close a packaging omission: the downloader lock does not itself fetch the local Azure asset inventory or the release-API snapshot required by arrival preparation. `source-documentation/inventory.json` records their original byte hashes; README/license copies are retained alongside them. A freshly fetched GitHub release-API response need not be byte-identical to the archived snapshot. Use the original metadata for byte-for-byte reconstruction; do not silently substitute changed metadata.

The accepted selection has content SHA256 `a0167d15f1f4214b438c584a2594c0676da4a3e4c88071f2d1d068f289196dea`:30 groups,64 documents each,1920 documents total. Full-text and decoded first512-token exact/near filtering occur **before** the pilot/final split. Content-v1's three offending prefix pairs remain in its leakage audit. The accepted independent audit reports zero qualifying cross-group prefix pairs. No answer, reference solution, test, assistant turn or chat template is included.

Optionally reproduce the separate check against all earlier captured inputs:

```bash
.venv-generalization/bin/python scripts/verify_generalization_previous_inputs.py \
  --sources "$GEN_REPRO/private/sources.json" \
  --public-manifest results/generalization/content-v2/source-manifest.json \
  --tokenizer "$GEN_REPRO/downloads/qwen-tokenizer.json" \
  --previous-capture results/payback/inputs results/optimization/capture results/resident/capture-v1 \
  --output "$GEN_REPRO/previous-input-overlap.json"
```

This checks1920 new prefixes against56 prior captured documents from28 banks, using both their first512 and full captured sequences. The archived audit found no exact/near pair under the same normalized5-shingle Jaccard≥0.8 rule. This does not exclude shared short fragments, semantic overlap, unrecorded material, or checkpoint pretraining. Older capture archives are needed to rerun this optional check.

Rebuild and independently audit the external source selection:

```bash
.venv-generalization/bin/python scripts/prepare_external_arrivals.py \
  --downloads "$GEN_REPRO/downloads" --output "$GEN_REPRO/arrivals"

.venv-generalization/bin/python scripts/verify_external_arrival_sources.py \
  --manifest "$GEN_REPRO/arrivals/arrival-manifest.json" \
  --candidate-root "$GEN_REPRO/arrivals" --downloads "$GEN_REPRO/downloads" \
  --output "$GEN_REPRO/arrival-source-audit.json"
```

The assets contain code May10–16 and conversation May12–18,2024. Pilot uses the first two dates **within each service**; final uses the remaining five. The seven-date final union includes overlapping service dates. This is not a global future-date split. The selector streams all44107694 rows without sorting, retains timestamp ties, audits171620 disjoint257-row candidates, and publishes12 pilot plus80 final candidate blocks. Each candidate includes256 release rows and a separate following boundary. Original request lengths remain metadata; they do not determine execution size.

## Capture and checkpoint reconstruction

For CPU reconstruction of existing records, use the **archived exact captures**, not recaptured approximations:

```text
results/generalization/capture-pilot-v3/  #6 banks,384 documents,26095 captured records
results/generalization/capture-final-v1/ #24 banks,1536 documents,109869 captured records
```

Their indexes pin each bank's SHA256. Both successful captures used source commit `66cc35b`; each NPZ pins the capture script, Transformers model/mask sources, checkpoint revision/tensors and source-document metadata. The accepted capture-script SHA256 is `b2e3d8a8c0725c8f24ab318370a2fac8ab3384fdfdfa0818eb739dda223c020a`. Check all bank bytes before using them:

```bash
.venv-generalization/bin/python - <<'PY'
import hashlib,json
from pathlib import Path
for name,expected in [('capture-pilot-v3',6),('capture-final-v1',24)]:
    root=Path('results/generalization')/name
    entries=json.loads((root/'capture-index.json').read_text())
    assert len(entries)==expected and len({x['block_id'] for x in entries})==expected
    for entry in entries:
        path=root/'banks'/(entry['block_id']+'.npz')
        assert entry['correctness_pass'] is True
        assert hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==entry['capture_sha256']
    print(name, 'all bank hashes match', expected)
PY
```

For an authorized GPU reconstruction, first reproduce the recorded environment and layer weights; see `docs/resident-dispatch-reproduction.md` for the official cu129 wheel/bootstrap details. With `SKB_ROOT` set to the intended project root:

```bash
source scripts/cluster-env.sh
python scripts/prepare_layer.py --layer 0 \
  --output "$SKB_ROOT/models/qwen3-layer0" --cache-dir "$SKB_ROOT/cache/huggingface"
```

The exported expert-weight file must hash to `80046eef8f68a7afd8d165e582190900854faa44076bd22c6a869730e26d4096`. Capture additionally needs the checkpoint embedding, attention, normalization and router tensors, all from `model-00001-of-00016.safetensors` at the pinned revision. `capture_generalization_inputs.py` obtains the required checkpoint/config/tokenizer files when its cache is not restricted to local files. It checks Transformers5.17.0 and exact model/masking source hashes. `--local-files-only` requires a previously populated cache; it is not an instruction to skip missing files.

The exact capture wrapper for the current reservation is:

```bash
export SLURM_SUBMIT_DIR="$PWD"
export GEN_SOURCE_JSON=/absolute/path/to/regenerated/private/sources.json
srun --jobid=4872248 --exact --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=2 \
  bash scripts/run-generalization-capture-kuma.sh \
  "$GEN_SOURCE_JSON" "$SKB_ROOT/results/generalization-capture-rerun-pilot-new" pilot
srun --jobid=4872248 --exact --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=2 \
  bash scripts/run-generalization-capture-kuma.sh \
  "$GEN_SOURCE_JSON" "$SKB_ROOT/results/generalization-capture-rerun-final-new" final
```

Run GPU steps sequentially, without a concurrent profiler or monitor workload. The wrapper uses only `cuda:0` for capture, checks both allocated devices, and requests no treatment timings. It needs `SOURCE_COMMIT` recording the actual deployed source identity. In a Git clone use `git rev-parse HEAD > SOURCE_COMMIT`; for a tarball record its verified archive identity and explicitly note the absence of Git metadata. Do not use Git commands in a source-only tarball.

Recapture is a new artifact: verify source/route/numerical attestations, then freeze its actual hashes in **new** plans. Do not overwrite archived captures or force new file hashes into an old manifest. Exported expert weights and raw prompt preparation remain outside Git.

## Native calibration, pilot planning and final execution

The following commands generate fresh output paths under the current frozen hardware/source contract. Set `GEN_CAPTURE_PILOT` and `GEN_CAPTURE_FINAL` to the chosen exact capture roots on the execution host, and `GEN_GPU_RUN` to a fresh result parent. A plan freezer runs on CPU; each serving wrapper runs under the held allocation. Both require the unchanged source checkout.

```bash
export GEN_CAPTURE_PILOT="$PWD/results/generalization/capture-pilot-v3"
export GEN_CAPTURE_FINAL="$PWD/results/generalization/capture-final-v1"
export GEN_GPU_RUN="$SKB_ROOT/results/generalization-reproduction-new"
mkdir -p "$GEN_GPU_RUN"
export SLURM_SUBMIT_DIR="$PWD"

python scripts/freeze_generalization.py --phase calibration --split pilot \
  --capture-root "$GEN_CAPTURE_PILOT" --output "$GEN_GPU_RUN/pilot-calibration-plan.json"
srun --jobid=4872248 --exact --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=2 \
  bash scripts/run-generalization-kuma.sh "$GEN_GPU_RUN/pilot-calibration-plan.json" \
  "$GEN_CAPTURE_PILOT" "$GEN_GPU_RUN/pilot-calibration"
python -m generalization.analysis --plan "$GEN_GPU_RUN/pilot-calibration-plan.json" \
  --records "$GEN_GPU_RUN/pilot-calibration/records.jsonl" \
  --output "$GEN_GPU_RUN/pilot-calibration/summary.json"

python scripts/freeze_generalization.py --phase pilot --split pilot \
  --capture-root "$GEN_CAPTURE_PILOT" \
  --calibration-plan "$GEN_GPU_RUN/pilot-calibration-plan.json" \
  --calibration-records "$GEN_GPU_RUN/pilot-calibration/records.jsonl" \
  --output "$GEN_GPU_RUN/pilot-plan.json"
srun --jobid=4872248 --exact --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=2 \
  bash scripts/run-generalization-kuma.sh "$GEN_GPU_RUN/pilot-plan.json" \
  "$GEN_CAPTURE_PILOT" "$GEN_GPU_RUN/pilot"
python -m generalization.analysis --plan "$GEN_GPU_RUN/pilot-plan.json" \
  --records "$GEN_GPU_RUN/pilot/records.jsonl" --output "$GEN_GPU_RUN/pilot/summary.json" \
  --allocation-output "$GEN_GPU_RUN/pilot/allocation.json"
```

Audit the complete pilot before freezing final allocation. The pilot has72 treatment trials; six native-only calibration trials are separate. The dispersion-only rule selects six or eight groups per dataset, not a favorable effect threshold. All six arm permutations are balanced **within each rate**, and every recorded local order must match its plan. The superseded pilot-v1 plan fails this revised order rule and was never executed for treatment.

```bash
python scripts/freeze_generalization.py --phase calibration --split final \
  --capture-root "$GEN_CAPTURE_FINAL" --allocation "$GEN_GPU_RUN/pilot/allocation.json" \
  --output "$GEN_GPU_RUN/final-calibration-plan.json"
srun --jobid=4872248 --exact --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=2 \
  bash scripts/run-generalization-kuma.sh "$GEN_GPU_RUN/final-calibration-plan.json" \
  "$GEN_CAPTURE_FINAL" "$GEN_GPU_RUN/final-calibration"
python -m generalization.analysis --plan "$GEN_GPU_RUN/final-calibration-plan.json" \
  --records "$GEN_GPU_RUN/final-calibration/records.jsonl" \
  --output "$GEN_GPU_RUN/final-calibration/summary.json"

python scripts/freeze_generalization.py --phase final --split final \
  --capture-root "$GEN_CAPTURE_FINAL" --allocation "$GEN_GPU_RUN/pilot/allocation.json" \
  --calibration-plan "$GEN_GPU_RUN/final-calibration-plan.json" \
  --calibration-records "$GEN_GPU_RUN/final-calibration/records.jsonl" \
  --pilot-calibration-plan "$GEN_GPU_RUN/pilot-calibration-plan.json" \
  --pilot-calibration-records "$GEN_GPU_RUN/pilot-calibration/records.jsonl" \
  --output "$GEN_GPU_RUN/final-plan.json"
srun --jobid=4872248 --exact --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=2 \
  bash scripts/run-generalization-kuma.sh "$GEN_GPU_RUN/final-plan.json" \
  "$GEN_CAPTURE_FINAL" "$GEN_GPU_RUN/final"
python -m generalization.analysis --plan "$GEN_GPU_RUN/final-plan.json" \
  --records "$GEN_GPU_RUN/final/records.jsonl" --output "$GEN_GPU_RUN/final/summary.json"
```

The final matrix is216 or288 treatment trials, according to the frozen allocation, plus18 or24 native calibrations. The protocol's optional24 diagnostic trials are not produced by this primary runner and must not be claimed completed from these commands. Preserve every failure and full completed matrix; there is no row-level deletion or best-repeat selection.

The archived completed pilot-v2 selected six groups per dataset using the declared dispersion-only rule. The completed final matrix contains216 treatment trials and18 native calibrations; both independent execution and statistics audits pass for the pilot, final and eligible calibrations. The final execution audit reconstructs9,792 assignments, and its statistics audit recomputes55,296 scheduled-job endpoints. These are evidence checks; interpretation of the effect belongs in the results report.

## Independent execution and statistics checks

For each complete calibration, pilot or final result, use its plan, records, captures and summary. For example, the existing six-group v2 calibration is:

```bash
export GEN_PLAN=configs/generalization-pilot-calibration-v2.json
export GEN_RAW=results/generalization/generalization-pilot-calibration-v2/records.jsonl
export GEN_CAPTURE=results/generalization/capture-pilot-v3
.venv-generalization/bin/python -m generalization.analysis \
  --plan "$GEN_PLAN" --records "$GEN_RAW" --output "$GEN_REPRO/summary.json"
.venv-generalization/bin/python scripts/verify_generalization_records.py \
  --plan "$GEN_PLAN" --records "$GEN_RAW" --capture-root "$GEN_CAPTURE" \
  --summary "$GEN_REPRO/summary.json" --output "$GEN_REPRO/execution-audit.json"
.venv-generalization/bin/python scripts/verify_generalization_statistics.py \
  --plan "$GEN_PLAN" --records "$GEN_RAW" --summary "$GEN_REPRO/summary.json" \
  --output "$GEN_REPRO/statistics-audit.json"
```

For pilot/final replace the variables with the exact archived or new paths; use fresh audit destinations. Both tools accept byte-preserving `.jsonl.gz` archives. The execution verifier searches `inputs/` beside the record file or its parent; use `--inputs-root DIR [DIR ...]` when NPZs are archived elsewhere. In a Git clone, `--source-revision 9263d53` explicitly checks the current study's declared files against that historical commit if the working tree has later changes. The superseded calibration-v1 instead requires `8a750f3`. A tarball cannot supply historical Git blobs: use a separately extracted matching source snapshot. The production analyzer and statistics auditor also require their frozen source files to match; `--source-revision` is not a bypass for those checks.

The execution audit independently regenerates all272 input permutations and both physical assignments, checks capture/token/prefix identities, every per-job count, three16-job proposal windows and the terminal window, four normal commit/shortcut calls, zero migration, maps, rank agreement, actual matrix/order and raw absolute-clock arithmetic. Native and shortcut physical windows must agree; combined windows can differ while collapsing to the same logical demand. Candidate/logical trajectories must agree across arms and rates. Resident-weight and numerical checks remain recorded GPU attestations, not an independent CPU re-execution of the GPU arithmetic.

The statistics audit imports no production analyzer/inference module. It reconstructs responses from all256 jobs, paired summaries and the pilot allocation or final bootstrap. Final inference resamples whole content bundles within the three datasets, using100000 PCG64 draws with seed471512; it verifies the draw-index digest, primary interval, seven Bonferroni guards, hierarchical secondary gate and date-deletion sensitivity. These intervals condition on the selected external trace material. Seven observed dates, jobs, token records and technical measurements are not independent production-day or hardware replications.

Correctness coverage is full-batch native-reference preflight for all272 unique inputs, plus native and independent FP32 checks at raw inputs16 and271 with16 evenly spaced FP32 token positions. Postflight repeats the two endpoint checks. This does not compare every future output element against FP32. Timing includes online selection, native observation/planning, any no-op agreement/commit, command/ACK, synchronization, all released jobs and drain. Input materialization, references and conditioning are excluded and logged. Architectural counts are explanatory proxies, not measured tensor-core/HBM counters or a substitute for response-time evidence.

## Separate bounded diagnostics

`generalization_diagnostics` implements the separate24-trial development-only subset in `diagnostic-protocol-v1.md`, with the instrumentation correction in `diagnostic-protocol-v2.md`. It uses three identifier-selected pilot groups, the original calibrated code calendars, main versus32-job cycles at phases0/16, and a common precomputed adapter for known native/online assignments. It changes no main runtime file, endpoint, allocation or acceptance rule. Its corrected frozen plan is `configs/generalization-diagnostics-v2.json`.

The first completed24-trial diagnostic attempt is retained under `generalization-diagnostics-v1` and is ineligible as mechanism evidence. Its recorded combining-weight digest used an integer-array hashing helper, which converted all fractional weights to zeros. The independent audit failed the float-byte comparison. V2 requires SHA256 of actual executed weights encoded as little-endian float32, C order, with no header, and records that encoding explicitly. The corrective attempt repeats the same24 states/conditions/orders; it does not change the main216 trials or inferential gates. Count both diagnostic attempts in the budget:72 pilot +216 final +48 diagnostics =336 trials. The failed audit also documents why this diagnostic-only field does not alter the main path's authenticated capture files, shared float32 input tensors or recorded numerical checks.

The corrected v2 attempt is complete and passes the strict independent audit at `results/generalization/generalization-diagnostics-v2/independent-execution-audit.json`:24 trials,1,632 reconstructed assignments across three development states,6,144 completed jobs, all30 descriptive paired comparisons, and all24 executed float32 weight digests. V1's failed audit remains alongside its unchanged records. The eligible CPU reproduction bundle includes v2 rather than v1.

```bash
export SLURM_SUBMIT_DIR="$PWD"
srun --jobid=4872248 --exact --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=2 \
  bash scripts/run-generalization-diagnostics-kuma.sh configs/generalization-diagnostics-v2.json \
  "$GEN_CAPTURE_PILOT" "$GEN_GPU_RUN/diagnostics"
python -m generalization_diagnostics.analysis --plan configs/generalization-diagnostics-v2.json \
  --records "$GEN_GPU_RUN/diagnostics/records.jsonl" --output "$GEN_GPU_RUN/diagnostics/summary.json"
.venv-generalization/bin/python scripts/verify_generalization_diagnostics.py \
  --plan configs/generalization-diagnostics-v2.json --records "$GEN_GPU_RUN/diagnostics/records.jsonl" \
  --capture-root "$GEN_CAPTURE_PILOT" --summary "$GEN_GPU_RUN/diagnostics/summary.json" \
  --output "$GEN_REPRO/diagnostic-audit.json"
```

Run after all other GPU measurements have stopped, with a fresh diagnostic output. To freeze a new diagnostic plan, use `scripts/freeze_generalization_diagnostics.py --pilot-plan PATH --output FRESH`; its referenced pilot plan must reside inside this repository because the plan stores a repository-relative reference. A fresh pilot requires its own new diagnostic freeze; do not rewrite the archived reference.

The independent audit reconstructs sequence frequency and phase, all native/online assignments and physical windows, guard/call counts, same-assignment adapter controls, raw-clock metrics and descriptive ratios. Precomputation is excluded while its runtime adapter is charged. Online versus precomputed differences are total route-path differences, not exclusive GPU selector time. Cycle phase also changes input-to-arrival alignment. Postflight still checks main-reference raw inputs16/271, not the cycle's final executed inputs47/31; full272-input preflight remains recorded. No diagnostic confidence interval or performance acceptance gate is created.

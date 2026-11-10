# Eight-GPU full-model extension — separate dataset pilots

## Full-model experiment

### Status and scope

This is a new full-model experiment, separate from the frozen two-GPU layer experiments. All five main dataset pilots completed 64/64 requests in each native EPLB-off/on arm on 2026-11-10, with no HTTP failures; the separate ShareGPT/BurstGPT timing diagnostic is still running. These are functional/calibration measurements, not final comparative evidence. Slurm job **4882284** requests two nodes with four H100s, 64 CPUs and 360 GB host memory per node for 48 hours. It began running on kh074 and kh077 at 2026-11-10 14:22:47 Zurich, with four 94 GB H100s per node and NV6 links within each node. Its 48 hours run from the actual allocation start. The account's verified limit is eight GPUs per job and per user, so a 16-GPU request is not currently feasible under this QOS. See `cluster/reservation-4882284.json`.

Use the existing **Qwen/Qwen3-30B-A3B**, revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, and vLLM `0.31.0+cu129`, source commit `db9527a46873454610df6dbedf79a36d6bf1a7f6`. The model has 48 layers, 128 routed experts per layer, top-8 routing, and a 40,960-token configured context. The complete checkpoint has 16 weight shards; downloading missing shards is preparation, not proof of a successful full-model load. Verify the index, all shard files, revision and checksums before final measurements. [Pinned model configuration](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/config.json)

### What changes relative to the layer benchmark

The old capture path saves layer-0 inputs, replaces expert computation with a recorder, and uses a 512-token prefix. The old executable benchmark supplies the same input to two ranks and combines their outputs. Its B128/top-8 dispatcher hard-codes 160 physical slots split across two ranks, a fixed stream and a reused output buffer. It is not a full inference server. Relevant code: `scripts/capture_residual_inputs.py`, `stock_comparison/native.py`, `resident_fastpath/fused.py`, and `resident_residual/compiled.py`.

The new experiment sends actual text to vLLM's HTTP chat endpoint, runs every model layer, uses the real scheduler and KV cache, and generates new output tokens until EOS or a declared output cap. There is no hidden-state replay and no 512-token input prefix cap. Response latency now refers to actual submitted requests. It is still an offline workload replay, not a production service.

**Do not install the old resident dispatcher in this experiment.** Different data-parallel ranks receive different requests. Their local routing histograms need not agree. The old assumption that identical local inputs imply identical replica decisions therefore fails. Preserving its globally coordinated choice would require demand communication and agreement at every applicable layer/step, plus support for dynamic batch sizes, CUDA graphs, all-to-all dispatch, all layers and concurrent execution. That is a separate implementation and correctness task. Native full-model baselines must work first; their results cannot establish a speedup from our layer-only dispatcher.

### Verified serving topology and first comparison

Use **DP=8, TP=1, EP enabled**, with four local DP ranks on each node. The pinned implementation makes EP size equal to DP × TP; DP greater than one activates communication of dispatched request activations. Attention is replicated across the eight TP1 ranks. This avoids treating TP-only replicated input execution as the new distributed request experiment.

Start with the supported `allgather_reducescatter` EP backend. It implements dispatch/combine with all-gather and reduce-scatter; it is **not** the optimized sparse DeepEP all-to-all backend. Record its identity in every result. The existing environment record does not establish an installed, working DeepEP stack or cross-node transport. An optimized backend is a later feasibility check, not an unannounced substitution. Verify node topology, GPU UUIDs, NCCL connectivity, resolved backend and actual dispatch/combine execution during the smoke run. [Pinned EP deployment guide](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/docs/serving/expert_parallel_deployment.md) · [Pinned parallel configuration](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/config/parallel.py)

The first two arms are:

1. **Unmodified native vLLM, EPLB off:** 128 physical expert slots per layer, 16 per EP rank.
2. **Unmodified native vLLM, EPLB on:** 32 redundant experts globally per layer, 160 physical slots, 20 per EP rank. Keep native asynchronous EPLB, window 1,000 and interval 3,000; archive the resolved weight-transfer communicator and ordinary completion/commit events. Pinned source inspection shows that absent NIXL, automatic asynchronous weight transport falls back to `torch_gloo` CPU staging; this is separate from activation collectives. Verify the resolved runtime choice rather than assuming NCCL weight transport. Do not advance its initial counter artificially.

This is a comparison of native configurations, including the memory and execution cost of extra replicas. It does not isolate replica policy from replica count. Record KV-cache capacity and scheduling limits because replica memory may change available KV capacity. Use identical model, dtype, scheduler settings, maximum context, cache policy, sampling, arrivals and request IDs. Preserve native graph execution unless a separately labelled eager-mode smoke diagnostic is required. Resolve and archive every default before freezing the final protocol. Instrumentation for detailed routing/commit diagnostics belongs in separate diagnostic runs unless its cost is measured and charged in both arms.

`launch_native_draft.sh` produces exact commands for this topology. It now explicitly selects multiprocessing DP and one API process; internal load balancing still reaches all eight DP engines. The primary and worker start concurrently in one two-task Slurm step. Its CLI flags were checked against the pinned [argument definitions](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/engine/arg_utils.py) and [serve entrypoint](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/entrypoints/cli/serve.py). CPU tests and dry runs pass. Installed bytes for 12 critical vLLM sources match the pinned commit (`results/full-serving/setup/installed-source-check.json`). This verifies selected files, not the entire wheel or a running GPU server.

### Actual requests and fresh data

Use bounded, previously unused WildChat conversations from the pinned dataset revision `f66566ceaaeb619dd98ffb0f3bf3ce1f86775ac4`; retain its ODC-By attribution and privacy/access constraints. Verify license and source metadata in the archived manifest. Split by conversation ID and normalized-text hash, excluding IDs used in earlier experiments and separating pilot from final data. Do not select by expert distribution or measured performance. [Dataset and license](https://huggingface.co/datasets/allenai/WildChat)

Retain natural variable-length user text and, when explicitly selected, the preceding conversation history. Use the pinned tokenizer's chat template with `enable_thinking=false` consistently; validate that setting in the smoke run and retain the serialized messages and actual token counts. Generate the next assistant reply; earlier assistant messages are conditioning history, not generated outputs. Do not concatenate unrelated documents to fabricate long prompts.

For the initial pilot, set maximum model context to 32,768, request maximum new tokens to 2,048, temperature to zero, and normal EOS termination. Reject over-limit requests explicitly rather than silently truncating; count exclusions and report the surviving length distribution. A request that stops at the output cap is not an EOS completion and must be reported as such. Include natural prompts longer than 512 tokens. If this source sample contains too few long requests, use a separately labelled, preselected 2–8k/8–30k length diagnostic; do not reweight the main sample after observing speed. Freeze this policy after checking memory and feasibility, before final measurements.

Use contiguous, disjoint blocks from the already licensed Azure public conversation/code arrival data. Assign actual requests to these timestamps with a fixed seed and preserve absolute arrivals under overload. This pairing provides external arrival patterns plus external content; it does not reconstruct a joint production trace. Record source block IDs, any timestamp rescaling and the independent content/arrival pairing. Submit via streaming HTTP so TTFT and per-token arrival times are observable. The load generator must not become the bottleneck; record actual send delay, concurrency and failed connections.

### Bounded execution plan

1. **Allocation and checkpoint preflight:** verify the running reservation, two hosts/four H100s each, model completeness, pinned package/source hashes, available RAM/KV capacity, network interfaces and backend. Start only one arm at a time across the same eight GPUs.
2. **Smoke:** complete a few real short and long requests through all 48 layers, including generated decode and EOS/cap accounting. Test both native arms. Confirm health, finite logits/output, identical tokenization, correct number of ranks, stable process completion and no failed collectives. Compare deterministic output tokens/log probabilities on a small correctness set with a documented numerical tolerance; flag divergent generations for investigation rather than assuming bitwise equality across parallel reductions.
3. **Pilot:** use excluded conversations and arrival blocks to choose at most two common absolute offered rates, one with headroom and one near useful capacity. Establish how many serving steps a block produces. Final on-runs must expose at least two ordinary EPLB decision opportunities after native startup, or be labelled insufficient to study adaptation. Do not shorten the interval merely to force a migration. Freeze workload IDs, counts/durations, ordering, sample allocation and criteria before final measurements.
4. **Final native comparison:** budget at most **24 measured runs** initially: six independent content/arrival block pairs × two rates × two arms. Alternate AB/BA order and restart each arm to a declared initial state. Warm model/kernels consistently and record any placement established during warmup; preserve native initial placement for the timed study by restart or an explicitly verified reset. Avoid warming on final requests. Charge initialization separately and all online dispatch/EPLB/communication in response and completion measurements. Keep arrival timestamps fixed while draining every admitted request. Extend only if pilot precision and the 48-hour walltime justify a preregistered smaller/larger allocation before final data are inspected.
5. **Diagnostics only if needed:** profile a few fixed requests and one decision boundary for layer compute, dispatch/combine, padding, rank imbalance, scheduler/KV behavior and weight movement. Native all-gather/reduce-scatter and a verified optimized backend would be separate studies, not mixed within the main comparison. A custom dispatcher arm requires its own full-model correctness gate and explicit implementation; it is not part of this baseline launch draft.

The 24-run outline is not a frozen protocol: pilot precision and actual run lengths are unknown. The freeze must fit measured duration, checkpoint load time, warmup, diagnostics and drain within the reservation, and must not cut an overloaded run short to improve reported latency.

### Records, decisions and falsification

Record git/source and environment hashes, checkpoint shard hashes, hardware/topology, all resolved server flags, model/tokenizer/template revisions, workload IDs and input/output lengths, generated token IDs or approved hashes, finish reasons, client send/first-token/token/end timestamps, status/errors, server queue/service metrics, KV usage/preemption and synchronized diagnostic timing. Preserve request-level records and run-level totals. For EPLB diagnostics record proposed map, transfer start/end, normal commit completion and map version on each rank; draining asynchronous work alone is not evidence of commit. Verify logical expert IDs, replica equivalence and rank agreement on a small instrumented subset. Record backend logs and profiler traces that show actual dispatch/combine communication.

Report TTFT, inter-token/TPOT latency, end-to-end mean and p95/p99 request latency, completed elapsed time including drain, achieved output tokens/s, failed/admitted/completed counts, generated-token counts and initialization separately. Do not call a lower MoE-kernel time an end-to-end speedup. Treat independent content/arrival blocks as experimental units; use paired run/block uncertainty and disclose tail sample size. Do not bootstrap individual requests as if they were independent runs.

Before final runs, predeclare a primary completed-elapsed or throughput endpoint at common offered rate and a 5% useful-gain target, with 2% acceptable latency-regression margins and multiplicity handling for the two load guards. A confidence interval spanning useful gain and material regression is inconclusive. Failure includes missing/corrupt requests, violated numerical/routing checks, inconsistent committed maps, reduced throughput hidden by normalization, or a method whose charged end-to-end benefit does not exceed uncertainty. Native EPLB may win, lose or be indistinguishable here; none is predetermined. Publish exact workload manifests, commands, raw timing records and an independent replay procedure. A native baseline success establishes full-model feasibility, not that our custom method is faster.


## Execution and datasets

### Implemented first execution and current evidence

`prepare_data.py` selected 448 real WildChat conversation prefixes from all 88,238 rows of the pinned first shard. It excluded earlier source identities and normalized text duplicates. All 448 requests were independently retokenized with Transformers 5.17.0 and the pinned Qwen chat template. Prompt lengths are **16–8,121 tokens**, median 307; **165 exceed 512 tokens** and 64 exceed 2,048. There is no 512-token prefix truncation. Eight smoke inputs span 20, 37, 96, 263, 444, 933, 1,946 and 8,121 tokens. Public identities and preparation checks are in `results/full-serving/data-v1/request-manifest.json`; private message JSONL remains outside Git. The 384 final requests are prepared candidates, not completed measurements or a frozen final allocation.

The automatic first execution is deliberately a **functional/calibration pilot**, with 64 excluded conversations split into disjoint smoke (8), simultaneous capacity calibration (32) and external-arrival (24) stages. It starts native EPLB-off, completes all three stages, terminates only that serving step, then repeats native EPLB-on with identical inputs/calendars. The external calendar preserves a previously unused contiguous Azure block's relative timestamps. Its common absolute rate is fixed at 0.6 times the off-arm pilot's completed-request rate before any on-arm result exists. A simultaneous burst is a calibration diagnostic, not a production arrival trace or final sustainable-capacity estimate. This fixed arm order and single pilot block cannot support a final superiority claim.

The source and checkpoint preflight hashes all 16 weight shards and checks every indexed tensor, including all 48 layers' 128 expert triplets. Each server launch archives actual hardware, node addresses, selected installed source hashes and unchanged checkpoint fingerprints. The model starts with native CUDA graphs/scheduling. Readiness requires health and model endpoints, followed by actual successful generated responses. Initialization is timed separately. Public request records contain content hashes, actual token counts, scheduled/send/first-content/end timestamps, finish reasons and errors; generated text and smoke logprobs stay in private files. SSE chunks are not individual tokens. Aggregate stream-boundary TPOT is labelled an estimate. HTTP alone does not expose server queue/service decomposition; native metric snapshots are retained separately.

Execution uses `wait_and_run.sh RUN_ID`, tied to allocation 4882284. It waits at most 24 hours for that allocation, then starts a four-CPU, zero-GPU controller step and two 56-CPU/four-GPU serving tasks. It does not submit another allocation, switch accounts, cancel the holding reservation, shorten EPLB windows, or install the layer-only dispatcher. All admitted requests are drained, including failures; missing usage or tokenization mismatches invalidate a run. Server shutdown uses the actual recorded Slurm step ID so native SIGTERM handlers can run. Native commit consistency remains a separate diagnostic; process exit or asynchronous drain is not counted as a commit.

The pilot writes `status.json`, `completed-phases.json`, per-arm raw request JSONL, server logs, metric snapshots, launch/source/checkpoint/hardware records and a smoke-output hash comparison under the server's `full-serving/RUN_ID/results/`. A failure stops further execution and preserves logs. Successful pilot completion still requires inspecting output/logprob differences, native decision/commit opportunities, backend behavior and capacity before freezing the independent final comparison. These pilots do not establish an end-to-end speedup.


### Separate named datasets

The extension keeps **WildChat, ShareGPT, InstructCoder, NuminaMath-1.5 and LongBench as five separate content workloads**, each with its own pilot and reserved final file. It does not pool their response times into a single headline speedup. The initial suite has 64 pilot and 64 final candidates per source (320 of each); the other 320 previously prepared WildChat final requests remain reserved. All final candidates remain unmeasured until the pilot supports a frozen allocation and duration. Source and license evidence live under `results/full-serving/suite-sources/`; `suite-v1/suite-manifest.json` records selection and hashes.

The separate suite is launched by `run_suite_allocation.sh RUN_ID`, using the
already running allocation. It starts fresh native off/on servers for each
source and then a separately labelled ShareGPT/BurstGPT timing diagnostic.
`run_suite.py` checks the admitted payload hashes before the first server starts;
`report_pilots.py` keeps every dataset and stage separate and withholds paired
ratios for missing, failed or mismatched requests. The executed source revision
is recorded in each cohort, independent of subsequent documentation updates.

The completed WildChat pilot is archived under `results/full-serving/pilots-v1/`;
its manifest hashes all 56 raw cohort files. Recompute its separate-stage report
with `python -m full_serving.report_pilots results/full-serving/pilots-v1/wildchat --output /tmp/wildchat-pilot-report.json`.
Both arms completed 64 requests without HTTP failures. Native logs contain three
ordinary asynchronous launch entries, but no successful-commit attestation.
Generated reply lengths differ across arms; these observations are descriptive
and do not satisfy the final numerical, repeated-block or speedup criteria.

ShareGPT uses the release recognized by the pinned vLLM benchmark documentation. Its upstream creators filtered and segmented conversations; our inputs retain complete released prefixes, not claimed original unsegmented sessions. We keep at most one released segment per underlying conversation and exclude its recorded target reply. The selected 128 prompts range from 22 to 5,682 tokens. [Official vLLM benchmark data](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/docs/benchmarking/cli.md) · [Released ShareGPT dataset](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)

InstructCoder and NuminaMath-1.5 connect the evaluation to METRO's named workloads and eight-GPU Qwen3 model configuration. METRO does not disclose an exact revision, row selection, template or generation cap, so this is aligned workload testing, not an exact reproduction. InstructCoder's selected prompts contain code-editing instructions plus input code (62–317 tokens); NuminaMath contains problems only (27–287 tokens), with no reference solution or answer supplied. Short natural inputs are retained. NuminaMath declares Apache-2.0. InstructCoder's official repository declares MIT, while its Hugging Face card lacks a separate explicit dataset-license declaration; that distinction remains recorded. [METRO methods, preprint](https://arxiv.org/html/2512.09277v1#S6.SS1) · [InstructCoder](https://huggingface.co/datasets/likaixin/InstructCoder) · [NuminaMath-1.5](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5)

The bounded original LongBench subset uses Qasper, GovReport and HotpotQA, with the pinned official task templates. Its 128 selected inputs range from 2,087 to 27,013 tokens. One question per context/document group prevents related Qasper questions from crossing the pilot/final boundary. Three oversized GovReport source prompts were rejected with their lengths recorded; none was truncated. These tasks also appear in FreeBalance, but its fixed-8K prefill measurements are not equivalent to natural-length autoregressive serving. Our generated outputs remain capped at 2,048 with normal EOS; the official task-specific answer caps are recorded as source metadata, not silently applied. [Original LongBench](https://github.com/THUDM/LongBench/tree/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench) · [FreeBalance evaluation, preprint](https://arxiv.org/html/2608.14205v1#S4)

**BurstGPT is a separate timing diagnostic, not a sixth prompt dataset.** Its published data provide request timestamps, sizes and failures, not authentic prompt text. A fixed contiguous 24-row block is paired with independently sourced ShareGPT pilot text. This repeated-content diagnostic restarts native servers and preserves absolute, explicitly scaled source arrival offsets. It does not reproduce BurstGPT's original content, expert routing, model or token lengths. The selected block happens to contain no failures; the source was failure-inclusive and no failed rows were filtered. Results remain labelled as a hybrid replay. [BurstGPT KDD 2025](https://doi.org/10.1145/3711896.3737413) · [Official artifact](https://github.com/HPMLL/BurstGPT)

The generic selector uses deterministic hash order, one source group per selected request, exact normalized text checks against all prior 448 WildChat requests and earlier selected suite sources, and the context-fit condition including output allowance. Private payloads stay outside Git; public metadata uses an explicit allowlist. Dataset source bytes and derived candidate files are admitted by hash before serving. The suite retains fresh payload distinctions: named-source pilots are new full-text content except the already declared WildChat pilot; BurstGPT reuses ShareGPT pilot content by design. The final comparison must report dataset-specific uncertainty and enough ordinary EPLB opportunities, rather than treating this pilot or 64 requests as evidence of sustained adaptation.


### Native observability limits

Native asynchronous EPLB logs show launch attempts, excluding profiling calls, but do not expose per-layer/final successful commit maps or a completion counter. The native counter starts at 2,250 against the 3,000-step interval, so the first nominal launch is after 750 calls; dummy DP steps and pending asynchronous work prevent reconstructing opportunities from request or token counts. A later ordinary launch is conditional evidence that the earlier cycle is no longer pending, not a proof of weight equality or the final committed map. `drain_async` discards transferred weights. Consequently the pilot reports launch counts and leaves commit consistency unverified until a separate diagnostic establishes it. `log_balancedness` remains disabled because enabling it adds collectives and synchronization. [Pinned EPLB state](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/distributed/eplb/eplb_state.py)

The smoke requests retain prompt/output token IDs and selected-token logprobs privately, with hashes/counts in public records. Prompt IDs must equal the frozen tokenizer's IDs; requested output diagnostics must be present and finite. These are HTTP-level checks, not proof of all internal logits or expert maps. Greedy off/on generations may diverge because of batching/reduction order. Compare logprobs only along a common token prefix; later positions have different conditioning. Native metric snapshots can supply per-engine request queue and RUNNING-time means, prefill/decode means, cache hits and preemptions. Request RUNNING time is not isolated GPU service time, and histogram buckets give only coarse tail bounds.


### Toolchain admission

The first complete eight-rank model load and native graph capture reached sampler warmup, which failed because the runtime-only Python environment lacked an `nvcc` executable. A private CUDA 12.9.1 compiler/header tree was assembled from NVIDIA's pinned redistributables, including the required cuRAND headers. Its native FlashInfer scalar/per-row top-k sampling test passed against PyTorch on both reserved nodes (Slurm step 4882284.10). This is an environment fix applied to both arms, not a vLLM algorithm modification. Python packages, system CUDA and the earlier frozen layer environments remain unchanged. Compiler/header hashes and version are checked before each model startup. The failed attempt remains under `results/full-serving/attempts/v3/`; it contains no completed benchmark requests. [NVIDIA CUDA 12.9.1 component manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.9.1.json)

The first separate suite completed WildChat and ShareGPT OFF, then ShareGPT ON
failed during startup: Ninja could not open its build log in the shared home
FlashInfer cache. Subsequent Gloo waits were consequences of the exited worker;
the underlying reason the cache directory became unavailable is not proven.
The complete attempt is retained under `results/full-serving/attempts/suite-v1-sharegpt/`.
The repair uses FlashInfer's supported `FLASHINFER_WORKSPACE_BASE` under project
storage, with a distinct directory per allocation/node and a single sampler
prebuild before workers fork. Native top-k/top-p checks passed on both nodes
in step 4882284.18. Model, Python package pins and inference configuration remain
unchanged. Startup now detects logged native worker failure promptly instead of
waiting for an otherwise-live API parent's full timeout.

Retry v2 restarts both ShareGPT arms and then InstructCoder, NuminaMath, LongBench
and the declared BurstGPT diagnostic. Its rules were recorded before new
measurements in `results/full-serving/setup/retry-protocol-v2.json`: use the first
complete admitted pair regardless of its timing, retain every failure, and do not
pair the old ShareGPT OFF with a repaired ON. WildChat's earlier complete pair
remains separate, with its previous cache location disclosed. These are repeated
pilot inputs after an infrastructure repair; final candidates stay unexecuted.

### Reproducing the closed pilot comparisons

The headline comparison uses each dataset's **24-request external-arrival
stage**, separately. Its other 40 pilot requests are eight smoke requests and a
32-request simultaneous calibration burst. Do not pool these stages. Response
time runs from the scheduled arrival until the complete HTTP reply; completed
elapsed time includes the arrival horizon and drain, but excludes server
startup, earlier phases and shutdown after completion. The HTML/CSV also expose
TTFT from actual send, output-token totals, cap finishes and completed output
tokens per second. This token rate includes gaps in the arrival calendar and is
not a peak-capacity measurement.

From the repository root, regenerate the five main datasets without a GPU:

```bash
python -m full_serving.report_pilots \
  results/full-serving/pilots-v1/wildchat \
  results/full-serving/pilots-v1/sharegpt \
  results/full-serving/pilots-v1/instructcoder \
  results/full-serving/pilots-v1/numinamath \
  results/full-serving/pilots-v1/longbench \
  --output /tmp/native-comparison.json
python -m full_serving.render_report /tmp/native-comparison.json \
  --output-prefix /tmp/native-comparison
python -m full_serving.report_pilots \
  results/full-serving/pilots-v1/sharegpt-burstgpt-diagnostic \
  --output /tmp/burstgpt-diagnostic.json
python -m full_serving.render_report /tmp/burstgpt-diagnostic.json \
  --output-prefix /tmp/burstgpt-diagnostic
python -m full_serving.audit_runtime --root results/full-serving/pilots-v1 \
  --output /tmp/runtime-audit.json
```

Paths and creation timestamps may differ across checkouts; the underlying
request counts and numerical summaries must match. Per-dataset archive
manifests hash the original raw bytes, including logs and metric snapshots.
The independent timestamp recomputation recipe is
`results/full-serving/audits-v1/recompute_external.py`.
Revalidating private prompt identity and smoke-token decoding additionally
requires reconstructing the admitted source data and pinned tokenizer; the
public timing records alone cannot establish those checks. Raw prompt text,
generated text, token-ID arrays and selected-token logprobs remain outside Git.

Each native pair runs OFF then ON once, on the same eight physical GPUs.
Consequently these observations have **no independent-run confidence interval**.
Generated replies and token counts can differ even with temperature zero, so
the observed response difference includes changed generated work. The comparison
also includes the memory and execution effects of 32 extra replicas in ON.
Neither a request-level bootstrap nor dividing by output tokens removes those
limitations. The declared final data and balanced repeated-run protocol remain
unused; this pilot cannot establish a general speedup or satisfy the final 5%
gain criterion.

The common external calendar is scaled using OFF's calibration. This preserves
the same absolute arrivals across arms, but does not establish a common fraction
of sustainable capacity: in particular, InstructCoder OFF logged MoE-kernel JIT
compilation during calibration. Phase-attributed JIT warnings, backend choices,
KV capacity and shutdown-only errors are retained in the runtime audit. Native
EPLB launch entries remain separate from unobserved successful commits.

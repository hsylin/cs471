# Why the resident path loses to EPLB-off

The largest removable overhead package in the matched development diagnostic is the recording/controller path, followed by online resident selection. Both execute despite a placement that never changes. This identifies the overhead relative to stockoff; it does not mean either package dominates the entire MoE computation.

The exact pinned source adds two per-job scalar writes for native forward/recording state, history copy/zero work during the sampled window, and a full proposal every 64 jobs that the existing SKIP policy always rejects. Its unchanged-map shortcut avoids movement/commit work but leaves this recording/planning path. The original resident selector additionally launches separate histogram, greedy-home and map/record kernels and performs repeated Python metadata checks and output allocation.

## Matched development evidence

We used six previously measured pilot content groups, each with 64 distinct authenticated batches cycled eight times, two reverse-order technical repetitions and five variants. All variants compute the real stock gate/top-k and expert layer. The four 160-slot variants have identical physical assignments and weights; stockoff is a separate 128-slot contextual anchor. The offline adapter excludes the known-home selection calculation, but still charges its map/record kernel. It is a diagnostic estimate, not a deployable comparator or a strict runtime bound.

| Variant | Mean service, µs/job |
|---|---:|
| Stock EPLB-off | 708.206 |
| Current online method | 819.908 |
| Current controller + exact known homes | 784.220 |
| Controller inactive + online three-kernel selector | 767.501 |
| Controller inactive + exact known homes | 732.882 |

Averaging the paired factorial contrasts gives 51.872 µs/job for the recording/control package and 35.154 µs/job for selection; the interaction is 1.068 µs/job. These are package effects with charged execution/coordination, not exclusive CPU or GPU component timers. Technical repetitions are averaged within each of six content groups. No confidence intervals or hypothesis tests are attached to these reused-input diagnostics. All 60 records, raw clocks, predetermined orders, input hashes, physical assignments and static/current state checks pass an independently implemented audit.

Stockoff already executes each logical expert on one physical copy. In the previous held-out study, the resident policy did not lower total active-copy or padded-tile counts versus stockoff. It redistributed work between ranks. Lower rank imbalance therefore did not establish enough execution savings to pay for extra dispatch and control. The residual known-home-versus-stockoff difference can include the charged remap adapter, 160/128-slot geometry and other execution/host effects; this diagnostic does not isolate them.

## Implemented repair and honest development limit

The chosen custom repair preserves the existing greedy assignment policy while removing the inactive recording/planning/migration path and fusing histogram, home choice and remapping into one CTA and one launch. It reuses physical-ID storage on the same sequential inference stream. Setup and post-episode audits verify fixed residency and stream use. This is a custom static 160-slot path, not supported upstream EPLB-disabled configuration; native enabled-layout buffers remain allocated. The native gate/top-k, expert computation and output reduction stay intact.

A separate 36-run development check compares this fused path, the unchanged current method and stockoff contemporaneously on the same six old pilot groups. All 18 adversarial case/dtype checks agree with two independently expressed integer references; actual-input whole-layer outputs also pass. Its group-paired service ratio is 0.892074 versus current, but 1.021542 versus stockoff. It remains slower than stockoff in every development group (ratios 1.007921–1.036421). Fusion versus the earlier three-kernel run is a cross-run descriptive comparison, not another matched factorial contrast.

The repair removes most of the current method’s penalty. It has not established a speed advantage over stockoff. The fresh confirmation retains the original current path as a control and keeps the 5% useful response-gain and 2% regression criteria unchanged. If those criteria fail, stockoff remains the supported recommendation for this tested scope. The experiment does not expand into kernel searches, placement searches, full-model serving or new hardware to obtain a positive result.

See [frozen protocol](protocol-v1.md), [data provenance](data-provenance.md), [factorial evidence](../../results/resident-fix/diagnostic-v1/reproduction/resident-fix-factorial-analysis.md), and [fused development evidence](../../results/resident-fix/fused-development-v1/reproduction/resident-fix-fused-analysis.md). Fresh treatment outcomes are not included in this development diagnosis.

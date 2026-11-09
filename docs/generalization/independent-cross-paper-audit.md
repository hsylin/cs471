# Independent extraction audit: eight conference papers

This is a targeted check of high-risk methods and evaluation claims against the cached full texts, independently of the two extraction authors. It covers denominators, uncertainty language and section/appendix references; it is not a reproduction. All eight PDF hashes match the reviewed inventories. No new papers or GPU runs were used.

The checked numerical headline results are supported. Four definite attribution/locator issues and two precision clarifications were sent to the owning reviewers before consolidation. Both owners have now confirmed the corrections in their extraction files. MegaBlocks' corrected metadata exclusion and absence of appendices were independently confirmed.

## MegaBlocks: Efficient Sparse Training with Mixture-of-Experts

Primary text: [megablocks](https://arxiv.org/pdf/2211.15841).

- **§6.3, Fig9; text607–655**: 18 uniform-routing problems, six operations per model;100 kernel executions per problem; sparse-metadata construction explicitly excluded.98.6% of cuBLAS and4% spread describe problemwise kernel throughput, not independent-run confidence.
- **§6.1–6.2, Figs7–8; text528–602**: Up to40% over best Tutel settings, and1.8–2.4× versus dense Megatron at comparable validation loss are distinct comparisons. No appendices in retrieved manuscript.
- No new factual correction identified in the targeted claims.

## Tutel: Adaptive Mixture-of-Experts at Scale

Primary text: [tutel](https://arxiv.org/pdf/2206.03382).

- **§5.2/Fig14; abstract; text694–777**: Layer gains4.96× at16GPUs and5.75× at2048GPUs are different scale cases. SwinV2-MoE1.55× training and2.11× inference are separate end-to-end denominators. Fig14 is an actual incremental feature breakdown.
- **Appendices B–C; §5.1**: Appendix kernel/capacity/top-k anchors exist;243 single-layer settings and up2048A10080GB are supported.
- No new factual correction identified in the targeted claims.

## Accelerating Distributed MoE Training and Inference with Lina

Primary text: [lina](https://www.usenix.org/system/files/atc23-li-jiamin.pdf).

- **§7.1; text516–544**: Training uses10 warmup and50 averaged steps; inference is top1 versus training top2; hierarchical all-to-all explicitly enabled for Lina and DeepSpeed.
- **§7.3; text676–705**: Five executions split into two unprofiled end-to-end timings and three profiling runs. Median TXL4/16expert gains1.54×/1.45× and16expert p95 TXL/BERT gains1.82×/1.68× compare with baseline; ideal balanced gates are a separate altered-routing reference.
- **Correction/clarification (baselines)**: Original extraction wrongly attached the hierarchical-alltoall statement to DeepSpeed and Tutel. Name Lina and DeepSpeed as the explicit enabled pair; retain Tutel as a separately compared baseline.

## SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations

Primary text: [sonicmoe](https://arxiv.org/pdf/2512.14080).

- **Abstract; §6; text28–30,891,970–1021**: 1.86× BF16 versus ScatterMoE is for fine-grained7B;64H100 achieves213Btokens/day versus225B on96H100, not equal-device throughput. Token rounding changes logical routing and is separately analyzed.
- **Appendices F–I/Table9; text1856–1863,2279,2600–2628**: These appendix/table anchors exist. Large token groups131072/81920/32768 and FineWeb-Edu training are supported. Current extended manuscript and conference summary remain distinct versions.
- No new factual correction identified in the targeted claims.

## DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving

Primary text: [distserve](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf).

- **§6.1,Table1,Fig7; text574–639**: OPT13/66/175B;32A10080GB over4nodes with25Gbps cross-node links; request data lack timestamps and use Poisson arrivals; LongBench input capped at2048.
- **§6.2/Fig8; text668–700; Appendix C**: Up to4.6× per-GPU chat rate versus vLLM at90% SLO attainment is valid;7.4× uses DeepSpeed-MII. MII175B is excluded for implementation/memory limits. Appendix C changes attainment target to99%.
- No new factual correction identified in the targeted claims.

## Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve

Primary text: [sarathi](https://www.usenix.org/system/files/osdi24-agrawal.pdf).

- **§5/Table1–4,Figs10–14; text704–758,825–827,1003–1008**: 3.7× is Yi34B strict-SLO ShareGPT capacity versus vLLM. Approx25% prefill overhead uses512-token chunks versus unchunked Yi34B. Request p50TTFT and token p99TBT are different observational units; sustainable-load rule also limits median scheduling delay to2s.
- **Correction/clarification (support_section_table_page)**: Original range cited Figs10–15 although retrieved proceedings end at Fig14. Use Figs10–14. §4.4 implementation exists and requires no change.

## AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving

Primary text: [alpaserve](https://arxiv.org/pdf/2302.11665).

- **§6/Table1–2; text699–765**: Up to2.3× fewer devices,10×rate and6×burstiness are separate objectives at99% SLO attainment. Table2 simulator/testbed attainment differs by up1.6percentage points; its79.0%/80.6% pair rules out a blanket less-than2% relative-error interpretation.
- **§6.3–6.4/Figs13–14; text918–938**: §6.3 is physical serving of very large models. Three repeated changing-traffic experiments are in §6.4, with separate randomly sliced assumed/actual one-hour traces. This does not establish repeated-run CIs for all results.
- **Correction/clarification (uncertainty)**: Original three-repeat locator was §6.3. Change to §6.4/Fig14.
- **Correction/clarification (uncertainty)**: Original <2% simulator/testbed discrepancy wording can be interpreted as relative error. Specify <2 percentage points of SLO attainment (Table2).

## DynamoLLM: Designing LLM Inference Clusters for Performance and Energy Efficiency

Primary text: [dynamollm](https://iacoma.cs.uiuc.edu/iacoma-papers/hpca25_2.pdf).

- **§V.B,D,E/Figs7,15–16; rawtext1082–1088,1295–1320**: One-hour35% and day42% measured energy savings and week47%conversation/56%code simulated savings all use SinglePool baseline. The one-hour setup provisions12servers×8H100 for the baselines; 24h setup11servers×8H100.
- **§III.A/TablesI–III; rawtext250–252; §V.A**: Characterization uses30-second runs repeated5times and reports means; this is distinct from long cluster experiments. No independent-run CI was identified in inspected evaluation. Public1h traces and paper-specific day/week trace experiments must not be conflated with proof of exact asset identity.
- **Correction/clarification (results)**: Percentages lacked an explicit comparator, though numbers were correct. State all four relative to SinglePool.
- **Correction/clarification (uncertainty/metrics_timing)**: Avoid erasing the stated five-repeat30-second characterization method when summarizing absence of CIs. Preserve the repeat detail and distinguish it from cluster runs.

The machine-readable audit records exact PDF/text hashes and locators. Request quantiles, averages over kernel iterations, cross-configuration spread, and repeated trace runs are distinct quantities; none should be silently promoted to population-level uncertainty.

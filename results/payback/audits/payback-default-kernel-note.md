# Default fused-MoE kernel configuration: scope note

The final initialization log, `work/payback-results/logs/skb-payback-final-4866456.log:22`, reports that `E=80,N=768,device_name=NVIDIA_H100.json` was absent and vLLM used its default MoE configuration. The same warning appears in the profile, smoke and both pilot logs. This is consistent configuration provenance, not a numerical failure. No tuning was changed between arms.

The existing profile audit had no fallback log in its original fetched inputs and therefore made no fallback diagnosis. The subsequently fetched initialization logs supply that missing evidence. Its other cautions still apply: the calibration does not establish memory/compute bottlenecks, and profiler rows are not an additive wall-time decomposition.

The installed `fused_moe.py` was read without executing a runtime or changing files. Its SHA256 is `21860340e4b409b01666e2ba0f0edae66b0f97b859756b174b573bab895cfaf7`, exactly the source file pinned in the frozen native manifest. At commit `db9527a46873454610df6dbedf79a36d6bf1a7f6`, [configuration lookup](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/layers/fused_moe/fused_moe.py#L1109) returns no tuned table after this warning; [selection](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/layers/fused_moe/fused_moe.py#L1419) then calls the default rule. Here `E=80` is the number of local physical slots per rank, including replicas, and `N=768` is the expert intermediate dimension used for the lookup.

The general BF16 fallback rule, evaluated statically for the two declared batch sizes, selects:

| Batch tokens | BLOCK_SIZE_M | BLOCK_SIZE_N | BLOCK_SIZE_K | GROUP_SIZE_M | SPLIT_K | num_warps | num_stages |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 64 | 128 | 64 | 1 | 1 | 4 | 3 |
| 1024 | 128 | 128 | 64 | 1 | 1 | 8 | 3 |

These values follow from the pinned source's default rule; this note did not collect a new runtime launch-configuration trace. The source comments' generic statements about small/large batch bottlenecks are heuristics, not hardware-counter evidence for this experiment.

The three-arm action comparison remains valid **within this fixed execution path**. Each paired arm uses the same batch shape, weight dimensions, local slot budget, kernel-selection code and fallback rule. Placement changes alter routed work and potentially expert activation/padding, which are part of the action effect being measured. A missing tuned table is therefore not an arm-specific implementation change that invalidates pairing.

The portability limit is material: optimized tile choices or another expert kernel could change the relative cost of active experts, padding and token imbalance, and therefore change free-install headroom and migration payback. The report should describe results as applying to **the pinned vLLM0.31 BF16 fused-MoE fallback on the tested two-H100 setup**. It should not claim optimized H100 performance, a hardware-wide impossibility result, or that another tuned runtime has the same candidate boundary. The warning alone does not establish how much improvement tuning would yield.

Suggested report sentence: “All arms used the same pinned fused-MoE default configuration because the E=80/N=768 H100 tuning table was absent; the measured action boundary is specific to this executed path, and optimized-kernel generalization remains untested.”

No final outcome records were read for this note. Only initialization log lines, the earlier profile audit and pinned source were inspected; no tuning, runtime execution or frozen-code change was performed.

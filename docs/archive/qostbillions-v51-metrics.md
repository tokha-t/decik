# QostBILLions — final v5.1 metrics and techniques

Validated on 31 October 2026. Measured code: [`d9ab004`](https://github.com/tokha-t/decik/commit/d9ab00419faa5d81ddf48fb35ec562284e5d62f9).
[Public G7 run, saved version 364261503](https://www.kaggle.com/code/toktarkhan/gpt-v51-final-validation?scriptVersionId=364261503).

## Final metrics

| Metric | Base | v5.1 |
|---|---:|---:|
| Whole `soup train` wall time | 85.21 s | 63.88 s |
| Peak GPU memory | 14,345 MiB | 2,035 MiB |
| Training-loop time | 60.64 s | 39.38 s |
| Public eval loss | 2.04967 | 2.04964 |

- **Speedup: 1.3339×. Memory reduction factor: 7.0491×.**
- Eval-loss change: **-0.001%** relative to base; within the 0.5% limit.
- **10/10 rule checks, 21 GPU tests, 79 Soup regression tests passed.**
- **9 original Triton kernels; 13.0% combined training-time share** in the lower-share run (steps 3+; minimum 5%). Supplied SwiGLU kernels are excluded.
- Solution loss-curve deviation: first step **0.03%**, mean over steps **0.01%** (limits 1% / 3%).
- Individual wall times: base **83.01 / 87.40 s**; solution **63.05 / 64.70 s**.
- Saved activations: approximately **13.4 KiB/token/layer**, including **9 KiB** of norm-input sources; excludes weights and temporary buffers.

Protocol: Qwen2.5-1.5B NF4, LoRA rank 16 on q/v; **64 steps, seed 1**, defaults only.
Two base and two solution runs, ordered base/solution/solution/base; table shows medians.
One Tesla T4 used in a T4 x2 environment. Compiler caches reset for each training run;
compilation and overlapping warmup are included in wall time. Evaluation: **256 public rows, 38,109 tokens**.

## Techniques used

1. **Remove padding within each batch:** pack real tokens, retain per-row positions/causal attention and the original dropout draws.
2. **Custom NF4 double-quant dequantization → fp16:** Triton unpacks weights; cuBLAS performs the matrix multiplications.
3. **Custom RMSNorm forward/backward:** one Triton launch each, with fp32 reductions.
4. **Keep embedding/lm_head weights fp16:** embedding output returns to the fp32 residual stream.
5. **Recompute MLP:** use fp16 gate/up intermediates and rebuild them during backward instead of saving them.
6. **Recompute attention:** save q/k/v without grouped copies or the attention output; repeat each row's forward in backward.
7. **Rebuild LoRA inputs:** reconstruct the norm output, bit-pack q/v dropout masks, and reuse the same Triton input kernel in forward/backward. Fuse Q/K preparation with RoPE and fp16 layout conversion.
8. **v5.1 norm-input reconstruction:** `AddRMSNormFn` rebuilds the post-attention norm input from the shared fp32 layer residual plus fp16 attention output; MLP reuses this pair. This is an autograd wrapper using the existing RMSNorm kernel; add and norm remain separate GPU operations.
9. **Chunk the head in groups of 512 tokens:** compute loss, entropy, accuracy, rewind losses and gradients without full-batch vocabulary logits.
10. **Background thread warmup:** compile original Triton kernels while the model loads, within measured time.

The v5.1 additions are item 8 and LoRA forward reuse in item 7. G5 subprocess workers and v5.2 attention experiments were not adopted.

Environment: torch 2.11.0+cu128, Triton 3.6.0, Soup 0.77.0,
transformers 5.16.1, PEFT 0.20.0, TRL 0.29.1,
bitsandbytes 0.50.2, accelerate 1.14.0.

Evidence: [original result](https://github.com/tokha-t/decik/blob/main/docs/validation/result.json),
[checks](https://github.com/tokha-t/decik/blob/main/docs/validation/checks.json),
[manifest and logs](https://github.com/tokha-t/decik/tree/main/docs/validation).
Public-data results do not establish hidden-data performance; official points depend on admitted entries and jury review.

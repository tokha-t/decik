# Submission: <team name>

## Links
- Repository, final commit: https://github.com/<owner>/<repo>/commit/<sha>
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link is in README.md):
  https://www.kaggle.com/code/<you>/<notebook>?scriptVersionId=<version>
- Demo video (up to 3 minutes): <link>
- Presentation: <link>

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| | | | | |

## Our kernels
| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `skc_nf4_dequant_kernel` | `solution/nf4.py` | bitsandbytes' NF4 dequantization before every 4-bit matmul, forward and backward (bit-identical) | <x> % |
| `skc_rmsnorm_fwd_kernel`, `skc_rmsnorm_bwd_kernel` | `solution/rmsnorm.py` | Qwen2RMSNorm: 6 fp32 ops each way | <x> % |
| `skc_head_stats_kernel` | `solution/head.py` | log_softmax / cross-entropy / entropy / argmax over the [tokens x 151,936] logits and their gradient | <x> % |
| `skc_rope_fwd_kernel`, `skc_rope_bwd_kernel` | `solution/attn.py` | fp32 casts, peft's LoRA add and the six RoPE ops on q and k | <x> % |

Ideas taken from other kernels, with links (we wrote the code ourselves):
- Liger Kernel, fused linear cross entropy: https://github.com/linkedin/Liger-Kernel
- Cut Cross-Entropy (Apple): https://arxiv.org/abs/2411.09009
- QLoRA / NF4 double quantization format: https://arxiv.org/abs/2305.14314 and bitsandbytes
- Padding-free training: https://huggingface.co/blog/packing-with-FA2
- Triton tutorials (layer norm, fused softmax): https://triton-lang.org/main/getting-started/tutorials/

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | <base> | <ours> | speedup <s>x |
| Peak GPU memory, MiB | <base> | <ours> | base/ours <m>x |
| Eval loss (public rows) | <base> | <ours> | <d> % vs base |

- [x] All 10 checks of `harness/check_rules.py` passed
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits

## Presentation covers
Where the base spends memory and time, which operations we changed and why, how each kernel works,
results against the base (speed, memory, quality), limits and next steps.

## Limitations
See README.md, "Ограничения и что дальше".

# Submission: QostBILLions

## Links

- Repository: [tokha-t/decik](https://github.com/tokha-t/decik).
- Measured v5.1 implementation: [`d9ab00419faa5d81ddf48fb35ec562284e5d62f9`](https://github.com/tokha-t/decik/commit/d9ab00419faa5d81ddf48fb35ec562284e5d62f9).
- [Public G7 notebook, saved version 364261503](https://www.kaggle.com/code/toktarkhan/gpt-v51-final-validation?scriptVersionId=364261503); `COMMIT` is the SHA above.
- Presentation: [editable PPTX](docs/qostbillions-v51-final-deck.pptx), [PDF](docs/qostbillions-v51-final-deck.pdf).
- [Video narration, target 2:55](docs/qostbillions-v51-video-script.md).
- Demo video, up to 3 minutes: **pending recording and link from the team**.
- [Final validation evidence](docs/validation/README.md).
- Previous validated v4: [saved G4 run](https://www.kaggle.com/code/toktarkhan/qostbillions-final-validation?scriptVersionId=364217337), [preserved evidence](docs/validation-v4/README.md).

Runtime identity is pinned to the measured implementation; publication/documentation commits are separate.

## Team (up to 4 people)

Team name: **QostBILLions**, confirmed by Toktarkhan from the team repository name.
Member names, roles and captain contact will be supplied by Toktarkhan at the end.

| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| Pending team confirmation | Pending | Pending | Pending | Pending |

## Our kernels

| Kernel (`@triton.jit` name) | File | What it replaces | Training-time share* |
|---|---|---|---:|
| `skc_nf4_dequant_kernel` | `solution/nf4.py` | NF4 double-quant dequantization → fp16 weights | 9.05% |
| `skc_rmsnorm_fwd_kernel` | `solution/rmsnorm.py` | RMSNorm forward in one launch | 1.12% |
| `skc_rmsnorm_bwd_kernel` | `solution/rmsnorm.py` | RMSNorm input gradient in one launch | 0.72% |
| `skc_rope_fwd_kernel` | `solution/attn.py` | LoRA addition, RoPE and fp16 Q/K layout | 0.37% |
| `skc_rope_bwd_kernel` | `solution/attn.py` | Q/K transform input gradients | 0.24% |
| `skc_keep_pack_kernel` | `solution/attn.py` | Bit packing of both LoRA dropout keep-masks | 0.29% |
| `skc_lora_inputs_kernel` | `solution/attn.py` | fp16 q/v LoRA inputs through dropout, forward and backward | 0.52% |
| `skc_lora_input_grad_kernel` | `solution/attn.py` | Combined q/v input gradient through dropout | 0.25% |
| `skc_head_stats_kernel` | `solution/head.py` | Chunked CE, entropy, argmax, rewind losses and logit gradient | 0.46% |

*Shares are kernel CUDA-event time / timed loop (steps 3+) in `solution_2`, the run with
the lowest combined share. The checker reports **13.0% combined** across nine original
kernels; the required minimum is 5%. Individual shares are rounded separately. `_keep` is an
inlined helper. Supplied SwiGLU example kernels are credited and excluded from this count/share.

The release removes padding within each batch, preserves dropout draws, keeps the embedding fp16
with fp32 residual output, and computes head loss/metrics/rewind in chunks. Attention backward
saves only q/k/v and recomputes per-row forward. LoRA/MLP inputs are rebuilt from norm inputs;
LoRA masks are bit-packed. In v5.1, `AddRMSNormFn` returns the post-attention residual and norm
output while rebuilding its input from the layer input plus fp16 attention output in backward;
the recomputed MLP uses the same source. The existing LoRA-input kernel is also used in forward.
MLP gate/up recomputation remains on. cuBLAS performs GEMMs. Original thread warmup overlaps model
setup inside measured time; rejected G5 subprocess code is absent. No ready-made kernel library is imported.

Ideas and references (implementation written in this repository):

- [Liger Kernel](https://github.com/linkedin/Liger-Kernel): chunked fused linear CE.
- [Apple, Cut Your Losses](https://arxiv.org/abs/2411.09009): avoiding full logits.
- [QLoRA](https://arxiv.org/abs/2305.14314) and [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes): NF4/double quantization.
- [Padding-free training](https://huggingface.co/blog/packing-with-FA2): per-row positions and attention.
- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/): reductions and fused kernels.
- Supplied [SwiGLU example](solution/swiglu.py): attributed and excluded from original kernels.

## Results on Kaggle T4

From the exact saved version's `runs/result.json` and `runs/checks.json`:

| | Base Soup 0.77.0 | Ours v5.1 | Ratio |
|---|---:|---:|---:|
| Time of `soup train`, s | 85.21 | 63.88 | speedup 1.3339× |
| Peak GPU memory, MiB | 14345 | 2035 | base/ours 7.0491× |
| Eval loss (public rows) | 2.04967 | 2.04964 | −0.001% vs base |

Two base and two solution runs, ordered base/solution/solution/base; medians shown. Fresh compiler
caches for every training run; compilation is included in wall time. One T4 is used in the T4 x2
environment, with defaults and no `SKC_` overrides. Evaluation: 256 public rows,
38109 tokens; training: 64 steps.

- [x] All 10 rule checks passed; `score.py` confirms all checks passed on this result.
- [x] 21 GPU numerical tests passed.
- [x] All 79 Soup regression tests passed with solution applied.
- [x] README includes the final measurement, reproduction and limits.

Result SHA-256: `9366fe839448718f404a08264e13887d98990ad6837402d35d82ad0bd74ddcf2`.

Solution source SHA-256: `bdf21a6adb90887482ad0facaa421dc0173e8ae075d8ba4678c4166d91fcaa42`.

Both hashes were checked against downloaded result bytes and exact candidate Git blobs.
[Manifest](docs/validation/manifest.json) records versions and SHA. The v4 manifest and bytes
remain in `docs/validation-v4/`; original local G4 materials and evidence are preserved.

## Presentation covers

Seven slides explain base memory costs, padding removal, recomputation, v5.1 norm-input
reconstruction, the chunked head, nine original kernels, measured speed/memory/quality,
correctness, limits and next steps. Narration reserves 7–10 seconds for the saved Kaggle outputs.

## Limitations and next steps

Public rows/seed 1/T4 are measured; hidden data and another seed may change time, memory and
numerics. Recompute adds GEMMs. Per-row SDPA, CPU launches, startup and full fp16 dequantized
weights remain. Saved tensors are about 13.4 KiB/token/layer, of which about 9 KiB are norm-input
sources (v4: 16 KiB total, 12 KiB norm inputs); these estimates are not full GPU peaks.
NF4 experiments and G5 workers were not adopted. Next work would reduce launches/per-row
attention, validated with fresh correctness and whole-process measurements. Official points
depend on other admitted entries and jury review.

## Remaining before platform submission

- Member names/roles and captain contact from Toktarkhan.
- Recorded demo video, at most 3 minutes, and its link.
- Team review and platform submission by Toktarkhan.

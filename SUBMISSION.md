# Submission: QostBILLions

## Links

- Repository: [tokha-t/decik](https://github.com/tokha-t/decik).
- Frozen implementation: [`5c1bbe560c08b0323eefeb843a0baf00e400e774`](https://github.com/tokha-t/decik/commit/5c1bbe560c08b0323eefeb843a0baf00e400e774).
- [Public Kaggle notebook, saved version 364217337](https://www.kaggle.com/code/toktarkhan/qostbillions-final-validation?scriptVersionId=364217337); `COMMIT` is the frozen SHA above.
- Presentation: [editable PPTX](docs/qostbillions-final-deck.pptx), [PDF](docs/qostbillions-final-deck.pdf).
- [Video narration, target 2:55](docs/qostbillions-video-script.md).
- Demo video, up to 3 minutes: **pending recording and link from the team**.
- [Raw validation evidence](docs/validation/README.md).

Later documentation commits preserve `solution/` byte-for-byte against the frozen implementation.

## Team (up to 4 people)

Team name: **QostBILLions**, confirmed by Toktarkhan from the team repository name.
Member names, roles and captain contact will be supplied by Toktarkhan at the end.

| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| Pending team confirmation | Pending | Pending | Pending | Pending |

## Our kernels

| Kernel (`@triton.jit` name) | File | What it replaces | Training-time share* |
|---|---|---|---:|
| `skc_nf4_dequant_kernel` | `solution/nf4.py` | NF4 dequantization with double-quant scales → fp16 weights | 8.92% |
| `skc_rmsnorm_fwd_kernel` | `solution/rmsnorm.py` | RMSNorm forward in one launch | 1.39% |
| `skc_rmsnorm_bwd_kernel` | `solution/rmsnorm.py` | RMSNorm input gradient in one launch | 0.83% |
| `skc_rope_fwd_kernel` | `solution/attn.py` | LoRA addition, RoPE and fp16 Q/K layout | 0.43% |
| `skc_rope_bwd_kernel` | `solution/attn.py` | Q/K transform input gradients | 0.25% |
| `skc_keep_pack_kernel` | `solution/attn.py` | Bit packing of both LoRA dropout keep-masks | 0.31% |
| `skc_lora_inputs_kernel` | `solution/attn.py` | Rebuilt fp16 q/v LoRA inputs through dropout | 0.31% |
| `skc_lora_input_grad_kernel` | `solution/attn.py` | Combined q/v input gradient through dropout | 0.33% |
| `skc_head_stats_kernel` | `solution/head.py` | Chunked CE, entropy, argmax, rewind losses and logit gradient | 0.49% |

*Shares are measured kernel time / timed training loop (steps 3+) in `solution_1`,
the run with the lowest combined share. The rule checker reports **13.3% combined** across
nine own kernels, above the required 5%; it uses the minimum of the two runs. Individual shares
are rounded separately. `_keep` is an inlined helper. Supplied SwiGLU example kernels are
credited and excluded from this count and share.

The release also removes padding within each batch, keeps the embedding in fp16 with fp32
residual output, and computes head loss/metrics/rewind by chunks. Attention backward saves
only q/k/v and recomputes the per-row forward. LoRA and MLP inputs are rebuilt from saved norm
inputs; LoRA masks are bit-packed. MLP gate/up are recomputed by default. cuBLAS performs GEMMs.
Compilation overlaps model setup inside the measured process. No ready-made kernel library is imported.

Ideas and references (implementation written in this repository):

- [Liger Kernel](https://github.com/linkedin/Liger-Kernel): fused linear CE and chunked forward gradients.
- [Apple, Cut Your Losses](https://arxiv.org/abs/2411.09009): avoiding full logits.
- [QLoRA](https://arxiv.org/abs/2305.14314) and [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes): NF4 and double quantization.
- [Padding-free training](https://huggingface.co/blog/packing-with-FA2): per-row positions and attention.
- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/): reductions and fused kernels.
- Supplied [SwiGLU example](solution/swiglu.py): used with attribution, excluded from our original kernels.

## Results on Kaggle T4

From the exact saved notebook's `runs/result.json` and `runs/checks.json`:

| | Base Soup 0.77.0 | Ours | Ratio |
|---|---:|---:|---:|
| Time of `soup train`, s | 88.35 | 65.80 | speedup 1.3427× |
| Peak GPU memory, MiB | 14345 | 2155 | base/ours 6.6566× |
| Eval loss (public rows) | 2.04966 | 2.04968 | +0.001% vs base |

Two base and two solution runs, ordered base/solution/solution/base; medians shown.
Compiler caches are cold for every training run, and compilation is included in wall time.
The unchanged harness trains on one T4 in the T4 x2 environment, with release defaults and no
`SKC_` overrides. Public evaluation: 256 rows, 38109 tokens; training: 64 steps.

- [x] All 10 checks of `harness/check_rules.py` passed; `score.py` confirms all checks passed on this result.
- [x] 20 GPU tests of solution outputs/gradients passed.
- [x] All 79 Soup regression tests passed with `apply()` enabled.
- [x] README describes the implementation, reproduction, final results and limits.

Result SHA-256: `7434230cff6fa598cada142143d6ded954a4c328531df6c2c5e89d3d030265f5`.

Solution source SHA-256: `d6ef76a142b4ffbbaaad584201fd25469136f15c69ca37106b062b2c9283e6ab`.

Both hashes have been verified against the downloaded result bytes and the exact frozen Git blobs.
[Manifest](docs/validation/manifest.json) records package versions and the frozen SHA.

## Presentation covers

Seven slides explain the base's memory costs, padding removal, recomputation, chunked head,
all nine original kernels, speed/memory/quality measurements, correctness, limits and next steps.
The narration reserves 7–10 seconds to show the saved Kaggle outputs and all-checks-passed line.

## Limitations and next steps

These measurements cover public rows, seed 1 and Kaggle T4. Hidden data and another seed can
change speed, memory and numerics. Recompute trades extra GEMMs for lower memory. Per-row SDPA
calls, CPU launch overhead, startup and the full fp16 weight written by NF4 dequantization remain.
NF4 word-load/LUT prototypes were correct but slower; the original 1024-byte/4-warp kernel stays
in the release. See [the measured experiment report](docs/nf4-experiments.md).
Next work would reduce launch overhead and per-row attention calls, with fresh correctness and
full-process measurements. Final official points depend on the best admitted entries and jury review.

## Remaining before platform submission

- Member names/roles and captain contact from Toktarkhan.
- Recorded demo video, at most 3 minutes, and its link.
- Team review and final platform submission by Toktarkhan.

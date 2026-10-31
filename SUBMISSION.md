# Soup Kernel Challenge — Decik candidate

## Release status

The final SHA and public final notebook will be filled after Claude freezes the integrated release at
19:30 Asia/Almaty. The measurements below describe a verified development checkpoint, not M5.

## Development source and validation

- Repository: https://github.com/tokha-t/decik
- Development implementation commit: https://github.com/tokha-t/decik/commit/233282f5a9c0290449847c3e1455f61e0da3115f
- Development validation notebook: https://www.kaggle.com/code/toktarkhan/decik-validated (version 1).
- The notebook pins that implementation commit and uses its defaults, without `SKC_` overrides.
- The notebook is currently private. Public saved notebook, team details, presentation and
  video are still required before a platform submission.

## Implementation

Six original Triton kernels run in training:

| Kernel | File | Operation |
|---|---|---|
| `skc_nf4_dequant_kernel` | `solution/nf4.py` | NF4 dequantization for frozen 4-bit projections |
| `skc_rmsnorm_fwd_kernel` | `solution/rmsnorm.py` | RMSNorm forward |
| `skc_rmsnorm_bwd_kernel` | `solution/rmsnorm.py` | RMSNorm input gradient |
| `skc_head_stats_kernel` | `solution/head.py` | Chunked CE, entropy, accuracy and logit gradients |
| `skc_rope_fwd_kernel` | `solution/attn.py` | LoRA addition, RoPE and fp16 Q/K layout |
| `skc_rope_bwd_kernel` | `solution/attn.py` | Q/K transform input gradients |

Other changes: fp16 MLP intermediates, optional gate/up recomputation in backward (enabled in this checkpoint), retained fp16 embedding
with fp32 residual output, padding removal within each batch, identical LoRA dropout draws,
and compilation overlapping model setup inside the measured process. cuBLAS computes matrix
products. The starter SwiGLU kernels are credited as the supplied example and do not count as ours.

Design references and kernel details are in `README.md`. No ready-made kernel package is imported.

## Correctness and evidence

`python -m pytest solution -q` covers the NF4 output, RMSNorm output/gradient, SwiGLU,
head loss/metrics/gradients, fused Q/K output/gradients and recomputed MLP output/gradients.
The unchanged rule checker also runs all 79 Soup regression tests with `apply()` enabled.

The validation notebook runs base, solution, solution, base with fresh compiler caches.
It exports `validation/result.json`, `validation/checks.json` and `validation/manifest.json`.
The checker binds its verdict to the result hash and the solution source hash.

## Remaining submission fields

- Team name, members and captain contact: pending confirmation.
- Public notebook: pending publication of the validated saved version.
- Demo video (up to 3 minutes): pending.
- Presentation: pending.
- Final platform submission: pending.

## Limits

The reported measurements use public rows on Kaggle T4. Hidden rows and another seed can change
performance and peak memory. Recompute trades speed for memory. Claude owns the integrated release default; the
checkpoint measured here has recomputation enabled. Official speed/memory points depend on
the best admitted entries. Jury execution and source review determine final admission.

## Verified development result (233282f; final result still pending)

| | Same-session base | Checkpoint | Ratio |
|---|---:|---:|---:|
| soup train wall, s | 80.30 | 58.01 | 1.3842x speedup |
| GPU peak, MiB | 14345 | 3047 | 4.7079x memory |
| Eval loss | 2.04967 | 2.04965 | -0.001% |

Two runs per side; all 10 checks passed; own kernels 13.7%; 23 GPU tests and 79 Soup tests passed.

## Presentation drafts

- [Editable slides](docs/decik-deck-draft.pptx)
- [PDF slides](docs/decik-deck-draft.pdf)
- [Video narration, target 2:55](docs/video-script-draft.md)

These drafts label the development checkpoint. Update them from the frozen release result before
recording or submitting.

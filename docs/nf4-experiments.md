# NF4 experiments — measured, not included in the frozen release

The frozen release retains the original 1,024-byte / 4-warp NF4 kernel. No experimental word-load or
local-lookup code was merged into `main`.

## Component comparison

Four model shapes: 1536×1536, 256×1536, 8960×1536 and 1536×8960. Timing uses three interleaved rounds;
the table sums median wrapper times with weights 2, 2, 2, 1, corresponding to projection occurrences.
Output allocation and Python dispatch are included in wrapper measurements. These component times
exclude compilation and are not training scores.

| Variant | Bytes / warps | Weighted wrapper, ms |
|---|---:|---:|
| Original, reference | 1024 / 4 | 0.654923 |
| Original, best component setting | 4096 / 4 | 0.637113 |
| Word loads, best setting | 512 / 4 | 0.744097 |
| Local 16-value LUT, best setting | 4096 / 4 | 0.645201 |

The word kernel loads 32-bit words and reconstructs one scale per quantization block. The LUT mode
loads the 16 values once and gathers from that vector. Both are bit-identical to bitsandbytes, but
neither improved on the original kernel at its best launch setting.

Source snapshots: word `659b935` (tag `experiment/nf4-word-659b935`), LUT `96d2dd4` on `gpt/nf4`.
CPU interpreter checks, explicit sm75 compilation, GPU parity and Soup tests passed. The LUT snapshot
passed 30 GPU tests and 79 Soup tests. CPU versions were torch 2.11 CPU, NumPy 2.2.6, Triton 3.6.0;
the incompatible preinstalled optional torchao 0.10 package was removed before CPU preflight.

## Full training A/B

Stable source `73ad7f773fb7c8a9f1591ddb7fd97f28010f3fd2`, recomputation off, both settings measured
in one Kaggle session. Each setting runs the unchanged harness with two base and two solution runs,
then all ten checks and `score.py`.

| | Original 1024 / 4 | Original 4096 / 4 |
|---|---:|---:|
| Matched base wall, s | 85.77 | 88.25 |
| Solution wall, s | 57.16 | 59.70 |
| Speedup | 1.5005× | 1.4782× |
| Peak memory, MiB | 4307 | 4307 |
| Memory ratio | 3.3306× | 3.3306× |
| Solution eval loss | 2.04965 | 2.04969 |
| Eval delta vs matched base | 0.000% | +0.001% |
| All checks | 10/10 | 10/10 |

The 4096-byte configuration did not improve full training: median loops were 34.30 versus 33.52 s,
and measured NF4 GPU time increased from about 3.01 to 3.47 s. Session drift is visible in the base;
the matched ratios also favor 1024. The original setting remains in the release.

[Component notebook](https://www.kaggle.com/code/toktarkhan/gpt-nf4-microbench), versions 2 and 3;
[full A/B notebook](https://www.kaggle.com/code/toktarkhan/gpt-nf4-lab), version 1. These development
notebooks are private. Result hashes and exact source hashes were checked locally against each
`checks.json`.

## Fused GEMM feasibility

The pinned Triton 3.6 compiler falls back to scalar FMA for ordinary `tl.dot` below compute capability
80, including T4 sm75. A proposed fused NF4 GEMM therefore needs a different tensor-core path and
fresh numerical and performance evidence; it was not added to this release.
[Triton 3.6 compiler gate](https://github.com/triton-lang/triton/blob/v3.6.0/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp#L344-L350).

# G6 — GPU-idle diagnosis on frozen v4

Diagnostic source `7ebed815c9cf3bbddc510474a6298b3254128802`; `solution/` byte-identical to
frozen `5c1bbe560c08b0323eefeb843a0baf00e400e774`.
[Kaggle saved version 364227255](https://www.kaggle.com/code/toktarkhan/gpt-v4-idle-trace?scriptVersionId=364227255)
(private development notebook). Both runs completed all 64 steps; steps 11–13 captured.

## Timing and profiler overhead

| Per-step mean, ms | With Python stacks | No-stack control |
|---|---:|---:|
| CPU step range | 730.292 | 528.804 |
| GPU busy (all streams union) | 239.850 | 245.731 |
| All GPU idle | 490.442 | 283.073 |
| Idle gaps strictly >0.2 ms | 292.856 | 91.738 |

The two runs trace the same three batches. Stack capture adds about
201.5 ms/step here. Unprofiled medians in those runs were
453.5 and 446.3 ms (across other, differently sized batches). Therefore these are diagnostic
observations, not unprofiled recoverable savings. The official TritonWatch CUDA-event
instrumentation is retained. The three captured batches differ in real-token count; their
GPU-busy average should not be substituted for a whole-training profile average.

## Top ten overlaps during idle gaps >0.2 ms, with stacks

Exclusive attribution: CUDA runtime > CPU op > Python, choosing the shortest enclosing
span on the training or CUDA-launch/autograd threads. Totals across all causes equal the
gap durations; the ten entries below are only the largest entries.

| Rank | CPU operation / Python function | ms/step overlapping idle |
|---|---|---:|
| 1 | `NF4Linear` | 26.342 |
| 2 | `LoRAPairFromNormBackward` | 23.567 |
| 3 | `torch/nn/modules/module.py(2882): named_modules` | 18.554 |
| 4 | `NF4LinearBackward` | 18.084 |
| 5 | `PackedSDPABackward` | 15.761 |
| 6 | `RecomputedMLPBackward` | 14.228 |
| 7 | `RMSNormFn` | 10.473 |
| 8 | `FusedQKRope` | 9.455 |
| 9 | `aten::empty` | 8.660 |
| 10 | `FusedQKRopeBackward` | 7.792 |

## No-stack control

| CPU operation / missing instrumentation | ms/step overlapping idle |
|---|---:|
| `<no recorded CPU event>` | 30.616 |
| `LoRAPairFromNormBackward` | 5.985 |
| `FusedQKRope` | 5.195 |
| `PackedSDPABackward` | 5.147 |
| `NF4Linear` | 4.925 |
| `FusedQKRopeBackward` | 4.753 |
| `NF4LinearBackward` | 4.377 |
| `aten::empty` | 3.992 |
| `cuLaunchKernelEx` | 2.223 |
| `RecomputedMLPBackward` | 2.162 |

The uninstrumented Python time remains explicitly unattributed in this control.
The per-gap files include every qualifying interval and its three largest overlaps.
Runtime details, including `cudaStreamSynchronize` and `cudaMemcpyAsync`, are retained in
`runtime_overlaps` even when they do not enter the top ten.

## Findings and handoff

1. CPU dispatch/autograd around q/v LoRA, NF4, RoPE, and per-row SDPA appears repeatedly in
   both views. Reducing Python/launches is a better candidate than removing synchronization
   based on this trace. Claude owns `attn.py`/`varlen.py`; its v5.2 batched attention work
   targets one of these sources. No solution change was made for G6.
2. Main-thread-only attribution misses autograd work: `pt_autograd_0` issues much of the
   backward CPU/CUDA activity. The final analyzer includes CPU threads in the same process
   that launch CUDA kernels, while ignoring unrelated background threads.
3. `named_modules`/parameter walking is visible with stacks (including Trainer FLOP bookkeeping),
   but stack profiling strongly magnifies Python overhead. Do not change training accounting
   or promise this time as savings without a separate unprofiled test.
4. In the final no-stack attribution, explicit device/stream synchronization overlaps only
   about 0.026 ms/step of the >0.2-ms gaps; memcpy runtime overlaps about
   0.039 ms/step. Synchronization may wait while GPU work is active;
   this number is only its overlap with intervals where no GPU work is recorded.
5. G5 cold-start work is independent: this steady-state trace starts after step10. Warm-up
   completion and first-step timings need their own experiment.

## Evidence integrity and corrections

All original trace/log/metric bytes match the notebook manifest's SHA-256 values. The original
notebook analysis counted GPU copies of step labels as additional steps and omitted the
autograd thread. It is superseded by `analysis-corrected.json`, recomputed locally with the
corrected analyzer; raw traces remain unchanged. Six synthetic tests cover stream union,
CPU/GPU label distinction, autograd threads, exclusive attribution, thresholds and normalization.

Raw trace SHA-256:
- with-stack: `81d960b9abb83078b4ced1bdebfe188beae364992bd97487b1d5c6a31c16ad1e`
- no-stack: `0f7233dfd2692fc2a7c287bfde957618e1aa155c975d2a8b1115243d6f8b0b3a`

Run `dev/analyze_trace.py TRACE --out REPORT` to reproduce. The 22-MiB original evidence ZIP
and raw compressed traces are retained locally; the corrected analyses and original metrics
are committed alongside this report, without the bulky traces.

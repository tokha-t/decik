# G5 cold-start experiment — complete; retain the original thread

Candidate [`dfe0ad6`](https://github.com/tokha-t/decik/commit/dfe0ad612265c02d7ded3bd8a99037f943fa87a3)
on `gpt/warmup` was tested against frozen v4 `5c1bbe5` in the same Kaggle T4 session.
[Development notebook](https://www.kaggle.com/code/toktarkhan/gpt-warmup-lab/log).
The original thread remains the release default; no candidate runtime code was merged.

## Full-process comparison

The order was thread / one worker / two workers / two workers / one worker / thread.
Each uses the unchanged official harness, its allocator and GPU visibility, a fresh compiler
cache, all 64 steps, seed1, and public evaluation. The only candidate runtime change is
`solution/warmup.py`; protected files are unchanged. The same-session base was86.99s,
14345MiB, eval2.04966. It is a one-repeat reference, not a new G4 validation.

| Configuration | Individual wall times, s | Median wall, s | Median loop, s | Peak, MiB | Individual eval losses |
|---|---|---:|---:|---:|---|
| Original thread | 59.37 / 60.84 | 60.105 | 36.750 | 2155 | 2.04965 / 2.04968 |
| One worker | 59.43 / 59.52 | 59.475 | 36.655 | 2155 | 2.04966 / 2.04962 |
| Two workers | 60.79 / 61.19 | 60.990 | 36.710 | 2155 | 2.04966 / 2.04967 |

One worker has a nominal0.630s median improvement, but the two thread endpoints differ
by1.47s and the first one-worker run is0.06s slower than its adjacent thread baseline.
Two repeats do not establish a clear gain of at least0.5s. Two workers are0.885s slower
at the median. Median official `apply_s`: thread8.565s, one8.150s, two9.575s; median first
steps remain about4s (4.034 /3.957 /3.972s).

Eval values vary slightly in all configurations. They remain within the existing numerical
tolerance, but they are not identical as the G5 adoption condition requires. This is not
evidence of a quality regression. **Do not adopt either process mode.** No further experiment
is needed to retain the validated thread under this time box.

## Warmup/first-step probe

| Mode | Warmup start→ready, s | Ready before first step, s | Parent warmed entries |
|---|---:|---:|---:|
| Thread | 3.868 | 6.883 | 11 |
| One worker | 7.239 | 4.016 | 11 |
| Two workers | 8.638 | 3.566 | 11 |

There are no compile/process errors or first-use wait events in these three probes.
Workers exit and are reaped, then the parent loads/warms all eleven entries. The parent
phase after workers is1.289s with one worker and1.303s with two. Workers inherit the run's
cache/GPU/allocator environment and use fresh Python interpreters rather than forking an
initialized CUDA context. Failure and timeout fall back to in-process compilation.

The ordinary thread is already ready before the first step in the diagnostic. This does
not establish an official-run startup cause: `dev/warmup_probe.py` imports/accesses
`transformers.Trainer` before invoking the official runner, changing its import sequence.
Its apply timings cannot be used as the official A/B result or proof about GIL contention.
The full-process table above comes from the unmodified official runner, without that probe.

## Validation and scope

- Five lifecycle tests pass: partition coverage, failure fallback, environment/cache inheritance,
  timeout cleanup, completion signaling and idempotent start.
- All six CPU interpreter suites pass; all nine original kernels compile for sm75.
- Both process settings pass20 GPU numerical tests and79 Soup tests.
- The warmup-file overlay on v5.1 `d9ab004` passes21 GPU numerical tests. Those tests do not
  call `warmup.start()`, so this is numerical compatibility, not a v5.1 subprocess lifecycle
  or full-training test. Real worker behavior is evidenced by the v4 training runs and probes.
- v4/v5.1 have the same eleven warmup entries (nine original plus two supplied SwiGLU kernels).
- The experiment is not full G4/G7 validation: it does not replace the public release result,
  run all ten rules on an adopted candidate, or justify updating the final slides/metrics.

CPU preflight: [saved notebook](https://www.kaggle.com/code/toktarkhan/gpt-warmup-cpu?scriptVersionId=364238415),
[logs and provenance](g5-cpu-validation/provenance.json). Fresh test subprocesses used NumPy2.2.6,
torch2.11 CPU and Triton3.6. The notebook's already-imported NumPy still printed2.1.3; this
distinction is recorded in the CPU provenance. Optional incompatible torchao was removed only
from the CPU preflight environment, as in earlier CPU gates.

## Evidence

All52 files named in the original [manifest](g5-validation/manifest.json) were verified
against their SHA-256 values and the original evidence ZIP. All seven run summaries were
checked against their original result JSON; all completed64 steps with seed1 and no package
changes. [Summary](g5-validation/summary.json), [derived statistics](g5-validation/derived-statistics.json),
[GPU logs](g5-validation/gpu-1.log), [second-mode GPU logs](g5-validation/gpu-2.log),
[v5.1 overlay log](g5-validation/v51-overlay-gpu.log), and complete per-run JSON/logs are retained.

The archive was present in the workspace on resume at19:14 Asia/Almaty. Kaggle connector
calls still returned access/authentication errors, so saved-version metadata could not be
rechecked. This report relies on the hash-verified complete archive, not a live status claim.
The last confirmed submission was saved version2; no duplicate GPU run was submitted.

G6 is complete in [its report](g6-idle-diagnosis.md). G5 is complete with no adoption.
Claude's G7 handoff for v5.1 remains separate: it needs full public validation before any
release change and before the section12 hard stop21:00 Asia/Almaty. The validated v4 release
and the official team repository remain unchanged.

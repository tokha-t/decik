# G5 cold-start experiment — awaiting GPU measurement

Candidate source: `dfe0ad612265c02d7ded3bd8a99037f943fa87a3` on `gpt/warmup`.
The validated v4 runtime on main is unchanged.

The experimental `SKC_WARMUP_PROCESSES=1` or `2` starts fresh Python interpreters to compile
exactly the current warm-up kernel specializations into the measured run's empty
`TRITON_CACHE_DIR`. The parent then loads those binaries into its JIT cache. Workers inherit
the harness GPU/allocator/cache environment, are reaped before warmup wait returns, and fall
back to in-process compilation on failure. There is no fork of an initialized CUDA context.
The ordinary thread remains the default. v4 and v5.1 use the same eleven warmup entries
(nine original kernels plus the two supplied SwiGLU kernels).

Passed gates:

- Five local lifecycle checks: shard coverage, child failure fallback, environment/cache
  inheritance, timeout cleanup, completion signaling and idempotent start.
- All six provided CPU interpreter suites: kernels, model, trainer, attention, bnb, recompute.
- Explicit sm75 compilation for all nine original kernels.
- [CPU preflight notebook, saved version](https://www.kaggle.com/code/toktarkhan/gpt-warmup-cpu?scriptVersionId=364238415).
- [Downloaded CPU logs and provenance](g5-cpu-validation/provenance.json).

The CPU notebook installed NumPy 2.2.6 before launching each test in a fresh subprocess.
Its own already-imported NumPy module still printed 2.1.3; this distinction is recorded in
provenance. Torch was 2.11.0+cpu and Triton 3.6.0. The optional incompatible preinstalled
torchao package was removed for this CPU-only preflight, as in earlier CPU gates.

Pending GPU experiment:

[GPT warmup lab](https://www.kaggle.com/code/toktarkhan/gpt-warmup-lab/log), saved version2.
Version1 was canceled and cancellation acknowledged before the identical retry. Exactly
one GPT GPU batch is queued; the CPU job has finished and has enable_gpu=false.

The notebook compares original frozen-v4 warmup against one and two compiler workers,
ordered thread/one/two/two/one/thread. Every training run uses the unchanged official harness
with fresh compiler caches and evaluation. It includes GPU and Soup tests, a separate
ready-versus-first-step timing probe, a check of all eleven warmed entries, and v5.1 GPU
compatibility with only the warmup file overlaid. Profile/probe timings are not A/B scores.

Adoption gate: same-session wall time decreases by at least0.5s and eval loss is identical;
retain the current thread otherwise. A release change would still need full G7 validation.
No GPU timing claim or adoption decision is supported yet.

Operational note: Kaggle's SDK defines `KernelWorkerStatus.QUEUED=0`, the default omitted
from serialized responses. Thus a successful empty `{}` session-status response denotes
queued. Explicit saved-version labels are `v1`, `v2`, etc. (bare numeric labels return404).
Sources: [Kaggle SDK](https://pypi.org/project/kagglesdk/),
[official CLI version-label handling](https://github.com/Kaggle/kaggle-api/blob/main/src/kaggle/api/kaggle_api_extended.py).

Section12 hard stop remains21:00 Asia/Almaty; v4 remains final unless a replacement passes
its required measurements and validation in time. The team repository remains untouched.

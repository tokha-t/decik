# G6 diagnostic trace

Run on the same pinned Kaggle environment as the final notebook:

```bash
python task/prepare_data.py
python dev/trace_run.py --solution --out trace_runs/with-stack
python dev/analyze_trace.py trace_runs/with-stack/trace.json.gz --out trace_runs/with-stack/analysis.json
```

`kaggle_gpt_trace.ipynb` pins the source and also runs a separate `--no-stack` control.
The diagnostic runner calls `run_task.make_config`, then starts the child with the official
allocator, GPU visibility, offline environment and empty per-run compiler caches. The child
uses `harness._train.TritonWatch`, the same Soup command, and the same synchronized step
boundaries. No protected harness file or `solution/` file is changed.

The model still trains all 64 steps. Steps 11, 12 and 13 are captured with CPU + CUDA profiling
and `with_stack=True`. The raw Chrome trace is gzip-compressed after capture. Profiling and
trace export affect this diagnostic's wall time; it is not a submission benchmark.

The analyzer unions kernel, memcpy and memset intervals across GPU streams. It reports all
idle time, and lists the requested gaps strictly greater than 0.2 ms within each traced step.
It splits those gaps at CPU event boundaries on the training thread, choosing a CUDA runtime
event first, then the innermost CPU operation, then the innermost Python function. This gives
one exclusive attribution per interval: totals sum to the selected GPU idle gaps. Python
leaf callers and operation/caller pairs provide separate views of the same time; do not add
them to the primary attribution totals. Missing events remain explicitly unattributed.

An overlap is evidence of what the CPU was doing while the GPU had no recorded work. It
does not by itself prove a cause, establish an available speedup, or show that another
thread/process was idle. Stack profiling and the official per-Triton-launch CUDA-event
watcher both add overhead. Compare traced and unprofiled step timings and the no-stack
control before proposing a fix. Logging between step callback ranges is outside the
per-step attribution window.

For aggregate attribution of gaps smaller than 0.2 ms too:

```bash
python dev/analyze_trace.py trace_runs/with-stack/trace.json.gz --min-gap-ms 0 --out all-idle.json
```

Synthetic checks cover overlapping GPU streams, exclusive nested CPU attribution, threshold
boundaries, normalization across steps and exclusion of unrelated CPU threads:

```bash
python -m unittest discover -s dev -p test_analyze_trace.py -v
```

"""Diagnostic Soup run with the official task and a short CPU/CUDA Chrome trace.

This does not replace harness/run_task.py or produce submission scores. The parent uses
run_task.make_config and common.run_logged; the child follows harness/_train.py, retaining
its Triton launch/event watcher and synchronized step boundaries. solution/ is untouched.
"""

import argparse
import gzip
import json
import os
from pathlib import Path
import shutil
import statistics
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO))


def child(args):
    from common import ALLOC_CONF, write_json
    from _train import TRAINER_ARGS, TIMED_FROM_STEP, TritonWatch, plain

    sys.argv = ["soup", "train", "--config", args.config, "--yes"]
    sys.modules["unsloth"] = None
    sys.modules["unsloth_zoo"] = None
    import torch
    torch.empty(1, device="cuda")
    import transformers

    watch = TritonWatch(torch)
    apply_s = None
    if args.solution:
        applying = time.perf_counter()
        import solution
        solution.apply()
        apply_s = time.perf_counter() - applying

    from torch.profiler import ProfilerActivity, profile, record_function
    prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                   with_stack=not args.no_stack, record_shapes=False,
                   profile_memory=False)
    marks = {"steps": [], "trainer": None}
    trace_path = Path(args.out) / "trace.json"
    tracing = False
    exported = False
    active_range = None

    def export():
        nonlocal tracing, exported
        if tracing:
            prof.stop()
            tracing = False
            prof.export_chrome_trace(str(trace_path))
            with trace_path.open("rb") as src, gzip.open(str(trace_path) + ".gz", "wb") as dst:
                shutil.copyfileobj(src, dst)
            trace_path.unlink()
            exported = True
            print("TRACE_EXPORTED", str(trace_path) + ".gz", flush=True)

    class Probe(transformers.TrainerCallback):
        def on_step_begin(self, train_args, state, control, **kwargs):
            nonlocal tracing, active_range
            step = state.global_step + 1
            if step == args.start_step:
                prof.start()
                tracing = True
            elif step == args.start_step + args.active_steps:
                export()
            torch.cuda.synchronize()
            self.step_start = time.perf_counter()
            watch.timing = len(marks["steps"]) >= TIMED_FROM_STEP
            if tracing:
                active_range = record_function(f"SKC_TRACE_STEP_{step}")
                active_range.__enter__()

        def on_step_end(self, train_args, state, control, **kwargs):
            nonlocal active_range
            torch.cuda.synchronize()
            watch.timing = False
            elapsed = time.perf_counter() - self.step_start
            if active_range is not None:
                active_range.__exit__(None, None, None)
                active_range = None
            marks["steps"].append({"step": state.global_step, "seconds": elapsed,
                                   "profiled": tracing})

    original_init = transformers.Trainer.__init__

    def init(self, *a, **kw):
        original_init(self, *a, **kw)
        self.add_callback(Probe())
        marks["trainer"] = {"class": type(self).__name__,
                            **{name: plain(getattr(self.args, name, None)) for name in TRAINER_ARGS}}

    transformers.Trainer.__init__ = init
    code = 0
    try:
        from soup_cli.cli import run
        run()
    except SystemExit as exc:
        code = 0 if exc.code is None else (exc.code if isinstance(exc.code, int) else 1)
    finally:
        export()
    regular = [x["seconds"] for x in marks["steps"]
               if not x["profiled"] and x["step"] > 2]
    profiled = [x["seconds"] for x in marks["steps"] if x["profiled"]]
    metrics = {"exit_code": code, "solution": args.solution, "apply_s": apply_s,
               "steps": marks["steps"], "trainer": marks["trainer"],
               "unprofiled_median_step_s": statistics.median(regular) if regular else None,
               "profiled_median_step_s": statistics.median(profiled) if profiled else None,
               "with_stack": not args.no_stack, "trace_exported": exported,
               "own_kernel_ms": watch.own_ms(), "watch_error": watch.error,
               "device": torch.cuda.get_device_name(0),
               "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
               "alloc_conf_expected": ALLOC_CONF}
    write_json(Path(args.out) / "metrics.json", metrics)
    if code:
        raise SystemExit(code)
    assert exported and len(profiled) == args.active_steps, "Trace did not capture requested steps"
    print("TRACE_METRICS", json.dumps({k:v for k,v in metrics.items() if k not in ("steps", "trainer", "own_kernel_ms")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--solution", action="store_true")
    parser.add_argument("--out", type=Path, default=REPO / "trace_runs/v4")
    parser.add_argument("--train-file", type=Path, default=REPO / "data/train.jsonl")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--start-step", type=int, default=11)
    parser.add_argument("--active-steps", type=int, default=3)
    parser.add_argument("--no-stack", action="store_true", help="Optional profiler-overhead control")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--config", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.child:
        child(args)
        return
    from common import OFFLINE_ENV, run_logged, write_json
    from run_task import make_config, warm_up
    warm_up(args.out)
    config = make_config(args.out, args.train_file.resolve(), args.seed)
    for name in ("triton_cache", "inductor_cache"):
        shutil.rmtree(args.out / name, ignore_errors=True)
    command = [sys.executable, str(Path(__file__).resolve()), "--child", "--config", str(config),
               "--out", str(args.out), "--start-step", str(args.start_step),
               "--active-steps", str(args.active_steps)]
    if args.solution:
        command.append("--solution")
    if args.no_stack:
        command.append("--no-stack")
    env = dict(OFFLINE_ENV, SOUP_DB_PATH=str(args.out / "experiments.db"),
               TRITON_CACHE_DIR=str(args.out / "triton_cache"),
               TORCHINDUCTOR_CACHE_DIR=str(args.out / "inductor_cache"))
    code, wall, peak, start = run_logged(command, args.out / "train.log", 1200, env)
    write_json(args.out / "run.json", {"exit_code": code, "diagnostic_only": True,
               "wall_s": wall, "gpu_peak_mib": peak, "gpu_start_mib": start,
               "source_commit": __import__("subprocess").check_output(
                   ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()})
    raise SystemExit(code)


if __name__ == "__main__":
    main()

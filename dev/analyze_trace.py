"""Attribute GPU-idle gaps in a trace_run.py Chrome trace without double counting.

Intervals from all GPU streams are merged first. Gaps > --min-gap-ms within each
SKC_TRACE_STEP range are split at CPU event boundaries on the training and CUDA-launch
threads (including the autograd worker).
The most specific CUDA runtime event wins, then the innermost CPU op, then Python.
Python leaf callers are reported separately. Overlap is attribution, not proof that an
operation caused the stall. Profiler/stack and official watcher overhead remain present.
"""

import argparse
from collections import Counter
import gzip
import heapq
import json
from pathlib import Path
import re


def merged(intervals):
    result = []
    for lo, hi in sorted(intervals):
        if hi <= lo:
            continue
        if result and lo <= result[-1][1]:
            result[-1][1] = max(hi, result[-1][1])
        else:
            result.append([lo, hi])
    return result


def gpu_event(event):
    categories = set(str(event.get("cat", "")).split(","))
    return bool(categories & {"kernel", "gpu_memcpy", "gpu_memset"})


def cpu_kind(event):
    cat = str(event.get("cat", ""))
    name = event.get("name", "")
    if cat in ("cuda_runtime", "cuda_driver"):
        return 0
    if cat == "cpu_op":
        return 1
    if cat == "python_function":
        return 2
    return None


def analyze(events, min_gap_ms=0.2):
    spans = [e for e in events if e.get("ph") == "X" and e.get("dur", 0) > 0]
    # Kineto copies user ranges onto CUDA streams as gpu_user_annotation. Those
    # are labels, not extra training steps (nor GPU kernels).
    steps = sorted([e for e in spans if e.get("cat") == "user_annotation"
                    and re.fullmatch(r"SKC_TRACE_STEP_\d+", e.get("name", ""))],
                   key=lambda e:e["ts"])
    if not steps:
        raise ValueError("No SKC_TRACE_STEP ranges in trace")
    if len({e["name"] for e in steps}) != len(steps):
        raise ValueError("Duplicate CPU step labels; analyze one trace capture at a time")
    gpu = [e for e in spans if gpu_event(e)]
    cpu = [e for e in spans if cpu_kind(e) is not None]
    launch_threads = {(e.get("pid"), str(e.get("tid"))) for e in cpu
                      if cpu_kind(e) == 0 and "LaunchKernel" in e.get("name", "")}
    causes, python, pairs = Counter(), Counter(), Counter()
    gaps = []
    step_rows = []
    for step in steps:
        lo, hi = step["ts"], step["ts"] + step["dur"]
        busy = merged((max(lo,e["ts"]), min(hi,e["ts"]+e["dur"])) for e in gpu
                      if e["ts"] < hi and e["ts"]+e["dur"] > lo)
        idle = []
        cursor = lo
        for start,end in busy + [[hi,hi]]:
            if start - cursor > min_gap_ms * 1000:
                idle.append([cursor,start])
            cursor = max(cursor,end)
        gap_causes = [Counter() for _ in idle]
        selected = [e for e in cpu if e.get("pid") == step.get("pid")
                    and (str(e.get("tid")) == str(step.get("tid"))
                         or (e.get("pid"), str(e.get("tid"))) in launch_threads)
                    and e["ts"] < hi and e["ts"]+e["dur"] > lo]
        points = []
        for idx,e in enumerate(selected):
            points.extend([(max(lo,e["ts"]),1,idx),(min(hi,e["ts"]+e["dur"]),0,idx)])
        for idx,(a,b) in enumerate(idle):
            points.extend([(a,3,idx),(b,2,idx)])
        points.sort()
        active = set()
        heaps = [[],[],[]]
        idle_index = None
        previous = lo

        def leaf(kind):
            heap = heaps[kind]
            while heap and heap[0][1] not in active:
                heapq.heappop(heap)
            return selected[heap[0][1]]["name"] if heap else None

        for ts,kind,idx in points:
            if ts > previous and idle_index is not None:
                runtime, op, py = leaf(0), leaf(1), leaf(2)
                key = ("runtime",runtime) if runtime else (("cpu_op",op) if op else ("python",py) if py else ("unattributed","<no recorded CPU event>"))
                delta = ts - previous
                causes[key] += delta
                python[py or "<no Python stack event>"] += delta
                pairs[(key[0], key[1], py or "<no Python stack event>")] += delta
                gap_causes[idle_index][key] += delta
            if kind == 0:
                active.discard(idx)
            elif kind == 1:
                active.add(idx)
                heapq.heappush(heaps[cpu_kind(selected[idx])], (selected[idx]["dur"],idx))
            elif kind == 2:
                idle_index = None
            else:
                idle_index = idx
            previous = ts
        for idx,(a,b) in enumerate(idle):
            top = gap_causes[idx].most_common(3)
            gaps.append({"step":step["name"], "start_us":a,"duration_ms":(b-a)/1000,
                         "top_overlaps":[{"kind":k[0],"name":k[1],"ms":v/1000} for k,v in top]})
        step_rows.append({"step":step["name"],"wall_ms":(hi-lo)/1000,
                          "gpu_busy_ms":sum(b-a for a,b in busy)/1000,
                          "all_idle_ms":((hi-lo)-sum(b-a for a,b in busy))/1000,
                          "gaps_over_threshold_ms":sum(b-a for a,b in idle)/1000,
                          "gap_count":len(idle)})
    n = len(steps)
    attributed = sum(causes.values()) / 1000
    threshold_idle = sum(x["gaps_over_threshold_ms"] for x in step_rows)
    assert abs(attributed-threshold_idle) < 1e-4, (attributed,threshold_idle)
    return {"steps":step_rows,"active_steps":n,"min_gap_ms":min_gap_ms,
            "wall_ms_per_step":sum(x["wall_ms"] for x in step_rows)/n,
            "gpu_busy_ms_per_step":sum(x["gpu_busy_ms"] for x in step_rows)/n,
            "all_idle_ms_per_step":sum(x["all_idle_ms"] for x in step_rows)/n,
            "gaps_over_threshold_ms_per_step":threshold_idle/n,
            "top_causes":[{"kind":k[0],"name":k[1],"ms_per_step":v/1000/n} for k,v in causes.most_common(10)],
            "all_causes":[{"kind":k[0],"name":k[1],"ms_per_step":v/1000/n} for k,v in causes.most_common()],
            "runtime_overlaps":[{"name":k[1],"ms_per_step":v/1000/n} for k,v in causes.most_common() if k[0] == "runtime"],
            "top_python_callers":[{"name":k,"ms_per_step":v/1000/n} for k,v in python.most_common(10)],
            "top_operation_callers":[{"kind":k[0],"operation":k[1],"python":k[2],"ms_per_step":v/1000/n} for k,v in pairs.most_common(20)],
            "gaps":sorted(gaps,key=lambda x:-x["duration_ms"]),
            "cpu_threads":sorted([str(tid) for pid,tid in launch_threads if pid == steps[0].get("pid")]),
            "method":"CPU step annotations only; GPU interval union across streams; exclusive CPU/runtime leaf attribution on training plus CUDA-launch threads, including autograd. Runtime > CPU op > Python, then shortest enclosing span. Python caller times are a separate view, not additive.",
            "caveat":"Temporal overlap, not causal proof. Profiler with_stack and harness TritonWatch instrumentation can inflate CPU overhead. Use unprofiled controls for performance decisions."}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("trace",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--min-gap-ms",type=float,default=0.2)
    args = parser.parse_args()
    opener = gzip.open if args.trace.suffix == ".gz" else open
    with opener(args.trace,"rt") as f:
        data = json.load(f)
    report = analyze(data["traceEvents"],args.min_gap_ms)
    report["trace_file"] = str(args.trace)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+"\n")
    for key in ("active_steps","wall_ms_per_step","gpu_busy_ms_per_step","all_idle_ms_per_step","gaps_over_threshold_ms_per_step"):
        print(key,round(report[key],3))
    for row in report["top_causes"]:
        print(f"{row['ms_per_step']:9.3f} ms/step  {row['kind']:12s} {row['name']}")
    print(report["caveat"])


if __name__ == "__main__":
    main()

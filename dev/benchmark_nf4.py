"""Compare original NF4, word/block NF4 and bitsandbytes on the four model shapes.

Run on T4 after correctness tests: python dev/benchmark_nf4.py . --out nf4_benchmark.json
All variants are checked bit-for-bit. Three timed rounds are interleaved; report the median.
Kernel-only times use preallocated output; wrapper times include allocation and Python dispatch.
This is a microbenchmark, not a scored training result.
"""

import argparse
import gc
import json
import random
import statistics
import sys
from pathlib import Path

import bitsandbytes.functional as bnbf
import torch
import triton
import triton.testing

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("repo")
parser.add_argument("--out", default="nf4_benchmark.json")
args = parser.parse_args()
sys.path.insert(0, args.repo)
from solution import nf4  # noqa: E402

shapes = [(1536, 1536), (256, 1536), (8960, 1536), (1536, 8960)]
weights = [2, 2, 2, 1]
configs = [(512, 4), (1024, 4), (2048, 4), (4096, 4), (1024, 8)]
records = []
bnb_records = []

for shape, count in zip(shapes, weights):
    torch.manual_seed(17)
    original = (torch.randn(shape, device="cuda") * 0.05).half()
    packed, qs = bnbf.quantize_4bit(original, blocksize=64, compress_statistics=True,
                                  quant_type="nf4")
    expected = bnbf.dequantize_4bit(packed, qs)
    output = torch.empty_like(expected)
    words = packed.reshape(-1).view(torch.int32)
    cases = [(impl, block, warps) for impl in ("legacy", "word") for block, warps in configs]
    samples = {case: {"kernel_ms": [], "wrapper_ms": []} for case in cases}

    def select(impl, block, warps):
        nf4.USE_WORD_LOADS = impl == "word"
        nf4.BYTES_PER_PROGRAM, nf4.NUM_WARPS = block, warps

        def launch():
            fn = nf4.skc_nf4_dequant_word_kernel if impl == "word" else nf4.skc_nf4_dequant_kernel
            storage = words if impl == "word" else packed
            fn[(triton.cdiv(packed.numel(), block),)](
                storage, qs.absmax, qs.state2.absmax, qs.state2.code, qs.offset, qs.code,
                output, packed.numel(), NESTED=True, QBLOCK=64, QBLOCK2=qs.state2.blocksize,
                BLOCK=block, num_warps=warps, enable_fp_fusion=False)

        launch()
        assert torch.equal(output, expected), (shape, impl, block, warps, "kernel")
        assert torch.equal(nf4.dequantize_nf4(packed, qs), expected), (shape, impl, "wrapper")
        return launch

    # Compile/check every configuration before timing any of them.
    for case in cases:
        select(*case)
    for round_no in range(3):
        order = cases.copy()
        random.Random(200 + round_no).shuffle(order)
        for case in order:
            launch = select(*case)
            samples[case]["kernel_ms"].append(triton.testing.do_bench(launch, warmup=10, rep=50))
            samples[case]["wrapper_ms"].append(triton.testing.do_bench(
                lambda: nf4.dequantize_nf4(packed, qs), warmup=10, rep=50))
    bnb_times = [triton.testing.do_bench(lambda: bnbf.dequantize_4bit(packed, qs),
                                      warmup=10, rep=50) for _ in range(3)]
    bnb_records.append({"shape": shape, "count_per_layer": count,
                        "wrapper_ms": statistics.median(bnb_times), "samples_ms": bnb_times})
    for (impl, block, warps), timings in samples.items():
        line = {"shape": shape, "count_per_layer": count, "impl": impl,
                "bytes_per_program": block, "num_warps": warps,
                **{key: statistics.median(vals) for key, vals in timings.items()},
                "samples": timings}
        records.append(line)
        print("NF4_CASE", json.dumps(line), flush=True)
    del original, packed, qs, expected, output, words
    gc.collect()
    torch.cuda.empty_cache()  # Between standalone benchmark shapes, never inside training.

scores = []
for impl in ("legacy", "word"):
    for block, warps in configs:
        rows = [r for r in records if (r["impl"], r["bytes_per_program"], r["num_warps"])
                == (impl, block, warps)]
        scores.append({"impl": impl, "bytes_per_program": block, "num_warps": warps,
                       **{key: sum(row[key] * row["count_per_layer"] for row in rows)
                          for key in ("kernel_ms", "wrapper_ms")}})
baseline = next(x for x in scores if (x["impl"], x["bytes_per_program"], x["num_warps"])
                == ("legacy", 1024, 4))
report = {"device": torch.cuda.get_device_name(), "torch": torch.__version__,
          "triton": triton.__version__, "baseline": baseline,
          "best_kernel": min(scores, key=lambda r: r["kernel_ms"]),
          "best_wrapper": min(scores, key=lambda r: r["wrapper_ms"]),
          "scores": scores, "bitsandbytes": bnb_records, "measurements": records}
Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
print("NF4_SUMMARY", json.dumps({key: report[key] for key in
                                  ("baseline", "best_kernel", "best_wrapper", "scores")}), flush=True)

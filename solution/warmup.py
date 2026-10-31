"""Compile our Triton kernels in a background thread while Soup loads the model.

Every measured run starts with an empty Triton cache, and the clock runs from process start, so
compilation is part of the score. Started in apply(), the compiles overlap the import of the
training stack and the loading of the model. It is a thread of the measured process, its time
counts like everything else, and it runs the same way in every run. If anything goes wrong here
the kernels simply compile on first use.
"""

import atexit
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

_thread = None
_done = threading.Event()
_processes = []
_error = None
# Keep one source of warmup calls for both thread and process modes. Partition by
# kernel name, never by training data or model state. v4 and v5.1 use these kernels.
_KERNELS = (
    "skc_nf4_dequant_kernel", "skc_rmsnorm_fwd_kernel", "skc_rmsnorm_bwd_kernel",
    "skc_rope_fwd_kernel", "skc_rope_bwd_kernel", "skc_keep_pack_kernel",
    "skc_lora_inputs_kernel", "skc_lora_input_grad_kernel",
    "example_swiglu_fwd_kernel", "example_swiglu_bwd_kernel", "skc_head_stats_kernel",
)


def _mark(event, **extra):
    path = os.environ.get("SKC_WARMUP_TRACE")
    if not path:
        return
    record = dict(event=event, pid=os.getpid(), time_ns=time.time_ns(), **extra)
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    except OSError:
        pass


def _compile(kernel, *args, shard=0, shards=1, **kwargs):
    if kernel.__name__ in _KERNELS and _KERNELS.index(kernel.__name__) % shards != shard:
        return
    _mark("kernel_begin", kernel=kernel.__name__, shard=shard)
    kernel.warmup(*args, **kwargs)
    _mark("kernel_end", kernel=kernel.__name__, shard=shard)



def _work(shard=0, shards=1):
    global _error
    _mark("compile_begin", shard=shard, shards=shards)
    try:
        import torch

        from . import attn, head, nf4, rmsnorm

        dev = torch.device("cuda", torch.cuda.current_device())
        grid = (1,)
        # nf4 dequant (nested absmax, blocksize 64): as dequantize_nf4 launches it.
        _compile(nf4.skc_nf4_dequant_kernel,
            torch.zeros(nf4.BYTES_PER_PROGRAM, dtype=torch.uint8, device=dev),
            torch.zeros(nf4.BYTES_PER_PROGRAM // 32 + 1, dtype=torch.uint8, device=dev),
            torch.zeros(1, dtype=torch.float32, device=dev),
            torch.zeros(256, dtype=torch.float32, device=dev),
            torch.zeros((), dtype=torch.float32, device=dev),
            torch.zeros(16, dtype=torch.float32, device=dev),
            torch.empty(2 * nf4.BYTES_PER_PROGRAM, dtype=torch.float16, device=dev),
            nf4.BYTES_PER_PROGRAM, NESTED=True, QBLOCK=64, QBLOCK2=256, BLOCK=nf4.BYTES_PER_PROGRAM,
            num_warps=nf4.NUM_WARPS,
            enable_fp_fusion=False, grid=grid, shard=shard, shards=shards)
        # RMSNorm over the 1536-wide fp32 residual stream with fp32 weights.
        n = 1536
        block = 2048
        x = torch.zeros((1, n), dtype=torch.float32, device=dev)
        w = torch.zeros(n, dtype=torch.float32, device=dev)
        r = torch.zeros(1, dtype=torch.float32, device=dev)
        _compile(rmsnorm.skc_rmsnorm_fwd_kernel, x, w, torch.empty_like(x), r, 1, n, n, n, 1e-6,
                                              BLOCK=block, num_warps=8, grid=grid, shard=shard, shards=shards)
        _compile(rmsnorm.skc_rmsnorm_bwd_kernel, torch.zeros_like(x), x, w, r, torch.empty_like(x), 1,
                                              n, n, n, n, BLOCK=block, num_warps=8, grid=grid, shard=shard, shards=shards)
        # RoPE (12 query heads, 2 kv heads, head_dim 128), forward and backward.
        q = torch.zeros((1, 12 * 128), dtype=torch.float16, device=dev)
        kk = torch.zeros((1, 2 * 128), dtype=torch.float16, device=dev)
        cs = torch.zeros((1, 128), dtype=torch.float32, device=dev)
        _compile(attn.skc_rope_fwd_kernel, q, q, kk, cs, cs, torch.empty_like(q), torch.empty_like(kk), 1,
                                        HQ=12, HK=2, D=128, BQ=2048, BK=256, num_warps=4,
                                        enable_fp_fusion=False, grid=grid, shard=shard, shards=shards)
        _compile(attn.skc_rope_bwd_kernel, q, kk, cs, cs, torch.empty_like(q), torch.empty_like(kk), 1,
                                        HQ=12, HK=2, D=128, BQ=2048, BK=256, num_warps=4,
                                        enable_fp_fusion=False, grid=grid, shard=shard, shards=shards)
        # q/v LoRA dropout masks: bit packing, rebuilt inputs and input gradient (attn.py).
        bq = torch.zeros((1, n // 8), dtype=torch.uint8, device=dev)
        keep = torch.zeros((1, n), dtype=torch.uint8, device=dev)
        _compile(attn.skc_keep_pack_kernel, keep, keep, bq, bq, 1, WIDTH=n, BYTES=256, num_warps=2,
                                         grid=grid, shard=shard, shards=shards)
        x16 = torch.empty((1, n), dtype=torch.float16, device=dev)
        _compile(attn.skc_lora_inputs_kernel, x, bq, bq, r, r, x16, x16, 1, WIDTH=n, BLOCK=block,
                                           num_warps=4, enable_fp_fusion=False, grid=grid, shard=shard, shards=shards)
        _compile(attn.skc_lora_input_grad_kernel, x16, x16, bq, bq, x, 1, 1.0526, 1.0526, WIDTH=n,
                                               BLOCK=block, num_warps=4, enable_fp_fusion=False,
                                               grid=grid, shard=shard, shards=shards)
        # The starter kit's SwiGLU (used by mlp.py with fp16 gate/up; it does not count as ours).
        from . import swiglu

        m = 8960
        g16 = torch.zeros(m, dtype=torch.float16, device=dev)
        _compile(swiglu.example_swiglu_fwd_kernel, g16, g16, torch.empty_like(g16), m,
                                                BLOCK=swiglu.BLOCK, grid=grid, shard=shard, shards=shards)
        _compile(swiglu.example_swiglu_bwd_kernel, g16, g16, g16, torch.empty_like(g16),
                                                torch.empty_like(g16), m, BLOCK=swiglu.BLOCK,
                                                grid=grid, shard=shard, shards=shards)
        # Head statistics over the 151,936-token vocabulary.
        vocab = 151936
        _compile(head.skc_head_stats_kernel,
            torch.zeros((1, vocab), dtype=torch.float16, device=dev),
            torch.zeros(1, dtype=torch.int64, device=dev),
            torch.empty(1, dtype=torch.float32, device=dev),
            torch.empty(1, dtype=torch.float32, device=dev),
            torch.empty(1, dtype=torch.float32, device=dev),
            torch.empty(1, dtype=torch.int32, device=dev),
            1, vocab, vocab, BLOCK_V=head.BLOCK_V, num_warps=8, grid=grid, shard=shard, shards=shards)
    except Exception as exc:  # noqa: BLE001 - first-use compilation is the fallback
        _error = repr(exc)
        _mark("compile_error", error=_error, shard=shard)
        return False
    finally:
        _mark("compile_end", shard=shard)
    return True



def _process_work(count):
    """Children compile to this run's empty cache; parent then loads identical keys.

    A fresh interpreter is essential: CUDA was initialized before apply(), so fork
    would be unsafe. Children finish before wait() returns, including on fallback.
    """
    root = str(Path(__file__).resolve().parents[1])
    code = ("import sys; sys.path.insert(0, sys.argv[1]); "
            "from solution import warmup; "
            "ok=warmup._work(int(sys.argv[2]),int(sys.argv[3])); "
            "sys.exit(0 if ok else 1)")
    _mark("processes_begin", count=count)
    try:
        for shard in range(count):
            proc = subprocess.Popen([sys.executable, "-c", code, root, str(shard), str(count)],
                                    env=dict(os.environ), stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
            _processes.append(proc)
        for proc in _processes:
            if proc.wait(timeout=120) != 0:
                _mark("worker_failed", child_pid=proc.pid)
    except Exception as exc:  # noqa: BLE001 - preserve ordinary in-process fallback
        _mark("process_error", error=repr(exc))
    finally:
        for proc in _processes:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        _mark("processes_end", count=count)
    # Populates the parent's JIT cache, normally loading the binaries just built by
    # the children. Also retries every kernel if a child failed or timed out.
    _work()


def _background(count):
    try:
        if count:
            _process_work(count)
        else:
            _work()
    finally:
        _done.set()
        _mark("ready", mode=count)


def start():
    global _thread
    if _thread is not None:
        return
    try:
        count = int(os.environ.get("SKC_WARMUP_PROCESSES", "0"))
    except ValueError:
        count = 0
    count = count if count in (1, 2) else 0
    _mark("start", mode=count)
    _thread = threading.Thread(target=_background, args=(count,),
                               name="skc-triton-warmup", daemon=True)
    _thread.start()
    atexit.register(wait)


def wait():
    """Wait for workers and parent cache loading (no-op if never started)."""
    if _thread is not None and not _done.is_set():
        _mark("wait_begin")
        _thread.join()
        _mark("wait_end")

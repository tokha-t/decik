"""Compile our Triton kernels in a background thread while Soup loads the model.

Every measured run starts with an empty Triton cache, and the clock runs from process start, so
compilation is part of the score. Started in apply(), the compiles overlap the import of the
training stack and the loading of the model. It is a thread of the measured process, its time
counts like everything else, and it runs the same way in every run. If anything goes wrong here
the kernels simply compile on first use.
"""

import atexit
import threading

_thread = None
_done = threading.Event()


def _work():
    try:
        import torch

        from . import attn, head, nf4, rmsnorm

        dev = torch.device("cuda", torch.cuda.current_device())
        grid = (1,)
        # nf4 dequant (nested absmax, blocksize 64): as dequantize_nf4 launches it.
        nf4.skc_nf4_dequant_kernel.warmup(
            torch.zeros(nf4.BYTES_PER_PROGRAM, dtype=torch.uint8, device=dev),
            torch.zeros(nf4.BYTES_PER_PROGRAM // 32 + 1, dtype=torch.uint8, device=dev),
            torch.zeros(1, dtype=torch.float32, device=dev),
            torch.zeros(256, dtype=torch.float32, device=dev),
            torch.zeros((), dtype=torch.float32, device=dev),
            torch.zeros(16, dtype=torch.float32, device=dev),
            torch.empty(2 * nf4.BYTES_PER_PROGRAM, dtype=torch.float16, device=dev),
            nf4.BYTES_PER_PROGRAM, NESTED=True, QBLOCK=64, QBLOCK2=256, BLOCK=nf4.BYTES_PER_PROGRAM,
            num_warps=nf4.NUM_WARPS,
            enable_fp_fusion=False, grid=grid)
        # RMSNorm over the 1536-wide fp32 residual stream with fp32 weights.
        n = 1536
        block = 2048
        x = torch.zeros((1, n), dtype=torch.float32, device=dev)
        w = torch.zeros(n, dtype=torch.float32, device=dev)
        r = torch.zeros(1, dtype=torch.float32, device=dev)
        rmsnorm.skc_rmsnorm_fwd_kernel.warmup(x, w, torch.empty_like(x), r, 1, n, n, n, 1e-6,
                                              BLOCK=block, num_warps=8, grid=grid)
        rmsnorm.skc_rmsnorm_bwd_kernel.warmup(torch.zeros_like(x), x, w, r, torch.empty_like(x), 1,
                                              n, n, n, n, BLOCK=block, num_warps=8, grid=grid)
        # RoPE (12 query heads, 2 kv heads, head_dim 128), forward and backward.
        q = torch.zeros((1, 12 * 128), dtype=torch.float16, device=dev)
        kk = torch.zeros((1, 2 * 128), dtype=torch.float16, device=dev)
        cs = torch.zeros((1, 128), dtype=torch.float32, device=dev)
        attn.skc_rope_fwd_kernel.warmup(q, q, kk, cs, cs, torch.empty_like(q), torch.empty_like(kk), 1,
                                        HQ=12, HK=2, D=128, BQ=2048, BK=256, num_warps=4,
                                        enable_fp_fusion=False, grid=grid)
        attn.skc_rope_bwd_kernel.warmup(q, kk, cs, cs, torch.empty_like(q), torch.empty_like(kk), 1,
                                        HQ=12, HK=2, D=128, BQ=2048, BK=256, num_warps=4,
                                        enable_fp_fusion=False, grid=grid)
        # The starter kit's SwiGLU (used by mlp.py with fp16 gate/up; it does not count as ours).
        from . import swiglu

        m = 8960
        g16 = torch.zeros(m, dtype=torch.float16, device=dev)
        swiglu.example_swiglu_fwd_kernel.warmup(g16, g16, torch.empty_like(g16), m,
                                                BLOCK=swiglu.BLOCK, grid=grid)
        swiglu.example_swiglu_bwd_kernel.warmup(g16, g16, g16, torch.empty_like(g16),
                                                torch.empty_like(g16), m, BLOCK=swiglu.BLOCK,
                                                grid=grid)
        # Head statistics over the 151,936-token vocabulary.
        vocab = 151936
        head.skc_head_stats_kernel.warmup(
            torch.zeros((1, vocab), dtype=torch.float16, device=dev),
            torch.zeros(1, dtype=torch.int64, device=dev),
            torch.empty(1, dtype=torch.float32, device=dev),
            torch.empty(1, dtype=torch.float32, device=dev),
            torch.empty(1, dtype=torch.float32, device=dev),
            torch.empty(1, dtype=torch.int32, device=dev),
            1, vocab, vocab, BLOCK_V=head.BLOCK_V, num_warps=8, grid=grid)
    except Exception:  # noqa: BLE001 - compiling on first use is the fallback
        pass
    finally:
        _done.set()


def start():
    global _thread
    if _thread is not None:
        return
    _thread = threading.Thread(target=_work, name="skc-triton-warmup", daemon=True)
    _thread.start()
    atexit.register(wait)


def wait():
    """Block until the warm-up finished (no-op when it never started)."""
    if _thread is not None and not _done.is_set():
        _thread.join()

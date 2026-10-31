"""Our solution: Triton kernels and patches that make the same Soup run faster and lighter.

The harness imports this package and calls `apply()` once, before Soup builds the model and the
trainer. Every patch is a class-level patch that checks its inputs and falls back to the
original code for anything it was not written for (CPU tensors, other dtypes, Soup's tests
with fake modules), so the regression tests see Soup unchanged.

What runs (see README.md for the numbers):
- nf4.py      our Triton NF4 dequantization inside every 4-bit linear, forward and backward
- rmsnorm.py  our Triton RMSNorm, forward and backward
- mlp.py      MLP with fp16 intermediates (same values; a third of the activation memory)
- embedding.py the tied embedding / lm_head kept in fp16 (same values; no fp32 copy)
- head.py     lm_head + loss + TRL metrics + rewind losses in chunks, with our Triton kernel
- varlen.py   padding-free decoder (attention per row, LoRA dropout with the base's draws)
- attn.py     LoRA add + RoPE + fp16 cast of q and k in one Triton launch (padding-free path)
- warmup.py   Triton compilation in a background thread while the model loads

SKC_<NAME>=0 in the environment turns one part off, for A/B runs while developing; the measured
runs use the defaults (everything on). SKC_RECOMPUTE=0 keeps the MLP's gate and up for backward
instead of recomputing them (about 1.5 GB more on the longest batch, a few seconds faster).
"""

import os


def _enabled(name):
    return os.environ.get(f"SKC_{name}", "1") != "0"


def apply():
    import torch

    from . import attn, embedding, head, mlp, nf4, rmsnorm, varlen, warmup

    if _enabled("WARMUP") and torch.cuda.is_available():
        warmup.start()
    if _enabled("NF4"):
        nf4.patch()
    if _enabled("RMSNORM"):
        rmsnorm.patch()
    if _enabled("MLP"):
        mlp.patch(recompute=os.environ.get("SKC_RECOMPUTE", "1") == "1")
    if _enabled("EMBED"):
        embedding.patch()
    if _enabled("HEAD"):
        head.patch()
        if _enabled("VARLEN"):
            varlen.patch(recompute_attention=_enabled("ATTN_RECOMPUTE"))
            if _enabled("ATTN"):
                attn.patch(recompute_lora=_enabled("LORA_RECOMPUTE"))

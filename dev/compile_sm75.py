"""Compile every Triton kernel of solution/ for a Tesla T4 (sm_75) without a GPU.

Catches what the CPU interpreter cannot: type errors in loops, unsupported ops on Turing,
register pressure, and produces the PTX/cubin exactly as Triton would on the T4.
"""
import sys
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

sys.path.insert(0, sys.argv[1])
from solution import attn, head, nf4, rmsnorm  # noqa: E402

TARGET = GPUTarget("cuda", 75, 32)


def build(fn, signature, constexprs, options):
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    kernel = triton.compile(src, target=TARGET, options=options)
    meta = kernel.metadata
    print(f"OK  {fn.__name__:28s} regs={getattr(meta, 'num_regs', '?')} spills="
          f"{getattr(meta, 'num_spills', '?')} shared={meta.shared} warps={meta.num_warps}")
    return kernel


ok = True
cases = [
    (nf4.skc_nf4_dequant_word_kernel,
     {"packed_ptr": "*i32", "absmax_ptr": "*u8", "absmax2_ptr": "*fp32", "code2_ptr": "*fp32",
      "offset_ptr": "*fp32", "code_ptr": "*fp32", "out_ptr": "*fp16", "n_bytes": "i32",
      "NESTED": "constexpr", "QBLOCK": "constexpr", "QBLOCK2": "constexpr", "BLOCK": "constexpr"},
     {"NESTED": True, "QBLOCK": 64, "QBLOCK2": 256, "BLOCK": nf4.BYTES_PER_PROGRAM},
     {"num_warps": nf4.NUM_WARPS, "enable_fp_fusion": False}),
    (nf4.skc_nf4_dequant_kernel,
     {"packed_ptr": "*u8", "absmax_ptr": "*u8", "absmax2_ptr": "*fp32", "code2_ptr": "*fp32",
      "offset_ptr": "*fp32", "code_ptr": "*fp32", "out_ptr": "*fp16", "n_bytes": "i32",
      "NESTED": "constexpr", "QBLOCK": "constexpr", "QBLOCK2": "constexpr", "BLOCK": "constexpr"},
     {"NESTED": True, "QBLOCK": 64, "QBLOCK2": 256, "BLOCK": nf4.BYTES_PER_PROGRAM},
     {"num_warps": 4, "enable_fp_fusion": False}),
    (rmsnorm.skc_rmsnorm_fwd_kernel,
     {"x_ptr": "*fp32", "w_ptr": "*fp32", "y_ptr": "*fp32", "rstd_ptr": "*fp32", "n_rows": "i32",
      "stride_x": "i32", "stride_y": "i32", "n_cols": "i32", "eps": "fp32", "BLOCK": "constexpr"},
     {"BLOCK": 2048}, {"num_warps": 8}),
    (rmsnorm.skc_rmsnorm_bwd_kernel,
     {"dy_ptr": "*fp32", "x_ptr": "*fp32", "w_ptr": "*fp32", "rstd_ptr": "*fp32",
      "dx_ptr": "*fp32", "n_rows": "i32", "stride_dy": "i32", "stride_x": "i32",
      "stride_dx": "i32", "n_cols": "i32", "BLOCK": "constexpr"},
     {"BLOCK": 2048}, {"num_warps": 8}),
    (head.skc_head_stats_kernel,
     {"logits_ptr": "*fp16", "target_ptr": "*i64", "lse_ptr": "*fp32", "entropy_ptr": "*fp32",
      "loss_ptr": "*fp32", "hit_ptr": "*i32", "n_rows": "i32", "stride": "i32",
      "n_vocab": "i32", "BLOCK_V": "constexpr"},
     {"BLOCK_V": head.BLOCK_V}, {"num_warps": 8}),
    (attn.skc_rope_fwd_kernel,
     {"qb_ptr": "*fp16", "ql_ptr": "*fp16", "k_ptr": "*fp16", "cos_ptr": "*fp32", "sin_ptr": "*fp32",
      "qo_ptr": "*fp16", "ko_ptr": "*fp16", "n_tok": "i32", "HQ": "constexpr", "HK": "constexpr",
      "D": "constexpr", "BQ": "constexpr", "BK": "constexpr"},
     {"HQ": 12, "HK": 2, "D": 128, "BQ": 2048, "BK": 256}, {"num_warps": 4, "enable_fp_fusion": False}),
    (attn.skc_rope_bwd_kernel,
     {"gq_ptr": "*fp16", "gk_ptr": "*fp16", "cos_ptr": "*fp32", "sin_ptr": "*fp32", "dq_ptr": "*fp16",
      "dk_ptr": "*fp16", "n_tok": "i32", "HQ": "constexpr", "HK": "constexpr", "D": "constexpr",
      "BQ": "constexpr", "BK": "constexpr"},
     {"HQ": 12, "HK": 2, "D": 128, "BQ": 2048, "BK": 256}, {"num_warps": 4, "enable_fp_fusion": False}),
]
for fn, sig, consts, opts in cases:
    try:
        build(fn, sig, consts, opts)
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
sys.exit(0 if ok else 1)

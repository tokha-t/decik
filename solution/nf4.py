"""NF4 weights of bitsandbytes' Linear4bit, unpacked to fp16 by our own Triton kernel.

For a batch of many tokens bitsandbytes (nf4, double quant, blocksize 64, compute dtype fp16)
computes forward = dequantize the whole weight to fp16 + F.linear, and backward = dequantize
again + one matmul for the input gradient. We do the same, with the dequantization in our
kernel and the same arithmetic: code[nibble] * absmax in fp32 rounded to fp16, the double-quant
absmax as code2[q] * absmax2 + offset with two separate roundings (no FMA). The weights come
out bit-identical to bitsandbytes' (see test_nf4.py).

It runs 2 x 7 x 28 = 392 times per training step, which is what keeps our kernels well above
5% of the training time.
"""

import os

import torch
import triton
import triton.language as tl

from . import warmup
from ._state import on_gpu

BYTES_PER_PROGRAM = int(os.environ.get("SKC_NF4_BYTES", "1024"))  # bytes per program (2 weights each)
NUM_WARPS = int(os.environ.get("SKC_NF4_WARPS", "4"))


@triton.jit(do_not_specialize=["n_bytes"])
def skc_nf4_dequant_kernel(packed_ptr, absmax_ptr, absmax2_ptr, code2_ptr, offset_ptr, code_ptr,
                           out_ptr, n_bytes, NESTED: tl.constexpr, QBLOCK: tl.constexpr,
                           QBLOCK2: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < n_bytes
    byte = tl.load(packed_ptr + offs, mask=valid, other=0).to(tl.int32)
    # Both 4-bit weights of a byte (high nibble first) fall in the same absmax block.
    block = (offs * 2) // QBLOCK
    if NESTED:
        q = tl.load(absmax_ptr + block, mask=valid, other=0).to(tl.int32)
        scale = tl.load(code2_ptr + q, mask=valid, other=0.0) * \
            tl.load(absmax2_ptr + block // QBLOCK2, mask=valid, other=0.0)
        scale = scale + tl.load(offset_ptr)
    else:
        scale = tl.load(absmax_ptr + block, mask=valid, other=0.0)
    high = tl.load(code_ptr + (byte >> 4), mask=valid, other=0.0) * scale
    low = tl.load(code_ptr + (byte & 15), mask=valid, other=0.0) * scale
    pair = tl.interleave(high, low).to(tl.float16)
    outs = pid * (2 * BLOCK) + tl.arange(0, 2 * BLOCK)
    tl.store(out_ptr + outs, pair, mask=outs < 2 * n_bytes)


def supported(quant_state, n_in=None):
    qs = quant_state
    try:
        ok = (qs is not None and qs.quant_type == "nf4" and qs.dtype == torch.float16
              and len(qs.shape) == 2 and (qs.shape[0] * qs.shape[1]) % 2 == 0
              and qs.blocksize % 2 == 0 and qs.code is not None)
        if ok and qs.nested:
            ok = qs.state2 is not None and qs.offset is not None and qs.absmax.dtype == torch.uint8
        if ok and n_in is not None:
            ok = qs.shape[1] == n_in
        return bool(ok)
    except AttributeError:
        return False


def dequantize_nf4(packed, quant_state):
    """[N, K] fp16 weights of a bitsandbytes nf4 tensor (packed uint8 + QuantState)."""
    warmup.wait()
    qs = quant_state
    n_out, n_in = qs.shape
    n_bytes = n_out * n_in // 2
    out = torch.empty((n_out, n_in), dtype=torch.float16, device=packed.device)
    code = qs.code if qs.code.device == packed.device else qs.code.to(packed.device)
    if qs.nested:
        absmax, absmax2, code2, offset = qs.absmax, qs.state2.absmax, qs.state2.code, qs.offset
        block2 = qs.state2.blocksize
    else:
        absmax = absmax2 = code2 = offset = qs.absmax
        block2 = 256
    grid = (triton.cdiv(n_bytes, BYTES_PER_PROGRAM),)
    skc_nf4_dequant_kernel[grid](packed, absmax, absmax2, code2, offset, code, out, n_bytes,
                                 NESTED=bool(qs.nested), QBLOCK=int(qs.blocksize),
                                 QBLOCK2=int(block2), BLOCK=BYTES_PER_PROGRAM, num_warps=NUM_WARPS,
                                 enable_fp_fusion=False)
    return out


class NF4Linear(torch.autograd.Function):
    """F.linear(x, dequant(W), bias) with fp16 x; the input gradient dequantizes W again, as
    bitsandbytes' MatMul4Bit does (the weight itself is frozen)."""

    @staticmethod
    def forward(ctx, x16, packed, bias16, quant_state):
        weight = dequantize_nf4(packed, quant_state)
        ctx.packed, ctx.quant_state = packed, quant_state
        return torch.nn.functional.linear(x16, weight, bias16)

    @staticmethod
    def backward(ctx, grad_out):
        grad_x = None
        if ctx.needs_input_grad[0]:
            weight = dequantize_nf4(ctx.packed, ctx.quant_state)
            grad_x = torch.matmul(grad_out.to(weight.dtype), weight)
        return grad_x, None, None, None


def patch():
    import bitsandbytes as bnb

    cls = bnb.nn.Linear4bit
    if getattr(cls.forward, "_skc", False):
        return
    original = cls.forward
    try:
        from bitsandbytes.nn.modules import fix_4bit_weight_quant_state_from_module as fix_state
    except ImportError:  # pragma: no cover - older bitsandbytes
        fix_state = None

    def forward(self, x):
        weight = getattr(self, "weight", None)
        if fix_state is not None and weight is not None and getattr(weight, "quant_state", None) is None:
            fix_state(self)
        qs = getattr(weight, "quant_state", None)
        bias = self.bias
        if not (isinstance(x, torch.Tensor) and on_gpu(x) and x.numel() > 0
                and weight is not None and on_gpu(weight) and weight.dtype == torch.uint8
                and not weight.requires_grad
                and getattr(self, "compute_dtype", None) == torch.float16
                and (bias is None or not bias.requires_grad)
                and supported(qs, x.shape[-1])):
            return original(self, x)
        inp_dtype = x.dtype
        bias16 = None if bias is None else bias.to(torch.float16)
        out = NF4Linear.apply(x.to(torch.float16), weight.data, bias16, qs)
        return out.to(inp_dtype)

    forward._skc = True
    cls.forward = forward

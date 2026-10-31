"""Qwen2RMSNorm in one Triton launch forward and one backward.

The base runs it in fp32 as six PyTorch ops each way (pow, mean, add, rsqrt, mul, mul) over the
fp32 residual stream. Same math here: y = w * (x * rsqrt(mean(x^2) + eps)) in fp32; backward
keeps only x (the residual, alive anyway) and one rstd per token. The norm weight is frozen.
"""

import torch
import triton
import triton.language as tl

from . import warmup
from ._state import STATE, on_gpu


@triton.jit(do_not_specialize=["n_rows"])
def skc_rmsnorm_fwd_kernel(x_ptr, w_ptr, y_ptr, rstd_ptr, n_rows, stride_x, stride_y, n_cols,
                           eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    valid = cols < n_cols
    x = tl.load(x_ptr + row * stride_x + cols, mask=valid, other=0.0).to(tl.float32)
    mean_sq = tl.sum(x * x, axis=0) / n_cols
    rstd = tl.math.rsqrt(mean_sq + eps)
    w = tl.load(w_ptr + cols, mask=valid, other=0.0).to(tl.float32)
    y = w * (x * rstd)
    tl.store(y_ptr + row * stride_y + cols, y.to(y_ptr.dtype.element_ty), mask=valid)
    tl.store(rstd_ptr + row, rstd)


@triton.jit(do_not_specialize=["n_rows"])
def skc_rmsnorm_bwd_kernel(dy_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr, n_rows, stride_dy, stride_x,
                           stride_dx, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    valid = cols < n_cols
    dy = tl.load(dy_ptr + row * stride_dy + cols, mask=valid, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row * stride_x + cols, mask=valid, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=valid, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)
    g = dy * w
    # d/dx of x * rstd(x): rstd * (g - x * rstd^2 * mean(g * x))
    dot = tl.sum(g * x, axis=0) / n_cols
    dx = rstd * (g - x * (rstd * rstd * dot))
    tl.store(dx_ptr + row * stride_dx + cols, dx.to(dx_ptr.dtype.element_ty), mask=valid)


def _warps(block):
    return 8 if block >= 2048 else 4


def _forward(x2, weight, eps):
    n_rows, n_cols = x2.shape
    y = torch.empty((n_rows, n_cols), dtype=x2.dtype, device=x2.device)
    rstd = torch.empty(n_rows, dtype=torch.float32, device=x2.device)
    block = triton.next_power_of_2(n_cols)
    skc_rmsnorm_fwd_kernel[(n_rows,)](x2, weight, y, rstd, n_rows, x2.stride(0), y.stride(0),
                                      n_cols, eps, BLOCK=block, num_warps=_warps(block))
    return y, rstd


def _rows(x):
    x2 = x.reshape(-1, x.shape[-1])
    return x2 if x2.stride(-1) == 1 else x2.contiguous()


def recompute(x, weight, eps):
    """The norm's output for an input it ran on, bit-identical (same kernel, same launch)."""
    y, _ = _forward(_rows(x), weight, eps)
    return y.view(x.shape)


class RMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        warmup.wait()
        x2 = _rows(x)
        y, rstd = _forward(x2, weight, eps)
        ctx.save_for_backward(x2, weight, rstd)
        ctx.shape = x.shape
        return y.view(x.shape)

    @staticmethod
    def backward(ctx, dy):
        x2, weight, rstd = ctx.saved_tensors
        n_rows, n_cols = x2.shape
        dy2 = dy.reshape(-1, n_cols)
        if dy2.stride(-1) != 1:
            dy2 = dy2.contiguous()
        dx = torch.empty((n_rows, n_cols), dtype=x2.dtype, device=x2.device)
        block = triton.next_power_of_2(n_cols)
        skc_rmsnorm_bwd_kernel[(n_rows,)](dy2, x2, weight, rstd, dx, n_rows, dy2.stride(0),
                                          x2.stride(0), dx.stride(0), n_cols, BLOCK=block,
                                          num_warps=_warps(block))
        return dx.view(ctx.shape), None, None


def patch():
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2RMSNorm
    if getattr(cls.forward, "_skc", False):
        return
    original = cls.forward

    def forward(self, hidden_states):
        weight = getattr(self, "weight", None)
        if not (isinstance(hidden_states, torch.Tensor) and on_gpu(hidden_states)
                and hidden_states.dtype == torch.float32 and hidden_states.numel() > 0
                and isinstance(weight, torch.Tensor) and on_gpu(weight)
                and weight.dtype in (torch.float32, torch.float16) and not weight.requires_grad
                and weight.dim() == 1 and hidden_states.shape[-1] == weight.shape[0]
                and hidden_states.shape[-1] <= 8192):
            return original(self, hidden_states)
        eps = float(self.variance_epsilon)
        out = RMSNormFn.apply(hidden_states, weight, eps)
        if STATE.packing_active:
            STATE.norm_link = (out, hidden_states, weight, eps)
        return out

    forward._skc = True
    cls.forward = forward

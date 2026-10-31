"""Qwen2 MLP with fp16 intermediates, and optionally gate/up recomputed in backward.

In the base, the residual stream is fp32 (prepare_model_for_kbit_training keeps the embedding in
fp32) and bitsandbytes' Linear4bit returns its fp16 matmul result cast back to the input dtype,
so gate and up live in fp32 and silu(gate), gate and up are all saved for backward in fp32:
3 x 8960 x 4 bytes per token per layer. Their values are exactly fp16 values (the output of an
fp16 matmul), so we feed the projections fp16 and keep gate and up in fp16: the same numbers,
a third of the memory, and none of the fp32 casts. Each projection casts the input itself, as
bitsandbytes does, so the input gradients still add up in fp32.

With recompute=True the whole MLP is one autograd function that saves only its fp16 input
(1536 x 2 bytes per token per layer instead of 2 x 8960 x 2) and recomputes gate and up in
backward with the same kernels and matmuls, so the numbers are the same as without recompute.
It costs two more dequantizations and matmuls per layer in backward. On the padding-free path
it does not even keep that input: backward rebuilds it from the post-attention norm's input
(kept by the norm for its own backward) with the norm kernel, bit-identically.

SwiGLU itself is the starter kit's example kernel (allowed to run; it does not count as ours).
"""

import torch
import triton

from . import nf4, rmsnorm
from ._state import STATE, on_gpu


def _linear16(x16, module):
    """What bitsandbytes' Linear4bit computes for an fp16 input: dequantize + F.linear in fp16."""
    weight = nf4.dequantize_nf4(module.weight.data, module.weight.quant_state)
    bias = module.bias
    return torch.nn.functional.linear(x16, weight, None if bias is None else bias.to(torch.float16))


def _swiglu_fwd(gate, up):
    from . import swiglu

    out = torch.empty_like(gate)
    n = gate.numel()
    swiglu.example_swiglu_fwd_kernel[(triton.cdiv(n, swiglu.BLOCK),)](gate, up, out, n,
                                                                     BLOCK=swiglu.BLOCK)
    return out


def _swiglu_bwd(grad, gate, up):
    from . import swiglu

    dgate, dup = torch.empty_like(gate), torch.empty_like(up)
    n = gate.numel()
    swiglu.example_swiglu_bwd_kernel[(triton.cdiv(n, swiglu.BLOCK),)](grad, gate, up, dgate, dup,
                                                                     n, BLOCK=swiglu.BLOCK)
    return dgate, dup


class RecomputedMLP(torch.autograd.Function):
    """down(silu(gate(x)) * up(x)) for the fp32 input x; returns fp16 like the fp16 path."""

    @staticmethod
    def forward(ctx, x, gate_proj, up_proj, down_proj, norm_input=None, norm_weight=None, eps=None):
        x16 = x.to(torch.float16).contiguous()
        gate = _linear16(x16, gate_proj)
        up = _linear16(x16, up_proj)
        hidden = _swiglu_fwd(gate, up)
        del gate, up
        out = _linear16(hidden, down_proj)
        if norm_input is not None:  # x = norm(norm_input): keep nothing new
            ctx.save_for_backward(norm_input, norm_weight)
            ctx.eps = eps
        else:
            ctx.save_for_backward(x16)
        ctx.from_norm = norm_input is not None
        ctx.modules = (gate_proj, up_proj, down_proj)
        ctx.in_dtype = x.dtype
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.from_norm:
            norm_input, norm_weight = ctx.saved_tensors
            x16 = rmsnorm.recompute(norm_input, norm_weight, ctx.eps).to(torch.float16).contiguous()
        else:
            (x16,) = ctx.saved_tensors
        gate_proj, up_proj, down_proj = ctx.modules
        g16 = grad_out.to(torch.float16).contiguous()
        w_down = nf4.dequantize_nf4(down_proj.weight.data, down_proj.weight.quant_state)
        grad_hidden = torch.matmul(g16, w_down)
        del w_down
        gate = _linear16(x16, gate_proj)
        up = _linear16(x16, up_proj)
        dgate, dup = _swiglu_bwd(grad_hidden.contiguous(), gate, up)
        del gate, up, grad_hidden
        w_gate = nf4.dequantize_nf4(gate_proj.weight.data, gate_proj.weight.quant_state)
        grad_x = torch.matmul(dgate, w_gate).to(ctx.in_dtype)
        del w_gate, dgate
        w_up = nf4.dequantize_nf4(up_proj.weight.data, up_proj.weight.quant_state)
        # Two separate casts summed in fp32, as autograd adds the two input gradients in the base.
        grad_x = grad_x + torch.matmul(dup, w_up).to(ctx.in_dtype)
        return grad_x, None, None, None, None, None, None


def patch(recompute=False):
    import bitsandbytes as bnb
    from transformers.models.qwen2 import modeling_qwen2

    from .swiglu import swiglu

    cls = modeling_qwen2.Qwen2MLP
    if getattr(cls.forward, "_skc", False):
        return
    original = cls.forward
    linear4bit = bnb.nn.Linear4bit

    def fp16_projection(module):
        return isinstance(module, linear4bit) and getattr(module, "compute_dtype", None) == torch.float16

    def recomputable(module, n_in):
        weight = getattr(module, "weight", None)
        bias = getattr(module, "bias", None)
        return (weight is not None and on_gpu(weight) and weight.dtype == torch.uint8
                and not weight.requires_grad and (bias is None or not bias.requires_grad)
                and nf4.supported(getattr(weight, "quant_state", None), n_in))

    def forward(self, x):
        config = getattr(self, "config", None)
        projections = [getattr(self, name, None) for name in ("gate_proj", "up_proj", "down_proj")]
        if not (isinstance(x, torch.Tensor) and on_gpu(x) and x.dtype == torch.float32
                and getattr(config, "hidden_act", None) == "silu"
                and all(fp16_projection(p) for p in projections)):
            return original(self, x)
        gate_proj, up_proj, down_proj = projections
        if (recompute and torch.is_grad_enabled() and x.numel() > 0
                and recomputable(gate_proj, x.shape[-1]) and recomputable(up_proj, x.shape[-1])
                and recomputable(down_proj, gate_proj.weight.quant_state.shape[0])):
            link = STATE.norm_link
            if link is not None and link[0] is x and link[1].shape == x.shape:
                return RecomputedMLP.apply(x, gate_proj, up_proj, down_proj, link[1], link[2],
                                           link[3])
            return RecomputedMLP.apply(x, gate_proj, up_proj, down_proj)
        gate = gate_proj(x.to(torch.float16))
        up = up_proj(x.to(torch.float16))
        return down_proj(swiglu(gate, up))

    forward._skc = True
    cls.forward = forward

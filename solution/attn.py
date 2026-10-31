"""Attention inputs in one Triton launch each way: LoRA add + RoPE + fp16 cast for q and k.

In the base, the q/k/v projections return fp32 (bitsandbytes casts its fp16 result back to the
fp32 input dtype), peft clones q and v and adds the LoRA output in fp32, RoPE runs in fp32 as six
element-wise ops on q and on k, and SDPA's autocast casts q, k, v back to fp16. Forward and
backward that is ~15 passes over [tokens x 1536] per layer.

Here (on the padding-free path) the projections take the fp16 input bitsandbytes would make
anyway and return fp16, peft's LoRA branch runs exactly as in peft (same dropout draws, same
fp16 matmuls under autocast), and one kernel computes, for every token,
    q = fp32(base) + fp32(lora);  q_rope = q * cos + rotate_half(q) * sin  (fp32, no FMA)
and stores it in fp16, head-major, as SDPA takes it; the same for k without LoRA. Backward is the
exact transpose: dq = g * cos + rotate_half^T(g * sin) in fp32, rounded to fp16 once, which is
what both the projection and the LoRA branch receive in the base.
"""

import torch
import triton
import triton.language as tl

import torch.nn.functional as F

from . import rmsnorm, warmup
from ._state import STATE, on_gpu


@triton.jit(do_not_specialize=["n_tok"])
def skc_rope_fwd_kernel(qb_ptr, ql_ptr, k_ptr, cos_ptr, sin_ptr, qo_ptr, ko_ptr, n_tok,
                        HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, BQ: tl.constexpr,
                        BK: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    HALF: tl.constexpr = D // 2
    # q: HQ heads of D values, token-major in, head-major out
    i = tl.arange(0, BQ)
    valid = i < HQ * D
    h = i // D
    d = i % D
    j = h * D + (d + HALF) % D
    qi = tl.load(qb_ptr + t * (HQ * D) + i, mask=valid, other=0.0).to(tl.float32) + \
        tl.load(ql_ptr + t * (HQ * D) + i, mask=valid, other=0.0).to(tl.float32)
    qj = tl.load(qb_ptr + t * (HQ * D) + j, mask=valid, other=0.0).to(tl.float32) + \
        tl.load(ql_ptr + t * (HQ * D) + j, mask=valid, other=0.0).to(tl.float32)
    c = tl.load(cos_ptr + t * D + d, mask=valid, other=0.0)
    s = tl.load(sin_ptr + t * D + d, mask=valid, other=0.0)
    rot = tl.where(d < HALF, -qj, qj)
    out = qi * c + rot * s
    tl.store(qo_ptr + h * (n_tok * D) + t * D + d, out.to(tl.float16), mask=valid)
    # k: HK heads, no LoRA
    i = tl.arange(0, BK)
    valid = i < HK * D
    h = i // D
    d = i % D
    j = h * D + (d + HALF) % D
    ki = tl.load(k_ptr + t * (HK * D) + i, mask=valid, other=0.0).to(tl.float32)
    kj = tl.load(k_ptr + t * (HK * D) + j, mask=valid, other=0.0).to(tl.float32)
    c = tl.load(cos_ptr + t * D + d, mask=valid, other=0.0)
    s = tl.load(sin_ptr + t * D + d, mask=valid, other=0.0)
    rot = tl.where(d < HALF, -kj, kj)
    out = ki * c + rot * s
    tl.store(ko_ptr + h * (n_tok * D) + t * D + d, out.to(tl.float16), mask=valid)


@triton.jit(do_not_specialize=["n_tok"])
def skc_rope_bwd_kernel(gq_ptr, gk_ptr, cos_ptr, sin_ptr, dq_ptr, dk_ptr, n_tok,
                        HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, BQ: tl.constexpr,
                        BK: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    HALF: tl.constexpr = D // 2
    i = tl.arange(0, BQ)
    valid = i < HQ * D
    h = i // D
    d = i % D
    dj = (d + HALF) % D
    gi = tl.load(gq_ptr + h * (n_tok * D) + t * D + d, mask=valid, other=0.0).to(tl.float32)
    gj = tl.load(gq_ptr + h * (n_tok * D) + t * D + dj, mask=valid, other=0.0).to(tl.float32)
    a = gi * tl.load(cos_ptr + t * D + d, mask=valid, other=0.0)
    b = gj * tl.load(sin_ptr + t * D + dj, mask=valid, other=0.0)
    dq = a + tl.where(d < HALF, b, -b)
    tl.store(dq_ptr + t * (HQ * D) + i, dq.to(tl.float16), mask=valid)
    i = tl.arange(0, BK)
    valid = i < HK * D
    h = i // D
    d = i % D
    dj = (d + HALF) % D
    gi = tl.load(gk_ptr + h * (n_tok * D) + t * D + d, mask=valid, other=0.0).to(tl.float32)
    gj = tl.load(gk_ptr + h * (n_tok * D) + t * D + dj, mask=valid, other=0.0).to(tl.float32)
    a = gi * tl.load(cos_ptr + t * D + d, mask=valid, other=0.0)
    b = gj * tl.load(sin_ptr + t * D + dj, mask=valid, other=0.0)
    dk = a + tl.where(d < HALF, b, -b)
    tl.store(dk_ptr + t * (HK * D) + i, dk.to(tl.float16), mask=valid)


class FusedQKRope(torch.autograd.Function):
    """(q_base + q_lora, k) -> RoPE'd fp16 q [1, HQ, T, D] and k [1, HK, T, D]."""

    @staticmethod
    def forward(ctx, q_base, q_lora, k, cos, sin, n_heads, n_kv_heads):
        warmup.wait()
        n_tok = q_base.shape[0]
        head_dim = cos.shape[-1]
        q_out = torch.empty((n_heads, n_tok, head_dim), dtype=torch.float16, device=q_base.device)
        k_out = torch.empty((n_kv_heads, n_tok, head_dim), dtype=torch.float16, device=q_base.device)
        bq = triton.next_power_of_2(n_heads * head_dim)
        bk = triton.next_power_of_2(n_kv_heads * head_dim)
        skc_rope_fwd_kernel[(n_tok,)](q_base, q_lora, k, cos, sin, q_out, k_out, n_tok,
                                      HQ=n_heads, HK=n_kv_heads, D=head_dim, BQ=bq, BK=bk,
                                      num_warps=4, enable_fp_fusion=False)
        ctx.save_for_backward(cos, sin)
        ctx.shape = (n_tok, n_heads, n_kv_heads, head_dim)
        return q_out.unsqueeze(0), k_out.unsqueeze(0)

    @staticmethod
    def backward(ctx, grad_q, grad_k):
        cos, sin = ctx.saved_tensors
        n_tok, n_heads, n_kv_heads, head_dim = ctx.shape
        grad_q = grad_q.to(torch.float16).contiguous()
        grad_k = grad_k.to(torch.float16).contiguous()
        dq = torch.empty((n_tok, n_heads * head_dim), dtype=torch.float16, device=cos.device)
        dk = torch.empty((n_tok, n_kv_heads * head_dim), dtype=torch.float16, device=cos.device)
        bq = triton.next_power_of_2(n_heads * head_dim)
        bk = triton.next_power_of_2(n_kv_heads * head_dim)
        skc_rope_bwd_kernel[(n_tok,)](grad_q, grad_k, cos, sin, dq, dk, n_tok,
                                      HQ=n_heads, HK=n_kv_heads, D=head_dim, BQ=bq, BK=bk,
                                      num_warps=4, enable_fp_fusion=False)
        # The projection and the LoRA branch both receive round_fp16(dq) in the base.
        return dq, dq.clone(), dk, None, None, None, None


@triton.jit(do_not_specialize=["n_rows"])
def skc_keep_pack_kernel(keep_q_ptr, keep_v_ptr, bits_q_ptr, bits_v_ptr, n_rows,
                         WIDTH: tl.constexpr, BYTES: tl.constexpr):
    """The q and v dropout keep-masks (one byte 0/1 per element) packed into bits, 8 per byte."""
    row = tl.program_id(0).to(tl.int64)
    byte = tl.arange(0, BYTES)
    valid = byte < WIDTH // 8
    src = row * WIDTH + byte * 8
    acc_q = tl.zeros([BYTES], dtype=tl.int32)
    acc_v = tl.zeros([BYTES], dtype=tl.int32)
    for j in tl.static_range(8):
        acc_q = acc_q | (tl.load(keep_q_ptr + src + j, mask=valid, other=0).to(tl.int32) << j)
        acc_v = acc_v | (tl.load(keep_v_ptr + src + j, mask=valid, other=0).to(tl.int32) << j)
    dst = row * (WIDTH // 8) + byte
    tl.store(bits_q_ptr + dst, acc_q.to(tl.uint8), mask=valid)
    tl.store(bits_v_ptr + dst, acc_v.to(tl.uint8), mask=valid)


@triton.jit
def _keep(bits_ptr, row, cols, valid, WIDTH: tl.constexpr):
    byte = tl.load(bits_ptr + row * (WIDTH // 8) + cols // 8, mask=valid, other=0).to(tl.int32)
    return ((byte >> (cols % 8)) & 1).to(tl.float32)


@triton.jit(do_not_specialize=["n_rows"])
def skc_lora_inputs_kernel(x_ptr, bits_q_ptr, bits_v_ptr, scale_q_ptr, scale_v_ptr, xq_ptr, xv_ptr,
                           n_rows, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    """fp16(x * dropout values) of the q and v LoRA branches, from x and the packed keep-masks:
    x * (keep * scale) in fp32, then rounded to fp16, as the forward computed it."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    valid = cols < WIDTH
    x = tl.load(x_ptr + row * WIDTH + cols, mask=valid, other=0.0)
    mq = _keep(bits_q_ptr, row, cols, valid, WIDTH) * tl.load(scale_q_ptr)
    mv = _keep(bits_v_ptr, row, cols, valid, WIDTH) * tl.load(scale_v_ptr)
    tl.store(xq_ptr + row * WIDTH + cols, (x * mq).to(tl.float16), mask=valid)
    tl.store(xv_ptr + row * WIDTH + cols, (x * mv).to(tl.float16), mask=valid)


@triton.jit(do_not_specialize=["n_rows"])
def skc_lora_input_grad_kernel(gq_ptr, gv_ptr, bits_q_ptr, bits_v_ptr, out_ptr, n_rows, scale_q,
                               scale_v, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    """The input gradient of both LoRA branches through dropout, q's plus v's in fp32:
    (fp32(g) * keep) * 1/(1-p) for each, as PyTorch's dropout backward and autograd sum it."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    valid = cols < WIDTH
    gq = tl.load(gq_ptr + row * WIDTH + cols, mask=valid, other=0.0).to(tl.float32)
    gv = tl.load(gv_ptr + row * WIDTH + cols, mask=valid, other=0.0).to(tl.float32)
    tq = (gq * _keep(bits_q_ptr, row, cols, valid, WIDTH)) * scale_q
    tv = (gv * _keep(bits_v_ptr, row, cols, valid, WIDTH)) * scale_v
    tl.store(out_ptr + row * WIDTH + cols, tq + tv, mask=valid)


def pack_keep(keep_q, keep_v):
    """Two bool keep-masks [T, W] (W divisible by 8) -> bit-packed uint8 [T, W / 8] each."""
    n_rows, width = keep_q.shape
    bits_q = torch.empty((n_rows, width // 8), dtype=torch.uint8, device=keep_q.device)
    bits_v = torch.empty_like(bits_q)
    skc_keep_pack_kernel[(n_rows,)](keep_q.view(torch.uint8), keep_v.view(torch.uint8), bits_q,
                                    bits_v, n_rows, WIDTH=width,
                                    BYTES=triton.next_power_of_2(width // 8), num_warps=2)
    return bits_q, bits_v


class LoRAPairFromNorm(torch.autograd.Function):
    """peft's LoRA branches of q_proj and v_proj, lora_B(lora_A(dropout(x))) * scaling under fp16
    autocast, for x = the output of our RMSNorm, with the same ops in the same order.

    For lora_A's weight gradient peft keeps fp16(dropout(x)) (3 KB per token per projection)
    and dropout keeps its mask as bytes (1.5 KB). Here only the masks are kept, packed to bits
    (0.19 KB): backward rebuilds fp16(dropout(x)) from the norm's input (kept by the norm for its
    own backward) with the norm kernel (bit-identical), and two of our kernels replace the mask
    ops of both branches: the rebuilt inputs, and the input gradient through dropout."""

    @staticmethod
    def forward(ctx, x, h, norm_weight, eps, scaled_q, mask_q, scale_q, scaled_v, mask_v, scale_v,
                a_q, b_q, a_v, b_v, scaling_q, scaling_v, p_q, p_v, packed):
        outs, us = [], []
        for scaled, a, b, scaling in ((scaled_q, a_q, b_q, scaling_q), (scaled_v, a_v, b_v, scaling_v)):
            xd = x * scaled if scaled is not None else x
            u = F.linear(xd.to(torch.float16), a.to(torch.float16))
            outs.append(F.linear(u, b.to(torch.float16)) * scaling)
            us.append(u)
        ctx.save_for_backward(h, norm_weight, mask_q, scale_q, mask_v, scale_v, us[0], us[1],
                              a_q, b_q, a_v, b_v)
        ctx.eps, ctx.scalings, ctx.ps, ctx.packed = eps, (scaling_q, scaling_v), (p_q, p_v), packed
        return outs[0], outs[1]

    @staticmethod
    def backward(ctx, grad_q, grad_v):
        h, norm_weight, mask_q, scale_q, mask_v, scale_v, u_q, u_v, a_q, b_q, a_v, b_v = \
            ctx.saved_tensors
        x = rmsnorm.recompute(h, norm_weight, ctx.eps)
        width = x.shape[-1]
        n_rows = x.numel() // width
        if ctx.packed:
            x16_q = torch.empty((n_rows, width), dtype=torch.float16, device=x.device)
            x16_v = torch.empty_like(x16_q)
            block = triton.next_power_of_2(width)
            skc_lora_inputs_kernel[(n_rows,)](x, mask_q, mask_v, scale_q, scale_v, x16_q, x16_v,
                                              n_rows, WIDTH=width, BLOCK=block, num_warps=4,
                                              enable_fp_fusion=False)
            inputs = (x16_q, x16_v)
        else:
            inputs = tuple((x * (m * s) if m is not None else x).to(torch.float16).reshape(-1, width)
                           for m, s in ((mask_q, scale_q), (mask_v, scale_v)))
        grads, input_grads = [], []
        branches = ((grad_q, u_q, a_q, b_q, ctx.scalings[0]), (grad_v, u_v, a_v, b_v, ctx.scalings[1]))
        for (grad, u, a, b, scaling), x16 in zip(branches, inputs):
            if grad is None:
                grad = torch.zeros(u.shape[:-1] + (b.shape[0],), dtype=torch.float16, device=u.device)
            # the same products autograd forms for out = (u @ b16^T) * scaling, u = x16 @ a16^T
            g = (grad * scaling).reshape(-1, grad.shape[-1])
            u2 = u.reshape(-1, u.shape[-1])
            grad_b = g.t().mm(u2)
            grad_u = g.mm(b.to(torch.float16))
            grads += [grad_u.t().mm(x16).to(torch.float32), grad_b.to(torch.float32)]
            input_grads.append(grad_u.mm(a.to(torch.float16)))
        if ctx.packed:
            grad_x = torch.empty((n_rows, width), dtype=torch.float32, device=x.device)
            skc_lora_input_grad_kernel[(n_rows,)](input_grads[0], input_grads[1], mask_q, mask_v,
                                                  grad_x, n_rows, 1.0 / (1.0 - ctx.ps[0]),
                                                  1.0 / (1.0 - ctx.ps[1]), WIDTH=width,
                                                  BLOCK=triton.next_power_of_2(width), num_warps=4,
                                                  enable_fp_fusion=False)
        else:
            grad_x = None
            for gx16, m, p in zip(input_grads, (mask_q, mask_v), ctx.ps):
                gx = gx16.to(torch.float32)
                if m is not None:
                    gx = gx * m.reshape(-1, width) * (1.0 / (1.0 - p))
                grad_x = gx if grad_x is None else grad_x + gx
        return (grad_x.view(x.shape), None, None, None, None, None, None, None, None, None,
                grads[0], grads[1], grads[2], grads[3], None, None, None, None, None)


def _lora_parts(layer, name):
    """(lora_A weight, lora_B weight, scaling, dropout p) of a plain peft LoRA branch, else None."""
    try:
        lora_a, lora_b = layer.lora_A[name], layer.lora_B[name]
        dropout, scaling = layer.lora_dropout[name], layer.scaling[name]
    except (AttributeError, KeyError, TypeError):
        return None
    if not (type(lora_a) is torch.nn.Linear and type(lora_b) is torch.nn.Linear
            and lora_a.bias is None and lora_b.bias is None
            and isinstance(scaling, (int, float)) and not isinstance(scaling, bool)):
        return None
    if type(dropout) is torch.nn.Identity:
        p = 0.0
    elif (type(dropout) is torch.nn.Dropout and dropout.training and not dropout.inplace
          and 0.0 < dropout.p < 1.0):
        p = float(dropout.p)
    else:
        return None
    a, b = lora_a.weight, lora_b.weight
    if not (a.dtype == torch.float32 and b.dtype == torch.float32 and on_gpu(a) and on_gpu(b)):
        return None
    return a, b, float(scaling), p


def _lora_pair(x, link, parts_q, parts_v, packing):
    """q and v LoRA deltas through LoRAPairFromNorm; dropout draws in the base's order (q, v)."""
    _, h, norm_weight, eps = link
    draws = []
    for _, _, _, p in (parts_q, parts_v):
        if p == 0.0:
            draws.append((None, None, None))
            continue
        scaled, keep = packing.dropout_draw(p, x.shape[-1], x.dtype, x.device)
        draws.append((scaled.view(x.shape), keep, packing.dropout_scale(p, scaled)))
    (sq, kq, cq), (sv, kv, cv) = draws
    packed = kq is not None and kv is not None and x.shape[-1] % 8 == 0
    if packed:
        mq, mv = pack_keep(kq, kv)
    else:
        mq = kq.view(x.shape) if kq is not None else None
        mv = kv.view(x.shape) if kv is not None else None
    return LoRAPairFromNorm.apply(x, h, norm_weight, eps, sq, mq, cq, sv, mv, cv,
                                  parts_q[0], parts_q[1], parts_v[0], parts_v[1],
                                  parts_q[2], parts_v[2], parts_q[3], parts_v[3], packed)


def _single_lora(layer, lora_cls):
    """The active adapter name of a plain (vanilla, unmerged) peft LoRA layer, else None."""
    try:
        if not isinstance(layer, lora_cls) or layer.disable_adapters or layer.merged:
            return None
        active = list(layer.active_adapters)
        if len(active) != 1:
            return None
        name = active[0]
        if name not in layer.lora_A or name in getattr(layer, "lora_variant", {}):
            return None
        if getattr(layer, "use_dora", {}).get(name, False):
            return None
        return name
    except AttributeError:
        return None


def _lora_delta(layer, name, x):
    """peft's LoRA branch for a 4-bit layer under autocast, unchanged."""
    return layer.lora_B[name](layer.lora_A[name](layer.lora_dropout[name](x))) * layer.scaling[name]


def patch(recompute_lora=True):
    import bitsandbytes as bnb
    from peft.tuners.lora.bnb import Linear4bit as LoraLinear4bit
    from peft.tuners.lora.layer import Linear as LoraLinear
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2Attention
    if getattr(cls.forward, "_skc", False):
        return
    original = cls.forward
    linear4bit = bnb.nn.Linear4bit

    def fp16_linear(module):
        return isinstance(module, linear4bit) and getattr(module, "compute_dtype", None) == torch.float16

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None,
                **kwargs):
        packing = STATE.packing
        lora_classes = (LoraLinear4bit, LoraLinear)
        q_name = _single_lora(getattr(self, "q_proj", None), lora_classes)
        v_name = _single_lora(getattr(self, "v_proj", None), lora_classes)
        cos, sin = position_embeddings if isinstance(position_embeddings, tuple) else (None, None)
        head_dim = getattr(self, "head_dim", 0)
        if not (STATE.packing_active and packing is not None and attention_mask is None
                and past_key_values is None and isinstance(hidden_states, torch.Tensor)
                and on_gpu(hidden_states) and hidden_states.dtype == torch.float32
                and hidden_states.dim() == 3 and hidden_states.shape[0] == 1
                and hidden_states.shape[1] == packing.n_tokens and torch.is_autocast_enabled()
                and q_name is not None and v_name is not None
                and fp16_linear(self.q_proj.base_layer) and fp16_linear(self.v_proj.base_layer)
                and fp16_linear(getattr(self, "k_proj", None))
                and isinstance(cos, torch.Tensor) and cos.dtype == torch.float32
                and cos.shape == (1, packing.n_tokens, head_dim) and sin.shape == cos.shape
                and head_dim % 2 == 0 and cos.is_contiguous() and sin.is_contiguous()
                and getattr(self, "sliding_window", None) is None):
            return original(self, hidden_states, position_embeddings, attention_mask,
                            past_key_values=past_key_values, **kwargs)
        n_tok = hidden_states.shape[1]
        n_heads = self.config.num_attention_heads
        n_kv = self.config.num_key_value_heads
        x = hidden_states
        link = STATE.norm_link
        parts_q = parts_v = None
        if (recompute_lora and link is not None and link[0] is x
                and isinstance(link[1], torch.Tensor) and link[1].shape == x.shape):
            parts_q, parts_v = _lora_parts(self.q_proj, q_name), _lora_parts(self.v_proj, v_name)
        # Same call order as the base (q, k, v): the LoRA dropout draws stay in step.
        q_base = self.q_proj.base_layer(x.to(torch.float16))
        if parts_q is None or parts_v is None:
            q_lora = _lora_delta(self.q_proj, q_name, x)
        k = self.k_proj(x.to(torch.float16))
        v_base = self.v_proj.base_layer(x.to(torch.float16))
        if parts_q is None or parts_v is None:
            v_lora = _lora_delta(self.v_proj, v_name, x)
        else:
            q_lora, v_lora = _lora_pair(x, link, parts_q, parts_v, packing)
        # v in fp32 as the base adds it, then the fp16 SDPA input autocast would make.
        v = (v_base.float() + v_lora.float()).to(torch.float16)
        q_rot, k_rot = FusedQKRope.apply(
            q_base.reshape(n_tok, -1).contiguous(), q_lora.to(torch.float16).reshape(n_tok, -1).contiguous(),
            k.reshape(n_tok, -1).contiguous(), cos.reshape(n_tok, -1), sin.reshape(n_tok, -1),
            n_heads, n_kv)
        v_heads = v.view(1, n_tok, n_kv, head_dim).transpose(1, 2)
        attention = ALL_ATTENTION_FUNCTIONS["sdpa"]
        out, _ = attention(self, q_rot, k_rot, v_heads, None,
                           dropout=0.0 if not self.training else self.attention_dropout,
                           scaling=self.scaling, **kwargs)
        out = out.reshape(1, n_tok, -1).contiguous()
        return self.o_proj(out), None

    forward._skc = True
    cls.forward = forward

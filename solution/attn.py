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

from . import warmup
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


def patch():
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
        # Same call order as the base (q, k, v): the LoRA dropout draws stay in step.
        q_base = self.q_proj.base_layer(x.to(torch.float16))
        q_lora = _lora_delta(self.q_proj, q_name, x)
        k = self.k_proj(x.to(torch.float16))
        v_base = self.v_proj.base_layer(x.to(torch.float16))
        v_lora = _lora_delta(self.v_proj, v_name, x)
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

"""Padding-free decoder: the layers run on the real tokens of the batch only.

TRL pads every batch on the right to its longest row; on this data about 44% of the token
slots of a step are padding (826 real tokens per step on average against 1,467 slots). The rules
allow removing padding inside a batch in our own code (not Soup's padding_free switch, and rows
never mix across batches).

For a training forward with labels (only when the fused head runs) the decoder gets the real
tokens as one [1, T, H] sequence, the positions 0..len-1 of each row for RoPE, and attention per
row (causal SDPA on the row's own tokens: exactly what the padded batch computes for its real
positions). Everything else (norms, projections, LoRA, MLP) is per token anyway.

LoRA dropout keeps the base's random numbers: the keep-mask is drawn with the same op on the
same [B, S, H] shape as the base draws it, and only its real-token rows are used, so the training
run is the same run as the base, not a statistically equivalent one.
"""

import torch

from ._state import STATE, on_gpu


class Packing:
    def __init__(self, index, positions, spans, batch, seq):
        self.index = index          # [T] flat positions of the real tokens in the [B, S] batch
        self.positions = positions  # [T] position of each real token in its row
        self.spans = spans          # [(start, length)] of each row in the packed sequence
        self.batch, self.seq = batch, seq
        self.n_tokens = int(index.numel())
        self._ones = {}
        self._scales = {}

    def dropout_draw(self, p, width, dtype, device):
        """What F.dropout draws for a [B, S, width] input, at the real tokens only: the scaled
        values and the keep-mask."""
        key = (width, dtype, device)
        ones = self._ones.get(key)
        if ones is None:
            ones = torch.ones((self.batch, self.seq, width), dtype=dtype, device=device)
            self._ones[key] = ones
        scaled, keep = torch.native_dropout(ones, p, True)
        return (scaled.reshape(-1, width).index_select(0, self.index),
                keep.reshape(-1, width).index_select(0, self.index))

    def dropout_keep(self, p, width, dtype, device):
        """The keep-mask of the same draw as dropout_draw (same op, same shape, so the same random
        numbers), at the real tokens, and the kept value as a 0-dim tensor."""
        key = (width, dtype, device)
        ones = self._ones.get(key)
        if ones is None:
            ones = torch.ones((self.batch, self.seq, width), dtype=dtype, device=device)
            self._ones[key] = ones
        scaled, keep = torch.native_dropout(ones, p, True)
        return keep.reshape(-1, width).index_select(0, self.index), self.dropout_scale(p, scaled)

    def dropout_scale(self, p, scaled):
        """The kept value of a draw (1 / (1 - p) as the dropout kernel rounds it), as a 0-dim
        tensor on the device: keep * scale rebuilds the draw's values exactly."""
        key = (p, scaled.dtype, scaled.device)
        scale = self._scales.get(key)
        if scale is None:
            scale = scaled.amax()
            self._scales[key] = scale
        return scale


def _row_attention(q, k, v, groups, scale):
    """Causal SDPA of one row; query head h reads kv head h // groups (repeat_kv)."""
    if groups > 1:
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=scale,
                                                            is_causal=True)


class PackedSDPA(torch.autograd.Function):
    """Causal attention over every row of the packed sequence, keeping only q, k, v for backward.

    Per-row SDPA under autograd keeps the GQA-expanded k and v and its output for backward:
    12 KB per token per layer, about 540 MB on the longest batch. Here backward re-runs each
    row's forward (the attention forward is about 1 % of the step) right before that row's
    backward, so only q, k and v are kept (4 KB). The rows' outputs go straight into one
    [1, T, heads, head_dim] buffer (no cat and transpose copies), and the gradients straight
    into their rows (no zero-filled full-size gradient per row)."""

    @staticmethod
    def forward(ctx, query, key, value, spans, groups, scale):
        out = query.new_empty((1, query.shape[2], query.shape[1], query.shape[3]))
        for start, length in spans:
            rows = slice(start, start + length)
            o = _row_attention(query[:, :, rows], key[:, :, rows], value[:, :, rows], groups, scale)
            out[:, rows].copy_(o.transpose(1, 2))
        ctx.save_for_backward(query, key, value)
        ctx.spans, ctx.groups, ctx.scale = spans, groups, scale
        return out

    @staticmethod
    def backward(ctx, grad_out):
        query, key, value = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(query), torch.empty_like(key), torch.empty_like(value)
        for start, length in ctx.spans:
            rows = slice(start, start + length)
            q = query[:, :, rows].detach().requires_grad_(True)
            k = key[:, :, rows].detach().requires_grad_(True)
            v = value[:, :, rows].detach().requires_grad_(True)
            with torch.enable_grad():
                o = _row_attention(q, k, v, ctx.groups, ctx.scale)
            gq, gk, gv = torch.autograd.grad(o, (q, k, v), grad_out[:, rows].transpose(1, 2))
            dq[:, :, rows] = gq
            dk[:, :, rows] = gk
            dv[:, :, rows] = gv
        return dq, dk, dv, None, None, None


class PackedDropout(torch.autograd.Function):
    """x * mask * scale with a given mask, saved as bool for backward like native dropout."""

    @staticmethod
    def forward(ctx, x, scaled, keep, p):
        ctx.save_for_backward(keep)
        ctx.scale = 1.0 / (1.0 - p)
        return x * scaled

    @staticmethod
    def backward(ctx, grad):
        (keep,) = ctx.saved_tensors
        return grad * keep * ctx.scale, None, None, None


def plan(model, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds,
         use_cache, kwargs):
    """A Packing for this forward, or None to run the original decoder."""
    config = getattr(model, "config", None)
    if not (STATE.request and model.training and torch.is_grad_enabled()
            and isinstance(input_ids, torch.Tensor) and isinstance(attention_mask, torch.Tensor)
            and input_ids.dim() == 2 and attention_mask.shape == input_ids.shape
            and inputs_embeds is None and position_ids is None and past_key_values is None
            and not use_cache and on_gpu(input_ids)
            and getattr(config, "_attn_implementation", None) == "sdpa"
            and not getattr(model, "has_sliding_layers", False)
            and all(t == "full_attention" for t in getattr(config, "layer_types", ["?"]))
            and not kwargs.get("output_attentions") and not kwargs.get("output_hidden_states")
            # recomputation would replay the layers outside our packing context
            and not getattr(model, "gradient_checkpointing", False)
            and not any(getattr(layer, "gradient_checkpointing", False)
                        for layer in getattr(model, "layers", []))):
        return None
    batch, seq = input_ids.shape
    mask = attention_mask.to(torch.bool)
    lengths = mask.sum(dim=1)
    steps = torch.arange(seq, device=mask.device)
    if not torch.equal(mask, steps.unsqueeze(0) < lengths.unsqueeze(1)):
        return None  # not right-padded: keep the original path
    lens = lengths.tolist()
    if min(lens) == 0:
        return None  # an empty row
    # A batch without padding still takes this path: the packed layers also keep less for
    # backward (attention, LoRA inputs), and the longest batch sets the peak.
    index = mask.reshape(-1).nonzero().squeeze(1)
    positions = steps.unsqueeze(0).expand(batch, seq).reshape(-1).index_select(0, index)
    spans, start = [], 0
    for length in lens:
        spans.append((start, int(length)))
        start += int(length)
    return Packing(index, positions, spans, batch, seq)


def patch(recompute_attention=True, fuse_residual=True):
    import torch.nn.functional as F
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen2 import modeling_qwen2

    model_cls = modeling_qwen2.Qwen2Model
    if getattr(model_cls.forward, "_skc", False):
        return
    model_forward = model_cls.forward

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs):
        packing = plan(self, input_ids, attention_mask, position_ids, past_key_values,
                       inputs_embeds, use_cache, kwargs)
        if packing is None:
            return model_forward(self, input_ids=input_ids, attention_mask=attention_mask,
                                 position_ids=position_ids, past_key_values=past_key_values,
                                 inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        ids = input_ids.reshape(-1).index_select(0, packing.index).unsqueeze(0)
        positions = packing.positions.unsqueeze(0)
        hidden = self.embed_tokens(ids)
        position_embeddings = self.rotary_emb(hidden, positions)
        STATE.packing, STATE.packing_active = packing, True
        try:
            for layer in self.layers[: self.config.num_hidden_layers]:
                hidden = layer(hidden, attention_mask=None, position_embeddings=position_embeddings,
                               position_ids=positions, past_key_values=None, use_cache=False,
                               **kwargs)
        finally:
            STATE.packing_active = False
            STATE.norm_link = None
        hidden = self.norm(hidden)
        return BaseModelOutputWithPast(last_hidden_state=hidden, past_key_values=None)

    forward._skc = True
    model_cls.forward = forward

    sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]

    def sdpa_packed(module, query, key, value, attention_mask, dropout=0.0, scaling=None,
                    is_causal=None, **kwargs):
        packing = STATE.packing
        if not (STATE.packing_active and packing is not None and attention_mask is None
                and query.dim() == 4 and query.shape[0] == 1
                and query.shape[2] == packing.n_tokens and key.shape[2] == packing.n_tokens
                and kwargs.get("position_bias") is None):
            return sdpa(module, query, key, value, attention_mask, dropout=dropout,
                        scaling=scaling, is_causal=is_causal, **kwargs)
        groups = getattr(module, "num_key_value_groups", 1)
        if (recompute_attention and dropout == 0.0 and query.dtype == torch.float16
                and key.dtype == torch.float16 and value.dtype == torch.float16
                and query.stride(-1) == 1 and key.stride(-1) == 1 and value.stride(-1) == 1):
            # fp16 q/k/v come from attn.py; SDPA would run on them as they are under autocast
            return PackedSDPA.apply(query, key, value, packing.spans, groups, scaling), None
        outputs = []
        for start, length in packing.spans:
            q = query[:, :, start:start + length]
            k = key[:, :, start:start + length]
            v = value[:, :, start:start + length]
            if groups > 1:  # repeat_kv: query head h reads kv head h // groups
                k = k.repeat_interleave(groups, dim=1)
                v = v.repeat_interleave(groups, dim=1)
            outputs.append(F.scaled_dot_product_attention(q, k, v, dropout_p=dropout,
                                                          scale=scaling, is_causal=True))
        return torch.cat(outputs, dim=2).transpose(1, 2).contiguous(), None

    ALL_ATTENTION_FUNCTIONS["sdpa"] = sdpa_packed

    from . import rmsnorm

    layer_cls = modeling_qwen2.Qwen2DecoderLayer
    layer_forward = layer_cls.forward

    def decoder_layer(self, hidden_states, attention_mask=None, position_ids=None,
                      past_key_values=None, use_cache=False, position_embeddings=None, **kwargs):
        packing = STATE.packing
        post_norm = getattr(self, "post_attention_layernorm", None)
        if not (fuse_residual and STATE.packing_active and packing is not None
                and attention_mask is None and past_key_values is None and not use_cache
                and isinstance(hidden_states, torch.Tensor) and hidden_states.dim() == 3
                and hidden_states.shape[0] == 1 and hidden_states.shape[1] == packing.n_tokens
                and rmsnorm.fusable(post_norm, hidden_states)):
            return layer_forward(self, hidden_states, attention_mask=attention_mask,
                                 position_ids=position_ids, past_key_values=past_key_values,
                                 use_cache=use_cache, position_embeddings=position_embeddings,
                                 **kwargs)
        # Qwen2DecoderLayer.forward with the residual add and the post-attention norm fused:
        # the norm's input is rebuilt in backward from the layer input and the fp16 attention
        # output instead of being kept (3 KB per token per layer instead of 6).
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out, _ = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask,
                                     position_ids=position_ids, past_key_values=past_key_values,
                                     use_cache=use_cache, position_embeddings=position_embeddings,
                                     **kwargs)
        if (isinstance(attn_out, torch.Tensor) and attn_out.dtype == torch.float16
                and attn_out.shape == residual.shape):
            eps = float(post_norm.variance_epsilon)
            hidden_states, normed = rmsnorm.AddRMSNormFn.apply(residual, attn_out, post_norm.weight,
                                                               eps)
            STATE.norm_link = (normed, (residual, attn_out), post_norm.weight, eps)
        else:
            hidden_states = residual + attn_out
            normed = post_norm(hidden_states)
        residual = hidden_states
        hidden_states = self.mlp(normed)
        return residual + hidden_states

    decoder_layer._skc = True
    layer_cls.forward = decoder_layer

    dropout_forward = torch.nn.Dropout.forward

    def dropout(self, input):
        packing = STATE.packing
        if (STATE.packing_active and packing is not None and self.training and 0.0 < self.p < 1.0
                and not self.inplace and isinstance(input, torch.Tensor) and input.dim() == 3
                and input.shape[0] == 1 and input.shape[1] == packing.n_tokens):
            scaled, keep = packing.dropout_draw(self.p, input.shape[-1], input.dtype, input.device)
            return PackedDropout.apply(input, scaled.unsqueeze(0), keep.unsqueeze(0), self.p)
        return dropout_forward(self, input)

    torch.nn.Dropout.forward = dropout

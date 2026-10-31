"""Output head, loss and TRL's metrics without the [tokens x 151,936] logits tensor.

The base computes fp16 logits for every position of the padded batch, upcasts them to fp32 for
the loss (log_softmax kept for backward, a dense fp32 gradient, its fp16 cast: ~3.5 GiB at a
4 x 512 batch), and TRL then reads the logits again for entropy and token accuracy, and Soup's
rewind recorder once more for per-row losses.

Here the head runs on the real tokens only, CHUNK_ROWS at a time: cuBLAS computes one chunk of
fp16 logits (as autocast does), and our Triton kernel makes one pass over each row for
logsumexp, entropy, argmax and the target's loss, and a second pass, for rows with a label,
that writes softmax - onehot over the logits in place. One more matmul turns that into the
gradient of the hidden states, kept in fp32 until backward scales it by the incoming gradient
(the fp16 loss scale) and 1 / num_items_in_batch. The chunk buffer is the only vocabulary-sized
tensor (CHUNK_ROWS x 151,936 fp16 = 148 MiB).

The metrics are logged by TRL's own formulas (entropy over all real tokens, accuracy over the
labelled ones, num_tokens from the attention mask), and the per-token losses go to Soup's
rewind recorder through the hook Soup itself provides for steps without logits
(`_soup_chunked_step.token_loss`, read by soup_cli.trainer.rewind_hf).
"""

import types

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from . import warmup
from ._state import STATE, on_gpu

IGNORE_INDEX = -100
CHUNK_ROWS = 512
BLOCK_V = 2048
STEP_ATTR = "_soup_chunked_step"


@triton.jit(do_not_specialize=["n_rows"])
def skc_head_stats_kernel(logits_ptr, target_ptr, lse_ptr, entropy_ptr, loss_ptr, hit_ptr,
                          n_rows, stride, n_vocab, BLOCK_V: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    base = logits_ptr + row * stride
    target = tl.load(target_ptr + row).to(tl.int32)
    labelled = target >= 0
    target = tl.where(labelled, target, 0)
    lanes = tl.arange(0, BLOCK_V)
    # Per-lane running max, sum of exp and sum of z * exp; first index of each lane's max.
    m_lane = tl.full([BLOCK_V], float("-inf"), tl.float32)
    s_lane = tl.zeros([BLOCK_V], tl.float32)
    w_lane = tl.zeros([BLOCK_V], tl.float32)
    at_lane = tl.zeros([BLOCK_V], tl.int32)
    for start in range(0, n_vocab, BLOCK_V):
        cols = start + lanes
        valid = cols < n_vocab
        z = tl.load(base + cols, mask=valid, other=float("-inf")).to(tl.float32)
        take = z > m_lane
        at_lane = tl.where(take, cols, at_lane)
        m_new = tl.maximum(m_lane, z)
        alpha = tl.where(m_lane == m_new, 1.0, tl.exp(m_lane - m_new))
        e = tl.where(valid, tl.exp(z - m_new), 0.0)
        s_lane = s_lane * alpha + e
        w_lane = w_lane * alpha + tl.where(valid, z * e, 0.0)
        m_lane = m_new
    m = tl.max(m_lane, axis=0)
    scale = tl.where(m_lane == float("-inf"), 0.0, tl.exp(m_lane - m))
    s = tl.sum(s_lane * scale, axis=0)
    w = tl.sum(w_lane * scale, axis=0)
    lse = m + tl.log(s)
    entropy = lse - w / s
    # torch.argmax returns the first maximal index.
    best_at = tl.min(tl.where(m_lane == m, at_lane, n_vocab), axis=0)
    z_target = tl.load(base + target).to(tl.float32)
    tl.store(lse_ptr + row, lse)
    tl.store(entropy_ptr + row, entropy)
    tl.store(loss_ptr + row, tl.where(labelled, lse - z_target, 0.0))
    tl.store(hit_ptr + row, (labelled & (best_at == target)).to(tl.int32))
    if labelled:
        for start in range(0, n_vocab, BLOCK_V):
            cols = start + lanes
            valid = cols < n_vocab
            z = tl.load(base + cols, mask=valid, other=0.0).to(tl.float32)
            p = tl.exp(z - lse)
            g = tl.where(cols == target, p - 1.0, p)
            tl.store(base + cols, g.to(tl.float16), mask=valid)


class FusedHeadLoss(torch.autograd.Function):
    """(sum of the labelled tokens' cross-entropy) / denominator, plus per-token entropy, loss
    and argmax hit; the gradient flows to `hidden` only (the head is frozen)."""

    @staticmethod
    def forward(ctx, hidden, weight16, targets, denominator):
        warmup.wait()
        n_tok, n_hidden = hidden.shape
        n_vocab = weight16.shape[0]
        device = hidden.device
        with torch.autocast(device_type=device.type, enabled=False):
            labelled = targets != IGNORE_INDEX
            # Labelled tokens first: their gradient matmul then runs on a contiguous prefix.
            order = torch.argsort((~labelled).to(torch.int8), stable=True)
            n_lab = int(labelled.sum().item())
            target_sorted = targets.index_select(0, order).contiguous()
            h16 = hidden.detach().index_select(0, order).to(torch.float16)
            lse = torch.empty(n_tok, dtype=torch.float32, device=device)
            entropy = torch.empty(n_tok, dtype=torch.float32, device=device)
            token_loss = torch.empty(n_tok, dtype=torch.float32, device=device)
            hit = torch.empty(n_tok, dtype=torch.int32, device=device)
            grad = torch.empty((n_lab, n_hidden), dtype=torch.float32, device=device)
            buffer = torch.empty((max(1, min(CHUNK_ROWS, n_tok)), n_vocab), dtype=torch.float16,
                                 device=device)
            for start in range(0, n_tok, CHUNK_ROWS):
                stop = min(start + CHUNK_ROWS, n_tok)
                rows = stop - start
                logits = buffer[:rows]
                torch.mm(h16[start:stop], weight16.t(), out=logits)
                skc_head_stats_kernel[(rows,)](logits, target_sorted[start:stop], lse[start:stop],
                                               entropy[start:stop], token_loss[start:stop],
                                               hit[start:stop], rows, logits.stride(0), n_vocab,
                                               BLOCK_V=BLOCK_V, num_warps=8)
                with_grad = min(stop, n_lab) - start
                if with_grad > 0:
                    grad[start:start + with_grad] = torch.mm(logits[:with_grad], weight16)
            loss = token_loss.sum() / denominator
            # Back to the caller's token order.
            entropy_out = torch.empty_like(entropy).index_copy_(0, order, entropy)
            loss_out = torch.empty_like(token_loss).index_copy_(0, order, token_loss)
            hit_out = torch.empty_like(hit).index_copy_(0, order, hit)
        ctx.save_for_backward(grad, order[:n_lab])
        ctx.n_tok, ctx.dtype, ctx.denominator = n_tok, hidden.dtype, denominator
        ctx.mark_non_differentiable(entropy_out, loss_out, hit_out)
        return loss, entropy_out, loss_out, hit_out

    @staticmethod
    def backward(ctx, grad_loss, *unused):
        grad, positions = ctx.saved_tensors
        grad_hidden = torch.zeros((ctx.n_tok, grad.shape[1]), dtype=torch.float32,
                                  device=grad.device)
        grad_hidden.index_copy_(0, positions, grad * (grad_loss.float() / ctx.denominator))
        return grad_hidden.to(ctx.dtype), None, None, None


def shift_labels(labels):
    """Position t is scored against token t + 1, as ForCausalLMLoss does."""
    return F.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:]


def run_head(model, hidden, labels, attention_mask, num_items_in_batch, packing):
    batch, seq = labels.shape
    n_hidden = hidden.shape[-1]
    if packing is not None:  # the decoder already ran on the real tokens only
        positions = packing.index
        flat = hidden.reshape(-1, n_hidden)
    else:
        positions = attention_mask.reshape(-1).nonzero().squeeze(1)
        flat = hidden.reshape(-1, n_hidden).index_select(0, positions)
    targets = shift_labels(labels).reshape(-1).index_select(0, positions).to(flat.device)
    weight = model.lm_head.weight
    weight16 = weight if weight.dtype == torch.float16 else weight.to(torch.float16)
    labelled = (targets != IGNORE_INDEX).sum()
    if num_items_in_batch is None:
        denominator = labelled.to(torch.float32)
    elif torch.is_tensor(num_items_in_batch):
        denominator = num_items_in_batch.to(flat.device)
    else:
        denominator = num_items_in_batch
    loss, entropy, token_loss, hit = FusedHeadLoss.apply(flat, weight16, targets, denominator)
    loss_grid = torch.zeros(batch * seq, dtype=torch.float32, device=flat.device)
    loss_grid.index_copy_(0, positions, token_loss)
    STATE.stats = types.SimpleNamespace(entropy_sum=entropy.sum(), correct=hit.sum(),
                                        labelled=labelled, token_loss=loss_grid.view(batch, seq))
    return loss


def _causal_lm(model):
    get_base_model = getattr(model, "get_base_model", None)
    return get_base_model() if callable(get_base_model) else model


def patch():
    import trl
    from transformers import Trainer
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.models.qwen2 import modeling_qwen2

    lm_cls = modeling_qwen2.Qwen2ForCausalLM
    if not getattr(lm_cls.forward, "_skc", False):
        lm_forward = lm_cls.forward

        def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                    past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                    logits_to_keep=0, **kwargs):
            head = getattr(self, "lm_head", None)
            weight = getattr(head, "weight", None)
            if not (STATE.request and self.training and torch.is_grad_enabled()
                    and isinstance(labels, torch.Tensor) and isinstance(input_ids, torch.Tensor)
                    and isinstance(attention_mask, torch.Tensor) and inputs_embeds is None
                    and attention_mask.dim() == 2 and labels.shape == input_ids.shape
                    and attention_mask.shape == input_ids.shape and on_gpu(input_ids)
                    and isinstance(logits_to_keep, int) and logits_to_keep == 0
                    and "shift_labels" not in kwargs and isinstance(weight, torch.Tensor)
                    and on_gpu(weight) and not weight.requires_grad
                    and getattr(self.config, "final_logit_softcapping", None) is None):
                return lm_forward(self, input_ids=input_ids, attention_mask=attention_mask,
                                  position_ids=position_ids, past_key_values=past_key_values,
                                  inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                                  logits_to_keep=logits_to_keep, **kwargs)
            num_items_in_batch = kwargs.pop("num_items_in_batch", None)
            STATE.packing = None
            try:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                     position_ids=position_ids, past_key_values=past_key_values,
                                     inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
                packing = STATE.packing
            finally:
                STATE.packing = None
            loss = run_head(self, outputs.last_hidden_state, labels, attention_mask,
                            num_items_in_batch, packing)
            return CausalLMOutputWithPast(loss=loss, logits=None)

        forward._skc = True
        lm_cls.forward = forward

    trainer_cls = trl.SFTTrainer
    if getattr(trainer_cls.compute_loss, "_skc", False):
        return
    trl_compute_loss = trainer_cls.compute_loss

    def applicable(trainer, model, inputs):
        args = getattr(trainer, "args", None)
        return (isinstance(model, torch.nn.Module) and isinstance(inputs, dict)
                and getattr(getattr(trainer, "model", None), "training", False)
                and isinstance(inputs.get("labels"), torch.Tensor)
                and isinstance(inputs.get("attention_mask"), torch.Tensor)
                and isinstance(inputs.get("input_ids"), torch.Tensor)
                and on_gpu(inputs["input_ids"])
                and "shift_labels" not in inputs and "position_ids" not in inputs
                and not getattr(args, "use_liger_kernel", False)
                and getattr(trainer, "num_virtual_tokens", 0) == 0
                and not getattr(trainer, "aux_loss_enabled", False)
                and getattr(trainer, "compute_loss_func", None) is None
                and getattr(trainer, "label_smoother", None) is None
                and isinstance(getattr(trainer, "_metrics", None), dict)
                and isinstance(_causal_lm(model), lm_cls))

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not applicable(self, model, inputs):
            return trl_compute_loss(self, model, inputs, return_outputs=return_outputs,
                                    num_items_in_batch=num_items_in_batch)
        mode = "train" if self.model.training else "eval"
        # As TRL's compute_loss does before it delegates.
        inputs.pop("_prediction_loss_only", None)
        inputs["use_cache"] = False
        STATE.request, STATE.stats = True, None
        try:
            loss, outputs = Trainer.compute_loss(self, model, inputs, return_outputs=True,
                                                 num_items_in_batch=num_items_in_batch)
        finally:
            STATE.request = False
        stats, STATE.stats = STATE.stats, None
        if stats is None:
            raise RuntimeError("solution.head: the fused head did not run for a training batch")
        attention_mask = inputs["attention_mask"]
        entropy_sum, n_tokens, correct, labelled = torch.stack([
            stats.entropy_sum.float(), attention_mask.sum().float(), stats.correct.float(),
            stats.labelled.float()]).tolist()
        # TRL's formulas: entropy over all real tokens, accuracy over the labelled ones.
        self._metrics[mode]["entropy"].append(entropy_sum / n_tokens)
        if mode == "train":
            self._total_train_tokens += int(n_tokens)
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]
        self._metrics[mode]["mean_token_accuracy"].append(
            correct / labelled if labelled > 0 else 0.0)
        # Per-token losses for Soup's rewind recorder (it reads them when outputs carry no logits).
        setattr(self, STEP_ATTR, types.SimpleNamespace(token_loss=stats.token_loss))
        return (loss, outputs) if return_outputs else loss

    compute_loss._skc = True
    trainer_cls.compute_loss = compute_loss

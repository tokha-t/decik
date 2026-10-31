"""Patched vs original on a tiny Qwen2 + LoRA (CPU, fp32, Triton interpreter).

Checks the plumbing of head.py and varlen.py: loss, TRL's entropy and accuracy, Soup's per-row
rewind losses, and the LoRA gradients, with LoRA dropout on (same seed -> same draws).
"""
import os
import sys

os.environ["TRITON_INTERPRET"] = "1"
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, sys.argv[1])
import solution._state as st  # noqa: E402
from solution import head, rmsnorm, varlen  # noqa: E402

for mod in (st, rmsnorm, head, varlen):
    mod.on_gpu = lambda t: True
head.BLOCK_V = 128

from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402
from trl.trainer.utils import entropy_from_logits  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def rel(a, b):
    return (a - b).abs().max().item() / max(b.abs().max().item(), 1e-12)


torch.manual_seed(0)
cfg = Qwen2Config(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                  tie_word_embeddings=True, rms_norm_eps=1e-6)
cfg._attn_implementation = "sdpa"
base = Qwen2ForCausalLM(cfg)
with torch.no_grad():  # fp16-representable head, as the real (fp16-loaded) model
    base.lm_head.weight.copy_(base.lm_head.weight.half().float())
base.requires_grad_(False)
model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.05,
                                        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
for name, p in model.named_parameters():  # non-zero B so LoRA matters in the gradients
    if "lora_B" in name:
        torch.nn.init.normal_(p, std=0.05)
model.train()

B, S = 4, 21
lengths = [21, 9, 15, 4]
input_ids = torch.randint(0, cfg.vocab_size, (B, S))
attention_mask = torch.zeros(B, S, dtype=torch.long)
labels = torch.full((B, S), -100)
for i, L in enumerate(lengths):
    attention_mask[i, :L] = 1
    labels[i, L // 2:L] = input_ids[i, L // 2:L]  # the "answer" half is supervised
input_ids[attention_mask == 0] = 0
num_items = (labels[:, 1:] != -100).sum()


def lora_grads():
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def run_original(seed):
    model.zero_grad()
    torch.manual_seed(seed)
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                num_items_in_batch=num_items)
    out.loss.backward()
    logits = out.logits.detach()
    ent = entropy_from_logits(logits)
    entropy = (ent * attention_mask).sum() / attention_mask.sum()
    pred = logits[:, :-1].argmax(-1)
    tgt = labels[:, 1:]
    mask = tgt != -100
    accuracy = ((pred == tgt) & mask).sum() / mask.sum()
    rows = []
    for i in range(B):
        ce = F.cross_entropy(logits[i, :-1], tgt[i], reduction="none", ignore_index=-100)
        rows.append(float((ce * mask[i]).sum() / mask[i].sum().clamp(min=1)))
    return out.loss.detach(), entropy, accuracy, torch.tensor(rows), lora_grads()


def run_patched(seed):
    model.zero_grad()
    torch.manual_seed(seed)
    st.STATE.request, st.STATE.stats = True, None
    try:
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                    num_items_in_batch=num_items)
    finally:
        st.STATE.request = False
    stats = st.STATE.stats
    out.loss.backward()
    entropy = stats.entropy_sum / attention_mask.sum()
    accuracy = stats.correct / stats.labelled
    tok = stats.token_loss
    n = (labels[:, 1:] != -100).sum(1)
    rows = tok.sum(1) / n.clamp(min=1)  # soup's row_losses_from_token_losses
    return out.loss.detach(), entropy, accuracy, rows, lora_grads(), out


ref = run_original(7)
rmsnorm.patch()
head.patch()
ours_padded = None
for label, use_varlen in (("head only", False), ("head + varlen", True)):
    if use_varlen:
        varlen.patch()
    loss, entropy, accuracy, rows, grads, out = run_patched(7)
    check(f"[{label}] logits not materialized", out.logits is None)
    check(f"[{label}] loss", abs(loss - ref[0]) / ref[0] < 2e-3, f"{loss.item():.6f} vs {ref[0].item():.6f}")
    check(f"[{label}] entropy", abs(entropy - ref[1]) / ref[1] < 2e-3, f"{entropy.item():.6f} vs {ref[1].item():.6f}")
    check(f"[{label}] accuracy", abs(accuracy - ref[2]) < 1e-6, f"{accuracy.item():.4f} vs {ref[2].item():.4f}")
    check(f"[{label}] rewind row losses", rel(rows, ref[3]) < 3e-3, f"rel {rel(rows, ref[3]):.2e}")
    worst = max(rel(grads[k], ref[4][k]) for k in grads)
    check(f"[{label}] LoRA grads (same dropout draws)", worst < 2e-2, f"worst rel {worst:.2e} over {len(grads)} tensors")
    if not use_varlen:
        ours_padded = (loss, grads)
    else:
        worst_vs_padded = max(rel(grads[k], ours_padded[1][k]) for k in grads)
        check("[varlen vs padded] LoRA grads identical up to fp32 rounding", worst_vs_padded < 1e-4,
              f"worst rel {worst_vs_padded:.2e}")

# outside a training request the model behaves exactly as before (eval / generation path)
model.eval()
with torch.no_grad():
    out = model(input_ids=input_ids, attention_mask=attention_mask)
check("eval forward untouched (logits returned)", out.logits is not None and out.logits.shape == (B, S, cfg.vocab_size))

print("\nALL MODEL TESTS PASSED" if not failures else f"\nFAILED: {failures}")
sys.exit(1 if failures else 0)

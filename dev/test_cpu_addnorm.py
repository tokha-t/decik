"""Fused residual add + RMSNorm (C5), on CPU with the Triton interpreter.

1. AddRMSNormFn vs `h = residual + delta; y = RMSNormFn(h)` under autograd: outputs and gradients
   bit-identical, and it keeps no fp32 copy of h.
2. A tiny Qwen2 + LoRA (4-bit attention and MLP, recomputed MLP) on the padding-free path, with the fused
   decoder layer vs without it: loss and every LoRA gradient bit-identical, fused path taken, less kept for
   backward.

Usage: python dev/test_cpu_addnorm.py <repo>
"""
import os
import sys

os.environ["TRITON_INTERPRET"] = "1"
import torch  # noqa: E402

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else ".")
import solution._state as st  # noqa: E402
from solution import attn, head, mlp, nf4, rmsnorm, varlen  # noqa: E402

for m in (st, attn, head, mlp, nf4, rmsnorm, varlen):
    m.on_gpu = lambda t: True
head.BLOCK_V = 128
fails = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        fails.append(name)


# 1. the function
torch.manual_seed(0)
T, H = 23, 256
w = torch.rand(H) + 0.5
res0 = torch.randn(1, T, H) * 2
delta0 = (torch.randn(1, T, H) * 0.5).half()
g_h = torch.randn(1, T, H)
g_y = torch.randn(1, T, H)

r1, d1 = res0.clone().requires_grad_(True), delta0.clone().requires_grad_(True)
h1 = r1 + d1
y1 = rmsnorm.RMSNormFn.apply(h1, w, 1e-6)
torch.autograd.backward([h1, y1], [g_h, g_y])

r2, d2 = res0.clone().requires_grad_(True), delta0.clone().requires_grad_(True)
saved = []
with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
    h2, y2 = rmsnorm.AddRMSNormFn.apply(r2, d2, w, 1e-6)
torch.autograd.backward([h2, y2], [g_h, g_y])
check("h bit-exact", torch.equal(h1, h2))
check("y bit-exact", torch.equal(y1, y2))
check("grad residual bit-exact", torch.equal(r1.grad, r2.grad))
check("grad delta bit-exact (fp16)", torch.equal(d1.grad, d2.grad) and d2.grad.dtype == torch.float16)
fp32_big = [t for t in saved if t.dtype == torch.float32 and t.numel() >= T * H
            and t.untyped_storage().data_ptr() != r2.untyped_storage().data_ptr()]
check("keeps no fp32 copy of h (only the residual it was given)", not fp32_big,
      f"{[(tuple(t.shape), str(t.dtype)) for t in saved]}")

# 2. the model, fused layer vs not
import bitsandbytes as bnb  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402
from transformers.models.qwen2 import modeling_qwen2  # noqa: E402

torch.manual_seed(0)
cfg = Qwen2Config(vocab_size=300, hidden_size=256, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=128,
                  tie_word_embeddings=True)
cfg._attn_implementation = "sdpa"
base = Qwen2ForCausalLM(cfg)


def q4(lin):
    layer = bnb.nn.Linear4bit(lin.in_features, lin.out_features, bias=lin.bias is not None,
                              compute_dtype=torch.float16, compress_statistics=True, quant_type="nf4")
    layer.weight = bnb.nn.Params4bit(lin.weight.data.half(), requires_grad=False, compress_statistics=True,
                                     quant_type="nf4")._quantize("cpu")
    if lin.bias is not None:
        layer.bias = torch.nn.Parameter(lin.bias.data.half().float(), requires_grad=False)
    return layer


for layer in base.model.layers:
    a = layer.self_attn
    a.q_proj, a.k_proj, a.v_proj, a.o_proj = q4(a.q_proj), q4(a.k_proj), q4(a.v_proj), q4(a.o_proj)
    f = layer.mlp
    f.gate_proj, f.up_proj, f.down_proj = q4(f.gate_proj), q4(f.up_proj), q4(f.down_proj)
with torch.no_grad():
    base.lm_head.weight.copy_(base.lm_head.weight.half().float())
base.requires_grad_(False)
model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.05,
                                        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
for n, p in model.named_parameters():
    if "lora_B" in n:
        torch.nn.init.normal_(p, std=0.05)
model.train()
B, S = 3, 17
ids = torch.randint(0, 300, (B, S))
am = torch.zeros(B, S, dtype=torch.long)
lab = torch.full((B, S), -100)
for i, L in enumerate([17, 6, 11]):
    am[i, :L] = 1
    lab[i, L // 2:L] = ids[i, L // 2:L]
nitems = (lab[:, 1:] != -100).sum()

original_layer = modeling_qwen2.Qwen2DecoderLayer.forward
nf4.patch()
rmsnorm.patch()
mlp.patch(recompute=True)
head.patch()
varlen.patch()
attn.patch()
fused_layer = modeling_qwen2.Qwen2DecoderLayer.forward
calls = {"addnorm": 0, "mlp_tuple": 0}
orig_addnorm = rmsnorm.AddRMSNormFn.forward
orig_mlp = mlp.RecomputedMLP.forward


def counting_addnorm(ctx, *a):
    calls["addnorm"] += 1
    return orig_addnorm(ctx, *a)


def counting_mlp(ctx, x, g, u, d, norm_input=None, norm_weight=None, eps=None, norm_delta=None):
    calls["mlp_tuple"] += norm_delta is not None
    return orig_mlp(ctx, x, g, u, d, norm_input, norm_weight, eps, norm_delta)


rmsnorm.AddRMSNormFn.forward = staticmethod(counting_addnorm)
mlp.RecomputedMLP.forward = staticmethod(counting_mlp)
real_autocast = torch.is_autocast_enabled
torch.is_autocast_enabled = lambda *a, **k: True  # the GPU run is under fp16 autocast


def run(layer_forward):
    modeling_qwen2.Qwen2DecoderLayer.forward = layer_forward
    model.zero_grad()
    torch.manual_seed(5)
    st.STATE.request = True
    kept = []
    try:
        with torch.autograd.graph.saved_tensors_hooks(lambda t: kept.append(t) or t, lambda t: t):
            out = model(input_ids=ids, attention_mask=am, labels=lab, num_items_in_batch=nitems)
    finally:
        st.STATE.request = False
    out.loss.backward()
    storages = {}
    for t in kept:
        storages[t.untyped_storage().data_ptr()] = t.untyped_storage().nbytes()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
    return out.loss.detach(), grads, sum(storages.values())


l4, g4, kept4 = run(original_layer)
l5, g5, kept5 = run(fused_layer)
torch.is_autocast_enabled = real_autocast
check("fused layer taken", calls["addnorm"] == cfg.num_hidden_layers, f"{calls['addnorm']} calls")
check("recomputed MLP rebuilds its input from (residual, attention output)",
      calls["mlp_tuple"] == cfg.num_hidden_layers, f"{calls['mlp_tuple']} calls")
check("loss bit-identical", torch.equal(l4, l5), f"{l5.item():.7f} vs {l4.item():.7f}")
same = all(torch.equal(g4[k], g5[k]) for k in g4)
worst = max(((g5[k] - g4[k]).abs().max() / g4[k].abs().max().clamp(min=1e-12)).item() for k in g4)
check("LoRA grads bit-identical", same, f"worst rel {worst:.2e} over {len(g4)} tensors")
check("keeps less for backward", kept5 < kept4, f"{kept5} vs {kept4} bytes")

print("ADDNORM TESTS PASSED" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)

"""Fused LoRA-add + RoPE kernel (bit-exact vs PyTorch) and the patched attention on a tiny model."""
import os, sys
os.environ["TRITON_INTERPRET"] = "1"
import torch
sys.path.insert(0, sys.argv[1])
import solution._state as st
from solution import attn, head, rmsnorm, varlen
for m in (st, attn, head, rmsnorm, varlen):
    m.on_gpu = lambda t: True
head.BLOCK_V = 128
fails = []
def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("  " + detail if detail else ""))
    if not ok: fails.append(name)

# 1. kernel vs reference with the base's rounding sequence
torch.manual_seed(0)
T, HQ, HK, D = 37, 12, 2, 128
qb = (torch.randn(T, HQ * D) * 3).half(); ql = (torch.randn(T, HQ * D) * 0.1).half()
kb = (torch.randn(T, HK * D) * 3).half()
pos = torch.arange(T).float()
inv = 1.0 / (1e6 ** (torch.arange(0, D, 2).float() / D))
fr = torch.outer(pos, inv); emb = torch.cat((fr, fr), -1)
cos, sin = emb.cos(), emb.sin()
qo, ko = attn.FusedQKRope.apply(qb, ql, kb, cos, sin, HQ, HK)
def ref_rope(x32, heads):
    x = x32.view(T, heads, D)
    rot = torch.cat((-x[..., D // 2:], x[..., :D // 2]), -1)
    return (x * cos[:, None] + rot * sin[:, None])
q32 = (qb.float() + ql.float()).requires_grad_(True)
k32 = kb.float().requires_grad_(True)
rq, rk = ref_rope(q32, HQ), ref_rope(k32, HK)
check("rope fwd q bit-exact", torch.equal(qo[0], rq.half().permute(1, 0, 2)))
check("rope fwd k bit-exact", torch.equal(ko[0], rk.half().permute(1, 0, 2)))
gq = torch.randn(1, HQ, T, D).half(); gk = torch.randn(1, HK, T, D).half()
torch.autograd.backward([rq, rk], [gq[0].permute(1, 0, 2).float(), gk[0].permute(1, 0, 2).float()])
dq, dq2, dk, *_ = attn.FusedQKRope.backward(type("C", (), {"saved_tensors": (cos, sin), "shape": (T, HQ, HK, D)})(), gq, gk)
check("rope bwd q bit-exact", torch.equal(dq, q32.grad.reshape(T, -1).half()), f"max {(dq.float() - q32.grad.reshape(T, -1)).abs().max().item():.3g}")
check("rope bwd k bit-exact", torch.equal(dk, k32.grad.reshape(T, -1).half()))
check("rope bwd lora grad = projection grad", torch.equal(dq, dq2))

# 2. patched attention inside the tiny Qwen2 + LoRA, padding-free path, vs the same path without it
from peft import LoraConfig, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM
import bitsandbytes as bnb
torch.manual_seed(0)
cfg = Qwen2Config(vocab_size=300, hidden_size=128 * 2, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=128, tie_word_embeddings=True)
cfg._attn_implementation = "sdpa"
base = Qwen2ForCausalLM(cfg)
def q4(lin):
    layer = bnb.nn.Linear4bit(lin.in_features, lin.out_features, bias=lin.bias is not None, compute_dtype=torch.float16, compress_statistics=True, quant_type="nf4")
    layer.weight = bnb.nn.Params4bit(lin.weight.data.half(), requires_grad=False, compress_statistics=True, quant_type="nf4")._quantize("cpu")
    if lin.bias is not None:
        layer.bias = torch.nn.Parameter(lin.bias.data.half().float(), requires_grad=False)
    return layer
for layer in base.model.layers:
    a = layer.self_attn
    a.q_proj, a.k_proj, a.v_proj, a.o_proj = q4(a.q_proj), q4(a.k_proj), q4(a.v_proj), q4(a.o_proj)
with torch.no_grad():
    base.lm_head.weight.copy_(base.lm_head.weight.half().float())
base.requires_grad_(False)
model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.05, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
for n, p in model.named_parameters():
    if "lora_B" in n: torch.nn.init.normal_(p, std=0.05)
model.train()
B, S = 3, 17
ids = torch.randint(0, 300, (B, S)); am = torch.zeros(B, S, dtype=torch.long); lab = torch.full((B, S), -100)
for i, L in enumerate([17, 6, 11]):
    am[i, :L] = 1; lab[i, L // 2:L] = ids[i, L // 2:L]
nitems = (lab[:, 1:] != -100).sum()
def run():
    model.zero_grad(); torch.manual_seed(5)
    st.STATE.request = True
    try:
        out = model(input_ids=ids, attention_mask=am, labels=lab, num_items_in_batch=nitems)
    finally:
        st.STATE.request = False
    out.loss.backward()
    return out.loss.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
from solution import nf4
nf4.on_gpu = lambda t: True
nf4.patch(); rmsnorm.patch(); head.patch(); varlen.patch()
real_autocast = torch.is_autocast_enabled
torch.is_autocast_enabled = lambda *a, **k: True   # the GPU run is under fp16 autocast
l0, g0 = run()
attn.patch()
calls = {"n": 0}
orig_fwd = attn.FusedQKRope.forward
def counting(ctx, *a):
    calls["n"] += 1
    return orig_fwd(ctx, *a)
attn.FusedQKRope.forward = staticmethod(counting)
pair_calls, sdpa_calls = {"n": 0}, {"n": 0}
orig_pair, orig_sdpa = attn.LoRAPairFromNorm.forward, varlen.PackedSDPA.forward
def counting_pair(ctx, *a):
    pair_calls["n"] += 1
    return orig_pair(ctx, *a)
def counting_sdpa(ctx, *a):
    sdpa_calls["n"] += 1
    return orig_sdpa(ctx, *a)
attn.LoRAPairFromNorm.forward = staticmethod(counting_pair)
varlen.PackedSDPA.forward = staticmethod(counting_sdpa)
l1, g1 = run()
torch.is_autocast_enabled = real_autocast
check("fused attention path taken", calls["n"] == cfg.num_hidden_layers, f"{calls['n']} calls")
check("LoRA pair from the norm taken", pair_calls["n"] == cfg.num_hidden_layers, f"{pair_calls['n']} calls")
check("packed SDPA with recompute taken", sdpa_calls["n"] == cfg.num_hidden_layers, f"{sdpa_calls['n']} calls")
check("loss vs unfused", abs(l1 - l0) / l0 < 2e-3, f"{l1.item():.6f} vs {l0.item():.6f}")
worst = max(((g1[k] - g0[k]).abs().max() / g0[k].abs().max().clamp(min=1e-12)).item() for k in g0)
check("LoRA grads vs unfused (q/k/v rounded to fp16 as on GPU)", worst < 3e-2, f"worst rel {worst:.2e}")
print("ATTN TESTS PASSED" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)

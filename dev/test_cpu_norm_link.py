"""Inputs rebuilt from the norm's input in backward (C2/C3), on CPU with the Triton interpreter.

- LoRAPairFromNorm vs peft's ops under fp16 autocast written out with autograd: bit-identical
  outputs and gradients, and it keeps no [tokens x hidden] fp16 copy.
- RecomputedMLP with the norm link vs without: bit-identical.

Usage: python dev/test_cpu_norm_link.py <repo>
"""
import os
import sys

os.environ["TRITON_INTERPRET"] = "1"
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else ".")
import solution._state as st  # noqa: E402
from solution import attn, mlp, nf4, rmsnorm  # noqa: E402

for m in (st, attn, mlp, nf4, rmsnorm):
    m.on_gpu = lambda t: True
fails = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        fails.append(name)


torch.manual_seed(0)
T, H, R, HV = 29, 256, 8, 64
eps = 1e-6
h = (torch.randn(1, T, H) * 3).requires_grad_(True)
w = torch.rand(H) + 0.5
a_q = (torch.randn(R, H) * 0.05).requires_grad_(True)
b_q = (torch.randn(H, R) * 0.05).requires_grad_(True)
a_v = (torch.randn(R, H) * 0.05).requires_grad_(True)
b_v = (torch.randn(HV, R) * 0.05).requires_grad_(True)
p, scaling = 0.05, 2.0
ones = torch.ones(1, T, H)
scaled_q, keep_q = torch.native_dropout(ones, p, True)
scaled_v, keep_v = torch.native_dropout(ones, p, True)
gq = torch.randn(1, T, H).half() * 64
gv = torch.randn(1, T, HV).half() * 64


def grads(*ts):
    out = [t.grad.clone() for t in ts]
    for t in ts:
        t.grad = None
    return out


# reference: the norm, then peft's branch with autocast's casts written out, under autograd
x = rmsnorm.RMSNormFn.apply(h, w, eps)
outs = []
for scaled, a, b in ((scaled_q, a_q, b_q), (scaled_v, a_v, b_v)):
    xd = x * scaled
    outs.append(F.linear(F.linear(xd.half(), a.half()), b.half()) * scaling)
torch.autograd.backward(outs, [gq, gv])
ref_out = [o.detach() for o in outs]
ref_grads = grads(h, a_q, b_q, a_v, b_v)

# ours
x = rmsnorm.RMSNormFn.apply(h, w, eps)
scale = scaled_q.amax()
saved = []
with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
    oq, ov = attn.LoRAPairFromNorm.apply(x, h, w, eps, scaled_q, keep_q, scale, scaled_v, keep_v,
                                         scale, a_q, b_q, a_v, b_v, scaling, scaling, p, p)
torch.autograd.backward([oq, ov], [gq, gv])
our_grads = grads(h, a_q, b_q, a_v, b_v)

check("q delta bit-exact", torch.equal(oq.detach(), ref_out[0]))
check("v delta bit-exact", torch.equal(ov.detach(), ref_out[1]))
for name, ours, ref in zip(("h", "A_q", "B_q", "A_v", "B_v"), our_grads, ref_grads):
    same = torch.equal(ours, ref)
    rel = ((ours - ref).abs().max() / ref.abs().max()).item()
    # h's gradient sums the q and v branches in another order than autograd: fp32 rounding only
    ok = same if name != "h" else rel < 1e-6
    check(f"grad {name}" + (" bit-exact" if name != "h" else " (fp32 rounding)"), ok,
          f"max rel {rel:.2e}")
big = [t for t in saved if t.dtype == torch.float16 and t.numel() >= T * H]
check("keeps no [T x H] fp16 input", not big, f"{len(saved)} saved, big fp16: {[tuple(t.shape) for t in big]}")
check("keeps the norm's own input (same storage)",
      any(t.untyped_storage().data_ptr() == h.untyped_storage().data_ptr() for t in saved))

# RecomputedMLP with and without the norm link
import bitsandbytes as bnb  # noqa: E402
from transformers import Qwen2Config  # noqa: E402
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP  # noqa: E402


def quantized_linear(n_in, n_out):
    layer = bnb.nn.Linear4bit(n_in, n_out, bias=False, compute_dtype=torch.float16,
                              compress_statistics=True, quant_type="nf4")
    weight = (torch.randn(n_out, n_in) * 0.05).to(torch.float16)
    layer.weight = bnb.nn.Params4bit(weight, requires_grad=False, compress_statistics=True,
                                     quant_type="nf4")._quantize("cpu")
    return layer


block = Qwen2MLP(Qwen2Config(hidden_size=64, intermediate_size=192, hidden_act="silu"))
block.gate_proj, block.up_proj, block.down_proj = (quantized_linear(64, 192), quantized_linear(64, 192),
                                                   quantized_linear(192, 64))
nf4.patch()
mlp.patch(recompute=True)
hm = torch.randn(1, 11, 64) * 2
wm = torch.rand(64) + 0.5
gy = torch.randn(1, 11, 64) * 1000


def run_mlp(link):
    hh = hm.clone().requires_grad_(True)
    y = rmsnorm.RMSNormFn.apply(hh, wm, eps)
    st.STATE.norm_link = (y, hh, wm, eps) if link else None
    try:
        out = block(y)
    finally:
        st.STATE.norm_link = None
    (out.float() * 1).backward(gy)
    return out.detach(), hh.grad


plain, linked = run_mlp(False), run_mlp(True)
check("MLP out with norm link bit-exact", torch.equal(plain[0], linked[0]))
check("MLP input grad with norm link bit-exact", torch.equal(plain[1], linked[1]))

print("NORM LINK TESTS PASSED" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)

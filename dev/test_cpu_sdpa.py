"""PackedSDPA (attention recomputed in backward) vs per-row SDPA under autograd, on CPU in fp16.

Usage: python dev/test_cpu_sdpa.py <repo>
"""
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else ".")
from solution import varlen  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        fails.append(name)


torch.manual_seed(0)
HQ, HK, D = 12, 2, 128
spans = [(0, 37), (37, 5), (42, 20), (62, 1)]
T = sum(length for _, length in spans)
groups = HQ // HK
scale = D ** -0.5
q = torch.randn(1, HQ, T, D).half()                          # head-major, as attn.py makes it
k = torch.randn(1, HK, T, D).half()
v = torch.randn(1, T, HK, D).half().transpose(1, 2)           # transposed view, as attn.py passes it
g = torch.randn(1, T, HQ, D).half()


def reference(q, k, v):
    outs = []
    for start, length in spans:
        kk = k[:, :, start:start + length].repeat_interleave(groups, dim=1)
        vv = v[:, :, start:start + length].repeat_interleave(groups, dim=1)
        outs.append(F.scaled_dot_product_attention(q[:, :, start:start + length], kk, vv,
                                                   dropout_p=0.0, scale=scale, is_causal=True))
    return torch.cat(outs, dim=2).transpose(1, 2).contiguous()


q0, k0, v0 = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
out0 = reference(q0, k0, v0)
out0.backward(g)

q1, k1, v1 = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
v1_view = v1  # a leaf with the transposed strides
out1 = varlen.PackedSDPA.apply(q1, k1, v1_view, spans, groups, scale)
out1.backward(g)

check("forward bit-exact", torch.equal(out0, out1))
check("output layout [1, T, H, D] contiguous", out1.shape == (1, T, HQ, D) and out1.is_contiguous())
check("dq bit-exact", torch.equal(q0.grad, q1.grad))
check("dk bit-exact", torch.equal(k0.grad, k1.grad))
check("dv bit-exact", torch.equal(v0.grad, v1.grad))
check("dv keeps the input strides", v1.grad.stride() == v1.stride(), f"{v1.grad.stride()} vs {v1.stride()}")

# Only q, k, v are saved: no expanded k/v and no output kept by the graph.
saved = []
with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
    out2 = varlen.PackedSDPA.apply(*(t.detach().requires_grad_(True) for t in (q, k, v)), spans,
                                   groups, scale)
check("saves only q, k, v", len(saved) == 3 and sum(t.numel() for t in saved) == q.numel() + k.numel() + v.numel(),
      f"{len(saved)} tensors, {sum(t.numel() for t in saved)} elements")

print("SDPA TESTS PASSED" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)

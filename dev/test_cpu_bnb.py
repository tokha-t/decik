"""bitsandbytes Linear4bit and Qwen2MLP: patched vs original on CPU (Triton interpreter)."""
import os
import sys

os.environ["TRITON_INTERPRET"] = "1"
import torch  # noqa: E402

sys.path.insert(0, sys.argv[1])
import solution._state as st  # noqa: E402
from solution import mlp, nf4  # noqa: E402

import bitsandbytes as bnb  # noqa: E402
from transformers import Qwen2Config  # noqa: E402
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def quantized_linear(n_in, n_out, bias):
    layer = bnb.nn.Linear4bit(n_in, n_out, bias=bias, compute_dtype=torch.float16,
                              compress_statistics=True, quant_type="nf4")
    w = (torch.randn(n_out, n_in) * 0.05).to(torch.float16)
    layer.weight = bnb.nn.Params4bit(w, requires_grad=False, compress_statistics=True,
                                     quant_type="nf4")._quantize("cpu")
    if bias:
        layer.bias = torch.nn.Parameter((torch.randn(n_out) * 0.1).float(), requires_grad=False)
    return layer


torch.manual_seed(0)
cfg = Qwen2Config(hidden_size=64, intermediate_size=192, hidden_act="silu")
block = Qwen2MLP(cfg)
block.gate_proj = quantized_linear(64, 192, False)
block.up_proj = quantized_linear(64, 192, False)
block.down_proj = quantized_linear(192, 64, False)
qkv = quantized_linear(64, 96, True)
x = torch.randn(2, 7, 64, dtype=torch.float32)
gy = torch.randn(2, 7, 64)
gq = torch.randn(2, 7, 96)


def run():
    xa = x.clone().requires_grad_(True)
    y = block(xa)
    y.backward(gy)
    xb = x.clone().requires_grad_(True)
    q = qkv(xb)
    q.backward(gq)
    return y.detach(), xa.grad, q.detach(), xb.grad


ref = run()
for mod in (st, nf4, mlp):
    mod.on_gpu = lambda t: True
nf4.patch()
mlp.patch()
ours = run()
names = ["mlp output", "mlp input grad", "linear+bias output", "linear+bias input grad"]
for name, a, b in zip(names, ours, ref):
    err = (a.float() - b.float()).abs().max().item() / max(b.abs().max().item(), 1e-9)
    check(name, err < 2e-3, f"rel {err:.2e}, dtype ours {a.dtype} base {b.dtype}")

print("\nBNB TESTS PASSED" if not failures else f"\nFAILED: {failures}")
sys.exit(1 if failures else 0)

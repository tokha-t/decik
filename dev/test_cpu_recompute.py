"""RecomputedMLP vs the fp16 MLP path vs bitsandbytes base, on CPU (Triton interpreter)."""
import os, sys
os.environ["TRITON_INTERPRET"] = "1"
import torch
sys.path.insert(0, sys.argv[1])
import solution._state as st
from solution import mlp, nf4
import bitsandbytes as bnb
from transformers import Qwen2Config
from transformers.models.qwen2 import modeling_qwen2
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

def quantized_linear(n_in, n_out):
    layer = bnb.nn.Linear4bit(n_in, n_out, bias=False, compute_dtype=torch.float16, compress_statistics=True, quant_type="nf4")
    w = (torch.randn(n_out, n_in) * 0.05).to(torch.float16)
    layer.weight = bnb.nn.Params4bit(w, requires_grad=False, compress_statistics=True, quant_type="nf4")._quantize("cpu")
    return layer

torch.manual_seed(0)
block = Qwen2MLP(Qwen2Config(hidden_size=64, intermediate_size=192, hidden_act="silu"))
block.gate_proj, block.up_proj, block.down_proj = quantized_linear(64, 192), quantized_linear(64, 192), quantized_linear(192, 64)
x = torch.randn(3, 11, 64)
gy = torch.randn(3, 11, 64) * 1000  # a loss-scaled gradient

def run():
    xa = x.clone().requires_grad_(True)
    y = block(xa)
    (y.float() * 1).backward(gy)
    return y.detach().float(), xa.grad.clone()

ref = run()
for m in (st, nf4, mlp):
    m.on_gpu = lambda t: True
orig = modeling_qwen2.Qwen2MLP.forward
nf4.patch(); mlp.patch(recompute=False)
v1 = run()
modeling_qwen2.Qwen2MLP.forward = orig
mlp.patch(recompute=True)
v2 = run()
fails = 0
for name, a, b in (("v1 vs base out", v1[0], ref[0]), ("v1 vs base grad", v1[1], ref[1]),
                   ("recompute vs v1 out", v2[0], v1[0]), ("recompute vs v1 grad", v2[1], v1[1])):
    same = torch.equal(a, b)
    fails += not same
    print(("PASS " if same else "FAIL ") + f"{name}: bit-identical={same} max|diff|={(a - b).abs().max().item():.3g}")
print("RECOMPUTE TESTS PASSED" if not fails else "FAILED")
sys.exit(1 if fails else 0)

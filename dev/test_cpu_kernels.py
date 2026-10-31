"""Kernel correctness on CPU through Triton's interpreter (TRITON_INTERPRET=1)."""
import os
import sys

os.environ["TRITON_INTERPRET"] = "1"
import torch  # noqa: E402

sys.path.insert(0, sys.argv[1])
import solution._state as st  # noqa: E402
from solution import head, nf4, rmsnorm  # noqa: E402

for mod in (st, nf4, rmsnorm, head):
    mod.on_gpu = lambda t: True
torch.manual_seed(0)
failures = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail else ""))
    if not ok:
        failures.append(name)


# 1. NF4 dequant vs bitsandbytes (CPU reference implementation), nested and plain absmax.
import bitsandbytes.functional as bnbf  # noqa: E402

for nested in (True, False):
    for shape in [(96, 128), (256, 64 * 40)]:
        w = (torch.randn(shape) * 0.05).to(torch.float16)
        packed, qs = bnbf.quantize_4bit(w, blocksize=64, compress_statistics=nested,
                                        quant_type="nf4", quant_storage=torch.uint8)
        ref = bnbf.dequantize_4bit(packed, qs)
        ours = nf4.dequantize_nf4(packed, qs)
        same = torch.equal(ref, ours)
        diff = (ref.float() - ours.float()).abs().max().item()
        check(f"nf4 dequant nested={nested} {shape}", same, f"max|diff|={diff:.3g} dtype={ours.dtype}")

# 2. RMSNorm forward / backward vs PyTorch in fp32.
for rows, cols in [(7, 1536), (3, 100)]:
    x = torch.randn(rows, cols, dtype=torch.float32, requires_grad=True)
    wt = torch.randn(cols, dtype=torch.float32) * 0.1 + 1.0
    dy = torch.randn(rows, cols, dtype=torch.float32)
    ref = wt * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6))
    ref.backward(dy)
    ref_dx = x.grad.clone()
    x.grad = None
    out = rmsnorm.RMSNormFn.apply(x, wt, 1e-6)
    out.backward(dy)
    check(f"rmsnorm fwd {rows}x{cols}", torch.allclose(out, ref, rtol=1e-5, atol=1e-5),
          f"max|diff|={(out - ref).abs().max().item():.3g}")
    check(f"rmsnorm bwd {rows}x{cols}", torch.allclose(x.grad, ref_dx, rtol=1e-4, atol=1e-4),
          f"max|diff|={(x.grad - ref_dx).abs().max().item():.3g}")

# 3. Fused head vs PyTorch: loss, entropy, accuracy hits, per-token loss, hidden gradient.
head.CHUNK_ROWS = 5  # several chunks, labelled/unlabelled mixed inside chunks
head.BLOCK_V = 64
for vocab in (300, 64, 1000):
    T, H = 23, 32
    hidden = torch.randn(T, H, dtype=torch.float32, requires_grad=True)
    weight16 = (torch.randn(vocab, H) * 0.3).to(torch.float16)
    targets = torch.randint(0, vocab, (T,))
    targets[torch.rand(T) < 0.4] = -100
    denom = torch.tensor(float((targets != -100).sum() + 3))  # like num_items_in_batch
    # make one exact tie for argmax: copy the top logit of row 0
    loss, ent, tok, hit = head.FusedHeadLoss.apply(hidden, weight16, targets, denom)
    gl = torch.tensor(3.0)
    (loss * gl).backward()
    g_ours = hidden.grad.clone()
    hidden.grad = None
    logits = (hidden.to(torch.float16) @ weight16.t()).float()  # fp16 logits, as autocast
    logp = torch.log_softmax(logits, -1)
    lab = targets != -100
    ref_tok = torch.where(lab, -logp.gather(1, targets.clamp(min=0)[:, None])[:, 0], 0.0)
    ref_loss = ref_tok.sum() / denom
    ref_ent = -(logp.exp() * logp).sum(-1)
    ref_hit = ((logits.argmax(-1) == targets) & lab).int()
    check(f"head loss V={vocab}", torch.allclose(loss, ref_loss, rtol=1e-5, atol=1e-6),
          f"{loss.item():.6f} vs {ref_loss.item():.6f}")
    check(f"head entropy V={vocab}", torch.allclose(ent, ref_ent, rtol=1e-4, atol=1e-5),
          f"max|diff|={(ent - ref_ent).abs().max().item():.3g}")
    check(f"head token loss V={vocab}", torch.allclose(tok, ref_tok, rtol=1e-5, atol=1e-5))
    check(f"head argmax hits V={vocab}", torch.equal(hit, ref_hit), f"{hit.sum().item()} hits")
    # reference gradient: (softmax - onehot) in fp16, matmul in fp16, scaled
    p = logp.exp()
    g = p.clone()
    rows = torch.nonzero(lab).squeeze(1)
    g[rows, targets[rows]] -= 1.0
    g[~lab] = 0.0
    ref_grad = (g.to(torch.float16) @ weight16).float() * (gl / denom)
    check(f"head grad V={vocab}", torch.allclose(g_ours, ref_grad, rtol=2e-3, atol=1e-5),
          f"max|diff|={(g_ours - ref_grad).abs().max().item():.3g} max|g|={ref_grad.abs().max().item():.3g}")

# argmax tie-break: first index, as torch.argmax
weight16 = torch.zeros(10, 4, dtype=torch.float16)
weight16[3] = 1.0
weight16[7] = 1.0  # rows 3 and 7 give the same logit -> argmax must be 3
hidden = torch.ones(2, 4, requires_grad=True)
targets = torch.tensor([3, 7])
_, _, _, hit = head.FusedHeadLoss.apply(hidden, weight16, targets, torch.tensor(2.0))
check("head argmax tie -> first index", hit.tolist() == [1, 0], str(hit.tolist()))

print("\nALL KERNEL TESTS PASSED" if not failures else f"\nFAILED: {failures}")
sys.exit(1 if failures else 0)

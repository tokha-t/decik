"""Our kernels against bitsandbytes / plain PyTorch on the GPU: `python -m pytest solution -q`."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("nested", [True, False])
@pytest.mark.parametrize("shape", [(1536, 1536), (256, 1536), (8960, 1536), (1536, 8960)])
def test_nf4_dequant_is_bit_identical_to_bitsandbytes(nested, shape):
    import bitsandbytes.functional as bnbf

    from solution.nf4 import dequantize_nf4

    torch.manual_seed(0)
    w = (torch.randn(shape, device="cuda") * 0.05).half()
    packed, qs = bnbf.quantize_4bit(w, blocksize=64, compress_statistics=nested, quant_type="nf4")
    assert torch.equal(dequantize_nf4(packed, qs), bnbf.dequantize_4bit(packed, qs))


@pytest.mark.parametrize("rows", [1, 77, 2048])
def test_rmsnorm_matches_pytorch(rows):
    from solution.rmsnorm import RMSNormFn

    torch.manual_seed(0)
    x = torch.randn(rows, 1536, device="cuda", requires_grad=True)
    w = torch.randn(1536, device="cuda") * 0.1 + 1
    dy = torch.randn(rows, 1536, device="cuda")
    ref = w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6))
    ref.backward(dy)
    ref_dx, x.grad = x.grad.clone(), None
    out = RMSNormFn.apply(x, w, 1e-6)
    out.backward(dy)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(x.grad, ref_dx, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("tokens", [1, 300, 1100])
def test_head_matches_pytorch(tokens):
    from solution.head import FusedHeadLoss

    torch.manual_seed(0)
    vocab, hidden = 151936, 1536
    h = torch.randn(tokens, hidden, device="cuda", requires_grad=True)
    w16 = (torch.randn(vocab, hidden, device="cuda") * 0.02).half()
    targets = torch.randint(0, vocab, (tokens,), device="cuda")
    targets[torch.rand(tokens, device="cuda") < 0.4] = -100
    if tokens > 1:
        targets[0] = 5
    denom = (targets != -100).sum().clamp(min=1).float()
    loss, ent, tok, hit = FusedHeadLoss.apply(h, w16, targets, denom)
    (loss * 1024.0).backward()  # a loss scale, as the fp16 GradScaler applies
    ours_grad, h.grad = h.grad.clone(), None

    logits = (h.half() @ w16.t()).float()
    logp = torch.log_softmax(logits, -1)
    lab = targets != -100
    ref_tok = torch.where(lab, -logp.gather(1, targets.clamp(min=0)[:, None])[:, 0], 0.0)
    (ref_tok.sum() / denom * 1024.0).backward()
    torch.testing.assert_close(loss, ref_tok.sum() / denom, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(tok, ref_tok, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(ent, -(logp.exp() * logp).sum(-1), rtol=1e-3, atol=1e-3)
    assert torch.equal(hit, ((logits.argmax(-1) == targets) & lab).int())
    torch.testing.assert_close(ours_grad, h.grad, rtol=2e-2, atol=2e-3)


def test_packed_attention_recompute_matches_per_row_sdpa():
    """PackedSDPA (q, k, v kept, rows recomputed in backward) vs per-row SDPA under autograd."""
    import torch.nn.functional as F

    from solution.varlen import PackedSDPA

    torch.manual_seed(0)
    hq, hk, d = 12, 2, 128
    spans = [(0, 401), (401, 77), (478, 512), (990, 3)]
    n = sum(length for _, length in spans)
    q = torch.randn(1, hq, n, d, device="cuda").half()
    k = torch.randn(1, hk, n, d, device="cuda").half()
    v = torch.randn(1, n, hk, d, device="cuda").half().transpose(1, 2)
    g = torch.randn(1, n, hq, d, device="cuda").half()

    def per_row(q, k, v):
        outs = []
        for start, length in spans:
            rows = slice(start, start + length)
            outs.append(F.scaled_dot_product_attention(
                q[:, :, rows], k[:, :, rows].repeat_interleave(hq // hk, dim=1),
                v[:, :, rows].repeat_interleave(hq // hk, dim=1), is_causal=True))
        return torch.cat(outs, dim=2).transpose(1, 2).contiguous()

    ref_in = [t.detach().clone().requires_grad_(True) for t in (q, k, v)]
    ref = per_row(*ref_in)
    ref.backward(g)
    our_in = [t.detach().clone().requires_grad_(True) for t in (q, k, v)]
    ours = PackedSDPA.apply(*our_in, spans, hq // hk, None)
    ours.backward(g)
    assert torch.equal(ours, ref)
    for a, b in zip(our_in, ref_in):  # SDPA's backward itself is not bit-deterministic
        torch.testing.assert_close(a.grad, b.grad, rtol=2e-3, atol=2e-3)


def test_lora_pair_from_norm_matches_peft_ops():
    """q/v LoRA with the input rebuilt from the norm's input vs peft's ops under fp16 autocast."""
    import torch.nn.functional as F

    from solution.attn import LoRAPairFromNorm, pack_keep
    from solution.rmsnorm import RMSNormFn

    torch.manual_seed(0)
    n, hidden, r = 700, 1536, 16
    h = (torch.randn(1, n, hidden, device="cuda") * 3).requires_grad_(True)
    w = torch.rand(hidden, device="cuda") + 0.5
    a_q, b_q = (torch.randn(r, hidden, device="cuda") * 0.05).requires_grad_(True), \
        (torch.randn(hidden, r, device="cuda") * 0.05).requires_grad_(True)
    a_v, b_v = (torch.randn(r, hidden, device="cuda") * 0.05).requires_grad_(True), \
        (torch.randn(256, r, device="cuda") * 0.05).requires_grad_(True)
    ones = torch.ones(1, n, hidden, device="cuda")
    (sq, kq), (sv, kv) = torch.native_dropout(ones, 0.05, True), torch.native_dropout(ones, 0.05, True)
    gq = torch.randn(1, n, hidden, device="cuda").half() * 64
    gv = torch.randn(1, n, 256, device="cuda").half() * 64
    params = (h, a_q, b_q, a_v, b_v)

    with torch.autocast("cuda", dtype=torch.float16):
        x = RMSNormFn.apply(h, w, 1e-6)
        ref = [F.linear(F.linear(x * s, a), b) * 2.0 for s, a, b in ((sq, a_q, b_q), (sv, a_v, b_v))]
    torch.autograd.backward(ref, [gq, gv])
    ref_grads = [p.grad.clone() for p in params]
    for p in params:
        p.grad = None
    with torch.autocast("cuda", dtype=torch.float16):
        x = RMSNormFn.apply(h, w, 1e-6)
        bits_q, bits_v = pack_keep(kq.reshape(n, hidden), kv.reshape(n, hidden))
        ours = LoRAPairFromNorm.apply(x, h, w, 1e-6, sq, bits_q, sq.amax(), sv, bits_v, sv.amax(),
                                      a_q, b_q, a_v, b_v, 2.0, 2.0, 0.05, 0.05, True)
    torch.autograd.backward(list(ours), [gq, gv])
    for a, b in zip(ours, ref):
        assert torch.equal(a, b)
    for p, g in zip(params, ref_grads):
        torch.testing.assert_close(p.grad, g, rtol=1e-3, atol=1e-5)


def test_add_rmsnorm_matches_add_then_norm():
    """Fused residual add + RMSNorm (h not kept, rebuilt in backward) vs the add and our norm."""
    from solution.rmsnorm import AddRMSNormFn, RMSNormFn

    torch.manual_seed(0)
    n, hidden = 1100, 1536
    w = torch.rand(hidden, device="cuda") + 0.5
    residual = torch.randn(1, n, hidden, device="cuda") * 2
    delta = (torch.randn(1, n, hidden, device="cuda") * 0.5).half()
    g_h, g_y = torch.randn(1, n, hidden, device="cuda"), torch.randn(1, n, hidden, device="cuda")
    r1, d1 = residual.clone().requires_grad_(True), delta.clone().requires_grad_(True)
    h1 = r1 + d1
    y1 = RMSNormFn.apply(h1, w, 1e-6)
    torch.autograd.backward([h1, y1], [g_h, g_y])
    r2, d2 = residual.clone().requires_grad_(True), delta.clone().requires_grad_(True)
    h2, y2 = AddRMSNormFn.apply(r2, d2, w, 1e-6)
    torch.autograd.backward([h2, y2], [g_h, g_y])
    assert torch.equal(h1, h2) and torch.equal(y1, y2)
    assert torch.equal(r1.grad, r2.grad) and torch.equal(d1.grad, d2.grad)


def test_varlen_attention_matches_per_row():
    """VarlenSDPA (one varlen launch over all rows) vs PackedSDPA (per-row SDPA), fp16 on the GPU."""
    from solution.varlen import PackedSDPA, VarlenSDPA, _varlen_available

    if not _varlen_available():
        pytest.skip("no varlen memory-efficient attention in this PyTorch")
    torch.manual_seed(0)
    hq, hk, d = 12, 2, 128
    spans = [(0, 401), (401, 77), (478, 512), (990, 3)]
    n = sum(length for _, length in spans)
    q = torch.randn(1, hq, n, d, device="cuda").half()
    k = torch.randn(1, hk, n, d, device="cuda").half()
    v = torch.randn(1, n, hk, d, device="cuda").half().transpose(1, 2)
    g = torch.randn(1, n, hq, d, device="cuda").half()
    cu = torch.tensor([s for s, _ in spans] + [n], dtype=torch.int32, device="cuda")
    ref_in = [t.detach().clone().requires_grad_(True) for t in (q, k, v)]
    ref = PackedSDPA.apply(*ref_in, spans, hq // hk, d ** -0.5)
    ref.backward(g)
    our_in = [t.detach().clone().requires_grad_(True) for t in (q, k, v)]
    ours = VarlenSDPA.apply(*our_in, cu, max(length for _, length in spans), hq // hk, d ** -0.5)
    ours.backward(g)
    torch.testing.assert_close(ours, ref, rtol=1e-3, atol=1e-3)
    for a, b in zip(our_in, ref_in):  # SDPA's backward itself is not bit-deterministic
        torch.testing.assert_close(a.grad, b.grad, rtol=2e-3, atol=2e-3)

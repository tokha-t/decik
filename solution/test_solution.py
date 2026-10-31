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

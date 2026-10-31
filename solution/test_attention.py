"""The fused attention input transform against the unfused PyTorch operations."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("tokens", [1, 77, 1100])
def test_fused_qk_rope_forward_and_input_gradients(tokens):
    from solution.attn import FusedQKRope

    torch.manual_seed(91)
    heads, kv_heads, dim = 12, 2, 128
    qb = torch.randn(tokens, heads * dim, device="cuda", dtype=torch.float16, requires_grad=True)
    ql = torch.randn_like(qb, requires_grad=True)
    k = torch.randn(tokens, kv_heads * dim, device="cuda", dtype=torch.float16, requires_grad=True)
    angles = torch.randn(tokens, dim // 2, device="cuda").repeat(1, 2)
    cos, sin = angles.cos(), angles.sin()

    def rope(x, count):
        x = x.float().view(tokens, count, dim).transpose(0, 1)
        rotated = torch.cat((-x[..., dim // 2:], x[..., :dim // 2]), dim=-1)
        return (x * cos.unsqueeze(0) + rotated * sin.unsqueeze(0)).half().unsqueeze(0)

    reference_q = rope(qb.float() + ql.float(), heads)
    reference_k = rope(k, kv_heads)
    dq, dk = torch.randn_like(reference_q), torch.randn_like(reference_k)
    torch.autograd.backward((reference_q, reference_k), (dq, dk))
    expected = [x.grad.clone() for x in (qb, ql, k)]
    for x in (qb, ql, k):
        x.grad = None

    actual_q, actual_k = FusedQKRope.apply(qb, ql, k, cos, sin, heads, kv_heads)
    torch.autograd.backward((actual_q, actual_k), (dq, dk))
    torch.testing.assert_close(actual_q, reference_q, rtol=0, atol=0)
    torch.testing.assert_close(actual_k, reference_k, rtol=0, atol=0)
    for x, grad in zip((qb, ql, k), expected):
        torch.testing.assert_close(x.grad, grad, rtol=0, atol=0)

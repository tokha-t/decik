"""Recomputed MLP outputs and input gradients against the non-recomputed path."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("tokens", [3, 77])
def test_recomputed_mlp_matches_saved_activations(tokens):
    import bitsandbytes as bnb

    from solution.mlp import RecomputedMLP
    from solution.nf4 import NF4Linear
    from solution.swiglu import swiglu

    torch.manual_seed(92)

    def layer(in_features, out_features):
        module = bnb.nn.Linear4bit(in_features, out_features, bias=False,
                                  compute_dtype=torch.float16, compress_statistics=True,
                                  quant_type="nf4").half().cuda()
        module.weight.requires_grad_(False)
        return module

    gate, up, down = layer(1536, 8960), layer(1536, 8960), layer(8960, 1536)
    x = torch.randn(tokens, 1536, device="cuda", requires_grad=True)
    grad = torch.randn(tokens, 1536, device="cuda", dtype=torch.float16)

    def linear(h, module):
        return NF4Linear.apply(h.half(), module.weight.data, None, module.weight.quant_state)

    reference = linear(swiglu(linear(x, gate), linear(x, up)), down)
    reference.backward(grad)
    expected_grad, x.grad = x.grad.clone(), None
    actual = RecomputedMLP.apply(x, gate, up, down)
    actual.backward(grad)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, expected_grad, rtol=0, atol=0)

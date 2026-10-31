"""Word-load NF4 parity, including partial groups and byte counts requiring fallback."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("nested", [True, False])
@pytest.mark.parametrize("shape", [(3, 64), (129, 64), (3, 1002)])
def test_nf4_word_loads_and_tail_fallback_match_bnb(monkeypatch, nested, shape):
    import bitsandbytes.functional as bnbf

    from solution import nf4

    torch.manual_seed(93)
    weights = (torch.randn(shape, device="cuda") * 0.05).half()
    packed, state = bnbf.quantize_4bit(weights, blocksize=64, compress_statistics=nested,
                                     quant_type="nf4")
    expected = bnbf.dequantize_4bit(packed, state)
    monkeypatch.setattr(nf4, "USE_WORD_LOADS", True)
    actual = nf4.dequantize_nf4(packed, state)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("nested", [True, False])
@pytest.mark.parametrize("shape", [(3, 64), (129, 64), (3, 1002)])
def test_nf4_local_lut_matches_bnb(monkeypatch, nested, shape):
    import bitsandbytes.functional as bnbf

    from solution import nf4

    torch.manual_seed(94)
    weights = (torch.randn(shape, device="cuda") * 0.05).half()
    packed, state = bnbf.quantize_4bit(weights, blocksize=64, compress_statistics=nested,
                                     quant_type="nf4")
    monkeypatch.setattr(nf4, "USE_WORD_LOADS", False)
    monkeypatch.setattr(nf4, "USE_LOCAL_LUT", True)
    torch.testing.assert_close(nf4.dequantize_nf4(packed, state),
                               bnbf.dequantize_4bit(packed, state), rtol=0, atol=0)

"""Qwen2 MLP with fp16 intermediates.

In the base, the residual stream is fp32 (prepare_model_for_kbit_training keeps the embedding in
fp32) and bitsandbytes' Linear4bit returns its fp16 matmul result cast back to the input dtype,
so gate and up live in fp32 and silu(gate), gate and up are all saved for backward in fp32:
3 x 8960 x 4 bytes per token per layer. Their values are exactly fp16 values (the output of an
fp16 matmul), so we feed the projections fp16 and keep gate and up in fp16: the same numbers,
a third of the memory, and none of the fp32 casts. Each projection casts the input itself, as
bitsandbytes does, so the input gradients still add up in fp32.

SwiGLU itself is the starter kit's example kernel (allowed to run; it does not count as ours).
"""

import torch

from ._state import on_gpu


def patch():
    import bitsandbytes as bnb
    from transformers.models.qwen2 import modeling_qwen2

    from .swiglu import swiglu

    cls = modeling_qwen2.Qwen2MLP
    if getattr(cls.forward, "_skc", False):
        return
    original = cls.forward
    linear4bit = bnb.nn.Linear4bit

    def fp16_projection(module):
        return isinstance(module, linear4bit) and getattr(module, "compute_dtype", None) == torch.float16

    def forward(self, x):
        config = getattr(self, "config", None)
        if not (isinstance(x, torch.Tensor) and on_gpu(x) and x.dtype == torch.float32
                and getattr(config, "hidden_act", None) == "silu"
                and all(fp16_projection(getattr(self, name, None))
                        for name in ("gate_proj", "up_proj", "down_proj"))):
            return original(self, x)
        gate = self.gate_proj(x.to(torch.float16))
        up = self.up_proj(x.to(torch.float16))
        return self.down_proj(swiglu(gate, up))

    forward._skc = True
    cls.forward = forward

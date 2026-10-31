"""Keep the tied embedding / lm_head in fp16.

Soup loads the non-quantized weights in fp16 on a T4, then peft's prepare_model_for_kbit_training
upcasts them to fp32. For the 151,936 x 1536 tied embedding that is 890 MiB, plus a 445 MiB fp16
copy that autocast makes for lm_head in every forward (kept for backward), plus a moment during
setup when the fp16 and the fp32 copies exist together. The fp32 values are exactly the fp16 ones,
so we skip that one upcast and turn the lookup result into fp32 right after the embedding: the
residual stream gets the same fp32 numbers as in the base. Norm weights and biases are upcast as
before.
"""

import torch


def _lookup_in_fp32(module, inputs, output):
    if isinstance(output, torch.Tensor) and output.dtype == torch.float16:
        return output.float()
    return output


def patch():
    import peft

    original = peft.prepare_model_for_kbit_training
    if getattr(original, "_skc", False):
        return

    def prepare_model_for_kbit_training(model, *args, **kwargs):
        embedding, param, kept = None, None, None
        try:
            if isinstance(model, torch.nn.Module) and callable(getattr(model, "get_input_embeddings", None)):
                embedding = model.get_input_embeddings()
                param = getattr(embedding, "weight", None)
                if isinstance(param, torch.nn.Parameter) and param.dtype == torch.float16 and param.is_cuda:
                    kept = param.data
                    # An empty fp32 stand-in: the upcast loop skips it.
                    param.data = torch.empty(0, dtype=torch.float32, device=kept.device)
        except Exception:  # noqa: BLE001 - anything unexpected: behave exactly like peft
            kept = None
        try:
            result = original(model, *args, **kwargs)
        finally:
            if kept is not None:
                param.data = kept
        if kept is not None and not getattr(embedding, "_skc_hook", False):
            embedding.register_forward_hook(_lookup_in_fp32)
            embedding._skc_hook = True
        return result

    prepare_model_for_kbit_training._skc = True
    peft.prepare_model_for_kbit_training = prepare_model_for_kbit_training

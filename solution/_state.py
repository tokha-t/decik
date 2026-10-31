"""State shared by the patches during one training step (one process, one thread)."""


class _State:
    def __init__(self):
        # Set by our SFTTrainer.compute_loss around the model call: only then do the fused
        # head and the packed (padding-free) decoder run.
        self.request = False
        # What the fused head leaves for the trainer's metrics (entropy, accuracy, rewind).
        self.stats = None
        # The packing of the current batch (varlen.Packing) while its forward runs.
        self.packing = None
        self.packing_active = False
        # (output, input, weight, eps) of the last RMSNorm run on the packed path: what follows
        # the norm (q/v LoRA, the recomputed MLP) rebuilds its input from the norm's input, which
        # the norm keeps for its own backward anyway, instead of keeping a copy.
        self.norm_link = None


STATE = _State()


def on_gpu(tensor):
    """Our kernels run on CUDA tensors only; everything else takes the original code path."""
    return tensor.is_cuda

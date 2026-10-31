"""Original vs patched TRL SFTTrainer on a tiny Qwen2 + LoRA (CPU): the logged metrics agree."""
import copy
import os
import sys
import tempfile

os.environ["TRITON_INTERPRET"] = "1"
import torch  # noqa: E402

sys.path.insert(0, sys.argv[1])
DATA = sys.argv[2]
import solution._state as st  # noqa: E402
from solution import head, rmsnorm, varlen  # noqa: E402

for mod in (st, rmsnorm, head, varlen):
    mod.on_gpu = lambda t: True
head.BLOCK_V = 8192

from datasets import Dataset  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

tok = AutoTokenizer.from_pretrained(DATA)
torch.manual_seed(0)
cfg = Qwen2Config(vocab_size=151936, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                  tie_word_embeddings=True)
cfg._attn_implementation = "sdpa"
base = Qwen2ForCausalLM(cfg)
with torch.no_grad():
    base.lm_head.weight.copy_(base.lm_head.weight.half().float())
init = copy.deepcopy(base.state_dict())

g = torch.Generator().manual_seed(1)
rows = []
for i in range(16):
    n_prompt = int(torch.randint(3, 20, (1,), generator=g))
    n_answer = int(torch.randint(2, 30, (1,), generator=g))
    ids = torch.randint(0, 150000, (n_prompt + n_answer,), generator=g).tolist()
    rows.append({"input_ids": ids, "completion_mask": [0] * n_prompt + [1] * n_answer})
dataset = Dataset.from_list(rows)


def train():
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg)
    model.load_state_dict(init)
    model.requires_grad_(False)
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.05,
                                             target_modules=["q_proj", "v_proj"],
                                             task_type="CAUSAL_LM"))
    with tempfile.TemporaryDirectory() as out:
        args = SFTConfig(output_dir=out, max_steps=4, per_device_train_batch_size=4,
                         logging_steps=1, report_to=[], use_cpu=True, seed=1, learning_rate=1e-2,
                         completion_only_loss=True, max_length=64, save_strategy="no",
                         gradient_checkpointing=False,
                         dataset_kwargs={"skip_prepare_dataset": True}, remove_unused_columns=False)
        trainer = SFTTrainer(model=model, args=args, train_dataset=dataset, processing_class=tok)
        trainer.train()
        logs = [e for e in trainer.state.log_history if "loss" in e]
        return logs, getattr(trainer, head.STEP_ATTR, None)


ref_logs, _ = train()
rmsnorm.patch()
head.patch()
varlen.patch()
our_logs, step = train()

failures = []
for key in ("loss", "entropy", "mean_token_accuracy", "num_tokens"):
    a = [e.get(key) for e in our_logs]
    b = [e.get(key) for e in ref_logs]
    if key == "num_tokens":
        ok = a == b
    else:
        ok = all(abs(x - y) <= 2e-3 * max(abs(y), 1e-6) + 1e-6 for x, y in zip(a, b)) and len(a) == len(b)
    print(("PASS " if ok else "FAIL ") + f"{key}: ours {a}\n      base {b}")
    if not ok:
        failures.append(key)
same_keys = sorted(our_logs[0]) == sorted(ref_logs[0])
print(("PASS " if same_keys else "FAIL ") + f"logged keys {sorted(our_logs[0])}")
ok_step = step is not None and tuple(step.token_loss.shape)[0] == 4
print(("PASS " if ok_step else "FAIL ") + "rewind hook set (token_loss [B, S])")
failures += [] if same_keys else ["keys"]
failures += [] if ok_step else ["rewind"]
print("\nTRAINER TEST PASSED" if not failures else f"\nFAILED: {failures}")
sys.exit(1 if failures else 0)

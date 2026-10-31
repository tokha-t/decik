# dev/: checks that run without a GPU

These are not part of the solution; the harness only reads `solution/`. They are how every kernel and
patch was checked before going to the Kaggle T4.

Environment (CPU): torch 2.11 (CPU build), triton 3.6.0, **numpy 2.2.6** (newer numpy breaks the
Triton interpreter), transformers 5.16.1, peft 0.20.0, trl 0.29.1, bitsandbytes 0.50.2,
accelerate 1.14.0, datasets.

| Script | What it checks |
|---|---|
| `compile_sm75.py <repo>` | Compiles every Triton kernel of `solution/` for the T4 (sm_75) without a GPU |
| `test_cpu_kernels.py <repo>` | Head statistics kernel vs PyTorch (Triton interpreter) |
| `test_cpu_bnb.py <repo>` | NF4 linear + fp16 MLP vs bitsandbytes, bit-identical |
| `test_cpu_recompute.py <repo>` | Recomputed MLP vs the fp16 MLP vs bitsandbytes, bit-identical |
| `test_cpu_sdpa.py <repo>` | Packed attention (q/k/v kept, rows recomputed) vs per-row SDPA, bit-identical |
| `test_cpu_norm_link.py <repo>` | q/v LoRA and the recomputed MLP rebuilding their inputs from the norm's input, bit-packed dropout masks: bit-identical |
| `test_cpu_attn.py <repo>` | Fused RoPE kernels bit-exact; patched attention in a tiny Qwen2 + LoRA takes every new path |
| `test_cpu_model.py <repo>` | Head + padding-free decoder vs the original model: loss, metrics, rewind losses, LoRA gradients |
| `test_cpu_trainer.py <repo> <data_dir>` | A few TRL SFTTrainer steps with all patches: logged metrics and keys, rewind hook. `data_dir` holds a Qwen2.5 tokenizer (`tokenizer.json`, `tokenizer_config.json`, `config.json`) and a small alpaca-style `kaz.json` |

Run from the repository root, e.g. `python dev/test_cpu_sdpa.py .`. The tests that need the Triton
interpreter set `TRITON_INTERPRET=1` themselves.

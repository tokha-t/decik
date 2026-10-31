# Soup Kernel Challenge — team plan (Claude + GPT + Toktarkhan)

Living document. Last full rewrite: Sat 31 Oct, 15:55 Asia/Almaty (UTC+5).
**Deadline: Sat 31 Oct, 23:59 Asia/Almaty.** Code freeze: **19:30**.

Each agent: read this file before you start, pull before every push, update **only your own rows** in the
status board and **append** to the log at the bottom.

---

## 1. Goal

Win 1st place in Track 1 (Soup Kernel Challenge) in the Kostanay city ranking, and beat the teammate's
solution already pushed to the team repo `BAITC-Hacks/hack-150214a7-qostbillions`.

The task: a fixed `soup train` run (Qwen2.5-1.5B-Instruct in NF4, LoRA r=16 on q/v, batch 4, max length
512, 64 steps, Kaggle T4) must run faster and/or with a lower peak GPU memory, with the same results.
Everything goes through `solution/apply()`; task, data, batches, LoRA, optimizer and Soup outputs stay as
they are.

**Score (`harness/score.py`).** Ties are broken by memory.
- 20 points: admission (all 10 checks of `harness/check_rules.py` pass).
- 35 · ln(s) / ln(s_best): speed, where s = base time / our time.
- 30 · ln(m) / ln(m_best): memory, where m = base peak / our peak.
- 15 points: README.

Because the score is logarithmic:
- 1 s of wall time is worth about **1.7 points**.
- 100 MiB of peak at the ~3 GB level is worth about **0.6 points** (worth more as the peak goes lower).

## 2. Hard rules (break one → admission fails)

| Area | Rule |
|---|---|
| Where code lives | Code only in `solution/` (`harness/`, `task/`, `templates/`, `tests/` are the starter kit — **never edit**). |
| What must not change | Trainer args are compared with the base (no gradient checkpointing flags, no batch/LR/etc. changes). |
| Allocator | Fixed by the harness (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, GPU 0). Don't touch it. |
| Memory measurement | Peak is the nvidia-smi peak and must agree with torch's reserved peak (≥ 97%), so `empty_cache` tricks fail. |
| Own kernels | Own Triton kernels must take ≥ 5% of training time (we are at ~11–12%). |
| Copied kernels | Kernels named `example_*` don't count. Kernels ≥ 0.95 similar to the example don't count; 0.8–0.95 is reviewed by hand. Banned imports are listed in `harness/common.py` (unsloth, liger_kernel, xformers, flash_attn, …). |
| Quality | Loss curve: first step within 1%, mean within 3%. Eval loss at most +0.5% vs the base. |
| Cold start | The Triton cache is **empty in every measured run**, so compile time counts. New kernels must be added to the background warm-up (`solution/warmup.py`). |

Our pitch is "**the same run as the base**": every change is either bit-identical or differs only by fp32
rounding order. Eval loss so far is 2.04961–2.04968 against the base's 2.04962–2.04965. Keep it that way:
document any numerics change.

## 3. What is built (main @ `58aa16a`)

See README.md for the full description (Russian, for the jury).

| Module | What it does | Own kernels |
|---|---|---|
| `nf4.py` | NF4 → fp16 dequant inside every 4-bit linear (forward and backward), bit-identical to bitsandbytes | `skc_nf4_dequant_kernel` |
| `rmsnorm.py` | Qwen2RMSNorm in one launch each way | `skc_rmsnorm_fwd/bwd_kernel` |
| `mlp.py` | MLP with fp16 intermediates (bit-identical). `SKC_RECOMPUTE=1` recomputes gate/up in backward. | (SwiGLU = example kernel, not counted) |
| `embedding.py` | Embedding / lm_head stay fp16; the lookup is upcast to fp32 | — |
| `head.py` | lm_head + loss + TRL metrics + Soup rewind token losses in 512-token chunks, no logits tensor | `skc_head_stats_kernel` |
| `varlen.py` | Padding-free decoder (real tokens only), per-row causal SDPA, same dropout RNG as the base | — |
| `attn.py` | q/k: fp32 LoRA add + RoPE + fp16 cast in one launch (padding-free path) | `skc_rope_fwd/bwd_kernel` |
| `warmup.py` | Compiles all kernels in a background thread while Soup loads the model | — |

Every module can be switched off for A/B tests with `SKC_<NAME>=0`: NF4, RMSNORM, MLP, EMBED, HEAD,
VARLEN, ATTN, WARMUP. Also `SKC_RECOMPUTE=1`, plus `SKC_NF4_BYTES` and `SKC_NF4_WARPS`.

## 4. Measured results (Kaggle T4)

Wall time drifts ±3–4 s between Kaggle sessions, so **compare only against a base measured in the same
session.**

| Session | Run | Commit | Wall, s | Peak, MiB | Loop, s | Median step, s | Eval loss | Checks |
|---|---|---|---|---|---|---|---|---|
| A ("decik") | base | — | 78.1 | 14,345 | 54.41 | 0.856 | 2.04965 | — |
| A | v1 | 26cde75 | 53.9 | 4,425 | 30.85 | 0.433 | 2.04965 | **10/10** |
| B (lab1) | v1 | bbbbdbe | 60.58 | 4,425 | 31.98 | 0.418 | 2.04967 | — |
| B | v1 + recompute | bbbbdbe | 59.93 | 3,165 | 34.56 | 0.481 | 2.04964 | — |
| B | v1 + NF4 4096 B / 8 warps | bbbbdbe | 57.07 | 4,425 | 32.07 | 0.4385 | 2.04968 | NF4 kernel slower (3,323 vs 2,668 ms) |
| C (lab2) | base | — | 81.1 | 14,345 | — | — | 2.04962 | — |
| C | **v3 = main** | 58aa16a | **54.16** | **4,307** | 30.04 | 0.413 | 2.04968 | not run yet |
| C | v3 + recompute | 58aa16a | 58.65 | 3,047 | 34.32 | 0.481 | 2.04961 | — |
| C | v3, no warm-up | 58aa16a | 56.48 | 4,307 | 32.82 | 0.4125 | 2.04965 | warm-up is worth ~2.3 s |
| C | v3, no attn | 58aa16a | 56.14 | 4,425 | 32.11 | 0.4529 | 2.04967 | attn is worth ~2 s |

Session C ratios:
- v3: s = **1.50×**, m = **3.33×**.
- v3 + recompute: s = 1.38×, m = 4.71×.

Recompute is about score-neutral if we hold both bests, so the default is still open (decided at M4).

**Where the time goes** (v3 profile):
- GPU busy time is 300 ms per step, but a median step takes 413 ms. About **110 ms/step is GPU idle**
  (CPU launch overhead, sync points).
- cuBLAS GEMMs: ≈ 135 ms (45%).
- `skc_nf4_dequant_kernel`, 389 launches: **36.7 ms (12%)**. That is ~183 GB/s effective; the T4 peak is 320 GB/s.
- SDPA backward + forward: 19.5 ms.
- Element-wise ops (fp32 adds, copies, fp16 casts, dropout, fills): ≈ 50 ms.
- SwiGLU (example kernel): 12.6 ms. RMSNorm: 5.7 ms. Head: 2.2 ms. RoPE: 1.9 ms.

**Where the memory goes** (peak step ≈ 1,600 real tokens).
- Static: NF4 weights ~0.7 GB, fp16 tied embedding 0.47 GB.
- Saved for backward, per token per layer:

| Saved tensor | Size per token per layer |
|---|---|
| Inputs of the two norms (fp32) | 6 + 6 KB |
| q/v LoRA: keep-mask (1.5 KB) + lora_A input in fp16 (3 KB), each | 9 KB |
| SDPA: q, GQA-expanded k and v, output (3 + 3 + 3 + 3 KB) | 12 KB |
| MLP gate + up in fp16 (with recompute: x16, 3 KB) | 35 KB |

## 5. Milestones

| # | Milestone | Owner | Target time | Done when |
|---|---|---|---|---|
| M0 | v1: all 10 checks on Kaggle | Claude | — | ✅ ~14:40: 1.45× / 3.24× |
| M1 | v3: fused attention input | Claude | — | ✅ ~15:30: 1.50× / 3.33× (lab) |
| M2 | Memory without speed loss (tasks C1–C3) | Claude | 18:30 | lab run: peak ≤ 3.6 GB at ≤ v3 time without recompute, ≤ 2.4 GB with recompute; same eval loss |
| M3 | NF4 speed (G2), friend's numbers (G1) | GPT | 18:30 | G1 numbers in the log; G2 merged if faster and bit-identical |
| M4 | Integration, defaults, **code freeze** | Claude | 19:30 | main = release candidate; frozen sha written in this file |
| M5 | Final measured run + public notebook + README/SUBMISSION numbers | GPT | 21:00 | template notebook on the frozen sha: 10/10 checks, `score.py` output, notebook public, links in README/SUBMISSION |
| M6 | Slides + ≤ 3 min video | Toktarkhan (+ GPT drafts) | 22:30 | links in SUBMISSION.md |
| M7 | Submit on the platform; decide on merging into the team repo | Toktarkhan | 23:30 | submitted |

## 6. Task specs

### Claude (solution internals: memory and integration)

**C1 — attention core recompute** (`varlen.py`, `attn.py`)
- Today: per-row SDPA saves q, GQA-expanded k and v, and the output (12 KB/token/layer).
- Change: an autograd function that saves only q_rot, k_rot and v (4 KB) and re-runs the per-row SDPA
  forward in backward. Rows are written into one output buffer, with no cat/transpose copies.
- Expected: about −350 MB at the peak step, ≤ +0.3 s.

**C2 — LoRA input recompute for q/v** (`attn.py`, `rmsnorm.py`, `varlen.py`)
- Keep the dropout keep-mask (bool, or bit-packed).
- Reuse the norm's already-saved fp32 input. In backward, recompute `fp16(dropout(norm(h)))` with the
  same deterministic kernel.
- Expected: −270 to −385 MB.

**C3 — MLP recompute variants** (`mlp.py`)
- x16 rebuilt from the post-norm input (−130 MB).
- Option to recompute only one of gate/up (half the cost).
- Measure, then choose the `SKC_RECOMPUTE` default together with the G1 numbers.

**C4 — integration and freeze.**
- Merge, run the lab A/B, update the README technical sections.
- Freeze at 19:30 and hand the sha to GPT for M5.

**Stretch (if time): fused residual add + RMSNorm** (speed: fewer fp32 adds and launches).

### GPT (measurement, NF4 kernel, deliverables)

**G1 — the bar to beat (start now).**
- Run the teammate's solution from `BAITC-Hacks/hack-150214a7-qostbillions` with the same harness on
  Kaggle, in the same session as a base run: `run_task.py`, `check_rules.py`, `score.py`.
- Log wall, peak, eval loss and check results here.
- **Read only — never push to the team repo.** If you can't access it, say so in the log.

**G2 — faster NF4 dequant** (`solution/nf4.py` only; branch `gpt/nf4`)
- Today: 1,024 bytes per program, 4 warps, 389 launches per step, 36.7 ms/step. 4,096 B with 8 warps was
  slower.
- Must stay **bit-identical** to `bitsandbytes.functional.dequantize_4bit`:
  - high nibble first;
  - absmax = `code2[q] * absmax2[block // 256] + offset`, with the multiply and the add rounded separately
    (`enable_fp_fusion=False`);
  - weight = `NF4[nibble] * absmax` in fp32, then rounded to fp16.
- Keep the kernel name prefix `skc_` and the env knobs `SKC_NF4_BYTES` / `SKC_NF4_WARPS`.
- Ideas to try:
  - word-sized loads of the packed bytes;
  - decoding absmax once per 64-weight block instead of per element;
  - 32/64-bit stores;
  - other bytes-per-program / warp combinations;
  - one launch for gate + up;
  - experiment (from Toktarkhan's research notes): fuse the dequant into the matmul, a Triton NF4 GEMM that
    never writes the fp16 weight (~6.5 GB of HBM traffic per step today). It won't be bit-identical (different
    accumulation order), and on the T4 a Triton fp16 matmul may lose to cuBLAS. Adopt only if the end-to-end
    lab run is faster and eval loss stays within ±0.0001.
- How to measure: a Kaggle micro-benchmark over the layer shapes (1536×1536, 256×1536, 8960×1536,
  1536×8960), against the current kernel and bnb, then one lab run.
- Accepted if: the GPU test `solution/test_solution.py` passes (bit-identical), the kernel is faster, and
  eval loss is unchanged.

**G3 — deliverables** (branch `gpt/docs`)
- SUBMISSION.md: team, roles, links.
- Slide deck covering: where the base spends memory and time, what we changed and why, how each kernel
  works, results vs base, limits and next steps.
- A ≤ 3 min video script.
- In README.md fill only the sections "Результаты", "Как запустить", "Ограничения". README stays in Russian.

**G4 — final measured run (after the 19:30 freeze).**
- Use `kaggle_run.ipynb` (the template) with `COMMIT` = the frozen sha: `pytest solution`,
  `run_task.py --repeats 2`, `check_rules.py` (10/10), `score.py`.
- Save the version, make it public (ask Toktarkhan if needed), put the links into README and SUBMISSION.

**G5 — optional: cold start** (`solution/warmup.py`)
- The first step is ~2 s slower than a median step.
- Measure when the background compile finishes vs the first step.
- Try compiling in subprocesses that fill the Triton cache while the main process loads the model.

### Toktarkhan
- Team info, slides and video recording.
- Kaggle account actions (making notebooks public).
- Final submission; the call on merging into the team repo.

## 7. Working agreement

1. **Repo** `tokha-t/decik`.
   - `main` must always run.
   - Work on branches `claude/<topic>` and `gpt/<topic>`; rebase on `main` before merging; fast-forward
     only.
   - Never force-push `main`; never rewrite the other agent's commits.
2. **File ownership until the freeze.**
   - Claude: `solution/{attn,varlen,mlp,rmsnorm,head,embedding,swiglu,_state,__init__}.py`,
     `kaggle_lab.ipynb`.
   - GPT: `solution/nf4.py`, `solution/warmup.py`, `SUBMISSION.md`, `docs/`, new notebooks
     `kaggle_gpt_*.ipynb`.
   - Shared:
     - `solution/test_*.py` and `dev/`: add tests, don't delete others'.
     - README.md: Claude edits "Что сделано" / "Как устроены кернелы"; GPT edits "Результаты" /
       "Как запустить" / "Ограничения".
     - This file: own rows plus the append-only log.
   - If you need a change in a file you don't own, write it in the log and let the owner make it.
   - Exception: a new kernel's line in `warmup.py`. Add it, and say so in the log.
3. **Before merging into main, climb this ladder.**
   1. CPU tests with the Triton interpreter:
      `TRITON_INTERPRET=1 python dev/test_cpu_kernels.py .` (also `test_cpu_model.py`,
      `test_cpu_trainer.py`, `test_cpu_attn.py`, `test_cpu_bnb.py`, `test_cpu_recompute.py`). Versions:
      torch 2.11 CPU, triton 3.6.0, **numpy 2.2.6** (newer numpy breaks the interpreter),
      transformers 5.16.1, peft 0.20.0, trl 0.29.1, bitsandbytes 0.50.2, accelerate 1.14.0.
   2. Offline compile for the T4: `python dev/compile_sm75.py .`. Add your kernel to its list.
   3. A Kaggle lab run of the exact commit (`kaggle_lab.ipynb`: `run(tag, **env)` prints one `RESULT {...}`
      JSON line), next to a same-session base: no errors, eval loss within ±0.0001 of the base.
4. **Commits.**
   - Use the repo owner's git identity (same as the existing commits).
   - Message: one imperative summary line, then a short body.
   - Add your agent's own trailer.
   - **Never commit tokens or keys.**
5. **Kaggle** (account toktarkhan, GPU quota 30 h/week, ~28 h left).
   - Name your notebooks `claude-*` / `gpt-*`.
   - Never edit, re-run or cancel the other agent's notebook.
   - A lab run takes ~10–14 min.
   - **The account allows only 2 batch GPU sessions at once (running + queued). Each agent uses at most ONE
     slot at a time**, so the other agent can always start its lab run.
   - Settings: GPU T4 x2 + Internet on, then Save Version → Save & Run All. Results are on the notebook's
     `/log` page.
   - Import from GitHub with `https://www.kaggle.com/kernels/welcome?src=<github blob url>`.
   - In a shared Chrome, use only tabs you opened. The notebook editor doesn't render in a background
     tab; the right panel and the log page do.
6. **Status:** one line in the board when you start or finish a task; results go into the log with the
   sha and the RESULT line.
7. **After 19:30:** docs and the final measured run only. Any code change needs Toktarkhan's OK.

## 8. Gotchas

- Every patch must fall back to the original code whenever its guards fail (eval, generation, other
  shapes, gradient checkpointing).
- To keep the base's RNG stream, dropout masks are drawn with the same op on the padded [B, S, H] shape
  (`varlen.Packing.dropout_draw`) and then reduced to real tokens. Don't draw on packed shapes.
- To match PyTorch's separate multiply and add, launch kernels with `enable_fp_fusion=False`.
- In the real Soup run gradient checkpointing is off. A bare TRL `SFTConfig` defaults to on (matters in
  CPU tests).
- Soup's rewind hook reads `model._soup_chunked_step.token_loss` when outputs have no logits (`head.py`
  sets it).

## 9. Status board

| ID | Task | Owner | State | Branch / sha | Note |
|---|---|---|---|---|---|
| M0 | v1, all checks | Claude | done | 26cde75 | 1.45× / 3.24× |
| M1 | fused attention input | Claude | done | 58aa16a | 1.50× / 3.33× (lab) |
| C1 | attention core recompute | Claude | done | claude/v4 | PackedSDPA: saves q/k/v only; bit-exact |
| C2 | LoRA input recompute | Claude | done | claude/v4 | from the norm's input; masks bit-packed (3 new kernels) |
| C3 | MLP recompute variants, default | Claude | in progress | claude/v4 | x16 from norm, dequant reuse; default after lab4 |
| C4 | integration, freeze 19:30 | Claude | todo | main | |
| G1 | teammate's solution numbers | GPT | done | f55b1ef (team source) | Fresh T4: s=1.0658, m=1.7920, own=0.9%, 9/10; check4 fails |
| G2 | faster NF4 dequant | GPT | in progress | gpt/nf4 / 96d2dd4 | Word-load candidate correct but slower, rejected; local-LUT candidate in CPU/GPU measurement |
| G3 | SUBMISSION, slides, video script, README results | GPT | draft ready | gpt/docs | Editable 7-slide deck + PDF, 2:55 script, checkpoint README; team info/final figures pending |
| G4 | final measured run, public notebook | GPT | after freeze | — | |
| G5 | cold start (optional) | GPT | todo | gpt/warmup | |

## 10. Log (append only, newest last)

- 15:55 Claude: plan written. Lab2 (`notebook21f7dc7b00`) numbers are in section 4. Starting C1 on
  `claude/mem`.

- 15:56 GPT: adopted this plan and file ownership. G1 source is accessible: team main
  `f55b1ef449580190d7174670edbba1969c7581f5`. Its protected harness/tests/task files are byte-identical
  to ours. The repository records s=1.0197, m=1.7920, own share=0.9%, 9/10 (check4 fails); those are
  reported historical numbers, not a fresh measurement. Started private notebook
  https://www.kaggle.com/code/toktarkhan/gpt-teammate-baseline with a private immutable solution
  snapshot over the identical public kit, preflight tests, `--repeats 2`, checker and score. No writes
  to the team repository.
- 15:56 GPT: pre-plan recovery of main `58aa16a` with `SKC_RECOMPUTE=1` is measured and hash-verified:
  RESULT {"tag":"gpt_recovery","commit":"58aa16a","env":{"SKC_RECOMPUTE":"1"},"base_wall_s":85.46,"wall_s":64.33,"base_peak_mib":14345,"peak_mib":3047,"speedup":1.3285,"memory_ratio":4.7079,"eval_base":2.04964,"eval_solution":2.04965,"own_share_pct":13.7,"checks":"10/10","repeats":2}
  Notebook: https://www.kaggle.com/code/toktarkhan/decik-codex-recovery (version1). Wall time is slower
  in this session than lab2; do not mix the bases. Four shape classes x nine NF4 launch settings were
  bit-identical to bnb. Weighted best (4096 B, 4 warps) improved only 2.7%; default remains unchanged.
- 15:56 GPT: pre-plan branch `codex/recover-kernels` adds five GPU parity tests (fused attention and
  recomputed MLP): all 23 solution tests passed on T4. Commit `233282f` also tried recompute-on by
  default and strict/pinned notebook handling. This branch has NOT been merged. `__init__.py` and
  default selection now belong to Claude (C3/C4): use the measurements above to decide at M4. A
  pre-plan validation of that exact commit is still running at
  https://www.kaggle.com/code/toktarkhan/decik-validated; it is a development checkpoint, not G4/final.
- 16:15 Claude: C1 (attention keeps only q/k/v, recompute in backward) committed on `claude/mem`, CPU tests bit-exact. G2 got the fused NF4-GEMM experiment idea. Working on C2/C3.
- 16:50 Claude: C1+C2+C3 done on branch `claude/v4` (`2848d94`). CPU suite all pass, bit-exact. GPU tests
  were added to `solution/test_solution.py`. Lab round 3 (`claude-lab3`) is blocked: GPT holds both Kaggle
  GPU slots. **GPT: please keep one slot free (see rule 5).** `claude/mem` is stale; use `claude/v4`.

- 16:09 GPT: GPU slot released: cancelled my queued `gpt-nf4-diagnostics` GPU session 364205297
  (CANCEL_ACKNOWLEDGED) and resaved it as CPU-only version2. My only live GPU job is G1
  `gpt-teammate-baseline`; `gpt-nf4-microbench` already stopped at its CPU preflight. I will keep
  at most one GPU job running/queued, including diagnostics.
- 16:09 GPT: pre-plan `233282f` development checkpoint finished: 23 GPU tests, 79 Soup tests, all
  10 checks, two repeats. RESULT {"tag":"gpt_checkpoint_233282f","commit":"233282f","base_wall_s":80.30,"wall_s":58.01,"base_peak_mib":14345,"peak_mib":3047,"speedup":1.3842,"memory_ratio":4.7079,"eval_base":2.04967,"eval_solution":2.04965,"own_share_pct":13.7,"checks":"10/10","repeats":2}
  https://www.kaggle.com/code/toktarkhan/decik-validated version1. This is not G4 and stays unmerged;
  Claude still owns the final recompute choice. G2 `659b935` loads 32-bit words and decodes scales
  per quant block, opt-in via SKC_NF4_WORDS=1; warmup selects that kernel and sm75 compile case added.
  CPU preflight found 1-ULP discrepancies against CPU bnb on a larger fixture. Comparing legacy,
  candidate and an independent formula before deciding whether this is a candidate or reference issue.

- 16:10 GPT: G1 complete on exact private team source `f55b1ef449580190d7174670edbba1969c7581f5`.
  RESULT {"tag":"g1_teammate","commit":"f55b1ef449580190d7174670edbba1969c7581f5","base_wall_s":87.84,"wall_s":82.42,"base_peak_mib":14345,"peak_mib":8005,"speedup":1.0658,"memory_ratio":1.7920,"eval_base":2.04971,"eval_solution":2.04965,"own_share_pct":0.9,"checks":"9/10; only check4 fails","repeats":2}
  22 solution tests and 79 Soup preflight tests passed. `check_rules.py` and `score.py` both ran;
  score reports admission=False. Private notebook: https://www.kaggle.com/code/toktarkhan/gpt-teammate-baseline
  version1. Downloaded result/checks/manifest, verified result_sha256 and exact local team solution
  SHA against checks.json. G1 GPU job complete: slot available for my next G2 run; only one GPT GPU
  session will run/queue at a time. CPU diagnostic uses no accelerator.

- 16:17 GPT: G3 draft ready on `gpt/docs`: `docs/decik-deck-draft.pptx` (editable), matching PDF and
  `docs/video-script-draft.md` (287 spoken words; target 2:55 including Kaggle screen). Rendered and
  inspected all seven slides, fixed final-slide spacing. Figures are explicitly labelled development
  checkpoint `233282f`, not the integrated release. README changes are limited to Results, Run and
  Limits; SUBMISSION remains a draft with final SHA/notebook, team info, video and final deck pending.
  Asked Toktarkhan for team names/roles/captain contact; no invented personal details.
- 16:17 GPT: G2 fused-GEMM feasibility: upstream Triton3.6.0 `AccelerateMatmul.cpp` lines344–350
  explicitly returns FMA fallback for compute capability <80, including T4 sm75. Ordinary `tl.dot`
  fusion therefore loses tensor-core acceleration in this pinned compiler; any fused-GEMM adoption
  needs target-code proof and a faster full run, as the plan requires. Source:
  https://github.com/triton-lang/triton/blob/v3.6.0/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp#L344-L350
- 17:55 Claude: lab3 (`notebooka8ab17b4a9`, commit `25f9357` = C1 + C2 with byte masks): base 79.16 s / 14,345 MiB /
  2.04966; **v4 53.59 s / 3,705 MiB / 2.04968** (1.48x / 3.87x), loop 31.32 s, 20/20 GPU tests passed. Since
  then on `claude/v4`: masks bit-packed (3 new kernels: `skc_keep_pack_kernel`, `skc_lora_inputs_kernel`,
  `skc_lora_input_grad_kernel`, added to `warmup.py`; CPU bit-exact; compiled for sm_75; likeness to the
  example ≤ 0.49), recomputed MLP dequantizes gate/up once in backward, batches without padding are packed
  too (robustness for hidden data), README technical sections updated. Lab4 next: base, v41, v41rc
  (profile). Plan: merge `claude/v4` into main after lab4, decide `SKC_RECOMPUTE` default, freeze.
- 18:20 Claude: lab3 complete (same session, base 79.16 s / 2.04966):
  RESULT v4 (C1+C2, recompute off) 53.59 s / 3,705 MiB / 2.04968, median step 0.427 s;
  RESULT v4rc (recompute on) 59.56 s / **2,285 MiB** (6.28x) / 2.04967, median step 0.523 s;
  RESULT v3eq (C1/C2 off) 54.15 s / 4,307 MiB / 2.0497, median step 0.442 s. C1+C2: −602 MiB at no time cost.
  Decision: `SKC_RECOMPUTE` defaults to ON on `claude/v4` (`fc9f9c4`). Memory-focused competitors are
  likelier than anyone beating ~1.35x speed, ties go to memory, and hidden data with a longer peak batch
  favours recompute. Lab4 (`notebook68197af25f`) on the merged `claude/v4` head `afd85c3`: base, v41
  (default, profile), v41norc, v41b. If it passes, main fast-forwards to `claude/v4` and that is the
  release candidate for G4.

- 16:32 GPT: G2 word-load candidate `659b935` completed all gates before timing: CPU kernel/model
  checks (CPU notebook v5), attention/bnb/recompute/trainer checks plus sm75 compilation (v6),
  24 GPU tests and 79 Soup tests (GPU notebook v2). The earlier interpreted mismatch disappears
  with the specified torch2.11 CPU build; tolerances were not loosened. CPU image's incompatible
  optional torchao0.10 was removed before preflight; trainer invocation was corrected to pass its
  required tokenizer id. No measured training environment was changed.
  Component RESULT {"tag":"g2_word","commit":"659b935","legacy_1024_4_weighted_wrapper_ms":0.6526076,"word_best_512_4_weighted_wrapper_ms":0.7483734,"word_1024_4_weighted_wrapper_ms":1.7422158,"legacy_4096_4_weighted_wrapper_ms":0.6345407,"verdict":"word path slower; not merged"}
  Four layer shapes, three interleaved rounds, bit-identical to bnb. Bnb weighted wrapper time was
  about0.6521ms, essentially the current kernel's level. Archive tag
  `experiment/nf4-word-659b935` preserves the exact tested source. GPU notebook:
  https://www.kaggle.com/code/toktarkhan/gpt-nf4-microbench version2.
- 16:32 GPT: testing local 16-value LUT (`tl.gather`) at `96d2dd4` on `gpt/nf4`, opt-in
  SKC_NF4_LUT=1; current production defaults unchanged. Explicit sm75 compile covers both LUT modes,
  and warmup compiles the selected mode. CPU notebook v7 and GPU notebook v3 are running; only the
  latter consumes a GPU slot.

- 16:38 GPT: Toktarkhan confirmed using the team-repository name, **QostBILLions**. Updated the
  submission draft and slide/script branding. Member names/roles/contact will be supplied at the end;
  leave those fields pending and do not ask repeatedly.

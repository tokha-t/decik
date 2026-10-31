# Final validation evidence — v5.2

Kaggle notebook `toktarkhan/notebook72649ae047`, version 1 (kernel session 364288750), measured implementation
[`8414ab4`](https://github.com/tokha-t/decik/commit/8414ab48cdc23c2ddb509f5418f41c2457d40811), defaults only.
Same protocol as G7: [`kaggle_claude_v52_final.ipynb`](../../kaggle_claude_v52_final.ipynb), GPU T4 x2 (one T4 used),
cold compiler caches, unmodified harness, `run_task.py --repeats 2` in ABBA order, then `check_rules.py` and `score.py`.
Previous evidence is preserved: [G7 / v5.1](../validation/README.md), [G4 / v4](../validation-v4/README.md).

| | Base Soup 0.77.0 | v5.2 | Ratio |
|---|---:|---:|---:|
| `soup train` wall time, s (runs) | 90.82 / 84.91 | 60.48 / 62.13 | |
| Wall time, s (median) | 87.86 | 61.30 | speedup **1.4333x** |
| Training loop, s (median) | 59.01 | 36.55 | 1.61x |
| GPU peak (nvidia-smi), MiB | 14,345 | 2,025 | base/ours **7.084x** |
| Eval loss (public rows) | 2.04963 | 2.04963 | +0.000% |
| Own kernels (check 4) | — | 13.4% | ≥ 5% |

All 10 checks pass; `score.py`: "all checks passed on this result: True". 22 GPU tests of the solution and
79 Soup regression tests pass (and the checker runs the 79 Soup tests again).

- [checks.json](checks.json): 10/10 verdict with result and source hashes, copied from the notebook output.
- [manifest.json](manifest.json): the notebook's final manifest (versions, SHA, test counts, summary). Rebuilt from the
  `V52_FINAL_RESULT` line of the log, which prints the same dictionary; same 1,160 bytes as the file in the output.
- The full `result.json` (237 kB) and `v52-evidence.zip` (42 kB, also with the test and checker logs) are in the
  notebook output `v52-evidence/`.

Result SHA-256: `57cca17d9ba4410a565622d6b028efc165ca1677bda034c2712f6431e839ec09`.

Solution SHA-256: `f80525c09da968d19323823b27b193fcedbe44da8fca6636c88b372a6327f78a`
(sorted relative paths and Git blob bytes of `solution/`, as the checker computes it).

Absolute times differ between Kaggle sessions (base 76–91 s across our sessions), so ratios are compared within one
session. The same-session A/B against v5.1 is lab 7: v5.2 53.25 / 53.66 s at 2,025 MiB vs v5.1 53.65 / 54.10 s
at 2,035 MiB.

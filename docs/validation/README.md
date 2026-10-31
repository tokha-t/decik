# Final G7 validation evidence — v5.1

[Public saved version 364261503](https://www.kaggle.com/code/toktarkhan/gpt-v51-final-validation?scriptVersionId=364261503), measured implementation
[`d9ab004`](https://github.com/tokha-t/decik/commit/d9ab00419faa5d81ddf48fb35ec562284e5d62f9), defaults only. Previous [v4/G4 evidence](../validation-v4/README.md) is preserved.

- [result.json](result.json): original harness output, two base and two solution runs.
- [checks.json](checks.json): 10/10 verdict and result/source hashes.
- [manifest.json](manifest.json): original G7 manifest, versions, SHA and summary.
- [gpu-tests.log](gpu-tests.log): 21 GPU tests passed.
- [soup-preflight.log](soup-preflight.log): 79 Soup tests passed with solution enabled.
- [checks.log](checks.log): rule checker output, including another 79 Soup tests.
- [score.log](score.log): all checks passed on this result; points require best team ratios and README review.
- [Original ZIP](g7-evidence.zip) and [independent verification](verification.json).
- [publication-provenance.json](publication-provenance.json): separate publication identity and hash notes.

Original Kaggle result/checks/manifest/log bytes are copied unchanged. Local publication provenance
is separate from the original notebook manifest/archive. Source hashing uses sorted relative paths
and Git blob bytes in `solution/`, as the checker does; it includes solution tests, excluding caches.

Result SHA-256: `9366fe839448718f404a08264e13887d98990ad6837402d35d82ad0bd74ddcf2`.

Solution SHA-256: `bdf21a6adb90887482ad0facaa421dc0173e8ae075d8ba4678c4166d91fcaa42`.

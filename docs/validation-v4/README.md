# Final validation evidence

[Public Kaggle saved version 364217337](https://www.kaggle.com/code/toktarkhan/qostbillions-final-validation?scriptVersionId=364217337), frozen implementation
[`5c1bbe5`](https://github.com/tokha-t/decik/commit/5c1bbe560c08b0323eefeb843a0baf00e400e774), defaults only.

- [result.json](result.json): original harness output (two base and two solution runs).
- [checks.json](checks.json): 10/10 verdict and hashes.
- [manifest.json](manifest.json): original notebook manifest, versions and summary.
- [gpu-tests.log](gpu-tests.log): 20 GPU tests passed.
- [soup-preflight.log](soup-preflight.log): 79 Soup tests passed with solution applied.
- [checks.log](checks.log): rule checker output, including a further 79 Soup tests.
- [score.log](score.log): `all checks passed on this result: True`; total points require other teams' best ratios and README review.
- [provenance.json](provenance.json): separately recorded public notebook identity and hash-verification notes.

The original Kaggle JSON/log bytes are preserved. `provenance.json` was added locally;
it is separate from the notebook's original manifest and ZIP.

Result SHA-256: `7434230cff6fa598cada142143d6ded954a4c328531df6c2c5e89d3d030265f5`.

Solution SHA-256: `d6ef76a142b4ffbbaaad584201fd25469136f15c69ca37106b062b2c9283e6ab`.

The source hash is computed as the checker does: sorted paths relative to `solution/`, followed
by each file's bytes. It was checked against the exact frozen Git blobs, not the working-tree label.
Later commits only add documentation and validation material; they do not change `solution/`.

# Repository instructions

The active Python implementation is `tensorbalance/` on `main`. `tensor_balance.py` and `export_csv.py` are command wrappers. Keep the README focused on three sections: the problem, how the current implementation solves it, and the code needed to use it. Do not add release history or migration guidance. Give the user one sentence describing each repository update.

Before changing behavior, read `README.md` and the affected modules. Explain planned changes in plain language before editing. Do not push or pull from GitHub without the user's explicit request.

Use Python 3.10 or newer. Install with `python -m pip install -e '.[parquet]'`. Run `python -m unittest discover -s tests -v` before and after behavior changes. For numerical changes, also run `python audit_numerics.py --out results/new_numerical_audit.json`. Put generated output in ignored `results/` directories.

Keep the numerical contract: input is a complete, finite, nonnegative, symmetric cube; replace only exact zeros with the same fixed `1e-12` across datasets, leaving positive counts unchanged; export `balanced_tensor.npy`, `bias_matrix.npy`, `bias_vector.npy`, and `reconstruction_scalar.npy`; and make every exported tensor fiber sum to one. Preserve checks that reject incomplete results and existing output paths. Segment-record Parquet needs an explicit bin size, while already-binned input retains its bins and resolution.

Preserve example fixtures and audit evidence. Never commit private data, credentials, or generated research results. Update covered-file hashes in `MANIFEST_SHA256.json` deliberately when files change. Avoid unrelated formatting changes.

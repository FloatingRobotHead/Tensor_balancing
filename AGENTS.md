# Repository Guidelines

## Project Structure & Module Organization

`tensorbalance/` contains the Python package: `cli.py` handles commands, `input.py` validates file formats, `decompose.py` exposes `balance`, and `storage.py` publishes outputs safely. Numerical methods live in `tensorbalance/solvers/`. Optional visualization uses `visualize.py` and packaged `view.html`.

`tests/` contains interface and numerical regression tests. `examples/` holds simulated inputs and reviewed reference outputs; preserve these fixtures. `audit/` separates current and historical evidence; the root audit report is an index. Read `handoff.md`, `README.md`, and `AUDIT_REPORT.md` before changing behavior. Put generated runs in ignored `results/` directories.

## Build, Test, and Development Commands

Use Python 3.10 or newer:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
tensor-balance examples/patchwork.npy --out results/smoke_dh
tensor-export-csv results/smoke_dh --out results/smoke_csv
```

Editable installation uses setuptools and installs NumPy/SciPy. The smoke commands normalize a tensor and export its results. Choose new output directories for repeated runs. CI installs the package and runs unittest on Python 3.11.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` functions/modules, `PascalCase` classes, and uppercase constants. Follow nearby Python/NumPy conventions and document mathematical changes explicitly. No formatter or linter is configured; avoid unrelated formatting changes.

## Testing Guidelines

Use standard-library `unittest`: files named `test_*.py`, methods named `test_*`, and temporary directories for generated files. Run the existing suite before and after behavioral changes. Solver changes also require known-factor recovery, reconstruction, marginal-balance, and reference comparisons across methods. No coverage percentage is configured.

For broader numerical validation, write a new report:

```sh
python audit_numerics.py --out results/new_numerical_audit.json
```

## Commit & Pull Request Guidelines

Existing commits use concise imperative subjects, such as “Add development handoff”; follow that style. PRs should explain the problem, resulting behavior, validation commands/results, and any mathematical or input/output changes. Link relevant issues; include screenshots for viewer changes. Update `handoff.md` when workflows or behavior change.

## Data & Numerical Contracts

Preserve the four default NPY outputs: balanced_tensor.npy, bias_matrix.npy, bias_vector.npy, and reconstruction_scalar.npy. Exported tensor fibers sum to one. The fixed `0.5` pseudocount applies to every cell. Preserve the symmetric complete-cube requirement and protection against overwrites or partial results. Keep visualization optional. Segment-record Parquet accepts an explicit bin size; already-binned inputs retain their resolution. Never commit credentials, private research data, or downloaded datasets. Update covered-file integrity manifests deliberately; distinguish fresh validation from historical audit claims.

## Canonical Source and Historical Copies

The active implementation is `tensorbalance/` in this repository on `main`.
The root `tensor_balance.py` and `export_csv.py` are thin entry-point wrappers.
The different modules in `tensorbalance/solvers/` are supported algorithms and
shared helpers, not obsolete copies. Read imports before considering removal.

Do not use `results/`, `build/`, old installed test targets, downloaded ZIPs, or
external archive folders as implementation sources. Scripts under `results/`
record historical experiments. Preserve their numerical outputs and provenance.
Current release evidence is in `audit/v0.2.0/`; older evidence is labeled in
`audit/v0.1.1/` and `audit/README.md`. Preserve historical example fixtures.

The user requires an explicit request before GitHub pushes or pulls. Cleanup
authorization for this task does not authorize future automatic synchronization.
Explain planned changes in plain language before making them.

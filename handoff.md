# Development handoff

Updated 2026-11-11 for Tensor Balance CLI 0.2.0.

Read README.md, START_HERE.txt, and RELEASE_NOTES.md for the current contract.
AUDIT_REPORT.md and the root numerical reports describe the historical 0.1.1
release. Reference outputs in examples/ retain their historical conventions.
Current release evidence lives in audit/v0.2.0/.

## User requirements

Explain actions in plain language before making changes. The user is new to
command-line software. Work locally unless the user explicitly requests a
GitHub push or pull. Authorization to publish this release does not authorize
future synchronization. Never commit private datasets or generated results.

The user selected 25 kb for the chromosome 21 test. Raw segment Parquet accepts
a chosen bin size. Already-binned inputs preserve their bins, order, and
resolution; physical width is unknown when no coordinate metadata exists.
The pseudocount remains 0.5 added once to every cell, including observed cells.
Earlier zero-only and pseudocount-sweep experiments did not change this default.

## Mathematical and output contract

Default output is four machine-readable float64 arrays:

- balanced_tensor.npy with shape (n,n,n), each fiber summing to 1.
- bias_matrix.npy with shape (n,n), row/column geometric means 1.
- bias_vector.npy with shape (n,), geometric mean 1.
- reconstruction_scalar.npy with shape (), the positive overall scale C.

Reconstruction is raw+0.5 = C*b_i*b_j*b_k*M_ij*M_ik*M_jk*T_ijk.
The tensor total is n*n and its mean is 1/n. The multiplier 1/n converts the
internal mean-one tensor to the exported tensor and has already been applied.
It is distinct from C. Do not apply it twice.

The balance(raw) Python API retains its historical mean-one tensor and vector
with absorbed scale. Use unit_fiber_result(balance(raw)) for the current
four-factor convention. Its field is reconstruction_scalar; scalar remains a
read-only compatibility alias. Exporters accept the early development filename
scalar.npy and reject conflicting scalar aliases. Legacy three-factor outputs
remain readable by CSV export. Legacy compact expansion converts to unit fibers.

DH, Log ANOVA and Binomial support complete, finite, nonnegative symmetric
cubes. Binomial requires actual exposures. The compact segment Parquet workflow
supports DH, pools all supplied rows, bins segment midpoints, expands distinct
permutations, and requires explicit acceptance of unobserved combinations as
zeros. It writes the complete tensor slabwise plus compact/provenance files.
No masked or structural-zero balancing is implemented.

## Development and packaging

Use Python 3.10 or newer. Install with `python -m pip install -e ".[parquet]"`.
Run `python -m unittest discover -s tests -v`. CI installs the Parquet extra.
For numerical changes also run `python audit_numerics.py --out NEW_REPORT.json`.
Use a new output directory for every command; preserve overwrite protection.

Core modules are decompose.py, input.py and cli.py in tensorbalance/. Compact
DH is in solvers/compact_dh.py; parquet_pipeline.py prepares segments and runs
normalization; export_dense.py expands compact results; export_csv.py exports
saved factors. Visualization is optional and labels display scaling separately.

Version and scripts are defined in pyproject.toml. MANIFEST.in includes source,
tests, documentation, public simulated examples and audit evidence in the source
distribution. Private inputs and large output files must never be packaged.
Refresh MANIFEST_SHA256.json deliberately when covered files change.

## Evidence and limits

The Linux chromosome 21 run at 25 kb produced 1,868 bins and checked all fiber
sums and all reconstructed cells. Large data and run artifacts remain local in
ignored results/. The naming change does not require recomputing those arrays.
Numerical checks do not establish biological interpretation or validate the
choice of 0.5 for every resolution. Native Windows/macOS and minimum dependency
versions have not been tested for this release.

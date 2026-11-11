# Tensor Balance CLI 0.2.0

This release accepts raw segment-record Parquet at a chosen genomic bin size,
preserves the resolution of already-binned tensors, and exports four numerical
factors. Visualization is optional. The pseudocount remains 0.5 added to every
cell.

## Outputs and migration

The four default NPY files are `balanced_tensor.npy`, `bias_matrix.npy`,
`bias_vector.npy`, and `reconstruction_scalar.npy`. Each tensor fiber sums to
one within numerical tolerance. The vector and each matrix row/column have
geometric mean one.

```text
raw[i,j,k] + 0.5 = C * b[i]*b[j]*b[k] * M[i,j]*M[i,k]*M[j,k] * T[i,j,k]
```

C is the reconstruction scalar. It restores the overall input scale when
recombining the factors. The internal mean-one tensor is divided by n before
saving, and this multiplier 1/n has already been applied. The saved tensor
has total n² and mean 1/n. Do not divide it by n again.

The Python `balance` API retains its legacy mean-one convention. Use
`unit_fiber_result(balance(raw))` and its `reconstruction_scalar` field for
this release's output convention. The former `scalar` field remains a read-only
alias. Exporters accept the early development filename `scalar.npy`; new
outputs use the explicit name. Conflicting aliases are rejected. Older
three-file CSV exports retain their original values. Dense expansion converts
legacy compact results to the new four-factor convention.

Historical example results and audit files retain their original conventions.
They are not examples of the new unit-fiber output layout.

## Resolution and data preparation

Install with `python -m pip install ".[parquet]"` from the source directory.
For raw segment Parquet, supply `--bin-size` in base pairs, `--chromosome`, and
`--unobserved-are-zero`. For example, 25000 selects 25 kb. The documented schema
uses segA/segB/segC chromosome, start and end fields. It bins segment midpoints,
counts every supplied row once, pools samples/haplotypes, and expands each
unordered triplet into distinct permutations. It does not add quality filtering
or deduplication. The zero assumption is explicit.

Already-binned inputs reject `--bin-size` and preserve shape, ordering, gaps,
and supplied coordinate metadata. A plain array does not reveal its physical
bin width. Sparse coordinate tables and masked tensors are unsupported.

Compact DH avoids allocating the full tensor during iteration. It still
exports the complete tensor to disk in slabs and checks every reconstruction
and every fiber sum. A 1,868-bin float64 tensor occupies about 52.15 GB.
Use an allocated compute node for large Voltron jobs.

## Validation

Current evidence is in `audit/v0.2.0/`. The regression suite has 46 tests,
including Parquet ingestion, resolution preservation, compact/dense agreement,
all three methods, scalar naming compatibility, and rejection of conflicting
scalar files. The numerical audit checks 28 inputs across three methods,
84 successful runs, with an independently implemented IPF reference. One
Binomial case needs the documented extended iteration limit.

The release package is built and installed separately for a command-line smoke
test. Validation runs on Linux with Python 3.11.6, NumPy 2.4.6, SciPy 1.17.1,
and PyArrow 26.0.1. Native Windows/macOS and minimum dependency versions have
not been tested for this release. Numerical agreement does not establish a
biological interpretation or validate the pseudocount choice at every resolution.

## Package contents

The source distribution includes code, tests, documentation, audit evidence,
and the public simulated patchwork example. It excludes chromosome datasets,
local configuration, and generated research outputs. The wheel installs the
`tensorbalance` package and four commands, `tensor-balance`, `tensor-parquet`,
`tensor-export-csv`, and `tensor-export-dense`.

`MANIFEST_SHA256.json` records source-file checksums. Downloaded release archives
have separate SHA-256 checksums. `START_HERE.txt` provides installation and usage
instructions. Root audit reports describe the historical 0.1.1 release.

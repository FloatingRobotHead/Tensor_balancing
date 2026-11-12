# Tensor Balance

## Problem

A three-way contact count tensor mixes an overall count scale, biases for individual bins, biases for pairs of bins, and a remaining three-way signal. Tensor Balance separates those parts so the remaining tensor can be compared across bins.

Input is a complete, finite, nonnegative, permutation-symmetric `n × n × n` cube. Each axis uses the same bin order. A missing measurement is not a zero. The program also accepts segment-record Parquet and bins it at a width you choose.

## How we do it

We add `0.5` to every count, balance the cube, and extract four factors:

```text
raw[i,j,k] + 0.5 ≈ C · b[i]b[j]b[k] · M[i,j]M[i,k]M[j,k] · T[i,j,k]
```

`C` is the overall scale, `b` is the bin bias vector, `M` is the pair bias matrix, and `T` is the balanced tensor. Every fiber of `T` sums to `1`; the geometric mean of `b` and of each row and column of `M` is `1`. Multiplying the factors and subtracting `0.5` reconstructs the input counts. The program checks convergence, fiber sums, and reconstruction before writing output.

DH is the default balancing method. Log ANOVA and Binomial are also available for already-binned tensors. Binomial needs the actual number of snapshots or exposures; it cannot be inferred from the counts. Segment-record Parquet uses DH, bins segment midpoints, and requires explicit acceptance that unobserved triplets have zero counts. Already-binned input keeps its bin order and resolution.

## Code

Python 3.10 or newer is required. From the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/tensor-balance my_tensor.npy --out results/my_run
```

Use a new output directory for each run. The input file is never changed. Supported already-binned inputs include NPY, NPZ, JSON, ZIP, and complete coordinate CSV/TSV with columns `i,j,k,value` and all `n³` entries.

The output directory contains four NumPy arrays:

| File | Contents |
| --- | --- |
| `reconstruction_scalar.npy` | `C`, one positive number |
| `bias_vector.npy` | `b`, shape `(n,)` |
| `bias_matrix.npy` | `M`, shape `(n,n)` |
| `balanced_tensor.npy` | `T`, shape `(n,n,n)` |

Select another method or export the saved arrays as CSV:

```sh
.venv/bin/tensor-balance my_tensor.npy --method log-anova --out results/my_log_anova
.venv/bin/tensor-balance my_tensor.npz --method binomial --out results/my_binomial
.venv/bin/tensor-export-csv results/my_run --out results/my_csv
```

For Binomial input without exposure metadata, add `--snapshots S` with your measured `S`. For segment-record Parquet, install the optional dependency and choose a bin size in base pairs:

```sh
.venv/bin/python -m pip install -e '.[parquet]'
.venv/bin/tensor-balance contacts.parquet --chromosome chr21 --bin-size 25000 --unobserved-are-zero --out results/chr21
```

Segment records need `segA`, `segB`, and `segC` chromosome, start, and end columns, such as `segA_chr`, `segA_start`, and `segA_end`. Coordinates are zero-based, half-open intervals. Full tensor output needs `8 × n³` bytes for `balanced_tensor.npy` alone.

The implementation is in `tensorbalance/`: `input.py` validates inputs, `decompose.py` extracts factors, `solvers/` implements the numerical methods, `parquet_pipeline.py` handles segment records, and `cli.py` runs the command. The Python API is:

```python
from tensorbalance import balance, unit_fiber_result

result = unit_fiber_result(balance(raw, method="dh"))
print(result.reconstruction_scalar, result.bias_vector, result.bias_matrix)
print(result.balanced_tensor)
```

Run the regression suite with `.venv/bin/python -m unittest discover -s tests -v`.

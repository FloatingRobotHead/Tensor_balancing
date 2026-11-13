# Tensor Balance

## Problem

A three-way contact count records how often bins i, j, and k occur together. Those counts mix an overall scale, effects associated with individual bins, effects associated with pairs of bins, and a remaining three-way signal. This program separates those components.

**What goes in:** one file containing either already-binned contact counts or genomic segment records:

| Input file | Required contents |
| --- | --- |
| `.npy` | One NumPy numeric array of shape `(n,n,n)`; create it with `numpy.save("counts.npy", counts)`. |
| `.npz` | A NumPy archive with the count array named `raw`; for Binomial, optionally include a scalar `snapshots`. |
| `.csv` / `.tsv` | Columns `i,j,k,value`, with zero-based bin indices and exactly one row for **every** ordered combination, including zero counts: `n³` rows. |
| `.json` | For example, `{"raw": [[[12]]], "snapshots": 100}`; `raw` is the nested count array, and `snapshots` is optional except when needed by Binomial. |
| `.zip` | One `raw.npy` plus optional `metadata.json`, or one supported tensor JSON/CSV/TSV file. |
| Binned `.parquet` | A complete coordinate table with `i,j,k,value` (or `count`), following the same rules as CSV. |
| Segment-record `.parquet` | One triplet per row, with the nine columns `segA_chr,segA_start,segA_end,segB_chr,segB_start,segB_end,segC_chr,segC_start,segC_end`. |

Already-binned counts must be finite, nonnegative, and symmetric under swapping i, j, and k. All three axes use the same bins. The program preserves their resolution and order. Missing measurements are not silently converted to zeros.

Segment coordinates are zero-based, half-open genomic intervals. Supply a chromosome and bin width in base pairs. Each segment is assigned to `floor(midpoint / bin_size)`; each row contributes one count to its unordered triplet, copied to each distinct permutation. The domain includes bins zero through the last observed midpoint. This route requires explicit agreement that unobserved triplets have zero counts.

**What comes out:** a new directory containing these four binary NumPy files, readable with `numpy.load(path, allow_pickle=False)`:

| Output file | Meaning | Shape |
| --- | --- | --- |
| `reconstruction_scalar.npy` | C: overall count scale | scalar |
| `bias_vector.npy` | b: one multiplicative bias per bin | `(n,)` |
| `bias_matrix.npy` | M: one multiplicative bias per bin pair | `(n,n)` |
| `balanced_tensor.npy` | T: remaining three-way weights after removing those biases | `(n,n,n)` |

T contains dimensionless weights, not raw counts or a joint probability distribution. For each fixed pair of indices, summing over the third gives 1. Segment-record runs also write `bins.csv` (index-to-genomic-interval mapping), `report.json`, `history.json`, `validation.json`, compact factors, and observed-triplet counts. Supplied coordinates are preserved in `metadata.json`. Running all methods writes separate results and a `summary.json`.

## How we do it

Let X be the input counts and Y the counts used by the solver. Every exact zero is replaced once with the fixed value `1e-12`, identical across datasets; positive counts are unchanged. Thus `Y = where(X == 0, 1e-12, X)`. There is no adaptive cap: for Z zeros, the total added mass is `1e-12 × Z`.

The reconstruction equation is:

```text
Y[i,j,k] ≈ C × b[i] × b[j] × b[k] × M[i,j] × M[i,k] × M[j,k] × T[i,j,k]
```

The equivalent **linear equation in log space** is:

```text
log Y[i,j,k] ≈ c + u[i] + u[j] + u[k] + v[i,j] + v[i,k] + v[j,k] + t[i,j,k]
c = log C;  u = log b;  v = log M;  t = log T
```

Thus the model is multiplicative in counts and additive in their logarithms. The geometric mean of b and of each row/column of M is 1, fixing how scale is divided between the factors. Multiplying the four factors reconstructs Y within numerical tolerance; setting originally zero cells back to zero recovers X. Use the original zero mask: a positive measurement can also equal `1e-12`.

Choose **Daryl Huntley (DH)**, **Log ANOVA**, or **Binomial** for already-binned input. DH is the default; its compact segment-record implementation keeps pair factors in memory instead of a full cube during solving. The other two methods use dense arrays. Binomial requires the measured number of snapshots/exposures S, with S at least the largest count; S cannot be inferred from the count file. Segment-record Parquet currently supports compact DH only.

Pseudocode:

```text
read one input file
if segment records:
    assign segment midpoints to bins and count triplets
else:
    load counts without changing bins
validate complete, finite, nonnegative, symmetric counts
choose DH by default, one requested method, or all three
for each selected method, independently:
    form positive counts Y using the smoothing rule
    iterate the chosen solver until its balance checks pass
    separate removed effects into overall scale C, bin vector b, and pair matrix M
    scale the remaining tensor T so every fiber sums to 1
    check convergence, fiber sums, and reconstruction of Y
    write the four NPY files only after all checks pass
    release this method's result before starting the next method
```

For DH, each iteration uses the total G, pair sums P, and single-bin sums Q of the current weights W:

```text
D[i,j,k] = G × P[i,j] × P[i,k] × P[j,k] / (Q[i] × Q[j] × Q[k])
W_next = W / sqrt(D), then rescale W_next to mean 1
```

Log ANOVA removes additive lower-order terms from log Y and repeatedly balances axis sums. Binomial uses fitted binomial-odds directions with a checked step size and a safe internal probability scale; its output is still relative weights. Failed convergence is reported as failure, never exported as a balanced result.

## Code

Python 3.10 or newer:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[parquet]'

# Default: DH.
.venv/bin/tensor-balance counts.npy --out results/dh

# Choose either of the other methods.
.venv/bin/tensor-balance counts.npy --method log-anova --out results/log_anova
.venv/bin/tensor-balance counts.npy --method binomial --snapshots 1000 --out results/binomial

# Run all three independently; replace 1000 with the measured exposure count.
.venv/bin/tensor-balance counts.npy --method all --snapshots 1000 --out results/all

# Start directly from genomic segment records.
.venv/bin/tensor-balance contacts.parquet --chromosome chr21 --bin-size 25000 --unobserved-are-zero --out results/chr21

# Optional: convert the four binary arrays to CSV files.
.venv/bin/tensor-export-csv results/dh --out results/dh_csv
```

All-method output goes into `results/all/dh/`, `results/all/log-anova/`, and `results/all/binomial/`. `summary.json` records success/failure, paths, timing, and diagnostics per method. Numerical failure in one method does not discard another's successful results. Exit codes are 0 for success, 2 for input/resource errors, 3 for numerical failure, and 130 for interruption. Existing output directories are never reused; the input file is never modified.

CSV export produces `reconstruction_scalar.csv`, `bias_vector.csv`, `bias_matrix.csv`, and `balanced_tensor.csv`; the tensor table has columns `i,j,k,value` and all `n³` entries. CSV is optional and can be much larger than NPY.

The full tensor file alone requires approximately `8 × n³` bytes. Dense solvers need several cube-sized arrays in memory. Compact segment-record DH reduces solving memory, but exporting the complete tensor still requires cubic disk space.

Python API:

```python
import numpy as np
from tensorbalance import balance, unit_fiber_result

counts = np.load("counts.npy", allow_pickle=False)
result = unit_fiber_result(balance(counts, method="dh"))
print(result.reconstruction_scalar)
print(result.bias_vector, result.bias_matrix, result.balanced_tensor)
```

Implementation: `tensorbalance/input.py` reads inputs, `decompose.py` extracts factors, `solvers/` contains the three methods, `parquet_pipeline.py` processes segment records, and `cli.py` runs the command. Run checks with `.venv/bin/python -m unittest discover -s tests -v`.

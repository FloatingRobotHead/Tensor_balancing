# Tensor Balance CLI — reviewed release 0.1.1

One **user-supplied raw contact tensor** in; a bias vector, a bias matrix, and a balanced remaining tensor out. The input filename is required. The program never substitutes demonstration data for your input.

This release includes the source, regression tests, a reproducible numerical audit, and readable example results. See `AUDIT_REPORT.md` for fixes, measured errors, and remaining limits.

**Start with `START_HERE.txt` for the complete own-data workflow**, including copyable Windows and Linux/macOS commands. The `my_data/` folder is ready for your tensor files.

## 1. Install

Requires Python 3.10 or later, NumPy >= 1.24, and SciPy >= 1.12. Installing the package installs the dependencies if needed. Normalization and viewing work offline after installation.

Extract this reviewed ZIP into a **new folder**, then open a terminal in the directory containing `pyproject.toml`. This avoids mixing the old installed source with the reviewed code. Existing input and result files can stay where they are.

### Linux / macOS / Unix shell

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
tensor-balance --version
```

The version should be `0.1.1`. In later terminal sessions, activate this environment again with `. .venv/bin/activate`.

### Windows Command Prompt

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install .
.venv\Scripts\tensor-balance.exe --version
```

These Windows commands explicitly use the environment and require no activation. If your installation provides `py` instead of `python`, use `py -m venv .venv` for the first command.

For developers installing from source, `python -m pip install -e .` is also supported. Source-folder entry points are `python tensor_balance.py ...` and `python export_csv.py ...`. Use the Python interpreter from the environment with the dependencies installed.

## 2. What the earlier demonstration used

The command

```text
python tensor_balance.py examples/patchwork.npy --out patchwork_test
```

explicitly supplied `examples/patchwork.npy` as its input. You did not have to upload data because the ZIP already contained this file.

It is a **simulated patchwork chain**, not chromosome 21 measurements:

- 123 monomers, binned into a **15 x 15 x 15** raw contact tensor: 3,375 entries.
- 100,000 pooled snapshots; provenance records four replicate seeds.
- Three chain sections with differing mobility, bending penalties, and backbone stiffness.
- ABC permits pair interactions; DEF permits pair and triple interactions.
- The saved counts range from 4,791 to 99,878. The array contains no zero entries.
- The NPZ example contains the identical raw tensor plus `snapshots=100000`.
- Recorded source: `All_Simulators_One_File.zip`, member `Simulator_Handoff/reference_tensors/six_families/patchwork/raw.npz`.

`examples/PROVENANCE.json` records these details. The raw float64 SHA-256 was checked against that record. This audit verified the packaged data and its provenance record; it did not rerun the original chain simulation.

`examples/checked_result/` contains reviewed DH output arrays and a visualization; `examples/checked_result_csv/` contains readable copies. These folders are reference results, so choose a different destination for a new run.

## 3. Prepare your own input

The input must be a **complete, finite, nonnegative, permutation-symmetric n x n x n cube**. All three axes refer to the same ordered set of bins. Swapping any axes must preserve values to relative tolerance `1e-12`.

The tool expects an already binned tensor. It does not read FASTQ, BAM, or Parquet, choose a genome assembly/bin size, or convert arbitrary sequencing-read records into contacts. Keep your bin-to-genomic-coordinate lookup alongside your input; output index order is preserved.

Every voxel must have a supplied value, including measured zeros and repeated-index cells. Missing or excluded cells are not measured zeros. Masks and structural-zero constraints are unsupported. A sparse list containing only observed triplets, or only `i <= j <= k`, is not a complete coordinate-table input. Prepare the complete cube in the upstream data pipeline using the correct counting and symmetry convention for your assay.

### Recommended: NPY or NPZ

If `raw` is your existing NumPy cube, save it as:

```python
import numpy as np

# raw is YOUR complete measured cube, shaped (n, n, n).
np.save("my_tensor.npy", raw, allow_pickle=False)
```

For Binomial, also save the **actual number of eligible snapshots/exposures**:

```python
# actual_snapshots must come from acquisition/simulation metadata.
np.savez_compressed("my_tensor.npz", raw=raw, snapshots=actual_snapshots)
```

NPZ selects `raw` preferentially, then `tensor` or `counts`, or an otherwise unique 3D array. An ambiguous archive is rejected. Do not supply a previously balanced result as if it were raw counts.

### Complete coordinate CSV / TSV

Required columns are `i,j,k,value`, with integer coordinates starting at zero. `count`, `counts`, or `raw` may replace `value`. All n³ coordinates must occur exactly once; row order does not matter.

Here is an entire **toy 2 x 2 x 2 tensor**, shown only to illustrate the format:

```csv
i,j,k,value
0,0,0,10
0,0,1,4
0,1,0,4
0,1,1,6
1,0,0,4
1,0,1,6
1,1,0,6
1,1,1,12
```

CSV cannot carry the snapshot count; supply `--snapshots` when using Binomial. Duplicate headers, coordinates, conflicting value columns, malformed row lengths, and incomplete cubes are rejected.

### JSON

Use a nested cube, or an object such as:

```json
{"shape": [2, 2, 2], "raw": [10, 4, 4, 6, 4, 6, 6, 12], "snapshots": 100}
```

A flat array uses C order: k changes fastest, then j, then i. Shape dimensions must be positive integers. `tensor`, `values`, or `data` can replace `raw`. JSON snapshots are optional except for a Binomial run without a command-line exposure.

### Other supported inputs

ZIP inputs can contain a unique `raw.npy`, a supported tensor JSON/table, or the established saved-session format (`session.json` plus `arrays.bin`, whose first n³ little-endian float64 entries are raw counts). ZIP members are read in memory; files are not extracted and no archive code is executed. Unsupported, ambiguous, masked, or malformed inputs fail explicitly.

Object/pickle arrays, complex numbers, booleans, NaN, infinity, and negative counts are rejected. Integer counts above 2^53 are rejected before conversion where their integer representation is available, including coordinate tables and JSON. Floating-point source files already have their own finite precision; lost precision cannot be recovered by the loader.

## 4. Normalize your data

Replace `my_tensor.npy` below with your actual input path. Quote paths containing spaces. Choose a **new output directory** for each run; existing directories are never overwritten. The input file is not modified.

### Linux / macOS, with the environment activated

```sh
tensor-balance "my_tensor.npy" --out "results/my_sample_dh"
```

### Windows Command Prompt

```bat
.venv\Scripts\tensor-balance.exe "my_tensor.npy" --out "results\my_sample_dh"
```

For an absolute input path on Windows, for example:

```bat
.venv\Scripts\tensor-balance.exe "C:\data\my_tensor.npy" --out "C:\data\my_sample_dh"
```

If `--out` is omitted, a new directory named `INPUT_STEM_balanced` is created beside the input file.

### Method selection

One invocation runs one method. DH is the default.

```sh
tensor-balance my_tensor.npy --method dh --out results/my_dh
tensor-balance my_tensor.npy --method log-anova --out results/my_loganova
tensor-balance my_tensor.npz --method binomial --out results/my_binomial
```

On Windows, replace `tensor-balance` at the start of each line with `.venv\Scripts\tensor-balance.exe`.

If the input lacks exposure metadata, explicitly supply the actual S. In this example, **100000 must be replaced if it is not your real exposure count**:

```sh
tensor-balance my_tensor.npy --method binomial --snapshots 100000 --out results/my_binomial
```

`--snapshots` overrides exposure metadata in the input. S must be positive and at least the largest raw entry, but the largest raw entry is **not an estimate of S**. DH and Log ANOVA do not require exposure metadata.

### Tolerance and iteration limits

Defaults: `--tolerance 1e-10 --max-iterations 5000`. The tolerance measures relative spread among fiber/plane sums. To preserve accurate reconstruction by a shared vector and pair matrix, the internal solver tolerance is `min(requested_tolerance, 1e-10)`. Asking for a looser tolerance does not stop factor extraction prematurely.

A difficult Binomial audit case required 7,017 iterations. If a run reaches its limit, it exits without writing a result. To allow more work:

```sh
tensor-balance my_tensor.npz --method binomial --max-iterations 10000 --out results/my_binomial_long
```

This extends the selected method; the program does not silently switch algorithms. A larger limit does not guarantee convergence for every possible input.

## 5. Collect and interpret the outputs

A successful default run writes exactly these three float64 NumPy files:

| File | Shape | Meaning |
| --- | --- | --- |
| `bias_vector.npy` | `(n,)` | Shared linear divisor factors, including the global scale. |
| `bias_matrix.npy` | `(n,n)` | Shared pairwise divisor factors, with row/column geometric means of 1. |
| `balanced_tensor.npy` | `(n,n,n)` | Remaining relative weights; arithmetic mean 1 and balanced fiber/plane sums. |

These are data files. Garbled symbols when opening an NPY file in a text editor are expected. Preserve the NPY files for computation, and use CSV export or the HTML view for inspection.

Read the output with Python:

```python
from pathlib import Path
import numpy as np

out = Path("results/my_sample_dh")
b = np.load(out / "bias_vector.npy", allow_pickle=False)
M = np.load(out / "bias_matrix.npy", allow_pickle=False)
R = np.load(out / "balanced_tensor.npy", allow_pickle=False)
print(b.shape, M.shape, R.shape)
```

The decomposition includes a fixed pseudocount of **0.5 added once to every raw entry**:

```text
raw[i,j,k] + 0.5 ≈ b[i] b[j] b[k] M[i,j] M[i,k] M[j,k] R[i,j,k]
```

The factors are divisors; their reciprocals are multiplicative correction weights. To reconstruct original counts, multiply the six bias factors by R and subtract 0.5. R is a relative-weight tensor, not a probability tensor. Its fiber sums target n and its plane sums target n². A residual peak alone does not establish a direct physical three-way interaction.

## 6. Export readable CSV — the entire tensor in one file

After normalization, run:

```sh
tensor-export-csv results/my_sample_dh --out results/my_sample_csv
```

Windows:

```bat
.venv\Scripts\tensor-export-csv.exe results\my_sample_dh --out results\my_sample_csv
```

This reads the saved arrays without rerunning normalization and creates:

- `bias_vector.csv`: index `i` and value.
- `bias_matrix.csv`: a rectangular table, with row indices in the first column and column indices in its header.
- `balanced_tensor.csv`: **all n³ values in one file**, with columns `i,j,k,value`. No sampling, slicing, or omitted zeros.

Float64 values are exported with 17 significant digits for numerical round trips. Open the CSV files in Excel or another table viewer. Large tensors can exceed spreadsheet row limits; keep their NPY files and inspect them in Python. The export folder must also be new.

## 7. Optional visualization

Request the view when running normalization:

```sh
tensor-balance my_tensor.npy --out results/my_sample_view --visualize
```

Windows:

```bat
.venv\Scripts\tensor-balance.exe my_tensor.npy --out results\my_sample_view --visualize
start "" "results\my_sample_view\visualization.html"
```

On macOS, use `open results/my_sample_view/visualization.html`; on Linux, use `xdg-open results/my_sample_view/visualization.html`, or open the file in your browser.

The optional fourth file is `visualization.html`: raw and remaining dot tensors, a vector graph, a matrix heatmap, and a tensor-slice slider. Drag a tensor to rotate it. The view embeds the actual input and output arrays, works offline, and sends no data anywhere.

Display conventions:

- Raw dots show positive counts on a logarithmic intensity scale.
- The remaining 3D view shows entries above 1; its opacity scale is stated beside the plot. It magnifies display contrast without changing numbers.
- The slice view includes entries below and above 1. Heatmap colors saturate at the labeled endpoints, 0.25 and 4.
- Large point clouds are sampled for display, as labeled. CSV and NPY outputs contain the complete tensor.

The viewer embeds full dense arrays and is intended for modest-sized tensors. For large datasets, omit `--visualize` and use the numerical outputs.

## 8. Logs, errors, and repeated runs

Progress and numerical diagnostics go to stderr. Stdout lists output paths, one per line. In a Unix shell, for example:

```sh
mkdir -p logs
tensor-balance my_tensor.npy --out results/my_run > logs/my_run_paths.txt 2> logs/my_run.log
```

Exit codes:

| Code | Meaning |
| --- | --- |
| 0 | Convergence, positivity, mean-one, marginal, and reconstruction checks passed; files were saved. |
| 2 | Input, argument, dependency, resource, or output-path error. Read the message. |
| 3 | Solver or final numerical validation failed; no normalized result was written. |
| 130 | Interrupted by the user. |

Use distinct output directories for different inputs, methods, or settings. An existing directory error is intentional protection against overwriting earlier results. Invalid/incomplete data is rejected rather than repaired by guessing.

This is a dense in-memory tool. One float64 n³ array uses `8*n**3` bytes (8 MB at n=100; 1 GB at n=500), and several arrays are needed during a run. Total memory usage can be substantially larger, especially for Binomial fitting and HTML export. An operating-system process kill cannot be turned into a Python error message.

## 9. Verification and Python API

The release was checked before packaging; you do not need to repeat the full audit to use it. To reproduce checks from the extracted source folder:

```sh
python -m unittest discover -s tests -v
python audit_numerics.py --out my_numerical_audit.json
```

Use `.venv\Scripts\python.exe` instead of `python` on Windows without activation. The audit report path must be new. `UNIT_TEST_LOG.txt`, `NUMERICAL_AUDIT.json`, `VALIDATION.json`, and `AUDIT_REPORT.md` contain the delivered evidence and its limits. Historical 0.1.0 validation is kept separately under `audit/`; those historical real-data comparisons were not rerun in this audit.

In your own Python code:

```python
from tensorbalance import balance
result = balance(raw, method="dh")
b = result.bias_vector
M = result.bias_matrix
R = result.balanced_tensor
print(result.diagnostics)
```

Binomial: `balance(raw, method="binomial", snapshots=actual_snapshots)`. The API validates the same tensor contract and raises on failure. The input array is not modified.

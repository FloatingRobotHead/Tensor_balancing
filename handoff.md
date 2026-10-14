# Development handoff

Updated: 2026-10-14.

## Start here in a new chat

Give the next developer or coding assistant this prompt:

> Continue developing https://github.com/FloatingRobotHead/Tensor_balancing.
> Clone the repository, read handoff.md, README.md, and AUDIT_REPORT.md, and inspect
> the current branch and working tree before editing. The main product is a Unix
> command line tool: one raw tensor in, a bias vector, bias matrix, and remaining
> tensor out. Visualization is optional. Run the existing regression tests before
> changing numerical behavior. Explain any changes to the input/output contract
> or mathematics. Do not assume access to files, credentials, or previous chats
> from the original development machine. Ask me for the next feature if I have
> not specified one.

GitHub is the source of truth. This document is sufficient to begin a fresh session;
the previous ChatGPT workspace and its download links are not required.

## Project goal and current state

Tensor Balance CLI splits a symmetric raw contact tensor into three factors:

1. A shared bias vector.
2. A shared bias matrix.
3. The balanced tensor that remains.

The user wants the core software usable from the command line by researchers.
Keep visualization separate from the required computation.

- Package: `tensor-balance-cli`, version `0.1.1`.
- Repository: https://github.com/FloatingRobotHead/Tensor_balancing
- Default branch: `main`.
- Initial complete source upload: commit
  `2d91d269effc53b5d2cba848409052723ef6b226`.
- All 48 files in that upload were verified against the prepared local source.
- All 36 unit/regression tests passed locally before the upload.
- GitHub Actions configuration is included. A successful hosted CI run was not
  verified during the initial upload; check the Actions tab.
- The numerical audit included in the release reports 84 numerical runs across
  28 tensors. That full audit was not rerun solely for the GitHub upload.
- The repository was created public by the user. Avoid committing credentials
  or private research data.
- No software license has been selected or included. Ask the owner before adding
  one; public visibility alone does not establish a reuse license.

## Set up on a remote Linux/macOS machine

Requires Git and Python 3.10 or newer. Installation downloads NumPy, SciPy, and
build dependencies; calculations work offline afterward.

```sh
git clone https://github.com/FloatingRobotHead/Tensor_balancing.git
cd Tensor_balancing
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
tensor-balance --version
python -m unittest discover -s tests -v
```

Expected version: `0.1.1`. The handoff baseline has 36 passing tests.

Smoke test using the bundled raw input:

```sh
tensor-balance examples/patchwork.npy --out results/smoke_dh
tensor-export-csv results/smoke_dh --out results/smoke_csv
```

A successful normalization prints the paths of three NPY files to stdout.
Progress and diagnostics go to stderr. Every output directory must be new;
use another name for a repeated run.

Check the other methods when changing numerical code:

```sh
tensor-balance examples/patchwork.npy --method log-anova --out results/smoke_loganova
tensor-balance examples/patchwork.npz --method binomial --out results/smoke_binomial
```

For optional viewing:

```sh
tensor-balance examples/patchwork.npy --out results/smoke_view --visualize
```

Download `results/smoke_view/visualization.html` from the remote machine and open
it in a browser. A headless remote machine does not need a graphical browser to
compute the outputs or create the HTML.

For Windows commands, see `START_HERE.txt` and `README.md`.

## Mathematical and file contracts

Input is a complete, finite, nonnegative, permutation-symmetric `n × n × n` cube.
All axes refer to the same ordered bins. Relative symmetry tolerance is
`1e-12`. Missing entries are not automatically interpreted as measured zeros.
Masks and structural-zero constraints are unsupported.

Supported input families include NPY, NPZ, JSON, complete coordinate CSV/TSV,
and supported tensor ZIP/saved-session archives. See README for exact schemas.
The program requires an explicit input filename and never replaces user input
with example data.

Default outputs, all float64:

| File | Shape |
| --- | --- |
| `bias_vector.npy` | `(n,)` |
| `bias_matrix.npy` | `(n,n)` |
| `balanced_tensor.npy` | `(n,n,n)` |

The decomposition is:

```text
raw[i,j,k] + 0.5 ≈
b[i] b[j] b[k] M[i,j] M[i,k] M[j,k] R[i,j,k]
```

- Add the fixed pseudocount `0.5` once.
- Bias factors are divisors; reciprocals are multiplicative correction weights.
- R has arithmetic mean 1, fiber sums targeting n, and plane sums targeting n².
- M has row/column geometric means of 1; b includes the global scale.
- To reconstruct original raw counts, multiply the factors and subtract 0.5.
- Index ordering is preserved.
- R contains relative weights, not probabilities. A residual peak alone is not
  proof of a physical three-way interaction.
- Existing destinations are protected; input files are not modified.
- Failed convergence must not publish a partial tensor as a balanced result.

Available methods: `dh` (default), `log-anova`, and `binomial`.
Binomial requires the actual number of eligible snapshots/exposures S, provided
in metadata or through `--snapshots`. Do not infer S from the largest count.

Defaults: `--tolerance 1e-10 --max-iterations 5000`. The solver uses at most
`1e-10` internal tolerance to retain accurate shared-factor reconstruction.
There is no silent fallback to another method.

Exit codes: 0 success, 2 input/resource/output errors, 3 numerical failure,
130 interruption.

## Code map

| Path | Responsibility |
| --- | --- |
| `pyproject.toml` | Installation, dependencies, package data, console entry points |
| `tensorbalance/cli.py` | Arguments, normalization command, diagnostics and exit codes |
| `tensorbalance/input.py` | Input formats and archive handling |
| `tensorbalance/checks.py` | Shared scalar/integer validation |
| `tensorbalance/decompose.py` | Public balance API, validation and bias factor extraction |
| `tensorbalance/solvers/dh_sqrt.py` | DH solver |
| `tensorbalance/solvers/balanced_loganova.py` | Log ANOVA solver |
| `tensorbalance/solvers/repaired_binomial.py` | Active Binomial solver |
| `tensorbalance/solvers/binomial_fit.py` | Binomial fitting support |
| `tensorbalance/storage.py` | Output reservation, publication and cleanup |
| `tensorbalance/export_csv.py` | Export saved outputs without rebalancing |
| `tensorbalance/visualize.py`, `view.html` | Optional self-contained viewer |
| `tests/test_cli.py`, `tests/test_audit.py` | Regression suite |
| `audit_numerics.py` | Reproducible numerical audit |
| `.github/workflows/tests.yml` | Install package and run tests on Python 3.11 |

The convenience scripts `tensor_balance.py` and `export_csv.py` call the same
package entry points. Python API: `from tensorbalance import balance`.

## Data, provenance and evidence

`examples/patchwork.npy` is a simulated patchwork-chain raw tensor, not
chromosome 21 measurements. It has shape `15 × 15 × 15`.
`examples/patchwork.npz` contains the same tensor with `snapshots=100000`.
See `examples/PROVENANCE.json`; the upstream simulator is not bundled here.

Reviewed reference outputs are in `examples/checked_result/` and
`examples/checked_result_csv/`. Keep these as fixtures rather than overwriting
them with new runs.

Read `AUDIT_REPORT.md`, `NUMERICAL_AUDIT.json`, `SOLVER_PROVENANCE.json`,
and `VALIDATION.json` for measured behavior and limits. Files under `audit/`
include historical evidence; do not describe historical checks as fresh runs.

`MANIFEST_SHA256.json` is the reviewed release's file-integrity manifest.
It is not a live inventory of new repository files such as this handoff.
If covered files change, update the manifest deliberately and document the
new evidence instead of retaining stale claims of release equivalence.

## Known limits and development priorities

- Dense memory use grows as n³; several arrays are needed. Binomial fitting and
  HTML export can require substantially more memory than the input alone.
- No FASTQ/BAM/Parquet ingestion, genomic binning, or sparse/masked balancing.
- One difficult audit case required 7,017 Binomial iterations, exceeding the
  default limit. A higher limit can be requested but does not guarantee success.
- Native Windows/macOS execution and minimum dependency versions were not
  verified in the included audit.
- Biological interpretation and the original chain simulations were not
  independently validated by the numerical checks.
- The earlier Cloudflare website is a separate deployment. Its complete source,
  chromosome 21 dataset, and original paper/ZIP materials are not in this repo.
  Obtain those materials from the user before changing that deployment or
  claiming reproducibility against them.

For future UI work, the user's preferences are simple, clean screens, few controls,
rotatable 3D tensors expressed as dots, and 2D IJ/XY slices selected by a K/Z slider.
Keep matrix/slice orientation consistent. Histograms were explicitly removed from
the earlier website design. Do not expand UI scope without a requested feature.

## Continuing work safely

1. Fetch current GitHub state and inspect local changes; this file records a
   baseline, not a promise that main will remain unchanged.
2. Use a feature branch, for example `git switch -c feature/short-name`.
3. Preserve the numerical and three-output contracts unless the user explicitly
   requests a change.
4. Run the existing suite for code changes. Numerical solver changes also need
   reconstruction, marginal-balance, known-factor and reference comparisons.
5. To reproduce the broader audit, choose a new report path:
   `python audit_numerics.py --out new_numerical_audit.json`.
6. Report actual checks, environment and remaining limits. Update this handoff
   when behavior, setup, provenance or outstanding work changes.
7. No next feature has been agreed yet; ask the user what to develop next.

Do not rely on credentials stored on the original machine. Public cloning needs
no authentication; pushing requires the developer's own authorized GitHub access.

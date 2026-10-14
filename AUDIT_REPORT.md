# Tensor Balance CLI 0.1.1 — audit report

## Result

The reviewed release passes **36 unit/regression tests** and **84 numerical runs across 28 input tensors**. Of those numerical runs, 83 converged within the default 5,000-iteration limit. One deliberately difficult Binomial run required 7,017 iterations and passed with a 10,000-iteration limit.

The original package's nine tests passed before this audit. The expanded checks nevertheless found reproducible input-validation and error-handling defects, which are fixed here. This report describes tested behavior; it is not a proof that every possible input or environment is bug-free.

## What the user ran

`examples/patchwork.npy` was explicitly named in the command. It is the bundled raw simulation, not newly generated or downloaded data and not chromosome 21.

Verified packaged array: 15 x 15 x 15, float64, 3,375 entries, minimum 4,791, maximum 99,878, no zero entries. `patchwork.npz` contains exactly the same array and `snapshots=100000`. Its provenance describes a 123-monomer patchwork chain, four pooled replicate seeds, ABC pair interactions, and DEF pair plus triple interactions.

The SHA-256 of the raw float64 bytes is:

```text
0c665550c9341b31dfa31a6cd5ae602bc9e2d589734b651de7f3ef3c65b8af44
```

This matches `examples/PROVENANCE.json`. All 21 entries in the original package's checksum manifest also matched before editing. The upstream simulation itself was not rerun.

## Confirmed defects and fixes

| Area | Earlier behavior | Reviewed behavior |
| --- | --- | --- |
| CSV duplicate headers | Two `value` headers silently selected the last value. | Duplicate headers rejected. |
| CSV malformed rows | Extra row fields could be ignored, losing information. | Header/row widths must agree; competing value columns rejected. |
| Integer precision | CSV `9007199254740993` became `9007199254740992.0`. Mixed JSON integer/float arrays could similarly round before validation. | Exact integer range checked before conversion; oversized integer exposures rejected too. |
| JSON shape metadata | A fractional `n`, such as 1.9, was truncated to 1. | Positive integer dimensions and consistent `n`/shape required. Duplicate JSON keys rejected. |
| Exposure metadata | A complex snapshot value could lose its imaginary part and be accepted. | Requires a finite positive real numeric scalar; booleans, strings, complex values, and nonscalar values rejected. |
| Missing-data masks | A NumPy masked array could lose its mask; masks inside ZIP metadata were not checked. | Both paths explicitly reject masks. |
| Archive metadata | `metadata.json` containing a list caused an AttributeError traceback. | Metadata type validated; clean input error. Duplicate archive members and disguised NPZ-as-NPY files rejected. |
| Symmetry threshold | Public validation accepted differences below 1e-12, but Binomial separately rejected some above 1e-13. | One shared 1e-12 threshold. Values are not silently symmetrized. |
| Loose solver tolerance | Log ANOVA at tolerance 0.1 on the sample stopped too soon to reconstruct with shared factors. | Internal tolerance is at most 1e-10, even if a looser threshold is requested. Default calculations unchanged. |
| Nested numerical failures | A RuntimeError from Newton-CG could escape as a traceback. | Wrapped as numerical failure, exit 3, without publishing a result. Fault-injection test included. |
| Invalid solver return | Finite, strictly positive, correctly shaped, mean-one output was partly assumed. | Explicit checks before factor extraction and export. |
| Output cleanup | Failure cleanup removed every file in the reserved destination, potentially including a different writer's file. | Cleans up only files published by this run. Concurrent-run and injected-write-failure tests included. |
| Directory input paths | Default destination construction could raise outside the error handler for `.` or a filesystem root. | Clean input error, no traceback. |
| Text encoding and viewing | HTML template loading relied on platform default encoding; very small enrichments rendered almost invisibly. | Explicit UTF-8; labeled opacity contrast for the remaining tensor. Single-point vectors render visibly; heatmap clipping endpoints are labeled. |

The solver update equations were retained. Changes to stopping/validation behavior are stated above and in `SOLVER_PROVENANCE.json`. There is no automatic fallback to another normalizer.

## Added result export

`tensor-export-csv RESULTS --out CSV_RESULTS` exports the existing vector and matrix plus **every tensor entry in one i,j,k,value CSV**. It does not recalculate normalization. Float64 round trips were tested exactly, and existing export directories are protected from overwriting.

This replaces the long one-line Python conversion commands previously supplied in chat. Normalization still writes exactly three NPY files by default; `--visualize` adds the optional HTML file.

## Numerical evidence

The deterministic battery includes zero/uniform cubes, singleton cubes, symmetric Poisson data, sparse observed counts, strong corner spikes, diagonal enrichment, lognormal variation at three strengths, and the packaged patchwork example. Sizes are 1, 2, 3, 4, 5, 8, and 15 bins per axis.

All three methods were checked for positivity, finite values, input preservation, mean one, fiber/plane targets, output symmetry, centered pair factors, and reconstruction of **raw + 0.5**. Results were compared with an independently implemented direct arithmetic iterative proportional fitting reference, starting from raw + 0.5. This reference is distinct code and shares the same mathematical target; agreement does not constitute an independent biological validation.

Worst observed errors across the 84 successful runs:

| Check | Maximum error |
| --- | ---: |
| Fiber sum relative to target n | 6.63 x 10^-11 |
| Plane sum relative to target n² | 1.66 x 10^-11 |
| Residual relative to the independent reference | 2.78 x 10^-9 |
| Reconstructed smoothed input, voxelwise relative error | 2.76 x 10^-9 |
| Vector/matrix/residual relative disagreement with DH | 2.79 x 10^-9 |

Unit tests additionally check known factor recovery, pair-only removal, triple-only preservation for constructed balanced controls, bin permutation, scaling of smoothed input, and preservation of three-way log contrasts. These controls concern the stated mathematical decomposition.

On the patchwork input, the three arrays for each default method were **exactly equal** to the original unmodified 0.1.0 arrays in the audit environment. See `audit/DEFAULT_COMPATIBILITY.json`. This comparison does not claim bit-for-bit identity between different operating systems or library versions.

### Iteration-limit case

The seeded `lognormal_10` 4 x 4 x 4 test did not reach tolerance with Binomial after 5,000 iterations. It was not reported as converged and no result was written. It converged in 7,017 iterations when allowed 10,000. DH and Log ANOVA converged on that case within their defaults. See `NUMERICAL_AUDIT.json` for all iteration counts and measured errors.

## Interface and packaging checks

- All documented input families: NPY, named/unnamed NPZ, nested/flat JSON, complete CSV/TSV with reordered rows, raw ZIP, and a fixture in the established saved-session format.
- Invalid values, shapes, masks, table rows, metadata, duplicate members, corrupted archives, precision limits, and insufficient exposure.
- Failure to converge, protected destinations, spaces and Unicode in paths, resource-error handling, interrupts, simultaneous jobs, and cleanup after simulated disk/write failures.
- Exact three-file interface; optional view's embedded arrays match the saved arrays.
- Wheel built and installed into a separate environment; both console commands run from outside the source folder. Packaged HTML and full CSV export worked.
- Viewer script rendered five nonblank plots using native canvases, and rotation/slice changes worked. The visible remaining-tensor plot was inspected after the contrast adjustment. See `audit/VIEW_CHECK.json`.

## Test environment and limits

Tests ran on **Linux, Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0**. Native Windows/macOS execution and minimum dependency versions were not tested here. Windows instructions use the same installed Python entry points, and the original 0.1.0 install/run was already demonstrated by the user; the reviewed Windows release still needs native-platform verification for a cross-platform certification claim.

A full browser layout test was unavailable. The actual viewer JavaScript and native canvas rendering were tested with a simulated DOM, including interaction handlers and exact data payloads.

The earlier chromosome 21/website comparisons are preserved in `audit/ORIGINAL_VALIDATION.json` as historical evidence. This audit did not retrieve and rerun those datasets. It does not resolve whether a particular real-data resolution has sufficient coverage or whether peaks represent direct biological triples.

The tool intentionally supports dense, complete, symmetric cubes. It does not support arbitrary missing-data masks, structural zeros, sparse-only inputs, or unlimited memory. Correctly rejecting unsupported input is expected behavior. Floating-point calculations and finite iteration budgets impose practical limits.

## Reproduce

From the installed source environment:

```sh
python -m unittest discover -s tests -v
python audit_numerics.py --out new_audit_report.json
```

`UNIT_TEST_LOG.txt`, `NUMERICAL_AUDIT.json`, `VALIDATION.json`, and the `audit/` directory contain the release evidence. `MANIFEST_SHA256.json` covers the delivered files, excluding the manifest itself.

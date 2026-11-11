# Audit evidence map

- `v0.2.0/` contains current numerical validation, test logs, and the package check.
- `v0.1.1/` preserves the former root audit report, numerical results, unit-test
  log, validation summary, and solver provenance without changing their content.
- `ORIGINAL_*` records the historical 0.1.0 baseline.
- `DEFAULT_COMPATIBILITY.json`, `VIEW_CHECK.json`, and `check_view.cjs` belong to
  the historical 0.1.1 validation workflow. They are not current validation claims.

Older reports retain their original filenames and references as provenance.
Historical root-file references resolve inside `v0.1.1/`; references to examples
and source modules refer to the repository root at the time of that audit.
The example fixtures remain in `examples/` because tests and reproducibility
checks use them. No historical audit is a second implementation to develop.

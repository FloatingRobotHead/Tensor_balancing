# Numerical audit index

Current software is Tensor Balance CLI 0.2.0. Its machine-readable output is
a tensor with fiber sums one, a centered bias matrix, a centered bias vector,
and an explicitly named reconstruction scalar.

Current release evidence is in [audit/v0.2.0](audit/v0.2.0/). The release passed
46 regression tests, 84 numerical audit runs, and an installed-package smoke
check. See [RELEASE_NOTES.md](RELEASE_NOTES.md) for scope and limitations.

The historical 0.1.1 audit, including its 36-test count and mean-one output
convention, is preserved in [audit/v0.1.1/AUDIT_REPORT.md](audit/v0.1.1/AUDIT_REPORT.md).
That report does not describe the current output convention.

See [audit/README.md](audit/README.md) for the historical evidence map.

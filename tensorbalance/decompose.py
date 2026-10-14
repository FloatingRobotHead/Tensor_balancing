"""The three-output wrapper; the existing solver equations are unchanged."""
from dataclasses import dataclass
import numpy as np
from .solvers import dh_sqrt
from .checks import MAX_EXACT_INTEGER, SYMMETRY_RTOL, check_python_integers, positive_integer, scalar

PSEUDOCOUNT = 0.5


@dataclass
class Result:
    bias_vector: np.ndarray
    bias_matrix: np.ndarray
    balanced_tensor: np.ndarray
    diagnostics: dict
    history: list


class ConvergenceError(RuntimeError):
    pass


def validate(raw):
    if np.ma.isMaskedArray(raw):
        raise ValueError("Masked arrays are unsupported. Supply a full measured cube, including zeros.")
    check_python_integers(raw)
    x = np.asarray(raw)
    if x.dtype.kind not in "uif":
        raise ValueError("Raw values must be real numbers; object, complex, and boolean arrays are unsupported.")
    if x.ndim != 3 or not x.size or len(set(x.shape)) != 1:
        raise ValueError("Input must be a nonempty n × n × n tensor.")
    if x.dtype.kind in "ui" and int(np.max(x)) > MAX_EXACT_INTEGER:
        raise ValueError("Integer counts above 2^53 cannot be represented exactly in float64.")
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("Input counts must be finite and nonnegative.")
    # One shared vector and one shared matrix require interchangeable axes.
    for axes in [(1, 0, 2), (0, 2, 1)]:
        if not np.allclose(x, x.transpose(axes), rtol=SYMMETRY_RTOL, atol=0):
            raise ValueError("The three-output interface requires a permutation-symmetric tensor. "
                             "An asymmetric tensor needs separate factors for each axis and pair; "
                             "this tool does not silently symmetrize data.")
    return x


def extract_factors(raw, residual):
    """Canonical log-ANOVA decomposition of the removed multiplicative bias.

    Absorb the scalar scale into the shared vector; pair row/column geometric
    means are one. This is a factor extraction, not a second normalization.
    """
    z = np.log(raw + PSEUDOCOUNT) - np.log(residual)
    c = float(z.mean())
    v = sum(z.mean(tuple(a for a in range(3) if a != axis)) for axis in range(3)) / 3 - c
    p = sum(z.mean(axis) for axis in range(3)) / 3 - c - v[:, None] - v[None, :]
    p = (p + p.T) / 2
    # Remove roundoff in the centering constraints without losing its scale.
    row = p.mean(axis=1)
    total = float(p.mean())
    p = p - row[:, None] - row[None, :] + total
    v = v + 2 * (row - total)
    c += 3 * total
    log_b = v + c / 3
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        b, matrix = np.exp(log_b), np.exp(p)
    if (b <= 0).any() or (matrix <= 0).any():
        raise FloatingPointError("Bias factors underflowed; rescale input counts before running.")
    # Check the actual three arrays that will be exported, in slabs to avoid
    # allocating another full reconstruction tensor.
    lb, lm = np.log(b), np.log(matrix)
    error = 0.0
    for i in range(len(raw)):
        rebuilt_log = (lb[i] + lb[:, None] + lb[None, :] + lm[i, :, None]
                       + lm[i, None, :] + lm + np.log(residual[i]))
        error = max(error, float(np.max(np.abs(np.expm1(rebuilt_log - np.log(raw[i] + PSEUDOCOUNT))))))
    if not np.isfinite(error) or error > 1e-8:
        raise FloatingPointError(f"Three-factor reconstruction failed (relative error {error:.3g}).")
    return b, matrix, error


def balance(raw, *, method="dh", snapshots=None, tolerance=1e-10, max_iterations=5000):
    """Return the bias vector, bias matrix and mean-one balanced tensor.

    raw + 0.5 ≈ b[i] b[j] b[k] M[i,j] M[i,k] M[j,k] R[i,j,k].
    Raises on failed convergence; never relabels a partial result as balanced.
    """
    x = validate(raw)
    tolerance = scalar(tolerance, "Tolerance")
    max_iterations = positive_integer(max_iterations, "Maximum iterations")
    # Shared factors need a more accurate solution than a loose user tolerance
    # may request, particularly after sequential (axis-asymmetric) updates.
    solver_tolerance = min(tolerance, 1e-10)
    options = dict(tolerance=solver_tolerance, max_rounds=max_iterations)
    if snapshots is not None:
        snapshots = scalar(snapshots, "Snapshot count")
    if method == "binomial":
        if snapshots is None:
            raise ValueError("Binomial requires the actual snapshot count: store snapshots in the input "
                             "NPZ/JSON/ZIP, or supply --snapshots S. It cannot be inferred from raw counts.")
        if snapshots < x.max():
            raise ValueError("Snapshot count must be at least the largest raw count.")
    if method not in ("dh", "log-anova", "binomial"):
        raise ValueError("Method must be dh, log-anova, or binomial.")
    try:
        if method == "dh":
            w, info, history = dh_sqrt.normalize(x, **options)
        elif method == "log-anova":
            from .solvers import balanced_loganova
            w, info, history = balanced_loganova.normalize(x, **options)
        else:
            from .solvers import repaired_binomial
            w, info, history = repaired_binomial.normalize(x, snapshots, **options)
    except (RuntimeError, FloatingPointError) as exc:
        raise ConvergenceError(f"{method} numerical solver failed: {exc}. No output files were written.") from exc
    if info["status"] != "converged":
        raise ConvergenceError(f"{method} did not converge: {info['status']} after {info['rounds']} iterations. "
                               "No output files were written.")
    if w.shape != x.shape or not np.isfinite(w).all() or (w <= 0).any():
        raise ConvergenceError("Solver returned a nonpositive, nonfinite, or wrong-shaped tensor. No output files were written.")
    if abs(float(w.mean()) - 1) > 1e-12:
        raise ConvergenceError("Final exported tensor failed the mean-one check.")
    try:
        b, matrix, error = extract_factors(x, w)
    except FloatingPointError as exc:
        raise ConvergenceError(f"Factor extraction failed: {exc}. No output files were written.") from exc
    scores = dh_sqrt.deviation(w)
    if not all(np.isfinite(v) for v in scores.values()) or max(scores.values()) >= tolerance:
        raise ConvergenceError("Final exported tensor failed the fiber/plane balance check.")
    return Result(b, matrix, w, dict(method=method, shape=list(x.shape), pseudocount=PSEUDOCOUNT,
                  tolerance=tolerance, solver_tolerance=solver_tolerance, snapshots=snapshots,
                  rounds=int(info["rounds"]), status="converged",
                  reconstruction_relative_error=error, **scores), history)

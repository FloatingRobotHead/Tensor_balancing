"""Tensor decomposition and conversion to unit-fiber output factors."""
from dataclasses import dataclass, replace
import numpy as np
from .solvers import dh_sqrt
from .checks import MAX_EXACT_INTEGER, SYMMETRY_RTOL, check_python_integers, positive_integer, scalar

from .pseudocount import PSEUDOCOUNT, protect_zeros


@dataclass
class Result:
    bias_vector: np.ndarray
    bias_matrix: np.ndarray
    balanced_tensor: np.ndarray
    diagnostics: dict
    history: list
    reconstruction_scalar: float | None = None

    @property
    def scalar(self):
        """Compatibility alias for the early 0.2.0 development API."""
        return self.reconstruction_scalar


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
            raise ValueError("The shared-factor interface requires a permutation-symmetric tensor. "
                             "An asymmetric tensor needs separate factors for each axis and pair; "
                             "this tool does not silently symmetrize data.")
    return x


def extract_factors(raw, residual):
    """Canonical log-ANOVA decomposition of the removed multiplicative bias.

    Absorb the scalar scale into the shared vector; pair row/column geometric
    means are one. This is a factor extraction, not a second normalization.
    """
    z = np.log(protect_zeros(raw)) - np.log(residual)
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
        error = max(error, float(np.max(np.abs(np.expm1(rebuilt_log - np.log(protect_zeros(raw[i])))))))
    if not np.isfinite(error) or error > 1e-8:
        raise FloatingPointError(f"Three-factor reconstruction failed (relative error {error:.3g}).")
    return b, matrix, error


def balance(raw, *, method="dh", snapshots=None, tolerance=1e-10, max_iterations=5000):
    """Return the bias vector, bias matrix and mean-one balanced tensor.

    protect_zeros(raw) ≈ b[i] b[j] b[k] M[i,j] M[i,k] M[j,k] R[i,j,k].
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


def separate_scale(bias_vector):
    """Center a mean-one-residual divisor vector and return the reconstruction scalar."""
    b = np.asarray(bias_vector, dtype=float)
    if b.ndim != 1 or not b.size or not np.isfinite(b).all() or np.any(b <= 0):
        raise ValueError("Bias vector must be a nonempty positive finite vector.")
    logs = np.log(b)
    center = float(logs.mean())
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        vector = np.exp(logs - center)
        overall = float(np.exp(np.log(b.size) + 3 * center))
    if overall <= 0 or np.any(vector <= 0):
        raise ValueError("Scalar or vector underflowed during factor separation.")
    return vector, overall


def unit_fiber_result(result):
    """Convert the internal mean-one result to four reconstructing output factors.

    protect_zeros(raw) = reconstruction_scalar * b_i*b_j*b_k * M_ij*M_ik*M_jk * T_ijk.
    T has unit fiber sums; b has geometric mean one.
    """
    if result.reconstruction_scalar is not None:
        return result
    n = result.bias_vector.size
    vector, overall = separate_scale(result.bias_vector)
    return replace(result, bias_vector=vector, balanced_tensor=result.balanced_tensor / n,
                   reconstruction_scalar=overall, diagnostics=dict(result.diagnostics, fiber_sum_target=1.0,
                   tensor_mean_target=1.0/n, mean_one_to_unit_fiber_multiplier=1.0/n,
                   output_convention="unit-fibers-v1"))

"""DH evaluated as W[i,j,k]=protect_zeros(raw)*A[i,j]*A[i,k]*A[j,k].

The epsilon background is removed at positive observations so they stay unchanged.
The update is algebraically the dense symmetric DH square-root update.
"""
from itertools import permutations
import numpy as np
from ..checks import positive_integer, scalar
from ..pseudocount import PSEUDOCOUNT


def ordered(i, j, k, count):
    """Expand canonical triples into distinct permutations, preserving counts."""
    columns = [np.asarray(i), np.asarray(j), np.asarray(k)]
    count = np.asarray(count)
    if any(column.shape != count.shape for column in columns) or count.ndim != 1:
        raise ValueError('Triplet coordinates and counts must be matching vectors.')
    indices, values, previous = [], [], []
    for permutation in permutations(range(3)):
        candidate = np.stack([columns[axis] for axis in permutation], axis=1)
        keep = np.ones(len(count), dtype=bool)
        for earlier in previous:
            keep &= np.any(candidate != earlier, axis=1)
        indices.append(candidate[keep])
        values.append(count[keep])
        previous.append(candidate)
    return np.concatenate(indices).astype(np.int64), np.concatenate(values).astype(float)


def solve(n, indices, counts, tolerance=1e-10, max_rounds=5000, progress=None):
    """Balance symmetric ordered counts, with 1e-12 only at zero-count cells.

    Caller supplies unique ordered coordinates with permutation-symmetric counts,
    normally produced by ordered() from unique sorted canonical triples.
    """
    n = positive_integer(n, 'Number of bins')
    max_rounds = positive_integer(max_rounds, 'Maximum rounds')
    tolerance = min(scalar(tolerance, 'Tolerance'), 1e-10)
    indices, counts = np.asarray(indices), np.asarray(counts, dtype=float)
    if indices.shape != (len(counts), 3) or indices.dtype.kind not in 'iu':
        raise ValueError('Expected integer coordinates with shape (number of counts, 3).')
    if np.any(indices < 0) or np.any(indices >= n) or not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError('Invalid sparse coordinates or counts.')
    positive = counts > 0
    indices, counts = indices[positive], counts[positive]
    i, j, k = indices.T
    ij, ik, jk = i*n+j, i*n+k, j*n+k
    total = PSEUDOCOUNT*(n**3-len(counts)) + counts.sum()
    factors = np.full((n, n), (n**3/total)**(1/3))

    def marginals(a):
        flat = a.ravel()
        weights = (counts-PSEUDOCOUNT)*flat[ij]*flat[ik]*flat[jk]
        observed = np.bincount(ij, weights=weights, minlength=n*n).reshape(n, n)
        result = PSEUDOCOUNT*a*(a@a.T) + observed
        if counts.size and counts.min() < PSEUDOCOUNT*1e8:
            # Avoid cancellation when genuine positive values are tiny.
            for first in range(n):
                slab = np.full((n, n), PSEUDOCOUNT)
                keep = i == first
                slab[j[keep], k[keep]] = counts[keep]
                result[first] = a[first]*np.sum(slab*a[first,None,:]*a, axis=1)
        if not np.isfinite(result).all() or np.any(result <= 0):
            raise FloatingPointError('Invalid compact DH marginals.')
        return result

    pair_sums = marginals(factors)
    history, streak = [], 0
    for iteration in range(1, max_rounds+1):
        plane = pair_sums.sum(1)
        log_update = -0.5*np.log(pair_sums) + 0.25*(np.log(plane)[:, None] + np.log(plane)[None, :])
        candidate = factors*np.exp(log_update)
        updated_sums = marginals(candidate)
        scale = (n**3/updated_sums.sum())**(1/3)
        candidate *= scale
        updated_sums *= scale**3
        # Conservative bound on change of ANY tensor entry from its three pairs.
        change_bound = float(np.expm1(3*np.max(np.abs(log_update+np.log(scale)))))
        plane = updated_sums.sum(1)
        stack = float((updated_sums.max()-updated_sums.min())/updated_sums.min())
        planes = float((plane.max()-plane.min())/plane.min())
        history.append(dict(round=iteration, max_relative_change_bound=change_bound,
                            stack_excess=stack, plane_excess=planes))
        factors, pair_sums = candidate, updated_sums
        streak = streak+1 if max(change_bound, stack, planes) < tolerance else 0
        if progress:
            progress(history[-1])
        if streak >= 3:
            break
    else:
        raise RuntimeError('DH did not converge; no balanced result published.')
    logs = -np.log(factors)
    row, center = logs.mean(1), logs.mean()
    b = np.exp(2*row-center)
    matrix = np.exp(logs-row[:, None]-row[None, :]+center)
    if not np.isfinite(b).all() or not np.isfinite(matrix).all() or np.any(b <= 0) or np.any(matrix <= 0):
        raise FloatingPointError('Invalid divisor factors.')
    diagnostics = dict(status='converged', rounds=iteration, tolerance=tolerance,
                       pseudocount=PSEUDOCOUNT, residual_mean=float(pair_sums.sum()/n**3),
                       **{key: value for key, value in history[-1].items() if key != 'round'})
    return b, matrix, factors, diagnostics, history

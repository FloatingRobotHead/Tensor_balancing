"""Binomial odds directions with outer-objective backtracking.

The output is a relative-weight tensor. A separate working scale may be
reduced to make the binomial parameterization feasible. No DH, ANOVA, or
reference answer is used in this solver. Positive symmetric full cubes only.
"""
import numpy as np
from scipy.special import logsumexp
from .binomial_fit import fit_eta
from ..checks import SYMMETRY_RTOL


def unit_log(z):
    return z - (logsumexp(z) - np.log(z.size))


def diagnostics(y):
    lines = np.concatenate([y.sum(a).ravel() for a in range(3)])
    planes = np.concatenate([y.sum(tuple(b for b in range(3) if b != a)).ravel()
                             for a in range(3)])
    return dict(line_ratio=float(lines.max()/lines.min()),
                plane_ratio=float(planes.max()/planes.min()))


def objective_delta(y, direction, alpha):
    """Stable change in log(mean(exp(z))) - mean(z), in extended precision.

    Subtracting nearly equal objective totals near convergence is avoided.
    Scaling y exactly to mean one is important for this local calculation.
    """
    yy = y.astype(np.longdouble)
    yy /= yy.mean()
    d = direction.astype(np.longdouble)
    d -= d.mean()
    slope = np.mean((yy-1)*d)
    t = np.longdouble(alpha)*d
    remainder = np.empty_like(t)
    small = np.abs(t) < 1e-3
    s = t[small]
    remainder[small] = s*s*(.5+s*(1/6+s*(1/24+s*(1/120+s/720))))
    with np.errstate(over='ignore', invalid='ignore'):
        remainder[~small] = np.expm1(t[~small])-t[~small]
        difference = np.log1p(alpha*slope+np.mean(yy*remainder))
    return float(difference), float(slope)


def normalize(raw, snapshots, *, tolerance=1e-10, max_rounds=2000,
              scale_policy='safe', armijo=1e-4):
    """Return (mean-one weights, diagnostics, accepted-step history).

    safe: numerical working mean <= 0.5/n; reduce it further if any
          intermediate accepted shape would have working probability > 0.5.
    original: preserve the input probability mean; infeasible targets may
              stop at a boundary, and are NEVER reported as converged.

    A balanced n-cube of mean one has every voxel <= n. The safe working
    scale therefore places its unique balanced target strictly below 0.5.
    Only a global constant is changed, not any relative input value.
    """
    x = np.asarray(raw, dtype=float)
    if x.ndim != 3 or len(set(x.shape)) != 1 or not np.isfinite(x).all():
        raise ValueError('A finite cubic tensor is required')
    if snapshots <= 0 or (x < 0).any() or (x > snapshots).any():
        raise ValueError('Require S > 0 and 0 <= counts <= S')
    if any(not np.allclose(x, x.transpose(o), rtol=SYMMETRY_RTOL, atol=0)
           for o in [(1,0,2),(0,2,1)]):
        raise ValueError('Symmetric full-cube counts required')
    n = len(x)
    z = unit_log(np.log(x+.5))
    y = np.exp(z)
    original_mu = float(np.mean((x+.5)/(snapshots+1)))
    if scale_policy == 'safe':
        mu = min(original_mu, .5/n, .5/float(y.max()))
    elif scale_policy == 'original':
        mu = original_mu
    else:
        raise ValueError('scale_policy must be safe or original')
    history = []
    status = 'max_rounds'
    reason = None
    streak = 0
    scale_reductions = 0
    for iteration in range(max_rounds):
        scores = diagnostics(y)
        balanced = max(scores['line_ratio']-1, scores['plane_ratio']-1) <= tolerance
        if balanced and (iteration == 0 or streak >= 3):
            status = 'converged'
            break
        eta, fit = fit_eta(mu*y)
        # Center coefficients BEFORE broadcasting their sum. Otherwise the
        # large constant intercept causes cancellation near convergence and
        # pollutes a direction of size ~1e-10 with errors of size ~1e-15.
        pair = -fit.pop('pair_coefficients')
        pair -= pair.mean()
        direction = pair[:,:,None]+pair[:,None,:]+pair[None,:,:]
        _, slope = objective_delta(y, direction, 0.)
        if not np.isfinite(slope) or (slope >= 0 and not balanced):
            status, reason = 'no_descent_direction', 'Fit direction is not descent at numerical precision'
            break
        alpha = 1.
        domain_rejects = progress_rejects = 0
        for trial in range(61):
            nz = unit_log(z+alpha*direction)
            ny = np.exp(nz)
            next_mu = min(mu, .5/float(ny.max())) if scale_policy == 'safe' else mu
            valid = np.isfinite(ny).all() and ny.min() > 0 and next_mu*ny.max() < 1
            delta, _ = objective_delta(y, direction, alpha)
            if valid and np.isfinite(delta) and (delta <= armijo*alpha*slope or
                    (balanced and abs(alpha*direction).max() < 1e-14)):
                break
            if not valid:
                domain_rejects += 1
            else:
                progress_rejects += 1
            alpha *= .5
        else:
            status, reason = 'line_search_failed', 'No acceptable positive step found'
            break
        change = float(np.max(np.abs(np.expm1(nz-z))))
        history.append(dict(round=iteration+1, step_fraction=alpha,
                            objective=float(-np.mean(nz.astype(np.longdouble))),
                            objective_delta=delta, directional_derivative=slope,
                            domain_rejects=domain_rejects, progress_rejects=progress_rejects,
                            max_relative_change=change, probability_max=float(next_mu*ny.max()),
                            working_mean=next_mu, scale_reduced=next_mu < mu,
                            fit_relative_gradient=fit['relative_gradient_max'],
                            **diagnostics(ny)))
        scale_reductions += int(next_mu < mu)
        y, z, mu = ny, nz, next_mu
        passed = max(history[-1]['line_ratio']-1, history[-1]['plane_ratio']-1, change) < tolerance
        streak = streak+1 if passed else 0
        if change <= 1e-15 and history[-1]['line_ratio']-1 > tolerance:
            status, reason = 'stalled', 'Tiny step with unresolved imbalance; not convergence'
            break
    else:
        if streak >= 3:
            status = 'converged'
    original_scale_max = float(original_mu*y.max())
    info = dict(status=status, reason=reason, rounds=len(history), **diagnostics(y),
                original_mean=original_mu, working_mean=mu,
                working_scale_ratio=mu/original_mu, scale_policy=scale_policy,
                working_probability_max=float(mu*y.max()),
                original_scale_max=original_scale_max,
                original_probability_target_feasible=(original_scale_max < 1) if status == 'converged' else None,
                progress_backtracks=sum(h['progress_rejects'] for h in history),
                domain_backtracks=sum(h['domain_rejects'] for h in history),
                additional_scale_reductions=scale_reductions,
                pseudocount=.5, tolerance=tolerance)
    return y, info, history

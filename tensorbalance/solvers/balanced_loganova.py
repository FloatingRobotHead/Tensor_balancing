"""Project balanced log ANOVA, with explicit convergence reporting.

Same projection and sequential marginal-balancing formulas as the recovered
normalizers.py. Replaces its fixed-round driver with a tolerance-based driver.
"""
import numpy as np
from scipy.special import logsumexp
from .dh_sqrt import deviation


def log_projection(z):
    return (z-z.mean(0,keepdims=True)-z.mean(1,keepdims=True)-z.mean(2,keepdims=True)
            +z.mean((0,1),keepdims=True)+z.mean((0,2),keepdims=True)
            +z.mean((1,2),keepdims=True)-z.mean())


def unit_log(z):
    return z-logsumexp(z)+np.log(z.size)


def normalize(raw, *, tolerance=1e-10, max_rounds=5000):
    x=np.asarray(raw,dtype=float)
    if x.ndim!=3 or not x.size or len(set(x.shape))!=1 or not np.isfinite(x).all() or (x<0).any():
        raise ValueError('Finite nonnegative nonempty cubic counts required')
    if not np.isfinite(tolerance) or tolerance<=0 or max_rounds<1:
        raise ValueError('Require positive tolerance and max_rounds')
    z=unit_log(log_projection(np.log(x+.5)))
    history=[];streak=0;status='max_rounds'
    for iteration in range(1,max_rounds+1):
        nz=z.copy()
        for axis in (0,1,2):
            nz=nz-logsumexp(nz,axis=axis,keepdims=True)+np.log(x.shape[axis])
        nz=unit_log(nz);y=np.exp(nz)
        if not np.isfinite(y).all() or y.min()<=0:
            raise FloatingPointError('Nonpositive or nonfinite update')
        change=float(abs(np.expm1(nz-z)).max());scores=deviation(y)
        history.append(dict(round=iteration,max_relative_change=change,**scores))
        z=nz
        passed=max(change,scores['stack_excess'],scores['plane_excess'])<tolerance
        streak=streak+1 if passed else 0
        if streak>=3:
            status='converged';break
    return y,dict(status=status,rounds=iteration,pseudocount=.5,tolerance=tolerance,**scores),history

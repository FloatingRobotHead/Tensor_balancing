"""The project's DH formula with a fixed square-root correction each round.

The sole mathematical change is raising the complete divisor D to alpha,
with default alpha=0.5. No reference tensor, adaptive line search, or other
normalizer is used. Full-cube mean-one relative weights are returned.
"""
import numpy as np


def normalize_mean(z):
    z = z-z.max()
    return z-np.log(np.exp(z).mean())


def log_divisor(w):
    """Log of the original DH correction, calculated without huge products."""
    return (np.log(w.sum())
            + np.log(w.sum(2))[:,:,None]
            + np.log(w.sum(1))[:,None,:]
            + np.log(w.sum(0))[None,:,:]
            - np.log(w.sum((1,2)))[:,None,None]
            - np.log(w.sum((0,2)))[None,:,None]
            - np.log(w.sum((0,1)))[None,None,:])


def deviation(w):
    a=w.astype(np.longdouble)
    lines=np.concatenate([a.sum(axis).ravel() for axis in range(3)])
    planes=np.concatenate([a.sum(tuple(b for b in range(3) if b!=axis)).ravel() for axis in range(3)])
    return dict(stack_excess=float((lines.max()-lines.min())/lines.min()),
                plane_excess=float((planes.max()-planes.min())/planes.min()))


def normalize(raw, *, alpha=0.5, pseudocount=0.5, tolerance=1e-10, max_rounds=5000):
    x=np.asarray(raw,dtype=float)
    if x.ndim!=3 or len(set(x.shape))!=1 or not np.isfinite(x).all() or (x<0).any():
        raise ValueError('Finite nonnegative cubic tensor required')
    if pseudocount<=0 or not 0<alpha<=1:
        raise ValueError('Require positive pseudocount and 0 < alpha <= 1')
    z=normalize_mean(np.log(x+pseudocount));w=np.exp(z)
    streak=0;history=[];status='max_rounds'
    for it in range(1,max_rounds+1):
        nz=normalize_mean(z-alpha*log_divisor(w));new=np.exp(nz)
        if not np.isfinite(new).all() or new.min()<=0:
            raise FloatingPointError('Nonpositive or nonfinite update')
        change=float(abs(np.expm1(nz-z)).max())
        scores=deviation(new)
        history.append(dict(round=it,max_relative_change=change,**scores))
        w,z=new,nz
        passed=max(change,scores['stack_excess'],scores['plane_excess'])<tolerance
        streak=streak+1 if passed else 0
        if streak>=3:
            status='converged';break
    return w,dict(status=status,rounds=it,alpha=alpha,tolerance=tolerance,
                  pseudocount=pseudocount,**scores),history

"""Original Newton-CG binomial background fitter, with coefficients exposed.

The objective, tolerance, and Newton steps are unchanged. Returning the
coefficients lets the outer solver remove the constant intercept before
broadcasting, avoiding cancellation near convergence. No normalizer fallback.
"""
import itertools
from functools import lru_cache
import numpy as np
from scipy.special import expit, logit
from scipy.sparse.linalg import LinearOperator, cg

FIT_TOL = 1e-13

@lru_cache(None)
def design(n):
    can = np.array(list(itertools.combinations_with_replacement(range(n), 3)))
    pairs = np.zeros((n, n), dtype=np.int32)
    for e, (i, j) in enumerate(itertools.combinations_with_replacement(range(n), 2)):
        pairs[i, j] = pairs[j, i] = e
    i, j, k = can.T
    edges = np.array([pairs[i, j], pairs[i, k], pairs[j, k]])
    weight = np.where(i == k, 1., np.where((i == j) | (j == k), 3., 6.))
    return can, edges, weight, pairs

def fit_eta(p):
    if not np.isfinite(p).all() or p.min() <= 0 or p.max() >= 1:
        raise ValueError(f'Working probabilities outside (0,1): {p.min()}, {p.max()}')
    if max(abs(p-p.transpose(o)).max() for o in [(1,0,2),(0,2,1)]) > 1e-12:
        raise ValueError('Symmetric contact tensor required')
    n=len(p);can,edges,weight,pairs=design(n);obs=p[tuple(can.T)]
    size=n*(n+1)//2;mu=float(np.average(obs,weights=weight));scale=mu*(1-mu)
    beta=np.full(size,logit(mu)/3)
    total_cg=0;backtracks=0
    for iteration in range(101):
        eta=beta[edges].sum(0);q=expit(eta)
        residual=weight*(q-obs)/n
        g=sum(np.bincount(e,weights=residual,minlength=size) for e in edges)
        rel=float(abs(g).max()/scale)
        if rel<=FIT_TOL:
            full_eta=beta[pairs][:,:,None]+beta[pairs][:,None,:]+beta[pairs][None,:,:]
            return full_eta,dict(pair_coefficients=beta[pairs].copy(),newton_iterations=iteration,cg_iterations=total_cg,
                backtracks=backtracks,relative_gradient_max=rel,gradient_max=float(abs(g).max()),
                eta_min=float(full_eta.min()),eta_max=float(full_eta.max()))
        if iteration==100:break
        curvature=weight*q*(1-q)/n
        def hv(v):
            z=curvature*v[edges].sum(0)
            return sum(np.bincount(e,weights=z,minlength=size) for e in edges)
        diag=sum(np.bincount(e,weights=curvature,minlength=size) for e in edges)
        for a,b in [(0,1),(0,2),(1,2)]:
            same=edges[a]==edges[b]
            diag+=np.bincount(edges[a,same],weights=2*curvature[same],minlength=size)
        counter=[0]
        def callback(_):counter[0]+=1
        direction,info=cg(LinearOperator((size,size),matvec=hv,dtype=float),-g,
            M=LinearOperator((size,size),matvec=lambda v:v/diag,dtype=float),
            rtol=1e-11,atol=0,maxiter=500,callback=callback)
        total_cg+=counter[0]
        if info!=0:raise RuntimeError(f'Newton-CG failed: {info}')
        slope=float(g@direction)
        if slope>=0:raise RuntimeError('Newton direction is not descent')
        deta=direction[edges].sum(0);step=1.
        # Stable cross-entropy objective DIFFERENCE near the optimum.
        for search in range(50):
            delta=step*deta;small=abs(delta)<1e-3;v=q*(1-q)
            change=np.empty_like(delta);z=delta[small];qq=q[small];vv=v[small]
            change[small]=(qq-obs[small])*z+vv*z*z*(.5+(1-2*qq)*z/6+(1-6*vv)*z*z/24)
            big=~small
            change[big]=np.logaddexp(0,eta[big]+delta[big])-np.logaddexp(0,eta[big])-obs[big]*delta[big]
            if float(weight@change/n)<=1e-4*step*slope:
                beta+=step*direction;break
            step*=.5;backtracks+=1
        else:raise RuntimeError(f'Newton line search failed at relative gradient {rel}')
    raise RuntimeError(f'Binomial fit failed: relative gradient {rel}')

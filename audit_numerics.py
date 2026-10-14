#!/usr/bin/env python3
"""Reproducible numerical audit with an independently implemented IPF reference."""
import argparse
import itertools
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import scipy
from tensorbalance import balance, __version__
from tensorbalance.decompose import ConvergenceError


def cases():
    rng=np.random.default_rng(20261013)
    def sym(a):return sum(a.transpose(p) for p in itertools.permutations(range(3)))/6
    result=[]
    for n in (1,2,3,5,8):
        result += [(f'zero_{n}',np.zeros((n,)*3)),(f'uniform_{n}',np.full((n,)*3,17.)),
                   (f'poisson_{n}',np.rint(sym(rng.poisson(20,size=(n,)*3))))]
    for n in (3,5,8):
        result.append((f'sparse_{n}',sym(rng.binomial(1,.08,size=(n,)*3)*rng.integers(1,100,size=(n,)*3))))
        x=np.zeros((n,)*3);x[0,0,0]=1e6;result.append((f'corner_spike_{n}',x))
        x=np.zeros((n,)*3);ii=np.arange(n);x[ii,ii,ii]=10000;result.append((f'diagonal_{n}',x))
    for strength in (1,4,10):
        result.append((f'lognormal_{strength}',np.exp(sym(rng.normal(size=(4,)*3)*strength))*100))
    result.append(('patchwork',np.load(Path(__file__).parent/'examples/patchwork.npy',allow_pickle=False)))
    return result


def reference_ipf(raw):
    # Independent arithmetic implementation: directly scale every fiber to n,
    # cycling through the axes, starting from raw+0.5 (no source-solver calls).
    n=len(raw);a=raw.copy()+.5;a/=a.mean()
    for iteration in range(100000):
        for axis in range(3):a*=n/a.sum(axis,keepdims=True)
        error=max(float(np.max(np.abs(a.sum(axis)/n-1))) for axis in range(3))
        if error<1e-12:return a,iteration+1
    raise RuntimeError('Audit reference failed to converge')


def audit():
    report={'version':__version__,'platform':platform.platform(),'python':sys.version.split()[0],
            'numpy':np.__version__,'scipy':scipy.__version__,'seed':20261013,'runs':[],'extended_cases':[]}
    for name,raw in cases():
        expected,steps=reference_ipf(raw)
        dh=None
        for method in ('dh','log-anova','binomial'):
            started=time.perf_counter();limit=5000;before=raw.copy()
            snapshots=max(float(raw.max()),1.)
            try:r=balance(raw,method=method,snapshots=snapshots,max_iterations=limit)
            except ConvergenceError as exc:
                report['extended_cases'].append({'case':name,'method':method,'default_limit':limit,'message':str(exc)})
                limit=10000;r=balance(raw,method=method,snapshots=snapshots,max_iterations=limit)
            b,m,w=r.bias_vector,r.bias_matrix,r.balanced_tensor
            np.testing.assert_array_equal(raw,before)
            assert np.isfinite(w).all() and (w>0).all()
            fiber=max(float(np.max(np.abs(w.sum(axis)/len(w)-1))) for axis in range(3))
            plane=max(float(np.max(np.abs(w.sum(tuple(q for q in range(3) if q!=axis))/len(w)**2-1))) for axis in range(3))
            ref_error=float(np.max(np.abs(w/expected-1)))
            rebuilt=b[:,None,None]*b[None,:,None]*b[None,None,:]*m[:,:,None]*m[:,None,:]*m[None,:,:]*w
            reconstruction=float(np.max(np.abs(rebuilt/(raw+.5)-1)))
            centering=float(np.max(np.abs(np.log(m).mean(axis=1))))
            symmetry=max(float(np.max(np.abs(w/w.transpose(p)-1))) for p in itertools.permutations(range(3)))
            assert fiber<1e-10 and plane<1e-10 and abs(float(w.mean())-1)<1e-12
            assert ref_error<1e-8 and reconstruction<1e-8 and centering<1e-12 and symmetry<1e-8
            if method=='dh':dh=r
            agreement=max(float(np.max(np.abs(a/b-1))) for a,b in
                          ((r.bias_vector,dh.bias_vector),(r.bias_matrix,dh.bias_matrix),(w,dh.balanced_tensor)))
            assert agreement<1e-8
            entry=dict(case=name,method=method,status='passed',shape=list(raw.shape),
                       max_iterations=limit,rounds=r.diagnostics['rounds'],reference_rounds=steps,
                       max_fiber_relative_error=fiber,max_plane_relative_error=plane,
                       reference_relative_error=ref_error,reconstruction_relative_error=reconstruction,
                       max_log_pair_row_mean=centering,symmetry_relative_error=symmetry,
                       agreement_with_dh=agreement,seconds=time.perf_counter()-started)
            report['runs'].append(entry)
            print(f'{name}: {method} passed ({entry["rounds"]} iterations)',file=sys.stderr,flush=True)
    report['case_count']=len(cases());report['run_count']=len(report['runs']);report['status']='passed'
    for field in ('max_fiber_relative_error','max_plane_relative_error','reference_relative_error',
                  'reconstruction_relative_error','agreement_with_dh'):
        report['worst_'+field]=max(item[field] for item in report['runs'])
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True,help='new JSON audit report')
    args=parser.parse_args()
    if args.out.exists():parser.error('Report already exists; choose another --out path.')
    report=audit();args.out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f'{report["run_count"]} runs passed; {len(report["extended_cases"])} required the extended iteration limit.')

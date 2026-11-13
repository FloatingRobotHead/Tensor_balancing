"""Expand a compact DH result into complete NPY arrays, one slab at a time."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from .solvers.compact_dh import ordered
from .storage import write_directory
from .decompose import separate_scale
from .input import load_reconstruction_scalar
from .pseudocount import protect_zeros


def export_dense(source, destination):
    source, destination = Path(source), Path(destination)
    b = np.load(source/'bias_vector.npy', allow_pickle=False)
    matrix = np.load(source/'bias_matrix.npy', allow_pickle=False)
    factors = np.load(source/'residual_pair_factors.npy', allow_pickle=False)
    n = len(b)
    if matrix.shape != (n,n) or factors.shape != (n,n):
        raise ValueError('Inconsistent compact factor shapes.')
    if any(not np.isfinite(a).all() or np.any(a <= 0) for a in (b, matrix, factors)):
        raise ValueError('Compact factors must be positive and finite.')
    overall = load_reconstruction_scalar(source)
    if overall is None:
        # Legacy compact files use a mean-one tensor with global scale inside b.
        b, overall = separate_scale(b)
        factors = factors / np.cbrt(n)
    with np.load(source/'triplet_counts.npz', allow_pickle=False) as data:
        epsilon = float(data['zero_pseudocount']) if 'zero_pseudocount' in data else None
        if epsilon is not None and (not np.isfinite(epsilon) or epsilon <= 0):
            raise ValueError('Invalid zero-only pseudocount.')
        ids, counts = ordered(data['i'], data['j'], data['k'], data['count'])
    if np.any(ids < 0) or np.any(ids >= n) or not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError('Invalid compact counts.')
    order = np.argsort(ids[:,0], kind='stable')
    ids, counts = ids[order], counts[order]
    offsets = np.r_[0, np.cumsum(np.bincount(ids[:,0], minlength=n))]

    def write(stage):
        required = 8*n**3 + 8*n*n + 8*n + 1024*1024
        if shutil.disk_usage(stage).free < required:
            raise OSError(f'Full export requires at least {required:,} free bytes.')
        tensor = np.lib.format.open_memmap(stage/'balanced_tensor.npy', mode='w+', dtype=np.float64, shape=(n,n,n))
        marginal0 = np.zeros((n,n))
        marginal1 = np.zeros((n,n))
        marginal2 = np.zeros((n,n))
        reconstruction_error = 0.0
        try:
            for i in range(n):
                raw = np.zeros((n,n))
                start, stop = offsets[i:i+2]
                cells = ids[start:stop]
                raw[cells[:,1], cells[:,2]] = counts[start:stop]
                protected = raw+0.5 if epsilon is None else protect_zeros(raw, epsilon)
                slab = protected*factors[i,:,None]*factors[i,None,:]*factors
                rebuilt = overall*slab*b[i]*b[:,None]*b[None,:]*matrix[i,:,None]*matrix[i,None,:]*matrix
                error = float(np.max(np.abs(rebuilt/protected-1)))
                if not np.isfinite(slab).all() or np.any(slab <= 0) or not np.isfinite(error):
                    raise ValueError('Invalid dense reconstruction.')
                reconstruction_error = max(reconstruction_error, error)
                marginal0 += slab
                marginal1[i] = slab.sum(0)
                marginal2[i] = slab.sum(1)
                tensor[i] = slab
                if i % 100 == 0:
                    print(f'Exported slab {i+1}/{n}', flush=True)
            tensor.flush()
        finally:
            del tensor
        fiber_error = max(float(np.max(np.abs(m-1))) for m in (marginal0, marginal1, marginal2))
        mean = float(marginal0.sum()/n**3)
        if reconstruction_error > 1e-8 or fiber_error > 1e-10 or abs(mean*n-1) > 1e-12:
            raise ValueError('Full output failed reconstruction or marginal validation.')
        for name in ('metadata.json', 'bins.csv'):
            if (source/name).is_file():
                shutil.copyfile(source/name, stage/name)
        np.save(stage/'bias_vector.npy', b)
        np.save(stage/'bias_matrix.npy', matrix)
        np.save(stage/'reconstruction_scalar.npy', np.asarray(overall))
        checks = dict(reconstruction_relative_error=reconstruction_error,
                      max_fiber_target_relative_error=fiber_error, residual_mean=mean,
                      fiber_sum_target=1.0, tensor_mean_target=1.0/n, output_convention="unit-fibers-v1",
                      reconstruction_scalar=overall, mean_one_to_unit_fiber_multiplier=1.0/n,
                      tensor_values=n**3, tensor_bytes=8*n**3)
        (stage/'validation.json').write_text(json.dumps(checks, indent=2)+'\n', encoding='utf-8')
    return write_directory(destination, write)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(export_dense(args.source, args.out))
        return 0
    except (ValueError, OSError, KeyError, MemoryError) as exc:
        print(f'tensor-export-dense: {exc}', file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    raise SystemExit(main())

"""Prepare segment-triplet Parquet and balance its complete cube using compact DH."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import sys
import shutil
import time

import numpy as np

from . import __version__
from .checks import positive_integer
from .decompose import Result, separate_scale
from .pseudocount import PSEUDOCOUNT, protect_zeros
from .solvers.compact_dh import ordered, solve
from .storage import write_directory
from .visualize import save_view


def prepare(path, chromosome, bin_size, max_bins=2500):
    """Count rows once using midpoint bins; retain bins 0 through the last observation."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    positive_integer(bin_size, 'Bin size')
    positive_integer(max_bins, 'Maximum bins')
    source = pq.ParquetFile(path)
    columns = [f'seg{x}_{field}' for x in 'ABC' for field in ('chr', 'start', 'end')]
    missing = set(columns) - set(source.schema_arrow.names)
    if missing:
        raise ValueError(f'Missing Parquet columns: {sorted(missing)}')
    if not source.metadata.num_rows:
        raise ValueError('Parquet contains no records.')
    if source.metadata.num_rows > 2**53:
        raise ValueError('Record count exceeds exact float64 integer precision.')
    blocks = []
    for batch in source.iter_batches(batch_size=250000, columns=columns):
        coords = []
        for segment in 'ABC':
            chrom = batch.column(f'seg{segment}_chr')
            if set(chrom.to_pylist()) != {chromosome}:
                raise ValueError(f'Every segment must belong to {chromosome}; mixed chromosomes are unsupported.')
            arrays = []
            for field in ('start', 'end'):
                column = batch.column(f'seg{segment}_{field}')
                if column.null_count or not pa.types.is_integer(column.type):
                    raise ValueError('Coordinates must be non-null integers.')
                values = column.to_numpy()
                if np.any(values < 0) or np.any(values > np.iinfo(np.int64).max // 2):
                    raise ValueError('Coordinate outside supported nonnegative integer range.')
                arrays.append(values.astype(np.int64))
            start, end = arrays
            if np.any(end <= start):
                raise ValueError('Intervals must have end > start.')
            bins = (start + end) // (2 * bin_size)
            if bins.max() >= max_bins:
                raise ValueError(f'Input exceeds the {max_bins}-bin resource limit. Use larger bins or explicitly raise --max-bins.')
            coords.append(bins)
        blocks.append(np.sort(np.stack(coords, axis=1), axis=1))
    triples, counts = np.unique(np.concatenate(blocks), axis=0, return_counts=True)
    return triples, counts, int(triples.max()) + 1, source.metadata.num_rows


def reconstruct_region(triples, counts, factors, first, stop):
    """Return raw and full-run balanced values for [first, stop) on every axis."""
    n = len(factors)
    if not 0 <= first < stop <= n:
        raise ValueError('Region must satisfy 0 <= first < stop <= number of bins.')
    keep = np.all((triples >= first) & (triples < stop), axis=1)
    ids, values = ordered(*(triples[keep] - first).T, counts[keep])
    raw = np.zeros((stop-first,) * 3)
    raw[tuple(ids.T)] = values
    a = factors[first:stop, first:stop]
    residual = protect_zeros(raw) * a[:, :, None] * a[:, None, :] * a[None, :, :]
    return raw, residual


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def run(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    started = time.perf_counter()
    if not args.unobserved_are_zero:
        raise ValueError('Confirm the counting convention with --unobserved-are-zero before preparing the complete cube.')
    if args.out.exists() or args.out.is_symlink():
        raise ValueError('Output already exists. Choose a new --out directory.')
    before = args.input.stat()
    print('Reading Parquet and counting midpoint triplets...', flush=True)
    triples, counts, n, rows = prepare(args.input, args.chromosome, args.bin_size, args.max_bins)
    print(f'{rows:,} rows; {n} bins; {n**3:,} complete tensor values; {8*n**3/1e9:.2f} GB as dense float64.', flush=True)
    ids, values = ordered(*triples.T, counts)
    def progress(item):
        if item['round'] == 1 or item['round'] % 10 == 0:
            print(f"DH iteration {item['round']}: fiber spread {item['stack_excess']:.3g}", flush=True)
    b, matrix, factors, diagnostics, history = solve(n, ids, values, tolerance=getattr(args, 'tolerance', 1e-10),
                                                       max_rounds=getattr(args, 'max_iterations', 5000), progress=progress)
    # Algebraically checks reconstruction for every possible triple, including zeros.
    pair_product = factors * matrix * np.sqrt(b[:, None] * b[None, :])
    error_bound = float(np.expm1(3 * np.max(np.abs(np.log(pair_product)))))
    if error_bound > 1e-8 or abs(diagnostics['residual_mean'] - 1) > 1e-12:
        raise ValueError('Final reconstruction or mean-one validation failed.')
    b, overall = separate_scale(b)
    factors = factors / np.cbrt(n)
    diagnostics = dict(diagnostics, residual_mean=diagnostics['residual_mean']/n,
                       fiber_sum_target=1.0, tensor_mean_target=1.0/n,
                       mean_one_to_unit_fiber_multiplier=1.0/n, output_convention='unit-fibers-v1')
    input_hash = sha256(args.input)
    after = args.input.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Input changed during processing. Retry with a stable file.')
    report = dict(status='complete', software_version=__version__, method='compact DH', reconstruction_scalar=overall, output_convention='unit-fibers-v1',
                  input=str(args.input.resolve()), input_sha256=input_hash,
                  rows=rows, shape=[n]*3, bin_size_bp=args.bin_size, chromosome=args.chromosome,
                  unique_unordered_triplets=len(counts), dense_tensor_bytes=8*n**3,
                  coordinates='Zero-based half-open intervals; floor(midpoint / bin_size).',
                  domain='All bins from zero through the last observed midpoint; assembly unspecified.',
                  counting='Every supplied row once; all accessions/haplotypes pooled; no additional filtering or deduplication.',
                  symmetry='Each unordered count copied to each distinct permutation, including repeated-index cells.',
                  zeros='Unobserved combinations assigned zero; fixed 1e-12 pseudocount only at zero counts.',
                  reconstruction_relative_error_bound=error_bound, diagnostics=diagnostics,
                  elapsed_seconds=time.perf_counter()-started,
                  versions=dict(python=sys.version.split()[0], numpy=np.__version__, pyarrow=pa.__version__))

    def write(stage):
        np.save(stage / "reconstruction_scalar.npy", np.asarray(overall))
        (stage / "reconstruction_scalar.csv").write_text("reconstruction_scalar\n" + format(overall, ".17g") + "\n", encoding="utf-8")
        metadata = dict(genomic_bin_width=args.bin_size, genomic_bins=list(range(n)), chromosome=args.chromosome)
        (stage / 'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
        for name, array in [('bias_vector', b), ('bias_matrix', matrix), ('residual_pair_factors', factors)]:
            np.save(stage / f'{name}.npy', array, allow_pickle=False)
            with (stage / f'{name}.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.writer(stream)
                if array.ndim == 1:
                    writer.writerow(['i', 'value'])
                    writer.writerows((i, format(v, '.17g')) for i, v in enumerate(array))
                else:
                    writer.writerow(['i', *range(n)])
                    writer.writerows([i, *[format(v, '.17g') for v in row]] for i, row in enumerate(array))
        np.savez_compressed(stage / 'triplet_counts.npz', i=triples[:, 0], j=triples[:, 1], k=triples[:, 2], count=counts, zero_pseudocount=PSEUDOCOUNT)
        i, j, k = triples.T
        observed = protect_zeros(counts) * factors[i,j] * factors[i,k] * factors[j,k]
        pq.write_table(pa.table(dict(i=i, j=j, k=k, raw_count=counts, balanced_value=observed)), stage / 'observed_triplets.parquet')
        with (stage / 'bins.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['index', 'chromosome', 'start_inclusive', 'end_exclusive'])
            writer.writerows((i, args.chromosome, i*args.bin_size, (i+1)*args.bin_size) for i in range(n))
        if args.view_region:
            first, stop = args.view_region
            if stop-first > 80:
                raise ValueError('Visualization region may contain at most 80 bins per axis.')
            raw, residual = reconstruct_region(triples, counts, factors, first, stop)
            region_result = Result(b[first:stop], matrix[first:stop, first:stop], residual, dict(diagnostics, method='dh'), [], reconstruction_scalar=overall)
            save_view(stage / 'visualization.html', raw, region_result)
            view = stage / 'visualization.html'
            label = (f'<p>Excerpt of full chromosome normalization. {html.escape(args.chromosome)}, '
                     f'{first*args.bin_size:,}–{stop*args.bin_size:,} bp. '
                     f'Local index 0 is genomic bin {first}. Full numerical results are in the compact files.</p>')
            view.write_text(view.read_text(encoding='utf-8').replace('<main>', '<main>'+label).replace(' · mean 1', ' · whole-chromosome normalization'), encoding='utf-8')
            np.save(stage / 'view_raw.npy', raw)
            np.save(stage / 'view_balanced.npy', residual)
            report['visualization_region_bins'] = [first, stop]
        (stage / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        (stage / 'history.json').write_text(json.dumps(history, indent=2)+'\n', encoding='utf-8')
        (stage / 'README.txt').write_text(
            'Complete chromosome normalization with numerical factors\n\n'
            'bins.csv maps every index to a genomic interval. report.json records preparation and checks.\n'
            'bias_vector and bias_matrix are divisor factors in NPY and CSV formats.\n'
            'triplet_counts.npz holds unordered observed counts. Absent triplets have raw count zero.\n'
            'residual_pair_factors.npy (A) represents the complete normalized tensor with the counts.\n'
            'For ANY i,j,k, R=where(raw==0,1e-12,raw)*A[i,j]*A[i,k]*A[j,k]. All zero-count cells are included.\n'
            'Recreate original binned counts by multiplying R by reconstruction_scalar*b[i]*b[j]*b[k]*M[i,j]*M[i,k]*M[j,k], then setting originally zero cells back to zero using triplet_counts.npz.\n'
            'observed_triplets.parquet contains only observed unordered triples; use the compact representation for other cells.\n'
            'If requested, visualization is a labeled regional excerpt of this full run.\n'
            'balanced_tensor.npy is the complete numerical tensor with every fiber sum equal to 1 within tolerance.\n'
            'reconstruction_scalar.npy is the positive overall reconstruction factor. The bias vector has geometric mean 1.\n'
            'The tensor total is n*n, and its mean is 1/n. It is not a joint probability distribution.\n'
            'Numerical validation does not establish biological interpretation.\n', encoding='utf-8')
        # The requested complete numerical tensor is published with the factors.
        from .export_dense import export_dense
        full = export_dense(stage, stage / 'full_tensor')
        for name in ('balanced_tensor.npy', 'validation.json'):
            (full / name).rename(stage / name)
        shutil.rmtree(full)
    destination = write_directory(args.out, write)
    print(destination, flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--chromosome', required=True)
    parser.add_argument('--bin-size', type=int, required=True, help='Genomic bin width in base pairs.')
    parser.add_argument('--unobserved-are-zero', action='store_true', help='Explicitly assign unobserved combinations zero counts.')
    parser.add_argument('--tolerance', type=float, default=1e-10)
    parser.add_argument('--max-iterations', type=int, default=5000)
    parser.add_argument('--max-bins', type=int, default=2500)
    parser.add_argument('--view-region', type=int, nargs=2, metavar=('FIRST_BIN', 'STOP_BIN'))
    args = parser.parse_args(argv)
    try:
        run(args)
        return 0
    except (ValueError, OSError, ImportError, MemoryError, RuntimeError, FloatingPointError) as exc:
        print(f'tensor-parquet: {exc}', file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    raise SystemExit(main())

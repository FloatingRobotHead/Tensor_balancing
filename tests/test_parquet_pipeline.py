import itertools
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tensorbalance import balance
from tensorbalance.parquet_pipeline import prepare, reconstruct_region, main
from tensorbalance.solvers.compact_dh import ordered, solve

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None


class CompactTests(unittest.TestCase):
    def test_dense_equivalence_and_reconstruction(self):
        rng = np.random.default_rng(72)
        for n in (1, 3, 7, 15):
            triples = np.array(list(itertools.combinations_with_replacement(range(n), 3)))
            counts = rng.poisson(2, len(triples))
            ids, values = ordered(*triples.T, counts)
            raw = np.zeros((n, n, n))
            raw[tuple(ids.T)] = values
            b, m, a, d, _ = solve(n, ids, values)
            _, residual = reconstruct_region(triples, counts, a, 0, n)
            reference = balance(raw)
            np.testing.assert_allclose(residual, reference.balanced_tensor, rtol=1e-8)
            np.testing.assert_allclose(b, reference.bias_vector, rtol=1e-8)
            np.testing.assert_allclose(m, reference.bias_matrix, rtol=1e-8)
            rebuilt = residual*b[:,None,None]*b[None,:,None]*b[None,None,:]*m[:,:,None]*m[:,None,:]*m[None,:,:]
            np.testing.assert_allclose(rebuilt, raw+0.5, rtol=1e-12)
            for axis in range(3):
                np.testing.assert_allclose(residual.sum(axis), n, rtol=1e-10)

    def test_empty_counts_and_failed_convergence(self):
        ids, values = ordered(*np.empty((0, 3), dtype=int).T, np.array([]))
        b, m, a, _, _ = solve(3, ids, values)
        np.testing.assert_allclose(0.5*a[0,0]**3, 1)
        with self.assertRaises(RuntimeError):
            solve(3, ids, values, max_rounds=1)

    @unittest.skipIf(pa is None, 'Optional pyarrow dependency absent')
    def test_parquet_entry_point_and_counting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = {}
            for segment, starts in zip('ABC', ([0,0,25000], [0,25000,0], [0,25000,25000])):
                data[f'seg{segment}_chr'] = ['chr21']*3
                data[f'seg{segment}_start'] = starts
                data[f'seg{segment}_end'] = [x+2 for x in starts]
            source = root/'input.parquet'
            pq.write_table(pa.table(data), source)
            triples, counts, n, rows = prepare(source, 'chr21', 25000)
            self.assertEqual((n, rows), (2, 3))
            np.testing.assert_array_equal(counts, [1, 2])
            args = [str(source), '--out', str(root/'result'), '--chromosome', 'chr21', '--bin-size', '25000']
            self.assertEqual(main(args), 2)
            self.assertEqual(main(args+['--unobserved-are-zero', '--view-region', '0', '2']), 0)
            self.assertTrue((root/'result'/'report.json').is_file())
            view = (root/'result'/'visualization.html').read_text()
            self.assertIn('Local index 0 is genomic bin 0', view)
            self.assertNotIn(' · mean 1', view)
            from tensorbalance.export_dense import export_dense
            export_dense(root/'result', root/'dense')
            np.testing.assert_array_equal(np.load(root/'dense'/'balanced_tensor.npy'), np.load(root/'result'/'view_balanced.npy'))
            self.assertEqual(main(args+['--unobserved-are-zero']), 2)
            with self.assertRaises(ValueError):
                prepare(source, 'chr22', 25000)
            with self.assertRaises(ValueError):
                prepare(source, 'chr21', 1, max_bins=2)

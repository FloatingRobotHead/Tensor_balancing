"""Fixed zero-only pseudocount and reconstruction regression tests."""
import itertools
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tensorbalance import balance
from tensorbalance.pseudocount import protect_zeros
from tensorbalance.solvers.compact_dh import ordered, solve
from tensorbalance.export_dense import export_dense


class PseudocountTests(unittest.TestCase):
    def test_fixed_across_datasets_and_positive_values_unchanged(self):
        for scale in (1e-100, 1., 1e8):
            raw = np.array([0., scale, 1e-12, 0.])
            original = raw.copy()
            protected = protect_zeros(raw)
            np.testing.assert_array_equal(protected[raw == 0], 1e-12)
            np.testing.assert_array_equal(protected[raw > 0], raw[raw > 0])
            np.testing.assert_array_equal(raw, original)

    def test_all_methods_reconstruct_fixed_zero_only_counts(self):
        raw = np.zeros((3,)*3)
        raw[np.arange(3), np.arange(3), np.arange(3)] = 100
        target = np.where(raw == 0, 1e-12, raw)
        for method in ('dh', 'log-anova', 'binomial'):
            with self.subTest(method=method):
                r = balance(raw, method=method, snapshots=1000)
                b, m, w = r.bias_vector, r.bias_matrix, r.balanced_tensor
                rebuilt = w*b[:,None,None]*b[None,:,None]*b[None,None,:]*m[:,:,None]*m[:,None,:]*m[None,:,:]
                np.testing.assert_allclose(rebuilt, target, rtol=1e-8, atol=0)
                self.assertEqual(r.diagnostics['pseudocount'], 1e-12)

    def test_compact_preserves_tiny_positive_values(self):
        raw = np.full((2,)*3, 1e-30)
        raw[np.indices(raw.shape).sum(0) % 2 == 0] = 0
        triples = np.array(list(itertools.combinations_with_replacement(range(2), 3)))
        ids, counts = ordered(*triples.T, raw[tuple(triples.T)])
        b, m, a, _, _ = solve(2, ids, counts)
        w = protect_zeros(raw)*a[:,:,None]*a[:,None,:]*a[None,:,:]
        rebuilt = w*b[:,None,None]*b[None,:,None]*b[None,None,:]*m[:,:,None]*m[:,None,:]*m[None,:,:]
        np.testing.assert_allclose(rebuilt, protect_zeros(raw), rtol=1e-8, atol=0)
        np.testing.assert_allclose(w.sum(0), 2, rtol=1e-10)
        np.testing.assert_allclose(w, balance(raw).balanced_tensor, rtol=1e-8, atol=1e-10)

    def test_legacy_compact_without_marker_keeps_old_smoothing(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'legacy'
            source.mkdir()
            np.save(source/'bias_vector.npy', np.full(2, np.cbrt(2.5)))
            np.save(source/'bias_matrix.npy', np.ones((2,2)))
            np.save(source/'residual_pair_factors.npy', np.full((2,2), 1/np.cbrt(2.5)))
            triples = np.array(list(itertools.combinations_with_replacement(range(2), 3)))
            np.savez(source/'triplet_counts.npz', i=triples[:,0], j=triples[:,1], k=triples[:,2], count=np.full(4, 2.))
            out = export_dense(source, Path(directory)/'dense')
            np.testing.assert_allclose(np.load(out/'balanced_tensor.npy'), .5, rtol=1e-12)

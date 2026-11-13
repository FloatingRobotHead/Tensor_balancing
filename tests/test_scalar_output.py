import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tensorbalance import balance, unit_fiber_result
from tensorbalance.cli import write_result
from tensorbalance.export_csv import export_csv
from tensorbalance.export_dense import export_dense
from tensorbalance.solvers.compact_dh import ordered, solve


class ScalarOutputTests(unittest.TestCase):
    def test_four_factors_reconstruct_and_unit_fibers_all_methods(self):
        raw=np.full((3,3,3),7.0)
        raw[1,1,1]=50
        for method in ['dh','log-anova','binomial']:
            internal=balance(raw,method=method,snapshots=100)
            result=unit_fiber_result(internal)
            b,m,t,s=result.bias_vector,result.bias_matrix,result.balanced_tensor,result.reconstruction_scalar
            for axis in range(3):
                np.testing.assert_allclose(t.sum(axis),1,rtol=1e-10)
            np.testing.assert_allclose(np.log(b).mean(),0,atol=1e-14)
            np.testing.assert_allclose(np.log(m).mean(0),0,atol=1e-14)
            rebuilt=s*t*b[:,None,None]*b[None,:,None]*b[None,None,:]*m[:,:,None]*m[:,None,:]*m[None,:,:]
            np.testing.assert_allclose(rebuilt,raw,rtol=1e-8)
            np.testing.assert_array_equal(t,internal.balanced_tensor/3)
            self.assertIs(unit_fiber_result(result),result)
            np.testing.assert_allclose(t.sum(),9)

    def test_saved_files_and_csv_scalar(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);raw=np.full((2,2,2),3.)
            source=write_result(root/'out',balance(raw),raw)
            self.assertEqual({p.name for p in source.iterdir()},{'reconstruction_scalar.npy','bias_vector.npy','bias_matrix.npy','balanced_tensor.npy'})
            self.assertEqual(np.load(source/'reconstruction_scalar.npy').shape,())
            np.testing.assert_allclose(np.load(source/'reconstruction_scalar.npy'),6)
            dest=export_csv(source,root/'csv')
            np.testing.assert_equal(np.loadtxt(dest/'reconstruction_scalar.csv',skiprows=1),np.load(source/'reconstruction_scalar.npy'))
            # Early development results used the ambiguous filename scalar.npy.
            (source/'reconstruction_scalar.npy').rename(source/'scalar.npy')
            migrated=export_csv(source,root/'migrated_csv')
            np.testing.assert_equal(np.loadtxt(migrated/'reconstruction_scalar.csv',skiprows=1),np.load(source/'scalar.npy'))
            np.save(source/'reconstruction_scalar.npy',8.)
            with self.assertRaisesRegex(ValueError,'Conflicting'):
                export_csv(source,root/'conflict_csv')
            (source/'scalar.npy').unlink()
            # Older three-file data remains exportable in its original convention.
            (source/'reconstruction_scalar.npy').unlink()
            legacy=export_csv(source,root/'legacy_csv')
            self.assertFalse((legacy/'reconstruction_scalar.csv').exists())

    def test_dense_expansion_accepts_legacy_and_new_compact(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'old';source.mkdir()
            triples=np.array([[0,0,0],[0,0,1],[0,1,1],[1,1,1]])
            counts=np.array([9,2,4,3])
            ids,values=ordered(*triples.T,counts)
            b,m,a,_,_=solve(2,ids,values)
            for name,value in [('bias_vector',b),('bias_matrix',m),('residual_pair_factors',a)]:
                np.save(source/f'{name}.npy',value)
            np.savez(source/'triplet_counts.npz',i=triples[:,0],j=triples[:,1],k=triples[:,2],count=counts,zero_pseudocount=1e-12)
            first=export_dense(source,root/'first')
            np.testing.assert_allclose(np.load(first/'balanced_tensor.npy').sum(0),1,rtol=1e-10)
            # Rewrite compact factors in the new convention and confirm expansion agrees.
            np.save(source/'bias_vector.npy',np.load(first/'bias_vector.npy'))
            np.save(source/'reconstruction_scalar.npy',np.load(first/'reconstruction_scalar.npy'))
            np.save(source/'residual_pair_factors.npy',a/np.cbrt(2))
            second=export_dense(source,root/'second')
            for name in ('bias_vector.npy','bias_matrix.npy','reconstruction_scalar.npy','balanced_tensor.npy'):
                np.testing.assert_array_equal(np.load(first/name),np.load(second/name))
            report=json.loads((second/'validation.json').read_text())
            self.assertEqual(report['fiber_sum_target'],1)
            self.assertEqual(report['mean_one_to_unit_fiber_multiplier'],0.5)
            self.assertEqual(report['reconstruction_scalar'],float(np.load(second/'reconstruction_scalar.npy')))
            (source/'reconstruction_scalar.npy').rename(source/'scalar.npy')
            third=export_dense(source,root/'third')
            np.testing.assert_array_equal(np.load(second/'balanced_tensor.npy'),np.load(third/'balanced_tensor.npy'))

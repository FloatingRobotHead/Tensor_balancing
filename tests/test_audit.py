"""Regression checks for the 0.1.1 audit; malformed data and failure safety."""
import contextlib
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile
import numpy as np
from tensorbalance import balance
from tensorbalance.cli import main, write_result, OUTPUTS
from tensorbalance.decompose import ConvergenceError
from tensorbalance.input import load_input
from tensorbalance.export_csv import export_csv
from test_cli import known_factors

ROOT = Path(__file__).resolve().parents[1]


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = known_factors()[0]
        self.input = self.root / 'raw.npy'
        np.save(self.input, self.raw)

    def cli(self, *args):
        return subprocess.run([sys.executable, str(ROOT/'tensor_balance.py'), *map(str,args)], capture_output=True, text=True)

    def archive(self, name='raw.zip', metadata=None, extra=None):
        path = self.root/name
        with zipfile.ZipFile(path,'w') as z:
            z.write(self.input,'raw.npy')
            if metadata is not None: z.writestr('metadata.json',json.dumps(metadata))
            if extra:
                for k,v in extra.items(): z.writestr(k,v)
        return path

    def test_malformed_metadata_is_clean_error(self):
        for metadata in ([], 'bad', 4, {'snapshots':[]}, {'snapshots':True}, {'mask':[1]}):
            with self.subTest(metadata=metadata):
                path=self.archive(metadata=metadata)
                run=self.cli(path)
                self.assertEqual(run.returncode,2,run.stderr)
                self.assertNotIn('Traceback',run.stderr)
                self.assertFalse((self.root/'raw_balanced').exists())

    def test_csv_rejects_malformed_fields(self):
        invalid = [
            'i,j,k,value,value\n0,0,0,10,99\n',
            'i,j,k,value\n0,0,0,10,99\n',
            'i,j,k,value\n0,0,0\n',
            'i,j,k,value\n0,0,0,10\n0,0,0,10\n',
            'i,j,k,value\n-1,0,0,10\n',
            'i,j,k,value\n0.5,0,0,10\n',
            'i,j,k,value,count\n0,0,0,10,11\n',
            'i,j,k,value\n0,0,0,NaN\n',
            'i,j,k,value\n0,0,0,Infinity\n',
            'i,j,k,value\n0,0,0,1e-999\n',
            'i,j,k,value\n0,0,0,"1\n',
        ]
        for content in invalid:
            with self.subTest(content=content):
                path=self.root/'bad.csv';path.write_text(content)
                run=self.cli(path)
                self.assertEqual(run.returncode,2,run.stderr)
                self.assertNotIn('Traceback',run.stderr)

    def test_exact_integer_limits_across_formats(self):
        for value in (2**53+1,2**63+1):
            for suffix,content in (
                ('csv',f'i,j,k,value\n0,0,0,{value}\n'),
                ('csv',f'i,j,k,value\n0,0,0,{value}.0\n'),
                ('json',json.dumps({'raw':[value,.5,.5,.5,.5,.5,.5,.5],'shape':[2,2,2]}))):
                with self.subTest(value=value,suffix=suffix):
                    path=self.root/f'big.{suffix}';path.write_text(content)
                    with self.assertRaisesRegex(ValueError,r'2\^53'):load_input(path)
        for dtype in (np.int64,np.uint64):
            with self.assertRaisesRegex(ValueError,r'2\^53'):
                balance(np.full((1,1,1),2**53+1,dtype=dtype))
        path=self.root/'limit.csv';path.write_text(f'i,j,k,value\n0,0,0,{2**53}\n')
        self.assertEqual(load_input(path)[0].item(),2**53)

    def test_json_dimensions_and_duplicate_keys(self):
        cases=[{'raw':[1],'n':1.9},{'raw':[1],'n':True},{'raw':[1],'n':0},
               {'raw':[1],'shape':[1,1,1.0]},{'raw':[1],'shape':[1,1]},
               {'raw':[1],'shape':[1,1,1],'n':2},{'raw':[[[1]]],'n':2}]
        for obj in cases:
            with self.subTest(obj=obj):
                path=self.root/'bad.json';path.write_text(json.dumps(obj))
                with self.assertRaises(ValueError):load_input(path)
        path.write_text('{"raw":[[[1]]],"raw":[[[2]]]}')
        with self.assertRaisesRegex(ValueError,'Duplicate'):load_input(path)

    def test_scalar_exposure_types(self):
        for s in (np.array([1,2]),100+2j,True,'100',np.nan,np.inf,-1,0):
            with self.subTest(s=str(s)):
                path=self.root/'bad.npz';np.savez(path,raw=self.raw,snapshots=s)
                with self.assertRaises(ValueError):load_input(path)
                with self.assertRaises(ValueError):balance(self.raw,method='binomial',snapshots=s)

    def test_duplicate_archive_members_and_embedded_masks(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            path=self.root/'duplicate.zip'
            with zipfile.ZipFile(path,'w') as z:
                z.write(self.input,'raw.npy')
                z.writestr('metadata.json','{}');z.writestr('metadata.json','{}')
        with self.assertRaisesRegex(ValueError,'duplicate'):load_input(path)
        for metadata in ({'mask':[]},{'structural_zeros':True}):
            with self.assertRaisesRegex(ValueError,'masks'):load_input(self.archive(metadata=metadata))

    def test_saved_session_and_table_variants(self):
        p=self.root/'session.zip'
        with zipfile.ZipFile(p,'w') as z:
            z.writestr('nested/session.json',json.dumps({'n':len(self.raw),'snapshots':1000}))
            z.writestr('nested/arrays.bin',self.raw.astype('<f8').tobytes()+np.zeros(10,dtype='<f8').tobytes())
        np.testing.assert_array_equal(load_input(p)[0],self.raw)
        self.assertEqual(load_input(p)[1],1000)
        for ext,sep in [('csv',','),('tsv','\t')]:
            p=self.root/f'full.{ext}'
            with p.open('w',encoding='utf-8-sig',newline='') as f:
                out=csv.writer(f,delimiter=sep);out.writerow(['i','j','k','value'])
                out.writerows((*idx,format(self.raw[idx],'.17g')) for idx in reversed(list(np.ndindex(self.raw.shape))))
            np.testing.assert_array_equal(load_input(p)[0],self.raw)

    def test_nested_json_and_text_archives(self):
        for obj in (self.raw.tolist(),{'raw':self.raw.tolist(),'snapshots':1000}):
            path=self.root/'nested.json';path.write_text(json.dumps(obj))
            np.testing.assert_array_equal(load_input(path)[0],self.raw)
            zipped=self.root/'json.zip'
            with zipfile.ZipFile(zipped,'w') as z:z.write(path,'raw.json')
            np.testing.assert_array_equal(load_input(zipped)[0],self.raw)
        table='i,j,k,value\n'+'\n'.join(','.join(map(str,(*idx,self.raw[idx]))) for idx in np.ndindex(self.raw.shape))
        zipped=self.root/'table.zip'
        with zipfile.ZipFile(zipped,'w') as z:z.writestr('data.csv',table)
        np.testing.assert_array_equal(load_input(zipped)[0],self.raw)

    def test_all_documented_input_formats_run(self):
        candidates=[self.input]
        for key in ('raw','tensor','counts','arr_0'):
            p=self.root/f'{key}.npz';np.savez(p,**{key:self.raw,'snapshots':1000});candidates.append(p)
        for key in ('raw','tensor','values','data'):
            p=self.root/f'{key}.json';p.write_text(json.dumps({key:self.raw.ravel().tolist(),'shape':list(self.raw.shape)}));candidates.append(p)
        candidates.append(self.archive())
        for count,p in enumerate(candidates):
            with self.subTest(p=p.name):
                out=self.root/f'result-{count}'
                run=self.cli(p,'--out',out)
                self.assertEqual(run.returncode,0,run.stderr)
                self.assertEqual(len(list(out.iterdir())),3)

    def test_api_rejects_mask_bool_and_bad_controls(self):
        for x in [np.ma.array(self.raw,mask=True),np.ones((2,2,2),dtype=bool), [[[True]]], [[[2**53+1]]]]:
            with self.subTest(type=type(x).__name__),self.assertRaises(ValueError):balance(x)
        for opts in ({'max_iterations':True},{'max_iterations':2.5},{'tolerance':True},{'tolerance':1+2j}):
            with self.subTest(opts=opts),self.assertRaises(ValueError):balance(self.raw,**opts)

    def test_symmetry_threshold_agrees_between_methods(self):
        x=np.ones((3,3,3));x[0,1,2]+=5e-13
        for method in ('dh','log-anova','binomial'):
            with self.subTest(method=method):balance(x,method=method,snapshots=2)
        x[0,1,2]+=1e-8
        for method in ('dh','log-anova','binomial'):
            with self.subTest(method=method),self.assertRaises(ValueError):balance(x,method=method,snapshots=2)

    def test_loose_tolerance_still_reconstructs(self):
        raw=np.load(ROOT/'examples/patchwork.npy')
        for method in ('dh','log-anova','binomial'):
            for tol in (.1,.001):
                with self.subTest(method=method,tol=tol):
                    r=balance(raw,method=method,tolerance=tol,snapshots=100000)
                    self.assertLess(r.diagnostics['reconstruction_relative_error'],1e-10)
                    self.assertEqual(r.diagnostics['solver_tolerance'],1e-10)

    def test_nested_solver_failure_is_clean(self):
        err=io.StringIO()
        with patch('tensorbalance.solvers.repaired_binomial.normalize',side_effect=RuntimeError('injected Newton failure')):
            with contextlib.redirect_stderr(err):
                code=main([str(self.input),'--method','binomial','--snapshots','1000'])
        self.assertEqual(code,3);self.assertNotIn('Traceback',err.getvalue())
        self.assertFalse((self.root/'raw_balanced').exists())

    def test_nonfinite_success_is_rejected(self):
        for w in (np.full(self.raw.shape,np.nan),np.zeros(self.raw.shape),np.full(self.raw.shape,2.)):
            with patch('tensorbalance.solvers.dh_sqrt.normalize',return_value=(w,{'status':'converged','rounds':3},[])):
                with self.assertRaises(ConvergenceError):balance(self.raw)

    def test_failed_write_rolls_back(self):
        result=balance(self.raw)
        with patch('tensorbalance.cli.np.save',side_effect=OSError('injected disk full')):
            with self.assertRaises(OSError):write_result(self.root/'failed',result,self.raw)
        self.assertFalse((self.root/'failed').exists())
        self.assertFalse(list(self.root.glob('.tensor-balance-*')))
        original=Path.rename; calls=0
        def failing_rename(source,dest):
            nonlocal calls
            calls+=1
            if calls==2: raise OSError('injected publication failure')
            return original(source,dest)
        with patch.object(Path,'rename',failing_rename):
            with self.assertRaises(OSError):write_result(self.root/'failed',result,self.raw)
        self.assertFalse((self.root/'failed').exists())
        self.assertFalse(list(self.root.glob('.tensor-balance-*')))

    def test_output_path_errors_are_clean(self):
        self.input.with_name('parent').write_text('a file')
        for out in (self.input,self.root/'parent'/'child'):
            run=self.cli(self.input,'--out',out)
            self.assertEqual(run.returncode,2);self.assertNotIn('Traceback',run.stderr)
        np.testing.assert_array_equal(np.load(self.input),self.raw)

    def test_directory_inputs_fail_cleanly(self):
        for source in ('.',str(Path.cwd().anchor),str(self.root)):
            with self.subTest(source=source):
                run=self.cli(source)
                self.assertEqual(run.returncode,2,run.stderr)
                self.assertNotIn('Traceback',run.stderr)

    def test_concurrent_jobs_do_not_overwrite(self):
        out=self.root/'race'
        command=[sys.executable,str(ROOT/'tensor_balance.py'),str(self.input),'--out',str(out)]
        jobs=[subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for _ in range(2)]
        for job in jobs:job.communicate(timeout=30)
        self.assertEqual(sorted(job.returncode for job in jobs),[0,2])
        self.assertEqual(sorted(p.name for p in out.iterdir()),sorted(OUTPUTS))
        np.testing.assert_allclose(np.load(out/'balanced_tensor.npy'),balance(self.raw).balanced_tensor,rtol=0,atol=0)

    def test_spaces_unicode_and_module_entry(self):
        p=self.root/'tensor data é.npy';np.save(p,self.raw)
        out=self.root/'results with spaces é'
        run=self.cli(p,'--out',out,'--visualize')
        self.assertEqual(run.returncode,0,run.stderr)
        self.assertIn('Tensor Balance',(out/'visualization.html').read_text(encoding='utf-8'))
        run=subprocess.run([sys.executable,'-m','tensorbalance','--version'],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(run.returncode,0);self.assertIn('0.1.1',run.stdout)

    def test_memory_and_interrupt_errors_are_clean(self):
        for error,expected in ((MemoryError(),2),(KeyboardInterrupt(),130)):
            with patch('tensorbalance.cli.load_input',side_effect=error),contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([str(self.input)]),expected)
        self.assertFalse((self.root/'raw_balanced').exists())

    def test_complete_csv_export_roundtrip(self):
        result=balance(self.raw)
        source=write_result(self.root/'result',result,self.raw)
        dest=export_csv(source,self.root/'csv')
        b=np.loadtxt(dest/'bias_vector.csv',delimiter=',',skiprows=1)[:,1]
        m=np.loadtxt(dest/'bias_matrix.csv',delimiter=',',skiprows=1)[:,1:]
        r=load_input(dest/'balanced_tensor.csv')[0]
        for actual,expected in ((b,result.bias_vector),(m,result.bias_matrix),(r,result.balanced_tensor)):
            np.testing.assert_array_equal(actual,expected)
        with (dest/'balanced_tensor.csv').open(newline='') as f:
            self.assertEqual(sum(1 for _ in csv.reader(f)),self.raw.size+1)
        before=(dest/'balanced_tensor.csv').read_bytes()
        with self.assertRaises(ValueError):export_csv(source,dest)
        self.assertEqual((dest/'balanced_tensor.csv').read_bytes(),before)

    def test_bad_export_input_and_scalar_export(self):
        source=write_result(self.root/'scalar',balance(np.zeros((1,1,1))),np.zeros((1,1,1)))
        out=export_csv(source,self.root/'scalar_csv')
        self.assertEqual(load_input(out/'balanced_tensor.csv')[0].shape,(1,1,1))
        np.save(source/'bias_vector.npy',np.array([1.,2.]))
        with self.assertRaises(ValueError):export_csv(source,self.root/'bad_csv')
        self.assertFalse((self.root/'bad_csv').exists())

    def test_disguised_npz_is_clean_error(self):
        data=io.BytesIO();np.savez(data,raw=self.raw)
        path=self.root/'fake.npy';path.write_bytes(data.getvalue())
        with self.assertRaisesRegex(ValueError,'disguised'):load_input(path)
        path=self.root/'fake.zip'
        with zipfile.ZipFile(path,'w') as z:z.writestr('raw.npy',data.getvalue())
        run=self.cli(path)
        self.assertEqual(run.returncode,2);self.assertNotIn('Traceback',run.stderr)

    def test_exposure_integer_precision(self):
        with self.assertRaisesRegex(ValueError,r'2\^53'):
            balance(self.raw,method='binomial',snapshots=2**53+1)

    def test_failure_cleanup_preserves_foreign_files(self):
        result=balance(self.raw);out=self.root/'failed'
        original=Path.rename;calls=0
        def failing_rename(source,dest):
            nonlocal calls
            calls+=1
            if calls==2:
                (out/'foreign.txt').write_text('keep')
                raise OSError('injected publication failure')
            return original(source,dest)
        with patch.object(Path,'rename',failing_rename):
            with self.assertRaises(OSError):write_result(out,result,self.raw)
        self.assertEqual((out/'foreign.txt').read_text(),'keep')
        self.assertEqual([p.name for p in out.iterdir()],['foreign.txt'])

    def test_known_pair_only_and_triple_only_controls(self):
        raw,b,m,r=known_factors()
        pair_only=b[:,None,None]*b[None,:,None]*b[None,None,:]*m[:,:,None]*m[:,None,:]*m[None,:,:]-.5
        triple_only=50*r-.5
        for method in ('dh','log-anova','binomial'):
            with self.subTest(method=method):
                a=balance(pair_only,method=method,snapshots=1000)
                np.testing.assert_allclose(a.balanced_tensor,1,rtol=1e-9)
                a=balance(triple_only,method=method,snapshots=1000)
                np.testing.assert_allclose(a.balanced_tensor,r,rtol=1e-9)
                np.testing.assert_allclose(a.bias_matrix,1,rtol=1e-9)

    def test_permutation_scaling_and_three_way_contrasts(self):
        order=np.array([4,1,3,0,2]);ix=np.ix_(order,order,order)
        for method in ('dh','log-anova','binomial'):
            with self.subTest(method=method):
                a=balance(self.raw,method=method,snapshots=1000)
                b=balance(self.raw[ix],method=method,snapshots=1000)
                np.testing.assert_allclose(b.balanced_tensor,a.balanced_tensor[ix],rtol=1e-9)
                np.testing.assert_allclose(b.bias_matrix,a.bias_matrix[np.ix_(order,order)],rtol=1e-9)
                np.testing.assert_allclose(b.bias_vector,a.bias_vector[order],rtol=1e-9)
                c=balance((self.raw+.5)*7-.5,method=method,snapshots=10000)
                np.testing.assert_allclose(c.balanced_tensor,a.balanced_tensor,rtol=1e-9)
                np.testing.assert_allclose(c.bias_matrix,a.bias_matrix,rtol=1e-9)
                np.testing.assert_allclose(c.bias_vector,a.bias_vector*np.cbrt(7),rtol=1e-9)
                contrast=lambda x: np.diff(np.diff(np.diff(np.log(x),axis=0),axis=1),axis=2)
                np.testing.assert_allclose(contrast(a.balanced_tensor),contrast(self.raw+.5),atol=2e-13)


if __name__=='__main__':unittest.main()

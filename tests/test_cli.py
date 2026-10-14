import io
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile
import numpy as np

from tensorbalance import balance
from tensorbalance.cli import OUTPUTS
from tensorbalance.decompose import ConvergenceError
from tensorbalance.input import load_input

ROOT = Path(__file__).resolve().parents[1]


def known_factors():
    n = 5
    f = np.linspace(-1, 1, n)
    v = np.exp(0.18 * f + np.log(50) / 3)
    m = np.exp(.24 * f[:, None] * f[None, :])
    r = 1 + .45 * f[:, None, None] * f[None, :, None] * f[None, None, :]
    a = v[:, None, None] * v[None, :, None] * v[None, None, :] * m[:, :, None] * m[:, None, :] * m[None, :, :] * r
    return a - .5, v, m, r


class NumericalTests(unittest.TestCase):
    def test_known_factors_all_methods(self):
        raw, vector, matrix, residual = known_factors()
        before = raw.copy()
        for method in ["dh", "log-anova", "binomial"]:
            with self.subTest(method=method):
                result = balance(raw, method=method, snapshots=1000)
                np.testing.assert_allclose(result.bias_vector, vector, rtol=2e-9)
                np.testing.assert_allclose(result.bias_matrix, matrix, rtol=2e-9)
                np.testing.assert_allclose(result.balanced_tensor, residual, rtol=2e-9)
                self.assertLess(result.diagnostics["reconstruction_relative_error"], 1e-11)
                np.testing.assert_allclose(np.log(result.bias_matrix).mean(0), 0, atol=1e-14)
                for axis in range(3):
                    np.testing.assert_allclose(result.balanced_tensor.sum(axis), len(raw), rtol=1e-10)
                np.testing.assert_array_equal(raw, before)

    def test_uniform_and_zero(self):
        for raw in [np.zeros((3, 3, 3)), np.full((3, 3, 3), 17.), np.ones((1, 1, 1))]:
            r = balance(raw)
            np.testing.assert_allclose(r.balanced_tensor, 1, atol=1e-13)
            np.testing.assert_allclose(r.bias_matrix, 1, atol=1e-13)
            np.testing.assert_allclose(r.bias_vector, np.cbrt(raw.mean() + .5))

    def test_invalid_tensor_and_settings(self):
        raw = known_factors()[0]
        asymmetric = raw.copy(); asymmetric[0, 1, 2] += 1
        invalid = [asymmetric, np.ones((2, 3, 2)), np.empty((0, 0, 0)), np.full((2, 2, 2), -1.),
                   np.full((2, 2, 2), np.nan), np.ones((2, 2, 2), dtype=complex), np.full((2, 2, 2), 2**54, dtype=np.uint64)]
        for value in invalid:
            with self.subTest(shape=value.shape), self.assertRaises(ValueError):
                balance(value)
        for opts in [dict(tolerance=0), dict(tolerance=np.nan), dict(max_iterations=0), dict(method="bad"),
                     dict(method="binomial"), dict(method="binomial", snapshots=1), dict(method="binomial", snapshots=np.inf)]:
            with self.subTest(opts=opts), self.assertRaises(ValueError):
                balance(raw, **opts)
        with self.assertRaises(ConvergenceError):
            balance(raw, max_iterations=1)


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = known_factors()[0]
        self.input = self.root / "raw.npy"
        np.save(self.input, self.raw)

    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(ROOT / "tensor_balance.py"), *map(str, args)],
                              text=True, capture_output=True)

    def test_one_input_three_outputs_and_no_overwrite(self):
        run = self.run_cli(self.input)
        self.assertEqual(run.returncode, 0, run.stderr)
        out = self.root / "raw_balanced"
        self.assertEqual(sorted(p.name for p in out.iterdir()), sorted(OUTPUTS))
        self.assertEqual(len(run.stdout.splitlines()), 3)
        b, m, r = [np.load(out / name) for name in OUTPUTS]
        reconstructed = b[:, None, None] * b[None, :, None] * b[None, None, :] * m[:, :, None] * m[:, None, :] * m[None, :, :] * r - .5
        np.testing.assert_allclose(reconstructed, self.raw, rtol=1e-12)
        before = (out / "bias_vector.npy").read_bytes()
        run = self.run_cli(self.input)
        self.assertEqual(run.returncode, 2)
        self.assertEqual((out / "bias_vector.npy").read_bytes(), before)

    def test_failure_writes_no_results(self):
        out = self.root / "failed"
        run = self.run_cli(self.input, "--out", out, "--max-iterations", "1")
        self.assertEqual(run.returncode, 3, run.stderr)
        self.assertFalse(out.exists())
        run = self.run_cli(self.input, "--out", out, "--method", "binomial")
        self.assertEqual(run.returncode, 2)
        self.assertIn("actual snapshot", run.stderr)
        self.assertFalse(out.exists())

    def test_optional_view_has_exact_outputs(self):
        import re
        out = self.root / "view"
        run = self.run_cli(self.input, "--out", out, "--visualize")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(len(list(out.iterdir())), 4)
        html = (out / "visualization.html").read_text()
        data = json.loads(re.search(r'<script id="tensor-data" type="application/json">(.*?)</script>', html, re.S).group(1))
        np.testing.assert_array_equal(data["tensor"], np.load(out / "balanced_tensor.npy").ravel())
        np.testing.assert_array_equal(data["raw"], self.raw.ravel())
        self.assertNotIn("https://", html)

    def test_input_formats_and_zip(self):
        for key in ["raw", "tensor", "arr_0"]:
            path = self.root / (key + ".npz")
            np.savez_compressed(path, **{key: self.raw, "snapshots": 1000})
            raw, s = load_input(path)
            np.testing.assert_array_equal(raw, self.raw)
            self.assertEqual(s, 1000)
        path = self.root / "raw.json"
        path.write_text(json.dumps(dict(raw=self.raw.ravel().tolist(), shape=list(self.raw.shape), snapshots=1000)))
        np.testing.assert_array_equal(load_input(path)[0], self.raw)
        path = self.root / "raw.csv"
        path.write_text("i,j,k,value\n" + "\n".join(f"{i},{j},{k},{self.raw[i,j,k]:.17g}" for i, j, k in np.ndindex(self.raw.shape)))
        np.testing.assert_array_equal(load_input(path)[0], self.raw)
        path = self.root / "raw.zip"
        with zipfile.ZipFile(path, "w") as z:
            z.write(self.input, "data/raw.npy")
            z.writestr("data/metadata.json", '{"snapshots": 1000}')
        np.testing.assert_array_equal(load_input(path)[0], self.raw)
        self.assertEqual(load_input(path)[1], 1000)
        run = self.run_cli(path, "--method", "binomial", "--out", self.root / "binomial")
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_ambiguous_missing_and_unsafe_arrays(self):
        path = self.root / "ambiguous.npz"
        np.savez(path, a=self.raw, b=self.raw)
        with self.assertRaises(ValueError): load_input(path)
        path = self.root / "missing.csv"
        path.write_text("i,j,k,value\n0,0,0,1\n1,1,1,1\n")
        with self.assertRaises(ValueError): load_input(path)
        np.save(self.input, np.array([{"bad": True}], dtype=object))
        with self.assertRaises(ValueError): load_input(self.input)

    def test_corrupt_archive_is_a_clean_failure(self):
        path = self.root / "broken.zip"
        path.write_bytes(b"PK broken")
        run = self.run_cli(path)
        self.assertEqual(run.returncode, 2)
        self.assertNotIn("Traceback", run.stderr)
        self.assertFalse((self.root / "broken_balanced").exists())


if __name__ == "__main__":
    unittest.main()

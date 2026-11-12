import io
import itertools
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np

from tensorbalance import balance, unit_fiber_result
from tensorbalance.cli import METHODS, OUTPUTS, main
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

    def test_one_input_four_outputs_and_no_overwrite(self):
        run = self.run_cli(self.input)
        self.assertEqual(run.returncode, 0, run.stderr)
        out = self.root / "raw_balanced"
        self.assertEqual(sorted(p.name for p in out.iterdir()), sorted(OUTPUTS))
        self.assertEqual(len(run.stdout.splitlines()), 4)
        b, m, r, scalar = [np.load(out / name) for name in OUTPUTS]
        reconstructed = scalar * b[:, None, None] * b[None, :, None] * b[None, None, :] * m[:, :, None] * m[:, None, :] * m[None, :, :] * r - .5
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

    def test_all_methods_separate_outputs_match_individual_runs(self):
        source = self.root / "raw.npz"
        np.savez(source, raw=self.raw, snapshots=1000)
        original = source.read_bytes()
        out = self.root / "all"
        run = self.run_cli(source, "--method", "all", "--out", out)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual({p.name for p in out.iterdir()}, {*METHODS, "summary.json"})
        self.assertEqual(len(run.stdout.splitlines()), 13)
        self.assertTrue(all(Path(p).is_file() for p in run.stdout.splitlines()))
        summary = json.loads((out / "summary.json").read_text())
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["snapshots"], 1000)
        for method in METHODS:
            with self.subTest(method=method):
                expected = unit_fiber_result(balance(self.raw, method=method, snapshots=1000))
                self.assertEqual({p.name for p in (out / method).iterdir()}, set(OUTPUTS))
                for name in OUTPUTS:
                    np.testing.assert_array_equal(np.load(out / method / name), getattr(expected, name[:-4]))
                b, m, r, c = [np.load(out / method / name) for name in OUTPUTS]
                rebuilt = c * b[:, None, None] * b[None, :, None] * b[None, None, :] * m[:, :, None] * m[:, None, :] * m[None, :, :] * r - .5
                np.testing.assert_allclose(rebuilt, self.raw, rtol=1e-10)
                for axis in range(3):
                    np.testing.assert_allclose(r.sum(axis), 1, rtol=1e-10)
                entry = summary["methods"][method]
                self.assertEqual(entry["status"], "success")
                self.assertEqual(entry["diagnostics"]["fiber_sum_target"], 1)
                self.assertEqual(set(entry["outputs"]), {f"{method}/{name}" for name in OUTPUTS})
        self.assertEqual(source.read_bytes(), original)
        saved = {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()}
        run = self.run_cli(source, "--method", "all", "--out", out)
        self.assertEqual(run.returncode, 2, run.stderr)
        self.assertEqual(saved, {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()})

    def test_all_methods_preflight_before_any_solver_or_output(self):
        out = self.root / "invalid_all"
        cases = [[], ["--snapshots", "0"], ["--snapshots", "nan"], ["--snapshots", "inf"],
                 ["--snapshots", "1"], ["--snapshots", "1000", "--tolerance", "0"],
                 ["--snapshots", "1000", "--max-iterations", "0"]]
        for options in cases:
            with self.subTest(options=options), patch("tensorbalance.cli.balance") as solver:
                with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
                    code = main([str(self.input), "--method", "all", "--out", str(out), *options])
                self.assertEqual(code, 2)
                solver.assert_not_called()
                self.assertFalse(out.exists())
        invalid = self.raw.copy()
        invalid[0, 1, 2] += 1
        np.save(self.input, invalid)
        with patch("tensorbalance.cli.balance") as solver, redirect_stderr(io.StringIO()):
            self.assertEqual(main([str(self.input), "--method", "all", "--snapshots", "1000"]), 2)
            solver.assert_not_called()
        self.assertFalse((self.root / "raw_balanced").exists())

    def test_all_methods_continue_after_each_solver_failure(self):
        for failed in METHODS:
            out = self.root / failed
            inputs = []

            def solver(raw, *, method, **options):
                inputs.append(raw)
                np.testing.assert_array_equal(raw, self.raw)
                self.assertFalse(raw.flags.writeable)
                if method == failed:
                    raise ConvergenceError("Deliberate convergence failure")
                return balance(raw, method=method, **options)

            with self.subTest(failed=failed), patch("tensorbalance.cli.balance", side_effect=solver) as mocked:
                with patch("tensorbalance.cli.load_input", wraps=load_input) as loader:
                    with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
                        code = main([str(self.input), "--method", "all", "--snapshots", "1000", "--out", str(out)])
                    loader.assert_called_once()
                self.assertEqual(code, 3)
                self.assertEqual([call.kwargs["method"] for call in mocked.call_args_list], list(METHODS))
                self.assertTrue(all(raw is inputs[0] for raw in inputs))
                summary = json.loads((out / "summary.json").read_text())
                self.assertEqual(summary["status"], "incomplete")
                self.assertEqual(summary["methods"][failed]["status"], "failed")
                self.assertIn("convergence failure", summary["methods"][failed]["error"])
                self.assertFalse((out / failed).exists())
                for method in set(METHODS) - {failed}:
                    self.assertEqual(summary["methods"][method]["status"], "success")
                    self.assertEqual({p.name for p in (out / method).iterdir()}, set(OUTPUTS))

    def test_all_methods_real_nonconvergence_saves_only_summary(self):
        out = self.root / "failed_all"
        run = self.run_cli(self.input, "--method", "all", "--snapshots", "1000",
                           "--max-iterations", "1", "--out", out)
        self.assertEqual(run.returncode, 3, run.stderr)
        self.assertEqual([p.name for p in out.iterdir()], ["summary.json"])
        summary = json.loads((out / "summary.json").read_text())
        self.assertEqual(summary["status"], "incomplete")
        self.assertTrue(all(entry["status"] == "failed" for entry in summary["methods"].values()))

    def test_all_methods_metadata_views_and_exposure_override(self):
        import re
        source = self.root / "raw.json"
        metadata = dict(genomic_bin_width=25000, genomic_bins=[0, 2, 4, 6, 8])
        source.write_text(json.dumps(dict(raw=self.raw.tolist(), snapshots=1, metadata=metadata)))
        out = self.root / "views"
        run = self.run_cli(source, "--method", "all", "--snapshots", "1000", "--visualize", "--out", out)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(len(run.stdout.splitlines()), 19)
        summary = json.loads((out / "summary.json").read_text())
        self.assertEqual(summary["snapshots"], 1000)
        for method in METHODS:
            self.assertEqual(json.loads((out / method / "metadata.json").read_text()), metadata)
            html = (out / method / "visualization.html").read_text()
            data = json.loads(re.search(r'<script id="tensor-data" type="application/json">(.*?)</script>', html, re.S).group(1))
            np.testing.assert_array_equal(data["raw"], self.raw.ravel())
            np.testing.assert_array_equal(data["tensor"], np.load(out / method / "balanced_tensor.npy").ravel())

    def test_all_methods_interruption_retains_completed_results(self):
        out = self.root / "interrupted"

        def solver(raw, *, method, **options):
            if method == "log-anova":
                raise KeyboardInterrupt
            return balance(raw, method=method, **options)

        with patch("tensorbalance.cli.balance", side_effect=solver):
            with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
                code = main([str(self.input), "--method", "all", "--snapshots", "1000", "--out", str(out)])
        self.assertEqual(code, 130)
        summary = json.loads((out / "summary.json").read_text())
        self.assertEqual(summary["status"], "incomplete")
        self.assertEqual(summary["methods"]["dh"]["status"], "success")
        self.assertEqual(summary["methods"]["log-anova"]["status"], "interrupted")
        self.assertEqual(summary["methods"]["binomial"]["status"], "pending")
        self.assertEqual({p.name for p in out.iterdir()}, {"dh", "summary.json"})

    def test_all_methods_segment_parquet_rejected_before_loading(self):
        out = self.root / "segments_out"
        with patch("tensorbalance.cli.parquet_kind", return_value="segments"):
            with patch("tensorbalance.cli.load_input") as loader, redirect_stderr(io.StringIO()):
                code = main([str(self.root / "segments.parquet"), "--method", "all", "--out", str(out),
                             "--bin-size", "25000", "--chromosome", "chr21", "--unobserved-are-zero"])
            loader.assert_not_called()
        self.assertEqual(code, 2)
        self.assertFalse(out.exists())

    def test_optional_view_has_exact_outputs(self):
        import re
        out = self.root / "view"
        run = self.run_cli(self.input, "--out", out, "--visualize")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(len(list(out.iterdir())), 5)
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

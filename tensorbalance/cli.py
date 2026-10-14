"""Unix command-line entry point: one input file and exactly three array outputs."""
import argparse
import csv
from pathlib import Path
import sys
import time
from zipfile import BadZipFile
import numpy as np
from . import __version__
from .decompose import balance, ConvergenceError
from .input import load_input
from .storage import write_directory

OUTPUTS = ("bias_vector.npy", "bias_matrix.npy", "balanced_tensor.npy")


def write_result(destination, result, raw, visualize=False):
    """Stage complete files; never overwrite a previous result directory."""
    def write(stage):
        for name in OUTPUTS:
            np.save(stage / name, getattr(result, name[:-4]), allow_pickle=False)
        if visualize:
            from .visualize import save_view
            save_view(stage / "visualization.html", raw, result)
    return write_directory(destination, write)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="tensor-balance", description="One symmetric raw contact tensor → bias vector, bias matrix, and remaining tensor.",
        epilog="Defaults match the website: DH, raw + 0.5, full cube, residual mean 1. "
               "Exactly three NPY files are written unless --visualize is requested.")
    parser.add_argument("input", type=Path, help="raw NPY/NPZ/JSON/CSV/TSV or tensor ZIP")
    parser.add_argument("--out", type=Path, help="new output directory (default: INPUT_balanced)")
    parser.add_argument("--method", choices=("dh", "log-anova", "binomial"), default="dh")
    parser.add_argument("--snapshots", type=float, help="actual exposure S for Binomial, if absent from the input")
    parser.add_argument("--visualize", action="store_true", help="also write a self-contained visualization.html of these outputs")
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)
    args = parser.parse_args(argv)
    try:
        output = args.out or args.input.with_name(args.input.stem + "_balanced")
        if output.exists() or output.is_symlink():
            raise ValueError(f"Output path already exists: {output}. Choose another --out directory.")
        started = time.perf_counter()
        raw, snapshots = load_input(args.input)
        if args.snapshots is not None:
            snapshots = args.snapshots
        result = balance(raw, method=args.method, snapshots=snapshots,
                         tolerance=args.tolerance, max_iterations=args.max_iterations)
        destination = write_result(output, result, raw, args.visualize)
        d = result.diagnostics
        print(f"{args.method}: converged in {d['rounds']} iterations ({time.perf_counter() - started:.2f} s)", file=sys.stderr)
        print(f"raw + 0.5; residual mean 1; fiber spread {d['stack_excess']:.3g}; "
              f"reconstruction error {d['reconstruction_relative_error']:.3g}", file=sys.stderr)
        # stdout is a clean list of paths, useful for Unix pipelines.
        for name in OUTPUTS:
            print(destination / name)
        if args.visualize:
            print(destination / "visualization.html")
        return 0
    except ConvergenceError as exc:
        print("tensor-balance: " + str(exc), file=sys.stderr)
        return 3
    except MemoryError:
        print("tensor-balance: Not enough memory for this dense tensor. Use fewer bins or a smaller complete region.", file=sys.stderr)
        return 2
    except (ValueError, OSError, FloatingPointError, KeyError, TypeError, OverflowError, EOFError, BadZipFile, ImportError, csv.Error, RecursionError) as exc:
        print("tensor-balance: " + str(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("tensor-balance: interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

"""Normalize segment records at a chosen bin size or preserve existing tensor bins."""
import argparse
import csv
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import time
from zipfile import BadZipFile
import numpy as np
from . import __version__
from .decompose import balance, ConvergenceError, unit_fiber_result
from .input import load_input, parquet_kind
from .storage import write_directory

OUTPUTS = ("bias_vector.npy", "bias_matrix.npy", "balanced_tensor.npy", "reconstruction_scalar.npy")


def write_result(destination, result, raw, visualize=False, metadata=None):
    """Stage complete files; never overwrite a previous result directory."""
    result = unit_fiber_result(result)
    def write(stage):
        for name in OUTPUTS:
            np.save(stage / name, getattr(result, name[:-4]), allow_pickle=False)
        if metadata:
            (stage / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if visualize:
            from .visualize import save_view
            save_view(stage / "visualization.html", raw, result)
    return write_directory(destination, write)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="tensor-balance", description="One symmetric raw contact tensor → reconstruction scalar, bias vector, bias matrix, and unit-fiber tensor.",
        epilog="Defaults: DH, raw + 0.5, full cube, output fiber sums 1. "
               "Binned input retains its bins; supplied coordinates are saved in metadata.json. "
               "Segment-record Parquet requires --bin-size and --chromosome.")
    parser.add_argument("input", type=Path, help="already-binned tensor or coordinate Parquet, or segment-record Parquet")
    parser.add_argument("--out", type=Path, help="new output directory (default: INPUT_balanced)")
    parser.add_argument("--method", choices=("dh", "log-anova", "binomial"), default="dh")
    parser.add_argument("--snapshots", type=float, help="actual exposure S for Binomial, if absent from the input")
    parser.add_argument("--visualize", action="store_true", help="also write a self-contained visualization.html of these outputs")
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--bin-size", type=int, help="Parquet segment records only: bin width in base pairs, e.g. 25000 or 1000000")
    parser.add_argument("--chromosome", help="Parquet segment records only: chromosome shared by all segments")
    parser.add_argument("--unobserved-are-zero", action="store_true", help="Parquet only: accept zero counts for unobserved combinations")
    parser.add_argument("--view-region", type=int, nargs=2, metavar=("FIRST_BIN", "STOP_BIN"), help="Parquet only: regional preview of the full normalization")
    parser.add_argument("--max-bins", type=int, help="Parquet only: bin-count resource limit (default 2500)")
    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)
    args = parser.parse_args(argv)
    try:
        output = args.out or args.input.with_name(args.input.stem + "_balanced")
        if output.exists() or output.is_symlink():
            raise ValueError(f"Output path already exists: {output}. Choose another --out directory.")
        if args.input.suffix.lower() == ".parquet" and parquet_kind(args.input) == "segments":
            if args.bin_size is None or not args.chromosome:
                raise ValueError("Segment-record Parquet requires --bin-size (e.g. 25000) and --chromosome (e.g. chr21).")
            if args.method != "dh" or args.snapshots is not None:
                raise ValueError("The compact Parquet workflow supports DH without --snapshots.")
            if args.visualize and not args.view_region:
                raise ValueError("For a Parquet preview, supply --view-region FIRST_BIN STOP_BIN.")
            from .parquet_pipeline import run
            args.out = output
            args.max_bins = 2500 if args.max_bins is None else args.max_bins
            with redirect_stdout(sys.stderr):
                run(args)
            print(output.resolve())
            return 0
        if args.bin_size is not None:
            raise ValueError("This input is already binned. Omit --bin-size to preserve its existing resolution.")
        if args.chromosome or args.unobserved_are_zero or args.view_region or args.max_bins is not None:
            raise ValueError("Parquet preparation options apply only to segment-record Parquet. Binned inputs retain their coordinates.")
        started = time.perf_counter()
        metadata = {}
        raw, snapshots = load_input(args.input, metadata=metadata)
        if args.snapshots is not None:
            snapshots = args.snapshots
        result = balance(raw, method=args.method, snapshots=snapshots,
                         tolerance=args.tolerance, max_iterations=args.max_iterations)
        destination = write_result(output, result, raw, args.visualize, metadata=metadata)
        d = result.diagnostics
        print(f"{args.method}: converged in {d['rounds']} iterations ({time.perf_counter() - started:.2f} s)", file=sys.stderr)
        print(f"raw + 0.5; exported fiber sums 1; fiber spread {d['stack_excess']:.3g}; "
              f"reconstruction error {d['reconstruction_relative_error']:.3g}", file=sys.stderr)
        # stdout is a clean list of paths, useful for Unix pipelines.
        for name in OUTPUTS:
            print(destination / name)
        if args.visualize:
            print(destination / "visualization.html")
        if metadata:
            print(destination / "metadata.json")
        return 0
    except (ConvergenceError, RuntimeError) as exc:
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

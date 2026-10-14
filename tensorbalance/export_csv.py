"""Export existing results to readable CSV without rerunning normalization."""
import argparse
import csv
from pathlib import Path
import sys
import numpy as np
from .storage import write_directory
from .input import load_npy

OUTPUTS = ("bias_vector.csv", "bias_matrix.csv", "balanced_tensor.csv")


def export_csv(source, destination):
    source = Path(source)
    b, m, r = (load_npy(source / name) for name in
               ("bias_vector.npy", "bias_matrix.npy", "balanced_tensor.npy"))
    if b.ndim != 1 or b.size == 0 or m.shape != (b.size, b.size) or r.shape != (b.size,) * 3:
        raise ValueError("Result shapes must be (n,), (n,n), and (n,n,n), with n > 0.")
    if any(a.dtype.kind not in "uif" or not np.isfinite(a).all() or (a <= 0).any() for a in (b, m, r)):
        raise ValueError("Result arrays must contain finite, positive, real values.")

    def write(stage):
        with (stage / OUTPUTS[0]).open("w", newline="", encoding="utf-8") as f:
            out = csv.writer(f)
            out.writerow(("i", "value"))
            out.writerows((i, format(float(value), ".17g")) for i, value in enumerate(b))
        with (stage / OUTPUTS[1]).open("w", newline="", encoding="utf-8") as f:
            out = csv.writer(f)
            out.writerow(("i", *range(b.size)))
            for i, row in enumerate(m):
                out.writerow((i, *(format(float(value), ".17g") for value in row)))
        with (stage / OUTPUTS[2]).open("w", newline="", encoding="utf-8") as f:
            out = csv.writer(f)
            out.writerow(("i", "j", "k", "value"))
            for i in range(b.size):
                for j in range(b.size):
                    out.writerows((i, j, k, format(float(value), ".17g")) for k, value in enumerate(r[i, j]))

    return write_directory(destination, write)


def main(argv=None):
    p = argparse.ArgumentParser(description="Export the three result arrays as CSV; the whole tensor goes into one coordinate table.")
    p.add_argument("results", type=Path, help="directory containing the three NPY outputs")
    p.add_argument("--out", type=Path, help="new CSV directory (default: RESULTS_csv)")
    args = p.parse_args(argv)
    try:
        source = args.results.resolve()
        destination = args.out or source.with_name(source.name + "_csv")
        out = export_csv(source, destination)
        for name in OUTPUTS:
            print(out / name)
        return 0
    except (ValueError, OSError, TypeError, EOFError, MemoryError) as exc:
        print(f"tensor-export-csv: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("tensor-export-csv: interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

"""Load complete raw arrays without pickle, executable code, or ZIP extraction."""
from pathlib import Path, PurePosixPath
from decimal import Decimal, InvalidOperation
import csv
import io
import json
import zipfile
import zlib
import numpy as np
from .checks import MAX_EXACT_INTEGER, check_python_integers, positive_integer, scalar

MASK_KEYS = ("mask", "masks", "structural_zeros")


def load_npy(source):
    value = np.load(source, allow_pickle=False)
    if not isinstance(value, np.ndarray):
        if hasattr(value, "close"):
            value.close()
        raise ValueError("NPY input must contain one array, not a disguised ZIP/NPZ container.")
    return value


def _decimal_float(text):
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid numeric value: {text!r}.") from exc
    if not value.is_finite():
        raise ValueError("Input numeric values must be finite.")
    if abs(value) > MAX_EXACT_INTEGER and value == value.to_integral_value():
        raise ValueError("Integer counts above 2^53 cannot be represented exactly in float64.")
    out = float(value)
    if not np.isfinite(out):
        raise ValueError("Input numeric value is outside the finite float64 range.")
    if out == 0 and value != 0:
        raise ValueError("Input numeric value underflows float64.")
    return out


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}.")
        result[key] = value
    return result


def _loads(payload):
    obj = json.loads(payload, object_pairs_hook=_pairs)
    return obj


def _metadata(obj, name):
    if not isinstance(obj, dict):
        raise ValueError(f"{name} must be a JSON object.")
    if any(obj.get(k) is not None for k in MASK_KEYS):
        raise ValueError("Missing-cell masks are unsupported. Supply a full measured cube, including zeros.")
    return obj


def _snapshots(value):
    return None if value is None else scalar(value, "Snapshot count")


def _json(payload):
    # Parse integer and decimal tokens before NumPy can silently round them.
    obj = json.loads(payload, object_pairs_hook=_pairs, parse_float=_decimal_float)
    if isinstance(obj, list):
        check_python_integers(obj)
        return np.asarray(obj), None
    obj = _metadata(obj, "Tensor JSON")
    keys = [k for k in ("raw", "tensor", "values", "data") if k in obj]
    if not keys:
        raise ValueError("JSON needs a raw, tensor, values, or data array.")
    check_python_integers(obj[keys[0]])
    x = np.asarray(obj[keys[0]])
    shape = obj.get("shape")
    if "n" in obj:
        n = positive_integer(obj["n"], "n")
        if shape is None and x.ndim == 1:
            shape = (n,) * 3
    if shape is not None:
        if not isinstance(shape, (list, tuple)) or len(shape) != 3:
            raise ValueError("JSON shape must contain three positive integer dimensions.")
        shape = tuple(positive_integer(d, "Shape dimension") for d in shape)
        if len(set(shape)) != 1:
            raise ValueError("JSON shape must be n x n x n.")
        if "n" in obj and shape != (n,) * 3:
            raise ValueError("JSON n and shape disagree.")
        x = x.reshape(shape, order="C")
    if "n" in obj and x.shape != (n,) * 3:
        raise ValueError("JSON n does not match its tensor dimensions.")
    return x, _snapshots(obj.get("snapshots"))


def _table(text):
    lines = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not lines:
        raise ValueError("The tensor table is empty.")
    delimiter = "\t" if "\t" in lines[0] else ","
    rows = csv.DictReader(lines, delimiter=delimiter, strict=True)
    fields = rows.fieldnames or []
    if len(fields) != len(set(fields)):
        raise ValueError("CSV/TSV column names must be unique.")
    values = [v for v in ("value", "count", "counts", "raw") if v in fields]
    if not all(k in fields for k in ("i", "j", "k")) or len(values) != 1:
        raise ValueError("CSV/TSV needs i,j,k and exactly one value/count/counts/raw column (zero-based).")
    value = values[0]
    cells, n = {}, 0
    for line_number, row in enumerate(rows, start=2):
        if None in row or any(v is None for v in row.values()):
            raise ValueError(f"CSV/TSV row {line_number} has a different number of fields from the header.")
        try:
            idx = tuple(int(row[k]) for k in ("i", "j", "k"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"CSV/TSV row {line_number}: i,j,k must be integers.") from exc
        if min(idx) < 0 or idx in cells:
            raise ValueError("Duplicate or negative coordinate in tensor table.")
        cells[idx] = _decimal_float(row[value])
        n = max(n, max(idx) + 1)
    if not n or len(cells) != n**3:
        raise ValueError("Coordinate table must contain every voxel, including measured zeros. Missing cells are not assumed zero.")
    x = np.empty((n, n, n))
    for idx, value in cells.items():
        x[idx] = value
    return x, None


def _archive_contents(z):
    names = [n for n in z.namelist() if not n.endswith("/") and not n.startswith("__MACOSX/")]
    if len(names) != len(set(names)):
        raise ValueError("Archive contains duplicate member names.")
    base = lambda n: PurePosixPath(n).name.lower()
    if any(PurePosixPath(base(n)).stem in MASK_KEYS for n in names):
        raise ValueError("Missing-cell masks are unsupported.")
    raw = [n for n in names if base(n) == "raw.npy"]
    if not raw:
        raw = [n for n in names if base(n) in ("tensor.npy", "counts.npy")]
    if len(raw) > 1:
        raise ValueError("Archive contains multiple raw tensors. Select one input.")
    if not raw:
        cubes = []
        for name in names:
            if name.lower().endswith(".npy"):
                a = load_npy(io.BytesIO(z.read(name)))
                if a.ndim == 3:
                    cubes.append(name)
        if len(cubes) > 1:
            raise ValueError("Archive contains multiple 3D arrays. Name the intended input raw.npy.")
        raw = cubes
    if raw:
        name = raw[0]
        prefix = name.rsplit("/", 1)[0] + "/" if "/" in name else ""
        meta_name = prefix + "metadata.json"
        metadata = _metadata(_loads(z.read(meta_name)), "metadata.json") if meta_name in names else {}
        snapshots = _snapshots(metadata.get("snapshots"))
        for key in ("snapshots.npy", "exposure.npy", "S.npy"):
            if prefix + key in names:
                snapshots = scalar(load_npy(io.BytesIO(z.read(prefix + key))), "Snapshot count")
                break
        return load_npy(io.BytesIO(z.read(name))), snapshots
    # In the established saved-session format the first n^3 float64 entries
    # are the raw tensor. Extra entries hold the saved session's other arrays.
    sessions = [n for n in names if base(n) == "session.json" and n[:-12] + "arrays.bin" in names]
    if len(sessions) > 1:
        raise ValueError("Archive contains multiple saved sessions. Select one input.")
    if sessions:
        name = sessions[0]
        meta = _metadata(_loads(z.read(name)), "session.json")
        n = positive_integer(meta["n"], "Saved-session n")
        values = np.frombuffer(z.read(name[:-12] + "arrays.bin"), dtype="<f8")
        if len(values) < n**3:
            raise ValueError("Saved raw tensor is truncated.")
        return values[:n**3].reshape((n,) * 3), _snapshots(meta.get("snapshots"))
    text = [n for n in names if base(n) in ("tensor.json", "raw.json", "counts.json")]
    if not text:
        text = [n for n in names if n.lower().endswith((".json", ".csv", ".tsv"))
                and base(n) not in ("metadata.json", "session.json")]
    if len(text) != 1:
        raise ValueError("Archive must contain one raw tensor; none or multiple candidates were found.")
    payload = z.read(text[0]).decode("utf-8-sig")
    return _json(payload) if text[0].lower().endswith(".json") else _table(payload)


def _archive(path):
    try:
        with zipfile.ZipFile(path) as z:
            return _archive_contents(z)
    except (RuntimeError, NotImplementedError, zlib.error) as exc:
        raise ValueError(f"Cannot read tensor archive: {exc}") from exc


def load_input(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return load_npy(path), None
    if suffix in (".npz", ".zip"):
        return _archive(path)
    if suffix == ".json":
        return _json(path.read_text(encoding="utf-8-sig"))
    if suffix in (".csv", ".tsv", ".txt"):
        return _table(path.read_text(encoding="utf-8-sig"))
    raise ValueError("Use NPY, NPZ, JSON, a full i,j,k,value CSV/TSV table, or a tensor ZIP.")

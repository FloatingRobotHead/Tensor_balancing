"""Validation shared by file readers and the public Python API."""
import numbers
import numpy as np

MAX_EXACT_INTEGER = 2**53
SYMMETRY_RTOL = 1e-12


def scalar(value, name, *, positive=True):
    a = np.asarray(value)
    if a.size != 1 or a.dtype.kind not in "uif":
        raise ValueError(f"{name} must be one real numeric scalar (not a boolean or complex value).")
    if a.dtype.kind in "ui" and abs(int(a.reshape(-1)[0])) > MAX_EXACT_INTEGER:
        raise ValueError(f"{name}: integers above 2^53 cannot be represented exactly in float64.")
    result = float(a.reshape(-1)[0])
    if not np.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be positive and finite.")
    return result


def positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def check_python_integers(value):
    """Check before NumPy can coerce mixed integer/float lists to float64."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Boolean values are not raw contact counts.")
    if isinstance(value, numbers.Integral):
        if abs(value) > MAX_EXACT_INTEGER:
            raise ValueError("Integer counts above 2^53 cannot be represented exactly in float64.")
    elif isinstance(value, (list, tuple)):
        for item in value:
            check_python_integers(item)

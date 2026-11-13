"""Fixed protection for zero counts, shared by all balancing methods."""
import numpy as np

PSEUDOCOUNT = 1e-12


def protect_zeros(raw, pseudocount=PSEUDOCOUNT):
    """Replace exact zeros only; keep every positive observation unchanged."""
    return np.where(np.asarray(raw) == 0, pseudocount, raw)

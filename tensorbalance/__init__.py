"""One raw tensor in; one bias vector, bias matrix, and residual tensor out."""
__version__ = "0.1.1"

from .decompose import balance, Result

__all__ = ["balance", "Result", "__version__"]

"""Optional visualization only; never recalculates or changes the CLI output."""
import json
from pathlib import Path
import numpy as np


def save_view(path, raw, result):
    n = len(raw)
    payload = dict(n=n, raw=np.asarray(raw).ravel().tolist(), vector=result.bias_vector.tolist(),
                   matrix=result.bias_matrix.ravel().tolist(), tensor=result.balanced_tensor.ravel().tolist(),
                   diagnostics=result.diagnostics)
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("view.html").read_text(encoding="utf-8")
    Path(path).write_text(template.replace("/*DATA*/", encoded), encoding="utf-8")

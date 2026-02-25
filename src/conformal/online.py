import numpy as np
from .utils import _as_1d

def online_conformal_point(y_stream, yhat_stream, alpha=0.1, step=0.01, q0=None):
    """
    Online conformal for point forecasts over a stream.
    Produces intervals sequentially and updates radius q_t based on coverage error.

    Update:
      e_t = 1{ |y_t - yhat_t| <= q_t } - (1 - alpha)
      q_{t+1} = max(0, q_t + step * e_t)

    Returns dict with lo, hi, q_series, coverage.
    """
    y = _as_1d(y_stream, "y_stream")
    yhat = _as_1d(yhat_stream, "yhat_stream")
    if len(y) != len(yhat):
        raise ValueError("y_stream and yhat_stream must have same length")

    n = len(y)
    q = float(np.median(np.abs(y - yhat)) if q0 is None else q0)

    lo = np.empty(n, dtype=float)
    hi = np.empty(n, dtype=float)
    q_series = np.empty(n, dtype=float)
    covered = np.empty(n, dtype=bool)

    target = 1.0 - alpha

    for t in range(n):
        lo[t] = yhat[t] - q
        hi[t] = yhat[t] + q
        err = abs(y[t] - yhat[t])
        covered[t] = (err <= q)
        q_series[t] = q

        e_t = (1.0 if covered[t] else 0.0) - target
        q = max(0.0, q + step * e_t)

    return {
        "lo": lo,
        "hi": hi,
        "q_series": q_series,
        "coverage": float(np.mean(covered)),
        "avg_width": float(np.mean(hi - lo)),
    }
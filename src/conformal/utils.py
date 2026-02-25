import numpy as np

def _as_1d(x, name="array"):
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {x.shape}")
    return x

def weighted_quantile(values, quantile, sample_weight=None):
    """
    Weighted quantile of 1D values.
    quantile in [0,1].
    """
    v = _as_1d(values, "values").astype(float)
    q = float(quantile)
    if not (0.0 <= q <= 1.0):
        raise ValueError("quantile must be in [0,1]")

    if sample_weight is None:
        return float(np.quantile(v, q, method="higher"))

    w = _as_1d(sample_weight, "sample_weight").astype(float)
    if len(w) != len(v):
        raise ValueError("sample_weight must match values length")
    if np.any(w < 0):
        raise ValueError("sample_weight must be nonnegative")

    if np.all(w == 0):
        # fallback: unweighted
        return float(np.quantile(v, q, method="higher"))

    idx = np.argsort(v)
    v_sorted = v[idx]
    w_sorted = w[idx]
    cw = np.cumsum(w_sorted)
    cw /= cw[-1]
    # first index where cumulative weight >= q
    j = int(np.searchsorted(cw, q, side="left"))
    j = min(max(j, 0), len(v_sorted) - 1)
    return float(v_sorted[j])

def coverage_and_width(y, lo, hi):
    y = _as_1d(y, "y")
    lo = _as_1d(lo, "lo")
    hi = _as_1d(hi, "hi")
    if not (len(y) == len(lo) == len(hi)):
        raise ValueError("y, lo, hi must have same length")
    covered = (y >= lo) & (y <= hi)
    width = hi - lo
    return {
        "n": int(len(y)),
        "coverage": float(np.mean(covered)),
        "avg_width": float(np.mean(width)),
        "median_width": float(np.median(width)),
    }

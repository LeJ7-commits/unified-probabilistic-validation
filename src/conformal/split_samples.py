import numpy as np
from .utils import _as_1d, weighted_quantile

def _sample_quantiles(samples, q_lo, q_hi):
    s = np.asarray(samples)
    if s.ndim != 2:
        raise ValueError(f"samples must be 2D, got {s.shape}")
    lo = np.quantile(s, q_lo, axis=1)
    hi = np.quantile(s, q_hi, axis=1)
    return lo, hi

def split_conformal_interval_samples(y_cal, samples_cal, samples_test, alpha=0.1, w_cal=None):
    """
    Split conformal interval using sample-based quantiles.
    Base interval: [q_{alpha/2}, q_{1-alpha/2}] from samples.
    Conformal score: s_i = max(lo_i - y_i, y_i - hi_i, 0).
    Final interval: [lo - q, hi + q] where q = quantile_{1-alpha}(scores).
    """
    y_cal = _as_1d(y_cal, "y_cal")
    s_cal = np.asarray(samples_cal)
    s_test = np.asarray(samples_test)

    if s_cal.ndim != 2 or s_test.ndim != 2:
        raise ValueError("samples_cal and samples_test must be 2D arrays (n, m)")
    if len(y_cal) != s_cal.shape[0]:
        raise ValueError("y_cal length must equal samples_cal first dimension")

    q_lo = alpha / 2.0
    q_hi = 1.0 - alpha / 2.0

    lo_cal, hi_cal = _sample_quantiles(s_cal, q_lo, q_hi)
    # CQR-style nonconformity
    scores = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
    scores = np.maximum(scores, 0.0)

    q = weighted_quantile(scores, 1 - alpha, sample_weight=w_cal)

    lo_test, hi_test = _sample_quantiles(s_test, q_lo, q_hi)
    lo = lo_test - q
    hi = hi_test + q
    return lo, hi, q, (lo_test, hi_test)
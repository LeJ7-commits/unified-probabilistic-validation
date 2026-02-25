import numpy as np
from .utils import _as_1d, weighted_quantile

def split_conformal_interval_point(y_cal, yhat_cal, yhat_test, alpha=0.1, w_cal=None):
    """
    Split conformal for point forecasts using absolute residuals.
    Returns (lo, hi, radius).
    """
    y_cal = _as_1d(y_cal, "y_cal")
    yhat_cal = _as_1d(yhat_cal, "yhat_cal")
    yhat_test = _as_1d(yhat_test, "yhat_test")
    if len(y_cal) != len(yhat_cal):
        raise ValueError("y_cal and yhat_cal must have same length")

    scores = np.abs(y_cal - yhat_cal)
    # Conformal uses quantile at (1 - alpha) with "higher" behavior
    q = weighted_quantile(scores, 1 - alpha, sample_weight=w_cal)
    lo = yhat_test - q
    hi = yhat_test + q
    return lo, hi, q
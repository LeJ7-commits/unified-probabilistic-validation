from __future__ import annotations
import numpy as np

from src.calibration.pit import pit_gof_tests, pit_independence_tests


def pit_uniformity_tests(u: np.ndarray) -> dict:
    """
    Wrapper for PIT goodness-of-fit tests (KS, CvM, AD-stat via z-transform).
    """
    return pit_gof_tests(u)


def pit_autocorrelation_tests(u: np.ndarray, lags: int | list[int] = 20) -> dict:
    """
    Wrapper for PIT independence tests (Ljung-Box on z=Phi^{-1}(u)).
    """
    return pit_independence_tests(u, lags=lags, use_inverse_normal=True)


def interval_coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """
    Compute empirical coverage of prediction intervals.
    """
    y = np.asarray(y, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return float(np.mean((y >= lower) & (y <= upper)))

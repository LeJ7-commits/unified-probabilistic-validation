# src/calibration/pit.py

import numpy as np
from scipy.stats import uniform
from scipy.stats import kstest
from scipy.stats import norm


def compute_pit(cdf_values: np.ndarray) -> np.ndarray:
    """
    compute PIT values u_t = F_t(y_t)

    parameters
    ----------
    cdf_values : np.ndarray
        Evaluated predictive CDF at realized value y_t.

    returns
    -------
    np.ndarray
        PIT values in (0,1)
    """
    eps = 1e-12
    return np.clip(cdf_values, eps, 1 - eps)


def pit_uniformity_test(pit_values: np.ndarray):
    """
    Kolmogorov-Smirnov test against Uniform(0,1)
    """
    stat, p_value = kstest(pit_values, 'uniform')
    return {
        "ks_statistic": stat,
        "p_value": p_value
    }


def pit_autocorrelation_test(pit_values: np.ndarray, lags: int = 10):
    """
    Ljung-Box autocorrelation check
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    result = acorr_ljungbox(pit_values, lags=[lags], return_df=True)
    return result.to_dict()

# src/calibration/pit.py

import numpy as np
from scipy.stats import uniform
from scipy.stats import kstest
from scipy.stats import norm



def compute_pit(y_true, samples):
    """
    compute Probability Integral Transform (PIT)
    from predictive samples.

    Parameters
    ----------
    y_true : array-like, shape (n_obs,)
    samples : array-like, shape (n_obs, n_samples)

    Returns
    -------
    u : ndarray, shape (n_obs,)
        PIT values in [0,1]
    """
    y_true = np.asarray(y_true)
    samples = np.asarray(samples)

    n_obs, n_samples = samples.shape

    # empirical CDF per observation
    u = np.mean(samples <= y_true[:, None], axis=1)

    return u


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

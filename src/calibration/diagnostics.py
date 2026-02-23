import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import kstest

def pit_autocorrelation(u, nlags=20):
    """
    Compute Ljung–Box test for PIT independence.
    Returns test statistic and p-value.
    """
    lb = acorr_ljungbox(u, lags=[nlags], return_df=True)
    return {
        "ljung_box_stat": float(lb["lb_stat"].iloc[0]),
        "p_value": float(lb["lb_pvalue"].iloc[0])
    }


def interval_coverage(y, lower, upper):
    """
    Compute empirical coverage of prediction intervals.
    """
    return float(np.mean((y >= lower) & (y <= upper)))

def pit_uniformity_ks(u):
    """
    KS test for Uniform(0,1).
    """
    stat, p = kstest(u, "uniform")
    return {"ks_statistic": float(stat), "p_value": float(p)}

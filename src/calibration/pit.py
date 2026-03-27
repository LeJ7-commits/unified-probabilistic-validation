from __future__ import annotations

import numpy as np
from scipy.stats import kstest, cramervonmises, norm
from statsmodels.stats.diagnostic import acorr_ljungbox


def compute_pit(y_true, samples) -> np.ndarray:
    """
    Compute Probability Integral Transform (PIT) values u_t in [0,1]
    from predictive samples via the empirical CDF.

    Parameters
    ----------
    y_true : array-like, shape (n_obs,)
    samples : array-like, shape (n_obs, n_samples)

    Returns
    -------
    u : ndarray, shape (n_obs,)
        PIT values in [0,1]
    """
    y_true = np.asarray(y_true, dtype=float)
    samples = np.asarray(samples, dtype=float)

    if samples.ndim != 2:
        raise ValueError("samples must have shape (n_obs, n_samples)")

    n_obs, _ = samples.shape
    if y_true.shape[0] != n_obs:
        raise ValueError("y_true and samples must have same n_obs")

    # empirical CDF per observation
    u = np.mean(samples <= y_true[:, None], axis=1)
    # numerical guard (avoid exactly 0/1 for inverse CDF)
    u = np.clip(u, 1e-12, 1 - 1e-12)
    return u


def pit_inverse_normal(pit_values: np.ndarray) -> np.ndarray:
    """
    Inverse-CDF transformation z_t = Phi^{-1}(u_t), recommended before
    autocorrelation testing on PIT values.
    """
    u = np.asarray(pit_values, dtype=float)
    u = np.clip(u, 1e-12, 1 - 1e-12)
    return norm.ppf(u)


def pit_gof_tests(pit_values: np.ndarray) -> dict:
    """
    Uniformity / goodness-of-fit tests for PIT against Uniform(0,1).

    Includes:
    - Kolmogorov-Smirnov (KS)
    - Cramér–von Mises (CvM)
    - Anderson–Darling style statistic:
        scipy does not provide AD for Uniform directly, so we transform:
        z = Phi^{-1}(u) should be N(0,1) if u ~ Uniform(0,1),
        then apply scipy.stats.anderson(z, dist='norm').

    Returns
    -------
    dict with test statistics and p-values where available.
    """
    u = np.asarray(pit_values, dtype=float)
    u = np.clip(u, 1e-12, 1 - 1e-12)

    # KS test vs Uniform(0,1)
    ks_stat, ks_p = kstest(u, "uniform")

    # CvM test vs Uniform(0,1) (has p-value in scipy)
    cvm_res = cramervonmises(u, "uniform")


    # AD for normality after inverse-normal transform.
    # scipy >= 1.17 requires method parameter; method='auto' returns a pvalue.
    # pit_ad_critvals and pit_ad_siglevels are kept as empty lists for
    # backwards compatibility — use pit_ad_pvalue for hypothesis testing.
    from scipy.stats import anderson  # local import

    z = pit_inverse_normal(u)

    ad_res = anderson(z, dist="norm", method="interpolate")

    return {
        "pit_ks_stat": float(ks_stat),
        "pit_ks_pvalue": float(ks_p),
        "pit_cvm_stat": float(cvm_res.statistic),
        "pit_cvm_pvalue": float(cvm_res.pvalue),
        "pit_ad_stat": float(ad_res.statistic),
        "pit_ad_pvalue": float(ad_res.pvalue),
    }

def pit_independence_tests(
    pit_values: np.ndarray,
    lags: int | list[int] = 20,
    use_inverse_normal: bool = True,
) -> dict:
    """
    Independence / autocorrelation check using Ljung-Box.

    IMPORTANT (per Rikard/enBW comment):
    - Perform inverse CDF transformation first (z = Phi^{-1}(u)).
    - Then run autocorrelation tests on z_t.

    Parameters
    ----------
    pit_values : ndarray, shape (n_obs,)
    lags : int or list[int]
        Ljung-Box lags. If int, evaluates at that lag.
        If list, returns dict for all provided lags.
    use_inverse_normal : bool
        If True (default), test is applied to z_t = Phi^{-1}(u_t).
        If False, test is applied directly to u_t (not recommended).

    Returns
    -------
    dict with Ljung-Box statistics/p-values.
    """
    u = np.asarray(pit_values, dtype=float)
    u = np.clip(u, 1e-12, 1 - 1e-12)

    x = pit_inverse_normal(u) if use_inverse_normal else u

    if isinstance(lags, int):
        lag_list = [lags]
    else:
        lag_list = list(lags)

    df = acorr_ljungbox(x, lags=lag_list, return_df=True)

    out = {}
    for lag in lag_list:
        out[f"pit_lb_stat_lag{lag}"] = float(df.loc[lag, "lb_stat"])
        out[f"pit_lb_pvalue_lag{lag}"] = float(df.loc[lag, "lb_pvalue"])

    out["pit_lb_input"] = "z=Phi^{-1}(u)" if use_inverse_normal else "u"

    # ACF at lag 1 — effect size for large-n Ljung-Box guard
    acf_lag1 = float(np.corrcoef(x[:-1], x[1:])[0, 1]) if len(x) > 1 else 0.0
    out["pit_acf_lag1"] = acf_lag1

    return out
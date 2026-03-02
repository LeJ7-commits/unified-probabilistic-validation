from __future__ import annotations

import numpy as np
from scipy.stats import binom, binomtest


def _traffic_light_from_exceedances(k: int, n: int, alpha: float) -> dict:
    """
    Basel/Anfuso-style traffic light classification via binomial quantiles.
    If the model is correct, exceedances ~ Binomial(n, alpha).

    We define:
      GREEN  if k <= q_95
      YELLOW if q_95 < k <= q_99_9
      RED    if k > q_99_9

    These cutoffs are tunable; this is a sensible default for governance.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    q95 = int(binom.ppf(0.95, n, alpha))
    q999 = int(binom.ppf(0.999, n, alpha))

    if k <= q95:
        label = "GREEN"
    elif k <= q999:
        label = "YELLOW"
    else:
        label = "RED"

    return {
        "traffic_light": label,
        "green_cutoff_q95": q95,
        "yellow_cutoff_q999": q999,
    }


def anfuso_var_backtest(y, var_quantile, alpha: float) -> dict:
    """
    One-sided exceedance backtest (VaR-style).

    Example:
      var_quantile = q_{1-alpha}(t) for upper-tail risk
      exceedance = y_t > var_quantile_t

    Parameters
    ----------
    y : array-like (n,)
    var_quantile : array-like (n,)
    alpha : exceedance probability (e.g., 0.01, 0.05, 0.10)

    Returns
    -------
    dict with exceedance count, rate, binomial test p-value, traffic light zone.
    """
    y = np.asarray(y, dtype=float)
    q = np.asarray(var_quantile, dtype=float)
    if y.shape != q.shape:
        raise ValueError("y and var_quantile must have same shape")

    exceed = (y > q)
    k = int(exceed.sum())
    n = int(len(y))

    # Binomial test: H0: exceedance rate == alpha
    bt = binomtest(k, n, alpha, alternative="greater")

    zone = _traffic_light_from_exceedances(k, n, alpha)

    return {
        "n": n,
        "alpha": float(alpha),
        "exceedances": k,
        "exceedance_rate": float(k / n),
        "binom_pvalue_greater": float(bt.pvalue),
        **zone,
    }


def anfuso_interval_backtest(y, lower, upper, alpha: float) -> dict:
    """
    Two-sided interval backtest ("accrue both sides").

    For a central (1-alpha) prediction interval [lower_t, upper_t],
    we test lower-tail breaches and upper-tail breaches separately:

      lower breach: y_t < lower_t   expected rate alpha/2
      upper breach: y_t > upper_t   expected rate alpha/2

    And also total breaches outside interval expected rate alpha.

    Returns traffic-light zones for:
      - total breaches (alpha)
      - lower breaches (alpha/2)
      - upper breaches (alpha/2)
    """
    y = np.asarray(y, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)

    if not (y.shape == lo.shape == hi.shape):
        raise ValueError("y, lower, upper must have same shape")

    n = int(len(y))

    lower_breach = (y < lo)
    upper_breach = (y > hi)
    total_breach = lower_breach | upper_breach

    k_lo = int(lower_breach.sum())
    k_hi = int(upper_breach.sum())
    k_tot = int(total_breach.sum())

    # Binomial tests
    bt_lo = binomtest(k_lo, n, alpha / 2, alternative="greater")
    bt_hi = binomtest(k_hi, n, alpha / 2, alternative="greater")
    bt_tot = binomtest(k_tot, n, alpha, alternative="greater")

    zone_lo = _traffic_light_from_exceedances(k_lo, n, alpha / 2)
    zone_hi = _traffic_light_from_exceedances(k_hi, n, alpha / 2)
    zone_tot = _traffic_light_from_exceedances(k_tot, n, alpha)

    return {
        "n": n,
        "alpha": float(alpha),

        "lower_breaches": k_lo,
        "upper_breaches": k_hi,
        "total_breaches": k_tot,

        "lower_breach_rate": float(k_lo / n),
        "upper_breach_rate": float(k_hi / n),
        "total_breach_rate": float(k_tot / n),

        "binom_pvalue_lower_greater": float(bt_lo.pvalue),
        "binom_pvalue_upper_greater": float(bt_hi.pvalue),
        "binom_pvalue_total_greater": float(bt_tot.pvalue),

        "traffic_light_lower": zone_lo["traffic_light"],
        "traffic_light_upper": zone_hi["traffic_light"],
        "traffic_light_total": zone_tot["traffic_light"],

        "cutoffs_lower_q95": zone_lo["green_cutoff_q95"],
        "cutoffs_lower_q999": zone_lo["yellow_cutoff_q999"],
        "cutoffs_upper_q95": zone_hi["green_cutoff_q95"],
        "cutoffs_upper_q999": zone_hi["yellow_cutoff_q999"],
        "cutoffs_total_q95": zone_tot["green_cutoff_q95"],
        "cutoffs_total_q999": zone_tot["yellow_cutoff_q999"],
    }
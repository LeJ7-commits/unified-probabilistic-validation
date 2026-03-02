from __future__ import annotations

import numpy as np

from src.calibration.pit import compute_pit
from src.calibration.diagnostics import (
    pit_uniformity_tests,
    pit_autocorrelation_tests,
    interval_coverage,
)
from src.scoring.crps import crps_sample


def evaluate_distribution(
    y_true,
    samples=None,
    quantiles=None,
    alpha: float = 0.1,
    lb_lags: int | list[int] = 20,
) -> dict:
    """
    Unified evaluation for probabilistic forecasts.

    Parameters
    ----------
    y_true : (n_obs,)
    samples : (n_obs, n_samples) optional
    quantiles : dict {p: array(n_obs,)} optional
    alpha : miscoverage level for central interval (0.1 -> 90% interval)
    lb_lags : int or list[int]
        Ljung-Box lags used for independence testing on PIT z-transform.

    Returns
    -------
    dict of evaluation metrics.
    """
    y_true = np.asarray(y_true, dtype=float)
    results: dict = {}

    # PIT + GOF + independence + CRPS (needs samples)
    if samples is not None:
        samples = np.asarray(samples, dtype=float)
        u = compute_pit(y_true, samples)

        results.update(pit_uniformity_tests(u))
        results.update(pit_autocorrelation_tests(u, lags=lb_lags))

        # CRPS mean (sample-based)
        results["crps_mean"] = float(
            np.mean([crps_sample(samples[i], float(y_true[i])) for i in range(len(y_true))])
        )

    # Empirical interval coverage (needs quantiles)
    if quantiles is not None:
        if (alpha / 2) not in quantiles or (1 - alpha / 2) not in quantiles:
            raise KeyError(
                f"quantiles must include keys {alpha/2} and {1-alpha/2} for central interval coverage"
            )
        lower = np.asarray(quantiles[alpha / 2], dtype=float)
        upper = np.asarray(quantiles[1 - alpha / 2], dtype=float)
        results["empirical_coverage"] = interval_coverage(y_true, lower, upper)

    return results
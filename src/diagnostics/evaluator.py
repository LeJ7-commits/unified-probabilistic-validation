import numpy as np

from src.calibration.pit import compute_pit
from src.calibration.diagnostics import (
    pit_uniformity_ks,
    pit_autocorrelation,
    interval_coverage,
)
from src.scoring.crps import crps_sample


def evaluate_distribution(y_true, samples=None, quantiles=None, alpha=0.1):
    """
    Unified evaluation for probabilistic forecasts.

    y_true: (n_obs,)
    samples: (n_obs, n_samples) optional
    quantiles: dict {p: array(n_obs,)} optional
    alpha: miscoverage level for central interval (0.1 -> 90% interval)
    """
    y_true = np.asarray(y_true)
    results = {}

    # PIT + independence + CRPS (needs samples)
    if samples is not None:
        samples = np.asarray(samples)
        u = compute_pit(y_true, samples)

        results.update(pit_uniformity_ks(u))
        results.update(pit_autocorrelation(u))

        results["crps_mean"] = float(
        np.mean([crps_sample(samples[i], float(y_true[i])) for i in range(len(y_true))])
        )

    # Empirical interval coverage (needs quantiles)
    if quantiles is not None:
        lower = np.asarray(quantiles[alpha / 2])
        upper = np.asarray(quantiles[1 - alpha / 2])
        results["empirical_coverage"] = interval_coverage(y_true, lower, upper)

    return results
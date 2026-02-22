# src/scoring/crps.py

import numpy as np


def crps_sample(forecast_samples: np.ndarray, y_true: float) -> float:
    """
    compute CRPS using sample-based approximation.

    parameters
    ----------
    forecast_samples : np.ndarray
        Samples from predictive distribution
    y_true : float
        Observed value

    returns
    -------
    float
    """
    n = len(forecast_samples)

    term1 = np.mean(np.abs(forecast_samples - y_true))
    term2 = 0.5 * np.mean(
        np.abs(forecast_samples[:, None] - forecast_samples[None, :])
    )

    return term1 - term2
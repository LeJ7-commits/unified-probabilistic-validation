from __future__ import annotations

import numpy as np
import pandas as pd

from src.diagnostics.evaluator import evaluate_distribution


def rolling_evaluation(
    y_true,
    samples,
    window: int = 250,
    step: int | None = None,
    mode: str = "overlapping",
    lb_lags: int | list[int] = 20,
) -> pd.DataFrame:
    """
    Rolling window evaluation.

    Parameters
    ----------
    y_true : (n_obs,)
    samples : (n_obs, n_samples)
    window : int
        window length
    step : int or None
        if None, uses:
          - overlapping: step=1
          - non_overlapping: step=window
    mode : {"overlapping", "non_overlapping"}
    lb_lags : Ljung-Box lags passed to evaluate_distribution

    Returns
    -------
    DataFrame with one row per window.
    """
    y_true = np.asarray(y_true, dtype=float)
    samples = np.asarray(samples, dtype=float)

    if mode not in {"overlapping", "non_overlapping"}:
        raise ValueError("mode must be 'overlapping' or 'non_overlapping'")

    if step is None:
        step = 1 if mode == "overlapping" else window

    rows = []
    n = len(y_true)

    for start in range(0, n - window + 1, step):
        end = start + window
        metrics = evaluate_distribution(
            y_true[start:end],
            samples=samples[start:end],
            lb_lags=lb_lags,
        )
        metrics["window_start"] = int(start)
        metrics["window_end"] = int(end)
        metrics["window_len"] = int(window)
        metrics["window_step"] = int(step)
        metrics["window_mode"] = mode
        rows.append(metrics)

    return pd.DataFrame(rows)
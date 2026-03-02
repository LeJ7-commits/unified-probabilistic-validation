from __future__ import annotations

import numpy as np
import pandas as pd
from src.diagnostics.evaluator import evaluate_distribution

def rolling_evaluation(
    y_true,
    samples=None,
    quantiles: dict[float, np.ndarray] | None = None,
    window: int = 250,
    step: int | None = None,
    mode: str = "overlapping",
    lb_lags: int | list[int] = 20,
) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=float)
    samples_arr = None if samples is None else np.asarray(samples, dtype=float)

    if mode not in {"overlapping", "non_overlapping"}:
        raise ValueError("mode must be 'overlapping' or 'non_overlapping'")

    if step is None:
        step = 1 if mode == "overlapping" else window

    n = len(y_true)
    if window <= 0 or window > n:
        raise ValueError(f"window must be in [1, n]. Got window={window}, n={n}")

    # basic input availability check
    if samples_arr is None and quantiles is None:
        raise ValueError("rolling_evaluation requires either `samples` or `quantiles` (refusing dummy zeros).")

    rows = []
    for start in range(0, n - window + 1, step):
        end = start + window

        q_win = None
        if quantiles is not None:
            q_win = {q: np.asarray(arr, dtype=float)[start:end] for q, arr in quantiles.items()}

        s_win = None
        if samples_arr is not None:
            s_win = samples_arr[start:end]

        metrics = evaluate_distribution(
            y_true=y_true[start:end],
            samples=s_win,
            quantiles=q_win,
            lb_lags=lb_lags,
        )
        metrics["window_start"] = int(start)
        metrics["window_end"] = int(end)
        metrics["window_len"] = int(window)
        metrics["window_step"] = int(step)
        metrics["window_mode"] = mode
        rows.append(metrics)

    return pd.DataFrame(rows)
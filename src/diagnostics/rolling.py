import numpy as np
import pandas as pd

from src.diagnostics.evaluator import evaluate_distribution


def rolling_evaluation(y_true, samples, window=250, step=50):
    """
    Fixed-length rolling window evaluation.

    Returns a DataFrame with one row per window.
    """
    y_true = np.asarray(y_true)
    samples = np.asarray(samples)

    rows = []
    n = len(y_true)

    for start in range(0, n - window + 1, step):
        end = start + window
        metrics = evaluate_distribution(y_true[start:end], samples=samples[start:end])
        metrics["window_start"] = start
        metrics["window_end"] = end
        rows.append(metrics)

    return pd.DataFrame(rows)
from src.diagnostics.evaluator import evaluate_distribution
from src.diagnostics.rolling import rolling_evaluation


def run_diagnostics_policy(
    *,
    model_class: str,  # "short_term" | "long_term"
    y_true,
    samples,
    rolling_window=250,
    rolling_step=50,
    enable_rolling_for_long_term=False,
):
    """
    2026-02-24 email exchange
      - Always run full-period diagnostics.
      - Rolling windows are complementary; default ON for short-term.
      - For long-term renewables, default OFF unless explicitly enabled.
    """
    out = {"full_sample": evaluate_distribution(y_true, samples=samples), "rolling": None}

    if model_class == "short_term" or (model_class == "long_term" and enable_rolling_for_long_term):
        out["rolling"] = rolling_evaluation(
            y_true=y_true,
            samples=samples,
            window=rolling_window,
            step=rolling_step,
        )

    return out
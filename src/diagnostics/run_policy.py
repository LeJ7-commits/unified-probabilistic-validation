from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.diagnostics.evaluator import evaluate_distribution
from src.diagnostics.rolling import rolling_evaluation
from src.governance.risk_classification import RiskPolicy, classify_risk
from src.governance.anfuso import anfuso_interval_backtest


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _to_jsonable(x: Any) -> Any:
    import numpy as _np
    if isinstance(x, (_np.integer, _np.floating)):
        return x.item()
    if isinstance(x, _np.ndarray):
        return x.tolist()
    return x


def run_diagnostics_policy(
    *,
    model_class: str,  # "short_term" | "long_term"
    y_true,
    samples=None,
    quantiles: dict[float, np.ndarray] | None = None,
    alpha: float = 0.1,
    rolling_window: int = 250,
    rolling_step: int = 50,
    enable_rolling_for_long_term: bool = False,
    lb_lags: int | list[int] = (5, 10, 20),
    coverage_target: float | None = None,
) -> dict:
    """
    Policy logic (per 2026-02-24 exchange):
      - Always run full-period diagnostics.
      - Rolling windows are complementary; default ON for short-term.
      - For long-term renewables, default OFF unless explicitly enabled.
    """
    y_true = np.asarray(y_true, dtype=float)
    samples_arr = None if samples is None else np.asarray(samples, dtype=float)

    # Full-sample diagnostics can use samples and/or quantiles (evaluator decides what it can compute)
    out: dict[str, Any] = {
        "full_sample": evaluate_distribution(
            y_true=y_true,
            samples=samples_arr,
            quantiles=quantiles,
            alpha=alpha,
            lb_lags=list(lb_lags) if isinstance(lb_lags, tuple) else lb_lags,
        ),
        "rolling_overlapping": None,
        "rolling_non_overlapping": None,
    }

    # governance label for full sample
    policy = RiskPolicy(coverage_target=coverage_target)
    out["full_sample_governance"] = classify_risk(out["full_sample"], policy=policy)

    do_rolling = (model_class == "short_term") or (
        model_class == "long_term" and enable_rolling_for_long_term
    )
    if do_rolling:
        # IMPORTANT: do NOT fabricate dummy samples. Rolling requires either real samples or quantiles.
        if samples_arr is None and quantiles is None:
            raise ValueError(
                "Rolling diagnostics require either `samples` or `quantiles` (refusing dummy zeros)."
            )

        # Overlapping rolling
        out["rolling_overlapping"] = rolling_evaluation(
            y_true=y_true,
            samples=samples_arr,     # may be None
            quantiles=quantiles,     # may be None
            window=rolling_window,
            step=rolling_step,
            mode="overlapping",
            lb_lags=list(lb_lags) if isinstance(lb_lags, tuple) else lb_lags,
        )

        # Non-overlapping rolling
        out["rolling_non_overlapping"] = rolling_evaluation(
            y_true=y_true,
            samples=samples_arr,     # may be None
            quantiles=quantiles,     # may be None
            window=rolling_window,
            step=None,               # rolling_evaluation will map this to step=window for non_overlapping
            mode="non_overlapping",
            lb_lags=list(lb_lags) if isinstance(lb_lags, tuple) else lb_lags,
        )

    return out


def write_run_artifacts(
    *,
    out_dir: str | Path,
    run_output: dict,
    alpha: float = 0.1,
    y_true=None,
    quantiles: dict[float, np.ndarray] | None = None,
    coverage_target: float | None = None,
) -> dict[str, str]:
    """
    Writes artifacts (JSON/CSV) from run_diagnostics_policy output.
    Optionally runs Anfuso interval backtest if y_true + quantiles are provided.
    """
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    # Save full sample metrics + governance
    full = {**run_output["full_sample"], **run_output.get("full_sample_governance", {})}
    with open(out_dir / "full_sample_metrics.json", "w", encoding="utf-8") as f:
        json.dump({k: _to_jsonable(v) for k, v in full.items()}, f, indent=2, ensure_ascii=False)

    # Save rolling outputs
    if run_output.get("rolling_overlapping") is not None:
        run_output["rolling_overlapping"].to_csv(out_dir / "rolling_overlapping.csv", index=False)
    if run_output.get("rolling_non_overlapping") is not None:
        run_output["rolling_non_overlapping"].to_csv(out_dir / "rolling_non_overlapping.csv", index=False)

    # Optional: Anfuso interval backtest
    if (
        y_true is not None
        and quantiles is not None
        and (alpha / 2) in quantiles
        and (1 - alpha / 2) in quantiles
    ):
        y = np.asarray(y_true, dtype=float)
        lo = np.asarray(quantiles[alpha / 2], dtype=float)
        hi = np.asarray(quantiles[1 - alpha / 2], dtype=float)
        anf = anfuso_interval_backtest(y, lo, hi, alpha=alpha)
        with open(out_dir / "anfuso_full_sample.json", "w", encoding="utf-8") as f:
            json.dump({k: _to_jsonable(v) for k, v in anf.items()}, f, indent=2, ensure_ascii=False)

    return {
        "full_sample_metrics": str(out_dir / "full_sample_metrics.json"),
        "rolling_overlapping": str(out_dir / "rolling_overlapping.csv"),
        "rolling_non_overlapping": str(out_dir / "rolling_non_overlapping.csv"),
        "anfuso_full_sample": str(out_dir / "anfuso_full_sample.json"),
    }
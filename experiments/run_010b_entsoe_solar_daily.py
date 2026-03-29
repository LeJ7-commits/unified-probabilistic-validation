"""
experiments/run_010b_entsoe_solar_daily.py
===========================================
Daily-aggregated robustness check for ENTSO-E Solar Germany (run_010).

MOTIVATION
----------
run_010 was classified RED due to PIT serial independence failure
(Ljung-Box, ACF lag-1 = 0.79) but with a PIT uniformity KS statistic
of only 0.0258 — below the 0.05 effect-size floor, flagged as large-n
sensitivity rather than genuine distributional failure.

Solar has an additional structural reason to prefer daily aggregation:
the nighttime exclusion applied in the hourly pipeline removes structurally
zero observations, but the remaining daytime-only hourly sequence still
has strong within-day solar irradiance cycles that introduce autocorrelation
in the reconstruction residuals. Daily means smooth over this within-day
structure and provide a cleaner test of inter-day calibration.

A GREEN/YELLOW classification at daily resolution confirms that the
hourly RED on independence reflects within-day solar irradiance structure
in the bootstrap reconstruction, not a fundamentally broken model.

OUTPUTS
-------
experiments/run_010b_entsoe_solar_daily/
    full_sample_metrics.json
    anfuso_full_sample.json
    rolling_overlapping.csv
    rolling_non_overlapping.csv
    governance_decision.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.diagnostics.run_policy import run_diagnostics_policy, write_run_artifacts
from src.diagnostics.diagnostics_input import Diagnostics_Input
from src.governance.decision_engine import DecisionEngine
from src.governance.risk_classification import RiskPolicy

# ── Config ────────────────────────────────────────────────────────────────────
ALPHA           = 0.10
COVERAGE_TARGET = 0.90
ROLLING_WINDOW  = 90
ROLLING_STEP    = 30

REPO     = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data" / "derived_entsoe_solar"
OUT_DIR  = REPO / "experiments" / "run_010b_entsoe_solar_daily"
T_PATH   = DATA_DIR / "entsoe_solar_t.npy"


def aggregate_to_daily(
    t: np.ndarray,
    y: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    samples: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Aggregate hourly arrays to daily means.

    For solar, daily aggregation additionally smooths the within-day
    irradiance cycle that causes systematic autocorrelation in the
    hourly bootstrap reconstruction.
    """
    t_dt  = pd.to_datetime(t.astype(str), utc=True).tz_convert("Europe/Berlin")
    dates = t_dt.normalize()

    df = pd.DataFrame({
        "date": dates,
        "y":    y,
        "lo":   lo,
        "hi":   hi,
    })

    agg  = df.groupby("date", sort=True).mean()
    y_d  = agg["y"].values
    lo_d = agg["lo"].values
    hi_d = agg["hi"].values

    if samples is not None:
        df_idx    = pd.Series(dates, name="date")
        day_groups = df_idx.groupby(df_idx).ngroup()
        n_days    = day_groups.max() + 1
        M         = samples.shape[1]
        s_d       = np.zeros((n_days, M), dtype=np.float32)
        counts    = np.zeros(n_days, dtype=int)
        for i, g in enumerate(day_groups):
            s_d[g] += samples[i]
            counts[g] += 1
        for g in range(n_days):
            if counts[g] > 0:
                s_d[g] /= counts[g]
        samples_d = s_d
    else:
        samples_d = None

    return y_d, lo_d, hi_d, samples_d


if __name__ == "__main__":
    print("=" * 60)
    print("  ENTSO-E Solar Germany — Daily Aggregation (run_010b)")
    print(f"  Robustness check: hourly n≈52k → daily n≈{52605//24:,}")
    print("=" * 60)

    # ── Load hourly artifacts ─────────────────────────────────────────────────
    y_h   = np.load(DATA_DIR / "entsoe_solar_y.npy").astype(float)
    lo_h  = np.load(DATA_DIR / "entsoe_solar_lo_base_90.npy").astype(float)
    hi_h  = np.load(DATA_DIR / "entsoe_solar_hi_base_90.npy").astype(float)
    t_h      = np.load(T_PATH, allow_pickle=True).astype(str)
    # t may include warmup rows — trim to match y
    if len(t_h) > len(y_h):
        t_h = t_h[-len(y_h):]

    samples_path = DATA_DIR / "entsoe_solar_samples.npy"
    samples_h = (
        np.load(samples_path).astype(float) if samples_path.exists() else None
    )
    if samples_h is None:
        print("  WARNING: samples not found — PIT diagnostics will be skipped")

    # ── Aggregate to daily ────────────────────────────────────────────────────
    y_d, lo_d, hi_d, samples_d = aggregate_to_daily(
        t_h, y_h, lo_h, hi_h, samples_h
    )
    n_daily = len(y_d)
    print(f"  Hourly n = {len(y_h):,} → Daily n = {n_daily:,}")

    quantiles_d = {ALPHA / 2: lo_d, 1 - ALPHA / 2: hi_d}

    # ── Run diagnostics ───────────────────────────────────────────────────────
    print("  Running diagnostics on daily-aggregated data...")
    run_out = run_diagnostics_policy(
        model_class            = "short_term",
        y_true                 = y_d,
        samples                = samples_d,
        quantiles              = quantiles_d,
        alpha                  = ALPHA,
        rolling_window         = ROLLING_WINDOW,
        rolling_step           = ROLLING_STEP,
        enable_rolling_for_long_term = False,
        lb_lags                = (5, 10, 20),
        coverage_target        = COVERAGE_TARGET,
    )

    # ── Write standard artifacts ──────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = write_run_artifacts(
        out_dir         = OUT_DIR,
        run_output      = run_out,
        alpha           = ALPHA,
        y_true          = y_d,
        quantiles       = quantiles_d,
        coverage_target = COVERAGE_TARGET,
    )
    print("  Artifacts written:")
    for k, v in paths.items():
        print(f"    {k}: {v}")

    # ── DecisionEngine governance decision ────────────────────────────────────
    di  = Diagnostics_Input(alpha=ALPHA)
    dro = di.from_arrays(
        y         = y_d,
        t         = np.arange(n_daily),
        model_id  = "entsoe_solar_de_daily",
        lo        = lo_d,
        hi        = hi_d,
        quantiles = quantiles_d,
        samples   = samples_d,
    )
    engine   = DecisionEngine(alpha=ALPHA,
                               global_policy=RiskPolicy(coverage_target=COVERAGE_TARGET))
    decision = engine.decide(dro)

    decision_path = OUT_DIR / "governance_decision.json"
    with open(decision_path, "w", encoding="utf-8") as f:
        json.dump(decision.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"\n  Governance decision (daily) : {decision.final_label}")
    print(f"  Reason codes               : "
          f"{[rc.value if hasattr(rc, 'value') else str(rc) for rc in decision.reason_codes]}")
    print(f"  Decision artifact          : {decision_path}")

    # ── Comparison summary ─────────────────────────────────────
    import json
    with open(OUT_DIR / "full_sample_metrics.json") as f:
        full_metrics = json.load(f)

    cov_daily  = full_metrics.get("empirical_coverage", float("nan"))
    ks_daily   = full_metrics.get("pit_ks_stat", float("nan"))
    acf_daily  = full_metrics.get("pit_acf_lag1", float("nan"))
    lb_p_daily = full_metrics.get("pit_lb_pvalue_lag5", float("nan"))

    print("\n  ── Comparison: Hourly vs Daily ──────────────────────────────")
    print(f"  {'Metric':<30} {'Hourly (run_010)':>18} {'Daily (run_010b)':>18}")
    print(f"  {'-'*66}")
    print(f"  {'n (eval)':<30} {'51,933':>18} {n_daily:>18,}")
    print(f"  {'Coverage':<30} {'89.94%':>18} {cov_daily:>17.2%}")
    print(f"  {'KS statistic':<30} {'0.0258':>18} {ks_daily:>18.4f}")
    print(f"  {'ACF lag-1':<30} {'0.787':>18} {acf_daily:>18.3f}")
    print(f"  {'LB p-value (lag 5)':<30} {'~0':>18} {lb_p_daily:>18.4f}")
    print(f"  {'Governance label':<30} {'RED':>18} {decision.final_label:>18}")
    print()

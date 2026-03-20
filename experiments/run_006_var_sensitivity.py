"""
experiments/run_006_var_sensitivity.py
=======================================
VaR Sensitivity Analysis: Economic Distortion Under Miscalibration

Two complementary analyses are computed for each model class:

Analysis 1 — Capital Multiplier Distortion (Basel VaR framework, adapted)
--------------------------------------------------------------------------
The Basel Committee (1996) framework uses raw exception counts in a 250-day
window against a 99% VaR (1% nominal breach rate). This thesis operates at
90% coverage (10% nominal breach rate), so a perfectly calibrated model
already produces 250 × 0.10 = 25 raw exceptions — far beyond Basel's RED
threshold of 10. Applying the raw schedule classifies every model as RED,
making the mapping uninformative.

The correct adaptation maps EXCESS exceptions above the nominal expectation:

  nominal_exceptions = window × nominal_breach_rate = 250 × 0.10 = 25
  excess = actual_exceptions − nominal_exceptions

  CONSERVATIVE : excess < −3   (under-breaching)   → multiplier 2.80
  GREEN        : −3 ≤ excess ≤ 2   (within noise)  → multiplier 3.00
  YELLOW       : 3 ≤ excess ≤ 8   (mild excess)    → multiplier 3.40
  RED          : excess > 8   (systematic excess)   → multiplier 4.00

Multiplier distortion = multiplier − 3.00 (GREEN baseline).

Analysis 2 — Operational Reserve Sizing Error
----------------------------------------------
If a model's 90% PI is used to size operational reserves and empirical
coverage differs from nominal 90%, reserves are over- or under-sized:

  reserve_error_pp = empirical_coverage − nominal_coverage
  direction        = "oversized" if positive, else "undersized"

Outputs saved to experiments/run_006_var_sensitivity/:
  var_sensitivity_results.json
  var_sensitivity_summary.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR   = REPO_ROOT / "experiments"
OUT_DIR   = EXP_DIR / "run_006_var_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NOMINAL_COVERAGE = 0.90
NOMINAL_ALPHA    = 0.10
WINDOW_DAYS      = 250
NOMINAL_EXCEPTIONS = WINDOW_DAYS * NOMINAL_ALPHA   # = 25.0

# Excess-based zone thresholds and multipliers
# excess = actual_exceptions - nominal_exceptions (25)
EXCESS_SCHEDULE = [
    # (excess_lo, excess_hi, zone, multiplier)
    (float("-inf"), -3,   "CONSERVATIVE", 2.80),
    (-3,             2,   "GREEN",        3.00),
    ( 3,             8,   "YELLOW",       3.40),
    ( 8, float("inf"), "RED",           4.00),
]

GREEN_MULTIPLIER = 3.00


def get_zone_and_multiplier(breach_rate: float, window: int = WINDOW_DAYS):
    actual_exceptions = breach_rate * window
    excess = actual_exceptions - NOMINAL_EXCEPTIONS
    for lo, hi, zone, mult in EXCESS_SCHEDULE:
        if lo <= excess < hi:
            return actual_exceptions, excess, zone, mult
    return actual_exceptions, excess, "RED", 4.00


def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS = [
    ("ENTSO-E",                    "run_001_entsoe",                               "Short-term electricity load"),
    ("PV Solar",                   "run_002_pv",                                   "Long-term PV generation"),
    ("Wind",                       "run_003_wind",                                 "Long-term wind generation"),
    ("Sim Price (well-spec)",      "run_004_simulation_price",                     "Simulation price — well-specified"),
    ("Sim Temp (well-spec)",       "run_004_simulation_temp",                      "Simulation temp — well-specified"),
    ("Sim Price — Var Inflation",  "run_004b_simulation_price_variance_inflation", "Variance inflation"),
    ("Sim Price — Mean Bias",      "run_004b_simulation_price_mean_bias",          "Mean bias"),
    ("Sim Price — Heavy Tails",    "run_004b_simulation_price_heavy_tails",        "Heavy tails"),
    ("Sim Temp — Var Inflation",   "run_004b_simulation_temp_variance_inflation",  "Variance inflation"),
    ("Sim Temp — Mean Bias",       "run_004b_simulation_temp_mean_bias",           "Mean bias"),
    ("Sim Temp — Heavy Tails",     "run_004b_simulation_temp_heavy_tails",         "Heavy tails"),
]

# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

results = []

for label, run_dir_name, description in DATASETS:
    run_dir = EXP_DIR / run_dir_name
    if not run_dir.exists():
        print(f"[SKIP] {label}: directory not found ({run_dir})")
        continue

    metrics = load_json(run_dir / "full_sample_metrics.json")
    anfuso  = load_json(run_dir / "anfuso_full_sample.json")

    if metrics is None:
        print(f"[SKIP] {label}: full_sample_metrics.json not found")
        continue

    emp_cov    = metrics.get("empirical_coverage", np.nan)
    risk_label = metrics.get("risk_label", "UNKNOWN")
    cov_error  = emp_cov - NOMINAL_COVERAGE

    breach_rate = anfuso.get("total_breach_rate", 1 - emp_cov) if anfuso else (1 - emp_cov)

    # --- Analysis 1: Capital multiplier (excess-based) ---
    actual_exc, excess_exc, adapted_zone, multiplier = get_zone_and_multiplier(breach_rate)

    capital_distortion_abs = multiplier - GREEN_MULTIPLIER
    capital_distortion_pct = capital_distortion_abs / GREEN_MULTIPLIER * 100

    if excess_exc < 0:
        capital_direction = f"over-capitalised ({abs(excess_exc):.1f} fewer exceptions than nominal)"
    else:
        capital_direction = f"under-capitalised ({excess_exc:.1f} excess exceptions above nominal)"

    # --- Analysis 2: Reserve sizing error ---
    reserve_error_pp  = cov_error * 100
    reserve_direction = "oversized" if cov_error > 0 else "undersized"

    record = {
        "dataset":                    label,
        "description":                description,
        "governance_label":           risk_label,
        "empirical_coverage":         round(emp_cov, 4),
        "coverage_error_pp":          round(reserve_error_pp, 2),
        "breach_rate":                round(breach_rate, 4),
        "actual_exceptions_250d":     round(actual_exc, 1),
        "nominal_exceptions_250d":    NOMINAL_EXCEPTIONS,
        "excess_exceptions":          round(excess_exc, 1),
        "adapted_zone":               adapted_zone,
        "capital_multiplier":         multiplier,
        "capital_distortion_abs":     round(capital_distortion_abs, 2),
        "capital_distortion_pct":     round(capital_distortion_pct, 1),
        "capital_direction":          capital_direction,
        "reserve_sizing_error_pp":    round(reserve_error_pp, 2),
        "reserve_direction":          reserve_direction,
    }
    results.append(record)
    print(
        f"[{adapted_zone:13s}] {label:40s}  "
        f"cov={emp_cov:.4f}  breach={breach_rate:.4f}  "
        f"actual_exc={actual_exc:.1f}  excess={excess_exc:+.1f}  "
        f"mult={multiplier:.2f}  reserve_err={reserve_error_pp:+.2f}pp"
    )

# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

(OUT_DIR / "var_sensitivity_results.json").write_text(json.dumps(results, indent=2))
df = pd.DataFrame(results)
df.to_csv(OUT_DIR / "var_sensitivity_summary.csv", index=False)

# ---------------------------------------------------------------------------
# Print formatted summaries
# ---------------------------------------------------------------------------

print("\n" + "=" * 90)
print("CAPITAL MULTIPLIER DISTORTION SUMMARY (excess-based Basel adaptation)")
print(f"Nominal exceptions in 250-day window: {NOMINAL_EXCEPTIONS:.0f}  "
      f"(250 × {NOMINAL_ALPHA:.0%} breach rate)")
print("=" * 90)
print(f"{'Dataset':<42} {'Gov':6} {'Adapted Zone':14} {'Exc':>5} {'Excess':>7} "
      f"{'Mult':>5} {'Distort%':>9}")
print("-" * 90)
for r in results:
    print(
        f"{r['dataset']:<42} {r['governance_label']:6} "
        f"{r['adapted_zone']:14} "
        f"{r['actual_exceptions_250d']:>5.1f} "
        f"{r['excess_exceptions']:>+7.1f} "
        f"{r['capital_multiplier']:>5.2f} "
        f"{r['capital_distortion_pct']:>8.1f}%"
    )

print("\n" + "=" * 90)
print("RESERVE SIZING ERROR SUMMARY")
print("=" * 90)
print(f"{'Dataset':<42} {'Gov':6} {'Cov Error (pp)':>15} {'Direction':>12}")
print("-" * 90)
for r in results:
    print(
        f"{r['dataset']:<42} {r['governance_label']:6} "
        f"{r['reserve_sizing_error_pp']:>+14.2f}pp "
        f"{r['reserve_direction']:>12}"
    )

print(f"\nResults saved to {OUT_DIR}")

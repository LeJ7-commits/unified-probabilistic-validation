"""
experiments/run_004b_simulation_misspec.py

Runs probabilistic validation diagnostics for three deliberate misspecification
scenarios on the simulation model class (price and temp series).

Scenarios evaluated
-------------------
1. variance_inflation  — DGP uses 2x sigma; model assumes sigma
2. mean_bias           — DGP mean shifted by +1 sigma; model uses true mean
3. heavy_tails         — DGP uses t(df=3); model assumes Gaussian

Each scenario is evaluated for both price and temp (6 runs total).

Expected governance outcomes
-----------------------------
- variance_inflation: RED — bilateral over-breaching, coverage well below 90%
- mean_bias:          RED — asymmetric tail failure (lower over, upper under)
- heavy_tails:        RED — excess tail breaching, PIT non-uniformity

These outcomes validate the discriminative power of the framework: it should
reliably return RED for known misspecification patterns, in contrast to the
GREEN classification obtained for the well-specified baseline in run_004.

Artifacts written to:
  experiments/run_004b_simulation_price_variance_inflation/
  experiments/run_004b_simulation_temp_variance_inflation/
  experiments/run_004b_simulation_price_mean_bias/
  experiments/run_004b_simulation_temp_mean_bias/
  experiments/run_004b_simulation_price_heavy_tails/
  experiments/run_004b_simulation_temp_heavy_tails/

Prerequisites
-------------
  Run scripts/build_simulation_misspec.py first.

Note on model_class
-------------------
  Uses model_class="simulation" with enable_rolling_for_long_term=True.
  Ensure src/diagnostics/run_policy.py routes "simulation" to rolling-enabled
  path (same fix applied for run_004).
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from src.diagnostics.run_policy import run_diagnostics_policy, write_run_artifacts

SCENARIOS = ["variance_inflation", "mean_bias", "heavy_tails"]
SERIES    = ["price", "temp"]
ALPHA     = 0.1


def run_one(
    scenario:  str,
    series:    str,
    repo_root: Path,
) -> dict:
    """Run diagnostics for one (scenario, series) combination."""

    data_dir = repo_root / "data" / f"derived_simulation_{series}_{scenario}"
    out_dir  = repo_root / "experiments" / f"run_004b_simulation_{series}_{scenario}"

    # --- load artifacts ---
    y     = np.load(data_dir / f"{series}_y.npy").astype(float)
    lower = np.load(data_dir / f"{series}_lo_base_90.npy").astype(float)
    upper = np.load(data_dir / f"{series}_hi_base_90.npy").astype(float)

    if lower.shape != y.shape or upper.shape != y.shape:
        raise ValueError(
            f"[{series}/{scenario}] Shape mismatch: y{y.shape}, "
            f"lower{lower.shape}, upper{upper.shape}. "
            "Re-run scripts/build_simulation_misspec.py."
        )

    quantiles = {ALPHA / 2: lower, 1 - ALPHA / 2: upper}

    # No samples for misspecified scenarios — interval diagnostics only.
    # PIT tests require samples; they will return null (same as run_004).
    samples = None

    # --- run policy diagnostics ---
    run_out = run_diagnostics_policy(
        model_class="simulation",
        y_true=y,
        samples=samples,
        quantiles=quantiles,
        alpha=ALPHA,
        rolling_window=50,
        rolling_step=10,
        enable_rolling_for_long_term=True,
        lb_lags=(5, 10, 20),
        coverage_target=0.90,
    )

    # --- write artifacts ---
    paths = write_run_artifacts(
        out_dir=out_dir,
        run_output=run_out,
        alpha=ALPHA,
        y_true=y,
        quantiles=quantiles,
        coverage_target=0.90,
    )

    # Print summary
    anf = run_out.get("full_sample", {})
    gov = run_out.get("full_sample_governance", {})
    cov = anf.get("empirical_coverage", float("nan"))
    label = gov.get("risk_label", "N/A")

    print(
        f"  [{series}/{scenario}] "
        f"coverage={cov:.4f}, "
        f"risk_label={label}"
    )

    return paths


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]

    print("run_004b — Misspecification Scenario Diagnostics")
    print("=" * 60)

    for scenario in SCENARIOS:
        print(f"\nScenario: {scenario.upper()}")
        for series in SERIES:
            run_one(scenario=scenario, series=series, repo_root=repo_root)

    print("\n" + "=" * 60)
    print("All misspecification runs complete.")
    print("Check experiments/run_004b_simulation_*/anfuso_full_sample.json")
    print("for breach rates and traffic-light classifications.")

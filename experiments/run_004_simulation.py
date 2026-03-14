"""
experiments/run_004_simulation.py

Runs probabilistic validation diagnostics for the simulation model class.

Two series are evaluated independently:
  - price: energy price simulation (jointly generated with temp, rho=0.5)
  - temp:  temperature simulation

Both are well-specified by construction (realised values drawn from the same
DGP as simulation paths). This provides a controlled positive-control baseline:
the diagnostic framework is expected to return GREEN or near-GREEN, confirming
that the framework does not produce false positives under a correctly specified
model.

Artifacts are written to:
  experiments/run_004_simulation_price/
  experiments/run_004_simulation_temp/

Prerequisites:
  Run scripts/build_simulation_derived.py first to generate derived artifacts.

Model class: "simulation"
  - If run_diagnostics_policy does not yet recognise "simulation" as a valid
    model_class string, add it to the policy routing logic in
    src/diagnostics/run_policy.py alongside "short_term" and "long_term".
    The simulation class should enable rolling diagnostics (same as long_term).

Rolling window: 50 steps / step 10
  - n_days=365 gives 365 evaluable observations.
  - Window=50 (~7 weeks of daily as-of dates) balances local adaptivity
    with having enough windows (~7 non-overlapping) for meaningful rolling
    diagnostics given the small sample.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from src.diagnostics.run_policy import run_diagnostics_policy, write_run_artifacts


def run_series(
    series: str,
    repo_root: Path,
    alpha: float = 0.1,
) -> None:
    """Run full diagnostic policy for one simulation series."""

    data_dir = repo_root / "data" / f"derived_simulation_{series}"
    out_dir  = repo_root / "experiments" / f"run_004_simulation_{series}"

    # --- load artifacts ---
    y     = np.load(data_dir / f"{series}_y.npy").astype(float)
    lower = np.load(data_dir / f"{series}_lo_base_90.npy").astype(float)
    upper = np.load(data_dir / f"{series}_hi_base_90.npy").astype(float)

    if lower.shape != y.shape or upper.shape != y.shape:
        raise ValueError(
            f"[{series}] Shape mismatch: y{y.shape}, "
            f"lower{lower.shape}, upper{upper.shape}. "
            "Re-run scripts/build_simulation_derived.py."
        )

    quantiles = {alpha / 2: lower, 1 - alpha / 2: upper}

    # No pre-built samples for simulation — samples could be added later
    # by saving the full path matrix from build_simulation_derived.py.
    samples = None

    # --- run policy diagnostics ---
    # model_class="simulation": treated as a long-horizon simulation model.
    # Rolling diagnostics enabled. Window=50 as-of dates (~7 weeks),
    # step=10. With n=365 this yields ~31 non-overlapping windows.
    #
    # NOTE: if run_diagnostics_policy raises ValueError on model_class="simulation",
    # add "simulation" to the routing logic in src/diagnostics/run_policy.py.
    # It should be treated identically to "long_term" (rolling enabled).
    run_out = run_diagnostics_policy(
        model_class="simulation",
        y_true=y,
        samples=samples,
        quantiles=quantiles,
        alpha=alpha,
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
        alpha=alpha,
        y_true=y,
        quantiles=quantiles,
        coverage_target=0.90,
    )

    print(f"\n[{series}] Run completed. Artifacts:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    alpha     = 0.1

    for series in ["price", "temp"]:
        print(f"\n{'='*60}")
        print(f"Running simulation diagnostics: {series.upper()}")
        print(f"{'='*60}")
        run_series(series=series, repo_root=repo_root, alpha=alpha)

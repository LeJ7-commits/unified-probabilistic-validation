from __future__ import annotations

from pathlib import Path
import numpy as np

from src.diagnostics.run_policy import run_diagnostics_policy, write_run_artifacts


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    data_dir  = repo_root / "data" / "derived_pv"
    out_dir   = repo_root / "experiments" / "run_002_pv"

    alpha = 0.1  # 90% central interval

    # --- load structural artifacts ---
    y = np.load(data_dir / "pv_y.npy").astype(float)

    # Use asymmetric lo/hi directly — built from rolling empirical quantiles
    # in build_renewables_derived.py. Symmetric yhat+scale reconstruction is
    # not used here because PV residuals are highly asymmetric (bounded below
    # by zero, strong diurnal and seasonal structure).
    lower = np.load(data_dir / "pv_lo_base_90.npy").astype(float)
    upper = np.load(data_dir / "pv_hi_base_90.npy").astype(float)

    if lower.shape != y.shape or upper.shape != y.shape:
        raise ValueError(
            f"Shape mismatch: y{y.shape}, lower{lower.shape}, upper{upper.shape}. "
            "All must be identical length. Re-run scripts/build_renewables_derived.py."
        )

    quantiles = {alpha / 2: lower, 1 - alpha / 2: upper}

    # samples optional
    samples_path = data_dir / "pv_samples.npy"
    samples = np.load(samples_path).astype(float) if samples_path.exists() else None

    # --- run policy diagnostics ---
    # model_class="long_term": rolling diagnostics are enabled.
    # rolling_window=720 (30 days of hourly data), rolling_step=168 (1 week).
    # This yields ~36 non-overlapping windows across the 3-year series,
    # consistent with the hourly data scale.
    run_out = run_diagnostics_policy(
        model_class="long_term",
        y_true=y,
        samples=samples,
        quantiles=quantiles,
        alpha=alpha,
        rolling_window=720,
        rolling_step=168,
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

    print("Run completed. Artifacts:")
    for k, v in paths.items():
        print(f"  {k}: {v}")

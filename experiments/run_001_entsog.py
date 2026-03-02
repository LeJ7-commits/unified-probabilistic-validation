from __future__ import annotations
from pathlib import Path
import numpy as np

from src.diagnostics.run_policy import run_diagnostics_policy, write_run_artifacts


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data" / "derived_long"   # adjust if needed
    out_dir  = repo_root / "experiments" / "run_001_entsog"

    alpha = 0.1  # 90% interval

    # --- load data ---
    y = np.load(data_dir / "entsog_y.npy").astype(float)
    lower = np.load(data_dir / "entsog_lo_base_90.npy").astype(float)
    upper = np.load(data_dir / "entsog_hi_base_90.npy").astype(float)
    quantiles = {alpha / 2: lower, 1 - alpha / 2: upper}

    # samples optional
    samples_path = data_dir / "entsog_samples.npy"
    samples = np.load(samples_path).astype(float) if samples_path.exists() else None

    # --- run policy diagnostics ---
    run_out = run_diagnostics_policy(
        model_class="short_term",
        y_true=y,
        samples=samples,
        quantiles=quantiles,
        alpha=alpha,
        rolling_window=250,
        rolling_step=50,
        enable_rolling_for_long_term=False,
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
        print(f"{k}: {v}")
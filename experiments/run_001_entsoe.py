from __future__ import annotations

from pathlib import Path
import numpy as np

from src.diagnostics.run_policy import run_diagnostics_policy, write_run_artifacts


def z_value_two_sided(alpha: float) -> float:
    """
    Two-sided Normal z for central (1 - alpha) interval.
    For alpha=0.1 -> z ≈ 1.6448536269514722 (90% central interval).
    Hardcoded via common values to avoid scipy dependency in scripts.
    """
    if abs(alpha - 0.10) < 1e-12:
        return 1.6448536269514722
    if abs(alpha - 0.05) < 1e-12:
        return 1.959963984540054
    if abs(alpha - 0.20) < 1e-12:
        return 1.2815515655446004
    raise ValueError(f"Unsupported alpha={alpha}. Add the z value or use scipy.stats.norm.ppf.")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data" / "derived_full"
    out_dir  = repo_root / "experiments" / "run_001_entsoe"

    alpha = 0.1  # 90% central interval

    # --- load structural artifacts ---
    y = np.load(data_dir / "entsoe_y.npy").astype(float)

    # Use asymmetric lo/hi directly — these are built from rolling empirical quantiles
    # in build_entsoe_derived.py and correctly capture the asymmetric residual structure.
    # The yhat+scale path assumed Gaussian symmetry which is not valid here.
    lo_path = data_dir / "entsoe_lo_base_90.npy"
    hi_path = data_dir / "entsoe_hi_base_90.npy"

    if lo_path.exists() and hi_path.exists():
        lower = np.load(lo_path).astype(float)
        upper = np.load(hi_path).astype(float)
    else:
        raise FileNotFoundError(
            "Missing required interval inputs: entsoe_lo_base_90.npy and entsoe_hi_base_90.npy\n"
            f"Looked in: {data_dir}\n"
            "Re-run scripts/build_entsoe_derived.py to regenerate."
        )

    quantiles = {alpha / 2: lower, 1 - alpha / 2: upper}

    # samples optional
    samples_path = data_dir / "entsoe_samples.npy"
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
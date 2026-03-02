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
    out_dir  = repo_root / "experiments" / "run_001_entsog"

    alpha = 0.1  # 90% central interval

    # --- load structural artifacts ---
    y = np.load(data_dir / "entsog_y.npy").astype(float)

    # Prefer constructing base interval from yhat + scale (cleaner than storing lo/hi npy)
    yhat_path = data_dir / "entsog_yhat.npy"
    scale_path = data_dir / "entsog_scale.npy"

    if yhat_path.exists() and scale_path.exists():
        yhat = np.load(yhat_path).astype(float)
        scale = np.load(scale_path).astype(float)

        if yhat.shape != y.shape or scale.shape != y.shape:
            raise ValueError(
                f"Shape mismatch: y{y.shape}, yhat{yhat.shape}, scale{scale.shape}. "
                "All must be identical length."
            )

        z = z_value_two_sided(alpha)
        lower = yhat - z * scale
        upper = yhat + z * scale

    else:
        # Fallback: if you still have stored lo/hi, use them; otherwise fail loudly.
        lo_path = data_dir / "entsog_lo_base_90.npy"
        hi_path = data_dir / "entsog_hi_base_90.npy"
        if lo_path.exists() and hi_path.exists():
            lower = np.load(lo_path).astype(float)
            upper = np.load(hi_path).astype(float)
        else:
            raise FileNotFoundError(
                "Missing required interval inputs. Provide either:\n"
                "  (a) entsog_yhat.npy + entsog_scale.npy  (preferred), OR\n"
                "  (b) entsog_lo_base_90.npy + entsog_hi_base_90.npy\n"
                f"Looked in: {data_dir}"
            )

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
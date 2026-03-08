"""
scripts/build_renewables_derived.py
=====================================
Builds derived artifacts for the long-term renewable generation forecasting
datasets: pv_student.csv and wind_student.csv.

Distribution reconstruction method:
    24-bucket hour-of-day conditioning with trailing rolling window.

    With 3 years of hourly data (~26,280 rows), each hour-of-day bucket
    contains ~1,095 observations, making 24-bucket conditioning stable.
    This approach captures diurnal structure in forecast errors, which is
    physically meaningful for both PV (strong diurnal cycle) and wind.

    Window parameter (justified in thesis methodology):
        W = 720  trailing hour-bucket-specific observations
                 (~30 days of same-hour observations per bucket)

PV nighttime exclusion:
    Observations where BOTH Simulation == 0 AND Actuals == 0 are excluded
    from calibration evaluation. These are structural nighttime zeros, not
    forecast errors, and including them would artificially inflate coverage
    metrics and distort PIT diagnostics.

    This exclusion is applied AFTER loading and is recorded in metadata.
    The exclusion criterion is: (|Simulation| < 1e-9) AND (|Actuals| < 1e-9).

Outputs (saved to data/derived_pv/ and data/derived_wind/):
    {asset}_y.npy             realised values after nighttime exclusion (n,)
    {asset}_yhat.npy          point forecasts after nighttime exclusion (n,)
    {asset}_scale.npy         robust scale estimate per observation (n,)
    {asset}_lo_base_90.npy    lower bound of 90% base interval (n,)
    {asset}_hi_base_90.npy    upper bound of 90% base interval (n,)
    {asset}_samples.npy       MC samples from Normal(yhat, scale) (n, n_samples)
    metadata.json             run metadata and parameter choices
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COL_T    = "Datetime"
COL_Y    = "Actuals"
COL_YHAT = "Simulation"

_W_DEFAULT     = 720    # trailing same-hour observations (~30 days)
_ALPHA_DEFAULT = 0.1    # 90% central interval
_ZERO_THRESH   = 1e-9   # threshold for nighttime zero detection (PV only)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _z_two_sided(alpha: float) -> float:
    table = {0.10: 1.6448536269514722, 0.05: 1.959963984540054, 0.20: 1.2815515655446004}
    if alpha not in table:
        raise ValueError(f"Unsupported alpha={alpha}.")
    return table[alpha]


def _robust_scale(x: np.ndarray) -> float:
    """1.4826 * MAD; falls back to std if MAD is degenerate."""
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    s   = 1.4826 * mad
    if not np.isfinite(s) or s <= 1e-12:
        s = float(np.std(x, ddof=1))
    return max(s, 1e-8)


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_derived(
    csv_path: Path,
    out_dir: Path,
    asset: str,
    *,
    exclude_nighttime_zeros: bool = False,
    alpha: float = _ALPHA_DEFAULT,
    W: int = _W_DEFAULT,
    n_samples: int = 500,
    seed: int = 42,
) -> dict:
    """
    Build and save derived artifacts for a single renewable asset.

    Parameters
    ----------
    csv_path : Path
        Raw CSV (pv_student.csv or wind_student.csv).
    out_dir : Path
        Directory to write .npy files and metadata.json.
    asset : str
        Short asset name used as filename prefix, e.g. 'pv' or 'wind'.
    exclude_nighttime_zeros : bool
        If True, exclude rows where both Simulation and Actuals are ~0.
        Should be True for PV, False for wind.
    alpha : float
        Miscoverage level (default 0.1 -> 90% central interval).
    W : int
        Trailing same-hour-bucket window size.
    n_samples : int
        Number of MC samples per observation.
    seed : int
        RNG seed for reproducibility.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load and clean ---
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    df[COL_T]    = pd.to_datetime(df[COL_T], utc=True, errors="coerce")
    df[COL_Y]    = pd.to_numeric(df[COL_Y],    errors="coerce")
    df[COL_YHAT] = pd.to_numeric(df[COL_YHAT], errors="coerce")

    n_raw     = len(df)
    na_counts = {c: int(df[c].isna().sum()) for c in [COL_T, COL_Y, COL_YHAT]}
    df = df.dropna(subset=[COL_T, COL_Y, COL_YHAT]).sort_values(COL_T).reset_index(drop=True)
    n_clean = len(df)

    if n_clean == 0:
        raise ValueError(f"No valid rows after cleaning. Raw={n_raw}, NaNs={na_counts}")

    # --- PV nighttime exclusion ---
    n_nighttime_excluded = 0
    if exclude_nighttime_zeros:
        night_mask = (
            (np.abs(df[COL_YHAT].to_numpy()) < _ZERO_THRESH) &
            (np.abs(df[COL_Y].to_numpy())    < _ZERO_THRESH)
        )
        n_nighttime_excluded = int(night_mask.sum())
        df = df[~night_mask].reset_index(drop=True)

    n_after_nighttime = len(df)

    if n_after_nighttime == 0:
        raise ValueError("No rows remain after nighttime exclusion.")

    y_all    = df[COL_Y].to_numpy(dtype=float)
    yhat_all = df[COL_YHAT].to_numpy(dtype=float)
    resid    = y_all - yhat_all

    # --- Hour-of-day bucket (0-23) ---
    hour = df[COL_T].dt.hour.to_numpy()

    # --- Rolling 24-bucket reconstruction ---
    lo_arr    = np.full(n_after_nighttime, np.nan)
    hi_arr    = np.full(n_after_nighttime, np.nan)
    scale_arr = np.full(n_after_nighttime, np.nan)

    for t in range(n_after_nighttime):
        h = hour[t]

        # Past observations in same hour bucket
        past_idx = np.where(hour[:t] == h)[0]
        if len(past_idx) < W:
            continue
        past_idx = past_idx[-W:]
        window   = resid[past_idx]

        lo_r = float(np.quantile(window, alpha / 2))
        hi_r = float(np.quantile(window, 1 - alpha / 2))

        lo_arr[t]    = yhat_all[t] + lo_r
        hi_arr[t]    = yhat_all[t] + hi_r
        scale_arr[t] = _robust_scale(window)

    # --- Trim to evaluable observations ---
    mask   = ~np.isnan(lo_arr)
    y      = y_all[mask]
    yhat   = yhat_all[mask]
    lo     = lo_arr[mask]
    hi     = hi_arr[mask]
    scale  = scale_arr[mask]
    n_eval = int(mask.sum())

    if n_eval == 0:
        raise ValueError(
            f"No evaluable observations after rolling reconstruction. "
            f"Reduce W={W} or check data length."
        )

    # --- MC samples ---
    rng     = np.random.default_rng(seed)
    samples = yhat[:, None] + scale[:, None] * rng.standard_normal((n_eval, n_samples))

    # --- Empirical coverage audit ---
    cov_90 = float(np.mean((y >= lo) & (y <= hi)))

    # --- Save ---
    np.save(out_dir / f"{asset}_y.npy",            y)
    np.save(out_dir / f"{asset}_yhat.npy",          yhat)
    np.save(out_dir / f"{asset}_scale.npy",         scale)
    np.save(out_dir / f"{asset}_lo_base_90.npy",    lo)
    np.save(out_dir / f"{asset}_hi_base_90.npy",    hi)
    np.save(out_dir / f"{asset}_samples.npy",       samples)

    meta = {
        "source":                    str(csv_path),
        "out_dir":                   str(out_dir),
        "asset":                     asset,
        "n_raw":                     int(n_raw),
        "n_clean":                   int(n_clean),
        "n_nighttime_excluded":      int(n_nighttime_excluded),
        "n_after_nighttime":         int(n_after_nighttime),
        "n_eval":                    int(n_eval),
        "na_before_drop":            na_counts,
        "alpha":                     float(alpha),
        "W":                         int(W),
        "n_samples":                 int(n_samples),
        "seed":                      int(seed),
        "empirical_coverage_90":     round(cov_90, 4),
        "exclude_nighttime_zeros":   exclude_nighttime_zeros,
        "nighttime_exclusion_note": (
            "Rows where both Simulation and Actuals are below 1e-9 are excluded "
            "from calibration evaluation. These are structural nighttime zeros "
            "for PV generation, not forecast errors. Including them would "
            "artificially inflate coverage metrics and distort PIT diagnostics. "
            "Applied to PV only; wind generates around the clock."
            if exclude_nighttime_zeros else "Not applied (wind dataset)."
        ),
        "reconstruction_method": (
            "24-bucket hour-of-day conditioning with trailing rolling window. "
            f"W={W} trailing same-hour observations (~30 days). "
            "Justified by 3 years of hourly data (~1,095 observations per "
            "hour-bucket), ensuring stable quantile estimation."
        ),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[{asset}] n_raw={n_raw}, n_clean={n_clean}, "
          f"n_nighttime_excluded={n_nighttime_excluded}, "
          f"n_eval={n_eval}, empirical_coverage_90={cov_90:.4f}")
    return meta


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    raw_dir   = repo_root / "data"

    configs = [
        {
            "csv_path":                raw_dir / "pv_student.csv",
            "out_dir":                 repo_root / "data" / "derived_pv",
            "asset":                   "pv",
            "exclude_nighttime_zeros": True,
        },
        {
            "csv_path":                raw_dir / "wind_student.csv",
            "out_dir":                 repo_root / "data" / "derived_wind",
            "asset":                   "wind",
            "exclude_nighttime_zeros": False,
        },
    ]

    for cfg in configs:
        csv_path = cfg.pop("csv_path")
        out_dir  = cfg.pop("out_dir")
        if not csv_path.exists():
            print(f"Skipping missing file: {csv_path}")
            continue
        meta = build_derived(csv_path, out_dir, **cfg)
        print(json.dumps(meta, indent=2))

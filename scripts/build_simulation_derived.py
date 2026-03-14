"""
scripts/build_simulation_derived.py

Generates derived artifacts for the simulation model class (price and temp).

Design decisions:
  - n_days = 365: one year of daily as-of dates. Gives enough evaluable
    observations for rolling diagnostics while running well under 15 minutes.
  - n_paths = 5000: matches original notebook setup.
  - n_horizons = 8760: full-year hourly horizon per as-of date.
  - Well-specified case only: realised values are drawn from the same DGP
    as the simulation paths. PIT values should be approximately U(0,1) by
    construction, providing a controlled positive-control baseline.
  - For each as-of date, lo/hi are the empirical alpha/2 and 1-alpha/2
    quantiles of the 5000 paths at horizon h=1 (the first forecast step).
    Using h=1 across all as-of dates gives a 365-observation evaluation
    series — sufficient for full-sample and rolling diagnostics.
  - Artifacts written to data/derived_simulation/{price,temp}/ to mirror
    the structure of derived_pv/ and derived_wind/.

NOTE: The simulation DGP is jointly specified for price and temp with
correlation rho=0.5. Each series is validated independently (univariate
marginal calibration). Joint/multivariate evaluation is deferred to
run_005_multivariate.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# DGP PARAMETERS — must match simulated_data.ipynb exactly
# ============================================================

SEED       = 42
N_DAYS     = 365
N_PATHS    = 5000
N_HORIZONS = 8760        # 365 * 24
ALPHA      = 0.1         # 90% central interval

BASE_PRICE      = 50.0
AMP_PRICE_DAY   = 10.0
AMP_PRICE_YEAR  = 5.0

BASE_TEMP       = 10.0
AMP_TEMP_DAY    = 8.0
AMP_TEMP_YEAR   = 12.0

SIGMA_PRICE = 5.0
SIGMA_TEMP  = 3.0
RHO         = 0.5

COV = np.array([
    [SIGMA_PRICE**2,                    RHO * SIGMA_PRICE * SIGMA_TEMP],
    [RHO * SIGMA_PRICE * SIGMA_TEMP,    SIGMA_TEMP**2                 ],
])
L = np.linalg.cholesky(COV)   # COV = L @ L.T


def hourly_means(asof: pd.Timestamp, n_horizons: int):
    """Return (mean_price, mean_temp) arrays of shape (n_horizons,)."""
    dt_index  = pd.date_range(asof, periods=n_horizons, freq="h")
    hod       = dt_index.hour.values
    doy       = dt_index.dayofyear.values

    day_phase  = 2 * np.pi * hod / 24.0
    year_phase = 2 * np.pi * (doy - 1) / 365.0

    mean_price = (
        BASE_PRICE
        + AMP_PRICE_DAY * (
            0.8 * np.sin(day_phase - np.pi)
            + 0.2 * np.sin(2 * (day_phase - np.pi))
        )
        + AMP_PRICE_YEAR * np.cos(year_phase)
    )

    mean_temp = (
        BASE_TEMP
        - AMP_TEMP_DAY  * np.cos(2 * np.pi * (hod - 5) / 24.0)
        + AMP_TEMP_YEAR * np.cos(year_phase + np.pi)
    )

    return mean_price, mean_temp


def build_simulation_derived(
    out_root: Path,
    n_days:     int   = N_DAYS,
    n_paths:    int   = N_PATHS,
    n_horizons: int   = N_HORIZONS,
    alpha:      float = ALPHA,
    seed:       int   = SEED,
) -> dict:
    """
    Generate simulation paths and realised values; compute empirical
    quantile intervals at horizon h=1 for each as-of date.

    Returns metadata dict.
    """
    rng = np.random.default_rng(seed)

    asof_dates = pd.date_range("2020-01-01", periods=n_days, freq="D")

    # Containers — one scalar per as-of date (h=1 evaluation)
    price_y    = np.empty(n_days)
    price_yhat = np.empty(n_days)
    price_lo   = np.empty(n_days)
    price_hi   = np.empty(n_days)

    temp_y     = np.empty(n_days)
    temp_yhat  = np.empty(n_days)
    temp_lo    = np.empty(n_days)
    temp_hi    = np.empty(n_days)

    for i, asof in enumerate(asof_dates):
        mp, mt = hourly_means(asof, n_horizons)

        # --- simulation paths (shape: n_horizons × n_paths) ---
        Z   = rng.standard_normal((n_horizons, n_paths, 2))
        eps = Z @ L.T                                    # correlated
        sims_price = mp[:, None] + eps[:, :, 0]         # (H, P)
        sims_temp  = mt[:, None] + eps[:, :, 1]

        # --- single realisation ---
        Z_real   = rng.standard_normal((n_horizons, 2))
        eps_real = Z_real @ L.T
        real_price = mp + eps_real[:, 0]
        real_temp  = mt + eps_real[:, 1]

        # --- evaluate at h=1 only ---
        h = 0   # index 0 = first hour ahead

        price_y[i]    = real_price[h]
        price_yhat[i] = mp[h]
        price_lo[i]   = float(np.quantile(sims_price[h, :], alpha / 2))
        price_hi[i]   = float(np.quantile(sims_price[h, :], 1 - alpha / 2))

        temp_y[i]     = real_temp[h]
        temp_yhat[i]  = mt[h]
        temp_lo[i]    = float(np.quantile(sims_temp[h, :], alpha / 2))
        temp_hi[i]    = float(np.quantile(sims_temp[h, :], 1 - alpha / 2))

    # ---- write artifacts ----
    for series, (y, yhat, lo, hi) in {
        "price": (price_y, price_yhat, price_lo, price_hi),
        "temp":  (temp_y,  temp_yhat,  temp_lo,  temp_hi),
    }.items():
        out_dir = out_root / f"derived_simulation_{series}"
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / f"{series}_y.npy",           y)
        np.save(out_dir / f"{series}_yhat.npy",        yhat)
        np.save(out_dir / f"{series}_lo_base_90.npy",  lo)
        np.save(out_dir / f"{series}_hi_base_90.npy",  hi)

        emp_cov = float(np.mean((y >= lo) & (y <= hi)))

        meta = {
            "series":               series,
            "out_dir":              str(out_dir),
            "n_days":               n_days,
            "n_paths":              n_paths,
            "n_horizons":           n_horizons,
            "eval_horizon_index":   0,
            "alpha":                alpha,
            "seed":                 seed,
            "empirical_coverage_90": emp_cov,
            "model_class":          "simulation",
            "specification":        "well_specified",
            "specification_note": (
                "Realised values are drawn from the same DGP as the simulation "
                "paths. PIT values should be approximately U(0,1) by construction. "
                "This serves as a controlled positive-control baseline: the "
                "diagnostic framework should return GREEN (or near-GREEN) for a "
                "correctly specified model."
            ),
            "dgp_parameters": {
                "rho":         RHO,
                "sigma_price": SIGMA_PRICE,
                "sigma_temp":  SIGMA_TEMP,
            },
            "reconstruction_method": (
                "Empirical alpha/2 and 1-alpha/2 quantiles of 5000 simulation "
                "paths at h=1 per as-of date. No rolling reconstruction needed — "
                "the distributional form is known exactly."
            ),
        }

        with open(out_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(
            f"[{series}] n_days={n_days}, empirical_coverage_90="
            f"{emp_cov:.4f}"
        )
        print(json.dumps(meta, indent=2))

    return meta


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    out_root  = repo_root / "data"
    build_simulation_derived(out_root)

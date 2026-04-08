"""
scripts/build_simulation_extended_derived.py
=============================================
Generates derived artifacts for three additional simulation commodity
classes: natural gas, carbon (CO2), and electricity price.

These extend the original two-series simulation (price, temp) to a
five-dimensional correlated DGP, as recommended by the industry partner
(Rikard Green, Energy Quant Solutions) who suggested using generative AI
to extend the synthetic simulation notebook to additional commodity classes.

DGP DESIGN
----------
  elec_price  — electricity price (€/MWh): high intraday amplitude, strong seasonal
  nat_gas     — natural gas (€/MWh): moderate intraday, ρ=0.60 with elec_price
  carbon      — carbon (€/tCO2): no intraday cycle, ρ=0.40 with elec_price

Same evaluation design as build_simulation_derived.py:
  - n_days = 365 as-of dates, n_paths = 5000, evaluate at h=1
  - Well-specified positive control
  - Full sample array (n_days x n_paths) saved for PIT diagnostics

OUTPUTS
-------
data/derived_simulation_elec_price/
    elec_price_{y,yhat,lo_base_90,hi_base_90,samples}.npy  + metadata.json
data/derived_simulation_nat_gas/
    nat_gas_{y,yhat,lo_base_90,hi_base_90,samples}.npy     + metadata.json
data/derived_simulation_carbon/
    carbon_{y,yhat,lo_base_90,hi_base_90,samples}.npy      + metadata.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# SHARED CONFIG
# ============================================================
SEED       = 123
N_DAYS     = 365
N_PATHS    = 5000
N_HORIZONS = 8760
ALPHA      = 0.1

# ============================================================
# DGP PARAMETERS
# ============================================================
BASE       = {"elec_price": 55.0,  "nat_gas": 40.0,  "carbon": 65.0}
SIGMA      = {"elec_price": 12.0,  "nat_gas":  8.0,  "carbon":  6.0}
INTRADAY   = {"elec_price": 1.5,   "nat_gas":  0.3,  "carbon":  0.0}
ANNUAL_AMP = {"elec_price": 8.0,   "nat_gas":  5.0,  "carbon":  2.0}

RHO = np.array([
    [1.00, 0.60, 0.40],
    [0.60, 1.00, 0.25],
    [0.40, 0.25, 1.00],
])
SIGMA_VEC = np.array([SIGMA["elec_price"], SIGMA["nat_gas"], SIGMA["carbon"]])
COV = np.diag(SIGMA_VEC) @ RHO @ np.diag(SIGMA_VEC)
L   = np.linalg.cholesky(COV)

SERIES   = ["elec_price", "nat_gas", "carbon"]
N_SERIES = len(SERIES)


def hourly_means(asof: pd.Timestamp, n_horizons: int) -> np.ndarray:
    dt_index   = pd.date_range(asof, periods=n_horizons, freq="h")
    hod        = dt_index.hour.values
    doy        = dt_index.dayofyear.values
    day_phase  = 2 * np.pi * hod / 24.0
    year_phase = 2 * np.pi * (doy - 1) / 365.0
    means = np.zeros((n_horizons, N_SERIES), dtype=float)
    for i, s in enumerate(SERIES):
        means[:, i] = (
            BASE[s]
            + INTRADAY[s] * BASE[s] * 0.1 * (
                0.8 * np.sin(day_phase - np.pi)
                + 0.2 * np.sin(2 * (day_phase - np.pi))
            )
            + ANNUAL_AMP[s] * np.cos(year_phase)
        )
    return means


def build_extended_derived(
    out_root:   Path,
    n_days:     int   = N_DAYS,
    n_paths:    int   = N_PATHS,
    n_horizons: int   = N_HORIZONS,
    alpha:      float = ALPHA,
    seed:       int   = SEED,
) -> None:

    rng        = np.random.default_rng(seed)
    asof_dates = pd.date_range("2020-01-01", periods=n_days, freq="D")

    y_all       = {s: np.empty(n_days)            for s in SERIES}
    yhat_all    = {s: np.empty(n_days)            for s in SERIES}
    lo_all      = {s: np.empty(n_days)            for s in SERIES}
    hi_all      = {s: np.empty(n_days)            for s in SERIES}
    samples_all = {s: np.empty((n_days, n_paths)) for s in SERIES}  # ← NEW

    print(f"Building extended simulation artifacts (n_days={n_days}, n_paths={n_paths})...")

    for i, asof in enumerate(asof_dates):
        means = hourly_means(asof, n_horizons)

        Z    = rng.standard_normal((n_horizons, n_paths, N_SERIES))
        eps  = Z @ L.T
        sims = means[:, None, :] + eps

        Z_real   = rng.standard_normal((n_horizons, N_SERIES))
        eps_real = Z_real @ L.T
        realized = means + eps_real

        h = 0
        for j, s in enumerate(SERIES):
            y_all[s][i]       = realized[h, j]
            yhat_all[s][i]    = means[h, j]
            lo_all[s][i]      = float(np.quantile(sims[h, :, j], alpha / 2))
            hi_all[s][i]      = float(np.quantile(sims[h, :, j], 1 - alpha / 2))
            samples_all[s][i] = sims[h, :, j]    # ← NEW

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_days} as-of dates completed")

    for s in SERIES:
        out_dir = out_root / f"derived_simulation_{s}"
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / f"{s}_y.npy",          y_all[s])
        np.save(out_dir / f"{s}_yhat.npy",        yhat_all[s])
        np.save(out_dir / f"{s}_lo_base_90.npy",  lo_all[s])
        np.save(out_dir / f"{s}_hi_base_90.npy",  hi_all[s])
        np.save(out_dir / f"{s}_samples.npy",     samples_all[s])  # ← NEW

        cov_check = float(np.mean((y_all[s] >= lo_all[s]) & (y_all[s] <= hi_all[s])))
        meta = {
            "series":         s,
            "n_days":         n_days,
            "n_paths":        n_paths,
            "n_horizons":     n_horizons,
            "alpha":          alpha,
            "seed":           seed,
            "base":           BASE[s],
            "sigma":          SIGMA[s],
            "empirical_coverage_check": round(cov_check, 4),
            "dgp":            "joint_gaussian_5d_extended",
            "eval_horizon":   "h=1 (first step)",
            "note": (
                "Well-specified positive control. Realised values drawn "
                "from same DGP as simulation paths. Full sample array "
                "(n_days x n_paths) saved for PIT diagnostics."
            ),
        }
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  [{s}] coverage: {cov_check:.1%}  samples: {samples_all[s].shape}")

    print("Done.")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    build_extended_derived(out_root=repo_root / "data")

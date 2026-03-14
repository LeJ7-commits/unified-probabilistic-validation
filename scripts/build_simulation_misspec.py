"""
scripts/build_simulation_misspec.py

Generates derived artifacts for three deliberate misspecification scenarios
applied to the simulation model class (price and temp series).

Scenarios
---------
1. variance_inflation (seed=43)
   - Model: Gaussian paths with true DGP parameters (sigma_price=5, sigma_temp=3)
   - DGP:   Realised values drawn with 2x standard deviation
   - Effect: Model intervals are too narrow by construction. The framework
             should detect bilateral over-breaching and RED classification.

2. mean_bias (seed=44)
   - Model: Gaussian paths centred on true mean
   - DGP:   Realised values drawn with mean shifted by +1 sigma
             (+5.0 for price, +3.0 for temp)
   - Effect: Systematic upward bias. Lower tail will over-breach, upper tail
             will under-breach. Framework should detect asymmetric tail failure.

3. heavy_tails (seed=45)
   - Model: Gaussian paths (thin-tailed assumption)
   - DGP:   Realised values drawn from t(df=3) scaled to match DGP variance
             (scale = sigma * sqrt((df-2)/df) so variance is preserved)
   - Effect: Tail events are more frequent than Gaussian model predicts.
             Framework should detect excess breaching concentrated in tails
             and PIT non-uniformity (U-shaped PIT histogram).

Design decisions
----------------
- Price and temp evaluated independently for all three scenarios.
- n_days=365, n_paths=5000, n_horizons=8760, h=1 evaluation (mirrors
  well-specified case in build_simulation_derived.py).
- Different seeds per scenario for realistic independent experiments.
- Model paths are ALWAYS drawn from the correctly-specified Gaussian DGP
  (sigma, rho as defined). Only the REALISED values are perturbed.
  This is the standard misspecification paradigm: the model is fixed,
  the world departs from the model's assumptions.

Output directories
------------------
  data/derived_simulation_price_variance_inflation/
  data/derived_simulation_temp_variance_inflation/
  data/derived_simulation_price_mean_bias/
  data/derived_simulation_temp_mean_bias/
  data/derived_simulation_price_heavy_tails/
  data/derived_simulation_temp_heavy_tails/
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# ── DGP parameters (must match build_simulation_derived.py exactly) ──────────

N_DAYS     = 365
N_PATHS    = 5000
N_HORIZONS = 8760
ALPHA      = 0.1

BASE_PRICE     = 50.0
AMP_PRICE_DAY  = 10.0
AMP_PRICE_YEAR = 5.0

BASE_TEMP      = 10.0
AMP_TEMP_DAY   = 8.0
AMP_TEMP_YEAR  = 12.0

SIGMA_PRICE = 5.0
SIGMA_TEMP  = 3.0
RHO         = 0.5

COV = np.array([
    [SIGMA_PRICE**2,                   RHO * SIGMA_PRICE * SIGMA_TEMP],
    [RHO * SIGMA_PRICE * SIGMA_TEMP,   SIGMA_TEMP**2                 ],
])
L = np.linalg.cholesky(COV)

SCENARIOS = {
    "variance_inflation": {"seed": 43},
    "mean_bias":          {"seed": 44},
    "heavy_tails":        {"seed": 45},
}


# ── hourly means (identical to build_simulation_derived.py) ──────────────────

def hourly_means(asof: pd.Timestamp, n_horizons: int):
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


# ── realised value generators per scenario ───────────────────────────────────

def draw_realised(
    scenario:    str,
    rng:         np.random.Generator,
    mean_price:  np.ndarray,
    mean_temp:   np.ndarray,
    h:           int = 0,
) -> tuple[float, float]:
    """
    Draw a single realised (price, temp) pair at horizon index h
    under the specified misspecification scenario.
    """
    if scenario == "variance_inflation":
        # DGP uses 2x sigma — realised values are more dispersed than model expects
        sigma_p = SIGMA_PRICE * 2.0
        sigma_t = SIGMA_TEMP  * 2.0
        cov_mis = np.array([
            [sigma_p**2,              RHO * sigma_p * sigma_t],
            [RHO * sigma_p * sigma_t, sigma_t**2             ],
        ])
        L_mis = np.linalg.cholesky(cov_mis)
        z = rng.standard_normal(2)
        eps = L_mis @ z
        return float(mean_price[h] + eps[0]), float(mean_temp[h] + eps[1])

    elif scenario == "mean_bias":
        # DGP mean shifted by +1 sigma for each series independently
        z = rng.standard_normal(2)
        eps = L @ z
        real_price = mean_price[h] + SIGMA_PRICE + eps[0]   # +1 sigma shift
        real_temp  = mean_temp[h]  + SIGMA_TEMP  + eps[1]   # +1 sigma shift
        return float(real_price), float(real_temp)

    elif scenario == "heavy_tails":
        # DGP uses t(df=3) scaled to preserve variance
        # scale = sigma * sqrt((df-2)/df)
        df = 3
        scale_p = SIGMA_PRICE * np.sqrt((df - 2) / df)
        scale_t = SIGMA_TEMP  * np.sqrt((df - 2) / df)
        # Draw correlated t shocks via Gaussian copula approach:
        # generate correlated normals, convert to uniform, then t quantile
        z = rng.standard_normal(2)
        eps_corr = L @ z / np.sqrt(np.array([SIGMA_PRICE**2, SIGMA_TEMP**2]))
        # eps_corr are N(0,1) with correlation rho
        u = scipy_stats.norm.cdf(eps_corr)          # uniform marginals
        t_price = scipy_stats.t.ppf(u[0], df=df) * scale_p
        t_temp  = scipy_stats.t.ppf(u[1], df=df) * scale_t
        return float(mean_price[h] + t_price), float(mean_temp[h] + t_temp)

    else:
        raise ValueError(f"Unknown scenario: {scenario}")


# ── build one scenario ────────────────────────────────────────────────────────

def build_scenario(
    scenario: str,
    seed:     int,
    out_root: Path,
) -> None:
    rng        = np.random.default_rng(seed)
    asof_dates = pd.date_range("2020-01-01", periods=N_DAYS, freq="D")

    price_y    = np.empty(N_DAYS)
    price_yhat = np.empty(N_DAYS)
    price_lo   = np.empty(N_DAYS)
    price_hi   = np.empty(N_DAYS)

    temp_y     = np.empty(N_DAYS)
    temp_yhat  = np.empty(N_DAYS)
    temp_lo    = np.empty(N_DAYS)
    temp_hi    = np.empty(N_DAYS)

    # Use a separate RNG for model paths (always well-specified Gaussian)
    # so that path randomness is independent of realised value randomness.
    rng_paths = np.random.default_rng(seed + 1000)

    for i, asof in enumerate(asof_dates):
        mp, mt = hourly_means(asof, N_HORIZONS)

        # --- model paths: always correctly-specified Gaussian ---
        Z   = rng_paths.standard_normal((N_HORIZONS, N_PATHS, 2))
        eps = Z @ L.T
        sims_price = mp[:, None] + eps[:, :, 0]
        sims_temp  = mt[:, None] + eps[:, :, 1]

        # --- misspecified realised value ---
        real_price, real_temp = draw_realised(scenario, rng, mp, mt, h=0)

        # --- h=1 evaluation ---
        price_y[i]    = real_price
        price_yhat[i] = mp[0]
        price_lo[i]   = float(np.quantile(sims_price[0, :], ALPHA / 2))
        price_hi[i]   = float(np.quantile(sims_price[0, :], 1 - ALPHA / 2))

        temp_y[i]     = real_temp
        temp_yhat[i]  = mt[0]
        temp_lo[i]    = float(np.quantile(sims_temp[0, :], ALPHA / 2))
        temp_hi[i]    = float(np.quantile(sims_temp[0, :], 1 - ALPHA / 2))

    # --- write artifacts ---
    for series, (y, yhat, lo, hi) in {
        "price": (price_y, price_yhat, price_lo, price_hi),
        "temp":  (temp_y,  temp_yhat,  temp_lo,  temp_hi),
    }.items():
        sigma_s = SIGMA_PRICE if series == "price" else SIGMA_TEMP
        out_dir = out_root / f"derived_simulation_{series}_{scenario}"
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / f"{series}_y.npy",           y)
        np.save(out_dir / f"{series}_yhat.npy",        yhat)
        np.save(out_dir / f"{series}_lo_base_90.npy",  lo)
        np.save(out_dir / f"{series}_hi_base_90.npy",  hi)

        emp_cov = float(np.mean((y >= lo) & (y <= hi)))

        # scenario-specific notes
        if scenario == "variance_inflation":
            spec_note = (
                f"DGP uses 2x sigma ({2*sigma_s:.1f}). "
                "Model paths use true sigma. Intervals are too narrow by construction. "
                "Expected: bilateral over-breaching, RED classification."
            )
        elif scenario == "mean_bias":
            spec_note = (
                f"DGP mean shifted by +1 sigma (+{sigma_s:.1f}). "
                "Model paths use true mean. "
                "Expected: lower tail over-breaches, upper tail under-breaches, RED."
            )
        else:  # heavy_tails
            spec_note = (
                "DGP uses t(df=3) scaled to preserve variance. "
                "Model paths use Gaussian. "
                "Expected: excess tail breaching, PIT non-uniformity (U-shaped), RED."
            )

        meta = {
            "series":                series,
            "scenario":              scenario,
            "out_dir":               str(out_dir),
            "n_days":                N_DAYS,
            "n_paths":               N_PATHS,
            "n_horizons":            N_HORIZONS,
            "eval_horizon_index":    0,
            "alpha":                 ALPHA,
            "seed_realised":         seed,
            "seed_paths":            seed + 1000,
            "empirical_coverage_90": emp_cov,
            "model_class":           "simulation",
            "specification":         f"misspecified_{scenario}",
            "specification_note":    spec_note,
            "dgp_parameters": {
                "rho":         RHO,
                "sigma_price": SIGMA_PRICE,
                "sigma_temp":  SIGMA_TEMP,
            },
        }

        with open(out_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(
            f"[{series}/{scenario}] seed={seed}, "
            f"empirical_coverage_90={emp_cov:.4f}"
        )


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    out_root  = repo_root / "data"

    for scenario, cfg in SCENARIOS.items():
        print(f"\n{'='*60}")
        print(f"Building scenario: {scenario.upper()}  (seed={cfg['seed']})")
        print(f"{'='*60}")
        build_scenario(scenario=scenario, seed=cfg["seed"], out_root=out_root)

    print("\nAll misspecification scenarios built.")

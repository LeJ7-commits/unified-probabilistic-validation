"""
experiments/run_005_multivariate.py

Multivariate joint diagnostic evaluation: PV + Wind (shared hourly 2013-2015 index).

Rationale
---------
Univariate diagnostics (run_002, run_003) evaluate each asset independently.
Energy market risk is inherently multivariate: wind and solar generation are
jointly driven by weather and exhibit contemporaneous and lagged dependence.
This run tests whether the marginal PIT residuals of PV and wind are jointly
independent — a necessary condition for the predictive distributions to be
usable for portfolio-level risk aggregation.

The evaluation is restricted to the shared daytime index (hours where PV is
non-zero) because:
  - PV nighttime rows are structurally excluded from calibration
  - Wind data is available at all hours
  - Joint evaluation requires a common time index
  - Restricting to daytime hours reduces n but ensures the joint residuals
    are directly comparable (same weather regimes drive both series)

Diagnostics computed
--------------------
1. Marginal PIT scores for PV and wind on the shared index
   - Computed from empirical sample CDFs (500 paths per observation)
   - Transformed: z = Phi^{-1}(u) for Ljung-Box

2. Multivariate Ljung-Box test on stacked z-residuals [z_pv, z_wind]
   - Tests for serial dependence in the joint residual vector
   - Lags: 5, 10, 20

3. Cross-correlation matrix of z-residuals at lags 0, 1, 6, 24
   - Lag 0: contemporaneous dependence (do PV and wind errors co-move?)
   - Lag 1: short-term lagged dependence (1 hour)
   - Lag 6: medium-term (6 hours, half a weather cycle)
   - Lag 24: diurnal cycle (same hour next day)

4. Energy score on the shared index
   - Requires samples from both series
   - Computed as the bivariate energy score using PV and wind sample paths
   - Provides a multivariate proper scoring rule beyond marginal evaluation

Artifacts written to: experiments/run_005_multivariate/

Prerequisites
-------------
  data/derived_pv/    (from scripts/build_renewables_derived.py)
  data/derived_wind/  (from scripts/build_renewables_derived.py)
  data/pv_student.csv   (raw, for timestamp reconstruction)
  data/wind_student.csv (raw, for timestamp reconstruction)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ── helpers ──────────────────────────────────────────────────────────────────

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _pit_from_samples(y: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """
    Compute empirical PIT scores from sample paths.

    For each observation i, PIT[i] = fraction of sample paths <= y[i].
    samples shape: (n, n_paths)
    Returns u of shape (n,) in (0, 1).
    """
    # Clip to (0,1) exclusive to avoid Phi^{-1} blowup
    u = np.mean(samples <= y[:, None], axis=1)
    return np.clip(u, 1e-6, 1 - 1e-6)


def _z_transform(u: np.ndarray) -> np.ndarray:
    """Probit transform: z = Phi^{-1}(u)."""
    return stats.norm.ppf(u)


def _ljungbox_multivariate(Z: np.ndarray, lags: list[int]) -> dict:
    """
    Multivariate Ljung-Box test on matrix Z (n x k).

    Uses the Hosking (1980) portmanteau statistic:
      Q(h) = n^2 * sum_{j=1}^{h} (1/(n-j)) * tr(C_j' C_0^{-1} C_j C_0^{-1})
    where C_j is the lag-j cross-covariance matrix.

    Under H0 (white noise), Q(h) ~ chi^2(k^2 * h).
    """
    n, k = Z.shape
    C0 = (Z.T @ Z) / n  # lag-0 covariance

    try:
        C0_inv = np.linalg.inv(C0)
    except np.linalg.LinAlgError:
        C0_inv = np.linalg.pinv(C0)

    results = {}
    for h in lags:
        Q = 0.0
        for j in range(1, h + 1):
            Cj = (Z[j:].T @ Z[:-j]) / n   # lag-j cross-covariance
            Q += (n**2 / (n - j)) * np.trace(Cj.T @ C0_inv @ Cj @ C0_inv)

        df = k * k * h
        pval = 1.0 - stats.chi2.cdf(Q, df=df)
        results[f"lag{h}"] = {
            "statistic": float(Q),
            "df":        int(df),
            "pvalue":    float(pval),
        }

    return results


def _cross_correlation(z1: np.ndarray, z2: np.ndarray, lags: list[int]) -> dict:
    """
    Cross-correlation of two z-residual series at specified lags.
    lag > 0: z1 leads z2 by `lag` steps.
    """
    n = len(z1)
    z1c = z1 - z1.mean()
    z2c = z2 - z2.mean()
    s1  = z1c.std()
    s2  = z2c.std()

    results = {}
    for lag in lags:
        if lag == 0:
            r = float(np.corrcoef(z1c, z2c)[0, 1])
        else:
            r = float(np.dot(z1c[:-lag], z2c[lag:]) / ((n - lag) * s1 * s2))
        results[f"lag{lag}"] = round(r, 6)

    return results


def _energy_score_bivariate(
    y1: np.ndarray,
    y2: np.ndarray,
    s1: np.ndarray,
    s2: np.ndarray,
) -> float:
    """
    Bivariate energy score.

    ES = E[||Y - X||] - 0.5 * E[||X - X'||]

    where Y = (y1, y2) is the realisation vector,
    X, X' are independent draws from the joint predictive distribution
    approximated by the paired sample paths (s1[i,:], s2[i,:]).

    s1, s2 shape: (n, n_paths)
    """
    n, m = s1.shape
    es_vals = np.empty(n)

    for i in range(n):
        real = np.array([y1[i], y2[i]])
        paths = np.column_stack([s1[i], s2[i]])   # (m, 2)

        # E[||Y - X||]
        diffs_real = paths - real[None, :]         # (m, 2)
        term1 = np.mean(np.sqrt(np.sum(diffs_real**2, axis=1)))

        # E[||X - X'||]: subsample for speed (use m//2 pairs)
        half = m // 2
        d = paths[:half] - paths[half:]            # (half, 2)
        term2 = 0.5 * np.mean(np.sqrt(np.sum(d**2, axis=1)))

        es_vals[i] = term1 - term2

    return float(np.mean(es_vals))


# ── main ─────────────────────────────────────────────────────────────────────

def run_multivariate(repo_root: Path) -> None:

    out_dir    = repo_root / "experiments" / "run_005_multivariate"
    pv_dir     = repo_root / "data" / "derived_pv"
    wind_dir   = repo_root / "data" / "derived_wind"
    pv_csv     = repo_root / "data" / "pv_student.csv"
    wind_csv   = repo_root / "data" / "wind_student.csv"
    _ensure_dir(out_dir)

    NIGHTTIME_THRESHOLD = 1e-9

    # ── 1. Reconstruct timestamps from raw CSVs ───────────────────────────
    print("Loading raw CSVs for timestamp reconstruction...")

    pv_raw   = pd.read_csv(pv_csv,   parse_dates=["Datetime"])
    wind_raw = pd.read_csv(wind_csv, parse_dates=["Datetime"])

    # Apply same nighttime exclusion as build_renewables_derived.py
    pv_mask = ~(
        (pv_raw["Simulation"].abs() < NIGHTTIME_THRESHOLD) &
        (pv_raw["Actuals"].abs()    < NIGHTTIME_THRESHOLD)
    )
    pv_daytime = pv_raw[pv_mask].copy().reset_index(drop=True)
    pv_ts      = pd.to_datetime(pv_daytime["Datetime"], utc=True)

    wind_ts = pd.to_datetime(wind_raw["Datetime"], utc=True)

    print(f"  PV daytime rows:  {len(pv_ts):,}")
    print(f"  Wind total rows:  {len(wind_ts):,}")

    # ── 2. Load derived arrays ────────────────────────────────────────────
    pv_y       = np.load(pv_dir   / "pv_y.npy").astype(float)
    wind_y     = np.load(wind_dir / "wind_y.npy").astype(float)
    pv_samples = np.load(pv_dir   / "pv_samples.npy").astype(float)
    wind_samp  = np.load(wind_dir / "wind_samples.npy").astype(float)

    print(f"  pv_y shape:       {pv_y.shape}, pv_samples shape: {pv_samples.shape}")
    print(f"  wind_y shape:     {wind_y.shape}, wind_samples shape: {wind_samp.shape}")

    # The derived arrays cover only the evaluable rows (after warmup burn).
    # build_renewables_derived.py burns the first W=720 same-hour observations
    # per bucket before starting evaluation. We need to align the timestamps
    # to the evaluable suffix of the daytime series.
    n_pv_eval   = len(pv_y)
    n_wind_eval = len(wind_y)

    pv_ts_eval   = pv_ts.iloc[-n_pv_eval:].reset_index(drop=True)
    wind_ts_eval = wind_ts.iloc[-n_wind_eval:].reset_index(drop=True)

    print(f"  PV eval timestamps:   {pv_ts_eval.iloc[0]} → {pv_ts_eval.iloc[-1]}")
    print(f"  Wind eval timestamps: {wind_ts_eval.iloc[0]} → {wind_ts_eval.iloc[-1]}")

    # ── 3. Find shared daytime index ──────────────────────────────────────
    pv_set   = pd.Index(pv_ts_eval)
    wind_set = pd.Index(wind_ts_eval)
    shared   = pv_set.intersection(wind_set).sort_values()

    print(f"\n  Shared daytime timestamps: {len(shared):,}")

    if len(shared) < 100:
        raise ValueError(
            f"Too few shared timestamps ({len(shared)}). "
            "Check that pv_student.csv and wind_student.csv share the same "
            "datetime index and that the evaluable suffix logic is correct."
        )

    pv_idx   = pv_ts_eval.isin(shared).values
    wind_idx = wind_ts_eval.isin(shared).values

    y_pv     = pv_y[pv_idx]
    y_wind   = wind_y[wind_idx]
    s_pv     = pv_samples[pv_idx]
    s_wind   = wind_samp[wind_idx]

    n_shared = len(y_pv)
    assert len(y_wind) == n_shared, "Alignment error: PV and wind shared arrays differ in length."
    print(f"  Aligned arrays: n = {n_shared:,}")

    # ── 4. Marginal PIT scores ────────────────────────────────────────────
    print("\nComputing marginal PIT scores...")

    u_pv   = _pit_from_samples(y_pv,   s_pv)
    u_wind = _pit_from_samples(y_wind, s_wind)

    z_pv   = _z_transform(u_pv)
    z_wind = _z_transform(u_wind)

    print(f"  z_pv   mean={z_pv.mean():.4f}, std={z_pv.std():.4f}")
    print(f"  z_wind mean={z_wind.mean():.4f}, std={z_wind.std():.4f}")

    # ── 5. Multivariate Ljung-Box ─────────────────────────────────────────
    print("\nRunning multivariate Ljung-Box test...")
    Z = np.column_stack([z_pv, z_wind])   # (n, 2)
    mv_lb = _ljungbox_multivariate(Z, lags=[5, 10, 20])

    for lag_key, res in mv_lb.items():
        print(f"  {lag_key}: Q={res['statistic']:.2f}, df={res['df']}, p={res['pvalue']:.4e}")

    # ── 6. Cross-correlation ──────────────────────────────────────────────
    print("\nComputing cross-correlation of z-residuals...")
    xcorr = _cross_correlation(z_pv, z_wind, lags=[0, 1, 6, 24])

    for lag_key, r in xcorr.items():
        print(f"  {lag_key}: r = {r:.4f}")

    # ── 7. Bivariate energy score ─────────────────────────────────────────
    print("\nComputing bivariate energy score (this may take ~1 min)...")
    es = _energy_score_bivariate(y_pv, y_wind, s_pv, s_wind)
    print(f"  Mean bivariate energy score: {es:.6f}")

    # ── 8. Write artifacts ────────────────────────────────────────────────
    results = {
        "n_shared":              n_shared,
        "assets":                ["pv", "wind"],
        "shared_index_note":     (
            "Joint evaluation restricted to daytime hours where both PV and "
            "wind have evaluable observations after nighttime exclusion and "
            "warmup burn. Daytime restriction ensures comparable weather "
            "regimes drive both series."
        ),
        "marginal_pit": {
            "pv":   {"mean_z": float(z_pv.mean()),   "std_z": float(z_pv.std())},
            "wind": {"mean_z": float(z_wind.mean()), "std_z": float(z_wind.std())},
        },
        "multivariate_ljungbox": mv_lb,
        "cross_correlation":     xcorr,
        "bivariate_energy_score": es,
        "independence_verdict": (
            "REJECT"
            if any(r["pvalue"] < 0.05 for r in mv_lb.values())
            else "FAIL TO REJECT"
        ),
    }

    out_path = out_dir / "multivariate_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Also save z-residuals as CSV for further analysis
    pd.DataFrame({
        "timestamp": shared.values,
        "z_pv":      z_pv,
        "z_wind":    z_wind,
        "u_pv":      u_pv,
        "u_wind":    u_wind,
    }).to_csv(out_dir / "joint_pit_residuals.csv", index=False)

    print(f"\nRun completed. Artifacts written to: {out_dir}")
    print(f"  multivariate_results.json")
    print(f"  joint_pit_residuals.csv")
    print(f"\nIndependence verdict: {results['independence_verdict']}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    run_multivariate(repo_root)

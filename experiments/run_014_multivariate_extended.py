"""
experiments/run_014_multivariate_extended.py
=============================================
Multivariate joint diagnostic evaluation across the five-series extended
simulation DGP: electricity price, natural gas, carbon, price (original),
and temperature.

MOTIVATION
----------
The supervisor guidance (and Gneiting and Katzfuss, 2014) requires that the
framework consider at least five multivariate commodity series. run_005
covers PV + wind on real data. This run extends multivariate evaluation to
the simulation model class using the five correlated Gaussian series built
by scripts/build_simulation_extended_derived.py and
scripts/build_simulation_derived.py.

The simulation architecture saves only quantile bounds (lo, hi) rather than
full sample paths — PIT-based evaluation and energy score are therefore not
available. The multivariate analysis uses interval breach indicators
(binary: 1 if y falls outside [lo, hi], 0 otherwise) as the evaluation
currency, computing:

  1. Cross-correlation matrix of breach indicators across all 5 series
     (contemporaneous and lagged correlations)
  2. Multivariate Ljung-Box test on the joint breach indicator vector
     (tests for serial dependence in the joint breach process)
  3. Joint coverage statistics (fraction of observations where ALL 5
     intervals simultaneously contain the realised value)
  4. Pairwise coverage co-failure rates (fraction of obs where both
     series simultaneously breach — relevant for portfolio risk)

Under the well-specified joint DGP, breaches should be approximately
independent across series at each time step (each marginal breach rate
~10%), and breach indicators should not be serially correlated. The
cross-series breach correlations directly measure whether miscalibration
is contemporaneously correlated across commodities — a key input for
portfolio-level reserve sizing.

ARTIFACTS
---------
experiments/run_014_multivariate_extended/
    multivariate_extended_results.json
    joint_breach_indicators.csv
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


def _ljungbox_multivariate(Z: np.ndarray, lags: list[int]) -> dict:
    """
    Multivariate Ljung-Box (Hosking 1980) on matrix Z (n × k).
    Under H0 (white noise), Q(h) ~ chi^2(k^2 * h).
    """
    n, k = Z.shape
    C0 = (Z.T @ Z) / n
    try:
        C0_inv = np.linalg.inv(C0)
    except np.linalg.LinAlgError:
        C0_inv = np.linalg.pinv(C0)

    results = {}
    for h in lags:
        Q = 0.0
        for j in range(1, h + 1):
            Cj = (Z[j:].T @ Z[:-j]) / n
            Q += (n**2 / (n - j)) * np.trace(Cj.T @ C0_inv @ Cj @ C0_inv)
        df   = k * k * h
        pval = float(1.0 - stats.chi2.cdf(Q, df=df))
        results[f"lag{h}"] = {
            "statistic": float(Q),
            "df":        int(df),
            "pvalue":    pval,
        }
    return results


def _cross_corr_matrix(B: np.ndarray, series_names: list[str],
                       lag: int = 0) -> dict:
    """
    Compute cross-correlation matrix of breach indicator matrix B (n × k)
    at specified lag. lag=0 = contemporaneous.
    """
    n, k = B.shape
    result = {}
    for i, s1 in enumerate(series_names):
        for j, s2 in enumerate(series_names):
            if lag == 0:
                r = float(np.corrcoef(B[:, i], B[:, j])[0, 1])
            else:
                b1 = B[:-lag, i] - B[:-lag, i].mean()
                b2 = B[lag:,  j] - B[lag:,  j].mean()
                denom = (n - lag) * B[:-lag, i].std() * B[lag:, j].std()
                r = float(np.dot(b1, b2) / denom) if denom > 0 else 0.0
            result[f"{s1}_x_{s2}"] = round(r, 4)
    return result


# ── main ─────────────────────────────────────────────────────────────────────

def run_multivariate_extended(repo_root: Path) -> None:

    out_dir  = repo_root / "experiments" / "run_014_multivariate_extended"
    data_dir = repo_root / "data"
    _ensure_dir(out_dir)

    ALPHA = 0.10   # 90% interval

    # Series definitions: (name, derived_dir_prefix, file_prefix)
    SERIES = [
        ("elec_price", "derived_simulation_elec_price", "elec_price"),
        ("nat_gas",    "derived_simulation_nat_gas",    "nat_gas"),
        ("carbon",     "derived_simulation_carbon",     "carbon"),
        ("price",      "derived_simulation_price",      "price"),
        ("temp",       "derived_simulation_temp",       "temp"),
    ]
    series_names = [s[0] for s in SERIES]
    k = len(SERIES)

    print("=" * 60)
    print("  Multivariate Extended Simulation (run_014)")
    print(f"  {k} series: {', '.join(series_names)}")
    print("=" * 60)

    # ── 1. Load artifacts ─────────────────────────────────────────────────
    y_arrays  = {}
    lo_arrays = {}
    hi_arrays = {}

    for name, dir_name, file_prefix in SERIES:
        d = data_dir / dir_name
        y_arrays[name]  = np.load(d / f"{file_prefix}_y.npy").astype(float)
        lo_arrays[name] = np.load(d / f"{file_prefix}_lo_base_90.npy").astype(float)
        hi_arrays[name] = np.load(d / f"{file_prefix}_hi_base_90.npy").astype(float)
        print(f"  Loaded {name}: n = {len(y_arrays[name])}")

    # Confirm all series have same length
    lengths = [len(y_arrays[n]) for n in series_names]
    assert len(set(lengths)) == 1, f"Series length mismatch: {lengths}"
    n = lengths[0]
    print(f"\n  All series: n = {n}\n")

    # ── 2. Compute breach indicators ──────────────────────────────────────
    print("Computing breach indicators...")
    B = np.zeros((n, k), dtype=float)   # breach indicator matrix
    marginal_coverage = {}

    for j, name in enumerate(series_names):
        inside = (y_arrays[name] >= lo_arrays[name]) & \
                 (y_arrays[name] <= hi_arrays[name])
        B[:, j] = (~inside).astype(float)   # 1 = breach, 0 = no breach
        cov = float(inside.mean())
        marginal_coverage[name] = round(cov, 4)
        print(f"  {name}: coverage = {cov:.2%}, "
              f"breach rate = {1-cov:.2%} (nominal 10%)")

    # ── 3. Joint coverage (all 5 inside simultaneously) ───────────────────
    all_inside  = np.all(B == 0, axis=1)
    joint_cov   = float(all_inside.mean())
    # Under independence: (0.90)^5 = 0.5905
    expected_joint = 0.90 ** k
    print(f"\n  Joint coverage (all 5 inside): {joint_cov:.2%} "
          f"(expected under independence: {expected_joint:.2%})")

    # ── 4. Pairwise co-failure rates ─────────────────────────────────────
    print("\n  Pairwise co-failure rates (both breach simultaneously):")
    pairwise_cofailure = {}
    for i in range(k):
        for j in range(i + 1, k):
            s1, s2 = series_names[i], series_names[j]
            rate = float(np.mean((B[:, i] == 1) & (B[:, j] == 1)))
            pairwise_cofailure[f"{s1}_x_{s2}"] = round(rate, 4)
            # Expected under independence: 0.10 * 0.10 = 0.01
            print(f"    {s1} × {s2}: {rate:.2%} (expected ~1.0%)")

    # ── 5. Contemporaneous cross-correlation of breach indicators ─────────
    print("\n  Cross-correlation of breach indicators (lag=0):")
    xcorr_lag0 = _cross_corr_matrix(B, series_names, lag=0)
    for pair, r in xcorr_lag0.items():
        if "_x_" in pair:
            s1, s2 = pair.split("_x_")
            if s1 != s2:
                print(f"    {s1} × {s2}: r = {r:.4f}")

    # ── 6. Lag-1 cross-correlation ────────────────────────────────────────
    xcorr_lag1 = _cross_corr_matrix(B, series_names, lag=1)

    # ── 7. Multivariate Ljung-Box on breach indicators ────────────────────
    print("\n  Multivariate Ljung-Box on joint breach indicator vector...")
    # Demean for LB
    B_dm = B - B.mean(axis=0)
    mv_lb = _ljungbox_multivariate(B_dm, lags=[5, 10, 20])

    for lag_key, res in mv_lb.items():
        print(f"    {lag_key}: Q = {res['statistic']:.2f}, "
              f"df = {res['df']}, p = {res['pvalue']:.4e}")

    independence_verdict = (
        "REJECT"
        if any(r["pvalue"] < 0.05 for r in mv_lb.values())
        else "FAIL TO REJECT"
    )

    # ── 8. Write artifacts ────────────────────────────────────────────────
    results = {
        "n":            n,
        "k":            k,
        "series":       series_names,
        "alpha":        ALPHA,
        "note": (
            "Multivariate evaluation uses interval breach indicators "
            "(binary: 1=breach, 0=inside). Full PIT and energy score "
            "are not computed for the simulation class because only "
            "quantile bounds (lo, hi) are available — not full sample paths."
        ),
        "marginal_coverage":    marginal_coverage,
        "joint_coverage": {
            "observed":  round(joint_cov, 4),
            "expected_under_independence": round(expected_joint, 4),
            "note": "Expected = 0.90^5 = 0.5905 under marginal independence",
        },
        "pairwise_cofailure_rates": pairwise_cofailure,
        "cross_correlation_lag0":   xcorr_lag0,
        "cross_correlation_lag1":   xcorr_lag1,
        "multivariate_ljungbox":    mv_lb,
        "independence_verdict":     independence_verdict,
    }

    out_path = out_dir / "multivariate_extended_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save breach indicator CSV
    df_breach = pd.DataFrame(B, columns=series_names)
    df_breach.to_csv(out_dir / "joint_breach_indicators.csv", index=False)

    print(f"\n  Run completed. Artifacts written to: {out_dir}")
    print(f"  multivariate_extended_results.json")
    print(f"  joint_breach_indicators.csv")
    print(f"\n  Independence verdict: {independence_verdict}")
    print(f"  Joint coverage: {joint_cov:.2%} "
          f"(expected {expected_joint:.2%})")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    run_multivariate_extended(repo_root)

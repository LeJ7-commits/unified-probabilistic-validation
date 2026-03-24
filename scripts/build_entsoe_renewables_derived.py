"""
scripts/build_entsoe_renewables_derived.py
===========================================
Builds derived artifacts for the ENTSO-E onshore wind and solar PV
generation forecasts for Germany (2020-2026).

Distribution reconstruction method:
    4 coarse time-of-day buckets (night/morning/afternoon/evening) with
    global shrinkage and bias correction, trailing rolling window.

    Mirrors build_entsoe_derived.py exactly — same reconstruction method,
    same window parameters, same MC sample generation.

    Bucket definitions (hourly, hour-of-day):
        0: Night     00:00-05:59
        1: Morning   06:00-11:59
        2: Afternoon 12:00-17:59
        3: Evening   18:00-23:59

    Window parameters:
        Wb = 40   trailing bucket-specific observations
        Wg = 672  trailing global observations (28 days of hourly data)

    Bias correction: residual mean within combined window subtracted
    before quantile estimation, then added back.
    Scale proxy: 1.4826 * MAD of centred residuals (robust sigma estimate).
    MC samples: yhat + scale * N(0, 1) — parametric Gaussian.

Solar: structural nighttime zeros excluded before reconstruction.
       Rows where both Simulation and Actuals < 1e-9 are removed.

Input files:
    data/entsoe_wind_onshore_de.csv   (Datetime, Simulation, Actuals)
    data/entsoe_solar_de.csv          (Datetime, Simulation, Actuals)

Outputs:
    data/derived_entsoe_wind/{prefix}_y.npy etc.
    data/derived_entsoe_solar/{prefix}_y.npy etc.
    metadata.json in each output directory
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

_BUCKET_BINS   = [-1, 5, 11, 17, 23]
_BUCKET_LABELS = [0, 1, 2, 3]
_Wb_DEFAULT    = 40
_Wg_DEFAULT    = 672
_ALPHA_DEFAULT = 0.1


def _robust_scale(x: np.ndarray) -> float:
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    s   = 1.4826 * mad
    if not np.isfinite(s) or s <= 1e-12:
        s = float(np.std(x, ddof=1))
    return max(s, 1e-8)


def build_derived(
    csv_path:             Path,
    out_dir:              Path,
    prefix:               str,
    nighttime_exclusion:  bool  = False,
    *,
    alpha:                float = _ALPHA_DEFAULT,
    Wb:                   int   = _Wb_DEFAULT,
    Wg:                   int   = _Wg_DEFAULT,
    n_samples:            int   = 500,
    seed:                 int   = 42,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Simulation": "y_hat", "Actuals": "y"})
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
    df["y"]        = pd.to_numeric(df["y"],     errors="coerce")
    df["y_hat"]    = pd.to_numeric(df["y_hat"], errors="coerce")

    n_raw     = len(df)
    na_counts = {c: int(df[c].isna().sum()) for c in ["Datetime", "y", "y_hat"]}

    n_night_excluded = 0
    if nighttime_exclusion:
        mask_night       = (df["y"].abs() < 1e-9) & (df["y_hat"].abs() < 1e-9)
        n_night_excluded = int(mask_night.sum())
        df               = df[~mask_night].reset_index(drop=True)

    df     = df.dropna(subset=["Datetime", "y", "y_hat"]).sort_values("Datetime").reset_index(drop=True)
    n_clean = len(df)

    if n_clean == 0:
        raise ValueError(f"No valid rows after cleaning. Raw={n_raw}")

    y_all    = df["y"].to_numpy(dtype=float)
    yhat_all = df["y_hat"].to_numpy(dtype=float)
    resid    = y_all - yhat_all

    local_hour = df["Datetime"].dt.tz_convert("Europe/Berlin").dt.hour.to_numpy()
    bucket     = pd.cut(local_hour, bins=_BUCKET_BINS, labels=_BUCKET_LABELS).astype(int)

    print(f"  Computing rolling quantile intervals (Wb={Wb}, Wg={Wg})...")
    lo_arr    = np.full(n_clean, np.nan)
    hi_arr    = np.full(n_clean, np.nan)
    scale_arr = np.full(n_clean, np.nan)

    for t in range(n_clean):
        b      = bucket[t]
        past_b = np.where(bucket[:t] == b)[0]
        if len(past_b) < Wb:
            continue
        past_b = past_b[-Wb:]
        if t < Wg:
            continue
        past_g = np.arange(t - Wg, t)
        window = np.concatenate([resid[past_b], resid[past_g]])
        bias   = float(window.mean())
        wc     = window - bias
        lo_arr[t]    = yhat_all[t] + float(np.quantile(wc, alpha / 2)) + bias
        hi_arr[t]    = yhat_all[t] + float(np.quantile(wc, 1 - alpha / 2)) + bias
        scale_arr[t] = _robust_scale(wc)

    mask   = ~np.isnan(lo_arr)
    y      = y_all[mask]
    yhat   = yhat_all[mask]
    lo     = lo_arr[mask]
    hi     = hi_arr[mask]
    scale  = scale_arr[mask]
    n_eval = int(mask.sum())

    if n_eval == 0:
        raise ValueError("No evaluable observations after rolling reconstruction.")

    print(f"  Generating MC samples (M={n_samples}, non-parametric bootstrap)...")
    rng     = np.random.default_rng(seed)
    resid_v = y - yhat  # residuals for the evaluable window
    samples = np.empty((n_eval, n_samples), dtype=np.float32)
    for i in range(n_eval):
        start        = max(0, i - Wg)
        pool         = resid_v[start:i] if i > 0 else resid_v[:1]
        if len(pool) < 2:
            pool = resid_v[:max(2, i + 1)]
        draws        = rng.choice(pool, size=n_samples, replace=True)
        samples[i]   = yhat[i] + draws
    print(f"  Samples shape: {samples.shape} ({samples.nbytes / 1e6:.0f} MB)")

    cov_90 = float(np.mean((y >= lo) & (y <= hi)))

    np.save(out_dir / f"{prefix}_y.npy",            y)
    np.save(out_dir / f"{prefix}_yhat.npy",          yhat)
    np.save(out_dir / f"{prefix}_scale.npy",         scale)
    np.save(out_dir / f"{prefix}_lo_base_90.npy",    lo)
    np.save(out_dir / f"{prefix}_hi_base_90.npy",    hi)
    np.save(out_dir / f"{prefix}_samples.npy",       samples)

    meta = {
        "source":                str(csv_path),
        "out_dir":               str(out_dir),
        "prefix":                prefix,
        "n_raw":                 int(n_raw),
        "n_night_excluded":      int(n_night_excluded),
        "n_clean":               int(n_clean),
        "n_eval":                int(n_eval),
        "na_before_drop":        na_counts,
        "alpha":                 float(alpha),
        "Wb":                    int(Wb),
        "Wg":                    int(Wg),
        "n_samples":             int(n_samples),
        "seed":                  int(seed),
        "empirical_coverage_90": round(cov_90, 4),
        "reconstruction_method": (
            "4-bucket coarse time-of-day conditioning with global shrinkage "
            "and bias correction. Identical to build_entsoe_derived.py."
        ),
        "sample_method": "Non-parametric bootstrap: yhat + resample(residual_pool, M)",
        "bucket_definitions": {
            "0_night":     "00:00-05:59",
            "1_morning":   "06:00-11:59",
            "2_afternoon": "12:00-17:59",
            "3_evening":   "18:00-23:59",
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"  n_raw={n_raw}, n_clean={n_clean}, n_eval={n_eval}, "
          f"coverage={cov_90:.4f}")
    return meta


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    data_dir  = repo_root / "data"

    print("=" * 60)
    print("  Building ENTSO-E Renewables Derived Artifacts")
    print("  Config: short-term (4-bucket, Wb=40, Wg=672, Gaussian samples)")
    print("=" * 60)

    for cfg in [
        dict(csv_path=data_dir/"entsoe_wind_onshore_de.csv",
             out_dir=data_dir/"derived_entsoe_wind",
             prefix="entsoe_wind", nighttime_exclusion=False,
             label="ENTSO-E Wind (Germany)"),
        dict(csv_path=data_dir/"entsoe_solar_de.csv",
             out_dir=data_dir/"derived_entsoe_solar",
             prefix="entsoe_solar", nighttime_exclusion=True,
             label="ENTSO-E Solar (Germany)"),
    ]:
        if not cfg["csv_path"].exists():
            print(f"[SKIP] Missing: {cfg['csv_path']}")
            continue
        print(f"\nBuilding: {cfg['label']}")
        build_derived(
            csv_path=cfg["csv_path"], out_dir=cfg["out_dir"],
            prefix=cfg["prefix"], nighttime_exclusion=cfg["nighttime_exclusion"],
        )

    print("\nAll derived artifacts built.")

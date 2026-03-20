"""
scripts/build_entsoe_derived.py
================================
Builds derived artifacts for the ENTSOG short-term load forecasting dataset.

Distribution reconstruction method:
    4 coarse time-of-day buckets (night/morning/afternoon/evening) with
    global shrinkage and bias correction, trailing rolling window.

    This method was selected based on the feasibility analysis in
    notebooks/02_entsoe_feasibility.ipynb, where it achieved the best
    coverage calibration (92.4% empirical vs 90% nominal) among the
    methods tested on the ENTSOG sample data.

    Bucket definitions (quarter-hourly, hour-of-day):
        0: Night     00:00-05:59
        1: Morning   06:00-11:59
        2: Afternoon 12:00-17:59
        3: Evening   18:00-23:59

    Window parameters (justified in thesis methodology):
        Wb = 40   trailing bucket-specific observations
        Wg = 672  trailing global observations (7 days of quarter-hourly)

    Bias correction: residual mean within combined window subtracted
    before quantile estimation, then added back.
    Scale proxy: 1.4826 * MAD of centred residuals (robust sigma estimate).

Outputs (saved to data/derived_full/):
    entsoe_y.npy             realised values (n,)
    entsoe_yhat.npy          point forecasts (n,)
    entsoe_scale.npy         robust scale estimate per observation (n,)
    entsoe_lo_base_90.npy    lower bound of 90% base interval (n,)
    entsoe_hi_base_90.npy    upper bound of 90% base interval (n,)
    entsoe_samples.npy       MC samples from Normal(yhat, scale) (n, n_samples)
    metadata.json            run metadata and parameter choices
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COL_T, COL_Y, COL_YHAT = "timestamp", "y", "y_hat"

# Bucket boundaries: hour-of-day -> coarse period index
#   0: Night 00-05, 1: Morning 06-11, 2: Afternoon 12-17, 3: Evening 18-23
_BUCKET_BINS   = [-1, 5, 11, 17, 23]
_BUCKET_LABELS = [0, 1, 2, 3]

# Window lengths (see module docstring for justification)
_Wb_DEFAULT = 40    # trailing bucket-specific observations
_Wg_DEFAULT = 672   # trailing global observations (7 days @ 15-min)

# Reconstruction alpha (90% central interval)
_ALPHA_DEFAULT = 0.1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _z_two_sided(alpha: float) -> float:
    table = {0.10: 1.6448536269514722, 0.05: 1.959963984540054, 0.20: 1.2815515655446004}
    if alpha not in table:
        raise ValueError(f"Unsupported alpha={alpha}. Add entry or use scipy.")
    return table[alpha]


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    if all(c in df.columns for c in [COL_T, COL_Y, COL_YHAT]):
        return df
    if "Load" in df.columns and "Load forecast" in df.columns:
        first = df.columns[0]
        return df.rename(columns={first: COL_T, "Load": COL_Y, "Load forecast": COL_YHAT})
    return df.rename(columns={df.columns[0]: COL_T, df.columns[1]: COL_Y, df.columns[2]: COL_YHAT})


def _robust_scale(x: np.ndarray) -> float:
    """1.4826 * MAD; falls back to std if MAD is degenerate."""
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    s = 1.4826 * mad
    if not np.isfinite(s) or s <= 1e-12:
        s = float(np.std(x, ddof=1))
    return max(s, 1e-8)


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_derived(
    csv_path: Path,
    out_dir: Path,
    *,
    alpha: float = _ALPHA_DEFAULT,
    Wb: int = _Wb_DEFAULT,
    Wg: int = _Wg_DEFAULT,
    n_samples: int = 500,
    seed: int = 42,
) -> dict:
    """
    Build and save derived artifacts for the ENTSOG dataset.

    Parameters
    ----------
    csv_path : Path
        Raw CSV (entsoe_full.csv or entsoe_sample_90days.csv).
    out_dir : Path
        Directory to write .npy files and metadata.json.
    alpha : float
        Miscoverage level for the base interval (default 0.1 -> 90%).
    Wb : int
        Trailing bucket-specific window size.
    Wg : int
        Trailing global window size for shrinkage.
    n_samples : int
        Number of MC samples per observation.
    seed : int
        RNG seed for reproducibility.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load and clean ---
    df = pd.read_csv(csv_path)
    df = _ensure_columns(df)
    df[COL_T]    = pd.to_datetime(df[COL_T], utc=True, errors="coerce")
    df[COL_Y]    = pd.to_numeric(df[COL_Y],    errors="coerce")
    df[COL_YHAT] = pd.to_numeric(df[COL_YHAT], errors="coerce")

    n_raw = len(df)
    na_counts = {c: int(df[c].isna().sum()) for c in [COL_T, COL_Y, COL_YHAT]}
    df = df.dropna(subset=[COL_T, COL_Y, COL_YHAT]).sort_values(COL_T).reset_index(drop=True)
    n_clean = len(df)

    if n_clean == 0:
        raise ValueError(f"No valid rows after cleaning. Raw={n_raw}, NaNs={na_counts}")

    y_all    = df[COL_Y].to_numpy(dtype=float)
    yhat_all = df[COL_YHAT].to_numpy(dtype=float)
    resid    = y_all - yhat_all

    # --- Assign 4-bucket time-of-day index ---
    hour   = df[COL_T].dt.hour.to_numpy()
    bucket = pd.cut(
        hour,
        bins=_BUCKET_BINS,
        labels=_BUCKET_LABELS,
    ).astype(int)

    # --- Rolling reconstruction ---
    lo_arr    = np.full(n_clean, np.nan)
    hi_arr    = np.full(n_clean, np.nan)
    scale_arr = np.full(n_clean, np.nan)

    for t in range(n_clean):
        b = bucket[t]

        # Past bucket-specific indices
        past_b = np.where(bucket[:t] == b)[0]
        if len(past_b) < Wb:
            continue
        past_b = past_b[-Wb:]

        # Past global indices
        if t < Wg:
            continue
        past_g = np.arange(t - Wg, t)

        # Combined window residuals
        window = np.concatenate([resid[past_b], resid[past_g]])

        # Bias correction: subtract mean of combined window
        bias   = float(window.mean())
        window_c = window - bias

        # Quantile bounds (centred), then re-add bias
        lo_r = float(np.quantile(window_c, alpha / 2))     + bias
        hi_r = float(np.quantile(window_c, 1 - alpha / 2)) + bias

        lo_arr[t]    = yhat_all[t] + lo_r
        hi_arr[t]    = yhat_all[t] + hi_r
        scale_arr[t] = _robust_scale(window_c)

    # --- Trim to evaluable observations ---
    mask  = ~np.isnan(lo_arr)
    y     = y_all[mask]
    yhat  = yhat_all[mask]
    lo    = lo_arr[mask]
    hi    = hi_arr[mask]
    scale = scale_arr[mask]
    n_eval = int(mask.sum())

    if n_eval == 0:
        raise ValueError(
            f"No evaluable observations after rolling reconstruction. "
            f"Increase dataset size or reduce Wb={Wb}/Wg={Wg}."
        )

    # --- MC samples ---
    rng     = np.random.default_rng(seed)
    samples = yhat[:, None] + scale[:, None] * rng.standard_normal((n_eval, n_samples))

    # --- Empirical coverage audit ---
    cov_90 = float(np.mean((y >= lo) & (y <= hi)))

    # --- Save ---
    np.save(out_dir / "entsoe_y.npy",            y)
    np.save(out_dir / "entsoe_yhat.npy",          yhat)
    np.save(out_dir / "entsoe_scale.npy",         scale)
    np.save(out_dir / "entsoe_lo_base_90.npy",    lo)
    np.save(out_dir / "entsoe_hi_base_90.npy",    hi)
    np.save(out_dir / "entsoe_samples.npy",       samples)

    meta = {
        "source":               str(csv_path),
        "out_dir":              str(out_dir),
        "n_raw":                int(n_raw),
        "n_clean":              int(n_clean),
        "n_eval":               int(n_eval),
        "na_before_drop":       na_counts,
        "alpha":                float(alpha),
        "Wb":                   int(Wb),
        "Wg":                   int(Wg),
        "n_samples":            int(n_samples),
        "seed":                 int(seed),
        "empirical_coverage_90": round(cov_90, 4),
        "reconstruction_method": (
            "4-bucket coarse time-of-day conditioning (night/morning/afternoon/evening) "
            "with global shrinkage and bias correction. "
            "Selected based on feasibility analysis in notebooks/02_entsoe_feasibility.ipynb."
        ),
        "bucket_definitions": {
            "0_night":     "00:00-05:59",
            "1_morning":   "06:00-11:59",
            "2_afternoon": "12:00-17:59",
            "3_evening":   "18:00-23:59",
        },
        "window_justification": (
            f"Wb={Wb} bucket-specific observations; "
            f"Wg={Wg} global observations (7 days of quarter-hourly data). "
            "Chosen to balance local adaptivity with estimation stability."
        ),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[entsoe] n_raw={n_raw}, n_clean={n_clean}, n_eval={n_eval}, "
          f"empirical_coverage_90={cov_90:.4f}")
    return meta


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    raw_dir   = repo_root / "data"

    configs = [
        (raw_dir / "entsoe_full.csv",          repo_root / "data" / "derived_full"),
        (raw_dir / "entsoe_sample_90days.csv",  repo_root / "data" / "derived_dev"),
    ]

    for csv_path, out_dir in configs:
        if not csv_path.exists():
            print(f"[entsoe] Skipping missing file: {csv_path}")
            continue
        meta = build_derived(csv_path, out_dir)
        print(json.dumps(meta, indent=2))

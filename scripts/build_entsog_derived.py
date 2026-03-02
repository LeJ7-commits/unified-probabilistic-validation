from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

COL_T, COL_Y, COL_YHAT = "timestamp", "y", "y_hat"


def z_value_two_sided(alpha: float) -> float:
    """
    Two-sided Normal z for central (1 - alpha) interval.
    alpha=0.1 -> z ≈ 1.6448536269514722 (90% central interval).
    """
    if abs(alpha - 0.10) < 1e-12:
        return 1.6448536269514722
    if abs(alpha - 0.05) < 1e-12:
        return 1.959963984540054
    if abs(alpha - 0.20) < 1e-12:
        return 1.2815515655446004
    raise ValueError(f"Unsupported alpha={alpha}. Add z value or use scipy.stats.norm.ppf.")


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Conform raw CSV to canonical names: timestamp, y, y_hat.
    For your entsog_full.csv: header is ',Load,Load forecast' so first col is unnamed.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # If canonical names already exist, we're done
    if all(c in df.columns for c in [COL_T, COL_Y, COL_YHAT]):
        return df

    # Explicit handling for your known column names (preferred)
    if "Load" in df.columns and "Load forecast" in df.columns:
        first = df.columns[0]
        df = df.rename(columns={first: COL_T, "Load": COL_Y, "Load forecast": COL_YHAT})
        return df

    # Fallback: first three columns
    df = df.rename(columns={df.columns[0]: COL_T, df.columns[1]: COL_Y, df.columns[2]: COL_YHAT})
    return df


def build_derived(
    csv_path: Path,
    out_dir: Path,
    *,
    alpha: float = 0.1,
    store_in_utc: bool = True,
    n_samples: int = 500,
    seed: int = 42,
    max_rows_for_samples: int | None = None,
) -> dict:
    """
    Builds derived artifacts for diagnostics:
      - entsog_y.npy
      - entsog_yhat.npy
      - entsog_scale.npy  (sigma under Normal assumption)
      - entsog_lo_base_90.npy / entsog_hi_base_90.npy
      - entsog_samples.npy  (MC samples from Normal(mu=yhat, sigma=scale))
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = _ensure_columns(df)

    # Parse timestamps robustly and audit drops
    df[COL_T] = pd.to_datetime(
        df[COL_T],
        utc=store_in_utc,
        errors="coerce",
    )

    # Coerce y and y_hat to numeric (handles strings)
    df[COL_Y] = pd.to_numeric(df[COL_Y], errors="coerce")
    df[COL_YHAT] = pd.to_numeric(df[COL_YHAT], errors="coerce")

    before = len(df)
    na_t = int(df[COL_T].isna().sum())
    na_y = int(df[COL_Y].isna().sum())
    na_yhat = int(df[COL_YHAT].isna().sum())

    df = df.dropna(subset=[COL_T, COL_Y, COL_YHAT]).sort_values(COL_T).reset_index(drop=True)
    after = len(df)

    if after == 0:
        raise ValueError(
            f"No valid rows after cleaning. "
            f"Before={before}, NaT={na_t}, NaN_y={na_y}, NaN_yhat={na_yhat}. "
            f"Columns={df.columns.tolist()}"
        )

    y = df[COL_Y].to_numpy(dtype=float)
    yhat = df[COL_YHAT].to_numpy(dtype=float)
    resid = y - yhat

    # --- Base interval placeholder (global residual quantiles) ---
    lo_q = float(np.quantile(resid, alpha / 2))
    hi_q = float(np.quantile(resid, 1 - alpha / 2))
    lo = yhat + lo_q
    hi = yhat + hi_q

    # --- Convert interval to Normal scale for sampling ---
    # For a central (1-alpha) interval under Normal:
    #   lo = mu - z*sigma, hi = mu + z*sigma  => sigma = (hi-lo)/(2z)
    z = z_value_two_sided(alpha)
    scale = (hi - lo) / (2.0 * z)

    # Guard against zero/negative scale (shouldn’t happen with hi>lo but keep safe)
    eps = 1e-8
    scale = np.maximum(scale, eps)

    # --- Generate samples (optional downsample for huge n) ---
    rng = np.random.default_rng(seed)
    n = len(y)

    if max_rows_for_samples is not None and n > max_rows_for_samples:
        # Keep files manageable: sample a subset of rows for MC samples,
        # but still store full y/yhat/intervals for other metrics.
        idx = rng.choice(n, size=max_rows_for_samples, replace=False)
        idx.sort()
        y_samples_mu = yhat[idx]
        y_samples_scale = scale[idx]
        samples = y_samples_mu[:, None] + y_samples_scale[:, None] * rng.standard_normal((len(idx), n_samples))
        samples_index = idx  # store mapping
    else:
        samples = yhat[:, None] + scale[:, None] * rng.standard_normal((n, n_samples))
        samples_index = None

    # --- Save exactly the filenames your run_001 expects ---
    np.save(out_dir / "entsog_y.npy", y)
    np.save(out_dir / "entsog_yhat.npy", yhat)
    np.save(out_dir / "entsog_scale.npy", scale)
    np.save(out_dir / "entsog_lo_base_90.npy", lo)
    np.save(out_dir / "entsog_hi_base_90.npy", hi)
    np.save(out_dir / "entsog_samples.npy", samples)

    meta = {
        "csv_path": str(csv_path),
        "out_dir": str(out_dir),
        "n_raw": int(before),
        "n_clean": int(after),
        "alpha": float(alpha),
        "z_two_sided": float(z),
        "timestamp_dtype": str(df[COL_T].dtype),
        "t_min": str(df[COL_T].iloc[0]),
        "t_max": str(df[COL_T].iloc[-1]),
        "columns_after_rename": [COL_T, COL_Y, COL_YHAT],
        "store_in_utc": bool(store_in_utc),
        "na_before_drop": {"timestamp": na_t, "y": na_y, "y_hat": na_yhat},
        "base_interval_note": "lo/hi built from global residual quantiles (placeholder).",
        "sampling_note": "samples generated from Normal(mu=yhat, sigma=(hi-lo)/(2z)).",
        "n_samples": int(n_samples),
        "seed": int(seed),
        "max_rows_for_samples": max_rows_for_samples,
        "samples_index": samples_index.tolist() if samples_index is not None else None,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]

    raw_dir = repo_root / "data"
    raw_full = raw_dir / "entsog_full.csv"
    raw_90d = raw_dir / "entsog_sample_90days.csv"

    out_full = repo_root / "data" / "derived_full"
    out_dev = repo_root / "data" / "derived_dev"

    if not raw_full.exists():
        raise FileNotFoundError(f"Missing full dataset: {raw_full}")

    meta_full = build_derived(
        raw_full,
        out_full,
        alpha=0.1,
        store_in_utc=True,
        n_samples=500,
        seed=42,
        max_rows_for_samples=None,  # set e.g. 50000 if file sizes get too big
    )
    print("Built FULL artifacts:\n", json.dumps(meta_full, indent=2))

    if raw_90d.exists():
        meta_dev = build_derived(
            raw_90d,
            out_dev,
            alpha=0.1,
            store_in_utc=True,
            n_samples=500,
            seed=42,
            max_rows_for_samples=None,
        )
        print("Built DEV artifacts:\n", json.dumps(meta_dev, indent=2))
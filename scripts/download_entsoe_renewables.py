"""
scripts/download_entsoe_renewables.py
======================================
Downloads day-ahead wind onshore and solar forecasts + actuals
from ENTSO-E Transparency Platform for Germany (2019-2023).

Outputs (saved to data/):
  entsoe_wind_onshore_de.csv   columns: Datetime, Simulation, Actuals
  entsoe_solar_de.csv          columns: Datetime, Simulation, Actuals

These files mirror the structure of pv_student.csv / wind_student.csv
and can be fed directly into build_renewables_derived.py.
"""

import pandas as pd
from entsoe import EntsoePandasClient
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
API_TOKEN  = "your-token-here"     # paste your token from transparency.entsoe.eu
ZONE       = "DE_LU"               # Germany-Luxembourg bidding zone
START = pd.Timestamp("2020-01-01", tz="Europe/Berlin")
END   = pd.Timestamp("2026-01-01", tz="Europe/Berlin")
OUT_DIR    = Path(__file__).resolve().parents[1] / "data"

# PSR type codes
WIND_ONSHORE = "B19"
SOLAR        = "B16"

# ─────────────────────────────────────────────────────────────────────────────

client = EntsoePandasClient(api_key=API_TOKEN)

print("Fetching day-ahead forecasts (14.1.D)...")
forecast_raw = client.query_wind_and_solar_forecast(
    ZONE, start=START, end=END, psr_type=None
)
print(f"  Forecast shape: {forecast_raw.shape}")
print(f"  Forecast columns: {forecast_raw.columns.tolist()}")

print("Fetching actual generation (16.1.B/C)...")
actuals_raw = client.query_generation(
    ZONE, start=START, end=END, psr_type=None
)
print(f"  Actuals shape: {actuals_raw.shape}")
print(f"  Actuals columns: {actuals_raw.columns.tolist()}")

# ── Helper: build forecast/actual pair for one technology ─────────────────────

def build_series(
    forecast_df: pd.DataFrame,
    actuals_df:  pd.DataFrame,
    tech_name:   str,           # column label to search for in forecast_df
    actual_name: str,           # column label to search for in actuals_df
    out_path:    Path,
) -> pd.DataFrame:

    # Identify forecast column — entsoe-py returns MultiIndex or plain columns
    # depending on version; handle both
    if isinstance(forecast_df.columns, pd.MultiIndex):
        fc_cols = [c for c in forecast_df.columns if tech_name in str(c)]
    else:
        fc_cols = [c for c in forecast_df.columns if tech_name in str(c)]

    if not fc_cols:
        raise ValueError(
            f"Could not find '{tech_name}' in forecast columns: "
            f"{forecast_df.columns.tolist()}"
        )
    fc_series = forecast_df[fc_cols[0]].copy()

    # Identify actuals column
    if isinstance(actuals_df.columns, pd.MultiIndex):
        ac_cols = [c for c in actuals_df.columns if actual_name in str(c)]
    else:
        ac_cols = [c for c in actuals_df.columns if actual_name in str(c)]

    if not ac_cols:
        raise ValueError(
            f"Could not find '{actual_name}' in actuals columns: "
            f"{actuals_df.columns.tolist()}"
        )
    # actuals sometimes has (Actual, Actual Aggregated) sub-columns — take first
    ac_series = actuals_df[ac_cols[0]]
    if isinstance(ac_series, pd.DataFrame):
        ac_series = ac_series.iloc[:, 0]

    # Align on common hourly index
    fc_h = fc_series.resample("h").mean()
    ac_h = ac_series.resample("h").mean()

    combined = pd.DataFrame({
        "Datetime":   fc_h.index,
        "Simulation": fc_h.values,
        "Actuals":    ac_h.reindex(fc_h.index).values,
    }).dropna(subset=["Simulation", "Actuals"])

    combined.to_csv(out_path, index=False)
    print(f"  Saved {len(combined):,} rows → {out_path}")
    return combined


# ── Wind onshore ──────────────────────────────────────────────────────────────
print("\nBuilding wind onshore series...")
wind_df = build_series(
    forecast_raw, actuals_raw,
    tech_name="Wind Onshore",
    actual_name="Wind Onshore",
    out_path=OUT_DIR / "entsoe_wind_onshore_de.csv",
)

# ── Solar ─────────────────────────────────────────────────────────────────────
print("\nBuilding solar series...")
solar_df = build_series(
    forecast_raw, actuals_raw,
    tech_name="Solar",
    actual_name="Solar",
    out_path=OUT_DIR / "entsoe_solar_de.csv",
)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\nDone.")
print(f"Wind onshore: {len(wind_df):,} hourly rows, "
      f"{wind_df['Datetime'].min()} → {wind_df['Datetime'].max()}")
print(f"Solar:        {len(solar_df):,} hourly rows, "
      f"{solar_df['Datetime'].min()} → {solar_df['Datetime'].max()}")
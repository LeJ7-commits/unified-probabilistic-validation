"""
experiments/run_008_report_cards.py
=====================================
Runs Governance_ReportCard for all evaluated datasets, producing
reproducible governance reports in each experiment directory.

For each dataset the script:
  1. Loads the rolling_non_overlapping.csv (primary) and
     rolling_overlapping.csv from the existing experiment directory
  2. Instantiates a ReportCardConfig with dataset-specific parameters
  3. Calls Governance_ReportCard.generate()
  4. Saves all report artifacts alongside existing run artifacts

Outputs per dataset (written into existing experiments/run_XYZ/ dirs):
  report_card_config.json           Reproducibility config
  report_card_window_table.csv      Window labels + reason codes
  report_card_stability.json        Transition matrix + entropy
  report_card_narrative.md          Why-label-changed narrative
  report_card_label_bands.png       Time-series coloured band plot
  report_card_regime_confusion.csv  (if regime_col configured)
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path

from src.governance.report_card import Governance_ReportCard, ReportCardConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR   = REPO_ROOT / "experiments"

# ---------------------------------------------------------------------------
# Dataset registry
# Each entry: (label, run_dir_name, window_size, rolling_step, scheme)
# ---------------------------------------------------------------------------

DATASETS = [
    # Real-data runs — use overlapping for richer band plot
    ("ENTSO-E",    "run_001_entsoe",  250,  50,  "overlapping"),
    ("PV Solar",   "run_002_pv",      720, 168,  "overlapping"),
    ("Wind",       "run_003_wind",    720, 168,  "overlapping"),
    # Simulation — use non-overlapping (only 7 windows)
    ("Sim Price (well-spec)",     "run_004_simulation_price", 250, 250, "non_overlapping"),
    ("Sim Temp (well-spec)",      "run_004_simulation_temp",  250, 250, "non_overlapping"),
    # Misspecification scenarios
    ("Sim Price — Var Inflation", "run_004b_simulation_price_variance_inflation",  250, 250, "non_overlapping"),
    ("Sim Price — Mean Bias",     "run_004b_simulation_price_mean_bias",           250, 250, "non_overlapping"),
    ("Sim Price — Heavy Tails",   "run_004b_simulation_price_heavy_tails",         250, 250, "non_overlapping"),
    ("Sim Temp — Var Inflation",  "run_004b_simulation_temp_variance_inflation",   250, 250, "non_overlapping"),
    ("Sim Temp — Mean Bias",      "run_004b_simulation_temp_mean_bias",            250, 250, "non_overlapping"),
    ("Sim Temp — Heavy Tails",    "run_004b_simulation_temp_heavy_tails",          250, 250, "non_overlapping"),
]


def main() -> None:
    for label, run_dir_name, window_size, step, scheme in DATASETS:
        run_dir  = EXP_DIR / run_dir_name
        csv_name = f"rolling_{scheme}.csv"
        csv_path = run_dir / csv_name

        if not csv_path.exists():
            print(f"[SKIP] {label}: {csv_path.name} not found in {run_dir}")
            continue

        df = pd.read_csv(csv_path)
        if len(df) < 2:
            print(f"[SKIP] {label}: only {len(df)} window(s) — too few for report")
            continue

        config = ReportCardConfig(
            dataset_label=label,
            alpha=0.10,
            window_size=window_size,
            rolling_step=step,
            coverage_target=0.90,
            pvalue_red=0.01,
            pvalue_yellow=0.05,
            coverage_tol_red=0.05,
            coverage_tol_yellow=0.02,
            min_windows_for_transition=5,
            entropy_threshold_unstable=1.2,
            regime_col=None,   # no regime column in current rolling CSVs
        )

        card    = Governance_ReportCard(config)
        outputs = card.generate(rolling_df=df, out_dir=run_dir)

        stab    = outputs.entropy_result.stability_label.upper()
        n_trans = sum(
            1 for i in range(1, len(outputs.window_labels))
            if outputs.window_labels[i] != outputs.window_labels[i - 1]
        )

        print(
            f"[DONE] {label:<42}  "
            f"n={len(outputs.window_labels)}  "
            f"stability={stab:8s}  "
            f"H={outputs.entropy_result.stationary_entropy:.3f}bits  "
            f"transitions={n_trans}  "
            f"artifacts={len(outputs.saved_paths)}"
        )

    print("\nAll report cards complete.")


if __name__ == "__main__":
    main()

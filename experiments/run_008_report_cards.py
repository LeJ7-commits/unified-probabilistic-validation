"""
experiments/run_008_report_cards.py
=====================================
Runs Governance_ReportCard + NarrativeGenerator for all evaluated datasets.

Outputs per dataset:
  report_card_config.json
  report_card_window_table.csv
  report_card_stability.json
  report_card_narrative.md
  report_card_label_bands.png
  narrative_technical.md        (AI or stub)
  narrative_plain.md            (AI or stub)
  narrative_combined.md         (AI or stub)

NarrativeGenerator requires ANTHROPIC_API_KEY environment variable.
If not set, stub narratives are written (clearly marked).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from src.governance.report_card import Governance_ReportCard, ReportCardConfig
from src.governance.narrative_generator import NarrativeGenerator

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR   = REPO_ROOT / "experiments"

DATASETS = [
    ("ENTSO-E",    "run_001_entsoe",  250,  50,  "overlapping",     "short_term",  "ENTSO-E hourly electricity load forecast for Germany"),
    ("PV Solar",   "run_002_pv",      720, 168,  "overlapping",     "long_term",   "PV solar hourly generation forecast"),
    ("Wind",       "run_003_wind",    720, 168,  "overlapping",     "long_term",   "onshore wind hourly generation forecast"),
    ("Sim Price (well-spec)",     "run_004_simulation_price", 250, 250, "non_overlapping", "simulation", "Monte Carlo energy price simulation (well-specified positive control)"),
    ("Sim Temp (well-spec)",      "run_004_simulation_temp",  250, 250, "non_overlapping", "simulation", "Monte Carlo temperature simulation (well-specified positive control)"),
    ("Sim Price — Var Inflation", "run_004b_simulation_price_variance_inflation",  250, 250, "non_overlapping", "simulation", "Monte Carlo price simulation with variance inflation misspecification"),
    ("Sim Price — Mean Bias",     "run_004b_simulation_price_mean_bias",           250, 250, "non_overlapping", "simulation", "Monte Carlo price simulation with mean bias misspecification"),
    ("Sim Price — Heavy Tails",   "run_004b_simulation_price_heavy_tails",         250, 250, "non_overlapping", "simulation", "Monte Carlo price simulation with heavy-tails misspecification"),
    ("Sim Temp — Var Inflation",  "run_004b_simulation_temp_variance_inflation",   250, 250, "non_overlapping", "simulation", "Monte Carlo temperature simulation with variance inflation misspecification"),
    ("Sim Temp — Mean Bias",      "run_004b_simulation_temp_mean_bias",            250, 250, "non_overlapping", "simulation", "Monte Carlo temperature simulation with mean bias misspecification"),
    ("Sim Temp — Heavy Tails",    "run_004b_simulation_temp_heavy_tails",          250, 250, "non_overlapping", "simulation", "Monte Carlo temperature simulation with heavy-tails misspecification"),
]


def _load_governance_decision(run_dir: Path) -> dict | None:
    path = run_dir / "governance_decision.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    narrator = NarrativeGenerator()

    if api_key:
        print("NarrativeGenerator: ANTHROPIC_API_KEY found — AI narratives enabled.\n")
    else:
        print("NarrativeGenerator: ANTHROPIC_API_KEY not set — stub narratives will be written.")
        print("  Set ANTHROPIC_API_KEY to enable AI-generated narratives.\n")

    for label, run_dir_name, window_size, step, scheme, model_class, commodity in DATASETS:
        run_dir  = EXP_DIR / run_dir_name
        csv_path = run_dir / f"rolling_{scheme}.csv"

        if not csv_path.exists():
            print(f"[SKIP] {label}: {csv_path.name} not found")
            continue

        df = pd.read_csv(csv_path)
        if len(df) < 2:
            print(f"[SKIP] {label}: only {len(df)} window(s)")
            continue

        # ── Report card ───────────────────────────────────────────────────
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
            regime_col=None,
        )
        card    = Governance_ReportCard(config)
        outputs = card.generate(rolling_df=df, out_dir=run_dir)

        stab    = outputs.entropy_result.stability_label.upper()
        n_trans = sum(
            1 for i in range(1, len(outputs.window_labels))
            if outputs.window_labels[i] != outputs.window_labels[i - 1]
        )

        # ── NarrativeGenerator ────────────────────────────────────────────
        decision_dict    = _load_governance_decision(run_dir)
        narrative_status = "no governance_decision.json"

        if decision_dict is not None:
            class _DecisionProxy:
                def __init__(self, d):
                    self.model_id     = d["model_id"]
                    self.final_label  = d["final_label"]
                    self.reason_codes = d["reason_codes"]
                    self._d = d
                def to_dict(self): return self._d

            proxy  = _DecisionProxy(decision_dict)
            result = narrator.generate(
                proxy,
                model_class       = model_class,
                commodity_context = commodity,
            )
            narrator.save(result, out_dir=run_dir)
            narrative_status = "AI" if result.api_used else "stub"

        print(
            f"[DONE] {label:<42}  "
            f"n={len(outputs.window_labels)}  "
            f"stability={stab:8s}  "
            f"H={outputs.entropy_result.stationary_entropy:.3f}bits  "
            f"transitions={n_trans}  "
            f"artifacts={len(outputs.saved_paths)}  "
            f"narrative={narrative_status}"
        )

    print("\nAll report cards complete.")


if __name__ == "__main__":
    main()

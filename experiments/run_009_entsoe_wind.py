"""
experiments/run_009_entsoe_wind.py
====================================
Probabilistic validation diagnostics for ENTSO-E onshore wind (Germany, 2020-2026).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import json

from src.diagnostics.run_policy import run_diagnostics_policy, write_run_artifacts
from src.diagnostics.diagnostics_input import Diagnostics_Input
from src.governance.decision_engine import DecisionEngine
from src.governance.risk_classification import RiskPolicy

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR  = REPO_ROOT / "data" / "derived_entsoe_wind"
OUT_DIR   = REPO_ROOT / "experiments" / "run_009_entsoe_wind"
ALPHA     = 0.1
MODEL_ID  = "entsoe_wind_de"


def main() -> None:
    print("=" * 60)
    print("Running ENTSO-E Wind (Germany) diagnostics")
    print("=" * 60)

    y       = np.load(DATA_DIR / "entsoe_wind_y.npy").astype(float)
    lower   = np.load(DATA_DIR / "entsoe_wind_lo_base_90.npy").astype(float)
    upper   = np.load(DATA_DIR / "entsoe_wind_hi_base_90.npy").astype(float)
    samples = np.load(DATA_DIR / "entsoe_wind_samples.npy").astype(float)

    quantiles = {ALPHA / 2: lower, 1 - ALPHA / 2: upper}

    run_out = run_diagnostics_policy(
        model_class     = "short_term",
        y_true          = y,
        samples         = samples,
        quantiles       = quantiles,
        alpha           = ALPHA,
        rolling_window  = 250,
        rolling_step    = 50,
        lb_lags         = (5, 10, 20),
        coverage_target = 0.90,
    )

    paths = write_run_artifacts(
        out_dir         = OUT_DIR,
        run_output      = run_out,
        alpha           = ALPHA,
        y_true          = y,
        quantiles       = quantiles,
        coverage_target = 0.90,
    )

    print("Run completed. Artifacts:")
    for k, v in paths.items():
        print(f"  {k}: {v}")

    di  = Diagnostics_Input(alpha=ALPHA)
    dro = di.from_arrays(
        y         = y,
        t         = np.arange(len(y)),
        model_id  = MODEL_ID,
        samples   = samples,
        lo        = lower,
        hi        = upper,
        quantiles = quantiles,
    )

    engine   = DecisionEngine(alpha=ALPHA, global_policy=RiskPolicy(coverage_target=0.90))
    decision = engine.decide(dro)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "governance_decision.json", "w") as f:
        json.dump(decision.to_dict(), f, indent=2, ensure_ascii=False)

    codes = [rc.value if hasattr(rc, "value") else str(rc) for rc in decision.reason_codes]
    print(f"\nGovernance decision : {decision.final_label}")
    print(f"Reason codes        : {codes}")
    print(f"Decision artifact   : {OUT_DIR / 'governance_decision.json'}")


if __name__ == "__main__":
    main()

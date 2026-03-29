"""
experiments/run_011_sim_elec_price.py
======================================
Probabilistic validation for the extended simulation — electricity price series.

Model class : Class 1 — Simulation (positive control, well-specified DGP)
Series      : elec_price  (base=55 €/MWh, σ=12, high intraday amplitude)
n           : 365 as-of dates (h=1 evaluation)
Expected    : GREEN — realised values drawn from same DGP as simulation paths

Part of the five-dimensional extended simulation (elec_price, nat_gas,
carbon, price, temp), built by scripts/build_simulation_extended_derived.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.diagnostics.run_policy import run_diagnostics_policy, write_run_artifacts
from src.diagnostics.diagnostics_input import Diagnostics_Input
from src.governance.decision_engine import DecisionEngine
from src.governance.risk_classification import RiskPolicy

ALPHA           = 0.10
COVERAGE_TARGET = 0.90
SERIES          = "elec_price"

REPO     = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data" / f"derived_simulation_{SERIES}"
OUT_DIR  = REPO / "experiments" / f"run_011_sim_{SERIES}"


if __name__ == "__main__":
    print("=" * 60)
    print(f"  Simulation — {SERIES} (run_011)")
    print("=" * 60)

    y     = np.load(DATA_DIR / f"{SERIES}_y.npy").astype(float)
    yhat  = np.load(DATA_DIR / f"{SERIES}_yhat.npy").astype(float)
    lower = np.load(DATA_DIR / f"{SERIES}_lo_base_90.npy").astype(float)
    upper = np.load(DATA_DIR / f"{SERIES}_hi_base_90.npy").astype(float)

    quantiles = {ALPHA / 2: lower, 1 - ALPHA / 2: upper}
    samples   = None   # simulation class: interval-only (no sample .npy)

    run_out = run_diagnostics_policy(
        model_class            = "simulation",
        y_true                 = y,
        samples                = samples,
        quantiles              = quantiles,
        alpha                  = ALPHA,
        rolling_window         = 250,
        rolling_step           = 250,
        enable_rolling_for_long_term = False,
        lb_lags                = (5, 10, 20),
        coverage_target        = COVERAGE_TARGET,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = write_run_artifacts(
        out_dir         = OUT_DIR,
        run_output      = run_out,
        alpha           = ALPHA,
        y_true          = y,
        quantiles       = quantiles,
        coverage_target = COVERAGE_TARGET,
    )
    print("  Artifacts written:")
    for k, v in paths.items():
        print(f"    {k}: {v}")

    # DecisionEngine
    di  = Diagnostics_Input(alpha=ALPHA)
    dro = di.from_arrays(
        y         = y,
        t         = np.arange(len(y)),
        model_id  = f"simulation_{SERIES}",
        lo        = lower,
        hi        = upper,
        quantiles = quantiles,
        samples   = samples,
    )
    engine   = DecisionEngine(alpha=ALPHA,
                               global_policy=RiskPolicy(coverage_target=COVERAGE_TARGET))
    decision = engine.decide(dro)

    decision_path = OUT_DIR / "governance_decision.json"
    with open(decision_path, "w", encoding="utf-8") as f:
        json.dump(decision.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"\n  Governance decision : {decision.final_label}")
    print(f"  Reason codes        : "
          f"{[rc.value if hasattr(rc, 'value') else str(rc) for rc in decision.reason_codes]}")
    print(f"  Decision artifact   : {decision_path}")

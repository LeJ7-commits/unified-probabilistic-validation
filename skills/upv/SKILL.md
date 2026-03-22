---
name: unified_probabilistic_validation
description: "Use this skill whenever a user wants to run the Unified Probabilistic Validation Framework pipeline, validate a new energy market model, interpret governance results, add a new commodity class, or understand why a model received a RED/YELLOW/GREEN classification. Triggers include: 'run the pipeline', 'validate my model', 'add a new dataset', 'why is my model RED', 'generate narratives', 'run all experiments'. Also use when the user asks about any component of the framework (DataContract, DecisionEngine, NarrativeGenerator, RegimeTagger, etc)."
license: MIT
---

# Unified Probabilistic Validation Framework

## Overview

This framework validates probabilistic energy market models by running a
battery of diagnostics (PIT uniformity, Ljung-Box independence, Anfuso
interval backtesting, CRPS, pinball loss, interval sharpness) and producing
a Basel-style GREEN/YELLOW/RED governance classification with full provenance.

**Repository:** `LeJ7-commits/unified-probabilistic-validation`
**Authors:** Jia Yang Le, Komila Askarova
**Industry Partner:** Energy Quant Solutions Sweden AB (EnBW)

---

## Quick Reference

| Task | Command |
|------|---------|
| Run full pipeline | `python run_all.py` |
| Run full pipeline (skip build) | `python run_all.py --skip-build` |
| Run specific stages | `python run_all.py --stages 2,3,10` |
| Dry run (preview) | `python run_all.py --dry-run` |
| Run single experiment | `python experiments/run_001_entsoe.py` |
| Run report cards + narratives | `python experiments/run_008_report_cards.py` |
| Run all tests | `python -m pytest tests/ -q` |
| Enable AI narratives | `set ANTHROPIC_API_KEY=<your-key>` (Windows) or `export ANTHROPIC_API_KEY=<your-key>` (Mac/Linux) |

---

## Pipeline Stages

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `scripts/build_*.py` | Build derived `.npy` artifacts from raw CSV |
| 2 | `experiments/run_001_entsoe.py` | ENTSO-E load diagnostics + governance |
| 3 | `experiments/run_002_pv.py` | PV solar diagnostics + governance |
| 4 | `experiments/run_003_wind.py` | Wind diagnostics + governance |
| 5 | `experiments/run_004_simulation.py` | Simulation positive control |
| 6 | `experiments/run_004b_simulation_misspec.py` | Misspecification scenarios |
| 7 | `experiments/run_005_multivariate.py` | Joint PV+Wind diagnostics |
| 8 | `experiments/run_006_var_sensitivity.py` | VaR capital distortion |
| 9 | `experiments/run_007_transition_metrics.py` | Rolling label stability |
| 10 | `experiments/run_008_report_cards.py` | Report cards + AI narratives |

---

## Architecture (component map)

```
Raw Data (CSV)
    │
    ▼
DataContract          src/core/data_contract.py
    │  validates schema, timestamps, NaN, quantile crossing
    ▼
Adapters
  Adapter_PointForecast    src/adapters/point_forecast.py
  Adapter_SimulationJoint  src/adapters/simulation_joint.py
  Adapter_Quantiles        src/adapters/quantile_adapter.py
    │  produce: ResidualPool / JointSimulationObject / QuantileFunctionObject
    ▼
Diagnostics_Input     src/diagnostics/diagnostics_input.py
    │  auto-detects capabilities (PIT, CRPS, pinball, interval, energy score)
    ▼
Diagnostic branches
  PIT + Ljung-Box      src/calibration/pit.py + diagnostics.py
  Score_Pinball        src/scoring/pinball.py
  Interval_Sharpness   src/diagnostics/interval_sharpness.py
  Anfuso backtest      src/governance/anfuso.py
  CRPS                 src/scoring/crps.py
    │
    ▼
RegimeTagger          src/governance/regime_tagger.py
    │  seasonal / volatility / break-flag rules
    ▼
ThresholdCalibrator   src/governance/threshold_calibrator.py
    │  regime-conditioned GREEN/YELLOW/RED thresholds
    ▼
DecisionEngine        src/governance/decision_engine.py
    │  single .decide() call → GovernanceDecision with full provenance
    ▼
NarrativeGenerator    src/governance/narrative_generator.py
    │  AI-generated technical + plain language narratives
    ▼
Governance_ReportCard src/governance/report_card.py
    │  rolling window table, stability, label band PNG
    ▼
Artifacts (experiments/run_XYZ/)
  full_sample_metrics.json
  governance_decision.json
  narrative_technical.md
  narrative_plain.md
  report_card_*.{json,csv,png,md}
```

---

## Adding a new dataset / commodity class

To add a new commodity (e.g. natural gas, carbon):

### Step 1 — Prepare data
Place the raw CSV in `data/` with columns: `timestamp`, `y`, `y_hat`
(or `Datetime`, `Actuals`, `Simulation` for renewables format).

### Step 2 — Build derived artifacts
```bash
python scripts/build_entsoe_derived.py   # for point-forecast models
# OR
python scripts/build_renewables_derived.py  # for renewables
```

### Step 3 — Create a run script
Copy `experiments/run_001_entsoe.py` to `experiments/run_009_newcommodity.py`.
Update:
- `data_dir` to point to new derived data
- `model_id` in the DecisionEngine block
- `model_class` and `commodity_context`

### Step 4 — Add to run_008
Add an entry to the `DATASETS` list in `experiments/run_008_report_cards.py`.

### Step 5 — Add to run_all.py
Add a new stage entry to `STAGES` in `run_all.py`.

### Step 6 — Run
```bash
python run_all.py --stages 9,10
```

---

## Interpreting governance decisions

### Traffic light labels
| Label | Meaning | Action |
|-------|---------|--------|
| GREEN | All diagnostic signals within policy thresholds | No action required |
| YELLOW | One or more signals borderline | Monitor closely; consider recalibration |
| RED | One or more signals strongly rejected | Model requires remediation before use in production |

### Reason codes
| Code | What failed |
|------|-------------|
| `undercoverage` | Empirical coverage < nominal − tolerance |
| `overcoverage` | Empirical coverage > nominal + tolerance |
| `PIT_uniformity_fail` | KS or CvM test rejects PIT uniformity (p < 0.05) |
| `ACF_dependence_fail` | Ljung-Box rejects PIT independence (p < 0.05) |
| `all_clear` | No issues detected |

### Capital implications (Basel adaptation)
| Governance zone | Capital multiplier | Reserve sizing |
|-----------------|-------------------|----------------|
| GREEN | 3.00× | At nominal |
| YELLOW | 3.40× | +13.3% |
| RED | 4.00× | +33.3% |

---

## NarrativeGenerator

Requires `ANTHROPIC_API_KEY` environment variable.

```python
from src.governance.narrative_generator import NarrativeGenerator
from src.governance.decision_engine import DecisionEngine

engine   = DecisionEngine()
decision = engine.decide(dro)

narrator = NarrativeGenerator()
result   = narrator.generate(
    decision,
    model_class       = "short_term",
    commodity_context = "ENTSO-E electricity load",
)
# result.technical_narrative  → for risk officers
# result.plain_narrative      → for management / regulators

narrator.save(result, out_dir=Path("experiments/run_001_entsoe"))
# writes: narrative_technical.md, narrative_plain.md, narrative_combined.md
```

**Cost:** ~$0.005 per dataset. Full pipeline across 11 datasets ≈ $0.05 total.

---

## Troubleshooting

### `FileNotFoundError: entsoe_lo_base_90.npy`
Run `scripts/build_entsoe_derived.py` first. Derived artifacts must be
built before running experiment scripts.

### `ANTHROPIC_API_KEY not found`
Set the environment variable. Stub narratives (clearly marked) are written
if the key is absent — the pipeline still runs to completion.

### `ModuleNotFoundError: src.governance.narrative_generator`
The `narrative_generator.py` file must be in `src/governance/`. Check
that `src/governance/__init__.py` imports it (or import directly).

### Tests failing
```bash
python -m pytest tests/ -q --tb=short
```
All 427 tests should pass. If a test fails after adding a new component,
check that the new src file was placed in the correct directory.

---

## File structure

```
unified-probabilistic-validation/
├── run_all.py                        Full pipeline orchestrator
├── src/
│   ├── core/
│   │   └── data_contract.py          DataContract + StandardizedModelObject
│   ├── adapters/
│   │   ├── point_forecast.py         Adapter_PointForecast
│   │   ├── simulation_joint.py       Adapter_SimulationJoint
│   │   └── quantile_adapter.py       Adapter_Quantiles + PAVA
│   ├── diagnostics/
│   │   ├── diagnostics_input.py      Diagnostics_Input gateway
│   │   ├── interval_sharpness.py     Interval_Sharpness
│   │   ├── evaluator.py              evaluate_distribution
│   │   ├── rolling.py                rolling_evaluation
│   │   └── run_policy.py             run_diagnostics_policy
│   ├── calibration/
│   │   ├── pit.py                    PIT computation + GOF tests
│   │   └── diagnostics.py            pit_uniformity_tests wrappers
│   ├── scoring/
│   │   ├── crps.py                   CRPS sample-based
│   │   └── pinball.py                Score_Pinball
│   └── governance/
│       ├── anfuso.py                 Anfuso interval backtest
│       ├── reason_codes.py           ReasonCode enum
│       ├── risk_classification.py    classify_risk + RiskPolicy
│       ├── stability.py              Stability_TransitionMatrix
│       ├── report_card.py            Governance_ReportCard
│       ├── regime_tagger.py          RegimeTagger + SeasonalRule etc
│       ├── threshold_calibrator.py   ThresholdCalibrator
│       ├── decision_engine.py        DecisionEngine (top-level orchestrator)
│       └── narrative_generator.py    NarrativeGenerator (AI narratives)
├── experiments/
│   ├── run_001_entsoe.py
│   ├── run_002_pv.py
│   ├── run_003_wind.py
│   ├── run_004_simulation.py
│   ├── run_004b_simulation_misspec.py
│   ├── run_005_multivariate.py
│   ├── run_006_var_sensitivity.py
│   ├── run_007_transition_metrics.py
│   └── run_008_report_cards.py
├── scripts/
│   ├── build_entsoe_derived.py
│   ├── build_renewables_derived.py
│   └── build_simulation_derived.py
├── tests/                            427 pytest tests
├── data/                             Raw CSV and derived .npy artifacts
└── skills/
    └── upv/
        └── SKILL.md                  This file
```

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
**Live app:** `unified-probabilistic-validation.streamlit.app`
**Authors:** Jia Yang Le, Komila Askarova
**Supervisor:** Luca Margaritella (Lund University LUSEM)
**Industry Partner:** Rikard Green, Energy Quant Solutions Sweden AB
**Thesis deadline:** 2026-05-27

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
| Build extended simulation | `python scripts/build_simulation_extended_derived.py` |
| Run all tests | `python -m pytest tests/ -q` |
| Build Word doc | `python build_thesis.py` |
| Enable AI narratives | `set ANTHROPIC_API_KEY=<your-key>` (Windows) |

---

## Pipeline Stages (18 total)

| Stage | Script | Description | Optional |
|-------|--------|-------------|----------|
| 1 | `scripts/build_*.py` (5 scripts) | Build all derived `.npy` artifacts from raw CSV | No |
| 2 | `experiments/run_001_entsoe.py` | ENTSO-E load diagnostics + governance | No |
| 3 | `experiments/run_002_pv.py` | PV solar diagnostics + governance | No |
| 4 | `experiments/run_003_wind.py` | Wind diagnostics + governance | No |
| 5 | `experiments/run_004_simulation.py` | Simulation positive control (price + temp) | No |
| 6 | `experiments/run_004b_simulation_misspec.py` | Misspecification scenarios (3 types × 2 series) | No |
| 7 | `experiments/run_011_sim_elec_price.py` | Extended simulation — electricity price | No |
| 8 | `experiments/run_012_sim_nat_gas.py` | Extended simulation — natural gas | No |
| 9 | `experiments/run_013_sim_carbon.py` | Extended simulation — carbon CO2 | No |
| 10 | `experiments/run_005_multivariate.py` | Joint PV+Wind PIT dependence + energy score | No |
| 11 | `experiments/run_014_multivariate_extended.py` | Joint breach correlation across 5 simulation series | No |
| 12 | `experiments/run_006_var_sensitivity.py` | VaR capital multiplier distortion + reserve sizing | No |
| 13 | `experiments/run_007_transition_metrics.py` | Rolling label stability + transition matrices | No |
| 14 | `experiments/run_008_report_cards.py` | Report cards + AI narratives (13 datasets) | No |
| 15 | `experiments/run_009_entsoe_wind.py` | ENTSO-E Wind Germany 2020–2026 | No |
| 16 | `experiments/run_010_entsoe_solar.py` | ENTSO-E Solar Germany 2020–2026 | No |
| 17 | `experiments/run_009b_entsoe_wind_daily.py` | Wind daily aggregation robustness | Yes |
| 18 | `experiments/run_010b_entsoe_solar_daily.py` | Solar daily aggregation robustness | Yes |

**Build scripts in Stage 1:**
- `build_entsoe_derived.py` — ENTSO-E load
- `build_renewables_derived.py` — PV + wind student datasets
- `build_entsoe_renewables_derived.py` — ENTSO-E Wind/Solar Germany
- `build_simulation_derived.py` — price + temp (2D DGP)
- `build_simulation_extended_derived.py` — elec_price + nat_gas + carbon (5D DGP)

---

## Empirical Results Summary

| Dataset | n | Coverage | Anfuso | KS stat | Overall |
|---------|---|----------|--------|---------|---------|
| ENTSO-E load (run_001) | 209,555 | 87.06% | RED | 0.1615 | **RED** |
| PV Solar (run_002) | 4,287 | 91.37% | GREEN | 0.1028 | **RED** |
| Wind (run_003) | 9,000 | 88.62% | RED (lower) | 0.1057 | **RED** |
| Sim price (run_004) | 365 | 88.49% | GREEN | n/a | **GREEN** |
| Sim temp (run_004) | 365 | 89.59% | GREEN | n/a | **GREEN** |
| Sim elec_price (run_011) | 365 | 92.05% | GREEN | n/a | **YELLOW** |
| Sim nat_gas (run_012) | 365 | 91.51% | GREEN | n/a | **GREEN** |
| Sim carbon (run_013) | 365 | 91.00% | GREEN | n/a | **GREEN** |
| ENTSO-E Wind DE (run_009) | 51,933 | 89.16% | GREEN | 0.0083† | **RED†** |
| ENTSO-E Solar DE (run_010) | 51,933 | 89.94% | GREEN | 0.0258† | **RED†** |

† RED driven by large-n statistical sensitivity (KS below 0.05 effect-size floor),
not substantive distributional failure. ACF remains genuinely elevated.

**Key findings:**
- PIT diagnostics fail universally on real data regardless of coverage (PV passes Anfuso but fails PIT — key multi-layer validation argument)
- Positive control GREEN confirmed across 5/6 extended simulation series
- Dual-criterion rule (KS effect-size floor + ACF floor) implemented in `src/governance/risk_classification.py`
- Daily aggregation (run_009b/010b) confirms horizon-specificity of rolling quantile reconstruction
- 5D multivariate extended simulation (run_014): joint coverage 63.0% vs 59.1% expected; price×temp co-failure 3.29% (reflects ρ=0.5 DGP)

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
  BuildDist_FromResiduals  src/adapters/build_dist_from_residuals.py
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
    │  dual-criterion large-n guard: KS effect-size floor + ACF floor
    ▼
NarrativeGenerator    src/governance/narrative_generator.py
    │  AI-generated technical + plain language narratives via Anthropic API
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

### Step 1 — Prepare data
Place raw CSV in `data/` with columns: `timestamp`, `y`, `y_hat`
(or `Datetime`, `Actuals`, `Simulation` for renewables format).

### Step 2 — Build derived artifacts
```bash
python scripts/build_entsoe_derived.py      # point-forecast models
# OR
python scripts/build_renewables_derived.py  # renewables (PV/wind)
# OR
python scripts/build_simulation_extended_derived.py  # new simulation DGP
```

### Step 3 — Create a run script
Copy the nearest existing run script. Update:
- `DATA_DIR` to point to new derived data directory
- `SERIES` name and file prefix
- `model_id` in the `Diagnostics_Input` call
- `model_class` (`"short_term"`, `"long_term"`, or `"simulation"`)

### Step 4 — Add to run_008
Add an entry to `DATASETS` in `experiments/run_008_report_cards.py`.
Note: simulation class runs skip rolling report cards (insufficient windows at n=365).

### Step 5 — Add to run_all.py
Add a new stage entry to `STAGES` list. Renumber subsequent stages.

### Step 6 — Run
```bash
python run_all.py --stages <new_stage>,14
```

---

## Interpreting governance decisions

### Traffic light labels
| Label | Meaning | Action |
|-------|---------|--------|
| GREEN | All diagnostic signals within policy thresholds | No action required |
| YELLOW | Borderline signal (typically sampling noise at small n) | Monitor; consider extending evaluation window |
| RED | One or more signals strongly rejected | Remediation required before production use |

### Reason codes
| Code | What failed |
|------|-------------|
| `undercoverage` | Empirical coverage < nominal − tolerance |
| `coverage_warn` | Coverage outside ±2 pp band (YELLOW trigger) |
| `PIT_uniformity_fail` | KS/CvM/AD test rejects PIT uniformity (p < 0.05 AND KS > 0.05 effect-size floor) |
| `ACF_dependence_fail` | Ljung-Box rejects PIT independence (p < 0.05 AND ACF lag-1 > 0.05 floor) |
| `all_clear` | No issues detected |

### Large-n dual-criterion rule
At n > 20,000, PIT tests achieve near-infinite power. The framework uses:
- **KS effect-size floor = 0.05:** KS < 0.05 → downgrade to WARN (not FAIL)
- **ACF lag-1 floor = 0.05:** ACF > 0.05 → genuine FAIL regardless of n

Both thresholds implemented in `src/governance/risk_classification.py`.

### Capital implications (Basel adaptation)
| Governance zone | Capital multiplier | Reserve sizing |
|-----------------|-------------------|----------------|
| GREEN | 3.00× | At nominal |
| YELLOW | 3.40× | +13.3% |
| RED | 4.00× | +33.3% |

### Governance action protocol
| Label | Immediate action |
|-------|-----------------|
| RED (interval + PIT + independence) | Suspend model; apply conformal expansion; escalate |
| RED (PIT only, interval OK) | Flag tail quantiles unreliable; schedule structural review |
| RED (tail asymmetry) | Directional conformal correction; investigate physical cause |
| YELLOW | Increase monitoring frequency; extend evaluation window if n < 500 |
| GREEN | Standard governance calendar (quarterly review) |

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

**Cost:** ~$0.005 per dataset. Full pipeline across 13 datasets ≈ $0.065 total.
If no API key is set, stub narratives are written and pipeline continues.

---

## Figures (in `figures/`)

| File | Used in | Description |
|------|---------|-------------|
| `pit_diagnostics_entsoe.png` | 03_results.md §1.2 | 4-panel PIT for ENTSO-E |
| `pit_diagnostics_pv.png` | 03_results.md §2.2 | 4-panel PIT for PV Solar |
| `pit_diagnostics_wind.png` | 03_results.md §3.2 | 4-panel PIT for Wind |
| `pit_diagnostics_sim.png` | 03_results.md §4b | 4-panel PIT for Simulation |
| `model_diagnostic_positioning.png` | 03_results.md §7.3 | KS/ACF vs n positioning |
| `power_vs_n.png` | 04_discussion.md §4.6 | Theoretical power curves |

Generate all figures: run `experiments/run_002_pv.py` through `run_004_simulation.py`
plus `scripts/generate_figures.py` (if it exists), or regenerate via `build_thesis.py`.

---

## Streamlit App

**URL:** `unified-probabilistic-validation.streamlit.app`

Features:
- CSV upload with auto column detection
- α toggle (0.05 / 0.10 / 0.20) → live coverage target update
- Rolling window size (50–1000) and step (10–250) controls
- Distribution reconstruction toggle (non_parametric / parametric)
- Sample paths M toggle (100 / 200 / 500)
- PIT 4-panel diagnostic plots (inline, generated from uploaded data)
- Rolling coverage time series chart
- Power vs n theoretical figure (on-demand, ~20s, dynamic n marker)
- Governance decision with GREEN/YELLOW/RED verdict block
- AI narrative (technical + plain language) via Anthropic API
- ZIP download of all artifacts

**Deployment:** push to GitHub → Streamlit Cloud auto-redeploys (~60s).
Set `ANTHROPIC_API_KEY` in Streamlit Cloud Settings → Secrets for AI narratives.

---

## Troubleshooting

### `FileNotFoundError: *_lo_base_90.npy`
Run the appropriate build script first. All `experiments/run_*.py` scripts
require derived `.npy` artifacts to exist in `data/derived_*/`.

### `ANTHROPIC_API_KEY not found`
Set the environment variable. Stub narratives are written if absent —
pipeline still runs to completion.

### `ModuleNotFoundError: src.*`
Run scripts from the repo root. If running from a subdirectory, add the
repo root to `PYTHONPATH`.

### `rolling_overlapping.csv not found` in run_008
Simulation runs (run_011/012/013) don't generate rolling CSVs —
they use `model_class="simulation"` which skips rolling evaluation.
This is expected and produces a SKIP in run_008 output.

### Streamlit: `StreamlitAPIException` on `icon=`
Streamlit 1.55+ only accepts standard emoji for `icon=` parameter.
Remove the `icon=` argument from `st.warning()` calls entirely.

### Streamlit: `keyboard_double` in sidebar
Non-standard Unicode characters (e.g. `▣`, `▸`) in `st.markdown("### ...")` 
headers are intercepted by Streamlit as Material icon names. Use plain text
or standard emoji only in sidebar headers.

### Tests failing
```bash
python -m pytest tests/ -q --tb=short
```
451+ tests should pass. If a test fails after adding a new component,
check that the new file is in the correct `src/` subdirectory and that
`__init__.py` imports are consistent.

---

## File structure

```
unified-probabilistic-validation/
├── run_all.py                              Full pipeline orchestrator (18 stages)
├── build_thesis.py                         Assembles Word doc from papers/
├── app.py                                  Streamlit app
├── src/
│   ├── core/
│   │   └── data_contract.py
│   ├── adapters/
│   │   ├── point_forecast.py
│   │   ├── simulation_joint.py
│   │   ├── quantile_adapter.py
│   │   └── build_dist_from_residuals.py
│   ├── diagnostics/
│   │   ├── diagnostics_input.py
│   │   ├── interval_sharpness.py
│   │   ├── evaluator.py
│   │   ├── rolling.py
│   │   └── run_policy.py
│   ├── calibration/
│   │   ├── pit.py
│   │   └── diagnostics.py
│   ├── scoring/
│   │   ├── crps.py
│   │   └── pinball.py
│   └── governance/
│       ├── anfuso.py
│       ├── reason_codes.py
│       ├── risk_classification.py          Dual-criterion large-n guard
│       ├── stability.py
│       ├── report_card.py
│       ├── regime_tagger.py
│       ├── threshold_calibrator.py
│       ├── decision_engine.py
│       └── narrative_generator.py
├── experiments/
│   ├── run_001_entsoe.py                   ENTSO-E load (RED)
│   ├── run_002_pv.py                       PV Solar (RED)
│   ├── run_003_wind.py                     Wind (RED)
│   ├── run_004_simulation.py               Positive control price+temp (GREEN)
│   ├── run_004b_simulation_misspec.py      Misspecification scenarios
│   ├── run_005_multivariate.py             PV+Wind joint diagnostics
│   ├── run_006_var_sensitivity.py          VaR capital distortion
│   ├── run_007_transition_metrics.py       Rolling label stability
│   ├── run_008_report_cards.py             Report cards + AI narratives (13 datasets)
│   ├── run_009_entsoe_wind.py              ENTSO-E Wind DE (RED†)
│   ├── run_009b_entsoe_wind_daily.py       Wind daily robustness (optional)
│   ├── run_010_entsoe_solar.py             ENTSO-E Solar DE (RED†)
│   ├── run_010b_entsoe_solar_daily.py      Solar daily robustness (optional)
│   ├── run_011_sim_elec_price.py           Extended sim electricity price (YELLOW)
│   ├── run_012_sim_nat_gas.py              Extended sim natural gas (GREEN)
│   ├── run_013_sim_carbon.py               Extended sim carbon CO2 (GREEN)
│   └── run_014_multivariate_extended.py    5D simulation joint breach analysis
├── scripts/
│   ├── build_entsoe_derived.py
│   ├── build_renewables_derived.py
│   ├── build_entsoe_renewables_derived.py
│   ├── build_simulation_derived.py
│   └── build_simulation_extended_derived.py
├── papers/
│   ├── 00_abstract.md
│   ├── 01_introduction.md
│   ├── 02_methodology.md
│   ├── 03_results.md
│   ├── 04_discussion.md
│   ├── 05_governance_implications.md
│   └── 06_references_and_ai_statement.md
├── figures/
│   ├── pit_diagnostics_entsoe.png
│   ├── pit_diagnostics_pv.png
│   ├── pit_diagnostics_wind.png
│   ├── pit_diagnostics_sim.png
│   ├── model_diagnostic_positioning.png
│   └── power_vs_n.png
├── thesis_output/
│   └── Unified_Probabilistic_Validation_Le_Askarova_2026.docx
├── tests/                                  451+ pytest tests
├── data/
│   ├── derived_entsoe_full/
│   ├── derived_pv/
│   ├── derived_wind/
│   ├── derived_simulation_price/
│   ├── derived_simulation_temp/
│   ├── derived_simulation_elec_price/
│   ├── derived_simulation_nat_gas/
│   ├── derived_simulation_carbon/
│   ├── derived_entsoe_wind/
│   └── derived_entsoe_solar/
└── skills/
    └── upv/
        └── SKILL.md                        This file
```

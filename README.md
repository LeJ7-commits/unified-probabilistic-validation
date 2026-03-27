# Unified Probabilistic Validation Framework

A production-grade reliability architecture for probabilistic energy market models.

**Authors:** Jia Yang Le & Komila Askarova  
**Institution:** Lund University 
**Industry Partner:** Energy Quant Solutions Sweden AB (also in collaboration with EnBW Group)  
**Live Demo:** [unified-probabilistic-validation.streamlit.app](https://unified-probabilistic-validation.streamlit.app)

---

## What this is

A unified statistical validation framework that evaluates heterogeneous energy market model classes — Monte Carlo simulation, short-term operational forecasting, long-term renewable generation — within a shared probabilistic reliability space.

**451 passing tests. 11 components. 3 model classes. 1 governance decision per dataset.**

---

## Quick start
```bash
# Clone and install
git clone https://github.com/LeJ7-commits/unified-probabilistic-validation
pip install -r requirements.txt

# Run full pipeline (skips build if derived data exists)
python run_all.py --skip-build

# Or use the web app — no installation required
# https://unified-probabilistic-validation.streamlit.app
```

---

## Architecture
```
CSV / Simulation Paths
        │
        ▼
DataContract              ← validates schema, timestamps, NaN, crossings
        │
        ▼
Adapters
  Adapter_PointForecast   ← residual pool, bucket-conditioned intervals
  Adapter_SimulationJoint ← joint Monte Carlo paths (d-dimensional)
  Adapter_Quantiles       ← pre-computed quantiles + PAVA crossing fix
        │
        ▼
BuildDist_FromResiduals   ← bootstrap or Gaussian sample reconstruction
        │
        ▼
Diagnostics_Input         ← capability-aware gateway (PIT/CRPS/pinball/ES)
        │
        ├── PIT uniformity + independence (KS, CvM, AD, Ljung-Box)
        ├── Anfuso interval backtest (bilateral, Basel-style)
        ├── Score_Pinball (quantile loss, regime-stratified)
        ├── Interval_Sharpness (width + coverage tradeoff)
        └── CRPS (proper scoring)
        │
        ▼
RegimeTagger              ← seasonal / volatility / break-flag rules
ThresholdCalibrator       ← regime-conditioned GREEN/YELLOW/RED thresholds
        │
        ▼
DecisionEngine            ← single .decide() → GovernanceDecision + provenance
        │
        ▼
NarrativeGenerator        ← AI technical + plain language summaries (Anthropic API)
        │
        ▼
Governance_ReportCard     ← rolling window table, stability, label band PNG
```

---

## Empirical results

| Dataset | Classification | Key finding |
|---------|---------------|-------------|
| ENTSO-E electricity load | RED | Undercoverage + PIT failure + ACF dependence |
| PV solar generation | RED | PIT failure without interval failure |
| Onshore wind generation | RED | Asymmetric lower-tail failure |
| Simulation (well-specified) | GREEN | Positive control confirmed |
| Simulation (variance inflation) | RED | Bilateral severe miscalibration |
| Simulation (mean bias) | RED | Directional tail failure |
| Simulation (heavy tails) | GREEN | Detection boundary at n=365 |

The PV result is the key finding: coverage-only governance would classify PV as acceptable (91.4% coverage); multi-layer governance correctly returns RED (PIT uniformity and independence strongly rejected). A coverage-only regulator would reduce capital for a structurally misspecified model.

---

## Pipeline stages

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `scripts/build_*.py` | Build derived artifacts from raw CSV |
| 2–4 | `experiments/run_001–003.py` | ENTSO-E, PV, Wind diagnostics |
| 5–6 | `experiments/run_004–004b.py` | Simulation + misspecification |
| 7 | `experiments/run_005.py` | Multivariate PV+Wind joint diagnostics |
| 8 | `experiments/run_006.py` | VaR capital distortion analysis |
| 9 | `experiments/run_007.py` | Rolling label stability + transitions |
| 10 | `experiments/run_008.py` | Report cards + AI narratives |

Run all stages: `python run_all.py --skip-build`

---

## Repository structure
```
unified-probabilistic-validation/
├── app.py                          Streamlit web application
├── run_all.py                      Full pipeline orchestrator
├── src/
│   ├── core/data_contract.py       DataContract + StandardizedModelObject
│   ├── adapters/                   PointForecast, SimulationJoint, Quantiles, BuildDist
│   ├── diagnostics/                Diagnostics_Input, Interval_Sharpness, evaluator, rolling
│   ├── calibration/                PIT computation + GOF tests
│   ├── scoring/                    CRPS, Score_Pinball
│   └── governance/                 Anfuso, RiskPolicy, ReasonCodes, RegimeTagger,
│                                   ThresholdCalibrator, DecisionEngine, NarrativeGenerator,
│                                   ReportCard, Stability
├── experiments/                    run_001 through run_008 + outputs
├── scripts/                        build_entsoe_derived, build_renewables_derived,
│                                   build_simulation_derived, build_simulation_misspec
├── tests/                          451 pytest tests
├── papers/                         Thesis chapters (01–05)
├── skills/upv/SKILL.md             Machine-readable framework documentation
└── data/                           Raw CSV files (derived artifacts excluded)
```

---

## Datasets

| Dataset | Source | n (eval) | Model class |
|---------|--------|----------|-------------|
| ENTSO-E electricity load | Public ENTSO-E Transparency Platform | 209,555 | Short-term |
| PV solar generation | Anonymised student dataset | 4,287 (daytime) | Long-term |
| Onshore wind generation | Anonymised student dataset | 9,000 | Long-term |
| Synthetic simulation | Generated (joint Gaussian DGP) | 365 as-of dates | Simulation |

Raw CSV files are included. Derived `.npy` artifacts are excluded and regenerated by `scripts/build_*.py`.

---

## Requirements
```bash
pip install -r requirements.txt
```

Optional — AI narrative generation:
```bash
export ANTHROPIC_API_KEY=sk-ant-...   # Mac/Linux
set ANTHROPIC_API_KEY=sk-ant-...      # Windows
```

---

## License

MIT License. See LICENSE file.

---

## Citation
```
Le, J.Y. & Askarova, K. (2026). Reliable Uncertainty in Energy Markets:
A Unified Calibration Framework for Simulation and Forecasting Models.
Master's Thesis, Lund University.
```

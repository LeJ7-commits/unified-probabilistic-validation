# Unified Probabilistic Validation

A modular reliability architecture for probabilistic simulation and forecasting models.

This repository implements a unified statistical validation framework designed to evaluate heterogeneous model classes within a shared probabilistic reliability space. The framework integrates distributional calibration diagnostics, strictly proper scoring rules, conformal prediction augmentation, and governance-oriented aggregation into a coherent validation stack.

Status: Research prototype (Master Thesis, Lund University)

---

## Motivation

Energy-market models span diverse categories including Monte Carlo simulation engines, short-term operational forecasting systems, and long-horizon renewable generation models.  

Despite their structural differences, these systems share a common objective: producing probabilistic statements about uncertain future quantities.

However, evaluation practices are typically fragmented:

- Simulation models are backtested via exceedance testing.
- Forecasting models are assessed via point-error metrics (MAE, RMSE).
- Quantile models rely on pinball loss without structured calibration diagnostics.

This repository proposes a unified architecture that evaluates all model classes under a common probabilistic reliability framework.

---

## Conceptual Architecture

The framework maps heterogeneous model outputs into a unified evaluation pipeline:

Model Output  
- Predictive Distribution
- Calibration Diagnostics
- Proper Scoring Evaluation
- Governance Aggregation  

The architecture explicitly separates:

1. **Distribution availability**  
2. **Distribution reconstruction (for deterministic forecasts)**  
3. **Calibration and sharpness diagnostics**  
4. **Governance-oriented risk classification**

Each layer is modular and independently testable.

---

## Core Statistical Principle

The central calibration diagnostic relies on the Probability Integral Transform (PIT):

u_t = F_t(y_t)

Under correct model specification:

u_t ~ Uniform(0,1)

Calibration, independence, and tail adequacy are assessed via transformed PIT sequences and formal statistical testing procedures.

---

## Methodological Components

The framework integrates four complementary layers:

### 1. Calibration Diagnostics
- Probability Integral Transform (PIT)
- Uniformity and independence testing
- Density forecast evaluation

### 2. Strictly Proper Scoring Rules
- Continuous Ranked Probability Score (CRPS)
- Quantile (Pinball) loss
- Multivariate Energy Score (planned)

### 3. Conformal Prediction Augmentation
- Finite-sample marginal coverage guarantees
- Covariate-shift-aware conformal methods
- Adaptive online recalibration (planned)

### 4. Governance Aggregation
- Hierarchical diagnostic thresholding
- Traffic-light style classification
- Rolling stability analysis (planned)

---

## Repository Structure

```
unified-probabilistic-validation/
│
├── src/
│   ├── calibration/
│   ├── scoring/
│   ├── conformal/
│   ├── governance/
│   └── utils/
│
├── experiments/
├── notebooks/
├── tests/
├── docs/
├── paper/
└── data/
```


---

## Data Policy

Raw datasets are intentionally excluded from version control.
The framework is dataset-agnostic and can be applied to:

- Operational load forecasts
- Renewable generation forecasts
- Simulation-based price scenarios
- TBC

See `data/README.md` for details.

---

## Scope

This repository focuses exclusively on probabilistic validation and governance translation.

It does not:
- Develop forecasting algorithms
- Optimize trading strategies
- Guarantee conditional coverage under arbitrary covariate partitions

The emphasis is on reliability diagnostics and structured evaluation.

---

## Development Roadmap
- [ ] Implement PIT diagnostics module
- [ ] Implement CRPS and scoring layer
- [ ] Add multivariate Energy Score
- [ ] Integrate conformal augmentation
- [ ] Develop governance classification engine
- [ ] Rolling regime-partitioned evaluation

---

## License
License to be determined upon supervisor approval.


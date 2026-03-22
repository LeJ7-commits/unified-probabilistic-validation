# Methodology

## 1. Framework Overview

This chapter describes the unified probabilistic validation architecture
developed to evaluate and govern heterogeneous energy market models. The
framework is organised into four layers, applied uniformly across all model
classes:

1. **Distribution reconstruction** — converting model outputs into evaluable
   predictive distributions where only point forecasts are available.
2. **Diagnostic evaluation** — applying a multi-layer battery of statistical
   tests to assess calibration, distributional correctness, and temporal
   independence.
3. **Conformal augmentation** — applying distribution-free post-hoc
   recalibration to restore nominal coverage under distributional shift.
4. **Governance classification** — aggregating diagnostic signals into an
   interpretable traffic-light decision for model use.

Each layer is described in turn below. The framework is implemented as a
reproducible Python pipeline; all build scripts, run scripts, and diagnostic
modules are version-controlled at
`github.com/LeJ7-commits/unified-probabilistic-validation`.

A fifth layer — **production integration** — wraps the above into a
unified execution architecture suitable for deployment in operational
settings. This layer comprises: a canonical data contract (DataContract)
that validates and normalises heterogeneous model inputs; a set of
model-class-specific adapters that convert raw outputs into evaluable
predictive distributions; a diagnostics gateway (Diagnostics_Input) that
routes inputs to the appropriate diagnostic branches based on available
data representations; a standalone decision engine (DecisionEngine) that
orchestrates the full diagnostic pipeline and produces a structured
governance decision with full provenance; an AI-powered narrative generator
(NarrativeGenerator) that converts structured decisions into plain-language
and technical governance summaries; and a Streamlit web application that
exposes the full pipeline to non-technical users via CSV upload. The
production layer is documented in Section 9.

---

## 2. Model Classes and Datasets

Four model classes are evaluated, spanning three real-world datasets and
one synthetic positive control.

### 2.1 Short-Term Electricity Load Forecasting (ENTSO-E)

The ENTSO-E dataset contains quarter-hourly electricity load observations
and corresponding point forecasts for a European transmission network,
covering a multi-year historical period (n = 210,427 raw rows; 209,555
evaluable after cleaning). The point forecast y_hat is treated as the
output of a machine learning model — specifically an ensemble of gradient
boosted trees (XGBoost or LightGBM class) trained on historical hourly
consumption data augmented with temperature lags and forecasts, and
calendar features (hour of day, day of week, month, public holiday
indicators). The forecast is generated at t = 0 to predict all 24 hours
of the following day. No distributional output is natively provided by
this class of model; the predictive distribution is reconstructed from
historical residuals (see Section 3).

### 2.2 Long-Term PV Generation (PV Solar)

The PV dataset contains hourly photovoltaic generation simulations and
corresponding actual generation measurements for a European solar site,
covering 2013–2015 (n = 26,280 raw rows). The simulation y_hat represents the expected power output derived from a
wind speed forecast via a power curve transformation: forecasted irradiance
is converted to expected generation using the turbine or panel's
characteristic response function, adjusted for air density, system
efficiency, and estimated downtime. This is a standard physical-parametric
approach (Lorenz et al., 2009) and does not produce a
distributional output; the predictive distribution is reconstructed from
residuals. Structural nighttime hours — defined as observations where both
simulation and actuals fall below a threshold of 10⁻⁹ — are excluded from
calibration evaluation (see Section 3.3). This exclusion is consistent
with standard practice in solar generation assessment and is documented
explicitly in dataset metadata.

### 2.3 Long-Term Wind Generation (Wind)

The wind dataset shares the same structure and time span as the PV dataset
(n = 26,280 raw rows, hourly, 2013–2015). The simulation y_hat is derived
by applying a wind turbine's power curve to a forecasted wind speed time
series: the power curve maps wind speed at hub height to expected power
output, incorporating rated power, cut-in and cut-out wind speed
thresholds, and optional adjustments for air density and efficiency. The
forecast is generated at a multi-year planning horizon informed by long-term
weather scenarios and site-specific technical parameters (turbine size,
hub height, geolocation). Wind generates power around the clock; no
nighttime exclusion is applied.

### 2.4 Synthetic Simulation Model (Price and Temperature)

The synthetic model generates correlated daily price and temperature
simulation paths from a known joint Gaussian data-generating process (DGP).
The DGP incorporates intraday seasonality (modelled via sine and cosine
functions of hour of day), annual seasonality (via cosine of day of year),
and cross-series correlation (ρ = 0.5) implemented via Cholesky
decomposition. Parameters are: base price 50, σ_price = 5.0; base
temperature 10, σ_temp = 3.0. The evaluation horizon is n_days = 365
as-of dates, each with 5,000 simulation paths of 8,760 hourly steps
(n_horizons = 8,760 ≈ one year). The predictive distribution is given
directly by the empirical quantiles of the 5,000 paths; no reconstruction
is required. This model class serves as a **positive control**: since the
evaluation draws realisations from the same DGP as the simulation paths,
a correctly implemented framework should return GREEN classifications.
Three deliberate misspecification scenarios are also evaluated
(see Section 2.5).

### 2.5 Misspecification Scenarios

Three misspecification scenarios are applied to both price and temperature
series to test the framework's discriminative validity:

- **Variance inflation:** Realised values are drawn with standard deviation
  2σ while the simulation paths use σ. This makes the predictive intervals
  systematically half as wide as required, producing bilateral over-breaching.
- **Mean bias:** Realised values are drawn with the mean shifted by +1σ
  (price: +5.0 units; temperature: +3.0 units) while the simulation paths
  use the unshifted mean. This produces systematic directional over-breaching
  on one tail.
- **Heavy tails:** Realised values are drawn from a Student's t-distribution
  with df = 3 (heavy tails) while the simulation paths remain Gaussian. This
  tests the framework's sensitivity to distributional shape misspecification.

Each scenario uses an independent random seed from the well-specified
positive control for methodological separation.

---

## 3. Distribution Reconstruction

For model classes that provide only point forecasts (ENTSO-E, PV, wind), a
predictive distribution must be reconstructed before any probabilistic
diagnostic can be applied. The reconstruction method serves as the
**pre-conformal baseline**: it represents the model's implied predictive
uncertainty as honestly as possible given the available information, without
augmentation. Its residual miscalibration, if any, motivates the conformal
layer (Section 5).

### 3.1 Residual-Based Reconstruction

For each observation t, define the point forecast residual as:

    r_t = y_t − ŷ_t

The predictive interval at nominal level (1 − α) is then:

    [ŷ_t + q̂_{α/2}(r), ŷ_t + q̂_{1−α/2}(r)]

where q̂_p(r) denotes the p-th empirical quantile of a trailing window of
past residuals, described below. A bias correction is applied by subtracting
the window mean from residuals before quantile estimation and adding it back
after, ensuring the interval is centred on the corrected forecast rather than
the raw point forecast.

### 3.2 ENTSO-E: 4-Bucket Time-of-Day Conditioning

Gas load residuals exhibit strong diurnal structure — the distribution of
forecast errors differs materially between night, morning, afternoon, and
evening consumption regimes. To capture this without over-parameterising
the rolling estimator, residuals are assigned to one of four coarse
time-of-day buckets:

    Night: 00:00–05:59 | Morning: 06:00–11:59
    Afternoon: 12:00–17:59 | Evening: 18:00–23:59

For each observation t in bucket b, the reconstruction uses a **combined
window** of:
- Wb = 40 trailing bucket-specific past observations (capturing local
  diurnal structure)
- Wg = 672 trailing global observations (7 days of quarter-hourly data;
  capturing recent volatility regime)

The combined window is concatenated and the bias-corrected empirical
quantiles are taken at levels α/2 and 1 − α/2. A shrinkage weight is
implicitly achieved by pooling the smaller bucket-specific window with
the larger global window. This specification was selected on the basis
of the feasibility analysis conducted on the 90-day ENTSO-E development
sample (notebooks/02_entsog_feasibility.ipynb), where it achieved 92.4%
empirical coverage against a 90% nominal target — the best result among
candidate methods tested.

A robust scale estimate is also computed per observation using the
normalised median absolute deviation (1.4826 × MAD) over the centred
combined window. This scale is used to generate Monte Carlo samples for
CRPS computation and as input to the conformal augmentation layer.

### 3.3 PV and Wind: 24-Bucket Hour-of-Day Conditioning

With 3 years of hourly data (~26,280 rows), each of the 24 hour-of-day
buckets contains approximately 1,095 observations, making fine-grained
conditioning stable. For each observation t at hour h, the reconstruction
uses W = 720 trailing same-hour observations (~30 days of same-hour
history). No global shrinkage window is used because the large per-bucket
sample size makes local estimation stable without pooling.

The nighttime exclusion for PV is applied before reconstruction: rows
where both Simulation and Actuals fall below 10⁻⁹ are removed from the
dataset. Of the 26,280 raw PV rows, 10,036 (38.2%) are excluded as
structural nighttime zeros. The rolling warmup then consumes the first
W = 720 same-hour observations per bucket before evaluation begins, leaving
n_eval = 4,287 evaluable PV daytime observations. For wind (no exclusion),
n_eval = 9,000.

### 3.4 Simulation: No Reconstruction Required

The synthetic simulation model provides 5,000 paths per as-of date. The
predictive interval bounds are computed directly as the α/2 and 1 − α/2
empirical quantiles of the path ensemble. No residual-based reconstruction
is applied.

---

## 4. Diagnostic Evaluation

The diagnostic layer applies three complementary evaluation approaches.
Together these constitute a multi-layer test battery that avoids the
limitations of any single metric.

### 4.1 Interval Backtesting — Anfuso Traffic-Light Framework

The Anfuso framework (Anfuso et al., 2017) evaluates whether the empirical
breach rate of a predictive interval is consistent with its nominal
miscoverage level α. A breach at time t is defined as:

    B_t = 𝟙{y_t < l_t} + 𝟙{y_t > u_t}

where l_t and u_t are the lower and upper bounds of the (1−α) interval.
Under correct calibration, E[B_t] = α. The test evaluates each tail
separately and jointly using a one-sided binomial test:

    H₀: breach rate ≤ nominal rate
    H₁: breach rate > nominal rate

The traffic-light classification maps the binomial exceedance p-value to
one of three outcomes: GREEN (model acceptable), YELLOW (model under
review), RED (model rejected). The cutoffs follow the Basel framework
convention (Basel Committee, 1996; 2010), adapted to energy market
governance context.

### 4.2 Distributional Diagnostics — PIT-Based Tests

The Probability Integral Transform (PIT) of a realisation y_t under the
predictive CDF F_t is:

    u_t = F_t(y_t)

Under correct probabilistic calibration, u_t ~ U(0, 1) i.i.d. (Dawid,
1984; Diebold et al., 1998). Two properties are tested:

**Uniformity** — whether the empirical distribution of {u_t} is consistent
with U(0,1):
- Kolmogorov–Smirnov (KS) test: nonparametric distance between empirical
  and uniform CDF (Kolmogorov, 1933; Smirnov, 1948).
- Cramér–von Mises (CvM) test: weighted quadratic distance, more sensitive
  to distributional centre (Cramér, 1928).
- Anderson–Darling (AD) test: weighted KS with enhanced sensitivity to
  distribution tails (Anderson and Darling, 1952).

**Independence** — whether the sequence {u_t} is serially independent.
Following Berkowitz (2001), PIT values are transformed to standard normal
scores z_t = Φ⁻¹(u_t), and the the Ljung–Box portmanteau test (Ljung and Box, 1978) is applied to
{z_t} at lags 5, 10, and 20.. Serial dependence in z_t indicates that the
model's predictive distribution fails to capture temporal autocorrelation
in the process.

PIT-based diagnostics require access to the full predictive CDF. For ENTSO-E
and renewable datasets, PIT scores are computed from the Monte Carlo sample
arrays (500 draws per observation) saved by the build scripts. For the
simulation model, only quantile bounds are passed to the run script in the
current pipeline; full PIT diagnostics are therefore unavailable for that
model class.

### 4.3 Proper Scoring — CRPS

The Continuous Ranked Probability Score (CRPS; Gneiting and Raftery, 2007)
is a strictly proper scoring rule that jointly rewards calibration and
sharpness. For a predictive CDF F and realisation y:

    CRPS(F, y) = ∫_{-∞}^{∞} [F(z) − 𝟙{z ≥ y}]² dz

A lower CRPS indicates a better probabilistic forecast. CRPS values are
not directly comparable across datasets because they are expressed in the
units of the target variable (electricity load for ENTSO-E; capacity factor
for PV and wind). Within-dataset comparisons across rolling windows and
model variants are meaningful.

### 4.4 Rolling Evaluation

Full-sample diagnostics capture aggregate performance but may obscure
localised failures or improvements over time. Rolling-window diagnostics
are therefore computed under two complementary schemes:

- **Non-overlapping windows:** each window is independent, providing an
  unbiased sequence of local diagnostic snapshots.
- **Overlapping windows:** consecutive windows share observations,
  producing smoother diagnostic trajectories that highlight gradual
  regime transitions.

There is no standard window length for rolling energy market diagnostics
(Rikard Engström, personal communication, 2026); the choice is
context-dependent and should reflect the trade-off between local
adaptivity and estimation stability. Rolling evaluation is particularly
valuable for detecting localised forecast failures and structural breaks
(Giacomini and Rossi, 2010). The window parameters used in this
study are:

| Dataset | Window | Step (non-overlapping) | Step (overlapping) |
|---------|--------|----------------------|-------------------|
| ENTSO-E | 250    | 250                  | 50                |
| PV      | 720    | 720                  | 168               |
| Wind    | 720    | 720                  | 168               |

For ENTSO-E (quarter-hourly), a window of 250 corresponds to approximately
62.5 hours. For PV and wind (hourly), a window of 720 corresponds to 30
days, and a step of 168 corresponds to one week.

---

## 5. Conformal Augmentation

Conformal prediction (Vovk, Gammerman and Shafer, 2005; Shafer and Vovk, 2008;
Tibshirani et al., 2019) provides distribution-free finite-sample coverage
guarantees under exchangeability.
In this framework, conformal augmentation is applied as a post-hoc
recalibration layer on top of the residual-based base intervals, not as
an alternative evaluation method.

### 5.1 Motivation

The residual-based reconstruction in Section 3 is a reasonable best-effort
baseline, but it does not guarantee nominal coverage — particularly under
distributional shift between the calibration period (from which residual
quantiles are estimated) and the evaluation period. The conformal layer
corrects for this shift without requiring re-estimation of the underlying
model.

### 5.2 Method: Base Interval + Conformal Expansion

Among five conformal variants evaluated on the ENTSO-E 90-day development
sample (04_conformal_wrapping.ipynb), **base interval + conformal
expansion** achieves the best calibration–sharpness trade-off. This method
retains the structural conditioning of the base interval (time-of-day
buckets, rolling residuals) and expands or contracts it by a conformal
quantile correction:

1. Split the calibration set into a proper training set and a calibration
   holdout.
2. Compute conformity scores s_t = max(l_t − y_t, y_t − u_t) on the
   calibration holdout, where l_t, u_t are the base interval bounds.
3. Set the conformal quantile q̂ as the ⌈(n+1)(1−α)⌉/n empirical
   quantile of {s_t}.
4. Expand the test-set intervals: [l_t − q̂, u_t + q̂].

Under exchangeability, the resulting intervals achieve marginal coverage of
at least 1 − α with finite-sample validity (Angelopoulos and Bates, 2023).

### 5.3 Distribution Shift Under Covariate Shift

To account for temporal nonstationarity — where test-period observations
are not exchangeable with calibration-period observations — covariate-shift
weights are estimated via a logistic density ratio on time features (elapsed
time, cyclic hour and month encodings). These weights are used to reweight
the calibration conformity scores before computing the conformal quantile,
following Tibshirani et al. (2019).

### 5.4 Limitations

The conformal augmentation layer was evaluated only on the ENTSO-E 90-day
development sample due to data availability constraints; full-dataset
conformal results and extension to PV and wind are deferred as future work.
The online CP variant with a fixed step size was found to produce
approximately 50% coverage regardless of nominal level on this dataset,
due to a scale mismatch: the step size of 0.01 is calibrated for
normalised residuals but ENTSO-E errors are in physical electricity load units
(O(10³)). This confirms that online CP step sizes must be set relative
to the residual scale of the target series.

---

## 6. Governance Classification

### 6.1 Traffic-Light Aggregation

Diagnostic signals from the interval backtesting and PIT layers are
aggregated into a single governance classification using a rule-based
policy. The classification is:

- **GREEN:** All monitored signals within policy thresholds. Model is
  considered acceptable for continued use.
- **YELLOW:** One or more signals mildly off-target but not in extreme
  failure territory. Model is flagged for review.
- **RED:** One or more signals strongly exceed thresholds. Model use
  should be restricted or the model should be re-estimated.

The specific thresholds are implemented in `src/governance/risk_classification.py`.
RED is triggered if any of the following hold:
- PIT uniformity is rejected at conventional significance levels
  (minimum p-value across KS, CvM, AD < threshold)
- PIT serial independence is rejected (minimum Ljung–Box p-value < threshold)
- Empirical coverage deviates from nominal by more than a policy-defined
  tolerance (coverage_error > tol)

### 6.2 Regulatory Framing

The traffic-light architecture is inspired by the Basel Committee's
backtesting framework for Value-at-Risk models (Basel Committee, 1996;
2010), which classifies models into green, yellow, and red zones based
on the number of VaR exceedances over a 250-day window. The adaptation
to probabilistic energy market models replaces exceedance counts with
the broader multi-layer diagnostic battery described above, and extends
the governance logic to cover distributional shape (PIT uniformity) and
temporal dependence (Ljung–Box independence) in addition to coverage.
This is consistent with the REMIT regulatory environment (European Parliament,
2011), which requires that energy market participants maintain demonstrably
reliable models for fundamental price formation and risk quantification.

---

## 7. Multivariate Extension

The univariate diagnostic framework evaluates each asset independently.
To assess joint calibration across correlated assets, a multivariate
dependency analysis is conducted for the PV and wind datasets, which share
a common hourly time index (2013–2015).

Joint PIT residual dependence is evaluated on the shared daytime evaluation
index (n = 4,287 observations) using:

- **Multivariate Ljung–Box test** applied to the stacked residual vector
  [z_PV, z_wind]ᵀ at lags 5, 10, and 20, testing whether the joint
  residual process is white noise.
- **Cross-correlation matrix** of PIT residuals at lags 0, 1, 6, and 24,
  characterising contemporaneous and lagged dependence between the two
  series.
- **Bivariate energy score** (Gneiting et al., 2008) as a joint proper
  scoring rule providing a summary measure of the joint predictive
  distribution quality.

The multivariate layer addresses the supervisor's recommendation that a
credible validation framework for portfolio-level energy risk should
evaluate joint dependence structure across multiple commodity classes,
not only marginal calibration per asset.

---

## 8. Reproducibility

All pipeline components are implemented in Python 3.x and organised as
follows:

- `scripts/build_*.py` — dataset preparation and distribution reconstruction
- `experiments/run_*.py` — diagnostic evaluation and artifact output
- `src/diagnostics/` — evaluator, rolling, and run policy modules
- `src/governance/` — risk classification module
- `papers/` — thesis write-up layer
- `notebooks/` — methodological workbenches (feasibility analysis,
  conformal wrapping exploration)

The notebooks serve as exploratory workbenches — they establish that
reconstruction methods are viable and conformal variants are comparable —
but are not the source of official results. Official results are produced
exclusively by the `experiments/run_*.py` scripts and stored as JSON
artifacts alongside rolling diagnostic CSVs. Random seeds are fixed
throughout (seed = 42 for all build scripts; independent seeds per
misspecification scenario). All experiments can be reproduced by running the build and run scripts in
sequence from a clean environment using the `requirements.txt` specification.
A single orchestrator script — `run_all.py` — automates the full sequence of
ten pipeline stages (build → run_001 through run_008), with progress
reporting and graceful error handling. Non-Python users may access the
framework through a Streamlit web application deployed at
`unified-probabilistic-validation.streamlit.app`, which accepts a forecast
CSV and executes the core validation pipeline (DataContract → Adapter →
BuildDist → Diagnostics_Input → DecisionEngine → NarrativeGenerator)
directly in the browser without requiring local installation. A SKILL.md
file (`skills/upv/SKILL.md`) documents the framework architecture and
operational procedures in a machine-readable format, enabling automated
orchestration by AI assistants.

---

## 9. Production Architecture

### 9.1 Motivation

The diagnostic pipeline described in Sections 3–7 is implemented as a set
of experiment scripts that operate on pre-built derived artifacts. While
this is sufficient for reproducible thesis results, a production deployment
requires a more robust architecture: one that validates inputs before
processing, routes heterogeneous model types through appropriate
transformations, and produces structured, auditable outputs that
non-technical stakeholders can consume. This section documents the
production architecture layer built on top of the diagnostic pipeline.

### 9.2 Data Contract and Standardised Model Objects

All inputs enter the framework through a DataContract validator
(`src/core/data_contract.py`) that enforces a canonical schema before any
downstream processing occurs. The contract validates:

- **Required fields:** timestamp array t (monotone, no gaps), realisation
  array y (no NaN), model identifier, split label.
- **Optional fields:** point forecast y_hat, quantile arrays Q_t(p),
  Monte Carlo sample paths S ∈ ℝ^(M×d), covariates x.
- **Sanity checks:** monotone timestamps, no NaN or Inf in samples, quantile
  non-crossing (1e-8 tolerance), sample size M ≥ threshold, consistent
  array lengths.

Split labels follow a controlled vocabulary: `train`, `test`, `window_{int}`
(rolling window identification), or `regime_{tag}` (regime-tagged windows,
e.g. `regime_winter`). Invalid inputs raise a `DataContractError` with a
precise diagnostic message; no silent failures occur.

Validated inputs are encapsulated in a frozen `StandardizedModelObject`
— an immutable dataclass that all downstream components consume. Freezing
prevents accidental mutation of shared state in a pipeline where the same
object flows through multiple components.

### 9.3 Model-Class Adapters

Three adapters convert `StandardizedModelObject` instances into
distribution representations appropriate for their model class:

**Adapter_PointForecast** (`src/adapters/point_forecast.py`) accepts a
point forecast y_hat and builds a bucket-conditioned residual pool via a
configurable `bucket_fn` callable. The bucketing strategy is pluggable:
`bucket_hourly_24` (24-bucket hour-of-day, default for renewables),
`bucket_coarse_4` (4-bucket coarse time-of-day, for ENTSO-E), or
`bucket_none` (global pool, no conditioning). Making the bucket function a
parameter rather than hardcoding it is the production-correct design choice:
different commodities have structurally different residual patterns, and
the framework must adapt to each without modifying core adapter code.
Sanity checks flag bias (|pool mean| > tolerance × scale), structural
breaks (variance ratio > threshold), and pool sizes below hard and soft
minimum thresholds.

**Adapter_SimulationJoint** (`src/adapters/simulation_joint.py`) accepts
either a sims_dict (the dict-of-dicts format produced by simulation
notebooks) or a 3D numpy array (n_timestamps, M, d). Both formats are
auto-detected. The adapter produces a `JointSimulationObject` containing
both per-variable marginal outputs (compatible with univariate diagnostics)
and the full joint sample array (required for Energy Score computation).

**Adapter_Quantiles** (`src/adapters/quantile_adapter.py`) accepts
pre-computed quantile arrays Q_t(p) for p in a grid. Quantile crossings
are detected and fixed via the Pool Adjacent Violators Algorithm (PAVA)
isotonic regression with a warning; non-crossing quantiles are required
for downstream CDF interpolation. A PCHIP (Piecewise Cubic Hermite
Interpolating Polynomial) monotone spline interpolator is fitted to the
quantile function at each observation, enabling PIT computation from
quantile-only inputs.

### 9.4 Distribution Reconstruction from Residuals

`BuildDist_FromResiduals` (`src/adapters/build_dist_from_residuals.py`)
converts a ResidualPool (output of Adapter_PointForecast) into a Monte
Carlo sample matrix of shape (n_obs, M) by reconstructing the predictive
distribution per observation. Two modes are available:

- **Non-parametric (default):** bootstrap resamples M residuals from the
  trailing residual pool at each t. Preserves skewness, heavy tails, and
  non-Gaussian structure. Correct for energy forecasts with asymmetric
  errors.
- **Parametric (Gaussian):** fits N(bias_t, scale_t) to the residual pool
  at each t using the pool_bias and pool_scale stored in the ResidualPool.
  Faster but understates tail risk for heavy-tailed commodities.

This component closes the gap between point-forecast model classes (which
produce only lo/hi bounds from the adapter) and PIT-based diagnostics
(which require sample paths). The reconstructed samples are directly
compatible with `evaluate_distribution()`, CRPS computation, and the
`Diagnostics_Input` gateway.

### 9.5 Diagnostics Gateway

`Diagnostics_Input` (`src/diagnostics/diagnostics_input.py`) normalises
adapter outputs or raw arrays into a `DiagnosticsReadyObject` — a
capability-annotated container that downstream diagnostic components
query before attempting computation. The capability interface exposes:

- `can_compute_pit` — True if samples or CDF callable available
- `can_compute_crps` — True if samples available
- `can_compute_pinball` — True if quantile arrays available
- `can_compute_interval` — True if lo/hi bounds available
- `can_compute_energy_score` — True if joint samples (d ≥ 2) available

This design prevents silent missing-data failures: a diagnostic component
calls `dro.require("crps")` before computing, which raises a
`DiagnosticsInputError` with an informative message if the capability is
unavailable. The gateway accepts both adapter output objects (auto-detected
by type) and raw numpy arrays, with validation on all inputs.

### 9.6 Scoring Components

Two scoring components complement the existing CRPS implementation:

**Score_Pinball** (`src/scoring/pinball.py`) computes pinball (quantile)
loss (Koenker and Bassett, 1978) L_p(q, y) = (y − q)·p if y ≥ q, else
(q − y)·(1 − p), across all quantile levels in the grid. The mean pinball loss averaged over a dense
level grid approximates CRPS; the per-level breakdown reveals which
quantile regions are most miscalibrated. Regime-stratified losses are
computed if regime tags are provided, enabling diagnostic decomposition
by market condition.

**Interval_Sharpness** (`src/diagnostics/interval_sharpness.py`) computes
interval width and the sharpness-coverage tradeoff: mean and median width,
standard deviation of widths, and a human-readable interpretation label
(sharp / acceptable / wide / uninformative) based on the ratio of mean
width to the interquartile range of the target variable. A risk label
(safe / risky / over-cautious / acceptable) is assigned based on the
deviation of empirical coverage from nominal. This operationalises the
diagnostic principle that good probabilistic forecasts should be
simultaneously sharp and calibrated: narrow intervals that under-cover
are risky, wide intervals that over-cover are uninformative.

### 9.7 Regime Tagging and Threshold Calibration

`RegimeTagger` (`src/governance/regime_tagger.py`) assigns regime labels
to rolling windows using a composable rule system. Rules are callable
objects with signature `(t, y) → str | None`; the first non-None result
wins. Three built-in rules are provided:

- **SeasonalRule:** assigns `winter` (months 11–2) or `summer` (months
  5–8) based on the dominant month in the window, with a configurable
  majority threshold.
- **VolatilityRule:** assigns `high_vol` or `low_vol` based on the window's
  standard deviation relative to pre-computed percentile thresholds. Must
  be fitted on a reference set of window statistics before tagging.
- **BreakFlagRule:** detects structural breaks by comparing the variance
  ratio between the first and second halves of the window.

Regime tags follow the SplitLabel vocabulary (`regime_{tag}`) for
compatibility with the DataContract.

`ThresholdCalibrator` (`src/governance/threshold_calibrator.py`) calibrates
GREEN/YELLOW/RED coverage thresholds per regime using a calibration split.
For each regime with sufficient data (N_min_hard ≥ 10 by default), the
GREEN coverage threshold is set at a specified quantile of the observed
coverage distribution in calibration windows (default: 10th percentile).
This data-driven approach extends the Basel fixed-cutoff framework to
heterogeneous market conditions: if `high_vol` windows systematically
achieve lower empirical coverage, the GREEN threshold is relaxed for that
regime rather than uniformly applied. A relax_factor parameter bounds
the maximum downward adjustment from the global target, preventing
over-relaxation on sparse regimes. Regimes with insufficient calibration
data fall back to the global policy.

### 9.8 Decision Engine

`DecisionEngine` (`src/governance/decision_engine.py`) is the top-level
orchestrator. A single call to `.decide(dro, regime_tag)` executes the
full diagnostic pipeline — computing only what the `DiagnosticsReadyObject`
is capable of, applying the regime-conditioned policy, and returning a
`GovernanceDecision` with:

- **final_label:** GREEN, YELLOW, or RED
- **reason_codes:** list of `ReasonCode` enum values indicating which
  diagnostic branches triggered the classification
- **metric_snapshot:** all computed diagnostic values in a flat dict
- **policy_used:** the RiskPolicy applied (global or regime-calibrated)
- **provenance:** full audit trail — which diagnostics were computed,
  which were skipped and why, policy source, decision timestamp

The DecisionEngine does not re-implement any diagnostic logic; it delegates
entirely to the existing `evaluate_distribution()`, `anfuso_interval_backtest()`,
`Score_Pinball`, `Interval_Sharpness`, and `classify_risk()` functions.
This preserves full backwards compatibility: the existing
`run_diagnostics_policy()` / `write_run_artifacts()` pipeline continues to
operate unchanged, with the DecisionEngine as an additive layer producing
a structured `governance_decision.json` artifact alongside the existing
outputs.

### 9.9 Narrative Generation

`NarrativeGenerator` (`src/governance/narrative_generator.py`) converts
a `GovernanceDecision` into plain-language governance summaries using the
Anthropic API (Claude Sonnet). Two narrative modes are produced per
dataset in a single API call:

- **Technical narrative:** 3–5 sentences for quantitative risk officers.
  References specific metric values, names the diagnostic branches that
  failed, quantifies deviations from nominal, and states the governance
  implication (capital multiplier impact, REMIT reporting obligation).
- **Plain language narrative:** 3–5 sentences for non-technical
  stakeholders. No jargon; uses analogies where appropriate; focuses on
  the business implication and required action.

The structured prompt injects the full `GovernanceDecision.to_dict()`
output, model class, and commodity context. A `<<<PLAIN>>>` delimiter
separates the two sections in the model response, enabling deterministic
parsing. If the API is unavailable (no key configured, insufficient
credits, or network failure), clearly labelled stub narratives are written
and the pipeline continues without interruption. At a cost of approximately
$0.005 per dataset, narrative generation for the full 11-dataset thesis
pipeline costs under $0.10 total.

### 9.10 Deployment and Accessibility

The full framework is accessible through three interfaces:

**Command-line orchestrator** (`run_all.py`): executes all ten pipeline
stages (build → run_001 through run_008) in sequence, with per-stage
progress reporting and graceful error handling on optional stages.
Supports `--skip-build`, `--stages`, and `--dry-run` flags.

**Streamlit web application** (`app.py`, deployed at
`unified-probabilistic-validation.streamlit.app`): a production-grade
browser interface that accepts a forecast CSV upload and executes the
core pipeline (DataContract → Adapter_PointForecast →
BuildDist_FromResiduals → Diagnostics_Input → DecisionEngine →
NarrativeGenerator) directly, without requiring local installation or
Python knowledge. Results are displayed with colour-coded governance
labels, metric cards, Anfuso backtest table, AI narratives (technical and
plain language tabs), decision provenance, and a downloadable ZIP of all
artifacts. The application enforces a 50,000-row limit for cloud
deployment and falls back to stub narratives if no API key is configured.

**SKILL.md documentation** (`skills/upv/SKILL.md`): a machine-readable
skill file that documents the full framework architecture, pipeline stages,
component map, CSV format requirements, governance interpretation guide,
and step-by-step instructions for adding new commodity classes. This
enables AI-assisted orchestration: a new analyst can ingest the skill
file and receive guided instructions for extending the framework to
additional commodities without modifying core source code.

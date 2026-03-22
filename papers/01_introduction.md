# Introduction

## 1.1 Background and Motivation

Energy markets are governed by uncertainty. The physical quantities that
determine price formation — electricity demand, wind generation, solar
output, temperature — are inherently stochastic, and the models that
represent them must ultimately be evaluated not only for their point
accuracy but for the reliability of the uncertainty they express. A
short-term load forecast that is directionally correct but consistently
overconfident in its distributional tails will underestimate reserve
requirements. A wind generation simulation that produces well-shaped paths
on average but fails to capture persistent low-generation regimes will
mislead portfolio hedging decisions. The question of whether a model's
stated probabilistic uncertainty is trustworthy is therefore not a
statistical nicety — it is a practical prerequisite for sound market and
risk management.

Yet validation practice across the energy industry remains largely
fragmented along model class boundaries. Simulation models — Monte Carlo
engines used for pricing and risk analysis — are typically assessed via
Value-at-Risk (VaR) exceedance testing and other coverage-based backtesting
procedures. Short-term operational forecasting models are most commonly
evaluated using deterministic error metrics such as MAE or RMSE, which
carry no information about distributional reliability. Quantile regression
and probabilistic forecasting models are evaluated via pinball loss, which
penalises individual quantile violations but does not diagnose the shape
or serial structure of the full predictive distribution. Each model class
has inherited its own evaluation culture; none of these cultures shares a
common language, and none provides a complete picture of probabilistic
reliability on its own.

This fragmentation creates at least three structural compromises. First,
reduced comparability across model classes: a GREEN result on a simulation
exceedance test and a GREEN result on a pinball loss benchmark are not
calibrated to the same standard. Second, opaque aggregation of diagnostics
into decision-ready risk segmentations — without a shared framework,
governance conclusions rest on informal judgements rather than transparent
diagnostic evidence. Third, a distal relationship between economic
interpretation and statistical diagnostics, where model risk decisions lack
a principled connection to the underlying calibration evidence.

This thesis addresses these insufficiencies by constructing a
reliability-based evaluation system anchored in distributional calibration.
Formally, a unified validation architecture is defined as a mapping ℳ that
transforms heterogeneous model outputs into predictive distribution space,
followed by evaluation under a common probabilistic loss and calibration
functional ℒ(F_t, y_t), and aggregation via statistically calibrated
governance thresholds. Under this definition, unification signifies
comparability within a shared probabilistic reliability space — not
methodological homogenisation of the underlying models.

The core diagnostic tool is the Probability Integral Transform (PIT),
which provides a model-class-agnostic basis for assessing distributional
calibration: under correct probabilistic specification, the PIT of any
forecast should be uniformly distributed and serially independent
(Diebold, Gunther, and Tay, 1998; Dawid, 1985). Deviations from these
properties are diagnosed using a multi-layer test battery — interval
backtesting via the Anfuso traffic-light framework, PIT uniformity and
independence testing via formal goodness-of-fit procedures, and proper
scoring via the Continuous Ranked Probability Score. A conformal prediction
augmentation layer extends the architecture to provide finite-sample
coverage guarantees under distributional shift (Tibshirani et al., 2019;
Angelopoulos et al., 2024). A governance aggregation layer translates
diagnostic signals into structured traffic-light classifications grounded
in the Basel backtesting framework (Basel Committee, 1996; 2010), adapted
to the energy market and REMIT regulatory context.

---

## 1.2 Research Questions

The thesis is structured around three core research questions:

**RQ1:** Can heterogeneous energy-market models (Monte Carlo simulation,
short-term operational forecasting, and long-term renewable forecasting)
be evaluated within a unified probabilistic reliability architecture without
compromising diagnostic depth or interpretability?

**RQ2:** Under which market conditions do conformal augmentation methods
improve finite-sample coverage stability relative to classical PIT-based
calibration diagnostics under regime shifts and temporal dependence?

**RQ3:** Can probabilistic calibration diagnostics and scoring-rule outcomes
be aggregated into statistically defensible governance classifications
comparable to Basel-style traffic-light backtesting systems?

---

## 1.3 Scope and Contributions

The framework is developed and evaluated on four model classes:

- **Short-term electricity load forecasting** (ENTSO-E dataset):
  a machine learning point forecast of next-day quarter-hourly electricity
  consumption for the German power system, representative of operational
  short-term energy forecasting.
- **Long-term PV generation simulation** (PV solar dataset): a
  physical-parametric model converting irradiance forecasts to expected
  power output via a panel response function, representative of long-horizon
  renewable capacity planning models.
- **Long-term wind generation simulation** (wind dataset): structurally
  analogous to PV, deriving expected power output from wind speed forecasts
  via a turbine power curve with site-specific technical parameters.
- **Synthetic Monte Carlo simulation** (synthetic dataset): a controlled
  positive-control case in which the model DGP is fully known, enabling
  the framework's discriminative validity to be tested against three
  deliberate misspecification scenarios.

For the first three model classes — which provide only point forecasts —
the framework includes a distribution reconstruction layer that constructs
evaluable predictive intervals from historical residuals using rolling
empirical quantile methods with time-of-day conditioning. This
reconstruction is a prerequisite for applying PIT-based diagnostics to
model classes that do not natively produce distributional output, and
follows the approach of Goude and Nédellec (2015).

The primary contributions of this thesis are:

1. A **modular, reproducible validation pipeline** that evaluates
   heterogeneous energy market model classes under a unified probabilistic
   standard, implemented as an open-source Python framework.

2. A **multi-layer diagnostic architecture** that separates interval
   coverage (Anfuso traffic-light), distributional shape (PIT uniformity),
   and temporal dependence (Ljung–Box independence), demonstrating that
   these layers are not redundant — a model can pass one layer while
   failing another.

3. An **empirical cross-dataset comparison** revealing structurally
   distinct failure modes: bilateral miscalibration in the short-term
   electricity load model, asymmetric lower-tail failure in wind generation,
   distributional failure without interval failure in PV solar, and
   full-pass in the synthetic positive control.

4. A **conformal augmentation feasibility study** demonstrating that base
   interval + conformal expansion restores near-nominal coverage (89.9%
   at α = 0.1) on the ENTSO-E series under temporal distribution shift,
   with a 12% interval width increase — the best calibration–sharpness
   trade-off among five evaluated conformal variants.

5. A **misspecification detection analysis** showing that the framework
   correctly classifies variance inflation and mean bias as RED while
   exposing a power boundary: heavy-tail misspecification at t(df=3) is
   not detected by interval backtesting at n = 365, with direct practical
   implications for minimum evaluation horizon requirements.

6. A **governance framework** aggregating diagnostic signals into
   structured traffic-light classifications, with explicit regulatory
   grounding in the Basel backtesting literature and the REMIT context,
   and a multivariate extension evaluating joint PIT residual dependence
   across correlated renewable assets.
7. A **production-grade validation architecture** implementing eleven
   software components — DataContract, three model-class adapters
   (Adapter_PointForecast, Adapter_SimulationJoint, Adapter_Quantiles),
   a distribution builder (BuildDist_FromResiduals), a diagnostics gateway
   (Diagnostics_Input), two scoring components (Score_Pinball,
   Interval_Sharpness), a regime tagger (RegimeTagger), a threshold
   calibrator (ThresholdCalibrator), and a decision engine (DecisionEngine)
   — together comprising 451 passing unit and integration tests. The
   architecture enforces strict input validation via a canonical data
   contract, routes heterogeneous model types through appropriate
   transformations, and produces structured governance decisions with full
   provenance audit trails.

8. An **AI-powered narrative generation layer** (NarrativeGenerator) that
   converts structured governance decisions into both technical summaries
   (for quantitative risk officers) and plain-language explanations (for
   non-technical stakeholders) via the Anthropic API, and a **Streamlit
   web application** deployed at
   `unified-probabilistic-validation.streamlit.app` that exposes the full
   validation pipeline to non-Python users via CSV upload, enabling
   governance classification without local installation.

The framework does not develop new forecasting algorithms, optimise
trading strategies, or guarantee conditional coverage under arbitrary
covariate partitions. Its scope is confined to probabilistic validation,
reliability diagnostics, and governance translation of model outputs.

---

## 1.4 Industry Context

This thesis was conducted in collaboration with Energy Quant Solutions
Sweden AB (EnBW group). The industry partnership shaped the applied scope
of the framework — in particular the selection of model classes, the
treatment of y_hat as a machine learning or physical-parametric point
forecast, the governance framing of the traffic-light classification, and
the practical interpretation of RED classifications in an institutional
model risk context. Research questions and methodology were developed
jointly with the academic supervisor and the industry partner.

---

## 1.5 Literature Positioning

This thesis sits at the confluence of three research streams.

The first is **density forecast evaluation and calibration theory**. Since
Dawid (1985) formalised calibration as an empirical notion of probability,
and Diebold, Gunther, and Tay (1998) introduced PIT as a practical
diagnostic instrument, the field has developed strictly proper scoring
rules (Gneiting and Raftery, 2007) and the principle of maximising
sharpness subject to calibration (Gneiting and Katzfuss, 2014). More
recently, conformal prediction provides distribution-free finite-sample
coverage guarantees (Tibshirani et al., 2019), with adaptive online
variants for non-stationary environments (Angelopoulos et al., 2024).

The second is **probabilistic forecasting in energy systems**. Industrial
practice has shifted from deterministic to probabilistic modelling (Goude
and Nédellec, 2015), with large-scale competitions such as GEFCOM
underscoring quantile-based evaluation (Hong et al., 2016). Recent work
documents benefits of integrating probabilistic forecasts into electricity
price forecasting (Uniejewski and Ziel, 2025). However, evaluation practice
remains predominantly metric-driven with limited systematic integration
across heterogeneous model classes.

The third is **financial backtesting and governance**. From Kupiec (1995)
and Christoffersen (1998) through Berkowitz (2001), statistical backtesting
procedures constitute the backbone of Basel regulatory traffic-light systems
(Basel Committee, 1996; 2010), extended to counterparty credit risk by
Ruiz (2012) and Kenyon and Stamm (2012). The translation of these
governance-oriented paradigms into energy analytics remains comparatively
limited.

The research frontier lies in the systematic integration of all three
streams — calibration theory, energy forecasting evaluation, and
financial-style governance — into a coherent reliability architecture
operating across diverse model classes.

---

## 1.6 Thesis Structure

**Chapter 2 — Methodology** describes the unified validation architecture:
distribution reconstruction, diagnostic evaluation, conformal augmentation,
governance classification, the multivariate extension, and the production
integration layer comprising the data contract, model-class adapters,
distribution builder, diagnostics gateway, scoring components, regime
tagger, threshold calibrator, decision engine, and deployment interfaces.

**Chapter 3 — Results** reports empirical findings across all four model
classes and misspecification scenarios, structured as a cross-dataset
comparison with full diagnostic tables.

**Chapter 4 — Discussion** interprets results in relation to the three
research questions, addresses limitations, and situates findings within
the broader literature.

**Chapter 5 — Governance Implications** translates diagnostic findings
into practical governance recommendations, with reference to Basel and
REMIT regulatory frameworks and the role of conformal augmentation in
model risk management.



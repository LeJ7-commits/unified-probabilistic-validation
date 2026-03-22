# Discussion

This chapter interprets the empirical results in relation to the three
research questions, situates the findings within the broader literature,
and addresses the principal limitations of the framework.

---

## 1. RQ1 — A Unified Probabilistic Reliability Architecture

**RQ1:** Can heterogeneous energy-market models be evaluated within a
unified probabilistic reliability architecture without compromising
diagnostic depth or interpretability?

The empirical results provide a positive answer, with an important
qualification about what "unified" means in practice.

### 1.1 What unification achieves

All four model classes — short-term electricity load forecasting (ENTSO-E),
long-term PV generation, long-term wind generation, and synthetic Monte
Carlo simulation — were evaluated under the same diagnostic pipeline
without modification. The framework consumed point forecasts and simulation
paths alike, converting both into a common evaluable object (predictive
interval + sample array) before applying an identical battery of tests.
The cross-dataset summary table in Section 7 of the Results confirms that
this produces meaningful, comparable governance classifications: three RED
results on real-data models and two GREEN results on the positive control.

Critically, the unification does not flatten the findings into a single
scalar. The traffic-light framework separates interval coverage, PIT
uniformity, and serial independence as distinct diagnostic dimensions, and
the real-data models fail on different combinations of these. ENTSO-E fails
on all three; PV fails on PIT and independence while passing coverage; wind
fails on interval backtesting (lower tail only) and distributional
diagnostics. These are structurally distinct failure modes that a unified
scalar metric would have obscured. The framework thus achieves what the
proposal described: comparability within a shared probabilistic reliability
space without imposing methodological homogenisation.

### 1.2 The PV finding as the key test case

The most important single finding for RQ1 is the PV result. PV passes
interval backtesting (GREEN Anfuso, 91.4% coverage) but fails PIT
uniformity and independence diagnostics (both strongly rejected). This divergence directly demonstrates that coverage-only governance — the
dominant evaluation paradigm in energy forecasting practice (Weron, 2014;
Nowotarski and Weron, 2018) — would classify PV as acceptable
while a distributional-layer diagnostic correctly flags systematic
misspecification. This is not a theoretical argument: it is an empirical
finding on real industry data.

This result is consistent with the broader literature on probabilistic
forecast evaluation. Gneiting and Katzfuss (2014) argue that calibration
and sharpness must be evaluated jointly, and that coverage metrics alone
cannot distinguish a well-calibrated sharp forecast from a
poorly-calibrated but accidentally correct one. The PV finding is a
concrete instantiation of this theoretical point in an energy market
context.

### 1.3 Distribution reconstruction as a prerequisite

For the three real-data model classes, the unified architecture required
a distribution reconstruction step before any probabilistic diagnostic
could be applied. This step — rolling empirical residual quantiles with
time-of-day conditioning — is not itself a contribution to forecasting
methodology, but it is a necessary architectural component. Without it,
PIT-based evaluation of point-forecast models is not possible, and the
unification claim collapses.

The reconstruction produces coverage rates of 87.1% (ENTSO-E), 91.4%
(PV), and 88.6% (wind) against a 90% nominal target. These imperfections
are not failures of the reconstruction — they are genuine findings about
the models being evaluated. A perfect reconstruction would itself be a
model improvement, not a validation tool. The framework's role is to
diagnose the imperfections, not to eliminate them.

---

## 2. RQ2 — Conformal Augmentation Under Regime Shifts

**RQ2:** Under which market conditions do conformal augmentation methods
improve finite-sample coverage stability relative to classical PIT-based
calibration diagnostics under regime shifts and temporal dependence?

The conformal feasibility study on the ENTSO-E development sample provides
a partial answer, with clear conditions identified for both success and
failure.

### 2.1 When conformal augmentation works

The base interval + conformal expansion method achieves 89.9% empirical
coverage at α = 0.1 on the 30% test split of the 90-day ENTSO-E sample,
up from 82.5% on the base interval alone — a 7.4 pp coverage restoration
at a 12% width cost. This is the best result of the five methods evaluated
and confirms that conformal augmentation is effective when:

- The base intervals are directionally reasonable but systematically
  too narrow (the ENTSO-E pre-conformal case).
- Distribution shift is present but moderate — the study confirmed a
  19–45% increase in median-to-mean absolute error from calibration to
  test periods, which is substantial but not structural.
- The covariate-shift weighting via logistic density ratio correctly
  identifies the calibration-to-test shift, downweighting older
  calibration observations in favour of more recent ones.

This result is consistent with Tibshirani et al. (2019), who show that
weighted conformal prediction achieves valid marginal coverage under
covariate shift provided the density ratio is correctly estimated.
Adaptive conformal methods for time series (Zaffran et al., 2022) address
similar non-stationarity concerns through online quantile updates, though
their application here was constrained by the step-size issue described
in Section 2.2. The
temporal covariate (elapsed time with cyclic encodings) provides a
well-identified shift direction in this context.

### 2.2 When conformal augmentation fails

Two conformal variants fail in instructive ways.

**Online CP with fixed step size** collapses to approximately 50% coverage
at both nominal levels. The cause is explicit: a step size of 0.01 is
calibrated for normalised residuals of O(1), but ENTSO-E electricity load
errors are of O(10³). The quantile update rule moves at 0.01 units per
step while the true scale requires adjustments of hundreds of units,
causing the algorithm to stagnate. This is a configuration failure, not
a method failure — Angelopoulos et al. (2024) specifically address step
size calibration for non-stationary environments. The finding confirms
that online CP requires scale-aware initialisation and that naive
application to energy market data without this adjustment is
counterproductive.

**Point split conformal** undercovers at 65.4% (α = 0.2) because it uses
a constant-width interval derived from calibration-period residuals. When
the test distribution shifts — as confirmed by the error increase — a
fixed quantile cannot adapt, and coverage deteriorates in direct proportion
to the magnitude of the shift. This establishes a clear condition under
which conformal augmentation does not improve over the base interval:
when the calibration-to-test shift is directional and the conformal
variant does not adapt to it.

### 2.3 What conformal augmentation does not fix

The conformal study was conducted on the ENTSO-E development sample, where
the pre-conformal coverage is 82.5% — clearly below the 90% nominal. But
conformal expansion corrects only interval width, not distributional shape.
The underlying PIT failures (uniformity and independence rejection) would
persist on the conformal-adjusted outputs. This is the critical distinction
for RQ2: conformal augmentation improves finite-sample coverage stability
under regime shifts, but it does not address distributional misspecification
or serial dependence. Classical PIT diagnostics and conformal augmentation
are therefore complementary, not substitutable. The conditions under which
conformal augmentation improves over PIT diagnostics alone are precisely
the conditions where the PIT failures are limited to coverage shortfall
rather than distributional shape or temporal structure.

---

## 3. RQ3 — Governance Classification

**RQ3:** Can probabilistic calibration diagnostics be aggregated into
statistically defensible governance classifications comparable to
Basel-style traffic-light backtesting systems?

The results demonstrate that such aggregation is feasible and produces
classifications that are both statistically grounded and institutionally
interpretable.

### 3.1 The traffic-light aggregation logic

The governance classification aggregates three diagnostic layers
(Anfuso interval, PIT uniformity, Ljung–Box independence) via a
hierarchical policy: RED is triggered if any layer fails beyond its
threshold. This is a conservative aggregation rule — a model must pass
all layers to achieve GREEN. The conservative direction is appropriate
for model governance: the cost of a false GREEN (using a miscalibrated
model) exceeds the cost of a false RED (triggering unnecessary review).

This logic follows the Basel framework (Basel Committee, 1996; 2010;
McNeil, Frey and Embrechts, 2005), where the yellow and red zones are
defined conservatively: models with more than 10 VaR exceedances over
250 days are automatically red-flagged regardless of other performance
indicators. The extension in this thesis
adds distributional and independence layers that Basel's simpler
exceedance count does not include, reflecting the richer diagnostic
information available from full density forecast evaluation.

### 3.2 Cross-model comparability

The governance framework produces classifications that are directly
comparable across model classes. All three real-data models receive RED
for reasons rooted in the same diagnostic vocabulary — PIT failure,
independence rejection, tail miscalibration — even though the underlying
failure mechanisms are structurally different. An institution operating
across all three model classes can apply the same governance policy,
with the failure mode detail in the risk_reasons field providing the
model-specific guidance for remediation.

The positive control (synthetic simulation) receives GREEN, validating
that the framework is not systematically over-strict. The three
misspecification scenarios correctly detect variance inflation and mean
bias as RED with the appropriate tail signature (bilateral for inflation,
unilateral for bias). These results confirm that the classification is
statistically defensible: it responds correctly to known model states.

The production architecture reinforces this defensibility through a
structured provenance mechanism. Every `GovernanceDecision` produced by
the `DecisionEngine` records which diagnostic branches were computed,
which were skipped and why, which policy was applied, and the timestamp
of the decision — a full audit trail in machine-readable JSON. This is
not merely an engineering convenience: it is a direct response to the
governance requirement that classification decisions be explainable and
reproducible. An institution presenting a RED classification to a regulator
or internal risk committee can accompany it with the exact metric snapshot,
reason codes, and policy parameters that produced it, satisfying the
auditability requirements implicit in REMIT Article 15 and the Basel
backtesting documentation obligations.

### 3.3 The heavy-tail boundary

The heavy-tails scenario receiving GREEN at n = 365 is a genuine
limitation of the framework as implemented, but it is also a
statistically honest finding. The Anfuso binomial test at n = 365 has
insufficient power to detect t(df=3) excess in symmetric tails because
at this sample size the rare excess events do not accumulate enough to
cross the rejection threshold. This is not a flaw in the classification
logic — it is a manifestation of the fundamental trade-off between
Type I and Type II error in finite-sample testing.

The governance implication is a minimum evaluation horizon requirement:
institutions applying this framework to models with short backtesting
windows (n < 500) should incorporate supplementary tail diagnostics
(e.g., PIT histogram inspection in the tails, or extreme value tests)
alongside the interval backtesting layer. This is consistent with the
Basel framework's requirement for a minimum 250-trading-day evaluation
window, extended here to account for the lower frequency of the
renewable datasets.

### 3.4 Governance Communication via AI Narratives

A governance classification is only useful if it can be communicated to
the right decision-makers in the right form. The `NarrativeGenerator`
component addresses this by converting structured `GovernanceDecision`
objects into two parallel narrative registers: a technical summary for
quantitative risk officers that references specific metric values, names
failed diagnostic branches, and states capital multiplier implications;
and a plain-language summary for non-technical stakeholders that explains
the classification without jargon and identifies the required action.

Both registers are generated from the same structured input in a single
API call, ensuring consistency between the technical and non-technical
accounts. This is architecturally significant: the narrative is downstream
of the classification, not upstream. The governance decision is produced
deterministically by the diagnostic pipeline; the language model only
translates it. This preserves the statistical defensibility of the
classification while extending its reach to audiences who could not
otherwise engage with the underlying diagnostics.

The Streamlit web application (`unified-probabilistic-validation.streamlit.app`)
operationalises this for non-technical users: a non-Python analyst at an
energy trading firm can upload a forecast CSV, receive a governance
decision with full metric snapshot and AI narrative, and download all
artifacts as a ZIP — without installing software or writing code. This
closes the last gap between the statistical framework and practical
institutional deployment.

---

## 4. Limitations

### 4.1 Distribution reconstruction is not unique

The rolling empirical quantile reconstruction used as the pre-conformal
baseline is one of several defensible choices. The feasibility analysis
(02_entsoe_feasibility.ipynb) compared multiple variants and selected
the 4-bucket coarse conditioning for ENTSO-E based on empirical coverage
performance. For PV and wind, 24-bucket conditioning was used without
comparative evaluation. Different reconstruction choices would produce
different pre-conformal baselines and potentially different governance
outcomes. The framework's outputs should therefore be understood as
conditional on the reconstruction method, not as absolute assessments
of the underlying model.

### 4.2 PIT diagnostics unavailable for simulation model class

The simulation runs pass only quantile bounds (lo, hi) rather than full
path arrays to the diagnostic layer. As a result, PIT-based uniformity
and independence statistics are not computed for run_004 and run_004b.
The positive control and misspecification results are therefore based
solely on interval backtesting, which is sufficient to distinguish the
scenarios but does not provide the full distributional diagnostic picture.
Extending to full PIT evaluation would require persisting the per-horizon
empirical CDF from the simulation path ensemble — a straightforward
pipeline extension.

### 4.3 Conformal evaluation is limited to one dataset and sample

The conformal augmentation study uses only the 90-day ENTSO-E development
sample (n = 2,544 test observations). Extension to the full ENTSO-E
dataset, PV, and wind would require re-running the conformal variants on
those datasets, which is deferred. The PV and wind coverage shortfalls
(−1.37 pp and −1.38 pp respectively) are smaller than ENTSO-E's (−2.94
pp), so the marginal benefit of conformal augmentation on those datasets
is expected to be smaller — but this is not empirically confirmed.

### 4.4 Multivariate extension covers only two assets

The run_005 multivariate analysis is restricted to PV and wind, the only
two datasets sharing a common time index. The industry partner identified
additional commodity classes (natural gas, carbon) as desirable scope
extensions. Synthetic extension via the simulation notebook is planned
but not yet implemented. The current multivariate results therefore
characterise intra-renewable dependence rather than cross-commodity
dependence, which would be the more economically relevant dimension for
portfolio risk management.

### 4.5 Exchangeability assumption in conformal prediction

Conformal prediction's finite-sample coverage guarantee relies on the
exchangeability of calibration and test observations (Vovk, Gammerman
and Shafer, 2005). Energy market data are not exchangeable — they exhibit
serial dependence, seasonal structure, and structural breaks. Extensions
of conformal prediction beyond exchangeability exist (Barber et al., 2023)
but require additional structural assumptions. The covariate-shift
weighting partially addresses this by reweighting calibration observations,
but it does not provide formal guarantees under strong temporal dependence. The conformal results
should therefore be interpreted as approximately valid rather than
strictly valid in the theoretical sense.

---

## 5. Relation to Prior Work

This thesis extends the calibration evaluation literature (Diebold et al.,
1998; Berkowitz, 2001) by applying density forecast diagnostics to energy
market model classes that are not normally evaluated in this way. The
finding that interval coverage is an insufficient governance criterion —
demonstrated empirically on the PV dataset — is consistent with
theoretical arguments in Gneiting and Katzfuss (2014) but contributes
a concrete applied demonstration in a domain where coverage-only
evaluation is standard practice.

The conformal prediction results extend Tibshirani et al. (2019) to an
energy market context, confirming that covariate-shift-aware conformal
methods are effective under temporal distribution shift while identifying
the conditions (scale mismatch, constant-width construction) under which
they fail. Prior applications of conformal prediction to electricity price
forecasting (Kath and Ziel, 2021; O'Connor et al., 2025) demonstrate
reliable coverage in day-ahead markets; this thesis extends that evidence
to load and renewable generation forecasting and embeds it within a
governance classification architecture. The online CP step-size failure specifically corroborates and
extends the practical guidance in Angelopoulos et al. (2024), whose
theoretical results on decaying step sizes are motivated by exactly the
kind of scale sensitivity observed here.

The governance aggregation layer contributes to the literature on
Basel-style backtesting (Kupiec, 1995; Christoffersen, 1998) by
demonstrating that the traffic-light architecture can be extended beyond
exceedance counting to incorporate full density diagnostics, producing
richer and more discriminative governance classifications without
sacrificing the institutional interpretability that makes the Basel
framework practically useful.

---

## 6. Economic Distortion and Governance Stability (RQ3 Extension)

### 6.1 Capital Multiplier and Reserve Sizing

The VaR sensitivity analysis (run_006) adds a concrete economic dimension
to the RQ3 findings. Two results stand out as particularly thesis-relevant.

The PV divergence is the most important single finding for the governance
argument. Under a coverage-only regulatory framework adapted from Basel,
PV would attract a reduced capital multiplier (−6.7% distortion,
CONSERVATIVE zone) because its empirical coverage exceeds the 90% nominal.
Under the full multi-layer framework, PV receives a RED governance
classification. These two conclusions point in opposite directions: a
regulator relying on the simpler framework would relax oversight of a
model that the richer framework identifies as structurally misspecified.
This is not a theoretical concern — it is demonstrated empirically on
real industry data. It constitutes the strongest quantitative argument
in the thesis for why coverage metrics alone are insufficient for
energy market model governance.

The reserve sizing analysis complements this by translating coverage
errors into operational terms. ENTSO-E and wind produce reserve undersizing
of 2.94 pp and 1.38 pp respectively — modest in absolute terms but
systematic and persistent across the full evaluation period. The
misspecification scenarios demonstrate that severe calibration failures
(variance inflation, mean bias) produce reserve undersizing of 20–35 pp,
which would require reserves 2–3.5× larger than modelled to achieve
intended coverage. This quantifies the operational cost of miscalibration
in terms directly interpretable by energy market practitioners.

### 6.2 Traffic-Light Stability and Absorbing States

The transition probability analysis (run_007) adds a temporal dimension
to the governance findings that the full-sample results cannot provide.
The identification of RED as an absorbing state for all three real-data
models (T_RR = 1.0, H = 0 across hundreds of rolling windows) is a
stronger governance statement than the full-sample RED classification
alone. It means the miscalibration is not a transient artefact of a
particular market regime — it is a stable structural property that
persists regardless of which subperiod is evaluated.

This has a direct implication for governance policy. A model that is
persistently in an absorbing RED state cannot be remediated by waiting
for market conditions to change. It requires structural intervention —
either re-estimation of the underlying model or deployment of a
recalibration layer (such as conformal augmentation) that actively
corrects coverage in real time.

The heavy-tails finding from the rolling analysis is also noteworthy.
The price heavy-tails scenario produces a stable YELLOW absorbing state
across all 7 non-overlapping rolling windows, despite the full-sample
Anfuso test returning GREEN. This reveals a detection mechanism: while
full-sample interval backtesting at n = 365 lacks power to detect
t(df=3) excess, the rolling window classification consistently identifies
borderline behaviour. Rolling diagnostics therefore serve as a
complementary detection tool for subtle misspecification that evades
full-sample tests — a practical finding with direct implications for
governance monitoring frequency.

### 6.3 Limitations of the Economic Analyses

The capital multiplier analysis is an adaptation of the Basel framework,
not a direct application. The original Basel thresholds were calibrated
for 99% VaR on trading book positions with 1-day horizon; the excess-based
adaptation for 90% coverage intervals is a methodological contribution
of this thesis rather than a regulatory standard. Practitioners applying
this framework should recalibrate the zone thresholds to their specific
regulatory context.

The reserve sizing analysis assumes a direct mapping from prediction
interval coverage to reserve adequacy, which ignores market-clearing
dynamics, reserve substitutability, and intraday balancing mechanisms.
The percentage point errors reported here are therefore indicative of
the direction and approximate magnitude of miscalibration impact, not
precise operational reserve shortfall estimates.

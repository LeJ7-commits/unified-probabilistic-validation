# Governance Implications

## 1. From Diagnostics to Decisions

The empirical results presented in Chapter 3 are not ends in themselves.
A RED classification carries consequences only if it is connected to a
decision — about whether a model should continue to be used, whether its
outputs should be adjusted, and what institutional actions are warranted.
This chapter translates the diagnostic findings into governance
recommendations, situates them within established regulatory frameworks,
and addresses the specific implications of conformal augmentation as a
recalibration tool within a model risk management context.

The starting point is the cross-dataset governance summary. Of the three
real-data model classes evaluated, all three receive RED overall
classifications. The synthetic positive control receives GREEN, confirming
that the framework is not systematically over-strict. The RED findings
are not uniform: each model class fails on a structurally distinct
combination of diagnostic dimensions, which has direct implications for
the nature of the required remediation.

---

## 2. Interpreting RED by Failure Mode

### 2.1 Bilateral Interval Miscalibration — ENTSO-E Load Forecasting

The ENTSO-E model fails on all three diagnostic layers: interval
backtesting (both tails RED), PIT uniformity (strongly rejected), and
serial independence (overwhelmingly rejected). Empirical coverage of
87.1% against a 90% nominal target, combined with Ljung–Box statistics
several orders of magnitude above critical values, indicates that the
model's predictive intervals are both too narrow and temporally
misspecified.

**Governance implication.** A model in this state should not be used
for risk quantification without explicit adjustment. The bilateral
nature of the breach — both tails significantly over-breach, not just
one — indicates that the miscalibration is not directional bias but
a general underestimation of uncertainty. The serial dependence finding
compounds this: the model fails to account for autocorrelation in load
residuals, meaning consecutive interval violations are not independent
events. For risk aggregation across time, this non-independence makes
the effective exceedance probability higher than the marginal rate
suggests.

Recommended actions: (i) re-estimate the residual distribution
reconstruction with longer rolling windows or regime-conditioned
quantiles; (ii) apply conformal expansion as a post-hoc coverage
correction pending re-estimation; (iii) escalate to model review
committee with full diagnostic evidence.

### 2.2 Distributional Failure Without Interval Failure — PV Solar

The PV model presents the most instructive case for the multi-layer
framework argument. It passes interval backtesting (GREEN Anfuso,
coverage 91.4%) while failing PIT uniformity and independence
diagnostics (both strongly rejected). Under a naive coverage-only
governance policy, this model would be classified as acceptable.
Under the unified framework, it is classified RED.

**Governance implication.** The interval width is approximately
correct on average, but the distributional shape is systematically
misspecified. This means the model produces reasonable 90% intervals
but unreliable quantiles at other levels — in particular, any
risk metric that depends on tail quantiles beyond the evaluated
interval (e.g., 95th or 99th percentile) cannot be trusted. For
a model used in renewable generation planning or capacity adequacy
assessment, this is a material limitation.

This case directly demonstrates that interval-coverage metrics alone
are insufficient for governance classification. The RED classification
is appropriate and should be communicated with the specific
diagnosis: the failure is distributional shape and temporal
dependence, not coverage level.

### 2.3 Asymmetric Lower-Tail Failure — Wind Generation

The wind model fails on lower-tail interval backtesting (RED, 6.44%
breach rate vs 5% nominal) while the upper tail is within tolerance
(GREEN, 4.93%). Total coverage is 88.6%. PIT uniformity and
independence are strongly rejected.

**Governance implication.** The asymmetric failure pattern has a
physically interpretable cause: wind generation is bounded below by
zero, and the model underestimates the probability of low-generation
events more than high-generation events. For risk management purposes,
this is a one-directional conservatism failure — the model is more
dangerous in downside scenarios (low wind, high demand) than in
upside scenarios. Any risk measure sensitive to the lower tail of
the generation distribution — such as capacity shortfall probability
or balancing reserve sizing — will be systematically underestimated
by this model. The governance response should target lower-tail
recalibration specifically, rather than global interval adjustment.

---

## 3. The Misspecification Detection Boundary

The controlled misspecification scenarios reveal where the framework's
discriminative power ends. Variance inflation and mean bias are detected
reliably and their failure mode character (bilateral vs unilateral) is
correctly identified. Heavy-tail misspecification is not detected at
n = 365.

**Governance implication.** This finding establishes a practical
boundary for governance confidence: the traffic-light framework is
reliable for detecting the most consequential model failures
(wrong scale, systematic bias) but requires larger evaluation samples
or supplementary tail diagnostics to detect distributional shape
misspecification. Institutions using this framework with short
evaluation horizons (n < 500) should be aware that a GREEN interval
classification does not rule out heavy-tail misspecification. Minimum
evaluation horizon recommendations should be incorporated into
governance policy as a procedural safeguard.

---

## 4. Regulatory Mapping

### 4.1 Basel Traffic-Light Alignment

The traffic-light architecture developed in this thesis is explicitly
modelled on the Basel Committee's backtesting framework for Value-at-Risk
models (Basel Committee, 1996; 2010), which classifies models into green,
yellow, and red zones based on the number of VaR exceedances over a
250-day evaluation window. The adaptation in this framework extends the
Basel logic in two directions.

First, the diagnostic battery is broadened. Basel's framework tests only
exceedance frequency (whether the observed breach rate exceeds the nominal
rate). This thesis adds PIT uniformity and serial independence as
mandatory governance criteria, reflecting the finding that interval
coverage alone is insufficient (the PV case). This is consistent with
the direction of Basel III and IV, which have progressively expanded the
scope of backtesting beyond simple VaR exceedance counting.

Second, the framework is applied across heterogeneous model classes.
Basel's framework was designed for a single model type (internal market
risk models for trading book positions). The extension to simulation
models, short-term forecasters, and long-term renewable models with
heterogeneous output formats and evaluation sample sizes requires the
distribution reconstruction and sample-size-aware power calibration
described in Chapter 2.

### 4.2 REMIT Regulatory Context

The REMIT regulation (Regulation on Wholesale Energy Market Integrity and
Transparency) requires that energy market participants maintain reliable
and auditable models for fundamental price formation and risk
quantification. While REMIT does not prescribe specific backtesting
methodologies, the governance framework developed here — with its
structured RED/YELLOW/GREEN classification, documented diagnostic
evidence, and explicit model use restrictions at RED — is directly
compatible with REMIT's requirement for transparent and auditable model
governance.

In particular, the traffic-light output is designed to be interpretable
by non-statistician stakeholders (trading desks, risk committees,
independent validation units) without loss of the underlying statistical
rigour. This addresses a practical gap: statistical calibration results
are often not directly consumable by institutional decision-makers
without a translation layer.

---

## 5. The Role of Conformal Augmentation in Governance

Conformal augmentation occupies a specific and bounded role in the
governance architecture. It is not a substitute for model re-estimation
and it does not correct distributional shape failures — it corrects
interval width. The distinction matters:

- If a model fails on **interval coverage only** (breach rate too high
  but PIT diagnostics acceptable), conformal expansion is an appropriate
  post-hoc correction that restores nominal coverage with finite-sample
  guarantees.
- If a model fails on **PIT uniformity or independence** (the ENTSO-E,
  PV, and wind cases), conformal expansion can restore marginal interval
  coverage but does not fix the underlying distributional misspecification.
  The model will still produce unreliable quantile estimates at levels
  other than the conformal-corrected interval.

The feasibility study on the ENTSO-E development sample (Chapter 3,
Section 8; 04_conformal_wrapping.ipynb) confirms this: base interval +
conformal expansion restores 89.9% empirical coverage at α = 0.1 (from
a pre-conformal 82.5%), but the underlying PIT failures would remain
present if tested on the conformal-adjusted outputs. Conformal
augmentation is therefore best understood as a **coverage stabiliser**
within the governance architecture — a first-line remediation while
structural model improvement is underway, not a permanent fix.

For governance purposes, a model that is RED pre-conformal and GREEN
on interval coverage post-conformal should be classified as
**GREEN (with conformal adjustment, pending structural review)** — not
simply GREEN. The governance record should document the adjustment and
require periodic re-assessment.

---

## 6. Limitations and Future Extensions

Several limitations of the current framework are relevant to governance
practitioners.

**PIT diagnostics unavailable for simulation model class.** The synthetic
simulation runs (run_004, run_004b) pass quantile-based interval
diagnostics but do not produce PIT statistics, because the current
pipeline passes only quantile bounds rather than full distributional
samples to the diagnostic layer. A governance policy that relies on
PIT-layer evidence for simulation models would require persisting the
full path ensemble quantile CDFs — a straightforward extension that is
deferred from the current implementation.

**Sample size constraints on renewable datasets.** After nighttime
exclusion and rolling warmup, the PV evaluation sample reduces to
n = 4,287 observations. At this sample size, the Anfuso interval test
has substantially reduced power relative to the ENTSO-E run (n = 209,555).
A GREEN Anfuso result on PV should be interpreted conservatively.

**Multivariate coverage is partial.** The run_005 multivariate analysis
covers only PV and wind on their shared hourly index. The industry
partner has expressed interest in expanding the framework to additional
commodity classes (natural gas, carbon, electricity price). The
architecture is designed to accommodate this expansion: the diagnostic
modules are asset-agnostic, and the build scripts can be extended to
additional datasets. Synthetic extensions to additional commodity classes
using the simulation notebook are planned as a next step.

**Conformal augmentation evaluated on development sample only.** The
conformal feasibility study uses the 90-day ENTSO-E development sample
(n = 2,544 test observations). Full-dataset conformal evaluation and
extension to PV and wind are deferred. The governance implications of
conformal augmentation for the renewable datasets — where the
pre-conformal coverage shortfalls are smaller — remain to be quantified.

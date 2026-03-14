# Governance Implications

## 1. From Diagnostic Results to Governance Decisions

Probabilistic model validation is only useful insofar as its outputs inform
decisions. The traffic-light framework implemented in this thesis — drawing
on the interval backtesting approach of Anfuso et al. (2017) and the
distributional diagnostics of Berkowitz (2001) and Diebold et al. (1998) —
produces governance classifications (GREEN / AMBER / RED) designed to trigger
proportionate responses from model owners and risk oversight functions.

The empirical results across all four model classes (ENTSOG, PV, wind,
simulation) demonstrate that this system behaves as intended: it returns RED
for models with genuine structural deficiencies, GREEN for a well-specified
positive control, and — critically — RED for a model (PV) that passes naive
coverage checks but fails distributional and independence diagnostics. This
last case is the most governance-relevant finding of the thesis: it shows that
a model can satisfy the weakest validation criterion (empirical coverage ≈
nominal) while exhibiting systematic departures from its assumed probabilistic
structure that would go undetected without multi-layer diagnostic evaluation.

---

## 2. What a RED Classification Means in Practice

A RED classification under this framework does not necessarily mean a model
must be withdrawn from use. It means the model's uncertainty estimates cannot
be relied upon at face value for risk-sensitive decisions — specifically:

- **Interval-based risk limits** (e.g. Value-at-Risk, scenario bounds) derived
  from RED-classified models carry unquantified additional uncertainty and
  should be supplemented with conservative buffers or stress adjustments.
- **Portfolio aggregation** using RED-classified marginal distributions
  propagates miscalibration into joint risk measures; correlations and tail
  dependencies computed from such distributions are unreliable.
- **Model-based hedging or dispatch decisions** that depend on the tails of
  the predictive distribution (e.g. extreme price or generation scenarios) are
  most exposed to the identified deficiencies, particularly the systematic
  under-coverage and serial dependence found in ENTSOG and wind.

The appropriate governance response to RED depends on which diagnostic layer
triggered the classification. Three cases arise from the empirical results:

**Case 1 — Coverage failure with distributional failure (ENTSOG, wind).**
The model's intervals are too narrow and the distributional form is misspecified.
Recommended response: apply conformal expansion to restore interval coverage
before use in risk calculations, and flag serial dependence as a structural
limitation requiring model re-estimation or residual autocorrelation correction.

**Case 2 — Coverage pass with distributional failure (PV).**
The model produces well-calibrated interval widths on average but fails PIT
uniformity and independence tests. Recommended response: do not treat coverage
statistics alone as evidence of calibration; flag the model for enhanced
monitoring and investigate whether the distributional failures are concentrated
in specific regimes (e.g. seasonal transitions, extreme irradiance events).

**Case 3 — Full pass (simulation positive control).**
All diagnostic layers pass. The model can be used for risk calculations with
standard uncertainty quantification. Periodic re-validation is still required
as the DGP may evolve.

---

## 3. Mapping to Regulatory Standards

The traffic-light architecture of this framework is directly analogous to the
Basel Committee's interval forecast backtesting framework for internal market
risk models (Basel Committee on Banking Supervision, 1996; 2010). Under Basel,
VaR models are evaluated against a one-year window of daily breach counts, with
GREEN, AMBER, and RED zones defined by binomial critical values — the same
statistical logic used here. The key extension in this thesis is the addition
of distributional and independence diagnostic layers that the Basel framework
does not require but that are necessary for energy market models where:

- Forecasts are probabilistic density estimates rather than single quantiles
- Time series exhibit strong seasonal and meteorological autocorrelation
- Model classes span short-term operational forecasting (ENTSOG) and long-term
  scenario simulation (PV, wind, pricing)

In the EU energy market context, REMIT (Regulation on Wholesale Energy Market
Integrity and Transparency) and the guidelines of ACER (Agency for the
Cooperation of Energy Regulators) require that market participants use
models that produce reliable price and volume forecasts for trading and
risk management. While REMIT does not prescribe a specific validation
methodology, the RED classifications produced here — particularly the severe
serial dependence and bilateral tail miscalibration of ENTSOG — would
constitute material model risk under any standard risk governance framework
and would require documented remediation.

The industry partner's confirmation that both full-sample and rolling window
evaluation are valuable and complementary (rather than one superseding the
other) aligns with the Basel supervisory guidance that internal model
validation should assess both unconditional and conditional coverage — i.e.
whether the model is calibrated on average and whether it is calibrated
consistently across time.

---

## 4. How Conformal Augmentation Changes Governance Outcomes

The conformal wrapping layer (RQ2) changes the governance picture in a specific
and bounded way. As demonstrated in `04_conformal_wrapping.ipynb`, the
base + conformal expansion method restores near-nominal interval coverage
(89.9% at α = 0.1) on the ENTSOG dev sample, reducing the coverage error from
−7.5 pp (base interval alone) to −0.1 pp after expansion.

This has a direct governance implication: a model that receives RED due to
coverage failure alone can be upgraded to a governance-acceptable interval
through conformal recalibration, **without re-estimating the underlying model**.
This is operationally significant — re-estimation of energy market simulation
models is expensive and time-consuming, while conformal expansion requires only
a calibration set of historical residuals.

However, conformal expansion does not address distributional misspecification
or serial dependence. A model that fails PIT uniformity or Ljung–Box
independence tests after conformal expansion remains RED on those dimensions.
The governance implication is that conformal augmentation should be understood
as a **coverage repair mechanism**, not a full model rehabilitation tool. It is
a necessary but not sufficient condition for governance compliance in frameworks
that require full distributional calibration.

This distinction — between interval coverage (which conformal can fix) and
distributional form (which it cannot) — is the central practical contribution
of the conformal layer to the governance architecture.

---

## 5. Limitations and Directions for Future Work

**Multivariate scope.** The joint PV–wind evaluation in `run_005_multivariate.py`
confirms that the marginal predictive distributions are jointly serially
dependent (multivariate Ljung-Box statistics of 15,427–24,714 against
chi-squared critical values, p ≈ 0 at all lags) and exhibit a modest but
persistent lag-24 cross-correlation of +0.075, reflecting shared diurnal
weather persistence. The bivariate energy score baseline of 2.017 provides a
reference for future joint model improvement. However, the framework currently
covers only PV and wind jointly. Energy market risk is inherently broader: gas
prices, carbon prices, and electricity prices are jointly distributed with
renewable generation and their correlations matter materially for portfolio
risk. The industry partner explicitly requested extension to at least three
additional commodity classes (natural gas, carbon, electricity). A full
multivariate extension using copula-based joint PIT evaluation across all five
or more asset classes remains the most important direction for future work.

**Simulation model PIT evaluation.** The h=1 artifact structure used for the
simulation positive control precludes full PIT computation. A natural extension
is to save the full empirical CDF per as-of date and evaluate the PIT score
analytically against the known Gaussian DGP, enabling the distributional
diagnostics to be applied to simulation models as well as forecasting models.

**Misspecification scenarios.** Only the well-specified simulation case was
evaluated. Deliberate misspecification scenarios (variance inflation, mean bias,
heavy tails) were deferred. These would strengthen the discriminative validity
of the framework by demonstrating that RED classifications are triggered
reliably across a range of known failure modes.

**Conformal adaptation under regime shifts.** The online CP step-update method
failed due to scale mismatch. A natural extension is adaptive step sizing
(proportional to recent residual scale), which would make the online method
viable for energy series with heteroskedastic volatility regimes.

**Operational integration.** The framework is currently implemented as a
batch evaluation pipeline. Integration into a live model monitoring system —
with automated traffic-light updates, alert thresholds, and escalation
protocols — would be required for production governance use. This is a
systems engineering extension beyond the scope of this thesis but is the
natural next step for the industry partner.

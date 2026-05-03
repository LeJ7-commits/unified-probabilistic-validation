# Governance Narrative: simulation_temp_variance_inflation

**Classification:** RED

---

## Technical Summary

The Monte Carlo temperature simulation model exhibits severe undercoverage with empirical coverage of 62.47% against the 90% target, representing a coverage error of -27.53 percentage points. The Anfuso traffic-light diagnostic returns RED for both lower and upper tails with total breach rate of 37.53%, substantially exceeding acceptable thresholds for Basel-compliant risk models. Mean pinball loss of 0.79 indicates poor probabilistic calibration, while the variance inflation misspecification manifests through excessive interval breaches (19.18% lower, 18.36% upper). This RED classification triggers immediate capital multiplier penalties under Basel III market risk framework and requires suspension of the model for REMIT position reporting until recalibration addresses the systematic undercoverage.

---

## Plain Language Summary

The temperature forecasting model is performing poorly and cannot be trusted for critical business decisions. Like a weather forecast that consistently underestimates storm intensity, this model's predictions are too narrow - it captures actual outcomes only 62% of the time when it should capture 90%. The model is failing basic reliability tests across all measures, meaning it's significantly underestimating the range of possible temperature scenarios. This creates serious risk exposure for energy trading positions and regulatory compliance, requiring immediate suspension of the model and implementation of alternative forecasting methods until the underlying technical issues are resolved.

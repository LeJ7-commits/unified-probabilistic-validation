# Technical Governance Narrative

**Model:** simulation_temp_variance_inflation  
**Classification:** RED  
**API generated:** True

---

The Monte Carlo temperature simulation model exhibits severe undercoverage with empirical coverage of 62.47% against the 90% target, representing a coverage error of -27.53 percentage points. The Anfuso traffic-light diagnostic returns RED for both lower and upper tails with total breach rate of 37.53%, substantially exceeding acceptable thresholds for Basel-compliant risk models. Mean pinball loss of 0.79 indicates poor probabilistic calibration, while the variance inflation misspecification manifests through excessive interval breaches (19.18% lower, 18.36% upper). This RED classification triggers immediate capital multiplier penalties under Basel III market risk framework and requires suspension of the model for REMIT position reporting until recalibration addresses the systematic undercoverage.

# Governance Narrative: entsoe_load

**Classification:** RED

---

## Technical Summary

The ENTSO-E load forecasting model has been classified RED under the validation framework due to systematic failures across multiple diagnostic branches. The probability integral transform (PIT) exhibits severe non-uniformity with Kolmogorov-Smirnov statistic of 0.161 (p<0.001) and pronounced serial dependence indicated by Ljung-Box statistics exceeding 846,000 at all tested lags (5, 10, 20) with p-values of zero. The empirical coverage of 87.06% falls materially short of the 90% policy target, generating a coverage error of -2.94 percentage points, while the Anfuso traffic-light system shows RED across all branches with total breach rate of 12.94%. Under Basel capital requirements, this RED classification triggers enhanced capital multipliers and necessitates immediate model remediation with potential REMIT disclosure obligations given the systematic undercoverage in critical electricity market forecasts.

---

## Plain Language Summary

The electricity demand forecasting model for Germany is performing poorly and has been flagged as high-risk. Think of it like a weather forecast that consistently underestimates storm intensity - the model is producing prediction ranges that are too narrow and missing actual outcomes about 13% of the time when it should only miss 10%. This systematic underestimation creates significant trading and operational risks, as energy companies may be inadequately prepared for actual demand levels. The model must be immediately taken out of service for repairs and recalibration, and additional capital reserves may be required to cover the increased uncertainty until the forecasting accuracy is restored to acceptable standards.

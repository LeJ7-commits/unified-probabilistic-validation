# Governance Narrative: wind_onshore

**Classification:** RED

---

## Technical Summary

The wind onshore probabilistic forecast model has been classified RED due to critical failures in two diagnostic branches: PIT uniformity and autocorrelation structure. The probability integral transform exhibits severe distributional distortion with a Kolmogorov-Smirnov statistic of 0.106 (p<0.001) and extreme temporal dependence evidenced by a lag-1 autocorrelation of 0.855, with Ljung-Box statistics exceeding 25,000 across all tested lags (p<0.001). The Anfuso traffic-light framework confirms RED classification with a total breach rate of 11.4% against the 90% coverage target, though the upper tail performs adequately (GREEN). Under Basel-aligned governance protocols, this RED classification triggers immediate model replacement requirements and may impose capital multipliers for market risk positions dependent on these forecasts, while also creating potential REMIT reporting obligations for forecast-dependent trading strategies.

---

## Plain Language Summary

The wind farm forecasting model is performing poorly and has been flagged as high-risk (RED status). Think of weather forecasting - if the model consistently predicted sunny days when it rained, or if today's wrong prediction made tomorrow's prediction wrong too, you'd lose trust in the forecast. That's essentially what's happening here: the model's predictions aren't matching reality often enough (failing about 11% more than acceptable), and its errors are creating patterns that compound over time. This means we cannot rely on this model for critical business decisions like energy trading or grid planning, and we must either fix it immediately or replace it with a more reliable alternative to avoid potential financial losses and regulatory issues.

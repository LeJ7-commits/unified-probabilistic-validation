# Governance Narrative: entsoe_wind_de

**Classification:** RED

---

## Technical Summary

The ENTSO-E day-ahead wind generation forecast model for Germany has received a RED classification due to critical failures in probability integral transform (PIT) diagnostics. The model exhibits severe temporal dependence with an ACF lag-1 coefficient of 0.861 and Ljung-Box statistics exceeding 101,000 across all tested lags (5, 10, 20), indicating systematic forecast bias patterns that violate the independence assumption required for probabilistic calibration. Additionally, uniformity tests show marginal warnings with Kolmogorov-Smirnov p-value of 0.0016 and Cramér-von Mises p-value of 0.0008, while Anfuso backtesting returns RED across all traffic-light categories with total breach rate of 10.84% against the 90% coverage target. Under Basel framework requirements, this RED classification triggers increased capital multipliers and necessitates immediate model remediation before continued deployment for REMIT reporting obligations.

---

## Plain Language Summary

The wind power forecasting model for Germany is not performing reliably and has been flagged as high-risk. Think of it like a weather app that consistently gets the forecast wrong in predictable patterns - when it's wrong today, it tends to be wrong tomorrow in the same way, which means we can't trust its confidence intervals. The model is missing its accuracy targets by failing to capture the right amount of uncertainty 89% of the time instead of the required 90%, and more concerning, these errors follow patterns that suggest the model has learned the wrong lessons from historical data. This means we need to stop using this model for regulatory reporting and trading decisions until it can be fixed, as continuing to use it could expose the company to significant financial and compliance risks.

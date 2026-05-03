# Technical Governance Narrative

**Model:** entsoe_wind_de  
**Classification:** RED  
**API generated:** True

---

The ENTSO-E day-ahead wind generation forecast model for Germany has received a RED classification due to critical failures in probability integral transform (PIT) diagnostics. The model exhibits severe temporal dependence with an ACF lag-1 coefficient of 0.861 and Ljung-Box statistics exceeding 101,000 across all tested lags (5, 10, 20), indicating systematic forecast bias patterns that violate the independence assumption required for probabilistic calibration. Additionally, uniformity tests show marginal warnings with Kolmogorov-Smirnov p-value of 0.0016 and Cramér-von Mises p-value of 0.0008, while Anfuso backtesting returns RED across all traffic-light categories with total breach rate of 10.84% against the 90% coverage target. Under Basel framework requirements, this RED classification triggers increased capital multipliers and necessitates immediate model remediation before continued deployment for REMIT reporting obligations.

# Technical Governance Narrative

**Model:** wind_onshore  
**Classification:** RED  
**API generated:** True

---

The wind onshore probabilistic forecast model has been classified RED due to critical failures in two diagnostic branches: PIT uniformity and autocorrelation structure. The probability integral transform exhibits severe distributional distortion with a Kolmogorov-Smirnov statistic of 0.106 (p<0.001) and extreme temporal dependence evidenced by a lag-1 autocorrelation of 0.855, with Ljung-Box statistics exceeding 25,000 across all tested lags (p<0.001). The Anfuso traffic-light framework confirms RED classification with a total breach rate of 11.4% against the 90% coverage target, though the upper tail performs adequately (GREEN). Under Basel-aligned governance protocols, this RED classification triggers immediate model replacement requirements and may impose capital multipliers for market risk positions dependent on these forecasts, while also creating potential REMIT reporting obligations for forecast-dependent trading strategies.

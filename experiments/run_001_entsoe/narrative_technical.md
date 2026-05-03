# Technical Governance Narrative

**Model:** entsoe_load  
**Classification:** RED  
**API generated:** True

---

The ENTSO-E load forecasting model has been classified RED under the validation framework due to systematic failures across multiple diagnostic branches. The probability integral transform (PIT) exhibits severe non-uniformity with Kolmogorov-Smirnov statistic of 0.161 (p<0.001) and pronounced serial dependence indicated by Ljung-Box statistics exceeding 846,000 at all tested lags (5, 10, 20) with p-values of zero. The empirical coverage of 87.06% falls materially short of the 90% policy target, generating a coverage error of -2.94 percentage points, while the Anfuso traffic-light system shows RED across all branches with total breach rate of 12.94%. Under Basel capital requirements, this RED classification triggers enhanced capital multipliers and necessitates immediate model remediation with potential REMIT disclosure obligations given the systematic undercoverage in critical electricity market forecasts.

# Technical Governance Narrative

**Model:** entsoe_load  
**Classification:** RED  
**API generated:** True

---

The ENTSO-E load forecast model has been classified RED under the probabilistic validation framework due to multiple diagnostic failures across coverage and temporal dependence criteria. The empirical coverage of 87.06% falls significantly below the 90% policy target, generating a -2.94pp coverage error that triggers undercoverage breach protocols. PIT uniformity diagnostics show complete failure with KS p-value = 0.0, CVM statistic = 2,032, and AD statistic = 6,136, indicating systematic bias in the probability integral transforms. The Ljung-Box autocorrelation tests demonstrate severe temporal dependence with statistics exceeding 846,000 at lag-5 (p-value = 0.0) and ACF(1) = 0.926, violating the independence assumption critical for risk capital calculations. Under Basel III market risk standards, this RED classification mandates immediate model remediation, potential capital multiplier penalties, and suspension from regulatory capital relief until validation standards are restored.

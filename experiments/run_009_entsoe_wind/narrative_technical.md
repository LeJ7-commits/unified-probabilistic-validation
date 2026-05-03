# Technical Governance Narrative

**Model:** entsoe_wind_de  
**Classification:** RED  
**API generated:** True

---

The ENTSO-E German wind forecast model (entsoe_wind_de) has been classified RED under our probabilistic validation framework due to critical failures in forecast calibration diagnostics. The probability integral transform (PIT) uniformity tests show significant deviation from expected uniform distribution (Kolmogorov-Smirnov p-value = 0.0016, Cramér-von Mises p-value = 0.0008), indicating systematic forecast bias. More critically, the autocorrelation function analysis reveals severe temporal dependence in forecast errors with lag-1 correlation of 0.86, while Ljung-Box tests confirm persistent serial correlation across all tested lags (p-values < 0.001). The Anfuso traffic-light system shows RED status across all breach categories with a total breach rate of 10.84% against our 90% coverage target, triggering enhanced capital requirements under our internal risk framework. This RED classification mandates immediate model recalibration, increased trading position limits, and potential REMIT reporting obligations for systematic forecast inadequacy in critical infrastructure forecasting.

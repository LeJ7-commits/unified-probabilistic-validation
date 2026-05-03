# Governance Narrative: entsoe_wind_de

**Classification:** RED

---

## Technical Summary

The ENTSO-E German wind forecast model (entsoe_wind_de) has been classified RED under our probabilistic validation framework due to critical failures in forecast calibration diagnostics. The probability integral transform (PIT) uniformity tests show significant deviation from expected uniform distribution (Kolmogorov-Smirnov p-value = 0.0016, Cramér-von Mises p-value = 0.0008), indicating systematic forecast bias. More critically, the autocorrelation function analysis reveals severe temporal dependence in forecast errors with lag-1 correlation of 0.86, while Ljung-Box tests confirm persistent serial correlation across all tested lags (p-values < 0.001). The Anfuso traffic-light system shows RED status across all breach categories with a total breach rate of 10.84% against our 90% coverage target, triggering enhanced capital requirements under our internal risk framework. This RED classification mandates immediate model recalibration, increased trading position limits, and potential REMIT reporting obligations for systematic forecast inadequacy in critical infrastructure forecasting.

---

## Plain Language Summary

Our wind power forecasting model for Germany has been flagged as unreliable and requires immediate attention. Think of weather forecasts - if they consistently predict sunny days but it keeps raining, and those wrong predictions follow a predictable pattern, you'd lose trust in the forecast service. That's essentially what's happening here: our model is making systematic errors in predicting wind power generation, and these errors are following patterns rather than being random. The model is failing to capture 10.8% more extreme events than it should, meaning we're underestimating risk in our trading positions. We must stop using this model for critical trading decisions, implement stricter risk controls, and urgently recalibrate the forecasting system before it can be trusted again for commercial operations.

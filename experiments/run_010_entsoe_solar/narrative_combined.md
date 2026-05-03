# Governance Narrative: entsoe_solar_de

**Classification:** RED

---

## Technical Summary

The ENTSO-E solar PV forecast model for Germany exhibits critical probabilistic failures warranting RED classification and immediate remedial action. The PIT uniformity diagnostics show severe deviation from expected uniform distribution (KS p-value = 0.0, CvM statistic = 7.45), while the ACF dependence tests reveal strong serial correlation at lag-1 (ρ = 0.788) with highly significant Ljung-Box statistics across all tested lags (p-values = 0.0). Despite the Anfuso traffic-light system showing GREEN status with acceptable breach rates (10.06% total vs 10% target), the fundamental probabilistic calibration failures indicate the forecast intervals lack proper statistical foundation. Under Basel energy trading book requirements, this RED classification mandates a capital multiplier penalty and triggers REMIT model performance reporting obligations until recalibration restores statistical validity.

---

## Plain Language Summary

Our solar power forecasting model for Germany has been classified as RED due to significant reliability issues that require immediate attention. While the model correctly captures about 90% of actual solar generation within its predicted ranges (which meets our target), it suffers from two critical problems: the forecast probabilities are not properly distributed, and consecutive forecasts show too much similarity to each other rather than reflecting genuine uncertainty. Think of it like a weather forecaster who gets the right temperature range most days, but whose confidence levels are systematically wrong and who repeats yesterday's forecast too often. This means we cannot trust the model's risk assessments for trading decisions, requiring us to increase our capital reserves and suspend automated trading strategies until the model is fixed and revalidated.

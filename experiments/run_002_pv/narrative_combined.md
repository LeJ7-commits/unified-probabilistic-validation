# Governance Narrative: pv_solar

**Classification:** RED

---

## Technical Summary

The PV solar forecasting model receives a RED classification due to fundamental violations in probability calibration, specifically failing both uniformity and independence requirements in the probability integral transform (PIT) diagnostics. The Kolmogorov-Smirnov test yields a statistic of 0.103 with p-value < 0.001, indicating systematic departure from uniform distribution, while the Ljung-Box tests demonstrate severe temporal dependence with statistics exceeding 3,986 across all lag structures (5, 10, 20) and p-values effectively zero. The autocorrelation function reveals persistent dependence with lag-1 correlation of 0.66, fundamentally violating the independence assumption required for valid probabilistic forecasts. Despite adequate Anfuso traffic-light performance (all GREEN) and reasonable coverage error of 1.37%, these PIT failures indicate the model's probability distributions are systematically miscalibrated, requiring immediate model recalibration before deployment in any Basel-regulated capital calculations or REMIT compliance reporting.

---

## Plain Language Summary

The solar power forecasting model has failed critical reliability tests and cannot be trusted for business decisions. Think of it like a weather forecast that consistently gets the chance of rain wrong - even if it sometimes predicts sunny days correctly, you can't rely on its confidence levels when making important plans. While the model's basic accuracy appears reasonable (hitting targets about 91% of the time), the underlying probability calculations are fundamentally flawed, showing both systematic bias and patterns that shouldn't exist in a properly working model. This means the model must be taken offline immediately and rebuilt before it can be used for regulatory reporting, risk management, or any business decisions that depend on understanding the uncertainty in solar generation forecasts.

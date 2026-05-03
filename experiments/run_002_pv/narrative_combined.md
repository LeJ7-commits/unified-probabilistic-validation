# Governance Narrative: pv_solar

**Classification:** RED

---

## Technical Summary

The PV solar forecasting model has been classified as RED due to critical failures in two fundamental validation branches. The Probability Integral Transform (PIT) uniformity tests show severe departures from expected uniform distribution with Kolmogorov-Smirnov p-value of 0.0 and Cramér-von Mises statistic of 14.48, indicating systematic forecast bias. Additionally, the autocorrelation function reveals persistent temporal dependence with lag-1 correlation of 0.66 and Ljung-Box statistics exceeding 3986 at all tested lags (p-values < 0.001), violating the independence assumption critical for probabilistic forecasts. While Anfuso backtesting shows GREEN status with empirical coverage of 91.37% (close to the 90% target), the fundamental distributional failures trigger mandatory model recalibration and potential capital multiplier penalties under regulatory frameworks. The model requires immediate remediation before continued deployment for trading book positions or REMIT reporting obligations.

---

## Plain Language Summary

Our solar power forecasting model has failed its quality checks and received a RED rating, meaning it cannot be trusted for important business decisions. Think of it like a weather forecast that consistently gets the temperature wrong in a predictable pattern - even if it sometimes hits the right range, the underlying predictions are fundamentally flawed. The model is showing two main problems: it's producing biased forecasts that don't follow the expected statistical patterns, and its predictions are connected to each other in ways that suggest it's missing important information about how solar generation actually behaves. While the model's coverage of actual outcomes looks reasonable at 91% (close to our 90% target), these deeper issues mean we must stop using it immediately and fix the underlying problems before we can rely on it again for trading decisions or regulatory reporting.

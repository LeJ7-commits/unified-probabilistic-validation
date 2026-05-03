# Technical Governance Narrative

**Model:** pv_solar  
**Classification:** RED  
**API generated:** True

---

The PV solar forecasting model has been classified as RED due to critical failures in two fundamental validation branches. The Probability Integral Transform (PIT) uniformity tests show severe departures from expected uniform distribution with Kolmogorov-Smirnov p-value of 0.0 and Cramér-von Mises statistic of 14.48, indicating systematic forecast bias. Additionally, the autocorrelation function reveals persistent temporal dependence with lag-1 correlation of 0.66 and Ljung-Box statistics exceeding 3986 at all tested lags (p-values < 0.001), violating the independence assumption critical for probabilistic forecasts. While Anfuso backtesting shows GREEN status with empirical coverage of 91.37% (close to the 90% target), the fundamental distributional failures trigger mandatory model recalibration and potential capital multiplier penalties under regulatory frameworks. The model requires immediate remediation before continued deployment for trading book positions or REMIT reporting obligations.

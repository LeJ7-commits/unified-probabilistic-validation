# Technical Governance Narrative

**Model:** entsoe_solar_de  
**Classification:** RED  
**API generated:** True

---

The ENTSO-E German solar PV forecast model has been classified RED due to critical failures in probabilistic forecast calibration. The probability integral transform (PIT) diagnostics show severe uniformity violations with Kolmogorov-Smirnov p-value of 0.0 and Cramér-von Mises statistic of 7.45, indicating systematic forecast bias. Most critically, the autocorrelation function reveals persistent temporal dependence with lag-1 correlation of 0.78 and Ljung-Box statistics exceeding 64,000 across all tested lags (p-values = 0.0), violating the independence assumption fundamental to probabilistic model validity. While Anfuso backtesting shows GREEN across all traffic lights with 89.9% empirical coverage close to the 90% target, the PIT failures trigger immediate governance action requiring model recalibration before continued use for REMIT reporting or capital allocation purposes.

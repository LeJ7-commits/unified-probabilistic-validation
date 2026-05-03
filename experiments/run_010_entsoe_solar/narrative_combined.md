# Governance Narrative: entsoe_solar_de

**Classification:** RED

---

## Technical Summary

The ENTSO-E German solar PV forecast model has been classified RED due to critical failures in probabilistic forecast calibration. The probability integral transform (PIT) diagnostics show severe uniformity violations with Kolmogorov-Smirnov p-value of 0.0 and Cramér-von Mises statistic of 7.45, indicating systematic forecast bias. Most critically, the autocorrelation function reveals persistent temporal dependence with lag-1 correlation of 0.78 and Ljung-Box statistics exceeding 64,000 across all tested lags (p-values = 0.0), violating the independence assumption fundamental to probabilistic model validity. While Anfuso backtesting shows GREEN across all traffic lights with 89.9% empirical coverage close to the 90% target, the PIT failures trigger immediate governance action requiring model recalibration before continued use for REMIT reporting or capital allocation purposes.

---

## Plain Language Summary

The solar power forecasting model for Germany has been flagged as requiring immediate attention due to significant reliability issues. While the model's prediction intervals are working well - capturing the actual solar output about 90% of the time as intended - there are serious problems with how the forecasts behave over time. Think of it like a weather forecaster who gets the right amount of sunny and rainy days but always predicts rain after rain and sun after sun, showing they don't understand the underlying patterns. The model's errors are too predictable and connected to each other, which means it's missing important information about how solar generation actually changes day to day. This model cannot be used for regulatory reporting or risk management decisions until these fundamental issues are fixed through recalibration.

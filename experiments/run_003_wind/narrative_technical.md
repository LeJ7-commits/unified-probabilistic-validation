# Technical Governance Narrative

**Model:** wind_onshore  
**Classification:** RED  
**API generated:** True

---

The onshore wind generation forecast model has been classified RED due to systematic failures in two critical probabilistic validation branches. The PIT uniformity diagnostics show severe departures from the required uniform distribution (KS statistic 0.106, p-value < 0.001; CvM statistic 21.28, p-value < 0.001), indicating fundamental miscalibration in the probability forecasts. Additionally, the ACF dependence tests reveal strong serial correlation in the probability integral transforms (lag-1 autocorrelation 0.855), with Ljung-Box statistics rejecting independence at all tested lags (5, 10, 20) with p-values effectively zero. The Anfuso traffic-light system confirms RED status despite marginal coverage performance (88.6% vs 90% target), with the total breach rate of 11.4% exceeding regulatory thresholds. Under Basel framework guidelines, this RED classification triggers enhanced capital multipliers and mandates immediate model recalibration before continued deployment for REMIT reporting obligations.

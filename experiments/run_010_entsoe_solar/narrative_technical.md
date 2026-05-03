# Technical Governance Narrative

**Model:** entsoe_solar_de  
**Classification:** RED  
**API generated:** True

---

The ENTSO-E solar PV forecast model for Germany exhibits critical probabilistic failures warranting RED classification and immediate remedial action. The PIT uniformity diagnostics show severe deviation from expected uniform distribution (KS p-value = 0.0, CvM statistic = 7.45), while the ACF dependence tests reveal strong serial correlation at lag-1 (ρ = 0.788) with highly significant Ljung-Box statistics across all tested lags (p-values = 0.0). Despite the Anfuso traffic-light system showing GREEN status with acceptable breach rates (10.06% total vs 10% target), the fundamental probabilistic calibration failures indicate the forecast intervals lack proper statistical foundation. Under Basel energy trading book requirements, this RED classification mandates a capital multiplier penalty and triggers REMIT model performance reporting obligations until recalibration restores statistical validity.

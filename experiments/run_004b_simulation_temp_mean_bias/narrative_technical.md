# Technical Governance Narrative

**Model:** simulation_temp_mean_bias  
**Classification:** RED  
**API generated:** True

---

The Monte Carlo temperature simulation model has been classified as RED due to systematic undercoverage in the probabilistic forecasts. The empirical coverage rate of 70.4% falls significantly short of the 90% target, representing a coverage error of -19.6 percentage points. Anfuso traffic-light analysis reveals the undercoverage is driven entirely by excessive upper tail breaches (29.3% vs expected ~5%), while lower tail performance remains acceptable (0.27% breach rate). The mean pinball loss of 0.505 and total breach rate of 29.6% confirm poor calibration, requiring immediate model recalibration and potential capital multiplier increases under Basel framework. REMIT reporting obligations may apply given the systematic bias in temperature risk quantification affecting energy derivative valuations.

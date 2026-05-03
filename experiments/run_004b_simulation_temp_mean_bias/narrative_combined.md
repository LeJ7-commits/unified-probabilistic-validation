# Governance Narrative: simulation_temp_mean_bias

**Classification:** RED

---

## Technical Summary

The Monte Carlo temperature simulation model has been classified as RED due to systematic undercoverage in the probabilistic forecasts. The empirical coverage rate of 70.4% falls significantly short of the 90% target, representing a coverage error of -19.6 percentage points. Anfuso traffic-light analysis reveals the undercoverage is driven entirely by excessive upper tail breaches (29.3% vs expected ~5%), while lower tail performance remains acceptable (0.27% breach rate). The mean pinball loss of 0.505 and total breach rate of 29.6% confirm poor calibration, requiring immediate model recalibration and potential capital multiplier increases under Basel framework. REMIT reporting obligations may apply given the systematic bias in temperature risk quantification affecting energy derivative valuations.

---

## Plain Language Summary

The temperature forecasting model is performing poorly and has been given a "red light" status. Think of it like a weather app that consistently underestimates how hot it will get - it's getting the extreme high temperatures wrong about 30% of the time when it should only miss 10% of the time. This means the model is not properly capturing the risk of very high temperatures, which could lead to unexpected losses in energy trading positions. The model must be fixed before it can be used for important business decisions, and additional capital may need to be held as a safety buffer until the problems are resolved.

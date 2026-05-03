# Governance Narrative: simulation_temp_mean_bias

**Classification:** RED

---

## Technical Summary

The Monte Carlo temperature simulation model exhibits severe undercoverage with empirical coverage of 70.4% against the 90% target, representing a coverage error of -19.6 percentage points. The Anfuso traffic-light system flags RED overall due to excessive upper tail breaches at 29.3% (vastly exceeding typical 5% thresholds), while lower tail performance remains GREEN at 0.27%. This asymmetric failure pattern indicates systematic mean bias in the simulation, likely understating temperature extremes in the upper distribution. The RED classification triggers immediate model suspension under Basel framework protocols, requiring capital multiplier escalation and mandatory REMIT reporting of the validation failure to energy regulators.

---

## Plain Language Summary

The temperature forecasting model is performing poorly and has been classified as unreliable for business use. Think of it like a weather forecast that consistently underestimates how hot it will get - the model correctly predicts cooler temperatures but fails badly when temperatures spike higher. Out of every 100 forecasts, the model should capture the actual temperature 90 times, but it's only succeeding 70 times, missing 30 critical high-temperature events. This creates significant risk for energy trading and planning decisions, so the model must be immediately withdrawn from use until the underlying bias is corrected and revalidated.

# Governance Narrative: simulation_price_heavy_tails

**Classification:** GREEN

---

## Technical Summary

The simulation_price_heavy_tails model achieves GREEN classification under normal regime conditions, demonstrating robust probabilistic calibration despite the known heavy-tails misspecification. Empirical coverage of 90.137% closely matches the 90% policy target with minimal coverage error of 0.137%, while the total breach rate of 9.863% remains within acceptable bounds. The Anfuso traffic-light system returns GREEN across all quantile levels (lower, upper, and total), indicating no systematic bias in tail risk estimation that would trigger capital multiplier penalties under Basel frameworks. Mean pinball loss of 0.593 reflects reasonable forecast sharpness, though the "wide" sharpness classification suggests potential for interval tightening without compromising coverage reliability. No REMIT reporting obligations are triggered, and the model maintains regulatory compliance for continued deployment in energy price forecasting applications.

---

## Plain Language Summary

The price forecasting model is performing well and meets all required standards for accuracy and reliability. Like a weather forecast that correctly predicts rain 90% of the time when it says there's a 90% chance, our model's predictions are appropriately calibrated - it captures actual price movements within its confidence intervals at the expected rate. While the prediction ranges are somewhat wider than ideal (meaning less precise forecasts), this conservative approach ensures we're not underestimating potential price swings, which is crucial for risk management. The model has passed all regulatory tests and can continue to be used for business decisions without any additional oversight or capital requirements.

# Governance Narrative: simulation_price_mean_bias

**Classification:** RED

---

## Technical Summary

The simulation price mean bias model has been classified as RED due to severe undercoverage in the upper tail, with empirical coverage of 70.4% falling significantly short of the 90% policy target (coverage error of -19.6%). The Anfuso traffic light system confirms this assessment with an overall RED classification, driven specifically by upper tail failures (upper breach rate of 29.0% versus acceptable lower breach rate of 0.5%). The mean pinball loss of 0.87 indicates poor probabilistic calibration, while interval sharpness remains acceptable at 16.4 mean width. This RED classification triggers immediate model remediation requirements and potential regulatory capital multiplier penalties under Basel frameworks, with REMIT reporting obligations for systematic forecast bias in wholesale energy markets.

---

## Plain Language Summary

This price forecasting model is significantly underperforming and poses material risk to trading operations. Think of it like a weather forecaster who consistently underestimates storm severity - the model is failing to capture 30% of actual price spikes that should fall within its predicted ranges. While the model correctly identifies when prices might fall below expectations, it systematically underestimates how high prices can go, leaving the business exposed to unexpected losses. The model must be immediately withdrawn from production use and rebuilt before it can support trading decisions, as continuing to rely on it could result in substantial financial losses and regulatory penalties.

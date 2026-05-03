# Technical Governance Narrative

**Model:** simulation_price_mean_bias  
**Classification:** RED  
**API generated:** True

---

The simulation price mean bias model has been classified as RED due to severe undercoverage in the upper tail, with empirical coverage of 70.4% falling significantly short of the 90% policy target (coverage error of -19.6%). The Anfuso traffic light system confirms this assessment with an overall RED classification, driven specifically by upper tail failures (upper breach rate of 29.0% versus acceptable lower breach rate of 0.5%). The mean pinball loss of 0.87 indicates poor probabilistic calibration, while interval sharpness remains acceptable at 16.4 mean width. This RED classification triggers immediate model remediation requirements and potential regulatory capital multiplier penalties under Basel frameworks, with REMIT reporting obligations for systematic forecast bias in wholesale energy markets.

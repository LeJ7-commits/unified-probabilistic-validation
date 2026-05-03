# Technical Governance Narrative

**Model:** simulation_price_mean_bias  
**Classification:** RED  
**API generated:** True

---

The Monte Carlo price simulation model has been classified RED due to systematic undercoverage in probabilistic forecasting performance. The model achieves only 70.4% empirical coverage against the 90% target, representing a coverage error of -19.6 percentage points that triggers the undercoverage reason code. Anfuso traffic-light analysis confirms this failure with an upper-tail RED classification (29.0% breach rate) while the lower tail performs adequately (GREEN, 0.5% breach rate), indicating systematic bias toward underestimating price volatility on the upside. The mean pinball loss of 0.867 and total breach rate of 29.6% substantially exceed acceptable thresholds for regime_normal conditions. This RED classification mandates immediate model recalibration, triggers enhanced capital allocation under Basel framework requirements, and may necessitate REMIT reporting disclosures for market risk exposure calculations.

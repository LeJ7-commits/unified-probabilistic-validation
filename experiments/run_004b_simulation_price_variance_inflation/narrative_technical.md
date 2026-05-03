# Technical Governance Narrative

**Model:** simulation_price_variance_inflation  
**Classification:** RED  
**API generated:** True

---

The simulation_price_variance_inflation model has been classified RED due to severe undercoverage, with empirical coverage at 55.34% against the 90% target—a deficit of 34.66 percentage points. The Anfuso traffic-light assessment confirms RED status across all dimensions (total, lower, and upper tails), with breach rates of 44.66% total (22.74% lower, 21.92% upper) indicating systematic underestimation of price volatility. The coverage error of -0.347 significantly exceeds typical Basel tolerance thresholds, while the mean pinball loss of 1.73 reflects poor probabilistic calibration. Under current governance frameworks, this RED classification mandates immediate model withdrawal from production use, triggers capital multiplier penalties, and requires enhanced REMIT reporting disclosures for any positions valued using this simulation approach.

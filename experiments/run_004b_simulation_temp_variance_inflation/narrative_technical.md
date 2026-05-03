# Technical Governance Narrative

**Model:** simulation_temp_variance_inflation  
**Classification:** RED  
**API generated:** True

---

The Monte Carlo temperature simulation model exhibits severe probabilistic miscalibration, achieving only 62.47% empirical coverage against the 90% policy target, resulting in a substantial coverage error of -27.53%. The Anfuso traffic-light system registers RED across all branches (lower, upper, total), with total breach rates of 37.53% significantly exceeding regulatory thresholds. The mean pinball loss of 0.79 confirms poor distributional accuracy, while the variance inflation misspecification appears to create systematic undercoverage in both tails. This RED classification mandates immediate model suspension under Basel governance protocols, requiring capital multiplier adjustments and potential REMIT reporting obligations given the energy market context.

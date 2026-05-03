# Technical Governance Narrative

**Model:** simulation_price_variance_inflation  
**Classification:** RED  
**API generated:** True

---

The simulation_price_variance_inflation model has been classified RED due to severe undercoverage, with empirical coverage of 55.34% against the required 90% target, representing a coverage error of -34.66 percentage points. Both lower and upper tail Anfuso traffic lights are RED, indicating systematic breach rates of 22.74% and 21.92% respectively, well above acceptable thresholds. The total breach rate of 44.66% demonstrates fundamental misspecification in the variance inflation mechanism, likely resulting in overconfident interval predictions. Under Basel backtesting frameworks, this RED classification would trigger immediate model restriction with potential capital multipliers of 3.4-4.0x, and the model must be withdrawn from REMIT reporting obligations until remediation is complete.

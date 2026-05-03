# Governance Narrative: simulation_temp_variance_inflation

**Classification:** RED

---

## Technical Summary

The Monte Carlo temperature simulation model exhibits severe probabilistic miscalibration, achieving only 62.47% empirical coverage against the 90% policy target, resulting in a substantial coverage error of -27.53%. The Anfuso traffic-light system registers RED across all branches (lower, upper, total), with total breach rates of 37.53% significantly exceeding regulatory thresholds. The mean pinball loss of 0.79 confirms poor distributional accuracy, while the variance inflation misspecification appears to create systematic undercoverage in both tails. This RED classification mandates immediate model suspension under Basel governance protocols, requiring capital multiplier adjustments and potential REMIT reporting obligations given the energy market context.

---

## Plain Language Summary

This temperature forecasting model is performing poorly and cannot be trusted for business decisions. Instead of correctly predicting temperature ranges 90% of the time as required, it's only getting it right 62% of the time - like a weather forecast that's wrong 4 days out of every 10. The model is consistently underestimating how variable temperatures can be, leaving the business exposed to unexpected temperature swings that could impact energy demand and pricing. We must immediately stop using this model and switch to our backup systems while the technical team fixes the underlying problems.

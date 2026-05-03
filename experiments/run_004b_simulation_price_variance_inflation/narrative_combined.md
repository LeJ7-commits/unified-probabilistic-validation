# Governance Narrative: simulation_price_variance_inflation

**Classification:** RED

---

## Technical Summary

The simulation_price_variance_inflation model has been classified RED due to severe undercoverage, with empirical coverage of 55.34% against the required 90% target, representing a coverage error of -34.66 percentage points. Both lower and upper tail Anfuso traffic lights are RED, indicating systematic breach rates of 22.74% and 21.92% respectively, well above acceptable thresholds. The total breach rate of 44.66% demonstrates fundamental misspecification in the variance inflation mechanism, likely resulting in overconfident interval predictions. Under Basel backtesting frameworks, this RED classification would trigger immediate model restriction with potential capital multipliers of 3.4-4.0x, and the model must be withdrawn from REMIT reporting obligations until remediation is complete.

---

## Plain Language Summary

The price simulation model is performing very poorly - imagine asking it to predict a range where electricity prices should fall 9 times out of 10, but it's only getting it right about 5-6 times out of 10. This means the model is being far too confident in its predictions and consistently underestimating how much prices might move up or down. The model cannot be used for any important business decisions, regulatory reporting, or risk management until it's completely rebuilt and fixed. This represents a significant operational risk as we may be underestimating our exposure to price volatility, and we need to immediately switch to backup models or manual processes for critical functions.

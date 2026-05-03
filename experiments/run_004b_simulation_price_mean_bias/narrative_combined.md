# Governance Narrative: simulation_price_mean_bias

**Classification:** RED

---

## Technical Summary

The Monte Carlo price simulation model has been classified RED due to systematic undercoverage in probabilistic forecasting performance. The model achieves only 70.4% empirical coverage against the 90% target, representing a coverage error of -19.6 percentage points that triggers the undercoverage reason code. Anfuso traffic-light analysis confirms this failure with an upper-tail RED classification (29.0% breach rate) while the lower tail performs adequately (GREEN, 0.5% breach rate), indicating systematic bias toward underestimating price volatility on the upside. The mean pinball loss of 0.867 and total breach rate of 29.6% substantially exceed acceptable thresholds for regime_normal conditions. This RED classification mandates immediate model recalibration, triggers enhanced capital allocation under Basel framework requirements, and may necessitate REMIT reporting disclosures for market risk exposure calculations.

---

## Plain Language Summary

Our price forecasting model is significantly underperforming and has been flagged as high-risk. Think of it like a weather forecast that claims there's a 90% chance of staying dry, but you actually get soaked 3 out of every 10 times you rely on it - that's essentially what's happening with our price predictions. The model is consistently underestimating how much prices might spike upward, which means we're not properly preparing for volatile market conditions that could cost us money. We need to stop using this model immediately and fix it before it can be trusted again, as continuing to rely on it could expose us to unexpected financial losses and regulatory scrutiny. The good news is that the model handles downward price movements reasonably well, so the problem is specific and should be fixable with proper recalibration.

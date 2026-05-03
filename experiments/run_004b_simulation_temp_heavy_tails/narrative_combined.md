# Governance Narrative: simulation_temp_heavy_tails

**Classification:** YELLOW

---

## Technical Summary

The simulation_temp_heavy_tails model receives a YELLOW classification under normal market regime conditions, indicating moderate validation concerns requiring enhanced monitoring. The model demonstrates acceptable calibration with empirical coverage of 92.33% against the 90% target, yielding a coverage error of +2.33 percentage points within acceptable bounds. Anfuso traffic-light diagnostics show GREEN across all tails, confirming no systematic directional bias in breach patterns, with total breach rate of 7.67% appropriately distributed between lower (2.74%) and upper (4.93%) tails. The mean pinball loss of 0.348 and interval width of 9.87 indicate reasonable forecast accuracy, though the YELLOW designation mandates increased model oversight and potential recalibration review given the heavy-tails misspecification in the temperature simulation framework. This classification requires enhanced documentation for regulatory reporting but does not trigger immediate capital multiplier adjustments or REMIT filing obligations.

---

## Plain Language Summary

The temperature forecasting model is performing reasonably well but needs closer attention - think of it like a car that's running fine but showing a yellow warning light on the dashboard. The model is correctly predicting extreme temperature events about 92% of the time, which is actually better than our minimum requirement of 90%, so it's not missing too many important weather patterns. However, because this model uses a simplified approach that doesn't fully capture how extreme temperature swings actually behave, we've flagged it for extra monitoring. The good news is that energy trading operations can continue as normal, but our risk team needs to watch this model more closely and may need to fine-tune it in the coming months to ensure it keeps performing well for business planning and regulatory reporting.

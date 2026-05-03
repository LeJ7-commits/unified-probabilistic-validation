# Technical Governance Narrative

**Model:** simulation_temp_mean_bias  
**Classification:** RED  
**API generated:** True

---

The Monte Carlo temperature simulation model exhibits severe undercoverage with empirical coverage of 70.4% against the 90% target, representing a coverage error of -19.6 percentage points. The Anfuso traffic-light system flags RED overall due to excessive upper tail breaches at 29.3% (vastly exceeding typical 5% thresholds), while lower tail performance remains GREEN at 0.27%. This asymmetric failure pattern indicates systematic mean bias in the simulation, likely understating temperature extremes in the upper distribution. The RED classification triggers immediate model suspension under Basel framework protocols, requiring capital multiplier escalation and mandatory REMIT reporting of the validation failure to energy regulators.

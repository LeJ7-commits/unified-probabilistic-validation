# Technical Governance Narrative

**Model:** simulation_temp_heavy_tails  
**Classification:** YELLOW  
**API generated:** True

---

The simulation_temp_heavy_tails model receives a YELLOW classification under normal market regime conditions, indicating moderate concerns with probabilistic accuracy. The model exhibits empirical coverage of 92.3% against a 90% target, representing a positive coverage error of +2.3 percentage points that exceeds typical tolerance bands. While Anfuso traffic-light diagnostics remain GREEN across all quantile regions, the total breach rate of 7.7% (with asymmetric distribution: 2.7% lower, 4.9% upper) suggests systematic bias in the upper tail predictions. The mean pinball loss of 0.348 and mean interval width of 9.87 units indicate acceptable sharpness characteristics, but the YELLOW status triggers enhanced monitoring requirements and may necessitate recalibration before the next validation cycle. Under Basel frameworks, this classification maintains current capital multipliers but requires documented remediation plans within the prescribed timeline.

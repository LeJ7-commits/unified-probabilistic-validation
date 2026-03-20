# Governance Narrative — Sim Temp (well-spec)
Window size: 250  |  Step: 250  |  α = 0.1  |  Coverage target: 90%

## Overall Stability
- Stationary entropy: 0.650 bits (max = 1.585 bits)
- Stability classification: **MODERATE**
- Interpretation: Classification shows moderate variation across states. Possible causes: mild regime shifts, seasonal effects on calibration quality, or borderline diagnostic thresholds.

## State Distribution
- GREEN: 14.3% of windows (1 windows)
- YELLOW: 85.7% of windows (6 windows)
- RED: 0.0% of windows (0 windows)

## Label Transitions (2 detected)
- Window 1 (start: 50): YELLOW → GREEN
  Reason: All monitored diagnostic signals within policy thresholds.
- Window 2 (start: 100): GREEN → YELLOW
  Reason: Coverage mildly off-target by 0.040 (target=0.900).
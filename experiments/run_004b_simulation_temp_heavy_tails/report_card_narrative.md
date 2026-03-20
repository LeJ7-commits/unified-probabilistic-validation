# Governance Narrative — Sim Temp — Heavy Tails
Window size: 250  |  Step: 250  |  α = 0.1  |  Coverage target: 90%

## Overall Stability
- Stationary entropy: 1.371 bits (max = 1.585 bits)
- Stability classification: **UNSTABLE**
- Interpretation: Classification is highly unstable. Possible causes: (1) genuine regime shifts in the underlying process, (2) rolling window too small relative to process memory, (3) diagnostic thresholds near a decision boundary.

## State Distribution
- GREEN: 14.3% of windows (1 windows)
- YELLOW: 57.1% of windows (4 windows)
- RED: 28.6% of windows (2 windows)

## Label Transitions (5 detected)
- Window 1 (start: 50): RED → YELLOW
  Reason: Coverage mildly off-target by 0.020 (target=0.900).
- Window 3 (start: 150): YELLOW → RED
  Reason: Coverage off-target by 0.060 (target=0.900).
- Window 4 (start: 200): RED → YELLOW
  Reason: Coverage mildly off-target by 0.020 (target=0.900).
- Window 5 (start: 250): YELLOW → GREEN
  Reason: All monitored diagnostic signals within policy thresholds.
- Window 6 (start: 300): GREEN → YELLOW
  Reason: Coverage mildly off-target by 0.040 (target=0.900).
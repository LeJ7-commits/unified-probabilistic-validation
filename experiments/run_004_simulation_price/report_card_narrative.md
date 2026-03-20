# Governance Narrative — Sim Price (well-spec)
Window size: 250  |  Step: 250  |  α = 0.1  |  Coverage target: 90%

## Overall Stability
- Stationary entropy: 1.571 bits (max = 1.585 bits)
- Stability classification: **UNSTABLE**
- Interpretation: Classification is highly unstable. Possible causes: (1) genuine regime shifts in the underlying process, (2) rolling window too small relative to process memory, (3) diagnostic thresholds near a decision boundary.

## State Distribution
- GREEN: 28.6% of windows (2 windows)
- YELLOW: 42.9% of windows (3 windows)
- RED: 28.6% of windows (2 windows)

## Label Transitions (5 detected)
- Window 2 (start: 100): YELLOW → RED
  Reason: Coverage off-target by 0.140 (target=0.900).
- Window 3 (start: 150): RED → GREEN
  Reason: All monitored diagnostic signals within policy thresholds.
- Window 4 (start: 200): GREEN → YELLOW
  Reason: Coverage mildly off-target by 0.040 (target=0.900).
- Window 5 (start: 250): YELLOW → GREEN
  Reason: All monitored diagnostic signals within policy thresholds.
- Window 6 (start: 300): GREEN → RED
  Reason: Coverage off-target by 0.060 (target=0.900).
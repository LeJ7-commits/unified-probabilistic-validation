# Results

## 1. Full-Sample Diagnostics (n = 210,227, α = 0.1)

### 1.1 Interval Backtesting (Anfuso Traffic-Light Framework)

For α = 0.1 (nominal 90% coverage), the full-sample breach statistics are:

- Total observations: 210,227
- Lower breaches: 8,561  (4.07%)
- Upper breaches: 13,033 (6.20%)
- Total breaches: 21,594 (10.27%)

Nominal expected breach rate: 10%.

#### Traffic-Light Classification

| Component | Result |
|------------|--------|
| Lower tail | GREEN |
| Upper tail | RED |
| Total      | RED |

Binomial exceedance p-values:

- Lower tail: p = 1.0 (no underestimation of risk)
- Upper tail: p ≈ 2.21 × 10⁻¹³¹ (extreme undercoverage)
- Total: p ≈ 1.78 × 10⁻⁵

##### Interpretation

- The lower tail is conservative (actual 4.07% < 5% expected).
- The upper tail exhibits statistically significant excess breaches.
- Total coverage (10.27%) significantly exceeds the nominal 10%.

This indicates asymmetric miscalibration concentrated in the upper tail.

---

### 1.2 Distributional Diagnostics (PIT-Based)

#### Uniformity Tests

- KS statistic: 0.0678 (p ≈ 0)
- Cramér–von Mises: 437.17 (p ≈ 1.32 × 10⁻⁷)
- Anderson–Darling: 1271.99 (>> critical values)

All uniformity tests strongly reject the null hypothesis of U(0,1).

#### Independence Tests (Ljung–Box)

Applied to transformed residuals \( z = \Phi^{-1}(u) \):

| Lag | Statistic | p-value |
|------|-----------|----------|
| 5    | 888,273   | 0 |
| 10   | 1,660,715 | 0 |
| 20   | 2,914,607 | 0 |

Serial independence is overwhelmingly rejected.

---

### 1.3 Proper Scoring

- Mean CRPS: 1466.45
- Empirical coverage: 89.73%
- Coverage error: +0.27%

Although the average coverage error is numerically small, statistical testing reveals systematic structural deficiencies.

---

## 2. Rolling Evaluation

Rolling-window diagnostics were conducted under two schemes:

1. Non-overlapping windows
2. Overlapping windows

Across both specifications:

- PIT uniformity is frequently rejected.
- Independence violations persist across windows.
- Upper-tail breach dominance is recurrent.
- Test statistics remain highly significant in most subperiods.

Overlapping windows produce smoother trajectories but confirm the same structural deficiencies observed in the non-overlapping scheme.

This suggests the miscalibration is not localized to isolated regimes but rather persistent across time.

---

## 3. Synthesis of Findings

The combined evidence indicates:

1. **Distributional Miscalibration**  
   Strong rejection of PIT uniformity across all major tests.

2. **Dynamic Misspecification**  
   Severe serial dependence in transformed PIT residuals.

3. **Tail Asymmetry**  
   Conservative lower tail, under-covered upper tail.

4. **Governance Implication**  
   The model receives a RED classification under the diagnostic policy framework.

Importantly, although the raw coverage error appears modest, formal statistical backtesting detects systematic deviations from the assumed probabilistic structure.

This highlights the necessity of multi-layer probabilistic validation beyond naive coverage metrics.

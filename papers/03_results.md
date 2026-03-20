# Results

This chapter reports empirical results across all four model classes evaluated
in the unified probabilistic validation framework. Results are organised by
dataset: ENTSO-E short-term electricity load forecasting (run_001), long-term PV
generation (run_002), long-term wind generation (run_003), the synthetic
simulation positive control and misspecification scenarios (run_004,
run_004b), and a joint multivariate dependency analysis of PV and wind
(run_005). Each dataset is evaluated under the same diagnostic architecture:
Anfuso traffic-light interval backtesting, PIT-based distributional and
independence diagnostics, proper scoring via CRPS, and rolling-window
stability analysis.

---

## 1. ENTSO-E Short-Term Electricity Load Forecasting (run_001)

**Dataset:** Full ENTSO-E electricity load series, quarter-hourly resolution.
**Evaluable observations:** n = 209,555 (200 NaN observations removed).
**Nominal level:** α = 0.1 (90% central interval).
**Base interval construction:** Rolling empirical residual quantiles with
4-bucket time-of-day conditioning (night/morning/afternoon/evening),
Wb = 40 bucket-specific observations, Wg = 672 global observations (7 days).

---

### 1.1 Interval Backtesting (Anfuso Traffic-Light Framework)

| Component  | Breach Rate | p-value (exceedance) | Traffic Light |
|------------|-------------|----------------------|---------------|
| Lower tail | 6.66%       | ≈ 1.17 × 10⁻²⁴¹      | RED           |
| Upper tail | 6.28%       | ≈ 2.17 × 10⁻¹⁴⁸      | RED           |
| Total      | 12.94%      | ≈ 0 (machine ε)      | RED           |

Nominal expected breach rates: 5% per tail, 10% total.

Both tails breach significantly above their nominal rates. The rolling
empirical reconstruction (replacing the earlier global-quantile placeholder)
correctly captures the asymmetric residual structure; the resulting
intervals are systematically too narrow, producing bilateral over-breaching.
Total breach rate of 12.94% represents a 2.94 percentage point excess over
the nominal 10% target, confirmed as statistically extreme under the
binomial test.

---

### 1.2 Distributional Diagnostics (PIT-Based)

#### Uniformity Tests

| Test               | Statistic | p-value              |
|--------------------|-----------|----------------------|
| Kolmogorov–Smirnov | 0.1615    | ≈ 0                  |
| Cramér–von Mises   | 2,032.08  | ≈ 3.85 × 10⁻⁷        |
| Anderson–Darling   | 6,136.34  | >> critical values   |

All uniformity tests strongly reject the null hypothesis of U(0,1) PIT
scores at all conventional significance levels.

#### Independence Tests (Ljung–Box on z = Φ⁻¹(u))

| Lag | Statistic  | p-value |
|-----|------------|---------|
| 5   | 846,049    | ≈ 0     |
| 10  | 1,563,119  | ≈ 0     |
| 20  | 2,699,489  | ≈ 0     |

Serial independence is overwhelmingly rejected at all lags. The magnitude
of the statistics — several orders above chi-squared critical values —
indicates severe autocorrelation in the probability integral transforms,
consistent with the model failing to capture persistent temporal structure
in electricity load.

---

### 1.3 Proper Scoring

- Mean CRPS: 1,515.23
- Empirical coverage: 87.06%
- Coverage error: −2.94 pp

The negative coverage error confirms that the base predictive intervals are
too narrow on average. CRPS reflects the absolute forecast scale (electricity load
in physical units); cross-asset CRPS comparisons should account for scale
differences.

---

### 1.4 Governance Classification

| Criterion              | Status                  |
|------------------------|-------------------------|
| PIT uniformity         | FAIL                    |
| PIT independence       | FAIL                    |
| Interval coverage      | −2.94 pp (under)        |
| Overall classification | **RED**                 |

Risk reasons: uniformity strongly rejected (min p ≈ 0); independence
strongly rejected (min Ljung–Box p ≈ 0).

---

### 1.5 Rolling Evaluation

Rolling diagnostics were conducted under non-overlapping (window = 250,
step = 250) and overlapping (window = 250, step = 50) schemes. Across both
specifications, PIT uniformity is frequently rejected, independence
violations persist, and both tails breach above nominal rates recurrently.
The consistency across specifications indicates that miscalibration is not
localised to isolated regimes but is persistent across time.

---

## 2. Long-Term PV Generation (run_002)

**Dataset:** pv_student.csv, hourly, 2013–2015.
**Evaluable observations:** n = 4,287 (daytime only; 10,036 nighttime rows
excluded; 11,957 warmup rows consumed by rolling window).
**Nominal level:** α = 0.1 (90% central interval).
**Base interval construction:** Rolling empirical residual quantiles, 24-bucket
hour-of-day conditioning, W = 720 trailing same-hour observations (~30 days).
Structural nighttime zeros (both Simulation and Actuals < 10⁻⁹) excluded from
calibration evaluation — consistent with standard industry practice.

---

### 2.1 Interval Backtesting (Anfuso Traffic-Light Framework)

| Component  | Breach Rate | p-value (exceedance) | Traffic Light |
|------------|-------------|----------------------|---------------|
| Lower tail | 3.64%       | 0.9999921            | GREEN         |
| Upper tail | 4.99%       | 0.5196               | GREEN         |
| Total      | 8.63%       | 0.9989               | GREEN         |

Both tails are within nominal bounds. The lower tail is conservative
(3.64% < 5% expected). Total breach rate of 8.63% is below the 10%
nominal — the reconstructed intervals are slightly wide rather than too
narrow. All binomial p-values are far from rejection: the model passes
interval backtesting at the Anfuso level.

**Note on statistical power.** With n = 4,287, the binomial test has
substantially less power than the ENTSO-E run (n = 209,555). A modest
true over-breaching could go undetected. This limitation is inherent
to the available evaluation sample after nighttime exclusion and
rolling warmup.

---

### 2.2 Distributional Diagnostics (PIT-Based)

#### Uniformity Tests

| Test               | Statistic | p-value              |
|--------------------|-----------|----------------------|
| Kolmogorov–Smirnov | 0.1028    | ≈ 7.23 × 10⁻⁴⁰       |
| Cramér–von Mises   | 14.48     | ≈ 1.05 × 10⁻⁹        |
| Anderson–Darling   | 316.30    | >> critical values   |

PIT uniformity is strongly rejected despite the GREEN Anfuso result.

#### Independence Tests (Ljung–Box on z = Φ⁻¹(u))

| Lag | Statistic | p-value |
|-----|-----------|---------|
| 5   | 3,986     | ≈ 0     |
| 10  | 4,741     | ≈ 0     |
| 20  | 5,012     | ≈ 0     |

Serial independence is overwhelmingly rejected.

---

### 2.3 Proper Scoring

- Mean CRPS: 0.7937
- Empirical coverage: 91.37%
- Coverage error: +1.37 pp

CRPS is low relative to ENTSO-E because PV generation values are
normalised (capacity factors), not physical load units.

---

### 2.4 Governance Classification

| Criterion              | Status                  |
|------------------------|-------------------------|
| PIT uniformity         | FAIL                    |
| PIT independence       | FAIL                    |
| Interval coverage      | +1.37 pp (conservative) |
| Anfuso interval test   | GREEN                   |
| Overall classification | **RED**                 |

**Key finding.** PV passes interval backtesting (GREEN Anfuso) but fails
distributional diagnostics. This divergence is precisely the argument for
multi-layer validation: naive coverage metrics alone would classify PV as
acceptable, while PIT-based diagnostics reveal systematic structural
deficiencies in the shape of the predictive distribution.

---

### 2.5 Rolling Evaluation

Rolling diagnostics (window = 720, step = 168) confirm that PIT
uniformity and independence violations are not localised to specific
subperiods. The GREEN Anfuso result is also stable across rolling windows,
reinforcing the interpretation that interval width is approximately
correct on average but the distributional shape is systematically
misspecified.

---

## 3. Long-Term Wind Generation (run_003)

**Dataset:** wind_student.csv, hourly, 2013–2015.
**Evaluable observations:** n = 9,000 (no nighttime exclusion; 17,280
warmup rows consumed by rolling window).
**Nominal level:** α = 0.1 (90% central interval).
**Base interval construction:** Same rolling 24-bucket hour-of-day
conditioning as PV, W = 720.

---

### 3.1 Interval Backtesting (Anfuso Traffic-Light Framework)

| Component  | Breach Rate | p-value (exceedance) | Traffic Light |
|------------|-------------|----------------------|---------------|
| Lower tail | 6.44%       | ≈ 8.97 × 10⁻¹⁰       | RED           |
| Upper tail | 4.93%       | 0.6209               | GREEN         |
| Total      | 11.38%      | ≈ 1.02 × 10⁻⁵         | RED           |

Asymmetric tail failure: the lower tail breaches significantly above its
5% nominal rate while the upper tail is within tolerance. This is
physically interpretable — wind generation is bounded below by zero, and
the model underestimates low-generation risk more than high-generation
risk. The lower tail asymmetry mirrors the structure observed in the
original (uncorrected) ENTSO-E run before the lo/hi fix, but here it is
a genuine feature of the data rather than a reconstruction artefact.

---

### 3.2 Distributional Diagnostics (PIT-Based)

#### Uniformity Tests

| Test               | Statistic | p-value              |
|--------------------|-----------|----------------------|
| Kolmogorov–Smirnov | 0.1057    | ≈ 5.90 × 10⁻⁸⁸       |
| Cramér–von Mises   | 21.28     | ≈ 2.47 × 10⁻⁹        |
| Anderson–Darling   | 519.20    | >> critical values   |

PIT uniformity strongly rejected.

#### Independence Tests (Ljung–Box on z = Φ⁻¹(u))

| Lag | Statistic | p-value |
|-----|-----------|---------|
| 5   | 25,553    | ≈ 0     |
| 10  | 40,250    | ≈ 0     |
| 20  | 56,247    | ≈ 0     |

Serial independence overwhelmingly rejected.

---

### 3.3 Proper Scoring

- Mean CRPS: 1.7893
- Empirical coverage: 88.62%
- Coverage error: −1.38 pp

---

### 3.4 Governance Classification

| Criterion              | Status                  |
|------------------------|-------------------------|
| PIT uniformity         | FAIL                    |
| PIT independence       | FAIL                    |
| Interval coverage      | −1.38 pp (under)        |
| Lower tail             | RED                     |
| Upper tail             | GREEN                   |
| Overall classification | **RED**                 |

---

### 3.5 Rolling Evaluation

Rolling diagnostics confirm that the lower-tail excess breach is persistent
across subperiods rather than concentrated in any specific weather regime.
PIT independence violations are similarly stable, consistent with a
structural misspecification in the wind speed forecast or power curve
assumption rather than a localised regime failure.

---

## 4. Synthetic Simulation Model (run_004)

**Dataset:** Synthetic price and temperature series generated from a known
joint Gaussian DGP (correlated, ρ = 0.5) with intraday and seasonal
seasonality. y_hat computed as empirical quantiles of 5,000 simulation
paths. This constitutes a **positive control**: the model is correctly
specified by construction, so the framework should return GREEN.
**Evaluable observations:** n = 365 (one per as-of date).
**Nominal level:** α = 0.1.

---

### 4.1 Interval Backtesting (Anfuso Traffic-Light Framework)

| Series | Lower Breach | Upper Breach | Total Breach | Lower TL | Upper TL | Total TL |
|--------|-------------|-------------|-------------|----------|----------|----------|
| Price  | 6.58%       | 4.93%       | 11.51%      | GREEN    | GREEN    | GREEN    |
| Temp   | 6.30%       | 4.11%       | 10.41%      | GREEN    | GREEN    | GREEN    |

All breach rates within or close to nominal bounds. Binomial exceedance
p-values comfortably above conventional rejection thresholds (price total
p = 0.190; temp total p = 0.422). The positive control behaves as expected.

---

### 4.2 Governance Classification

| Criterion              | Price   | Temp    |
|------------------------|---------|---------|
| Interval coverage      | 88.49%  | 89.59%  |
| Coverage error         | −1.51pp | −0.41pp |
| Overall classification | GREEN   | GREEN   |

**Note on PIT diagnostics.** Full-sample PIT uniformity and independence
statistics (min_p_uniformity, min_p_ljungbox) are null for the simulation
runs because PIT computation requires full distributional samples; the
run_004 architecture passes only quantile bounds (lo, hi) rather than
the full 5,000-path sample array. PIT diagnostics are therefore not
available for this model class under the current pipeline. Extending to
full PIT evaluation would require persisting the per-horizon empirical
CDF from the simulation paths, which is deferred as a future extension.

---

## 5. Synthetic Simulation — Misspecification Scenarios (run_004b)

Three deliberate misspecification scenarios are evaluated against both the
price and temperature series to test the framework's discriminative
validity — its ability to detect and characterise known model failures.

**Scenario definitions:**

- **Variance inflation:** Realised values drawn with σ × 2; model intervals
  built from σ × 1 paths. Intervals are half as wide as needed.
- **Mean bias:** Realised values drawn with mean shifted by +1σ (price: +5.0,
  temp: +3.0); model uses unshifted mean. Systematic directional breach on
  one tail expected.
- **Heavy tails:** Realised values drawn from t(df=3); model uses Gaussian
  paths. Excess tail events expected if the framework is sensitive enough
  to detect them at n = 365.

---

### 5.1 Anfuso Results — All Scenarios

| Scenario            | Series | Lower   | Upper   | Total   | Lower TL | Upper TL | Total TL | Coverage |
|---------------------|--------|---------|---------|---------|----------|----------|----------|----------|
| Variance inflation  | Price  | 22.74%  | 21.92%  | 44.66%  | RED      | RED      | RED      | 55.3%    |
| Mean bias           | Price  | 0.55%   | 29.04%  | 29.59%  | GREEN    | RED      | RED      | ~70.4%   |
| Heavy tails         | Price  | 4.38%   | 5.48%   | 9.86%   | GREEN    | GREEN    | GREEN    | 90.1%    |
| Variance inflation  | Temp   | 19.18%  | 18.36%  | 37.53%  | RED      | RED      | RED      | 62.5%    |
| Mean bias           | Temp   | 0.27%   | 29.32%  | 29.59%  | GREEN    | RED      | RED      | ~70.4%   |
| Heavy tails         | Temp   | 2.74%   | 4.93%   | 7.67%   | GREEN    | GREEN    | GREEN    | 92.3%    |

---

### 5.2 Interpretation by Scenario

**Variance inflation** is detected strongly and symmetrically. Both tails
breach at approximately 4× the nominal rate (22–23% vs 5%), producing
total breach rates of 44.7% (price) and 37.5% (temp). The bilateral
symmetry is expected — inflating σ by 2× makes realisations equally
likely to exceed either tail. Coverage collapses to 55.3% and 62.5%
respectively. Both series receive RED with extreme p-values
(p < 10⁻²⁹ per tail).

**Mean bias** produces a clearly asymmetric signal. The upper tail
accumulates nearly all breaches (29.0%/29.3%) while the lower tail is
near-zero (0.55%/0.27%), consistent with a rightward shift in the
realised distribution pushing observations out of the upper prediction
bound. The lower tail is GREEN (realisations rarely fall below the
lower bound of an unshifted interval), while the upper tail is strongly
RED. This asymmetric pattern directly identifies the direction of the
bias and demonstrates that the traffic-light framework localises the
failure mode, not just its existence.

**Heavy tails** is not detected by the Anfuso interval test at n = 365.
Both price and temperature return GREEN, with near-nominal coverage
(90.1% and 92.3%). This is a genuine finding about the limits of the
framework's discriminative power at small sample sizes, not an error.
The t(df=3) distribution produces more extreme realisations than a
Gaussian, but at n = 365 these excess events are rare and roughly
symmetric across both tails, so the binomial interval test lacks
sufficient power to flag them. Detecting heavy-tail misspecification
reliably requires either a larger evaluation sample or tail-specific
diagnostics (e.g., extreme value tests or PIT histogram inspection in
the tails). The temperature series receives YELLOW (coverage error
+2.33pp), a marginal signal that is consistent with mild
over-conservatism from the heavier tails without reaching RED.

---

### 5.3 Governance Classification — Misspecification Scenarios

| Scenario           | Series | Classification | Primary Signal                        |
|--------------------|--------|----------------|---------------------------------------|
| Variance inflation | Price  | RED            | Bilateral symmetric over-breaching    |
| Variance inflation | Temp   | RED            | Bilateral symmetric over-breaching    |
| Mean bias          | Price  | RED            | Unilateral upper-tail excess          |
| Mean bias          | Temp   | RED            | Unilateral upper-tail excess          |
| Heavy tails        | Price  | GREEN          | Not detected (insufficient power)     |
| Heavy tails        | Temp   | YELLOW         | Marginal coverage excess (n=365)      |

The misspecification scenarios validate the framework's discriminative
capacity for the two largest failure modes (variance inflation and mean
bias) while exposing a power limitation for distributional shape
misspecification at small sample sizes.

---

## 6. Multivariate Dependency Analysis — PV and Wind (run_005)

**Motivation.** Univariate diagnostics evaluate each asset's calibration
independently. In practice, a portfolio of renewable assets is exposed to
shared weather drivers, creating cross-asset dependence in forecast errors.
run_005 evaluates joint PIT residual dependence between PV and wind on
their shared hourly evaluation index.

**Shared evaluation index:** n = 4,287 daytime hours (PV daytime restriction
applied to both series; wind warmup-eligible observations intersected with
PV daytime timestamps, 2013–2015).

---

### 6.1 Marginal PIT Residuals

| Asset | Mean z = Φ⁻¹(u) | Std z  |
|-------|-----------------|--------|
| PV    | +0.1037         | 1.9414 |
| Wind  | −0.0514         | 1.5997 |

Both series show std(z) > 1.0, consistent with overdispersion in the
marginal distributions (confirmed RED in univariate runs). PV shows a
slight positive bias (model slightly underestimates daytime generation);
wind shows a slight negative bias.

---

### 6.2 Multivariate Ljung–Box (Joint PIT Residual Vector)

| Lag | Statistic | df | p-value |
|-----|-----------|----|---------|
| 5   | 15,427    | 20 | ≈ 0     |
| 10  | 20,692    | 40 | ≈ 0     |
| 20  | 24,714    | 80 | ≈ 0     |

Joint independence is overwhelmingly rejected. The joint residual process
inherits the severe serial dependence of both individual series.

---

### 6.3 Cross-Correlation of PIT Residuals

| Lag | Correlation |
|-----|-------------|
| 0   | −0.040      |
| 1   | −0.041      |
| 6   | −0.031      |
| 24  | +0.075      |

Contemporaneous correlation is weakly negative (lag 0: −0.040), consistent
with the physical tendency for calm, sunny conditions to correlate with
lower wind and higher solar irradiance. The lag-24 value (+0.075) reflects
shared diurnal weather persistence — the same-hour correlation one day
ahead is positive, indicating that jointly favourable or unfavourable
weather conditions persist day-to-day. All cross-correlations are modest
in magnitude, indicating that the two marginal distributions are
approximately contemporaneously independent while not serially independent.

---

### 6.4 Bivariate Energy Score

Bivariate energy score: **2.017**

This serves as a reference value for the joint predictive distribution
under the current base intervals. A correctly calibrated joint predictive
distribution would achieve a lower score. The value provides a benchmark
for comparison after conformal augmentation or model improvement.

---

### 6.5 Independence Verdict

**REJECT** — the joint PIT residual process is not white noise.

The multivariate results confirm that treating PV and wind as independent
for portfolio risk aggregation would underestimate joint tail exposure.
Any composite risk measure (e.g., a combined renewable portfolio VaR) must
account for the persistent serial structure and the lag-24 positive
cross-correlation.

---

## 7. Cross-Dataset Synthesis

### 7.1 Summary Table

| Dataset     | n       | Coverage | Cov. Error | Anfuso TL | PIT Uniform | PIT Indep | Overall |
|-------------|---------|----------|------------|-----------|-------------|-----------|---------|
| ENTSO-E     | 209,555 | 87.06%   | −2.94 pp   | RED       | FAIL        | FAIL      | **RED** |
| PV          | 4,287   | 91.37%   | +1.37 pp   | GREEN     | FAIL        | FAIL      | **RED** |
| Wind        | 9,000   | 88.62%   | −1.38 pp   | RED (lower)| FAIL       | FAIL      | **RED** |
| Sim (price) | 365     | 88.49%   | −1.51 pp   | GREEN     | n/a         | n/a       | **GREEN** |
| Sim (temp)  | 365     | 89.59%   | −0.41 pp   | GREEN     | n/a         | n/a       | **GREEN** |

### 7.2 Key Cross-Dataset Findings

**1. PIT diagnostics fail universally on real data, regardless of coverage.**
All three real-data models (ENTSO-E, PV, wind) strongly reject PIT
uniformity and serial independence. This holds even for PV, which passes
interval backtesting. The implication is that interval-coverage metrics
alone are insufficient for governance classification — a model can appear
calibrated at the interval level while being systematically misspecified
at the distributional level.

**2. The positive control confirms framework discriminative validity.**
The well-specified simulation model returns GREEN across both series and
all interval diagnostics. The framework correctly distinguishes a
correctly specified model from miscalibrated ones.

**3. Misspecification scenarios demonstrate targeted failure mode detection.**
Variance inflation and mean bias are detected strongly and their
character (bilateral vs. unilateral) is correctly identified by the
traffic-light tail decomposition. Heavy-tail misspecification is not
detected at n = 365, exposing a power boundary of the Anfuso interval
test at small evaluation sample sizes.

**4. Tail asymmetry is dataset-specific and physically interpretable.**
ENTSO-E shows bilateral over-breaching; wind shows lower-tail dominance
(bounded-below generation); PV passes the interval test entirely. Each
pattern is physically motivated by the underlying generation or
forecasting process.

**5. Joint dependence between PV and wind is modest but present.**
Cross-asset PIT residuals show weak contemporaneous negative correlation
and meaningful lag-24 positive correlation. For portfolio risk
aggregation, ignoring this structure would be conservative in some
regimes and non-conservative in others.

**6. Conformal augmentation is motivated by the coverage shortfalls.**
ENTSO-E (−2.94 pp) and wind (−1.38 pp) both show systematic under-coverage
in the base reconstruction. The conformal augmentation layer (Section 5
of this chapter, and 04_conformal_wrapping.ipynb) addresses this by
expanding base intervals to restore near-nominal coverage guarantees,
as demonstrated on the ENTSO-E development sample.

---

## 8. Conformal Augmentation Results (Reference)

*Full conformal results are reported in 04_conformal_wrapping.ipynb.
Key finding: base interval + conformal expansion achieves 89.9% empirical
coverage against a 90% nominal target on the ENTSO-E development sample
(n = 2,544 test observations), with a 12% width increase over the base
interval — the best calibration–sharpness trade-off of all methods
evaluated. See the notebook for full method comparison at α = 0.1 and
α = 0.2.*

---

## 9. VaR Sensitivity Analysis (run_006)

Two complementary economic distortion analyses are computed across all
model classes and misspecification scenarios, quantifying the practical
consequences of miscalibration in terms of regulatory capital and
operational reserve sizing.

### 9.1 Capital Multiplier Distortion

The Basel Committee (1996) traffic-light framework maps exception counts
in a 250-day evaluation window to capital multiplier zones. Applying this
framework at 90% coverage (10% nominal breach rate) requires an adaptation:
a perfectly calibrated model already produces 250 × 0.10 = 25 exceptions,
far above Basel's raw RED threshold of 10. The adapted mapping therefore
operates on *excess exceptions* above the nominal expectation of 25, with
GREEN defined as ±2 excess, YELLOW as +3 to +8, and RED above +8.

| Dataset | Gov Label | Adapted Zone | Actual Exc | Excess | Multiplier | Distortion |
|---|---|---|---|---|---|---|
| ENTSO-E | RED | YELLOW | 32.3 | +7.3 | 3.40 | +13.3% |
| PV Solar | RED | CONSERVATIVE | 21.6 | −3.4 | 2.80 | −6.7% |
| Wind | RED | YELLOW | 28.4 | +3.4 | 3.40 | +13.3% |
| Sim Price (well-spec) | GREEN | YELLOW | 28.8 | +3.8 | 3.40 | +13.3% |
| Sim Temp (well-spec) | GREEN | GREEN | 26.0 | +1.0 | 3.00 | 0.0% |
| Sim Price — Var Inflation | RED | RED | 111.6 | +86.6 | 4.00 | +33.3% |
| Sim Price — Mean Bias | RED | RED | 74.0 | +49.0 | 4.00 | +33.3% |
| Sim Price — Heavy Tails | GREEN | GREEN | 24.7 | −0.3 | 3.00 | 0.0% |
| Sim Temp — Var Inflation | RED | RED | 93.8 | +68.8 | 4.00 | +33.3% |
| Sim Temp — Mean Bias | RED | RED | 74.0 | +49.0 | 4.00 | +33.3% |
| Sim Temp — Heavy Tails | YELLOW | CONSERVATIVE | 19.2 | −5.8 | 2.80 | −6.7% |

**Key finding — PV divergence.** PV Solar receives a governance RED
classification under the full diagnostic framework but maps to CONSERVATIVE
under the capital multiplier analysis (−6.7% distortion). This divergence
arises because PV's empirical coverage is 91.4% — above nominal — making
it appear over-conservative to a coverage-only regulatory framework. A
pure coverage-based regime would reduce capital requirements for PV, while
the full diagnostic framework correctly identifies systematic distributional
misspecification and serial dependence. This is the clearest empirical
demonstration that coverage metrics alone are insufficient for governance
classification.

**Sampling noise at small n.** The well-specified simulation price series
(governance GREEN) lands in the YELLOW capital zone (+3.8 excess) due to
sampling noise at n = 365. The temperature series (GREEN, +1.0 excess)
correctly maps to GREEN. This confirms that capital zone stability is
sensitive to evaluation sample size at the margins — a governance policy
implication for short evaluation horizons.

**Misspecification severity is proportional.** Variance inflation produces
excess exceptions of +86.6 / +68.8 (price / temp), approximately 1.8×
worse than mean bias (+49.0 for both). This ordering is consistent with
the reserve sizing error results below.

---

### 9.2 Operational Reserve Sizing Error

| Dataset | Gov Label | Coverage Error | Reserve Direction |
|---|---|---|---|
| ENTSO-E | RED | −2.94 pp | Undersized |
| PV Solar | RED | +1.37 pp | Oversized |
| Wind | RED | −1.38 pp | Undersized |
| Sim Price (well-spec) | GREEN | −1.51 pp | Undersized |
| Sim Temp (well-spec) | GREEN | −0.41 pp | Undersized |
| Sim Price — Var Inflation | RED | −34.66 pp | Undersized |
| Sim Price — Mean Bias | RED | −19.59 pp | Undersized |
| Sim Price — Heavy Tails | GREEN | +0.14 pp | Oversized |
| Sim Temp — Var Inflation | RED | −27.53 pp | Undersized |
| Sim Temp — Mean Bias | RED | −19.59 pp | Undersized |
| Sim Temp — Heavy Tails | YELLOW | +2.33 pp | Oversized |

ENTSO-E and wind produce undersized reserves (−2.94 pp and −1.38 pp
respectively), meaning that operational reserves sized from these models'
90% intervals would be systematically insufficient. For a grid operator
relying on wind generation intervals to size balancing reserves, a −1.38 pp
shortfall corresponds to roughly 1 in 7 hours being outside the reserved
range rather than the intended 1 in 10.

PV produces oversized reserves (+1.37 pp) — a conservative error that
wastes reserve capacity but does not create shortfall risk. The governance
RED classification for PV therefore reflects distributional shape failure
rather than operational risk in the conventional reserve-sizing sense.

The misspecification scenarios illustrate the magnitude of economic
distortion under severe miscalibration: variance inflation produces reserve
undersizing of 27–35 pp, meaning reserves would need to be 2.7–3.5×
larger than modelled to achieve the intended coverage. Mean bias produces
undersizing of ~20 pp, with the entire shortfall concentrated on one tail.

---

## 10. Traffic-Light Stability Analysis (run_007)

### 10.1 Real-Data Models — Absorbing RED States

| Dataset | Scheme | n Windows | % RED | T_RR | H (bits) | Stability |
|---|---|---|---|---|---|---|
| ENTSO-E | Non-overlapping | 838 | 100% | 1.000 | 0.000 | Stable |
| ENTSO-E | Overlapping | 4,187 | 100% | 1.000 | 0.000 | Stable |
| PV Solar | Non-overlapping | 5 | 100% | 1.000 | 0.000 | Stable |
| Wind | Non-overlapping | 12 | 100% | 1.000 | 0.000 | Stable |

All three real-data models show T_RR = 1.0 across both rolling schemes —
once a window enters RED it never leaves. Stationary entropy H = 0 bits
confirms a degenerate absorbing state: the RED classification is not a
transient regime localised to specific subperiods but a persistent
structural property of the model's miscalibration. For ENTSO-E this is
confirmed across 838 non-overlapping windows and 4,187 overlapping windows,
leaving no ambiguity. The finding strengthens the governance implication:
these models require structural intervention, not routine monitoring.

### 10.2 Simulation Well-Specified — Expected Instability at Small n

| Dataset | Scheme | n Windows | % GREEN | % YELLOW | % RED | H (bits) | Stability |
|---|---|---|---|---|---|---|---|
| Sim Price (well-spec) | Non-overlapping | 7 | 28.6% | 42.9% | 28.6% | 1.571 | Unstable |
| Sim Temp (well-spec) | Non-overlapping | 7 | 14.3% | 85.7% | 0.0% | 0.650 | Moderate |

The well-specified simulation model shows high entropy (1.571 bits for
price) and mixed classifications across 7 windows. This is expected
behaviour: at n = 250 per window with a 10% nominal breach rate, the
binomial sampling noise is sufficient to push rolling coverage across
zone boundaries. The temperature series, which had the smallest full-sample
coverage error (−0.41 pp), shows more stable YELLOW-dominant behaviour
(H = 0.650). Neither result indicates genuine model instability — both
reflect the statistical limits of small rolling windows on a correctly
specified model.

### 10.3 Misspecification Scenarios — Stable vs Ambiguous Signals

| Scenario | n Windows | Distribution | T_RR | H (bits) | Stability |
|---|---|---|---|---|---|
| Price — Var Inflation | 7 | 100% RED | 1.000 | 0.000 | Stable |
| Price — Mean Bias | 7 | 100% RED | 1.000 | 0.000 | Stable |
| Price — Heavy Tails | 7 | 100% YELLOW | — | 0.000 | Stable |
| Temp — Var Inflation | 7 | 100% RED | 1.000 | 0.000 | Stable |
| Temp — Mean Bias | 7 | 100% RED | 1.000 | 0.000 | Stable |
| Temp — Heavy Tails | 7 | 14% GREEN / 57% YELLOW / 28% RED | — | 1.371 | Unstable |

Variance inflation and mean bias produce stable absorbing RED states
(H = 0, T_RR = 1.0), consistent with large systematic miscalibration.
Heavy tails is more nuanced: the price series locks into a stable
YELLOW absorbing state across all 7 non-overlapping windows, while
temperature fluctuates across all three zones (H = 1.371, unstable).

The consistently YELLOW classification for price heavy-tails across
rolling windows is more informative than the full-sample GREEN result:
while the full-sample Anfuso test lacks power to reject at n = 365,
the rolling analysis reveals that every subperiod produces marginal
YELLOW signals. This suggests the heavy-tail misspecification is
detectable at the rolling level even when the full-sample test does
not formally reject — a finding that motivates rolling-window
diagnostics as a complementary detection tool for subtle misspecification.

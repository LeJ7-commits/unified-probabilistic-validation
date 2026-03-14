# Results

This section reports full-sample and rolling diagnostic results for three model
classes evaluated within the unified probabilistic validation framework:
short-term gas load forecasting (ENTSOG), long-term PV generation simulation,
and long-term wind generation simulation. Each dataset is assessed under the
same four-layer diagnostic protocol: interval backtesting (Anfuso traffic-light
framework), distributional diagnostics (PIT-based uniformity and independence
tests), proper scoring (CRPS and empirical coverage), and governance
classification.

---

## 1. ENTSOG — Short-Term Gas Load Forecasting

**Dataset:** Full historical sample, quarter-hourly resolution  
**Evaluable observations:** n = 209,555 (after removal of 200 missing values)  
**Nominal coverage target:** 90% (α = 0.1)  
**Base reconstruction method:** Rolling empirical quantiles with 4-bucket
coarse time-of-day conditioning (night / morning / afternoon / evening) and
global shrinkage. Window Wg = 672 steps (7 days); bucket window Wb = 40
observations. Selected on the basis of feasibility analysis in
`02_entsog_feasibility.ipynb`.

---

### 1.1 Interval Backtesting (Anfuso Traffic-Light Framework)

| Component  | Breaches | Breach Rate | Nominal | Traffic Light |
|------------|----------|-------------|---------|---------------|
| Lower tail | 13,948   | 6.66%       | 5.00%   | RED           |
| Upper tail | 13,162   | 6.28%       | 5.00%   | RED           |
| Total      | 27,110   | 12.94%      | 10.00%  | RED           |

Binomial exceedance p-values (one-sided, H₁: breach rate > nominal):

- Lower tail: p ≈ 1.17 × 10⁻²⁴¹
- Upper tail: p ≈ 2.17 × 10⁻¹⁴⁸
- Total:      p ≈ 0 (below machine precision)

**Interpretation.** Both tails breach significantly above their nominal 5%
targets. The bilateral over-breaching pattern indicates the reconstructed
predictive intervals are systematically too narrow across the full historical
sample — a finding that motivates the conformal augmentation layer addressed
in RQ2.

---

### 1.2 Distributional Diagnostics (PIT-Based)

#### Uniformity Tests

| Test               | Statistic | p-value              |
|--------------------|-----------|----------------------|
| Kolmogorov–Smirnov | 0.1615    | ≈ 0                  |
| Cramér–von Mises   | 2,032.08  | ≈ 3.85 × 10⁻⁷        |
| Anderson–Darling   | 6,136.34  | >> critical values   |

All three uniformity tests strongly reject the null hypothesis of U(0,1) PIT
scores at all conventional significance levels.

#### Independence Tests (Ljung–Box on z = Φ⁻¹(u))

| Lag | Statistic   | p-value |
|-----|-------------|---------|
| 5   | 846,049     | ≈ 0     |
| 10  | 1,563,119   | ≈ 0     |
| 20  | 2,699,489   | ≈ 0     |

Serial independence is overwhelmingly rejected at all lags. The magnitude of
the statistics — several orders above chi-squared critical values — indicates
severe autocorrelation in the PIT residuals, consistent with the model failing
to capture persistent temporal structure in gas load.

---

### 1.3 Proper Scoring

| Metric             | Value       |
|--------------------|-------------|
| Mean CRPS          | 1,515.23    |
| Empirical coverage | 87.06%      |
| Coverage error     | −2.94 pp    |

The negative coverage error confirms systematic under-coverage. The CRPS
reflects absolute forecast scale (gas load in physical units); cross-asset
comparisons of CRPS should account for differences in scale and units.

---

### 1.4 Governance Classification

| Criterion              | Status   |
|------------------------|----------|
| PIT uniformity         | FAIL     |
| PIT independence       | FAIL     |
| Coverage error         | −2.94 pp |
| Overall classification | **RED**  |

---

### 1.5 Rolling Evaluation

Rolling-window diagnostics were conducted under two schemes: non-overlapping
(window = 250, step = 250) and overlapping (window = 250, step = 50) windows.
Across both specifications, PIT uniformity is frequently rejected, independence
violations persist, and both tails breach above nominal rates recurrently.
Overlapping windows produce smoother diagnostic trajectories but confirm the
same structural deficiencies. The consistency across both schemes indicates the
miscalibration is persistent across time rather than localised to isolated
regimes.

---

## 2. PV — Long-Term Solar Generation Simulation

**Dataset:** Hourly resolution, 2013–2015  
**Raw observations:** n = 26,280  
**Nighttime exclusion:** 10,036 rows where both simulation and actuals are
below 1 × 10⁻⁹ are excluded from calibration evaluation. These are structural
nighttime zeros for PV generation, not forecast errors; including them would
artificially inflate coverage metrics and distort PIT diagnostics.  
**Evaluable observations after nighttime exclusion and warmup:** n = 4,287  
**Nominal coverage target:** 90% (α = 0.1)  
**Base reconstruction method:** Rolling empirical quantiles with 24-bucket
hour-of-day conditioning. Window W = 720 trailing same-hour observations
(~30 days). Justified by 3 years of hourly data (~1,095 observations per
hour-bucket), ensuring stable quantile estimation.

---

### 2.1 Interval Backtesting (Anfuso Traffic-Light Framework)

| Component  | Breaches | Breach Rate | Nominal | Traffic Light |
|------------|----------|-------------|---------|---------------|
| Lower tail | 156      | 3.64%       | 5.00%   | GREEN         |
| Upper tail | 214      | 4.99%       | 5.00%   | GREEN         |
| Total      | 370      | 8.63%       | 10.00%  | GREEN         |

Binomial exceedance p-values:

- Lower tail: p ≈ 1.000 (strongly conservative)
- Upper tail: p ≈ 0.520 (no significant over-breaching)
- Total:      p ≈ 0.999 (overall conservative)

**Interpretation.** The PV model passes interval backtesting at all levels.
Total coverage of 91.37% slightly exceeds the 90% nominal target, indicating
the reconstructed intervals are mildly conservative. The lower tail is
particularly conservative (3.64% vs 5% nominal), consistent with the physical
lower bound of zero for solar generation creating a floor effect in the
residual distribution.

**Statistical power caveat.** With n = 4,287 evaluable daytime observations,
the binomial test has substantially less power to detect modest over-breaching
than in the ENTSOG case (n = 209,555). The GREEN classification reflects both
genuine calibration and reduced detection power, and should be interpreted
accordingly.

---

### 2.2 Distributional Diagnostics (PIT-Based)

#### Uniformity Tests

| Test               | Statistic | p-value              |
|--------------------|-----------|----------------------|
| Kolmogorov–Smirnov | 0.1028    | ≈ 7.23 × 10⁻⁴⁰      |
| Cramér–von Mises   | 14.48     | ≈ 1.05 × 10⁻⁹        |
| Anderson–Darling   | 316.30    | >> critical values   |

All uniformity tests reject U(0,1) strongly, though test statistics are
substantially smaller than those observed for ENTSOG, reflecting both the
smaller sample and a less severe departure from uniformity.

#### Independence Tests (Ljung–Box on z = Φ⁻¹(u))

| Lag | Statistic | p-value |
|-----|-----------|---------|
| 5   | 3,986     | ≈ 0     |
| 10  | 4,741     | ≈ 0     |
| 20  | 5,012     | ≈ 0     |

Serial independence is strongly rejected. PV generation exhibits pronounced
diurnal and seasonal autocorrelation; the rolling reconstruction partially
captures this structure but does not eliminate residual serial dependence.

---

### 2.3 Proper Scoring

| Metric             | Value    |
|--------------------|----------|
| Mean CRPS          | 0.7937   |
| Empirical coverage | 91.37%   |
| Coverage error     | +1.37 pp |

The positive coverage error is consistent with the mildly conservative
interval classification. The mean CRPS of 0.79 is substantially lower than
ENTSOG (1,515.23), reflecting the difference in physical scale — PV generation
is expressed as a capacity factor (0–1 range) rather than absolute load.

---

### 2.4 Governance Classification

| Criterion              | Status   |
|------------------------|----------|
| PIT uniformity         | FAIL     |
| PIT independence       | FAIL     |
| Coverage error         | +1.37 pp |
| Overall classification | **RED**  |

**Key finding.** The PV model receives a RED governance classification despite
passing the interval backtesting layer. This illustrates a central argument of
the thesis: naive coverage-based metrics are insufficient for probabilistic
model validation. A model can produce well-calibrated interval widths on
average while still exhibiting systematic departures from the assumed
distributional form and significant serial dependence in its probability
integral transforms.

---

### 2.5 Rolling Evaluation

Rolling-window diagnostics were conducted under non-overlapping and overlapping
schemes (window = 720, step = 168). PIT uniformity rejections and independence
violations persist across subperiods with no evidence of improvement in later
windows. The rolling results are consistent with the full-sample findings.

---

## 3. Wind — Long-Term Wind Generation Simulation

**Dataset:** Hourly resolution, 2013–2015  
**Raw observations:** n = 26,280  
**Nighttime exclusion:** Not applied. Wind generation occurs around the clock;
no structural zeros are present.  
**Evaluable observations after warmup:** n = 9,000  
**Nominal coverage target:** 90% (α = 0.1)  
**Base reconstruction method:** Rolling empirical quantiles with 24-bucket
hour-of-day conditioning. Window W = 720 trailing same-hour observations
(~30 days).

---

### 3.1 Interval Backtesting (Anfuso Traffic-Light Framework)

| Component  | Breaches | Breach Rate | Nominal | Traffic Light |
|------------|----------|-------------|---------|---------------|
| Lower tail | 580      | 6.44%       | 5.00%   | RED           |
| Upper tail | 444      | 4.93%       | 5.00%   | GREEN         |
| Total      | 1,024    | 11.38%      | 10.00%  | RED           |

Binomial exceedance p-values:

- Lower tail: p ≈ 8.97 × 10⁻¹⁰
- Upper tail: p ≈ 0.621 (no significant over-breaching)
- Total:      p ≈ 1.02 × 10⁻⁵

**Interpretation.** Wind exhibits asymmetric tail miscalibration concentrated
in the lower tail. This is physically interpretable: wind generation is bounded
below by zero and exhibits extended low-generation periods during calm weather
that the rolling reconstruction underestimates. The upper tail — representing
high-generation events — is captured accurately, likely because high-wind
residuals are more stable across seasons.

---

### 3.2 Distributional Diagnostics (PIT-Based)

#### Uniformity Tests

| Test               | Statistic | p-value              |
|--------------------|-----------|----------------------|
| Kolmogorov–Smirnov | 0.1057    | ≈ 5.90 × 10⁻⁸⁸      |
| Cramér–von Mises   | 21.28     | ≈ 2.47 × 10⁻⁹        |
| Anderson–Darling   | 519.20    | >> critical values   |

All uniformity tests strongly reject U(0,1). Test statistics are larger than
the PV case, consistent with wind's more complex residual structure and the
absence of nighttime filtering.

#### Independence Tests (Ljung–Box on z = Φ⁻¹(u))

| Lag | Statistic | p-value |
|-----|-----------|---------|
| 5   | 25,553    | ≈ 0     |
| 10  | 40,250    | ≈ 0     |
| 20  | 56,247    | ≈ 0     |

Serial independence is strongly rejected. Wind speed exhibits well-known
meteorological persistence; the Ljung–Box statistics grow substantially with
lag, indicating autocorrelation extending well beyond short-term dependence.

---

### 3.3 Proper Scoring

| Metric             | Value    |
|--------------------|----------|
| Mean CRPS          | 1.7893   |
| Empirical coverage | 88.62%   |
| Coverage error     | −1.38 pp |

The negative coverage error is consistent with the RED lower-tail
classification. Mean CRPS of 1.79 is higher than PV (0.79) in absolute terms,
reflecting the wider residual distribution of wind relative to the bounded
capacity factor range.

---

### 3.4 Governance Classification

| Criterion              | Status   |
|------------------------|----------|
| PIT uniformity         | FAIL     |
| PIT independence       | FAIL     |
| Coverage error         | −1.38 pp |
| Overall classification | **RED**  |

---

### 3.5 Rolling Evaluation

Rolling-window diagnostics were conducted under non-overlapping and overlapping
schemes (window = 720, step = 168). Lower-tail over-breaching is recurrent
across subperiods. PIT independence violations are persistent and grow with
lag, consistent with meteorological persistence in wind speed. No subperiods
show sustained improvement across all diagnostic criteria simultaneously.

---

## 4. Simulation — Pricing and Risk Analysis Models (Positive Control)

**Dataset:** Synthetic joint price–temperature simulation (correlated Gaussian DGP)  
**Series evaluated:** Price and temperature independently (univariate marginal calibration)  
**Evaluable observations:** n = 365 (one as-of date per day, h=1 horizon evaluation)  
**Nominal coverage target:** 90% (α = 0.1)  
**Specification:** Well-specified — realised values drawn from the same DGP as simulation
paths (ρ = 0.5, σ_price = 5.0, σ_temp = 3.0). This is a controlled positive-control
baseline: the diagnostic framework should return GREEN under a correctly specified model.  
**Reconstruction method:** Empirical α/2 and 1−α/2 quantiles of 5,000 simulation paths
at h=1 per as-of date. No rolling reconstruction required — the distributional form is
known exactly.

---

### 4.1 Interval Backtesting (Anfuso Traffic-Light Framework)

#### Price

| Component  | Breaches | Breach Rate | Nominal | Traffic Light |
|------------|----------|-------------|---------|---------------|
| Lower tail | 24       | 6.58%       | 5.00%   | GREEN         |
| Upper tail | 18       | 4.93%       | 5.00%   | GREEN         |
| Total      | 42       | 11.51%      | 10.00%  | GREEN         |

Binomial exceedance p-values: lower p = 0.107, upper p = 0.558, total p = 0.190.

#### Temperature

| Component  | Breaches | Breach Rate | Nominal | Traffic Light |
|------------|----------|-------------|---------|---------------|
| Lower tail | 23       | 6.30%       | 5.00%   | GREEN         |
| Upper tail | 15       | 4.11%       | 5.00%   | GREEN         |
| Total      | 38       | 10.41%      | 10.00%  | GREEN         |

Binomial exceedance p-values: lower p = 0.154, upper p = 0.815, total p = 0.422.

**Interpretation.** Both series pass interval backtesting at all levels. Breach
rates are close to nominal and no p-value falls below conventional significance
thresholds. This confirms the framework does not produce false positives under a
correctly specified model — a necessary property of any valid diagnostic system.
Minor deviations from nominal rates (e.g. price lower tail at 6.58%) are
consistent with expected sampling noise at n = 365.

---

### 4.2 Distributional Diagnostics (PIT-Based)

PIT-based uniformity and independence tests were not computed for the simulation
series. The h=1 evaluation approach provides only the empirical quantile bounds
(lo/hi) per as-of date, not a full CDF evaluated at each realisation. Computing
PIT scores would require either saving the full path matrix or evaluating the
known Gaussian CDF analytically at each realisation. This is a known limitation
of the current h=1 artifact structure and is noted as a direction for extension.

---

### 4.3 Proper Scoring

| Series      | Empirical Coverage | Coverage Error |
|-------------|-------------------|----------------|
| Price       | 88.49%            | −1.51 pp       |
| Temperature | 89.59%            | −0.41 pp       |

Both series fall marginally short of 90% nominal coverage, consistent with
expected finite-sample variability at n = 365. Neither deviation is
statistically significant.

---

### 4.4 Governance Classification

| Series      | PIT Uniformity | PIT Independence | Coverage Error | Overall   |
|-------------|----------------|------------------|----------------|-----------|
| Price       | N/A            | N/A              | −1.51 pp       | **GREEN** |
| Temperature | N/A            | N/A              | −0.41 pp       | **GREEN** |

**Key finding.** Both simulation series receive GREEN governance classifications.
This validates a critical property of the framework: under a well-specified model,
the diagnostic system does not generate false positive RED classifications. The
positive-control result strengthens the interpretability of RED findings for
ENTSOG, PV, and wind — those failures are attributable to genuine model
deficiencies, not to artefacts of the diagnostic procedure.

---

### 4.5 Rolling Evaluation

Rolling-window diagnostics were conducted (window = 50 as-of dates, step = 10),
yielding approximately 31 non-overlapping windows. Interval coverage remained
close to nominal across subperiods for both series with no systematic
deterioration over time, consistent with the stationary DGP — no distributional
drift is present by construction.

---

## 5. Cross-Dataset Synthesis

### 5.1 Summary Table

| Dataset    | n       | Coverage | Error    | Lower TL | Upper TL | Total TL | PIT Unif | PIT Indep | Overall   |
|------------|---------|----------|----------|----------|----------|----------|----------|-----------|-----------|
| ENTSOG     | 209,555 | 87.06%   | −2.94 pp | RED      | RED      | RED      | FAIL     | FAIL      | **RED**   |
| PV         | 4,287   | 91.37%   | +1.37 pp | GREEN    | GREEN    | GREEN    | FAIL     | FAIL      | **RED**   |
| Wind       | 9,000   | 88.62%   | −1.38 pp | RED      | GREEN    | RED      | FAIL     | FAIL      | **RED**   |
| Sim. Price | 365     | 88.49%   | −1.51 pp | GREEN    | GREEN    | GREEN    | N/A      | N/A       | **GREEN** |
| Sim. Temp  | 365     | 89.59%   | −0.41 pp | GREEN    | GREEN    | GREEN    | N/A      | N/A       | **GREEN** |

---

### 5.2 Key Findings

**Finding 1 — The framework correctly distinguishes well-specified from
misspecified models.**  
The simulation positive control (GREEN) and the three real-data models (all RED)
confirm that the diagnostic system is both sensitive and specific: it flags
genuine miscalibration while avoiding false positives under a correctly specified
DGP. This is the foundational property required for governance credibility.

**Finding 2 — All real-data models receive RED governance classification.**  
Despite differences in asset class, data frequency, and reconstruction method,
ENTSOG, PV, and wind all fail the distributional and independence diagnostic
layers. Systematic miscalibration is not idiosyncratic to a single model or
asset class but is a structural feature of simulation and forecasting models
when evaluated under a rigorous probabilistic validation framework.

**Finding 3 — Interval backtesting alone is insufficient.**  
PV passes all interval backtesting checks (GREEN across both tails and total)
yet receives a RED governance classification due to distributional and
independence failures. This directly validates the multi-layer architecture of
the framework: a model satisfying naive coverage criteria can still exhibit
systematic departures from the assumed probabilistic structure.

**Finding 4 — Tail asymmetry is asset-specific and physically interpretable.**  
ENTSOG exhibits bilateral over-breaching. Wind exhibits lower-tail dominance
consistent with meteorological persistence and the zero lower bound. PV
exhibits conservative lower-tail behaviour consistent with the structural zero
floor from nighttime generation. A unified framework must accommodate
asset-specific tail behaviour rather than applying symmetric interval
assumptions.

**Finding 5 — Serial dependence is universal and severe in real-data models.**  
All three real-data datasets produce Ljung–Box statistics orders of magnitude
above critical values at all lags tested. None of the base predictive
distributions adequately capture the temporal autocorrelation structure of
their respective processes — a finding with direct implications for risk
aggregation and scenario generation in energy market models. This stands in
contrast to the simulation positive control, where no serial dependence is
present by DGP construction.

**Finding 6 — Sample size affects statistical power materially.**  
The PV evaluable sample (n = 4,287) and simulation series (n = 365) are two
to three orders of magnitude smaller than ENTSOG (n = 209,555). Traffic-light
classifications should always be accompanied by sample size reporting and power
considerations in governance documentation.

**Finding 7 — Joint serial dependence is confirmed across asset classes.**  
The multivariate Ljung-Box test on the joint PV–wind PIT residual vector
rejects independence overwhelmingly at all lags, extending the univariate
serial dependence findings to the joint distribution. This confirms that
portfolio-level risk aggregation using these marginal distributions would
propagate miscalibrated dependence structure, not just marginal errors.

---

## 6. Multivariate Joint Evaluation: PV and Wind

**Assets:** PV (solar) and wind, evaluated jointly on their shared daytime
hourly index (2013–2015).  
**Shared evaluable observations:** n = 4,287 (daytime hours where both
series have evaluable derived artifacts after nighttime exclusion and warmup).  
**Method:** Marginal PIT scores computed from empirical sample CDFs (500 paths
per observation), transformed via z = Φ⁻¹(u), then evaluated jointly.

---

### 6.1 Marginal PIT Summary on Shared Index

| Series | Mean z  | Std z  | Interpretation                           |
|--------|---------|--------|------------------------------------------|
| PV     | +0.104  | 1.941  | Slight positive bias; overdispersed      |
| Wind   | −0.051  | 1.600  | Near-zero bias; moderately overdispersed |

Under a correctly specified model, z-residuals should have mean ≈ 0 and
std ≈ 1. Both series show std > 1, consistent with the distributional
overdispersion identified in the univariate RED findings. PV's higher std
reflects the greater heterogeneity of daytime solar residuals across seasons.

---

### 6.2 Multivariate Ljung-Box Test

Applied to the stacked joint residual matrix Z = [z_pv, z_wind] of shape
(4,287 × 2). Under H₀ (joint white noise), the Hosking (1980) portmanteau
statistic follows χ²(k²·h) with k=2.

| Lag | Statistic | df | p-value |
|-----|-----------|----|---------|
| 5   | 15,427    | 20 | ≈ 0     |
| 10  | 20,692    | 40 | ≈ 0     |
| 20  | 24,714    | 80 | ≈ 0     |

**Verdict: REJECT joint independence at all lags.**

The statistics are orders of magnitude above chi-squared critical values.
Joint serial dependence in the PIT residual vector is overwhelmingly confirmed.
This extends the univariate Ljung-Box findings: not only is each series
individually serially dependent, the joint residual process is not white noise.
For portfolio-level applications — such as joint generation forecasting or
renewable energy risk aggregation — this means the marginal predictive
distributions cannot be combined via independence assumptions without
materially misrepresenting the joint uncertainty.

---

### 6.3 Cross-Correlation of PIT Residuals

| Lag | Correlation | Interpretation                                   |
|-----|-------------|--------------------------------------------------|
| 0   | −0.040      | Weak negative contemporaneous dependence         |
| 1   | −0.041      | Near-identical to lag 0; no decay over 1 hour    |
| 6   | −0.031      | Modest decay over 6 hours                        |
| 24  | +0.075      | Positive correlation at same hour next day       |

The contemporaneous correlation of −0.040 is physically plausible: during
daytime hours, calm sunny conditions (high PV, low wind) and cloudy windy
conditions (low PV, high wind) create a weak negative co-movement in
generation errors. The positive lag-24 correlation reflects shared diurnal
weather persistence — errors at the same hour on consecutive days tend to
co-move positively, consistent with multi-day weather regimes. None of the
cross-correlations are large in magnitude, suggesting approximate
contemporaneous independence in a practical sense; however, the serial
structure within each series dominates the joint dependence picture.

---

### 6.4 Bivariate Energy Score

| Metric                      | Value |
|-----------------------------|-------|
| Mean bivariate energy score | 2.017 |

The bivariate energy score (Gneiting and Raftery, 2007) is a proper scoring
rule for multivariate predictive distributions. This value serves as the
pre-conformal baseline for the joint PV–wind predictive distribution, enabling
future comparison after model improvement or joint conformal augmentation.

---

### 6.5 Governance Implication

| Criterion                    | Status                       |
|------------------------------|------------------------------|
| Joint serial independence    | FAIL                         |
| Contemporaneous independence | MARGINAL (r ≈ −0.04)         |
| Joint overall classification | **RED**                      |

Portfolio-level applications using these marginal distributions should not
assume independence between PV and wind generation errors, particularly at
the same hour across consecutive days. A copula-based or joint conformal
approach would be required to address the dependence structure beyond
marginal recalibration.

---

## 7. Motivation for Conformal Augmentation (RQ2)

The consistent finding of coverage shortfalls and distributional miscalibration
across all real-data model classes motivates the conformal prediction
augmentation layer evaluated in RQ2. Conformal methods offer finite-sample
marginal coverage guarantees that are distribution-free and do not require the
base model to be correctly specified. The pre-conformal results reported here
serve as the baseline against which post-conformal coverage and calibration
improvements will be assessed. The simulation positive control further confirms
that conformal augmentation is necessary for the real-data models specifically,
not an artefact of the validation procedure.

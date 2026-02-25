# Appendix A — Theoretical Interpretation of PIT Shapes

## A.1 Probability Integral Transform (PIT)

For a predictive cumulative distribution function ( F_t ) and realization ( y_t ), the Probability Integral Transform is defined as:

$
u_t = F_t(y_t)
$

If the predictive distribution is correctly specified, then:

$
u_t \sim \text{Uniform}(0,1)
$

Systematic deviations from uniformity therefore reveal specific forms of distributional misspecification.

---

## A.2 Variance Misspecification

### Overdispersion (Forecast Variance Too Large)

If the predictive distribution is too wide (i.e., forecast variance exceeds true variance), realizations tend to fall near the center of the predictive distribution.

Consequently:

* PIT values cluster around 0.5
* Fewer observations appear near 0 or 1
* The PIT histogram becomes **hump-shaped**

Intuition:
When forecasts are overly diffuse, realizations rarely lie in the tails relative to the forecast.

---

### Underdispersion (Forecast Variance Too Small)

If the predictive distribution is too narrow, realizations frequently fall in the tails relative to the forecast.

Consequently:

* PIT values concentrate near 0 and 1
* The PIT histogram becomes **U-shaped**

Intuition:
The forecast distribution underestimates variability, causing realizations to fall outside central regions more often than expected.

---

## A.3 Mean Misspecification (Location Bias)

If the predictive mean is systematically shifted relative to the true process:

[
\hat{\mu} \neq \mu
]

then realizations consistently fall to one side of the predictive distribution.

Consequently:

* PIT values accumulate near 0 or near 1
* The PIT histogram becomes **skewed**

For example, if the forecast mean is too large, realizations tend to fall below the forecast center, generating PIT values concentrated near 0.

---

## A.4 Implications for Diagnostic Design

These theoretical signatures justify the use of:

* PIT histograms (shape diagnostics)
* Uniformity tests (distributional deviation)
* Proper scoring rules (magnitude of error)
* Rolling evaluation (temporal persistence)

Because different misspecification types generate distinct PIT patterns, no single diagnostic is sufficient in isolation. A layered approach is therefore necessary for reliable model validation.

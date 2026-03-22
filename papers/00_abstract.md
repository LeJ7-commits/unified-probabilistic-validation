# Abstract

Energy market models span structurally heterogeneous classes — Monte Carlo
simulation engines, short-term operational forecasting systems, and
long-horizon renewable generation models — yet validation practice remains
fragmented along model class boundaries, with each class inheriting its
own evaluation culture and none providing a complete picture of
probabilistic reliability. This thesis constructs a unified probabilistic
validation framework that evaluates all three model classes within a shared
reliability space defined by distributional calibration, proper scoring,
and governance-oriented aggregation.

The framework comprises eleven software components organised into five
layers: a canonical data contract and model-class adapters that normalise
heterogeneous inputs into evaluable predictive distributions; a
capability-aware diagnostics gateway routing inputs to PIT uniformity,
serial independence, interval backtesting, pinball loss, and sharpness
diagnostics; a regime tagger and threshold calibrator enabling
regime-conditioned governance; a decision engine producing structured
governance decisions with full provenance audit trails; and an AI-powered
narrative generator translating decisions into technical and plain-language
summaries. The framework is deployed as a Streamlit web application and
validated by 451 automated tests.

Empirically, all three real-data model classes receive RED governance
classifications, each with a structurally distinct failure mode: bilateral
miscalibration in the ENTSO-E electricity load model, distributional
failure without interval failure in PV solar generation, and asymmetric
lower-tail failure in onshore wind generation. The synthetic positive
control confirms the framework is not over-strict. The key governance
finding is the PV divergence: coverage-only regulation would reduce
capital requirements for a model that multi-layer diagnostics correctly
identify as structurally misspecified — a concrete empirical demonstration
that interval coverage alone is insufficient for energy market governance.
Conformal augmentation restores near-nominal coverage under temporal
distribution shift but does not resolve distributional misspecification,
establishing the two methods as complementary rather than substitutable.

The framework is open-source at
github.com/LeJ7-commits/unified-probabilistic-validation.
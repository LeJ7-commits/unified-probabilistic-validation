"""
tests/test_interval_sharpness.py
==================================
Pytest suite for Interval_Sharpness and SharpnessResult.

Groups:
  1. Interval_Sharpness.compute — happy path
  2. Sharpness and risk labels
  3. Regime-stratified sharpness
  4. Rolling sharpness
  5. Error conditions
  6. SharpnessResult.to_dict
  7. compute_from_dro

Run with:
  python -m pytest tests/test_interval_sharpness.py -v
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.diagnostics.interval_sharpness import (
    Interval_Sharpness,
    SharpnessResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def n():
    return 500

@pytest.fixture
def y(n):
    rng = np.random.default_rng(42)
    return rng.normal(50, 5, size=n)

@pytest.fixture
def lo(y):
    return y - 4.0

@pytest.fixture
def hi(y):
    return y + 4.0

@pytest.fixture
def sharpness():
    return Interval_Sharpness(alpha=0.1)

@pytest.fixture
def result(sharpness, lo, hi, y):
    return sharpness.compute(lo=lo, hi=hi, y=y)


# ---------------------------------------------------------------------------
# 1. Happy path
# ---------------------------------------------------------------------------

class TestHappyPath:

    def test_returns_sharpness_result(self, result):
        assert isinstance(result, SharpnessResult)

    def test_n_obs(self, result, n):
        assert result.n_obs == n

    def test_mean_width_correct(self, result):
        np.testing.assert_allclose(result.mean_width, 8.0, rtol=1e-10)

    def test_median_width_correct(self, result):
        np.testing.assert_allclose(result.median_width, 8.0, rtol=1e-10)

    def test_widths_shape(self, result, n):
        assert result.widths.shape == (n,)

    def test_widths_non_negative(self, result):
        assert np.all(result.widths >= 0)

    def test_coverage_indicator_shape(self, result, n):
        assert result.coverage_indicator.shape == (n,)

    def test_coverage_indicator_boolean(self, result):
        assert result.coverage_indicator.dtype == bool

    def test_empirical_coverage_for_symmetric_interval(self, n, y):
        """lo = y-4, hi = y+4 → all obs inside → coverage = 1.0."""
        sharpness = Interval_Sharpness(alpha=0.1)
        result = sharpness.compute(lo=y - 4, hi=y + 4, y=y)
        np.testing.assert_allclose(result.empirical_coverage, 1.0)

    def test_empirical_coverage_zero_for_inverted(self, n, y):
        """lo > y > hi → no obs inside → coverage = 0.0."""
        sharpness = Interval_Sharpness(alpha=0.1)
        result = sharpness.compute(lo=y + 1, hi=y + 2, y=y)
        np.testing.assert_allclose(result.empirical_coverage, 0.0)

    def test_coverage_error_sign(self, n, y):
        """Perfect coverage (100%) → positive error vs 90% nominal."""
        sharpness = Interval_Sharpness(alpha=0.1)
        result = sharpness.compute(lo=y - 10, hi=y + 10, y=y)
        assert result.coverage_error > 0   # over-covered

    def test_alpha_stored(self, result):
        assert result.alpha == 0.1

    def test_nominal_coverage(self, result):
        np.testing.assert_allclose(result.nominal_coverage, 0.9)

    def test_std_width_zero_for_constant_width(self, n, y):
        """Constant-width intervals → std = 0."""
        sharpness = Interval_Sharpness(alpha=0.1)
        result = sharpness.compute(lo=y - 3, hi=y + 3, y=y)
        np.testing.assert_allclose(result.std_width, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 2. Sharpness and risk labels
# ---------------------------------------------------------------------------

class TestLabels:

    def test_sharp_label(self, n, y):
        """Very narrow intervals → sharp."""
        sharpness = Interval_Sharpness(alpha=0.1)
        # Width = 0.1 × IQR(y) ≈ 0.1 × (53.3 - 46.7) ≈ 0.66 → relative = 0.1
        iqr = float(np.percentile(y, 75) - np.percentile(y, 25))
        result = sharpness.compute(
            lo=y - 0.05 * iqr, hi=y + 0.05 * iqr, y=y
        )
        assert result.sharpness_label == "sharp"

    def test_uninformative_label(self, n, y):
        """Very wide intervals → uninformative."""
        sharpness = Interval_Sharpness(alpha=0.1, wide_threshold=2.0)
        iqr = float(np.percentile(y, 75) - np.percentile(y, 25))
        result = sharpness.compute(
            lo=y - 5 * iqr, hi=y + 5 * iqr, y=y
        )
        assert result.sharpness_label == "uninformative"

    def test_risky_label_for_undercoverage(self, n, y):
        """Sharp intervals shifted away from y → severe undercoverage → risky."""
        sharpness = Interval_Sharpness(alpha=0.1)
        # Shift intervals far above y — observations all fall below lo
        result = sharpness.compute(lo=y + 100, hi=y + 101, y=y)
        assert result.risk_label == "risky"

    def test_over_cautious_for_overcoverage(self, n, y):
        """Very wide intervals → over-cautious."""
        sharpness = Interval_Sharpness(alpha=0.1)
        # Intervals that capture everything
        result = sharpness.compute(lo=y - 100, hi=y + 100, y=y)
        assert result.risk_label == "over-cautious"

    def test_interpretation_is_string(self, result):
        assert isinstance(result.interpretation, str)
        assert len(result.interpretation) > 0


# ---------------------------------------------------------------------------
# 3. Regime-stratified sharpness
# ---------------------------------------------------------------------------

class TestRegimeStratification:

    def test_regime_stats_keys(self, sharpness, lo, hi, y, n):
        tags = ["winter"] * (n // 2) + ["summer"] * (n - n // 2)
        result = sharpness.compute(lo=lo, hi=hi, y=y, regime_tags=tags)
        assert set(result.regime_stats.keys()) == {"winter", "summer"}

    def test_regime_stats_fields(self, sharpness, lo, hi, y, n):
        tags = ["a"] * n
        result = sharpness.compute(lo=lo, hi=hi, y=y, regime_tags=tags)
        required = {
            "mean_width", "median_width", "empirical_coverage",
            "coverage_error", "n_obs"
        }
        assert required.issubset(set(result.regime_stats["a"].keys()))

    def test_single_regime_n_obs(self, sharpness, lo, hi, y, n):
        tags = ["all"] * n
        result = sharpness.compute(lo=lo, hi=hi, y=y, regime_tags=tags)
        assert result.regime_stats["all"]["n_obs"] == n

    def test_regime_n_obs_sum(self, sharpness, lo, hi, y, n):
        n1 = n // 3
        n2 = n - n1
        tags = ["a"] * n1 + ["b"] * n2
        result = sharpness.compute(lo=lo, hi=hi, y=y, regime_tags=tags)
        total = (
            result.regime_stats["a"]["n_obs"]
            + result.regime_stats["b"]["n_obs"]
        )
        assert total == n

    def test_regime_tags_wrong_length_raises(self, sharpness, lo, hi, y, n):
        with pytest.raises(ValueError, match="length"):
            sharpness.compute(
                lo=lo, hi=hi, y=y, regime_tags=["a"] * (n - 1)
            )


# ---------------------------------------------------------------------------
# 4. Rolling sharpness
# ---------------------------------------------------------------------------

class TestRollingSharpness:

    def test_rolling_widths_computed(self, lo, hi, y, n):
        window = 100
        sharpness = Interval_Sharpness(alpha=0.1, rolling_window=window)
        result = sharpness.compute(lo=lo, hi=hi, y=y)
        assert result.rolling_widths is not None
        expected_windows = n - window + 1
        assert result.rolling_widths.shape == (expected_windows,)

    def test_rolling_coverage_computed(self, lo, hi, y, n):
        window = 100
        sharpness = Interval_Sharpness(alpha=0.1, rolling_window=window)
        result = sharpness.compute(lo=lo, hi=hi, y=y)
        assert result.rolling_coverage is not None
        assert result.rolling_coverage.shape == result.rolling_widths.shape

    def test_no_rolling_by_default(self, result):
        assert result.rolling_widths is None
        assert result.rolling_coverage is None

    def test_rolling_window_too_large_warns(self, lo, hi, y, n):
        sharpness = Interval_Sharpness(alpha=0.1, rolling_window=n + 100)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = sharpness.compute(lo=lo, hi=hi, y=y)
            assert any("rolling_window" in str(x.message) for x in w)
        assert result.rolling_widths is None

    def test_rolling_widths_positive(self, lo, hi, y, n):
        sharpness = Interval_Sharpness(alpha=0.1, rolling_window=50)
        result = sharpness.compute(lo=lo, hi=hi, y=y)
        assert np.all(result.rolling_widths >= 0)

    def test_constant_interval_constant_rolling_width(self, y, n):
        """Constant-width intervals → all rolling windows have same width."""
        sharpness = Interval_Sharpness(alpha=0.1, rolling_window=50)
        result = sharpness.compute(lo=y - 3, hi=y + 3, y=y)
        np.testing.assert_allclose(
            result.rolling_widths,
            np.full_like(result.rolling_widths, 6.0),
            rtol=1e-10,
        )


# ---------------------------------------------------------------------------
# 5. Error conditions
# ---------------------------------------------------------------------------

class TestErrors:

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            Interval_Sharpness(alpha=1.5)

    def test_lo_wrong_shape_raises(self, sharpness, hi, y, n):
        with pytest.raises(ValueError, match="shape"):
            sharpness.compute(lo=np.ones(n - 1), hi=hi, y=y)

    def test_hi_wrong_shape_raises(self, sharpness, lo, y, n):
        with pytest.raises(ValueError, match="shape"):
            sharpness.compute(lo=lo, hi=np.ones(n - 1), y=y)

    def test_lo_gt_hi_warns(self, sharpness, y, n):
        lo_bad = y + 2
        hi_bad = y - 2
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = sharpness.compute(lo=lo_bad, hi=hi_bad, y=y)
            assert any("lo > hi" in str(x.message) for x in w)


# ---------------------------------------------------------------------------
# 6. SharpnessResult.to_dict
# ---------------------------------------------------------------------------

class TestToDict:

    def test_to_dict_keys(self, result):
        d = result.to_dict()
        required = {
            "mean_width", "median_width", "std_width",
            "empirical_coverage", "coverage_error",
            "sharpness_label", "risk_label", "interpretation",
            "n_obs", "alpha", "nominal_coverage", "regime_stats",
        }
        assert required.issubset(set(d.keys()))

    def test_to_dict_values_finite(self, result):
        d = result.to_dict()
        assert np.isfinite(d["mean_width"])
        assert np.isfinite(d["empirical_coverage"])

    def test_to_dict_regime_stats_in_dict(self, sharpness, lo, hi, y, n):
        tags = ["a"] * n
        result = sharpness.compute(lo=lo, hi=hi, y=y, regime_tags=tags)
        d = result.to_dict()
        assert "a" in d["regime_stats"]


# ---------------------------------------------------------------------------
# 7. compute_from_dro
# ---------------------------------------------------------------------------

class TestComputeFromDRO:

    def test_from_dro_matches_direct(self, sharpness, lo, hi, y, n):
        from src.diagnostics.diagnostics_input import Diagnostics_Input
        t = pd.date_range("2021-01-01", periods=n, freq="h")
        di = Diagnostics_Input()
        dro = di.from_arrays(y=y, t=t, model_id="m", lo=lo, hi=hi)

        result_dro    = sharpness.compute_from_dro(dro)
        result_direct = sharpness.compute(lo=lo, hi=hi, y=y)

        np.testing.assert_allclose(
            result_dro.mean_width, result_direct.mean_width, rtol=1e-10
        )
        np.testing.assert_allclose(
            result_dro.empirical_coverage,
            result_direct.empirical_coverage,
            rtol=1e-10,
        )

    def test_from_dro_no_interval_capability_raises(self, sharpness, y, n):
        from src.diagnostics.diagnostics_input import (
            Diagnostics_Input,
            DiagnosticsInputError,
        )
        rng = np.random.default_rng(99)
        S = rng.normal(size=(n, 100))
        t = pd.date_range("2021-01-01", periods=n, freq="h")
        di = Diagnostics_Input()
        dro = di.from_arrays(y=y, t=t, model_id="m", samples=S)

        with pytest.raises(DiagnosticsInputError, match="interval"):
            sharpness.compute_from_dro(dro)

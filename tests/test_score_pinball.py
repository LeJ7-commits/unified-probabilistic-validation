"""
tests/test_score_pinball.py
============================
Pytest suite for Score_Pinball, PinballResult, and pinball_loss.

Groups:
  1. pinball_loss (element-wise)
  2. Score_Pinball.compute — happy path
  3. Score_Pinball.compute — regime stratification
  4. Score_Pinball.compute — crossing detection
  5. Score_Pinball.compute — error conditions
  6. PinballResult.to_dict
  7. Score_Pinball.compute_from_dro

Run with:
  python -m pytest tests/test_score_pinball.py -v
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.scoring.pinball import (
    PinballResult,
    Score_Pinball,
    pinball_loss,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def n():
    return 200

@pytest.fixture
def y(n):
    rng = np.random.default_rng(42)
    return rng.normal(50, 5, size=n)

@pytest.fixture
def quantiles(y, n):
    rng = np.random.default_rng(1)
    return {
        0.1:  y - 8 + rng.normal(0, 0.5, n),
        0.25: y - 4 + rng.normal(0, 0.5, n),
        0.5:  y     + rng.normal(0, 0.5, n),
        0.75: y + 4 + rng.normal(0, 0.5, n),
        0.9:  y + 8 + rng.normal(0, 0.5, n),
    }

@pytest.fixture
def scorer():
    return Score_Pinball()

@pytest.fixture
def result(scorer, quantiles, y):
    return scorer.compute(quantiles=quantiles, y=y)


# ---------------------------------------------------------------------------
# 1. pinball_loss (element-wise)
# ---------------------------------------------------------------------------

class TestPinballLossElementwise:

    def test_perfect_forecast_at_median(self):
        y = np.array([1.0, 2.0, 3.0])
        q = np.array([1.0, 2.0, 3.0])
        loss = pinball_loss(y, q, level=0.5)
        np.testing.assert_allclose(loss, [0.0, 0.0, 0.0])

    def test_over_prediction(self):
        """q > y: loss = (q - y) * (1 - p)"""
        y = np.array([1.0])
        q = np.array([3.0])
        loss = pinball_loss(y, q, level=0.5)
        expected = (3.0 - 1.0) * 0.5
        np.testing.assert_allclose(loss, [expected])

    def test_under_prediction(self):
        """q < y: loss = (y - q) * p"""
        y = np.array([5.0])
        q = np.array([2.0])
        loss = pinball_loss(y, q, level=0.9)
        expected = (5.0 - 2.0) * 0.9
        np.testing.assert_allclose(loss, [expected])

    def test_exact_hit(self):
        """y == q: loss = 0 for any level."""
        y = np.array([4.0, 4.0])
        q = np.array([4.0, 4.0])
        loss = pinball_loss(y, q, level=0.3)
        np.testing.assert_allclose(loss, [0.0, 0.0])

    def test_non_negative(self):
        rng = np.random.default_rng(99)
        y = rng.normal(50, 5, size=100)
        q = rng.normal(50, 5, size=100)
        loss = pinball_loss(y, q, level=0.7)
        assert np.all(loss >= 0)

    def test_symmetry_at_median(self):
        """At p=0.5, loss is symmetric: |y-q| * 0.5 regardless of direction."""
        y = np.array([10.0, 10.0])
        q = np.array([8.0, 12.0])
        loss = pinball_loss(y, q, level=0.5)
        np.testing.assert_allclose(loss[0], loss[1])
        np.testing.assert_allclose(loss[0], 1.0)   # |10-8| * 0.5 = 1.0

    def test_higher_level_penalises_under_prediction_more(self):
        """p=0.9 penalises under-prediction (y > q) more than p=0.1."""
        y = np.array([5.0])
        q = np.array([3.0])   # under-prediction: y > q
        loss_90 = pinball_loss(y, q, level=0.9)
        loss_10 = pinball_loss(y, q, level=0.1)
        assert loss_90 > loss_10


# ---------------------------------------------------------------------------
# 2. Score_Pinball.compute — happy path
# ---------------------------------------------------------------------------

class TestComputeHappyPath:

    def test_returns_pinball_result(self, result):
        assert isinstance(result, PinballResult)

    def test_n_obs(self, result, n):
        assert result.n_obs == n

    def test_levels_sorted(self, result):
        assert np.all(np.diff(result.levels) > 0)

    def test_levels_correct(self, result):
        np.testing.assert_allclose(result.levels, [0.1, 0.25, 0.5, 0.75, 0.9])

    def test_loss_per_level_shape(self, result):
        assert result.loss_per_level.shape == (5,)

    def test_loss_per_level_non_negative(self, result):
        assert np.all(result.loss_per_level >= 0)

    def test_mean_pinball_non_negative(self, result):
        assert result.mean_pinball >= 0

    def test_mean_pinball_equals_mean_of_loss_per_level(self, result):
        np.testing.assert_allclose(
            result.mean_pinball,
            result.loss_matrix.mean(),
            rtol=1e-10,
        )

    def test_loss_matrix_shape(self, result, n):
        assert result.loss_matrix.shape == (n, 5)

    def test_loss_matrix_non_negative(self, result):
        assert np.all(result.loss_matrix >= 0)

    def test_no_crossings_in_clean_data(self, result):
        assert result.n_crossing_pairs == 0

    def test_no_regime_losses_by_default(self, result):
        assert result.regime_losses == {}
        assert result.regime_loss_per_level == {}

    def test_perfect_forecast_zero_loss(self, n, y):
        """Perfect quantile forecasts at every level → zero pinball loss."""
        q = {p: y.copy() for p in [0.1, 0.5, 0.9]}
        scorer = Score_Pinball()
        result = scorer.compute(quantiles=q, y=y)
        np.testing.assert_allclose(result.mean_pinball, 0.0, atol=1e-12)

    def test_loss_consistent_with_element_wise(self, y, n):
        """Score_Pinball matrix should match element-wise pinball_loss."""
        rng = np.random.default_rng(5)
        q_arr = y + rng.normal(0, 2, n)
        q = {0.5: q_arr}
        scorer = Score_Pinball()
        result = scorer.compute(quantiles=q, y=y)
        expected = pinball_loss(y, q_arr, level=0.5)
        np.testing.assert_allclose(
            result.loss_matrix[:, 0], expected, rtol=1e-10
        )


# ---------------------------------------------------------------------------
# 3. Regime stratification
# ---------------------------------------------------------------------------

class TestRegimeStratification:

    def test_regime_losses_keys(self, scorer, quantiles, y, n):
        tags = ["winter"] * (n // 2) + ["summer"] * (n - n // 2)
        result = scorer.compute(quantiles=quantiles, y=y, regime_tags=tags)
        assert set(result.regime_losses.keys()) == {"winter", "summer"}

    def test_regime_loss_per_level_keys(self, scorer, quantiles, y, n):
        tags = ["winter"] * (n // 2) + ["summer"] * (n - n // 2)
        result = scorer.compute(quantiles=quantiles, y=y, regime_tags=tags)
        assert set(result.regime_loss_per_level.keys()) == {"winter", "summer"}

    def test_regime_loss_per_level_shape(self, scorer, quantiles, y, n):
        tags = ["a"] * n
        result = scorer.compute(quantiles=quantiles, y=y, regime_tags=tags)
        assert result.regime_loss_per_level["a"].shape == (len(quantiles),)

    def test_single_regime_matches_overall(self, scorer, quantiles, y, n):
        """One regime for all obs → regime loss == overall mean pinball."""
        tags = ["all"] * n
        result = scorer.compute(quantiles=quantiles, y=y, regime_tags=tags)
        np.testing.assert_allclose(
            result.regime_losses["all"],
            result.mean_pinball,
            rtol=1e-10,
        )

    def test_three_regimes(self, scorer, quantiles, y, n):
        tags = (
            ["low"] * (n // 3)
            + ["mid"] * (n // 3)
            + ["high"] * (n - 2 * (n // 3))
        )
        result = scorer.compute(quantiles=quantiles, y=y, regime_tags=tags)
        assert len(result.regime_losses) == 3

    def test_regime_tags_wrong_length_raises(self, scorer, quantiles, y, n):
        with pytest.raises(ValueError, match="length"):
            scorer.compute(
                quantiles=quantiles, y=y,
                regime_tags=["a"] * (n - 1)
            )

    def test_numpy_array_tags_accepted(self, scorer, quantiles, y, n):
        tags = np.array(["win"] * n)
        result = scorer.compute(quantiles=quantiles, y=y, regime_tags=tags)
        assert "win" in result.regime_losses


# ---------------------------------------------------------------------------
# 4. Crossing detection
# ---------------------------------------------------------------------------

class TestCrossingDetection:

    def test_crossing_detected_and_warned(self, scorer, y, n):
        rng = np.random.default_rng(70)
        base = rng.normal(50, 5, n)
        q_cross = {
            0.1: base + 5,   # higher than 0.9 — crossing
            0.9: base - 5,
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = scorer.compute(quantiles=q_cross, y=y)
            cross_warns = [x for x in w if "crossing" in str(x.message)]
            assert len(cross_warns) >= 1
        assert result.n_crossing_pairs > 0

    def test_no_warning_with_warn_disabled(self, y, n):
        rng = np.random.default_rng(71)
        base = rng.normal(50, 5, n)
        q_cross = {0.1: base + 5, 0.9: base - 5}
        scorer_silent = Score_Pinball(warn_on_crossings=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = scorer_silent.compute(quantiles=q_cross, y=y)
            cross_warns = [x for x in w if "crossing" in str(x.message)]
            assert len(cross_warns) == 0
        assert result.n_crossing_pairs > 0   # still counted


# ---------------------------------------------------------------------------
# 5. Error conditions
# ---------------------------------------------------------------------------

class TestErrors:

    def test_empty_quantiles_raises(self, scorer, y):
        with pytest.raises(ValueError, match="empty"):
            scorer.compute(quantiles={}, y=y)

    def test_wrong_quantile_length_raises(self, scorer, y, n):
        with pytest.raises(ValueError, match="length"):
            scorer.compute(
                quantiles={0.5: np.ones(n - 1)},
                y=y,
            )

    def test_single_level_accepted(self, scorer, y, n):
        """Single quantile level is valid."""
        result = scorer.compute(quantiles={0.5: y}, y=y)
        assert result.mean_pinball == 0.0   # perfect median forecast


# ---------------------------------------------------------------------------
# 6. PinballResult.to_dict
# ---------------------------------------------------------------------------

class TestPinballResultToDict:

    def test_to_dict_keys(self, result):
        d = result.to_dict()
        required = {
            "mean_pinball", "loss_per_level", "n_obs",
            "n_crossing_pairs", "regime_losses"
        }
        assert required.issubset(set(d.keys()))

    def test_to_dict_mean_pinball_finite(self, result):
        d = result.to_dict()
        assert np.isfinite(d["mean_pinball"])

    def test_to_dict_with_regime(self, scorer, quantiles, y, n):
        tags = ["a"] * n
        result = scorer.compute(quantiles=quantiles, y=y, regime_tags=tags)
        d = result.to_dict()
        assert "a" in d["regime_losses"]

    def test_to_dict_loss_per_level_keys_are_floats(self, result):
        d = result.to_dict()
        for k in d["loss_per_level"]:
            assert isinstance(k, float)


# ---------------------------------------------------------------------------
# 7. compute_from_dro
# ---------------------------------------------------------------------------

class TestComputeFromDRO:

    def test_from_dro_matches_direct(self, scorer, quantiles, y, n):
        """compute_from_dro should give same result as direct compute."""
        from src.diagnostics.diagnostics_input import Diagnostics_Input
        import pandas as pd

        t = pd.date_range("2021-01-01", periods=n, freq="h")
        di = Diagnostics_Input()
        dro = di.from_arrays(y=y, t=t, model_id="m", quantiles=quantiles)

        result_dro    = scorer.compute_from_dro(dro)
        result_direct = scorer.compute(quantiles=quantiles, y=y)

        np.testing.assert_allclose(
            result_dro.mean_pinball,
            result_direct.mean_pinball,
            rtol=1e-10,
        )

    def test_from_dro_no_pinball_capability_raises(self, scorer, y, n):
        from src.diagnostics.diagnostics_input import Diagnostics_Input
        import pandas as pd

        t = pd.date_range("2021-01-01", periods=n, freq="h")
        di = Diagnostics_Input()
        # Only interval — no quantiles
        dro = di.from_arrays(y=y, t=t, model_id="m", lo=y-1, hi=y+1)

        from src.diagnostics.diagnostics_input import DiagnosticsInputError
        with pytest.raises(DiagnosticsInputError, match="pinball"):
            scorer.compute_from_dro(dro)

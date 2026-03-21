"""
tests/test_adapter_quantiles.py
=================================
Pytest suite for Adapter_Quantiles and QuantileFunctionObject.

Groups:
  1. PAVA isotonic regression (_pava_isotonic)
  2. Adapter_Quantiles — happy path
  3. Adapter_Quantiles — crossing detection and PAVA fix
  4. Adapter_Quantiles — error conditions
  5. Adapter_Quantiles — sanity flags and warnings
  6. QuantileFunctionObject — get_interval, to_quantiles_dict, interpolate_cdf

Run with:
  python -m pytest tests/test_adapter_quantiles.py -v
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.adapters.quantile_adapter import (
    Adapter_Quantiles,
    QuantileAdapterError,
    QuantileFunctionObject,
    _pava_isotonic,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def n():
    return 200

@pytest.fixture
def timestamps(n):
    return pd.date_range("2021-01-01", periods=n, freq="h")

@pytest.fixture
def y(n):
    rng = np.random.default_rng(10)
    return rng.normal(50, 5, size=n)

@pytest.fixture
def clean_quantiles(y, n):
    """Non-crossing quantiles centered around y."""
    rng = np.random.default_rng(20)
    return {
        0.1: y - 8 + rng.normal(0, 0.1, n),
        0.25: y - 4 + rng.normal(0, 0.1, n),
        0.5:  y + rng.normal(0, 0.1, n),
        0.75: y + 4 + rng.normal(0, 0.1, n),
        0.9:  y + 8 + rng.normal(0, 0.1, n),
    }

@pytest.fixture
def adapter():
    return Adapter_Quantiles(alpha=0.1, model_id="test_q")

@pytest.fixture
def qfo(adapter, clean_quantiles, timestamps, y):
    return adapter.transform(quantiles=clean_quantiles, t=timestamps, y=y)


# ---------------------------------------------------------------------------
# 1. PAVA isotonic regression
# ---------------------------------------------------------------------------

class TestPAVA:

    def test_already_sorted_unchanged(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _pava_isotonic(x)
        np.testing.assert_allclose(result, x)

    def test_single_violation_fixed(self):
        x = np.array([1.0, 3.0, 2.0, 4.0])
        result = _pava_isotonic(x)
        assert np.all(np.diff(result) >= 0)

    def test_all_equal_unchanged(self):
        x = np.array([5.0, 5.0, 5.0])
        result = _pava_isotonic(x)
        np.testing.assert_allclose(result, x)

    def test_fully_reversed_becomes_constant(self):
        x = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = _pava_isotonic(x)
        assert np.all(np.diff(result) >= 0)
        # PAVA on reversed array → all equal to mean
        np.testing.assert_allclose(result, np.mean(x) * np.ones(5), rtol=1e-10)

    def test_output_non_decreasing(self):
        rng = np.random.default_rng(99)
        x = rng.normal(size=50)
        result = _pava_isotonic(x)
        assert np.all(np.diff(result) >= -1e-12)

    def test_output_same_length(self):
        x = np.array([3.0, 1.0, 2.0])
        result = _pava_isotonic(x)
        assert len(result) == len(x)

    def test_single_element(self):
        x = np.array([7.0])
        result = _pava_isotonic(x)
        np.testing.assert_allclose(result, x)

    def test_two_elements_reversed(self):
        x = np.array([3.0, 1.0])
        result = _pava_isotonic(x)
        assert result[0] <= result[1]
        np.testing.assert_allclose(result[0], result[1])  # both = mean = 2.0


# ---------------------------------------------------------------------------
# 2. Adapter_Quantiles — happy path
# ---------------------------------------------------------------------------

class TestAdapterHappyPath:

    def test_returns_qfo(self, qfo):
        assert isinstance(qfo, QuantileFunctionObject)

    def test_dist_type(self, qfo):
        assert qfo.dist_type == "quantile_function"

    def test_model_id(self, qfo, adapter):
        assert qfo.model_id == adapter.model_id

    def test_n_obs(self, qfo, n):
        assert qfo.n_obs == n

    def test_levels_sorted(self, qfo):
        assert np.all(np.diff(qfo.levels) > 0)

    def test_levels_correct(self, qfo):
        expected = [0.1, 0.25, 0.5, 0.75, 0.9]
        np.testing.assert_allclose(qfo.levels, expected)

    def test_all_arrays_shape(self, qfo, n):
        for p, arr in qfo.quantile_arrays.items():
            assert arr.shape == (n,), f"Shape mismatch for p={p}"

    def test_no_crossings_in_clean_data(self, qfo):
        assert qfo.n_crossings_fixed == 0

    def test_alpha_stored(self, qfo, adapter):
        assert qfo.alpha == adapter.alpha

    def test_y_stored(self, qfo, y):
        np.testing.assert_array_equal(qfo.y, y)

    def test_without_y(self, adapter, clean_quantiles, timestamps):
        qfo = adapter.transform(quantiles=clean_quantiles, t=timestamps)
        assert qfo.y is None
        assert qfo.sanity_flags["median_coverage_ok"] is None

    def test_model_id_override(self, adapter, clean_quantiles, timestamps):
        qfo = adapter.transform(
            quantiles=clean_quantiles, t=timestamps, model_id="override"
        )
        assert qfo.model_id == "override"

    def test_numeric_timestamps_accepted(self, adapter, clean_quantiles, n):
        t_num = np.arange(n, dtype=float)
        qfo = adapter.transform(quantiles=clean_quantiles, t=t_num)
        assert qfo.n_obs == n


# ---------------------------------------------------------------------------
# 3. Crossing detection and PAVA fix
# ---------------------------------------------------------------------------

class TestCrossingFix:

    def test_crossings_detected_and_fixed(self, adapter, timestamps, n):
        rng = np.random.default_rng(55)
        base = rng.normal(50, 5, size=n)
        # Deliberately introduce crossings: q_0.9 < q_0.1 for some obs
        q = {
            0.1: base + 8,     # higher than 0.9 — crossing!
            0.5: base,
            0.9: base - 8,     # lower than 0.1 — crossing!
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            qfo = adapter.transform(quantiles=q, t=timestamps)
            crossing_warns = [x for x in w if "crossings" in str(x.message)]
            assert len(crossing_warns) >= 1

        assert qfo.n_crossings_fixed > 0

    def test_fixed_quantiles_non_crossing(self, adapter, timestamps, n):
        rng = np.random.default_rng(55)
        base = rng.normal(50, 5, size=n)
        q = {
            0.1: base + 8,
            0.5: base,
            0.9: base - 8,
        }
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            qfo = adapter.transform(quantiles=q, t=timestamps)

        # After fix, must be non-crossing
        lo = qfo.quantile_arrays[0.1]
        med = qfo.quantile_arrays[0.5]
        hi = qfo.quantile_arrays[0.9]
        assert np.all(lo <= med + 1e-10)
        assert np.all(med <= hi + 1e-10)

    def test_partial_crossings_fixed(self, adapter, timestamps, n):
        """Only some observations cross — only those should be fixed."""
        rng = np.random.default_rng(77)
        base = rng.normal(50, 2, size=n)
        lo  = base - 3
        med = base.copy()
        hi  = base + 3
        # Introduce crossing between lo and hi at index 5 and 10
        lo[5]  = hi[5]  + 1.0
        lo[10] = hi[10] + 2.0
        q = {0.1: lo, 0.5: med, 0.9: hi}
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            qfo = adapter.transform(quantiles=q, t=timestamps)
        assert qfo.n_crossings_fixed == 2


# ---------------------------------------------------------------------------
# 4. Error conditions
# ---------------------------------------------------------------------------

class TestErrors:

    def test_too_few_levels_raises(self, adapter, timestamps, n):
        rng = np.random.default_rng(1)
        base = rng.normal(50, 5, n)
        with pytest.raises(QuantileAdapterError, match="levels required"):
            adapter.transform(
                quantiles={0.5: base},   # only 1 level, min=3
                t=timestamps,
            )

    def test_level_out_of_01_raises(self, adapter, timestamps, n):
        rng = np.random.default_rng(2)
        base = rng.normal(50, 5, n)
        with pytest.raises(QuantileAdapterError, match="out of"):
            adapter.transform(
                quantiles={0.0: base, 0.5: base, 1.0: base},
                t=timestamps,
            )

    def test_nan_in_array_raises(self, adapter, timestamps, n):
        rng = np.random.default_rng(3)
        base = rng.normal(50, 5, n)
        bad = base.copy()
        bad[7] = np.nan
        with pytest.raises(QuantileAdapterError, match="NaN or Inf"):
            adapter.transform(
                quantiles={0.1: bad, 0.5: base, 0.9: base + 5},
                t=timestamps,
            )

    def test_mismatched_lengths_raises(self, adapter, timestamps, n):
        rng = np.random.default_rng(4)
        base = rng.normal(50, 5, n)
        short = rng.normal(50, 5, n - 10)
        with pytest.raises(QuantileAdapterError, match="same length"):
            adapter.transform(
                quantiles={0.1: short, 0.5: base, 0.9: base + 5},
                t=timestamps,
            )

    def test_t_wrong_length_raises(self, adapter, clean_quantiles, n):
        t_short = pd.date_range("2021-01-01", periods=n - 5, freq="h")
        with pytest.raises(QuantileAdapterError, match="length"):
            adapter.transform(quantiles=clean_quantiles, t=t_short)

    def test_y_wrong_shape_raises(self, adapter, clean_quantiles, timestamps, n):
        y_bad = np.ones(n - 1)
        with pytest.raises(QuantileAdapterError, match="shape"):
            adapter.transform(
                quantiles=clean_quantiles, t=timestamps, y=y_bad
            )

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            Adapter_Quantiles(alpha=1.1)

    def test_min_levels_below_2_raises(self):
        with pytest.raises(ValueError, match="min_levels"):
            Adapter_Quantiles(min_levels=1)

    def test_2d_quantile_array_raises(self, adapter, timestamps, n):
        rng = np.random.default_rng(5)
        base_2d = rng.normal(size=(n, 2))
        with pytest.raises(QuantileAdapterError, match="1-dimensional"):
            adapter.transform(
                quantiles={0.1: base_2d, 0.5: np.ones(n), 0.9: np.ones(n)},
                t=timestamps,
            )


# ---------------------------------------------------------------------------
# 5. Sanity flags and warnings
# ---------------------------------------------------------------------------

class TestSanityFlags:

    def test_median_calibration_ok_for_well_specified(self, qfo):
        # y was drawn close to median — should pass
        assert qfo.sanity_flags.get("median_coverage_ok") is True

    def test_median_calibration_fails_for_biased(self, adapter, timestamps, n):
        rng = np.random.default_rng(60)
        y_actual = rng.normal(50, 5, n)
        # Shift median 20 units above actuals → P(y ≤ q_0.5) ≈ 0
        q = {
            0.1: y_actual + 15,
            0.5: y_actual + 20,    # grossly over-estimates median
            0.9: y_actual + 25,
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            qfo = adapter.transform(quantiles=q, t=timestamps, y=y_actual)
            median_warns = [x for x in w if "Median calibration" in str(x.message)]
            assert len(median_warns) >= 1
        assert qfo.sanity_flags["median_coverage_ok"] is False

    def test_jump_flag_detected(self, adapter, timestamps, n):
        rng = np.random.default_rng(70)
        base = rng.normal(50, 1, n)
        # Introduce a huge jump between 0.5 and 0.75 for all observations
        q = {
            0.1: base - 1,
            0.5: base,
            0.75: base + 1000.0,   # enormous jump
            0.9: base + 1001.0,
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            qfo = adapter.transform(quantiles=q, t=timestamps)
            jump_warns = [x for x in w if "discontinuous jumps" in str(x.message)]
            assert len(jump_warns) >= 1
        assert qfo.sanity_flags["n_jump_flagged"] > 0

    def test_no_jump_for_uniform_gaps(self, qfo):
        assert qfo.sanity_flags["n_jump_flagged"] == 0


# ---------------------------------------------------------------------------
# 6. QuantileFunctionObject methods
# ---------------------------------------------------------------------------

class TestQuantileFunctionObject:

    def test_get_interval_exact_levels(self, qfo):
        lo, hi = qfo.get_interval(alpha=0.1)
        assert lo.shape == (qfo.n_obs,)
        assert hi.shape == (qfo.n_obs,)
        assert np.all(lo <= hi)

    def test_get_interval_default_alpha(self, qfo):
        """alpha=0.1 → seeks 0.05 and 0.95; fixture has 0.1 and 0.9 as nearest."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            lo, hi = qfo.get_interval()
        # Nearest to 0.05 is 0.1; nearest to 0.95 is 0.9
        np.testing.assert_array_equal(lo, qfo.quantile_arrays[0.1])
        np.testing.assert_array_equal(hi, qfo.quantile_arrays[0.9])

    def test_get_interval_nearest_warn(self, adapter, timestamps, n):
        """If alpha/2 level not present, warn and use nearest."""
        rng = np.random.default_rng(80)
        base = rng.normal(50, 5, n)
        q = {0.1: base - 5, 0.5: base, 0.9: base + 5}
        qfo = adapter.transform(quantiles=q, t=timestamps)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            lo, hi = qfo.get_interval(alpha=0.2)   # 0.1 and 0.9 not exact
            # Should warn about nearest level
        assert lo.shape == (n,)
        assert hi.shape == (n,)

    def test_to_quantiles_dict_keys(self, qfo):
        d = qfo.to_quantiles_dict()
        assert set(d.keys()) == set(qfo.quantile_arrays.keys())

    def test_to_quantiles_dict_compatible_shape(self, qfo, n):
        d = qfo.to_quantiles_dict()
        for arr in d.values():
            assert arr.shape == (n,)

    def test_interpolate_cdf_range(self, qfo, y):
        """CDF values must be in [0, 1]."""
        cdf = qfo.interpolate_cdf(y)
        assert cdf.shape == (qfo.n_obs,)
        assert np.all(cdf >= 0.0)
        assert np.all(cdf <= 1.0)

    def test_interpolate_cdf_monotone_at_single_obs(self, qfo):
        """For a sequence of increasing y values at obs 0, CDF should increase."""
        q_vals = [qfo.quantile_arrays[p][0] for p in qfo.levels]
        y_increasing = np.array([
            q_vals[0] - 10,    # below q_0.1
            q_vals[0],         # at q_0.1
            q_vals[2],         # at q_0.5
            q_vals[-1],        # at q_0.9
            q_vals[-1] + 10,   # above q_0.9
        ])
        # Build a minimal QFO with 5 obs all equal to obs 0
        q_rep = {p: np.full(5, qfo.quantile_arrays[p][0])
                 for p in qfo.levels}
        adapter = Adapter_Quantiles(alpha=0.1)
        t_rep = np.arange(5)
        qfo_rep = adapter.transform(quantiles=q_rep, t=t_rep)
        cdf = qfo_rep.interpolate_cdf(y_increasing)
        assert np.all(np.diff(cdf) >= -1e-10), "CDF must be non-decreasing"

    def test_interpolate_cdf_wrong_shape_raises(self, qfo):
        with pytest.raises(ValueError, match="shape"):
            qfo.interpolate_cdf(np.ones(qfo.n_obs + 1))

    def test_summary_keys(self, qfo):
        s = qfo.summary()
        required = {
            "dist_type", "model_id", "n_obs", "n_levels",
            "levels", "n_crossings_fixed", "alpha",
            "median_coverage_ok", "n_jump_flagged",
        }
        assert required.issubset(set(s.keys()))

    def test_summary_values(self, qfo, n):
        s = qfo.summary()
        assert s["dist_type"] == "quantile_function"
        assert s["n_obs"] == n
        assert s["n_levels"] == 5
        assert s["n_crossings_fixed"] == 0

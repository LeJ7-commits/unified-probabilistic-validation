"""
tests/test_adapter_point_forecast.py
======================================
Pytest suite for Adapter_PointForecast, ResidualPool, and bucket functions.

Groups:
  1. Bucket functions
  2. Adapter_PointForecast — happy path
  3. Adapter_PointForecast — error conditions
  4. Adapter_PointForecast — sanity flags and warnings
  5. ResidualPool properties and to_quantiles()

Run with:
  python -m pytest tests/test_adapter_point_forecast.py -v
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.core.data_contract import DataContract
from src.adapters.point_forecast import (
    Adapter_PointForecast,
    AdapterError,
    ResidualPool,
    bucket_coarse_4,
    bucket_hourly_24,
    bucket_none,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def contract():
    return DataContract(min_samples=10, min_obs=2)


@pytest.fixture
def n():
    return 5000  # enough for warmup + meaningful pool


@pytest.fixture
def timestamps(n):
    return pd.date_range("2020-01-01", periods=n, freq="h")


@pytest.fixture
def y(n):
    rng = np.random.default_rng(0)
    return rng.normal(50, 5, size=n)


@pytest.fixture
def y_hat(y):
    rng = np.random.default_rng(1)
    return y + rng.normal(0, 1, size=len(y))


@pytest.fixture
def std_obj(contract, timestamps, y, y_hat):
    return contract.validate(
        t=timestamps, y=y, model_id="test_pf", split="window_0", y_hat=y_hat
    )


@pytest.fixture
def adapter():
    return Adapter_PointForecast(
        W=100,
        N_min_hard=20,
        N_min_soft=40,
        bucket_fn=bucket_hourly_24,
    )


# ---------------------------------------------------------------------------
# 1. Bucket functions
# ---------------------------------------------------------------------------

class TestBucketFunctions:

    def test_bucket_hourly_24_range(self, timestamps):
        ids = bucket_hourly_24(timestamps.values)
        assert ids.min() >= 0
        assert ids.max() <= 23
        assert len(ids) == len(timestamps)

    def test_bucket_hourly_24_correct_hours(self):
        t = pd.date_range("2020-01-01 00:00", periods=24, freq="h")
        ids = bucket_hourly_24(t.values)
        np.testing.assert_array_equal(ids, np.arange(24))

    def test_bucket_coarse_4_range(self, timestamps):
        ids = bucket_coarse_4(timestamps.values)
        assert set(ids).issubset({0, 1, 2, 3})
        assert len(ids) == len(timestamps)

    def test_bucket_coarse_4_boundaries(self):
        t = pd.DatetimeIndex([
            "2020-01-01 00:00",  # Night -> 0
            "2020-01-01 05:59",  # Night -> 0
            "2020-01-01 06:00",  # Morning -> 1
            "2020-01-01 11:59",  # Morning -> 1
            "2020-01-01 12:00",  # Afternoon -> 2
            "2020-01-01 17:59",  # Afternoon -> 2
            "2020-01-01 18:00",  # Evening -> 3
            "2020-01-01 23:59",  # Evening -> 3
        ])
        ids = bucket_coarse_4(t.values)
        expected = [0, 0, 1, 1, 2, 2, 3, 3]
        np.testing.assert_array_equal(ids, expected)

    def test_bucket_none_all_zeros(self, timestamps):
        ids = bucket_none(timestamps.values)
        assert np.all(ids == 0)
        assert len(ids) == len(timestamps)

    def test_bucket_none_shape(self, timestamps):
        ids = bucket_none(timestamps.values)
        assert ids.shape == (len(timestamps),)


# ---------------------------------------------------------------------------
# 2. Adapter_PointForecast — happy path
# ---------------------------------------------------------------------------

class TestAdapterHappyPath:

    def test_transform_returns_residual_pool(self, adapter, std_obj):
        pool = adapter.transform(std_obj)
        assert isinstance(pool, ResidualPool)

    def test_dist_type(self, adapter, std_obj):
        pool = adapter.transform(std_obj)
        assert pool.dist_type == "residual_reconstruction"

    def test_model_id_forwarded(self, adapter, std_obj):
        pool = adapter.transform(std_obj)
        assert pool.model_id == std_obj.model_id

    def test_n_obs_less_than_input(self, adapter, std_obj):
        """Warmup burn reduces evaluable observations."""
        pool = adapter.transform(std_obj)
        assert pool.n_obs < std_obj.n_obs
        assert pool.n_obs > 0

    def test_lo_le_hi(self, adapter, std_obj):
        pool = adapter.transform(std_obj)
        assert np.all(pool.pool_lo <= pool.pool_hi)

    def test_pool_sizes_ge_n_min_hard(self, adapter, std_obj):
        pool = adapter.transform(std_obj)
        assert np.all(pool.pool_sizes >= adapter.N_min_hard)

    def test_shapes_consistent(self, adapter, std_obj):
        pool = adapter.transform(std_obj)
        n = pool.n_obs
        assert pool.y_eval.shape == (n,)
        assert pool.y_hat_eval.shape == (n,)
        assert pool.residuals_eval.shape == (n,)
        assert pool.pool_lo.shape == (n,)
        assert pool.pool_hi.shape == (n,)
        assert pool.pool_bias.shape == (n,)
        assert pool.pool_scale.shape == (n,)
        assert pool.pool_sizes.shape == (n,)
        assert pool.bucket_ids.shape == (n,)

    def test_residuals_correct(self, adapter, std_obj):
        pool = adapter.transform(std_obj)
        expected = pool.y_eval - pool.y_hat_eval
        np.testing.assert_allclose(pool.residuals_eval, expected, rtol=1e-10)

    def test_alpha_stored(self, adapter, std_obj):
        pool = adapter.transform(std_obj)
        assert pool.alpha == adapter.alpha

    def test_W_stored(self, adapter, std_obj):
        pool = adapter.transform(std_obj)
        assert pool.W == adapter.W

    def test_bucket_fn_name_stored(self, adapter, std_obj):
        pool = adapter.transform(std_obj)
        assert pool.bucket_fn_name == "bucket_hourly_24"

    def test_bucket_none_produces_more_obs(self, contract, timestamps, y, y_hat):
        """bucket_none pools all past residuals, warmup is shorter."""
        obj = contract.validate(
            t=timestamps, y=y, model_id="m", split="window_0", y_hat=y_hat
        )
        adapter_none = Adapter_PointForecast(
            W=100, N_min_hard=20, bucket_fn=bucket_none
        )
        adapter_24 = Adapter_PointForecast(
            W=100, N_min_hard=20, bucket_fn=bucket_hourly_24
        )
        pool_none = adapter_none.transform(obj)
        pool_24   = adapter_24.transform(obj)
        # bucket_none warms up faster — more evaluable obs
        assert pool_none.n_obs >= pool_24.n_obs

    def test_coarse_4_bucket_ids_in_range(self, contract, timestamps, y, y_hat):
        obj = contract.validate(
            t=timestamps, y=y, model_id="m", split="window_0", y_hat=y_hat
        )
        adapter = Adapter_PointForecast(
            W=100, N_min_hard=20, bucket_fn=bucket_coarse_4
        )
        pool = adapter.transform(obj)
        assert set(pool.bucket_ids).issubset({0, 1, 2, 3})

    def test_no_bias_correction_option(self, contract, timestamps, y, y_hat):
        obj = contract.validate(
            t=timestamps, y=y, model_id="m", split="window_0", y_hat=y_hat
        )
        adapter = Adapter_PointForecast(
            W=100, N_min_hard=20,
            bucket_fn=bucket_none,
            apply_bias_correction=False,
        )
        pool = adapter.transform(obj)
        assert pool.n_obs > 0


# ---------------------------------------------------------------------------
# 3. Adapter_PointForecast — error conditions
# ---------------------------------------------------------------------------

class TestAdapterErrors:

    def test_no_y_hat_raises(self, adapter, contract, timestamps, y):
        obj_no_yhat = contract.validate(
            t=timestamps, y=y, model_id="m", split="train"
        )
        with pytest.raises(AdapterError, match="requires y_hat"):
            adapter.transform(obj_no_yhat)

    def test_insufficient_data_raises(self, contract):
        """Very short series — can't build any pool."""
        n = 5
        t = pd.date_range("2020-01-01", periods=n, freq="h")
        y = np.ones(n)
        y_hat = np.ones(n)
        obj = contract.validate(
            t=t, y=y, model_id="short", split="train", y_hat=y_hat
        )
        adapter = Adapter_PointForecast(W=100, N_min_hard=50, bucket_fn=bucket_none)
        with pytest.raises(AdapterError, match="No evaluable observations"):
            adapter.transform(obj)

    def test_invalid_W_raises(self):
        with pytest.raises(ValueError, match="W must be positive"):
            Adapter_PointForecast(W=0)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha must be in"):
            Adapter_PointForecast(alpha=1.5)

    def test_invalid_N_min_hard_raises(self):
        with pytest.raises(ValueError, match="N_min_hard must be positive"):
            Adapter_PointForecast(N_min_hard=0)

    def test_bucket_fn_wrong_shape_raises(self, std_obj):
        def bad_bucket(t):
            return np.zeros(5, dtype=int)   # wrong length

        adapter = Adapter_PointForecast(
            W=100, N_min_hard=20, bucket_fn=bad_bucket
        )
        with pytest.raises(AdapterError, match="shape"):
            adapter.transform(std_obj)

    def test_bucket_fn_exception_raises(self, std_obj):
        def exploding_bucket(t):
            raise RuntimeError("Intentional error")

        adapter = Adapter_PointForecast(
            W=100, N_min_hard=20, bucket_fn=exploding_bucket
        )
        with pytest.raises(AdapterError, match="bucket_fn raised"):
            adapter.transform(std_obj)


# ---------------------------------------------------------------------------
# 4. Sanity flags and warnings
# ---------------------------------------------------------------------------

class TestSanityFlagsAndWarnings:

    def test_sanity_flags_keys_present(self, adapter, std_obj):
        pool = adapter.transform(std_obj)
        assert "n_bias_flagged" in pool.sanity_flags
        assert "n_break_flagged" in pool.sanity_flags
        assert "variance_ratio" in pool.sanity_flags
        assert "break_flag" in pool.sanity_flags

    def test_well_specified_no_bias_flags(self, contract):
        """Zero-mean residuals → no bias flags."""
        n = 5000
        t = pd.date_range("2020-01-01", periods=n, freq="h")
        y = np.ones(n) * 50
        y_hat = np.ones(n) * 50   # perfect forecast → zero residuals
        obj = contract.validate(
            t=t, y=y, model_id="perfect", split="train", y_hat=y_hat
        )
        adapter = Adapter_PointForecast(
            W=100, N_min_hard=20,
            bucket_fn=bucket_none,
            bias_tol=0.3,
        )
        pool = adapter.transform(obj)
        assert pool.sanity_flags["n_bias_flagged"] == 0

    def test_soft_minimum_warns(self, contract, timestamps, y, y_hat):
        obj = contract.validate(
            t=timestamps, y=y, model_id="m", split="train", y_hat=y_hat
        )
        adapter = Adapter_PointForecast(
            W=10,          # tiny pool
            N_min_hard=5,
            N_min_soft=500,  # unreachably high soft threshold
            bucket_fn=bucket_none,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pool = adapter.transform(obj)
            soft_warns = [x for x in w if "soft threshold" in str(x.message)]
            assert len(soft_warns) >= 1

    def test_break_flag_detected_for_heteroscedastic(self, contract):
        """Residuals with huge variance shift → break flag raised."""
        n = 2000
        t = pd.date_range("2020-01-01", periods=n, freq="h")
        rng = np.random.default_rng(99)
        y = rng.normal(50, 1, size=n)
        # First half: low noise, second half: very high noise
        noise = np.concatenate([
            rng.normal(0, 0.1, size=n // 2),
            rng.normal(0, 20.0, size=n - n // 2),
        ])
        y_hat = y - noise
        obj = contract.validate(
            t=t, y=y, model_id="het", split="train", y_hat=y_hat
        )
        adapter = Adapter_PointForecast(
            W=50, N_min_hard=20,
            bucket_fn=bucket_none,
            break_var_ratio=3.0,
        )
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            pool = adapter.transform(obj)
        assert pool.sanity_flags["break_flag"] is True


# ---------------------------------------------------------------------------
# 5. ResidualPool properties and to_quantiles()
# ---------------------------------------------------------------------------

class TestResidualPool:

    def test_to_quantiles_keys(self, adapter, std_obj):
        pool = adapter.transform(std_obj)
        q = pool.to_quantiles()
        expected_keys = {adapter.alpha / 2, 1 - adapter.alpha / 2}
        assert set(q.keys()) == expected_keys

    def test_to_quantiles_shapes(self, adapter, std_obj):
        pool = adapter.transform(std_obj)
        q = pool.to_quantiles()
        for arr in q.values():
            assert arr.shape == (pool.n_obs,)

    def test_to_quantiles_lo_le_hi(self, adapter, std_obj):
        pool = adapter.transform(std_obj)
        q = pool.to_quantiles()
        lo = q[adapter.alpha / 2]
        hi = q[1 - adapter.alpha / 2]
        assert np.all(lo <= hi)

    def test_summary_keys(self, adapter, std_obj):
        pool = adapter.transform(std_obj)
        s = pool.summary()
        required = {
            "dist_type", "model_id", "n_obs", "alpha",
            "W", "bucket_fn_name", "mean_pool_size",
            "mean_bias", "n_bias_flagged", "n_break_flagged",
        }
        assert required.issubset(set(s.keys()))

    def test_summary_values_correct(self, adapter, std_obj):
        pool = adapter.transform(std_obj)
        s = pool.summary()
        assert s["dist_type"] == "residual_reconstruction"
        assert s["model_id"] == std_obj.model_id
        assert s["n_obs"] == pool.n_obs
        assert s["alpha"] == adapter.alpha
        assert s["W"] == adapter.W

    def test_pool_scale_positive(self, adapter, std_obj):
        pool = adapter.transform(std_obj)
        assert np.all(pool.pool_scale > 0)

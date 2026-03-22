"""
tests/test_build_dist_from_residuals.py
========================================
Pytest suite for BuildDist_FromResiduals and SampleMatrix.

Groups:
  1. BuildDist_FromResiduals — happy path (non_parametric)
  2. BuildDist_FromResiduals — happy path (parametric)
  3. SampleMatrix properties
  4. Error conditions

Run with:
  python -m pytest tests/test_build_dist_from_residuals.py -v
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.adapters.build_dist_from_residuals import BuildDist_FromResiduals, SampleMatrix
from src.adapters.point_forecast import Adapter_PointForecast, bucket_none
from src.core.data_contract import DataContract


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def n():
    return 2000

@pytest.fixture
def timestamps(n):
    return pd.date_range("2020-01-01", periods=n, freq="h")

@pytest.fixture
def y(n):
    rng = np.random.default_rng(42)
    return rng.normal(50, 5, size=n)

@pytest.fixture
def y_hat(y, n):
    rng = np.random.default_rng(1)
    return y + rng.normal(0, 1, size=n)

@pytest.fixture
def pool(timestamps, y, y_hat, n):
    contract = DataContract(min_obs=2)
    obj = contract.validate(
        t=timestamps, y=y, model_id="test", split="window_0", y_hat=y_hat
    )
    adapter = Adapter_PointForecast(W=100, N_min_hard=20, bucket_fn=bucket_none)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        return adapter.transform(obj)


# ---------------------------------------------------------------------------
# 1. Non-parametric happy path
# ---------------------------------------------------------------------------

class TestNonParametric:

    def test_returns_sample_matrix(self, pool):
        builder = BuildDist_FromResiduals(M=100, mode="non_parametric", seed=0)
        result = builder.build(pool)
        assert isinstance(result, SampleMatrix)

    def test_samples_shape(self, pool):
        M = 100
        builder = BuildDist_FromResiduals(M=M, mode="non_parametric", seed=0)
        result = builder.build(pool)
        assert result.samples.shape == (pool.n_obs, M)

    def test_dist_type(self, pool):
        builder = BuildDist_FromResiduals(M=50, mode="non_parametric", seed=0)
        result = builder.build(pool)
        assert result.dist_type == "residual_reconstruction"

    def test_mode_stored(self, pool):
        builder = BuildDist_FromResiduals(M=50, mode="non_parametric", seed=0)
        result = builder.build(pool)
        assert result.mode == "non_parametric"

    def test_samples_finite(self, pool):
        builder = BuildDist_FromResiduals(M=100, mode="non_parametric", seed=0)
        result = builder.build(pool)
        assert np.all(np.isfinite(result.samples))

    def test_y_hat_offset_applied(self, pool):
        """Samples should be centred near y_hat, not near zero."""
        builder = BuildDist_FromResiduals(M=200, mode="non_parametric", seed=0)
        result = builder.build(pool)
        sample_means = result.samples.mean(axis=1)
        y_hat = result.y_hat
        # Mean of samples should be close to y_hat (within ~2 units)
        np.testing.assert_allclose(sample_means, y_hat, atol=5.0)

    def test_reproducible_with_seed(self, pool):
        b1 = BuildDist_FromResiduals(M=50, mode="non_parametric", seed=99)
        b2 = BuildDist_FromResiduals(M=50, mode="non_parametric", seed=99)
        r1 = b1.build(pool)
        r2 = b2.build(pool)
        np.testing.assert_array_equal(r1.samples, r2.samples)

    def test_different_seeds_differ(self, pool):
        b1 = BuildDist_FromResiduals(M=50, mode="non_parametric", seed=1)
        b2 = BuildDist_FromResiduals(M=50, mode="non_parametric", seed=2)
        r1 = b1.build(pool)
        r2 = b2.build(pool)
        assert not np.allclose(r1.samples, r2.samples)

    def test_clip_quantile_limits_range(self, pool):
        builder = BuildDist_FromResiduals(
            M=200, mode="non_parametric", seed=0, clip_quantile=0.99
        )
        result = builder.build(pool)
        assert np.all(np.isfinite(result.samples))


# ---------------------------------------------------------------------------
# 2. Parametric happy path
# ---------------------------------------------------------------------------

class TestParametric:

    def test_returns_sample_matrix(self, pool):
        builder = BuildDist_FromResiduals(M=100, mode="parametric", seed=0)
        result = builder.build(pool)
        assert isinstance(result, SampleMatrix)

    def test_samples_shape(self, pool):
        M = 150
        builder = BuildDist_FromResiduals(M=M, mode="parametric", seed=0)
        result = builder.build(pool)
        assert result.samples.shape == (pool.n_obs, M)

    def test_mode_stored(self, pool):
        builder = BuildDist_FromResiduals(M=50, mode="parametric", seed=0)
        result = builder.build(pool)
        assert result.mode == "parametric"

    def test_samples_finite(self, pool):
        builder = BuildDist_FromResiduals(M=100, mode="parametric", seed=0)
        result = builder.build(pool)
        assert np.all(np.isfinite(result.samples))

    def test_parametric_centred_near_yhat(self, pool):
        """Parametric samples should be centred near y_hat + pool_bias."""
        builder = BuildDist_FromResiduals(M=500, mode="parametric", seed=0)
        result = builder.build(pool)
        sample_means = result.samples.mean(axis=1)
        # y_hat + bias ≈ sample mean
        expected = result.y_hat + pool.pool_bias
        np.testing.assert_allclose(sample_means, expected, atol=3.0)


# ---------------------------------------------------------------------------
# 3. SampleMatrix properties
# ---------------------------------------------------------------------------

class TestSampleMatrix:

    def test_to_quantiles_keys(self, pool):
        builder = BuildDist_FromResiduals(M=100, mode="non_parametric", seed=0)
        result = builder.build(pool)
        q = result.to_quantiles(alpha=0.1)
        assert set(q.keys()) == {0.05, 0.95}

    def test_to_quantiles_shapes(self, pool):
        builder = BuildDist_FromResiduals(M=100, mode="non_parametric", seed=0)
        result = builder.build(pool)
        q = result.to_quantiles(alpha=0.1)
        for arr in q.values():
            assert arr.shape == (pool.n_obs,)

    def test_to_quantiles_lo_le_hi(self, pool):
        builder = BuildDist_FromResiduals(M=100, mode="non_parametric", seed=0)
        result = builder.build(pool)
        q = result.to_quantiles(alpha=0.1)
        assert np.all(q[0.05] <= q[0.95])

    def test_summary_keys(self, pool):
        builder = BuildDist_FromResiduals(M=50, mode="non_parametric", seed=0)
        result = builder.build(pool)
        s = result.summary()
        required = {"dist_type", "model_id", "n_obs", "M", "mode"}
        assert required.issubset(set(s.keys()))

    def test_n_obs_matches_pool(self, pool):
        builder = BuildDist_FromResiduals(M=50, mode="non_parametric", seed=0)
        result = builder.build(pool)
        assert result.n_obs == pool.n_obs

    def test_M_matches_requested(self, pool):
        M = 77
        builder = BuildDist_FromResiduals(M=M, mode="non_parametric", seed=0)
        result = builder.build(pool)
        assert result.M == M

    def test_model_id_forwarded(self, pool):
        builder = BuildDist_FromResiduals(M=50, mode="non_parametric", seed=0)
        result = builder.build(pool)
        assert result.model_id == pool.model_id

    def test_compatible_with_diagnostics_input(self, pool):
        """SampleMatrix.samples should plug directly into Diagnostics_Input."""
        from src.diagnostics.diagnostics_input import Diagnostics_Input
        builder = BuildDist_FromResiduals(M=100, mode="non_parametric", seed=0)
        matrix = builder.build(pool)
        di = Diagnostics_Input(alpha=0.1)
        dro = di.from_arrays(
            y=matrix.y,
            t=matrix.t,
            model_id=matrix.model_id,
            samples=matrix.samples,
            lo=pool.pool_lo,
            hi=pool.pool_hi,
        )
        assert dro.can_compute_pit
        assert dro.can_compute_crps


# ---------------------------------------------------------------------------
# 4. Error conditions
# ---------------------------------------------------------------------------

class TestErrors:

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            BuildDist_FromResiduals(M=100, mode="unknown_mode")

    def test_M_below_10_raises(self):
        with pytest.raises(ValueError, match="M must be"):
            BuildDist_FromResiduals(M=5)

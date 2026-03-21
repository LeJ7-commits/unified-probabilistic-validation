"""
tests/test_diagnostics_input.py
=================================
Pytest suite for Diagnostics_Input and DiagnosticsReadyObject.

Groups:
  1. DiagnosticsReadyObject capabilities
  2. Diagnostics_Input.from_adapter — ResidualPool
  3. Diagnostics_Input.from_adapter — MarginalSamples
  4. Diagnostics_Input.from_adapter — QuantileFunctionObject
  5. Diagnostics_Input.from_adapter — JointSimulationObject
  6. Diagnostics_Input.from_arrays — raw arrays
  7. Error conditions (no representation, type errors, alignment)

Run with:
  python -m pytest tests/test_diagnostics_input.py -v
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.adapters.point_forecast import Adapter_PointForecast, bucket_none
from src.adapters.quantile_adapter import Adapter_Quantiles
from src.adapters.simulation_joint import Adapter_SimulationJoint
from src.core.data_contract import DataContract
from src.diagnostics.diagnostics_input import (
    Diagnostics_Input,
    DiagnosticsInputError,
    DiagnosticsReadyObject,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def n():
    return 300

@pytest.fixture
def timestamps(n):
    return pd.date_range("2021-01-01", periods=n, freq="h")

@pytest.fixture
def y(n):
    rng = np.random.default_rng(42)
    return rng.normal(50, 5, size=n)

@pytest.fixture
def di():
    return Diagnostics_Input(alpha=0.1)

@pytest.fixture
def contract():
    return DataContract(min_samples=10, min_obs=2)

# ── ResidualPool fixture ───────────────────────────────────────────────────

@pytest.fixture
def residual_pool(contract, timestamps, y, n):
    y_hat = y + np.random.default_rng(1).normal(0, 1, n)
    obj = contract.validate(
        t=timestamps, y=y, model_id="pool_model",
        split="window_0", y_hat=y_hat
    )
    adapter = Adapter_PointForecast(W=50, N_min_hard=20, bucket_fn=bucket_none)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        return adapter.transform(obj)

# ── MarginalSamples fixture ────────────────────────────────────────────────

@pytest.fixture
def marginal_samples(n, timestamps, y):
    rng = np.random.default_rng(2)
    S = rng.normal(50, 5, size=(n, 200))
    adapter = Adapter_SimulationJoint(
        variable_names=["price"], M_min=100, model_id="sim_marginal"
    )
    joint = adapter.from_array(S=S[:, :, np.newaxis], y=y.reshape(-1, 1),
                               t=timestamps)
    return joint.get_marginal("price")

# ── QuantileFunctionObject fixture ─────────────────────────────────────────

@pytest.fixture
def qfo(n, timestamps, y):
    rng = np.random.default_rng(3)
    base = y
    q = {
        0.1:  base - 8 + rng.normal(0, 0.1, n),
        0.5:  base      + rng.normal(0, 0.1, n),
        0.9:  base + 8  + rng.normal(0, 0.1, n),
    }
    adapter = Adapter_Quantiles(alpha=0.1, model_id="quant_model")
    return adapter.transform(quantiles=q, t=timestamps, y=y)

# ── JointSimulationObject fixture ──────────────────────────────────────────

@pytest.fixture
def joint_obj(n, timestamps, y):
    rng = np.random.default_rng(4)
    d = 3
    S = rng.normal(50, 5, size=(n, 200, d))
    y_2d = rng.normal(50, 5, size=(n, d))
    adapter = Adapter_SimulationJoint(
        variable_names=["price", "temp", "gas"],
        M_min=100, model_id="joint_model"
    )
    return adapter.from_array(S=S, y=y_2d, t=timestamps)


# ---------------------------------------------------------------------------
# 1. DiagnosticsReadyObject capabilities
# ---------------------------------------------------------------------------

class TestCapabilities:

    def test_samples_only_enables_pit_crps(self, n, timestamps, y):
        rng = np.random.default_rng(10)
        S = rng.normal(size=(n, 100))
        di = Diagnostics_Input()
        obj = di.from_arrays(
            y=y, t=timestamps, model_id="m", samples=S
        )
        assert obj.can_compute_pit
        assert obj.can_compute_crps
        assert not obj.can_compute_pinball
        assert not obj.can_compute_interval
        assert not obj.can_compute_energy_score

    def test_interval_only_enables_interval(self, n, timestamps, y):
        di = Diagnostics_Input()
        obj = di.from_arrays(
            y=y, t=timestamps, model_id="m",
            lo=y - 5, hi=y + 5
        )
        assert obj.can_compute_interval
        assert not obj.can_compute_pit
        assert not obj.can_compute_crps

    def test_quantiles_only_enables_pinball(self, n, timestamps, y):
        di = Diagnostics_Input()
        obj = di.from_arrays(
            y=y, t=timestamps, model_id="m",
            quantiles={0.1: y - 5, 0.9: y + 5}
        )
        assert obj.can_compute_pinball
        assert not obj.can_compute_pit

    def test_cdf_fn_enables_pit(self, n, timestamps, y):
        di = Diagnostics_Input()
        obj = di.from_arrays(
            y=y, t=timestamps, model_id="m",
            lo=y - 5, hi=y + 5,
            cdf_fn=lambda x: np.clip(x / 100, 0, 1)
        )
        assert obj.can_compute_pit

    def test_joint_2d_enables_energy_score(self, n, timestamps, y):
        rng = np.random.default_rng(11)
        S_joint = rng.normal(size=(n, 100, 2))
        S_marg  = S_joint[:, :, 0]
        di = Diagnostics_Input()
        obj = di.from_arrays(
            y=y, t=timestamps, model_id="m",
            samples=S_marg,
            joint_samples=S_joint,
            variable_names=["a", "b"]
        )
        assert obj.can_compute_energy_score

    def test_joint_1d_does_not_enable_energy_score(self, n, timestamps, y):
        rng = np.random.default_rng(12)
        S_joint = rng.normal(size=(n, 100, 1))
        S_marg  = S_joint[:, :, 0]
        di = Diagnostics_Input()
        obj = di.from_arrays(
            y=y, t=timestamps, model_id="m",
            samples=S_marg,
            joint_samples=S_joint,
        )
        assert not obj.can_compute_energy_score

    def test_capabilities_dict_keys(self, n, timestamps, y):
        di = Diagnostics_Input()
        obj = di.from_arrays(
            y=y, t=timestamps, model_id="m", lo=y-1, hi=y+1
        )
        assert set(obj.capabilities.keys()) == {
            "pit", "crps", "pinball", "interval", "energy_score"
        }

    def test_require_raises_for_missing(self, n, timestamps, y):
        di = Diagnostics_Input()
        obj = di.from_arrays(
            y=y, t=timestamps, model_id="m", lo=y-1, hi=y+1
        )
        with pytest.raises(DiagnosticsInputError, match="crps"):
            obj.require("crps")

    def test_require_passes_for_available(self, n, timestamps, y):
        rng = np.random.default_rng(13)
        S = rng.normal(size=(n, 100))
        di = Diagnostics_Input()
        obj = di.from_arrays(
            y=y, t=timestamps, model_id="m", samples=S
        )
        obj.require("crps")   # should not raise

    def test_require_unknown_capability_raises(self, n, timestamps, y):
        di = Diagnostics_Input()
        obj = di.from_arrays(
            y=y, t=timestamps, model_id="m", lo=y-1, hi=y+1
        )
        with pytest.raises(DiagnosticsInputError, match="Unknown capability"):
            obj.require("nonexistent")


# ---------------------------------------------------------------------------
# 2. from_adapter — ResidualPool
# ---------------------------------------------------------------------------

class TestFromResidualPool:

    def test_returns_dro(self, di, residual_pool):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            obj = di.from_adapter(residual_pool)
        assert isinstance(obj, DiagnosticsReadyObject)

    def test_source_dist_type(self, di, residual_pool):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            obj = di.from_adapter(residual_pool)
        assert obj.source_dist_type == "residual_reconstruction"

    def test_can_compute_interval(self, di, residual_pool):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            obj = di.from_adapter(residual_pool)
        assert obj.can_compute_interval

    def test_can_compute_pinball(self, di, residual_pool):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            obj = di.from_adapter(residual_pool)
        assert obj.can_compute_pinball

    def test_model_id_override(self, di, residual_pool):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            obj = di.from_adapter(residual_pool, model_id="overridden")
        assert obj.model_id == "overridden"

    def test_y_shape(self, di, residual_pool):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            obj = di.from_adapter(residual_pool)
        assert obj.y.shape == (obj.n_obs,)

    def test_lo_le_hi(self, di, residual_pool):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            obj = di.from_adapter(residual_pool)
        assert np.all(obj.lo <= obj.hi)


# ---------------------------------------------------------------------------
# 3. from_adapter — MarginalSamples
# ---------------------------------------------------------------------------

class TestFromMarginalSamples:

    def test_returns_dro(self, di, marginal_samples):
        obj = di.from_adapter(marginal_samples)
        assert isinstance(obj, DiagnosticsReadyObject)

    def test_source_dist_type(self, di, marginal_samples):
        obj = di.from_adapter(marginal_samples)
        assert obj.source_dist_type == "empirical_joint"

    def test_can_compute_pit(self, di, marginal_samples):
        obj = di.from_adapter(marginal_samples)
        assert obj.can_compute_pit

    def test_can_compute_crps(self, di, marginal_samples):
        obj = di.from_adapter(marginal_samples)
        assert obj.can_compute_crps

    def test_samples_shape(self, di, marginal_samples):
        obj = di.from_adapter(marginal_samples)
        assert obj.samples.shape == (obj.n_obs, 200)


# ---------------------------------------------------------------------------
# 4. from_adapter — QuantileFunctionObject
# ---------------------------------------------------------------------------

class TestFromQuantileFunctionObject:

    def test_returns_dro(self, di, qfo):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            obj = di.from_adapter(qfo)
        assert isinstance(obj, DiagnosticsReadyObject)

    def test_source_dist_type(self, di, qfo):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            obj = di.from_adapter(qfo)
        assert obj.source_dist_type == "quantile_function"

    def test_can_compute_pit_via_cdf(self, di, qfo):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            obj = di.from_adapter(qfo)
        assert obj.can_compute_pit
        assert obj.cdf_fn is not None

    def test_can_compute_pinball(self, di, qfo):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            obj = di.from_adapter(qfo)
        assert obj.can_compute_pinball

    def test_can_compute_interval(self, di, qfo):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            obj = di.from_adapter(qfo)
        assert obj.can_compute_interval

    def test_cdf_fn_callable(self, di, qfo, y):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            obj = di.from_adapter(qfo)
        result = obj.cdf_fn(y)
        assert result.shape == (len(y),)
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_meta_has_levels(self, di, qfo):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            obj = di.from_adapter(qfo)
        assert "n_levels" in obj.meta


# ---------------------------------------------------------------------------
# 5. from_adapter — JointSimulationObject
# ---------------------------------------------------------------------------

class TestFromJointSimulationObject:

    def test_returns_dro(self, di, joint_obj):
        obj = di.from_adapter(joint_obj)
        assert isinstance(obj, DiagnosticsReadyObject)

    def test_source_dist_type(self, di, joint_obj):
        obj = di.from_adapter(joint_obj)
        assert obj.source_dist_type == "empirical_joint"

    def test_can_compute_energy_score(self, di, joint_obj):
        obj = di.from_adapter(joint_obj)
        assert obj.can_compute_energy_score

    def test_joint_samples_shape(self, di, joint_obj, n):
        obj = di.from_adapter(joint_obj)
        assert obj.joint_samples.shape == (n, 200, 3)

    def test_variable_names_stored(self, di, joint_obj):
        obj = di.from_adapter(joint_obj)
        assert obj.variable_names == ["price", "temp", "gas"]

    def test_primary_var_in_meta(self, di, joint_obj):
        obj = di.from_adapter(joint_obj)
        assert obj.meta["primary_var"] == "price"

    def test_can_compute_crps_from_first_marginal(self, di, joint_obj):
        obj = di.from_adapter(joint_obj)
        assert obj.can_compute_crps


# ---------------------------------------------------------------------------
# 6. from_arrays — raw arrays
# ---------------------------------------------------------------------------

class TestFromArrays:

    def test_samples_only(self, di, n, timestamps, y):
        rng = np.random.default_rng(20)
        S = rng.normal(size=(n, 100))
        obj = di.from_arrays(y=y, t=timestamps, model_id="m", samples=S)
        assert obj.can_compute_crps
        assert obj.source_dist_type == "raw_arrays"

    def test_quantiles_and_interval(self, di, n, timestamps, y):
        q = {0.1: y - 5, 0.9: y + 5}
        obj = di.from_arrays(
            y=y, t=timestamps, model_id="m",
            quantiles=q, lo=y - 5, hi=y + 5
        )
        assert obj.can_compute_pinball
        assert obj.can_compute_interval

    def test_alpha_override(self, di, n, timestamps, y):
        obj = di.from_arrays(
            y=y, t=timestamps, model_id="m",
            lo=y-1, hi=y+1, alpha=0.2
        )
        assert obj.alpha == 0.2

    def test_source_dist_type_override(self, di, n, timestamps, y):
        obj = di.from_arrays(
            y=y, t=timestamps, model_id="m",
            lo=y-1, hi=y+1,
            source_dist_type="custom_model"
        )
        assert obj.source_dist_type == "custom_model"

    def test_summary_keys(self, di, n, timestamps, y):
        obj = di.from_arrays(y=y, t=timestamps, model_id="m", lo=y-1, hi=y+1)
        s = obj.summary()
        required = {"model_id", "n_obs", "source_dist_type",
                    "alpha", "capabilities", "has_joint", "variable_names"}
        assert required.issubset(set(s.keys()))


# ---------------------------------------------------------------------------
# 7. Error conditions
# ---------------------------------------------------------------------------

class TestErrors:

    def test_no_representation_raises(self, di, n, timestamps, y):
        with pytest.raises(DiagnosticsInputError, match="at least one"):
            di.from_arrays(y=y, t=timestamps, model_id="m")

    def test_unrecognised_adapter_type_raises(self, di):
        with pytest.raises(DiagnosticsInputError, match="Unrecognised"):
            di.from_adapter("not_an_adapter")

    def test_y_nan_raises(self, di, n, timestamps, y):
        y_bad = y.copy()
        y_bad[0] = np.nan
        with pytest.raises(DiagnosticsInputError, match="NaN"):
            di.from_arrays(
                y=y_bad, t=timestamps, model_id="m", lo=y-1, hi=y+1
            )

    def test_y_wrong_shape_raises(self, di, n, timestamps, y):
        y_2d = np.ones((n, 2))
        with pytest.raises(DiagnosticsInputError, match="1D"):
            di.from_arrays(
                y=y_2d, t=timestamps, model_id="m", lo=y-1, hi=y+1
            )

    def test_t_wrong_length_raises(self, di, n, timestamps, y):
        t_short = np.arange(n - 5)
        with pytest.raises(DiagnosticsInputError, match="alignment"):
            di.from_arrays(
                y=y, t=t_short, model_id="m", lo=y-1, hi=y+1
            )

    def test_samples_wrong_shape_raises(self, di, n, timestamps, y):
        S_bad = np.ones((n - 1, 100))
        with pytest.raises(DiagnosticsInputError, match="shape"):
            di.from_arrays(
                y=y, t=timestamps, model_id="m", samples=S_bad
            )

    def test_lo_without_hi_raises(self, di, n, timestamps, y):
        with pytest.raises(DiagnosticsInputError, match="both"):
            di.from_arrays(
                y=y, t=timestamps, model_id="m", lo=y-1
            )

    def test_hi_without_lo_raises(self, di, n, timestamps, y):
        with pytest.raises(DiagnosticsInputError, match="both"):
            di.from_arrays(
                y=y, t=timestamps, model_id="m", hi=y+1
            )

    def test_lo_hi_wrong_shape_raises(self, di, n, timestamps, y):
        with pytest.raises(DiagnosticsInputError, match="shape"):
            di.from_arrays(
                y=y, t=timestamps, model_id="m",
                lo=np.ones(n - 1), hi=np.ones(n)
            )

    def test_joint_wrong_shape_raises(self, di, n, timestamps, y):
        rng = np.random.default_rng(30)
        S = rng.normal(size=(n, 100))
        J_bad = rng.normal(size=(n - 1, 100, 2))
        with pytest.raises(DiagnosticsInputError, match="shape"):
            di.from_arrays(
                y=y, t=timestamps, model_id="m",
                samples=S, joint_samples=J_bad
            )

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            Diagnostics_Input(alpha=2.0)

    def test_lo_gt_hi_warns(self, di, n, timestamps, y):
        lo_bad = y + 10   # lo > hi
        hi_bad = y - 10
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj = di.from_arrays(
                y=y, t=timestamps, model_id="m",
                lo=lo_bad, hi=hi_bad
            )
            assert any("lo > hi" in str(x.message) for x in w)
        assert obj.can_compute_interval   # still built, just warned

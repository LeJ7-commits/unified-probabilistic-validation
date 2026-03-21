"""
tests/test_adapter_simulation_joint.py
========================================
Pytest suite for Adapter_SimulationJoint, JointSimulationObject,
and MarginalSamples.

Groups:
  1. Adapter_SimulationJoint — Format B (3D array), happy path
  2. Adapter_SimulationJoint — Format A (sims_dict), happy path
  3. Adapter_SimulationJoint — error conditions
  4. JointSimulationObject properties
  5. MarginalSamples — to_samples, to_quantiles

Run with:
  python -m pytest tests/test_adapter_simulation_joint.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.adapters.simulation_joint import (
    Adapter_SimulationJoint,
    JointSimulationObject,
    MarginalSamples,
    SimulationAdapterError,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def n_ts():
    return 50

@pytest.fixture
def M():
    return 200

@pytest.fixture
def d():
    return 3

@pytest.fixture
def var_names():
    return ["price", "temp", "gas"]

@pytest.fixture
def timestamps(n_ts):
    return pd.date_range("2020-01-01", periods=n_ts, freq="D")

@pytest.fixture
def S_3d(n_ts, M, d):
    rng = np.random.default_rng(42)
    return rng.normal(50, 5, size=(n_ts, M, d))

@pytest.fixture
def y_2d(n_ts, d):
    rng = np.random.default_rng(0)
    return rng.normal(50, 5, size=(n_ts, d))

@pytest.fixture
def adapter(var_names):
    return Adapter_SimulationJoint(
        variable_names=var_names,
        alpha=0.1,
        M_min=100,
        model_id="test_sim",
    )

@pytest.fixture
def joint_obj(adapter, S_3d, y_2d, timestamps):
    return adapter.from_array(S=S_3d, y=y_2d, t=timestamps)


# ---------------------------------------------------------------------------
# Helpers to build sims_dict format
# ---------------------------------------------------------------------------

def make_sims_dict(asof_dates, series_names, M, n_horizons=24, seed=42):
    """Build a minimal sims_dict in notebook format."""
    rng = np.random.default_rng(seed)
    path_cols = [f"path_{i}" for i in range(1, M + 1)]
    hours = np.arange(1, n_horizons + 1)
    sims_dict = {}
    for asof in asof_dates:
        sims_dict[asof] = {}
        for s in series_names:
            data = rng.normal(50, 5, size=(n_horizons, M))
            sims_dict[asof][s] = pd.DataFrame(
                data, index=hours, columns=path_cols
            )
    return sims_dict


def make_realized_dict(asof_dates, series_names, n_horizons=24, seed=99):
    """Build a realized_dict in notebook format."""
    rng = np.random.default_rng(seed)
    hours = np.arange(1, n_horizons + 1)
    realized_dict = {}
    for s in series_names:
        realized_dict[s] = pd.DataFrame(
            rng.normal(50, 5, size=(len(asof_dates), n_horizons)),
            index=asof_dates,
            columns=hours,
        )
    return realized_dict


# ---------------------------------------------------------------------------
# 1. Format B (3D array) — happy path
# ---------------------------------------------------------------------------

class TestFormatBHappyPath:

    def test_returns_joint_object(self, joint_obj):
        assert isinstance(joint_obj, JointSimulationObject)

    def test_dist_type(self, joint_obj):
        assert joint_obj.dist_type == "empirical_joint"

    def test_model_id(self, joint_obj, adapter):
        assert joint_obj.model_id == adapter.model_id

    def test_n_timestamps(self, joint_obj, n_ts):
        assert joint_obj.n_timestamps == n_ts

    def test_M(self, joint_obj, M):
        assert joint_obj.M == M

    def test_d(self, joint_obj, d):
        assert joint_obj.d == d

    def test_variable_names(self, joint_obj, var_names):
        assert joint_obj.variable_names == var_names

    def test_samples_joint_shape(self, joint_obj, n_ts, M, d):
        assert joint_obj.samples_joint.shape == (n_ts, M, d)

    def test_y_joint_shape(self, joint_obj, n_ts, d):
        assert joint_obj.y_joint.shape == (n_ts, d)

    def test_marginals_keys(self, joint_obj, var_names):
        assert set(joint_obj.marginals.keys()) == set(var_names)

    def test_marginals_count(self, joint_obj, d):
        assert len(joint_obj.marginals) == d

    def test_no_weights_by_default(self, joint_obj):
        assert joint_obj.weights is None

    def test_d1_array_accepted(self, timestamps):
        """2D array (n_ts, M) should be treated as d=1."""
        rng = np.random.default_rng(7)
        S_2d = rng.normal(size=(20, 150))
        y_1d = rng.normal(size=20)
        adapter = Adapter_SimulationJoint(
            variable_names=["price"], M_min=100
        )
        obj = adapter.from_array(S=S_2d, y=y_1d, t=np.arange(20))
        assert obj.d == 1
        assert obj.samples_joint.shape == (20, 150, 1)

    def test_default_variable_names_inferred(self, S_3d, y_2d, timestamps):
        """No variable_names → auto-generate var_0, var_1, ..."""
        adapter = Adapter_SimulationJoint(M_min=100)
        obj = adapter.from_array(S=S_3d, y=y_2d, t=timestamps)
        assert obj.variable_names == ["var_0", "var_1", "var_2"]

    def test_weights_normalised(self, S_3d, y_2d, timestamps, M, var_names):
        rng = np.random.default_rng(5)
        w = rng.uniform(0, 1, size=M)
        adapter = Adapter_SimulationJoint(
            variable_names=var_names, M_min=100
        )
        obj = adapter.from_array(S=S_3d, y=y_2d, t=timestamps, weights=w)
        assert obj.weights is not None
        np.testing.assert_allclose(obj.weights.sum(), 1.0, rtol=1e-10)

    def test_meta_keys(self, joint_obj):
        required = {"M", "d", "variable_names", "model_id", "alpha",
                    "horizon_agg", "has_weights"}
        assert required.issubset(set(joint_obj.meta.keys()))


# ---------------------------------------------------------------------------
# 2. Format A (sims_dict) — happy path
# ---------------------------------------------------------------------------

class TestFormatAHappyPath:

    def test_from_sims_dict_returns_joint(self):
        asof_dates = pd.date_range("2020-01-01", periods=30, freq="D")
        series = ["price", "temp"]
        sd = make_sims_dict(asof_dates, series, M=150)
        rd = make_realized_dict(asof_dates, series)
        adapter = Adapter_SimulationJoint(M_min=100)
        obj = adapter.from_sims_dict(sd, rd, series_names=series)
        assert isinstance(obj, JointSimulationObject)
        assert obj.d == 2
        assert obj.n_timestamps == 30
        assert obj.M == 150

    def test_series_names_inferred_if_none(self):
        asof_dates = pd.date_range("2020-01-01", periods=10, freq="D")
        series = ["carbon", "gas", "price"]
        sd = make_sims_dict(asof_dates, series, M=150)
        rd = make_realized_dict(asof_dates, series)
        adapter = Adapter_SimulationJoint(M_min=100)
        obj = adapter.from_sims_dict(sd, rd)
        assert set(obj.variable_names) == set(series)

    def test_horizon_agg_mean(self):
        asof_dates = pd.date_range("2020-01-01", periods=20, freq="D")
        series = ["price"]
        sd = make_sims_dict(asof_dates, series, M=150, n_horizons=24)
        rd = make_realized_dict(asof_dates, series, n_horizons=24)
        adapter = Adapter_SimulationJoint(M_min=100, horizon_agg="mean")
        obj = adapter.from_sims_dict(sd, rd)
        assert obj.n_timestamps == 20

    def test_horizon_agg_first(self):
        asof_dates = pd.date_range("2020-01-01", periods=20, freq="D")
        series = ["price"]
        sd = make_sims_dict(asof_dates, series, M=150, n_horizons=24)
        rd = make_realized_dict(asof_dates, series, n_horizons=24)
        adapter = Adapter_SimulationJoint(M_min=100, horizon_agg="first")
        obj = adapter.from_sims_dict(sd, rd)
        assert obj.n_timestamps == 20

    def test_marginals_have_correct_shape(self):
        asof_dates = pd.date_range("2020-01-01", periods=25, freq="D")
        series = ["price", "temp"]
        sd = make_sims_dict(asof_dates, series, M=200)
        rd = make_realized_dict(asof_dates, series)
        adapter = Adapter_SimulationJoint(M_min=100)
        obj = adapter.from_sims_dict(sd, rd, series_names=series)
        for vname in series:
            m = obj.marginals[vname]
            assert m.samples.shape == (25, 200)
            assert m.y.shape == (25,)


# ---------------------------------------------------------------------------
# 3. Error conditions
# ---------------------------------------------------------------------------

class TestErrors:

    def test_M_below_min_raises(self, timestamps, y_2d):
        rng = np.random.default_rng(1)
        S = rng.normal(size=(len(timestamps), 50, 2))   # M=50 < M_min=100
        adapter = Adapter_SimulationJoint(M_min=100)
        with pytest.raises(SimulationAdapterError, match="M=50"):
            adapter.from_array(S=S, y=y_2d[:, :2], t=timestamps)

    def test_nan_in_S_raises(self, timestamps, y_2d, M, d):
        rng = np.random.default_rng(2)
        S = rng.normal(size=(len(timestamps), M, d))
        S[0, 0, 0] = np.nan
        adapter = Adapter_SimulationJoint(M_min=100)
        with pytest.raises(SimulationAdapterError, match="NaN or Inf"):
            adapter.from_array(S=S, y=y_2d, t=timestamps)

    def test_inf_in_S_raises(self, timestamps, y_2d, M, d):
        rng = np.random.default_rng(3)
        S = rng.normal(size=(len(timestamps), M, d))
        S[1, 2, 1] = np.inf
        adapter = Adapter_SimulationJoint(M_min=100)
        with pytest.raises(SimulationAdapterError, match="NaN or Inf"):
            adapter.from_array(S=S, y=y_2d, t=timestamps)

    def test_degenerate_variance_raises(self, timestamps, y_2d, M):
        rng = np.random.default_rng(4)
        S = rng.normal(size=(len(timestamps), M, 2))
        S[:, :, 1] = 42.0   # zero variance in dim 1
        adapter = Adapter_SimulationJoint(M_min=100, var_min=1e-8)
        with pytest.raises(SimulationAdapterError, match="Degenerate"):
            adapter.from_array(S=S, y=y_2d[:, :2], t=timestamps)

    def test_nan_in_y_raises(self, timestamps, S_3d):
        y_bad = np.ones((len(timestamps), 3))
        y_bad[0, 0] = np.nan
        adapter = Adapter_SimulationJoint(M_min=100)
        with pytest.raises(SimulationAdapterError, match="realizations"):
            adapter.from_array(S=S_3d, y=y_bad, t=timestamps)

    def test_y_wrong_shape_raises(self, timestamps, S_3d):
        y_bad = np.ones((len(timestamps), 5))   # d=5 but S has d=3
        adapter = Adapter_SimulationJoint(M_min=100)
        with pytest.raises(SimulationAdapterError, match="shape"):
            adapter.from_array(S=S_3d, y=y_bad, t=timestamps)

    def test_wrong_variable_names_length_raises(self, timestamps, S_3d, y_2d):
        adapter = Adapter_SimulationJoint(M_min=100)
        with pytest.raises(SimulationAdapterError, match="variable_names"):
            adapter.from_array(
                S=S_3d, y=y_2d, t=timestamps,
                variable_names=["price"]   # d=3 but only 1 name
            )

    def test_negative_weights_raises(self, S_3d, y_2d, timestamps, M):
        w = np.ones(M)
        w[0] = -1.0
        adapter = Adapter_SimulationJoint(M_min=100)
        with pytest.raises(SimulationAdapterError, match="non-negative"):
            adapter.from_array(S=S_3d, y=y_2d, t=timestamps, weights=w)

    def test_wrong_weights_shape_raises(self, S_3d, y_2d, timestamps):
        adapter = Adapter_SimulationJoint(M_min=100)
        with pytest.raises(SimulationAdapterError, match="shape"):
            adapter.from_array(
                S=S_3d, y=y_2d, t=timestamps, weights=np.ones(5)
            )

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            Adapter_SimulationJoint(alpha=2.0)

    def test_invalid_horizon_agg_raises(self):
        with pytest.raises(ValueError, match="horizon_agg"):
            Adapter_SimulationJoint(horizon_agg="median")

    def test_empty_sims_dict_raises(self):
        adapter = Adapter_SimulationJoint(M_min=100)
        with pytest.raises(SimulationAdapterError, match="empty"):
            adapter.from_sims_dict({}, {})

    def test_missing_series_raises(self):
        asof_dates = pd.date_range("2020-01-01", periods=10, freq="D")
        sd = make_sims_dict(asof_dates, ["price"], M=150)
        rd = make_realized_dict(asof_dates, ["price"])
        adapter = Adapter_SimulationJoint(M_min=100)
        with pytest.raises(SimulationAdapterError, match="not found"):
            adapter.from_sims_dict(sd, rd, series_names=["price", "missing"])

    def test_inconsistent_M_raises(self):
        """sims_dict with different M across timestamps should raise."""
        asof_dates = pd.date_range("2020-01-01", periods=3, freq="D")
        series = ["price"]
        sd = make_sims_dict(asof_dates, series, M=150)
        # Corrupt one entry to have different M
        hours = np.arange(1, 25)
        sd[asof_dates[1]]["price"] = pd.DataFrame(
            np.random.normal(size=(24, 200)),   # M=200 instead of 150
            index=hours,
            columns=[f"path_{i}" for i in range(1, 201)],
        )
        rd = make_realized_dict(asof_dates, series)
        adapter = Adapter_SimulationJoint(M_min=100)
        with pytest.raises(SimulationAdapterError, match="Inconsistent M"):
            adapter.from_sims_dict(sd, rd, series_names=series)


# ---------------------------------------------------------------------------
# 4. JointSimulationObject properties
# ---------------------------------------------------------------------------

class TestJointSimulationObjectProperties:

    def test_get_marginal_returns_correct(self, joint_obj, var_names):
        m = joint_obj.get_marginal(var_names[0])
        assert isinstance(m, MarginalSamples)
        assert m.variable_name == var_names[0]

    def test_get_marginal_missing_raises(self, joint_obj):
        with pytest.raises(KeyError, match="not found"):
            joint_obj.get_marginal("nonexistent_variable")

    def test_summary_keys(self, joint_obj):
        s = joint_obj.summary()
        required = {"dist_type", "model_id", "n_timestamps",
                    "M", "d", "variable_names"}
        assert required.issubset(set(s.keys()))

    def test_summary_values(self, joint_obj, n_ts, M, d, var_names):
        s = joint_obj.summary()
        assert s["dist_type"] == "empirical_joint"
        assert s["n_timestamps"] == n_ts
        assert s["M"] == M
        assert s["d"] == d
        assert s["variable_names"] == var_names


# ---------------------------------------------------------------------------
# 5. MarginalSamples — to_samples, to_quantiles
# ---------------------------------------------------------------------------

class TestMarginalSamples:

    def test_to_samples_shape(self, joint_obj, var_names, n_ts, M):
        m = joint_obj.get_marginal(var_names[0])
        s = m.to_samples()
        assert s.shape == (n_ts, M)

    def test_to_quantiles_keys(self, joint_obj, var_names):
        m = joint_obj.get_marginal(var_names[0])
        q = m.to_quantiles()
        expected_keys = {m.alpha / 2, 1 - m.alpha / 2}
        assert set(q.keys()) == expected_keys

    def test_to_quantiles_lo_le_hi(self, joint_obj, var_names):
        m = joint_obj.get_marginal(var_names[0])
        q = m.to_quantiles()
        lo = q[m.alpha / 2]
        hi = q[1 - m.alpha / 2]
        assert np.all(lo <= hi)

    def test_to_quantiles_shape(self, joint_obj, var_names, n_ts):
        m = joint_obj.get_marginal(var_names[0])
        q = m.to_quantiles()
        for arr in q.values():
            assert arr.shape == (n_ts,)

    def test_marginal_y_shape(self, joint_obj, var_names, n_ts):
        m = joint_obj.get_marginal(var_names[0])
        assert m.y.shape == (n_ts,)

    def test_marginal_summary_keys(self, joint_obj, var_names):
        m = joint_obj.get_marginal(var_names[0])
        s = m.summary()
        required = {"variable_name", "model_id", "n_timestamps", "M", "alpha"}
        assert required.issubset(set(s.keys()))

    def test_all_marginals_consistent_n_timestamps(self, joint_obj, n_ts):
        for m in joint_obj.marginals.values():
            assert m.n_timestamps == n_ts

    def test_all_marginals_consistent_M(self, joint_obj, M):
        for m in joint_obj.marginals.values():
            assert m.samples.shape[1] == M

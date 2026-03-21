"""
tests/test_data_contract.py
============================
Pytest suite for DataContract and StandardizedModelObject.

Tests are organised into five groups:
  1. SplitLabel validation
  2. Required field validation (t, y, model_id, split)
  3. Optional field validation (h, y_hat, quantiles, S, x)
  4. Sanity checks (monotone timestamps, non-crossing quantiles, min_samples)
  5. StandardizedModelObject properties and convenience methods

Run with:
  pytest tests/test_data_contract.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.data_contract import (
    DataContract,
    DataContractError,
    StandardizedModelObject,
    validate_split_label,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def n():
    return 100


@pytest.fixture
def timestamps(n):
    return pd.date_range("2020-01-01", periods=n, freq="h")


@pytest.fixture
def y(n):
    rng = np.random.default_rng(42)
    return rng.normal(50, 5, size=n)


@pytest.fixture
def contract():
    return DataContract(min_samples=10, min_obs=2)


@pytest.fixture
def valid_obj(contract, timestamps, y):
    return contract.validate(
        t=timestamps, y=y, model_id="test_model", split="window_0"
    )


# ---------------------------------------------------------------------------
# 1. SplitLabel validation
# ---------------------------------------------------------------------------

class TestSplitLabel:

    @pytest.mark.parametrize("label", [
        "train", "test",
        "window_0", "window_1", "window_42",
        "regime_winter", "regime_high_vol", "regime_summer",
        "regime_low_vol_2020",
    ])
    def test_valid_labels(self, label):
        result = validate_split_label(label)
        assert result == label.lower().strip()

    @pytest.mark.parametrize("label", [
        "Train", "TEST", "TRAIN",      # uppercase — normalised to lowercase
        "regime_UPPER",                # uppercase regime tag — normalised
        "window_0",                    # already lower
    ])
    def test_normalises_to_lowercase(self, label):
        result = validate_split_label(label)
        assert result == label.lower().strip()

    @pytest.mark.parametrize("bad_label", [
        "",                      # empty
        "  ",                    # whitespace only
        "window",                # missing index
        "window_abc",            # non-integer index
        "window_-1",             # negative index (not matched by \d+)
        "regime",                # missing tag
        "regime_",               # empty tag
        "fold_0",                # unknown prefix
        "calibration",           # not in vocabulary
        "window_3_extra",        # too many parts
    ])
    def test_invalid_labels_raise(self, bad_label):
        with pytest.raises(ValueError, match="Invalid SplitLabel"):
            validate_split_label(bad_label)


# ---------------------------------------------------------------------------
# 2. Required field validation
# ---------------------------------------------------------------------------

class TestRequiredFields:

    def test_valid_minimal(self, contract, timestamps, y):
        obj = contract.validate(
            t=timestamps, y=y, model_id="m1", split="train"
        )
        assert isinstance(obj, StandardizedModelObject)
        assert obj.model_id == "m1"
        assert obj.split == "train"
        assert obj.n_obs == len(y)

    def test_y_nan_raises(self, contract, timestamps, y):
        y_bad = y.copy()
        y_bad[5] = np.nan
        with pytest.raises(DataContractError, match="NaN"):
            contract.validate(t=timestamps, y=y_bad, model_id="m", split="train")

    def test_y_2d_raises(self, contract, timestamps):
        y_2d = np.ones((100, 2))
        with pytest.raises(DataContractError, match="1-dimensional"):
            contract.validate(t=timestamps, y=y_2d, model_id="m", split="train")

    def test_empty_model_id_raises(self, contract, timestamps, y):
        with pytest.raises(DataContractError, match="model_id"):
            contract.validate(t=timestamps, y=y, model_id="  ", split="train")

    def test_invalid_split_raises(self, contract, timestamps, y):
        with pytest.raises(ValueError, match="Invalid SplitLabel"):
            contract.validate(t=timestamps, y=y, model_id="m", split="bad_label")

    def test_length_mismatch_raises(self, contract, timestamps, y):
        t_short = timestamps[:50]
        with pytest.raises(DataContractError, match="same length"):
            contract.validate(t=t_short, y=y, model_id="m", split="train")

    def test_min_obs_raises(self):
        contract = DataContract(min_obs=10)
        t = pd.date_range("2020-01-01", periods=5, freq="h")
        y = np.ones(5)
        with pytest.raises(DataContractError, match="Too few observations"):
            contract.validate(t=t, y=y, model_id="m", split="train")

    def test_numeric_timestamps_accepted(self, contract, y):
        t_numeric = np.arange(len(y), dtype=float)
        obj = contract.validate(t=t_numeric, y=y, model_id="m", split="train")
        assert obj.n_obs == len(y)


# ---------------------------------------------------------------------------
# 3. Optional field validation
# ---------------------------------------------------------------------------

class TestOptionalFields:

    def test_y_hat_accepted(self, contract, timestamps, y):
        y_hat = y + np.random.normal(0, 0.5, size=len(y))
        obj = contract.validate(
            t=timestamps, y=y, model_id="m", split="train", y_hat=y_hat
        )
        assert obj.has_point_forecast
        np.testing.assert_array_equal(obj.y_hat, y_hat)

    def test_y_hat_wrong_shape_raises(self, contract, timestamps, y):
        y_hat_bad = np.ones(50)
        with pytest.raises(DataContractError, match="y_hat"):
            contract.validate(
                t=timestamps, y=y, model_id="m", split="train", y_hat=y_hat_bad
            )

    def test_samples_accepted_n_M(self, contract, timestamps, y):
        n = len(y)
        S = np.random.normal(size=(n, 50))
        obj = contract.validate(
            t=timestamps, y=y, model_id="m", split="train", S=S
        )
        assert obj.has_samples
        assert obj.S.shape == (n, 50)
        assert obj.n_samples == 50

    def test_samples_accepted_M_n_transposed(self, contract, timestamps, y):
        """S in (M, n) format should be transposed to (n, M)."""
        n = len(y)
        S = np.random.normal(size=(50, n))
        obj = contract.validate(
            t=timestamps, y=y, model_id="m", split="train", S=S
        )
        assert obj.S.shape == (n, 50)

    def test_samples_below_min_raises(self, contract, timestamps, y):
        n = len(y)
        S = np.random.normal(size=(n, 5))  # min_samples=10
        with pytest.raises(DataContractError, match="Sample size"):
            contract.validate(
                t=timestamps, y=y, model_id="m", split="train", S=S
            )

    def test_samples_nan_raises(self, contract, timestamps, y):
        n = len(y)
        S = np.random.normal(size=(n, 20))
        S[0, 0] = np.nan
        with pytest.raises(DataContractError, match="NaN or Inf"):
            contract.validate(
                t=timestamps, y=y, model_id="m", split="train", S=S
            )

    def test_quantiles_accepted(self, contract, timestamps, y):
        n = len(y)
        q = {
            0.05: y - 2,
            0.50: y,
            0.95: y + 2,
        }
        obj = contract.validate(
            t=timestamps, y=y, model_id="m", split="train", quantiles=q
        )
        assert obj.has_quantiles
        assert obj.quantile_levels == [0.05, 0.50, 0.95]

    def test_quantile_crossing_raises(self, contract, timestamps, y):
        n = len(y)
        q_crossing = {
            0.05: y + 5,   # higher than 0.95 — crossing!
            0.95: y,
        }
        with pytest.raises(DataContractError, match="crossing"):
            contract.validate(
                t=timestamps, y=y, model_id="m", split="train",
                quantiles=q_crossing
            )

    def test_quantile_out_of_01_raises(self, contract, timestamps, y):
        with pytest.raises(DataContractError, match="out of"):
            contract.validate(
                t=timestamps, y=y, model_id="m", split="train",
                quantiles={0.0: y, 0.5: y}
            )

    def test_quantile_wrong_shape_raises(self, contract, timestamps, y):
        with pytest.raises(DataContractError, match="shape"):
            contract.validate(
                t=timestamps, y=y, model_id="m", split="train",
                quantiles={0.5: np.ones(50)}   # wrong n
            )

    def test_h_scalar_broadcast(self, contract, timestamps, y):
        obj = contract.validate(
            t=timestamps, y=y, model_id="m", split="train", h=24
        )
        assert obj.h is not None
        assert obj.h.shape == (len(y),)
        assert np.all(obj.h == 24)

    def test_h_array_accepted(self, contract, timestamps, y):
        h_arr = np.arange(len(y))
        obj = contract.validate(
            t=timestamps, y=y, model_id="m", split="train", h=h_arr
        )
        np.testing.assert_array_equal(obj.h, h_arr)

    def test_h_wrong_shape_raises(self, contract, timestamps, y):
        with pytest.raises(DataContractError, match="shape"):
            contract.validate(
                t=timestamps, y=y, model_id="m", split="train",
                h=np.ones(50, dtype=int)
            )

    def test_covariates_accepted(self, contract, timestamps, y):
        x = np.random.normal(size=(len(y), 3))
        obj = contract.validate(
            t=timestamps, y=y, model_id="m", split="train", x=x
        )
        assert obj.x is not None
        assert obj.x.shape == (len(y), 3)

    def test_covariates_1d_reshaped(self, contract, timestamps, y):
        x = np.ones(len(y))
        obj = contract.validate(
            t=timestamps, y=y, model_id="m", split="train", x=x
        )
        assert obj.x.shape == (len(y), 1)

    def test_covariates_wrong_rows_raises(self, contract, timestamps, y):
        x = np.ones((50, 2))
        with pytest.raises(DataContractError, match="rows"):
            contract.validate(
                t=timestamps, y=y, model_id="m", split="train", x=x
            )


# ---------------------------------------------------------------------------
# 4. Sanity checks
# ---------------------------------------------------------------------------

class TestSanityChecks:

    def test_non_monotone_timestamps_raises(self, contract, y):
        t = pd.date_range("2020-01-01", periods=len(y), freq="h")
        t_bad = t.values.copy()
        t_bad[5] = t_bad[3]  # duplicate
        with pytest.raises(DataContractError, match="monotonically"):
            contract.validate(t=t_bad, y=y, model_id="m", split="train")

    def test_backwards_timestamps_raises(self, contract, y):
        t = pd.date_range("2020-01-01", periods=len(y), freq="h")
        t_bad = t.values[::-1]   # reversed
        with pytest.raises(DataContractError, match="monotonically"):
            contract.validate(t=t_bad, y=y, model_id="m", split="train")

    def test_quantile_nearly_crossing_tolerance(self, contract, timestamps, y):
        """Tiny numerical noise within 1e-8 should not raise."""
        eps = 1e-10
        q = {
            0.05: y - 1,
            0.95: y - 1 + eps,  # barely above p=0.05 — within tolerance
        }
        # Should not raise
        obj = contract.validate(
            t=timestamps, y=y, model_id="m", split="train", quantiles=q
        )
        assert obj.has_quantiles


# ---------------------------------------------------------------------------
# 5. StandardizedModelObject properties
# ---------------------------------------------------------------------------

class TestStandardizedModelObject:

    def test_split_type_train(self, contract, timestamps, y):
        obj = contract.validate(t=timestamps, y=y, model_id="m", split="train")
        assert obj.split_type == "train"

    def test_split_type_window(self, contract, timestamps, y):
        obj = contract.validate(t=timestamps, y=y, model_id="m", split="window_7")
        assert obj.split_type == "window"
        assert obj.split_index == 7

    def test_split_index_none_for_non_window(self, contract, timestamps, y):
        obj = contract.validate(t=timestamps, y=y, model_id="m", split="train")
        assert obj.split_index is None

    def test_split_regime(self, contract, timestamps, y):
        obj = contract.validate(
            t=timestamps, y=y, model_id="m", split="regime_high_vol"
        )
        assert obj.split_type == "regime"
        assert obj.split_regime == "high_vol"

    def test_split_regime_multipart(self, contract, timestamps, y):
        obj = contract.validate(
            t=timestamps, y=y, model_id="m", split="regime_low_vol_2020"
        )
        assert obj.split_regime == "low_vol_2020"

    def test_has_samples_false_by_default(self, valid_obj):
        assert not valid_obj.has_samples
        assert valid_obj.n_samples is None

    def test_has_quantiles_false_by_default(self, valid_obj):
        assert not valid_obj.has_quantiles
        assert valid_obj.quantile_levels == []

    def test_has_point_forecast_false_by_default(self, valid_obj):
        assert not valid_obj.has_point_forecast

    def test_summary_keys(self, valid_obj):
        s = valid_obj.summary()
        required_keys = {
            "model_id", "split", "n_obs", "has_samples",
            "n_samples", "has_quantiles", "quantile_levels",
            "has_point_forecast", "has_covariates",
        }
        assert required_keys.issubset(set(s.keys()))

    def test_summary_values(self, contract, timestamps, y):
        n = len(y)
        S = np.random.normal(size=(n, 20))
        q = {0.1: y - 1, 0.9: y + 1}
        obj = contract.validate(
            t=timestamps, y=y, model_id="entsoe_load",
            split="window_3", S=S, quantiles=q
        )
        s = obj.summary()
        assert s["model_id"] == "entsoe_load"
        assert s["split"] == "window_3"
        assert s["n_obs"] == n
        assert s["has_samples"] is True
        assert s["n_samples"] == 20
        assert s["has_quantiles"] is True
        assert s["quantile_levels"] == [0.1, 0.9]

    def test_frozen_immutable(self, valid_obj):
        """StandardizedModelObject is frozen — mutation should raise."""
        with pytest.raises(Exception):
            valid_obj.model_id = "changed"

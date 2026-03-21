"""
tests/test_threshold_calibrator.py
=====================================
Pytest suite for ThresholdCalibrator and CalibratedThresholds.

Groups:
  1. ThresholdCalibrator.calibrate — happy path
  2. Threshold clamping (relax / tighten)
  3. Sparse regime fallback
  4. calibrate_from_rolling_results
  5. CalibratedThresholds.get_policy
  6. CalibratedThresholds.to_dict
  7. Error conditions

Run with:
  python -m pytest tests/test_threshold_calibrator.py -v
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.governance.risk_classification import RiskPolicy
from src.governance.threshold_calibrator import (
    CalibratedThresholds,
    ThresholdCalibrator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fallback_policy():
    return RiskPolicy(coverage_target=0.90)

@pytest.fixture
def calibrator():
    return ThresholdCalibrator(
        coverage_quantile_green=0.10,
        coverage_quantile_red=0.05,
        min_regime_samples=5,
        relax_factor=0.15,
        tighten_factor=0.05,
    )

@pytest.fixture
def coverage_by_regime():
    rng = np.random.default_rng(42)
    return {
        "winter": rng.normal(0.85, 0.03, 50),   # slightly lower coverage
        "summer": rng.normal(0.91, 0.02, 50),   # slightly higher
        "high_vol": rng.normal(0.80, 0.05, 30), # lower coverage in high vol
    }

@pytest.fixture
def thresholds(calibrator, coverage_by_regime, fallback_policy):
    return calibrator.calibrate(
        coverage_by_regime=coverage_by_regime,
        fallback_policy=fallback_policy,
    )


# ---------------------------------------------------------------------------
# 1. Happy path
# ---------------------------------------------------------------------------

class TestHappyPath:

    def test_returns_calibrated_thresholds(self, thresholds):
        assert isinstance(thresholds, CalibratedThresholds)

    def test_all_regimes_calibrated(self, thresholds, coverage_by_regime):
        for regime in coverage_by_regime:
            assert regime in thresholds.regime_policies

    def test_regime_policies_are_risk_policies(self, thresholds):
        for policy in thresholds.regime_policies.values():
            assert isinstance(policy, RiskPolicy)

    def test_calibrated_targets_in_valid_range(self, thresholds):
        for policy in thresholds.regime_policies.values():
            assert 0 < policy.coverage_target < 1

    def test_fallback_policy_stored(self, thresholds, fallback_policy):
        assert thresholds.fallback_policy is fallback_policy

    def test_regime_stats_keys(self, thresholds, coverage_by_regime):
        for regime in coverage_by_regime:
            assert regime in thresholds.regime_stats

    def test_regime_stats_contains_n_samples(self, thresholds):
        for stats in thresholds.regime_stats.values():
            assert "n_samples" in stats
            assert stats["n_samples"] > 0

    def test_calibration_metrics_keys(self, thresholds):
        m = thresholds.calibration_metrics
        required = {
            "n_regimes_total", "n_regimes_calibrated",
            "n_regimes_sparse", "global_target"
        }
        assert required.issubset(set(m.keys()))

    def test_quantiles_stored(self, thresholds, calibrator):
        assert thresholds.coverage_quantile_green == calibrator.coverage_quantile_green
        assert thresholds.coverage_quantile_red == calibrator.coverage_quantile_red

    def test_optional_pit_stats_stored(self, calibrator, coverage_by_regime, fallback_policy):
        rng = np.random.default_rng(1)
        pit = {"winter": rng.uniform(0, 0.5, 50)}
        thresholds = calibrator.calibrate(
            coverage_by_regime=coverage_by_regime,
            fallback_policy=fallback_policy,
            pit_stat_by_regime=pit,
        )
        assert "mean_pit_ks" in thresholds.regime_stats["winter"]

    def test_optional_pinball_stored(self, calibrator, coverage_by_regime, fallback_policy):
        rng = np.random.default_rng(2)
        pb = {"winter": rng.uniform(1, 5, 50)}
        thresholds = calibrator.calibrate(
            coverage_by_regime=coverage_by_regime,
            fallback_policy=fallback_policy,
            pinball_by_regime=pb,
        )
        assert "mean_pinball" in thresholds.regime_stats["winter"]


# ---------------------------------------------------------------------------
# 2. Threshold clamping
# ---------------------------------------------------------------------------

class TestThresholdClamping:

    def test_relaxed_target_not_below_lower_bound(self, calibrator, fallback_policy):
        """Low coverage regime → target clamped at global - relax_factor."""
        # All coverage values are 0.50 → 10th pct = 0.50
        # But lower_bound = 0.90 - 0.15 = 0.75
        coverage = {"low_regime": np.full(50, 0.50)}
        thresholds = calibrator.calibrate(
            coverage_by_regime=coverage,
            fallback_policy=fallback_policy,
        )
        policy = thresholds.regime_policies["low_regime"]
        assert policy.coverage_target >= 0.90 - 0.15 - 1e-10

    def test_tightened_target_not_above_upper_bound(self, calibrator, fallback_policy):
        """High coverage regime → target clamped at global + tighten_factor."""
        # All coverage values are 1.0 → 10th pct = 1.0
        # But upper_bound = 0.90 + 0.05 = 0.95
        coverage = {"high_regime": np.full(50, 1.0)}
        thresholds = calibrator.calibrate(
            coverage_by_regime=coverage,
            fallback_policy=fallback_policy,
        )
        policy = thresholds.regime_policies["high_regime"]
        assert policy.coverage_target <= 0.90 + 0.05 + 1e-10

    def test_target_near_global_for_well_specified_regime(self, fallback_policy):
        """Coverage centred at 0.90 → calibrated target ≈ global target."""
        calibrator = ThresholdCalibrator(
            coverage_quantile_green=0.10,
            min_regime_samples=5,
            relax_factor=0.15,
        )
        rng = np.random.default_rng(99)
        coverage = {"normal": rng.normal(0.90, 0.01, 100)}
        thresholds = calibrator.calibrate(
            coverage_by_regime=coverage,
            fallback_policy=fallback_policy,
        )
        policy = thresholds.regime_policies["normal"]
        # 10th percentile of N(0.90, 0.01) ≈ 0.887 → within relax bounds
        assert abs(policy.coverage_target - 0.90) <= 0.15 + 1e-6


# ---------------------------------------------------------------------------
# 3. Sparse regime fallback
# ---------------------------------------------------------------------------

class TestSparseRegimeFallback:

    def test_sparse_regime_not_in_policies(self, calibrator, fallback_policy):
        coverage = {
            "sparse": np.array([0.85, 0.86, 0.87]),  # n=3 < min=5
            "normal": np.full(50, 0.90),
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            thresholds = calibrator.calibrate(
                coverage_by_regime=coverage,
                fallback_policy=fallback_policy,
            )
            sparse_warns = [x for x in w if "sparse" in str(x.message)]
            assert len(sparse_warns) >= 1

        assert "sparse" not in thresholds.regime_policies
        assert "sparse" in thresholds.regime_stats

    def test_sparse_regime_gets_fallback_policy(self, calibrator, fallback_policy):
        coverage = {"sparse": np.array([0.85, 0.86])}
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            thresholds = calibrator.calibrate(
                coverage_by_regime=coverage,
                fallback_policy=fallback_policy,
            )
        policy = thresholds.get_policy("sparse")
        assert policy is fallback_policy

    def test_calibration_metrics_counts_sparse(self, calibrator, fallback_policy):
        coverage = {
            "sparse": np.array([0.85]),
            "normal": np.full(50, 0.90),
        }
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            thresholds = calibrator.calibrate(
                coverage_by_regime=coverage,
                fallback_policy=fallback_policy,
            )
        assert thresholds.calibration_metrics["n_regimes_sparse"] == 1
        assert thresholds.calibration_metrics["n_regimes_calibrated"] == 1


# ---------------------------------------------------------------------------
# 4. calibrate_from_rolling_results
# ---------------------------------------------------------------------------

class TestCalibrateFromRollingResults:

    def test_from_rolling_returns_thresholds(self, calibrator, fallback_policy):
        n = 50
        rng = np.random.default_rng(5)
        df = pd.DataFrame({
            "empirical_coverage": rng.normal(0.88, 0.03, n),
            "window_start": range(n),
        })
        tags = ["regime_winter"] * 25 + ["regime_summer"] * 25
        thresholds = calibrator.calibrate_from_rolling_results(
            rolling_df=df,
            regime_tags=tags,
            fallback_policy=fallback_policy,
        )
        assert isinstance(thresholds, CalibratedThresholds)

    def test_from_rolling_strips_regime_prefix(self, calibrator, fallback_policy):
        n = 30
        rng = np.random.default_rng(6)
        df = pd.DataFrame({"empirical_coverage": rng.normal(0.90, 0.02, n)})
        tags = ["regime_winter"] * n
        thresholds = calibrator.calibrate_from_rolling_results(
            rolling_df=df, regime_tags=tags, fallback_policy=fallback_policy
        )
        # "winter" (stripped) should be in policies
        assert "winter" in thresholds.regime_policies

    def test_from_rolling_wrong_length_raises(self, calibrator, fallback_policy):
        df = pd.DataFrame({"empirical_coverage": np.ones(10)})
        with pytest.raises(ValueError, match="length"):
            calibrator.calibrate_from_rolling_results(
                rolling_df=df,
                regime_tags=["regime_a"] * 5,   # wrong length
                fallback_policy=fallback_policy,
            )

    def test_from_rolling_missing_column_raises(self, calibrator, fallback_policy):
        df = pd.DataFrame({"wrong_col": np.ones(10)})
        with pytest.raises(ValueError, match="not found"):
            calibrator.calibrate_from_rolling_results(
                rolling_df=df,
                regime_tags=["regime_a"] * 10,
                fallback_policy=fallback_policy,
            )


# ---------------------------------------------------------------------------
# 5. CalibratedThresholds.get_policy
# ---------------------------------------------------------------------------

class TestGetPolicy:

    def test_get_policy_by_raw_tag(self, thresholds):
        p = thresholds.get_policy("winter")
        assert isinstance(p, RiskPolicy)

    def test_get_policy_by_splitlabel_tag(self, thresholds):
        p = thresholds.get_policy("regime_winter")
        assert isinstance(p, RiskPolicy)

    def test_get_policy_raw_and_splitlabel_same(self, thresholds):
        p_raw   = thresholds.get_policy("winter")
        p_split = thresholds.get_policy("regime_winter")
        assert p_raw.coverage_target == p_split.coverage_target

    def test_unknown_regime_returns_fallback(self, thresholds, fallback_policy):
        p = thresholds.get_policy("regime_unknown_xyz")
        assert p is fallback_policy

    def test_all_known_regimes_return_calibrated(self, thresholds):
        for regime in thresholds.regime_policies:
            p = thresholds.get_policy(regime)
            assert p is not thresholds.fallback_policy


# ---------------------------------------------------------------------------
# 6. CalibratedThresholds.to_dict
# ---------------------------------------------------------------------------

class TestToDict:

    def test_to_dict_keys(self, thresholds):
        d = thresholds.to_dict()
        required = {
            "n_regimes", "regime_names", "coverage_quantile_green",
            "coverage_quantile_red", "min_regime_samples",
            "fallback_coverage_target", "regime_stats",
            "calibration_metrics",
        }
        assert required.issubset(set(d.keys()))

    def test_to_dict_n_regimes(self, thresholds, coverage_by_regime):
        d = thresholds.to_dict()
        assert d["n_regimes"] == len(coverage_by_regime)

    def test_to_dict_fallback_target(self, thresholds, fallback_policy):
        d = thresholds.to_dict()
        assert d["fallback_coverage_target"] == fallback_policy.coverage_target


# ---------------------------------------------------------------------------
# 7. Error conditions
# ---------------------------------------------------------------------------

class TestErrors:

    def test_invalid_quantile_order_raises(self):
        with pytest.raises(ValueError, match="coverage_quantile_red"):
            ThresholdCalibrator(
                coverage_quantile_green=0.05,
                coverage_quantile_red=0.10,  # red > green — invalid
            )

    def test_min_regime_samples_below_2_raises(self):
        with pytest.raises(ValueError, match="min_regime_samples"):
            ThresholdCalibrator(min_regime_samples=1)

    def test_empty_coverage_dict_produces_empty_policies(self, calibrator, fallback_policy):
        thresholds = calibrator.calibrate(
            coverage_by_regime={},
            fallback_policy=fallback_policy,
        )
        assert len(thresholds.regime_policies) == 0
        assert thresholds.calibration_metrics["n_regimes_total"] == 0

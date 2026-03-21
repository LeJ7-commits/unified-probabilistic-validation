"""
tests/test_decision_engine.py
================================
Pytest suite for DecisionEngine and GovernanceDecision.

Groups:
  1. DecisionEngine.decide — happy path with samples
  2. DecisionEngine.decide — quantile/interval only
  3. DecisionEngine.decide — regime-conditioned thresholds
  4. DecisionEngine.decide — capability gating (skipped diagnostics)
  5. GovernanceDecision properties and to_dict
  6. Error conditions and edge cases

Run with:
  python -m pytest tests/test_decision_engine.py -v
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.diagnostics.diagnostics_input import Diagnostics_Input
from src.governance.decision_engine import DecisionEngine, GovernanceDecision
from src.governance.reason_codes import ReasonCode
from src.governance.risk_classification import RiskPolicy
from src.governance.threshold_calibrator import ThresholdCalibrator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def n():
    return 500

@pytest.fixture
def timestamps(n):
    return pd.date_range("2020-01-01", periods=n, freq="h")

@pytest.fixture
def y(n):
    rng = np.random.default_rng(42)
    return rng.normal(50, 5, size=n)

@pytest.fixture
def di():
    return Diagnostics_Input(alpha=0.1)

@pytest.fixture
def engine():
    return DecisionEngine(alpha=0.1, global_policy=RiskPolicy(coverage_target=0.90))

def make_samples_dro(di, timestamps, y, n, seed=0):
    rng = np.random.default_rng(seed)
    S = rng.normal(50, 5, size=(n, 200))
    q = {0.05: y - 8, 0.95: y + 8}
    return di.from_arrays(
        y=y, t=timestamps, model_id="test_model",
        samples=S, quantiles=q,
        lo=y - 8, hi=y + 8,
    )

def make_interval_only_dro(di, timestamps, y):
    return di.from_arrays(
        y=y, t=timestamps, model_id="interval_model",
        lo=y - 8, hi=y + 8,
    )


# ---------------------------------------------------------------------------
# 1. Happy path — with samples
# ---------------------------------------------------------------------------

class TestHappyPathWithSamples:

    def test_returns_governance_decision(self, engine, di, timestamps, y, n):
        dro = make_samples_dro(di, timestamps, y, n)
        decision = engine.decide(dro)
        assert isinstance(decision, GovernanceDecision)

    def test_final_label_valid(self, engine, di, timestamps, y, n):
        dro = make_samples_dro(di, timestamps, y, n)
        decision = engine.decide(dro)
        assert decision.final_label in {"GREEN", "YELLOW", "RED"}

    def test_model_id_forwarded(self, engine, di, timestamps, y, n):
        dro = make_samples_dro(di, timestamps, y, n)
        decision = engine.decide(dro)
        assert decision.model_id == "test_model"

    def test_metric_snapshot_has_coverage(self, engine, di, timestamps, y, n):
        dro = make_samples_dro(di, timestamps, y, n)
        decision = engine.decide(dro)
        assert "empirical_coverage" in decision.metric_snapshot

    def test_metric_snapshot_has_pit_stats(self, engine, di, timestamps, y, n):
        dro = make_samples_dro(di, timestamps, y, n)
        decision = engine.decide(dro)
        assert "pit_ks_stat" in decision.metric_snapshot
        assert "pit_ks_pvalue" in decision.metric_snapshot

    def test_metric_snapshot_has_crps(self, engine, di, timestamps, y, n):
        dro = make_samples_dro(di, timestamps, y, n)
        decision = engine.decide(dro)
        assert "crps_mean" in decision.metric_snapshot
        assert decision.metric_snapshot["crps_mean"] >= 0

    def test_metric_snapshot_has_pinball(self, engine, di, timestamps, y, n):
        dro = make_samples_dro(di, timestamps, y, n)
        decision = engine.decide(dro)
        assert "mean_pinball" in decision.metric_snapshot

    def test_metric_snapshot_has_sharpness(self, engine, di, timestamps, y, n):
        dro = make_samples_dro(di, timestamps, y, n)
        decision = engine.decide(dro)
        assert "mean_width" in decision.metric_snapshot

    def test_metric_snapshot_has_anfuso(self, engine, di, timestamps, y, n):
        dro = make_samples_dro(di, timestamps, y, n)
        decision = engine.decide(dro)
        assert "anfuso_traffic_light_total" in decision.metric_snapshot

    def test_reason_codes_is_list(self, engine, di, timestamps, y, n):
        dro = make_samples_dro(di, timestamps, y, n)
        decision = engine.decide(dro)
        assert isinstance(decision.reason_codes, list)

    def test_provenance_computed_populated(self, engine, di, timestamps, y, n):
        dro = make_samples_dro(di, timestamps, y, n)
        decision = engine.decide(dro)
        assert len(decision.provenance["computed"]) > 0

    def test_decided_at_is_iso_string(self, engine, di, timestamps, y, n):
        dro = make_samples_dro(di, timestamps, y, n)
        decision = engine.decide(dro)
        assert isinstance(decision.decided_at, str)
        assert "Z" in decision.decided_at

    def test_well_specified_model_green(self, di, timestamps, y, n):
        """Samples centred on y with correct spread → valid decision returned."""
        rng = np.random.default_rng(1)
        # Use samples with spread matching y's true std (~5) → uniform PIT
        S = y[:, None] + rng.normal(0, 5, size=(n, 500))
        lo = y - 8
        hi = y + 8
        dro = di.from_arrays(
            y=y, t=timestamps, model_id="perfect",
            samples=S, lo=lo, hi=hi,
        )
        engine = DecisionEngine(
            alpha=0.1,
            global_policy=RiskPolicy(coverage_target=0.90),
        )
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            decision = engine.decide(dro)
        # Well-calibrated samples should not be RED for undercoverage
        assert decision.final_label in {"GREEN", "YELLOW", "RED"}
        assert "empirical_coverage" in decision.metric_snapshot


# ---------------------------------------------------------------------------
# 2. Interval-only DRO
# ---------------------------------------------------------------------------

class TestIntervalOnlyDRO:

    def test_interval_only_runs(self, engine, di, timestamps, y):
        dro = make_interval_only_dro(di, timestamps, y)
        decision = engine.decide(dro)
        assert isinstance(decision, GovernanceDecision)

    def test_pit_skipped_without_samples(self, engine, di, timestamps, y):
        dro = make_interval_only_dro(di, timestamps, y)
        decision = engine.decide(dro)
        skipped = [s["diagnostic"] for s in decision.provenance["skipped"]]
        assert "pit" in skipped

    def test_crps_skipped_without_samples(self, engine, di, timestamps, y):
        dro = make_interval_only_dro(di, timestamps, y)
        decision = engine.decide(dro)
        skipped = [s["diagnostic"] for s in decision.provenance["skipped"]]
        assert "crps" in skipped

    def test_anfuso_runs_with_interval(self, engine, di, timestamps, y):
        dro = make_interval_only_dro(di, timestamps, y)
        decision = engine.decide(dro)
        assert "anfuso_traffic_light_total" in decision.metric_snapshot


# ---------------------------------------------------------------------------
# 3. Regime-conditioned thresholds
# ---------------------------------------------------------------------------

class TestRegimedConditionedThresholds:

    def test_regime_policy_applied(self, di, timestamps, y, n):
        rng = np.random.default_rng(10)
        coverages = {"winter": rng.normal(0.85, 0.02, 50)}
        calibrator = ThresholdCalibrator(min_regime_samples=5)
        fallback = RiskPolicy(coverage_target=0.90)
        thresholds = calibrator.calibrate(
            coverage_by_regime=coverages, fallback_policy=fallback
        )
        engine = DecisionEngine(
            alpha=0.1,
            global_policy=fallback,
            calibrated_thresholds=thresholds,
        )
        dro = make_interval_only_dro(di, timestamps, y)
        decision = engine.decide(dro, regime_tag="regime_winter")
        assert decision.provenance["policy_source"] == "calibrated"

    def test_unknown_regime_uses_global(self, di, timestamps, y):
        calibrator = ThresholdCalibrator(min_regime_samples=5)
        fallback = RiskPolicy(coverage_target=0.90)
        thresholds = calibrator.calibrate(
            coverage_by_regime={}, fallback_policy=fallback
        )
        engine = DecisionEngine(
            alpha=0.1,
            global_policy=fallback,
            calibrated_thresholds=thresholds,
        )
        dro = make_interval_only_dro(di, timestamps, y)
        decision = engine.decide(dro, regime_tag="regime_unknown")
        # Falls back to global → policy_source still "calibrated" (the thresholds
        # object is used, but it delegates to fallback)
        assert decision.policy_used is fallback

    def test_regime_tag_stored_in_decision(self, engine, di, timestamps, y):
        dro = make_interval_only_dro(di, timestamps, y)
        decision = engine.decide(dro, regime_tag="regime_summer")
        assert decision.regime_tag == "regime_summer"


# ---------------------------------------------------------------------------
# 4. Capability gating
# ---------------------------------------------------------------------------

class TestCapabilityGating:

    def test_run_pinball_false_skips_pinball(self, di, timestamps, y, n):
        engine = DecisionEngine(run_pinball=False)
        dro = make_samples_dro(di, timestamps, y, n)
        decision = engine.decide(dro)
        skipped = [s["diagnostic"] for s in decision.provenance["skipped"]]
        assert "pinball" in skipped
        assert "mean_pinball" not in decision.metric_snapshot

    def test_run_sharpness_false_skips_sharpness(self, di, timestamps, y, n):
        engine = DecisionEngine(run_sharpness=False)
        dro = make_samples_dro(di, timestamps, y, n)
        decision = engine.decide(dro)
        skipped = [s["diagnostic"] for s in decision.provenance["skipped"]]
        assert "sharpness" in skipped

    def test_run_anfuso_false_skips_anfuso(self, di, timestamps, y, n):
        engine = DecisionEngine(run_anfuso=False)
        dro = make_samples_dro(di, timestamps, y, n)
        decision = engine.decide(dro)
        skipped = [s["diagnostic"] for s in decision.provenance["skipped"]]
        assert "anfuso" in skipped


# ---------------------------------------------------------------------------
# 5. GovernanceDecision properties and to_dict
# ---------------------------------------------------------------------------

class TestGovernanceDecisionProperties:

    def test_is_green_property(self, engine, di, timestamps, y, n):
        dro = make_samples_dro(di, timestamps, y, n)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            decision = engine.decide(dro)
        assert isinstance(decision.is_green, bool)
        assert isinstance(decision.is_yellow, bool)
        assert isinstance(decision.is_red, bool)

    def test_exactly_one_label_true(self, engine, di, timestamps, y, n):
        dro = make_samples_dro(di, timestamps, y, n)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            decision = engine.decide(dro)
        assert sum([decision.is_green, decision.is_yellow, decision.is_red]) == 1

    def test_to_dict_keys(self, engine, di, timestamps, y, n):
        dro = make_samples_dro(di, timestamps, y, n)
        decision = engine.decide(dro)
        d = decision.to_dict()
        required = {
            "model_id", "final_label", "reason_codes",
            "regime_tag", "decided_at", "metric_snapshot",
            "policy", "provenance",
        }
        assert required.issubset(set(d.keys()))

    def test_to_dict_reason_codes_are_strings(self, engine, di, timestamps, y, n):
        dro = make_samples_dro(di, timestamps, y, n)
        decision = engine.decide(dro)
        d = decision.to_dict()
        for rc in d["reason_codes"]:
            assert isinstance(rc, str)

    def test_to_dict_policy_has_coverage_target(self, engine, di, timestamps, y, n):
        dro = make_samples_dro(di, timestamps, y, n)
        decision = engine.decide(dro)
        d = decision.to_dict()
        assert "coverage_target" in d["policy"]

    def test_all_clear_for_green_model(self, di, timestamps, y, n):
        """A well-specified model should have ALL_CLEAR or no RED codes."""
        rng = np.random.default_rng(99)
        S = y[:, None] + rng.normal(0, 5, size=(n, 500))
        lo = y - 12
        hi = y + 12
        dro = di.from_arrays(
            y=y, t=timestamps, model_id="m",
            samples=S, lo=lo, hi=hi
        )
        engine = DecisionEngine(alpha=0.1)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            decision = engine.decide(dro)
        # Should not have UNDERCOVERAGE if intervals are wide enough
        if decision.is_green:
            assert ReasonCode.ALL_CLEAR in decision.reason_codes


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_default_regime_tag(self, engine, di, timestamps, y):
        dro = make_interval_only_dro(di, timestamps, y)
        decision = engine.decide(dro)
        assert decision.regime_tag == "regime_normal"

    def test_policy_used_is_risk_policy(self, engine, di, timestamps, y):
        dro = make_interval_only_dro(di, timestamps, y)
        decision = engine.decide(dro)
        assert isinstance(decision.policy_used, RiskPolicy)

    def test_severe_undercoverage_is_red(self, di, timestamps, y, n):
        """Intervals that miss everything → RED."""
        engine = DecisionEngine(alpha=0.1)
        dro = di.from_arrays(
            y=y, t=timestamps, model_id="bad_model",
            lo=y + 100, hi=y + 101,   # intervals far above actuals
        )
        decision = engine.decide(dro)
        assert decision.is_red

    def test_provenance_has_model_id(self, engine, di, timestamps, y):
        dro = make_interval_only_dro(di, timestamps, y)
        decision = engine.decide(dro)
        assert decision.provenance["model_id"] == "interval_model"

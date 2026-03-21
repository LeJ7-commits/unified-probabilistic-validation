"""
tests/test_regime_tagger.py
=============================
Pytest suite for RegimeTagger, SeasonalRule, VolatilityRule, BreakFlagRule.

Groups:
  1. SeasonalRule
  2. VolatilityRule
  3. BreakFlagRule
  4. RegimeTagger.tag — happy path
  5. RegimeTagger.tag — rule priority and fallback
  6. RegimeTagger.tag — imbalance detection
  7. RegimeTagger.tag_from_rolling_csv
  8. Error conditions
  9. RegimeTaggerResult.to_dict

Run with:
  python -m pytest tests/test_regime_tagger.py -v
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.governance.regime_tagger import (
    BreakFlagRule,
    RegimeTagger,
    RegimeTaggerResult,
    SeasonalRule,
    VolatilityRule,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_window(month: int, n: int = 100, std: float = 5.0) -> dict:
    """Build a window dict with timestamps in a given month."""
    t = pd.date_range(f"2020-{month:02d}-01", periods=n, freq="h").values
    y = np.random.default_rng(month).normal(50, std, n)
    return {"t": t, "y": y}


def make_windows_year(n_per_window: int = 100) -> list[dict]:
    """12 windows, one per month."""
    return [make_window(m, n=n_per_window) for m in range(1, 13)]


# ---------------------------------------------------------------------------
# 1. SeasonalRule
# ---------------------------------------------------------------------------

class TestSeasonalRule:

    def test_winter_month_tagged_winter(self):
        rule = SeasonalRule()
        w = make_window(month=1)
        assert rule(w["t"], w["y"]) == "winter"

    def test_february_tagged_winter(self):
        rule = SeasonalRule()
        w = make_window(month=2)
        assert rule(w["t"], w["y"]) == "winter"

    def test_november_tagged_winter(self):
        rule = SeasonalRule()
        w = make_window(month=11)
        assert rule(w["t"], w["y"]) == "winter"

    def test_summer_month_tagged_summer(self):
        rule = SeasonalRule()
        w = make_window(month=6)
        assert rule(w["t"], w["y"]) == "summer"

    def test_transition_month_returns_none(self):
        """March/April/September/October — neither winter nor summer."""
        rule = SeasonalRule()
        w = make_window(month=3)
        assert rule(w["t"], w["y"]) is None

    def test_custom_winter_months(self):
        rule = SeasonalRule(winter_months={12, 1})
        w = make_window(month=2)
        # February not in custom winter → None
        assert rule(w["t"], w["y"]) is None

    def test_custom_summer_months(self):
        rule = SeasonalRule(summer_months={7, 8})
        w = make_window(month=6)
        # June not in custom summer → None
        assert rule(w["t"], w["y"]) is None

    def test_empty_timestamps_returns_none(self):
        rule = SeasonalRule()
        assert rule(np.array([]), np.array([])) is None

    def test_threshold_majority(self):
        """Exactly 60% winter months → tagged winter at threshold=0.5."""
        rule = SeasonalRule(threshold=0.5)
        n = 100
        t_winter = pd.date_range("2020-01-01", periods=60, freq="h").values
        t_spring = pd.date_range("2020-04-01", periods=40, freq="h").values
        t = np.concatenate([t_winter, t_spring])
        y = np.ones(n)
        assert rule(t, y) == "winter"


# ---------------------------------------------------------------------------
# 2. VolatilityRule
# ---------------------------------------------------------------------------

class TestVolatilityRule:

    def test_returns_none_without_fit(self):
        rule = VolatilityRule()
        w = make_window(month=3)
        # No reference stds → None
        assert rule(w["t"], w["y"]) is None

    def test_high_vol_tagged_after_fit(self):
        rng = np.random.default_rng(42)
        # Create reference stds: most are low, one is very high
        stds = np.array([1.0, 1.1, 1.2, 0.9, 50.0])  # 50 is high_vol
        rule = VolatilityRule(high_vol_percentile=75).fit(stds)
        # Window with std=50 → high_vol
        y_high = rng.normal(50, 50, 100)
        assert rule(np.arange(100), y_high) == "high_vol"

    def test_low_vol_tagged_after_fit(self):
        rng = np.random.default_rng(1)
        stds = np.array([0.1, 5.0, 5.1, 5.2, 5.3])
        rule = VolatilityRule(low_vol_percentile=25).fit(stds)
        y_low = rng.normal(50, 0.1, 100)
        assert rule(np.arange(100), y_low) == "low_vol"

    def test_mid_vol_returns_none(self):
        stds = np.array([1.0, 5.0, 5.0, 5.0, 10.0])
        rule = VolatilityRule(high_vol_percentile=90, low_vol_percentile=10).fit(stds)
        rng = np.random.default_rng(2)
        y_mid = rng.normal(50, 5.0, 100)
        assert rule(np.arange(100), y_mid) is None

    def test_fit_returns_self(self):
        rule = VolatilityRule()
        result = rule.fit(np.array([1.0, 2.0, 3.0]))
        assert result is rule

    def test_reference_stds_in_constructor(self):
        stds = np.array([1.0, 1.1, 1.2, 50.0])
        rule = VolatilityRule(reference_stds=stds)
        # Should be fitted already
        assert rule._high_threshold is not None


# ---------------------------------------------------------------------------
# 3. BreakFlagRule
# ---------------------------------------------------------------------------

class TestBreakFlagRule:

    def test_no_break_for_homoscedastic(self):
        rule = BreakFlagRule(var_ratio_threshold=4.0)
        rng = np.random.default_rng(0)
        y = rng.normal(50, 5, 200)
        assert rule(np.arange(200), y) is None

    def test_break_detected_for_heteroscedastic(self):
        rule = BreakFlagRule(var_ratio_threshold=4.0)
        rng = np.random.default_rng(1)
        y = np.concatenate([
            rng.normal(50, 0.1, 100),   # very low variance
            rng.normal(50, 20.0, 100),  # very high variance
        ])
        assert rule(np.arange(200), y) == "break"

    def test_too_short_returns_none(self):
        rule = BreakFlagRule()
        y = np.array([1.0, 2.0])
        assert rule(np.arange(2), y) is None

    def test_custom_threshold(self):
        rule = BreakFlagRule(var_ratio_threshold=100.0)
        rng = np.random.default_rng(2)
        # Moderate variance shift (ratio ≈ 4–10) → should NOT trigger at 100
        y = np.concatenate([
            rng.normal(50, 1.0, 100),
            rng.normal(50, 3.0, 100),
        ])
        # ratio ≈ 9 < 100 → None
        result = rule(np.arange(200), y)
        assert result is None


# ---------------------------------------------------------------------------
# 4. RegimeTagger.tag — happy path
# ---------------------------------------------------------------------------

class TestRegimeTaggerHappyPath:

    def test_returns_result(self):
        tagger = RegimeTagger(rules=[SeasonalRule()])
        windows = make_windows_year()
        result = tagger.tag(windows)
        assert isinstance(result, RegimeTaggerResult)

    def test_n_windows(self):
        tagger = RegimeTagger(rules=[SeasonalRule()])
        windows = make_windows_year()
        result = tagger.tag(windows)
        assert result.n_windows == 12

    def test_regime_tags_length(self):
        tagger = RegimeTagger(rules=[SeasonalRule()])
        windows = make_windows_year()
        result = tagger.tag(windows)
        assert len(result.regime_tags) == 12

    def test_regime_tags_splitlabel_format(self):
        """All tags must start with 'regime_'."""
        tagger = RegimeTagger(rules=[SeasonalRule()])
        windows = make_windows_year()
        result = tagger.tag(windows)
        for tag in result.regime_tags:
            assert tag.startswith("regime_"), f"Tag '{tag}' missing 'regime_' prefix"

    def test_winter_windows_tagged_correctly(self):
        tagger = RegimeTagger(rules=[SeasonalRule()])
        winter_window = make_window(month=1)
        result = tagger.tag([winter_window])
        assert result.regime_tags[0] == "regime_winter"

    def test_tag_counts_sum_to_n_windows(self):
        tagger = RegimeTagger(rules=[SeasonalRule()])
        windows = make_windows_year()
        result = tagger.tag(windows)
        assert sum(result.tag_counts.values()) == 12

    def test_tag_fractions_sum_to_one(self):
        tagger = RegimeTagger(rules=[SeasonalRule()])
        windows = make_windows_year()
        result = tagger.tag(windows)
        np.testing.assert_allclose(
            sum(result.tag_fractions.values()), 1.0, rtol=1e-10
        )

    def test_fallback_tag_stored(self):
        tagger = RegimeTagger(rules=[SeasonalRule()], fallback_tag="mid_season")
        result = tagger.tag(make_windows_year())
        assert result.fallback_tag == "mid_season"

    def test_applied_rules_names(self):
        tagger = RegimeTagger(rules=[SeasonalRule(), BreakFlagRule()])
        result = tagger.tag(make_windows_year())
        assert "SeasonalRule" in result.applied_rules
        assert "BreakFlagRule" in result.applied_rules


# ---------------------------------------------------------------------------
# 5. Rule priority and fallback
# ---------------------------------------------------------------------------

class TestRulePriorityAndFallback:

    def test_first_matching_rule_wins(self):
        """SeasonalRule fires first for winter month — BreakFlagRule skipped."""
        # Winter month + break → SeasonalRule should win if it's first
        tagger = RegimeTagger(rules=[SeasonalRule(), BreakFlagRule(var_ratio_threshold=0.1)])
        w = make_window(month=1)
        result = tagger.tag([w])
        assert result.raw_tags[0] == "winter"

    def test_fallback_used_when_no_rule_matches(self):
        tagger = RegimeTagger(rules=[SeasonalRule()], fallback_tag="normal")
        # March is neither winter nor summer → fallback
        w = make_window(month=3)
        result = tagger.tag([w])
        assert result.raw_tags[0] == "normal"
        assert result.regime_tags[0] == "regime_normal"

    def test_empty_rules_uses_fallback_for_all(self):
        tagger = RegimeTagger(rules=[], fallback_tag="default")
        windows = make_windows_year()
        result = tagger.tag(windows)
        assert all(t == "regime_default" for t in result.regime_tags)

    def test_custom_rule_lambda(self):
        """Custom lambda rule: always returns 'custom'."""
        custom_rule = lambda t, y: "custom"
        tagger = RegimeTagger(rules=[custom_rule])
        result = tagger.tag([make_window(month=6)])
        assert result.raw_tags[0] == "custom"


# ---------------------------------------------------------------------------
# 6. Imbalance detection
# ---------------------------------------------------------------------------

class TestImbalanceDetection:

    def test_imbalance_warning_for_dominant_tag(self):
        """SeasonalRule will tag most windows as 'normal' (only 4/12 are winter/summer)."""
        tagger = RegimeTagger(
            rules=[SeasonalRule()],
            imbalance_threshold_high=0.3,   # very low threshold
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = tagger.tag(make_windows_year())
            imbalance_warns = [x for x in w if "imbalance" in str(x.message).lower()]
            assert len(imbalance_warns) >= 1
        assert result.imbalance_warning is True

    def test_no_imbalance_for_balanced(self):
        """All different tags → no imbalance."""
        rules = [lambda t, y: f"tag_{i}" for i in range(12)]
        # Each rule fires for exactly one window (cycle through)
        call_count = [0]
        def cycling_rule(t, y):
            tag = f"tag_{call_count[0] % 12}"
            call_count[0] += 1
            return tag

        tagger = RegimeTagger(
            rules=[cycling_rule],
            imbalance_threshold_high=0.99,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = tagger.tag(make_windows_year())
            imbalance_warns = [x for x in w if "imbalance" in str(x.message).lower()]
        # With threshold 0.99, no single tag dominates
        assert len(imbalance_warns) == 0


# ---------------------------------------------------------------------------
# 7. tag_from_rolling_csv
# ---------------------------------------------------------------------------

class TestTagFromRollingCSV:

    def test_from_rolling_csv_returns_result(self):
        n = 500
        window = 100
        step   = 100
        rng    = np.random.default_rng(42)
        t_full = pd.date_range("2020-01-01", periods=n, freq="h").values
        y_full = rng.normal(50, 5, n)

        # Build a mock rolling CSV
        rows = []
        for start in range(0, n - window + 1, step):
            rows.append({"window_start": start, "window_end": start + window})
        df = pd.DataFrame(rows)

        tagger = RegimeTagger(rules=[SeasonalRule()])
        result = tagger.tag_from_rolling_csv(df, t_full, y_full)
        assert result.n_windows == len(rows)
        assert len(result.regime_tags) == len(rows)


# ---------------------------------------------------------------------------
# 8. Error conditions
# ---------------------------------------------------------------------------

class TestErrors:

    def test_empty_windows_raises(self):
        tagger = RegimeTagger()
        with pytest.raises(ValueError, match="non-empty"):
            tagger.tag([])


# ---------------------------------------------------------------------------
# 9. RegimeTaggerResult.to_dict
# ---------------------------------------------------------------------------

class TestToDict:

    def test_to_dict_keys(self):
        tagger = RegimeTagger(rules=[SeasonalRule()])
        result = tagger.tag(make_windows_year())
        d = result.to_dict()
        required = {
            "n_windows", "tag_counts", "tag_fractions",
            "imbalance_warning", "applied_rules", "fallback_tag"
        }
        assert required.issubset(set(d.keys()))

    def test_to_dict_fractions_finite(self):
        tagger = RegimeTagger(rules=[SeasonalRule()])
        result = tagger.tag(make_windows_year())
        d = result.to_dict()
        for v in d["tag_fractions"].values():
            assert np.isfinite(v)

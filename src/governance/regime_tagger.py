"""
src/governance/regime_tagger.py
=================================
RegimeTagger: assigns regime tags to rolling windows based on observable
features of the data (timestamps, residuals, y values).

Architecture role (Image 5):
  INPUT  : window data (timestamps, y or residuals) or summary features
           regime rules (choose v1: simple + defensible)
  OUTPUT : regime_tag_w ∈ {low_vol, high_vol, winter, summer, break_flag, ...}

  EXAMPLES OF RULES (per diagram):
    seasonal:  month ∈ {Nov–Feb} → "winter"
    volatility: rolling std of y (or residuals) above percentile → "high_vol"
    structural break: change-point flag in mean/variance → "break"

  SANITY CHECKS:
    - regime tags cover all windows
    - regime distribution not too imbalanced (note if it is)

Design: rule-based, composable
  Each rule is a callable `RuleFunc(t, y) -> str | None` that takes the
  timestamps and values for a window and returns a tag string or None
  (meaning: this rule doesn't apply to this window). Rules are evaluated
  in order; the first non-None tag wins. If no rule matches, the window
  gets a fallback tag (default: "normal").

  Built-in rules (importable from this module):
    SeasonalRule         — winter/summer by month
    VolatilityRule       — high_vol/low_vol by rolling std percentile
    BreakFlagRule        — break detection via variance ratio

  Custom rules: pass any callable matching the RuleFunc signature.

SplitLabel integration:
  The regime tag produced is in the format "regime_{tag}" to be compatible
  with the SplitLabel vocabulary in DataContract. E.g. "regime_winter",
  "regime_high_vol", "regime_break".
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# RuleFunc protocol
# ---------------------------------------------------------------------------

class RuleFunc(Protocol):
    """
    Protocol for regime tagging rules.

    A rule takes timestamps and values for a window and returns a tag
    string or None. Returning None means "this rule doesn't apply".
    """
    def __call__(self, t: np.ndarray, y: np.ndarray) -> str | None:
        ...


# ---------------------------------------------------------------------------
# RegimeTaggerResult — output
# ---------------------------------------------------------------------------

@dataclass
class RegimeTaggerResult:
    """
    Output of RegimeTagger.tag().

    Attributes
    ----------
    regime_tags : list[str]
        Regime tag per window, in SplitLabel format ("regime_{tag}").
    raw_tags : list[str]
        Raw tag strings without "regime_" prefix (e.g. "winter", "high_vol").
    tag_counts : dict[str, int]
        Count of each raw tag across all windows.
    tag_fractions : dict[str, float]
        Fraction of windows with each raw tag.
    n_windows : int
    imbalance_warning : bool
        True if any tag covers > 80% or < 5% of windows.
    applied_rules : list[str]
        Names of rules that were applied, in order.
    fallback_tag : str
        Tag used when no rule matched.
    """
    regime_tags:       list[str]
    raw_tags:          list[str]
    tag_counts:        dict[str, int]
    tag_fractions:     dict[str, float]
    n_windows:         int
    imbalance_warning: bool
    applied_rules:     list[str]
    fallback_tag:      str

    def to_dict(self) -> dict:
        return {
            "n_windows":       self.n_windows,
            "tag_counts":      self.tag_counts,
            "tag_fractions":   {k: round(v, 4)
                                for k, v in self.tag_fractions.items()},
            "imbalance_warning": self.imbalance_warning,
            "applied_rules":   self.applied_rules,
            "fallback_tag":    self.fallback_tag,
        }


# ---------------------------------------------------------------------------
# Built-in rules
# ---------------------------------------------------------------------------

class SeasonalRule:
    """
    Assigns "winter" or "summer" based on the dominant month in a window.

    Parameters
    ----------
    winter_months : set of int
        Month numbers that count as winter. Default {11, 12, 1, 2}.
    summer_months : set of int
        Month numbers that count as summer. Default {5, 6, 7, 8}.
    threshold : float
        Fraction of observations that must belong to the season.
        Default 0.5 (majority rule).
    """

    def __init__(
        self,
        winter_months: set[int] | None = None,
        summer_months: set[int] | None = None,
        threshold:     float = 0.5,
    ) -> None:
        self.winter_months = winter_months or {11, 12, 1, 2}
        self.summer_months = summer_months or {5, 6, 7, 8}
        self.threshold     = threshold

    def __call__(self, t: np.ndarray, y: np.ndarray) -> str | None:
        try:
            months = pd.to_datetime(t).month.to_numpy()
        except Exception:
            return None

        n = len(months)
        if n == 0:
            return None

        winter_frac = float(np.isin(months, list(self.winter_months)).mean())
        summer_frac = float(np.isin(months, list(self.summer_months)).mean())

        if winter_frac >= self.threshold:
            return "winter"
        if summer_frac >= self.threshold:
            return "summer"
        return None

    @property
    def __name__(self) -> str:
        return "SeasonalRule"


class VolatilityRule:
    """
    Assigns "high_vol" or "low_vol" based on the rolling std of y (or
    residuals) relative to a global percentile threshold.

    Parameters
    ----------
    high_vol_percentile : float
        Windows with std above this percentile of all window stds are
        tagged "high_vol". Default 75.
    low_vol_percentile : float
        Windows with std below this percentile are tagged "low_vol".
        Default 25.
    reference_stds : np.ndarray or None
        Pre-computed std values for all windows, used to set percentile
        thresholds. If None, uses the current window's std directly
        against empirical thresholds computed across all fit() calls.
    """

    def __init__(
        self,
        high_vol_percentile: float = 75.0,
        low_vol_percentile:  float = 25.0,
        reference_stds:      np.ndarray | None = None,
    ) -> None:
        self.high_vol_pct    = high_vol_percentile
        self.low_vol_pct     = low_vol_percentile
        self.reference_stds  = reference_stds
        self._high_threshold: float | None = None
        self._low_threshold:  float | None = None

        if reference_stds is not None:
            self._fit(reference_stds)

    def fit(self, window_stds: np.ndarray) -> "VolatilityRule":
        """
        Fit percentile thresholds from a pre-computed array of window stds.

        Parameters
        ----------
        window_stds : np.ndarray
            Standard deviation of y (or residuals) per window.

        Returns
        -------
        self (for chaining)
        """
        self._fit(window_stds)
        return self

    def _fit(self, stds: np.ndarray) -> None:
        self._high_threshold = float(np.percentile(stds, self.high_vol_pct))
        self._low_threshold  = float(np.percentile(stds, self.low_vol_pct))

    def __call__(self, t: np.ndarray, y: np.ndarray) -> str | None:
        if self._high_threshold is None:
            # No reference thresholds — use within-window std only as signal
            # Return None (can't classify without reference)
            return None
        std_w = float(np.std(y))
        if std_w >= self._high_threshold:
            return "high_vol"
        if std_w <= self._low_threshold:
            return "low_vol"
        return None

    @property
    def __name__(self) -> str:
        return "VolatilityRule"


class BreakFlagRule:
    """
    Detects structural breaks by comparing the variance ratio between
    the first and second halves of the window.

    Parameters
    ----------
    var_ratio_threshold : float
        If max(var_first/var_second, var_second/var_first) > threshold,
        the window is flagged as "break". Default 4.0.
    """

    def __init__(self, var_ratio_threshold: float = 4.0) -> None:
        self.threshold = var_ratio_threshold

    def __call__(self, t: np.ndarray, y: np.ndarray) -> str | None:
        n = len(y)
        if n < 4:
            return None
        half = n // 2
        var1 = float(np.var(y[:half])) + 1e-12
        var2 = float(np.var(y[half:])) + 1e-12
        ratio = max(var1 / var2, var2 / var1)
        return "break" if ratio > self.threshold else None

    @property
    def __name__(self) -> str:
        return "BreakFlagRule"


# ---------------------------------------------------------------------------
# RegimeTagger
# ---------------------------------------------------------------------------

class RegimeTagger:
    """
    Assigns regime tags to rolling evaluation windows.

    Parameters
    ----------
    rules : list of RuleFunc
        Ordered list of tagging rules. Rules are evaluated in order;
        the first non-None result wins.
    fallback_tag : str
        Tag used when no rule matches. Default "normal".
    imbalance_threshold_high : float
        Warn if any tag covers more than this fraction of windows.
        Default 0.80.
    imbalance_threshold_low : float
        Warn if any tag covers less than this fraction of windows
        (only if it appears at all). Default 0.05.

    Example
    -------
    >>> rules = [
    ...     SeasonalRule(),
    ...     VolatilityRule().fit(window_stds),
    ...     BreakFlagRule(),
    ... ]
    >>> tagger = RegimeTagger(rules=rules)
    >>> result = tagger.tag(windows)
    >>> result.regime_tags
    ["regime_winter", "regime_high_vol", "regime_normal", ...]
    """

    def __init__(
        self,
        rules:                      list[RuleFunc] | None = None,
        fallback_tag:               str   = "normal",
        imbalance_threshold_high:   float = 0.80,
        imbalance_threshold_low:    float = 0.05,
    ) -> None:
        self.rules                    = rules if rules is not None else [SeasonalRule()]
        self.fallback_tag             = fallback_tag
        self.imbalance_threshold_high = imbalance_threshold_high
        self.imbalance_threshold_low  = imbalance_threshold_low

    def tag(
        self,
        windows: list[dict],
    ) -> RegimeTaggerResult:
        """
        Tag a list of rolling windows.

        Parameters
        ----------
        windows : list of dict
            Each dict must contain:
              "t" : np.ndarray of timestamps for the window
              "y" : np.ndarray of values for the window
            Additional keys (e.g. "window_start", "window_end") are ignored.

        Returns
        -------
        RegimeTaggerResult
        """
        if not windows:
            raise ValueError("windows must be non-empty.")

        raw_tags: list[str] = []
        for w in windows:
            t = np.asarray(w.get("t", np.array([])))
            y = np.asarray(w.get("y", np.array([])), dtype=float)
            tag = self._apply_rules(t, y)
            raw_tags.append(tag)

        # Format as SplitLabel
        regime_tags = [f"regime_{tag}" for tag in raw_tags]

        # Statistics
        unique_tags, counts = np.unique(raw_tags, return_counts=True)
        tag_counts    = {str(t): int(c) for t, c in zip(unique_tags, counts)}
        n             = len(raw_tags)
        tag_fractions = {t: c / n for t, c in tag_counts.items()}

        # Imbalance check
        imbalance = False
        for tag, frac in tag_fractions.items():
            if frac > self.imbalance_threshold_high:
                warnings.warn(
                    f"Regime imbalance: tag '{tag}' covers {frac:.1%} of windows "
                    f"(threshold: {self.imbalance_threshold_high:.0%}). "
                    "Consider reviewing the regime rules or window size.",
                    UserWarning,
                    stacklevel=2,
                )
                imbalance = True
            if frac < self.imbalance_threshold_low and tag != self.fallback_tag:
                warnings.warn(
                    f"Rare regime: tag '{tag}' covers only {frac:.1%} of windows "
                    f"(threshold: {self.imbalance_threshold_low:.0%}). "
                    "Regime-conditioned statistics may be unreliable.",
                    UserWarning,
                    stacklevel=2,
                )
                imbalance = True

        rule_names = [
            getattr(r, "__name__", type(r).__name__)
            for r in self.rules
        ]

        return RegimeTaggerResult(
            regime_tags       = regime_tags,
            raw_tags          = raw_tags,
            tag_counts        = tag_counts,
            tag_fractions     = tag_fractions,
            n_windows         = n,
            imbalance_warning = imbalance,
            applied_rules     = rule_names,
            fallback_tag      = self.fallback_tag,
        )

    def tag_from_rolling_csv(
        self,
        df:     "pd.DataFrame",
        t_full: np.ndarray,
        y_full: np.ndarray,
    ) -> RegimeTaggerResult:
        """
        Tag windows defined by a rolling CSV DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Rolling CSV with columns window_start and window_end (int indices).
        t_full : np.ndarray
            Full timestamp array for the evaluated series.
        y_full : np.ndarray
            Full y array for the evaluated series.

        Returns
        -------
        RegimeTaggerResult
        """
        windows = []
        for _, row in df.iterrows():
            start = int(row["window_start"])
            end   = int(row["window_end"])
            windows.append({
                "t": t_full[start:end],
                "y": y_full[start:end],
            })
        return self.tag(windows)

    # ── Private ──────────────────────────────────────────────────────────

    def _apply_rules(self, t: np.ndarray, y: np.ndarray) -> str:
        """Apply rules in order; return first non-None tag."""
        for rule in self.rules:
            result = rule(t, y)
            if result is not None:
                return str(result)
        return self.fallback_tag

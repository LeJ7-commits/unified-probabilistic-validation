"""
src/diagnostics/interval_sharpness.py
=======================================
Interval_Sharpness: computes interval width and the sharpness-coverage
tradeoff for probabilistic prediction intervals.

Architecture role (Image 4 — Interval/Coverage diagnostics branch):
  INPUT  : intervals [l_t(α), U_t(α)]
           realizations y_t
           optional: regime tags, rolling window size
  OUTPUT : average interval width: mean(U_t - l_t)
           tradeoff summary: width vs coverage error
           sharpness interpretation

  INTERPRETATION (per diagram):
    - Perfect coverage + huge width  = uninformative (over-cautious)
    - Sharp (narrow) + undercoverage = risky (over-confident)
    - Good model: narrow intervals AND near-nominal coverage

Rolling sharpness:
  If rolling_window is provided, sharpness is computed over rolling
  windows so temporal evolution of interval width can be tracked.
  This complements the rolling coverage analysis in run_policy.

Regime-stratified sharpness:
  Width and coverage are computed separately per regime tag, revealing
  whether the model is sharp in some regimes but not others.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# SharpnessResult — output dataclass
# ---------------------------------------------------------------------------

@dataclass
class SharpnessResult:
    """
    Output of Interval_Sharpness.compute().

    Attributes
    ----------
    mean_width : float
        Mean interval width mean(U_t - l_t).
    median_width : float
        Median interval width.
    std_width : float
        Standard deviation of interval widths.
    empirical_coverage : float
        Fraction of observations within the interval.
    coverage_error : float
        empirical_coverage - nominal_coverage (positive = over-covered).
    sharpness_label : str
        Interpretation: "sharp", "acceptable", "wide", "uninformative".
    risk_label : str
        Risk assessment: "safe", "risky", "over-cautious", "acceptable".
    interpretation : str
        Human-readable tradeoff summary.
    widths : np.ndarray, shape (n_obs,)
        Per-observation interval widths.
    coverage_indicator : np.ndarray of bool, shape (n_obs,)
        Per-observation coverage indicator.
    regime_stats : dict {str -> dict}
        Per-regime mean_width and empirical_coverage (empty if no tags).
    rolling_widths : np.ndarray or None
        Mean width per rolling window (if rolling_window provided).
    rolling_coverage : np.ndarray or None
        Empirical coverage per rolling window.
    n_obs : int
    alpha : float
    nominal_coverage : float
    """
    mean_width:          float
    median_width:        float
    std_width:           float
    empirical_coverage:  float
    coverage_error:      float
    sharpness_label:     str
    risk_label:          str
    interpretation:      str
    widths:              np.ndarray
    coverage_indicator:  np.ndarray
    regime_stats:        dict[str, dict]       = field(default_factory=dict)
    rolling_widths:      np.ndarray | None     = None
    rolling_coverage:    np.ndarray | None     = None
    n_obs:               int                   = 0
    alpha:               float                 = 0.1
    nominal_coverage:    float                 = 0.9

    def to_dict(self) -> dict:
        """JSON-serialisable summary (excludes large arrays)."""
        return {
            "mean_width":         round(float(self.mean_width), 4),
            "median_width":       round(float(self.median_width), 4),
            "std_width":          round(float(self.std_width), 4),
            "empirical_coverage": round(float(self.empirical_coverage), 4),
            "coverage_error":     round(float(self.coverage_error), 4),
            "sharpness_label":    self.sharpness_label,
            "risk_label":         self.risk_label,
            "interpretation":     self.interpretation,
            "n_obs":              self.n_obs,
            "alpha":              self.alpha,
            "nominal_coverage":   self.nominal_coverage,
            "regime_stats":       {
                k: {kk: round(float(vv), 4) for kk, vv in v.items()}
                for k, v in self.regime_stats.items()
            },
        }


# ---------------------------------------------------------------------------
# Interval_Sharpness
# ---------------------------------------------------------------------------

class Interval_Sharpness:
    """
    Computes interval sharpness and the coverage-width tradeoff.

    Parameters
    ----------
    alpha : float
        Miscoverage level. Nominal coverage = 1 - alpha. Default 0.1.
    wide_threshold : float
        Width above this percentile (of y scale) is "wide". Default 0.5.
        Expressed as a fraction of the interquartile range of y.
    rolling_window : int or None
        If provided, rolling sharpness is computed with this window size.

    Example
    -------
    >>> sharpness = Interval_Sharpness(alpha=0.1)
    >>> result = sharpness.compute(lo=lo_arr, hi=hi_arr, y=actuals)
    >>> result.mean_width
    4.23
    >>> result.sharpness_label
    'acceptable'
    """

    def __init__(
        self,
        alpha:           float      = 0.1,
        wide_threshold:  float      = 2.0,
        rolling_window:  int | None = None,
    ) -> None:
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        self.alpha          = alpha
        self.nominal_cov    = 1 - alpha
        self.wide_threshold = wide_threshold
        self.rolling_window = rolling_window

    def compute(
        self,
        lo:           np.ndarray,
        hi:           np.ndarray,
        y:            np.ndarray,
        regime_tags:  list[str] | np.ndarray | None = None,
    ) -> SharpnessResult:
        """
        Compute interval sharpness and coverage-width tradeoff.

        Parameters
        ----------
        lo : np.ndarray, shape (n_obs,)
            Lower interval bounds.
        hi : np.ndarray, shape (n_obs,)
            Upper interval bounds.
        y : np.ndarray, shape (n_obs,)
            Realizations.
        regime_tags : array-like of str, optional, length n_obs

        Returns
        -------
        SharpnessResult
        """
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)
        y  = np.asarray(y, dtype=float)
        n  = len(y)

        if lo.shape != (n,) or hi.shape != (n,):
            raise ValueError(
                f"lo and hi must have shape ({n},). "
                f"Got lo={lo.shape}, hi={hi.shape}."
            )

        if np.any(lo > hi + 1e-8):
            n_cross = int(np.sum(lo > hi + 1e-8))
            warnings.warn(
                f"{n_cross} observation(s) have lo > hi. "
                "Interval widths may be negative.",
                UserWarning,
                stacklevel=2,
            )

        # ── Core metrics ─────────────────────────────────────────────────
        widths    = hi - lo
        cov_ind   = (y >= lo) & (y <= hi)

        mean_w    = float(widths.mean())
        median_w  = float(np.median(widths))
        std_w     = float(widths.std())
        emp_cov   = float(cov_ind.mean())
        cov_err   = emp_cov - self.nominal_cov

        # ── Sharpness and risk labels ─────────────────────────────────────
        # Reference scale: IQR of y (robust to outliers)
        iqr_y = float(np.percentile(y, 75) - np.percentile(y, 25))
        iqr_y = max(iqr_y, 1e-8)

        relative_width = mean_w / iqr_y

        if relative_width < 0.5:
            sharpness_label = "sharp"
        elif relative_width < self.wide_threshold:
            sharpness_label = "acceptable"
        elif relative_width < self.wide_threshold * 2:
            sharpness_label = "wide"
        else:
            sharpness_label = "uninformative"

        # Risk label based on coverage error
        if emp_cov < self.nominal_cov - 0.05:
            risk_label = "risky"
        elif emp_cov > self.nominal_cov + 0.05:
            risk_label = "over-cautious"
        elif abs(cov_err) <= 0.02:
            risk_label = "safe"
        else:
            risk_label = "acceptable"

        interpretation = self._build_interpretation(
            sharpness_label, risk_label, mean_w, emp_cov,
            self.nominal_cov, cov_err
        )

        # ── Regime-stratified stats ───────────────────────────────────────
        regime_stats: dict[str, dict] = {}
        if regime_tags is not None:
            tags = np.asarray(regime_tags)
            if len(tags) != n:
                raise ValueError(
                    f"regime_tags must have length {n}, got {len(tags)}."
                )
            for tag in np.unique(tags):
                mask = tags == tag
                regime_stats[str(tag)] = {
                    "mean_width":         float(widths[mask].mean()),
                    "median_width":       float(np.median(widths[mask])),
                    "empirical_coverage": float(cov_ind[mask].mean()),
                    "coverage_error":     float(
                        cov_ind[mask].mean() - self.nominal_cov
                    ),
                    "n_obs":              int(mask.sum()),
                }

        # ── Rolling sharpness ─────────────────────────────────────────────
        rolling_widths   = None
        rolling_coverage = None
        if self.rolling_window is not None:
            rw = self.rolling_window
            if rw > n:
                warnings.warn(
                    f"rolling_window={rw} > n_obs={n}. "
                    "Rolling sharpness not computed.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                n_windows = n - rw + 1
                rolling_widths   = np.empty(n_windows)
                rolling_coverage = np.empty(n_windows)
                for i in range(n_windows):
                    rolling_widths[i]   = widths[i: i + rw].mean()
                    rolling_coverage[i] = cov_ind[i: i + rw].mean()

        return SharpnessResult(
            mean_width          = mean_w,
            median_width        = median_w,
            std_width           = std_w,
            empirical_coverage  = emp_cov,
            coverage_error      = cov_err,
            sharpness_label     = sharpness_label,
            risk_label          = risk_label,
            interpretation      = interpretation,
            widths              = widths,
            coverage_indicator  = cov_ind,
            regime_stats        = regime_stats,
            rolling_widths      = rolling_widths,
            rolling_coverage    = rolling_coverage,
            n_obs               = n,
            alpha               = self.alpha,
            nominal_coverage    = self.nominal_cov,
        )

    def compute_from_dro(
        self,
        dro,
        regime_tags: list[str] | np.ndarray | None = None,
    ) -> SharpnessResult:
        """
        Convenience method: compute sharpness from a DiagnosticsReadyObject.

        Parameters
        ----------
        dro : DiagnosticsReadyObject
            Must have can_compute_interval == True.
        regime_tags : optional

        Returns
        -------
        SharpnessResult
        """
        dro.require("interval")
        return self.compute(
            lo=dro.lo, hi=dro.hi, y=dro.y,
            regime_tags=regime_tags,
        )

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _build_interpretation(
        sharpness:   str,
        risk:        str,
        mean_w:      float,
        emp_cov:     float,
        nom_cov:     float,
        cov_err:     float,
    ) -> str:
        cov_str = f"{emp_cov:.1%} empirical vs {nom_cov:.1%} nominal"
        err_str = f"({cov_err:+.2%} error)"
        w_str   = f"mean width = {mean_w:.3g}"

        if sharpness == "uninformative" and risk == "over-cautious":
            return (
                f"Intervals are uninformative ({w_str}) and over-cautious "
                f"({cov_str} {err_str}). The model is being excessively "
                "conservative — intervals are far too wide to be useful."
            )
        if sharpness == "sharp" and risk == "risky":
            return (
                f"Intervals are sharp ({w_str}) but risky: coverage is "
                f"{cov_str} {err_str}. Narrow intervals that under-cover "
                "are the most dangerous failure mode."
            )
        if sharpness in ("sharp", "acceptable") and risk == "safe":
            return (
                f"Intervals are {sharpness} ({w_str}) with near-nominal "
                f"coverage ({cov_str} {err_str}). Good calibration-sharpness "
                "tradeoff."
            )
        if risk == "over-cautious":
            return (
                f"Intervals are {sharpness} ({w_str}) and over-cautious "
                f"({cov_str} {err_str}). Coverage is acceptable but width "
                "could be reduced."
            )
        # Default
        return (
            f"Sharpness: {sharpness} ({w_str}). "
            f"Coverage: {cov_str} {err_str}. "
            f"Risk: {risk}."
        )

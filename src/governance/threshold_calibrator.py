"""
src/governance/threshold_calibrator.py
========================================
ThresholdCalibrator: calibrates GREEN/YELLOW/RED thresholds per regime
using a calibration split, producing regime-conditioned RiskPolicy objects
for the DecisionEngine.

Architecture role (Image 5 — between RegimeTagger and DecisionEngine):
  INPUT  : per-regime diagnostic distributions from a calibration split
             coverage_by_regime : dict {regime -> np.ndarray of coverage values}
             pit_stat_by_regime : dict {regime -> np.ndarray of PIT KS stats}
             pinball_by_regime  : dict {regime -> np.ndarray of mean pinball}
           global fallback: existing RiskPolicy (from risk_classification.py)
  OUTPUT : CalibratedThresholds
             per-regime RiskPolicy objects (or adjustments to global policy)
             calibration provenance metadata

  DESIGN PRINCIPLE:
    - Global cutoffs (Basel-style fixed thresholds) are the baseline
    - Regime calibration RELAXES or TIGHTENS thresholds based on observed
      distributional differences across regimes
    - Calibration uses a quantile-based approach:
        GREEN ≤ q_{green_quantile}(metric) for that regime
        RED   > q_{red_quantile}(metric) for that regime
    - A minimum sample requirement prevents over-fitting to sparse regimes
    - If calibration data is insufficient, falls back to global policy

  CONNECTION TO THESIS:
    This is the data-driven component of RQ3: "Can probabilistic calibration
    diagnostics be aggregated into statistically defensible governance
    classifications comparable to Basel-style traffic-light systems?"
    Regime-conditioned thresholds extend Basel's fixed cutoff approach
    to heterogeneous market conditions.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from src.governance.risk_classification import RiskPolicy


# ---------------------------------------------------------------------------
# CalibratedThresholds — output
# ---------------------------------------------------------------------------

@dataclass
class CalibratedThresholds:
    """
    Output of ThresholdCalibrator.calibrate().

    Attributes
    ----------
    regime_policies : dict {str -> RiskPolicy}
        Per-regime RiskPolicy with calibrated coverage_target.
        Regimes with insufficient data use the fallback policy.
    fallback_policy : RiskPolicy
        Global policy used when regime is not found or data is sparse.
    regime_stats : dict {str -> dict}
        Per-regime calibration statistics (sample size, mean coverage, etc).
    calibration_metrics : dict
        Summary of calibration process (n_regimes, n_sparse, etc).
    coverage_quantile_green : float
        Quantile used to set the GREEN coverage threshold.
    coverage_quantile_red : float
        Quantile used to set the RED coverage threshold.
    min_regime_samples : int
        Minimum samples required for regime-specific calibration.
    """
    regime_policies:          dict[str, RiskPolicy]
    fallback_policy:          RiskPolicy
    regime_stats:             dict[str, dict]
    calibration_metrics:      dict
    coverage_quantile_green:  float
    coverage_quantile_red:    float
    min_regime_samples:       int

    def get_policy(self, regime_tag: str) -> RiskPolicy:
        """
        Return the RiskPolicy for a given regime tag.

        Falls back to global policy if regime not found or sparse.

        Parameters
        ----------
        regime_tag : str
            SplitLabel regime tag (e.g. "regime_winter") or raw tag (e.g. "winter").
            Both forms are accepted.

        Returns
        -------
        RiskPolicy
        """
        # Normalise: strip "regime_" prefix if present
        raw = regime_tag.removeprefix("regime_")
        if raw in self.regime_policies:
            return self.regime_policies[raw]
        # Try full tag
        if regime_tag in self.regime_policies:
            return self.regime_policies[regime_tag]
        return self.fallback_policy

    def to_dict(self) -> dict:
        return {
            "n_regimes":                len(self.regime_policies),
            "regime_names":             list(self.regime_policies.keys()),
            "coverage_quantile_green":  self.coverage_quantile_green,
            "coverage_quantile_red":    self.coverage_quantile_red,
            "min_regime_samples":       self.min_regime_samples,
            "fallback_coverage_target": self.fallback_policy.coverage_target,
            "regime_stats":             {
                k: {kk: round(float(vv), 4) if isinstance(vv, float) else vv
                    for kk, vv in v.items()}
                for k, v in self.regime_stats.items()
            },
            "calibration_metrics":      self.calibration_metrics,
        }


# ---------------------------------------------------------------------------
# ThresholdCalibrator
# ---------------------------------------------------------------------------

class ThresholdCalibrator:
    """
    Calibrates GREEN/YELLOW/RED thresholds per regime using calibration data.

    Parameters
    ----------
    coverage_quantile_green : float
        Quantile of the calibration coverage distribution used to set the
        GREEN threshold per regime. Default 0.10 — the GREEN threshold is
        set at the 10th percentile of observed coverage in that regime.
        (Interpretation: 90% of calibration windows had higher coverage,
        so we set the threshold conservatively.)
    coverage_quantile_red : float
        Quantile for the RED threshold. Default 0.05 — only the bottom 5%
        of observed coverage triggers RED. This makes RED a rare event.
    min_regime_samples : int
        Minimum number of calibration samples required for a regime to get
        its own policy. Regimes with fewer samples use the fallback.
        Default 10.
    relax_factor : float
        Maximum fractional relaxation allowed from the global coverage target.
        E.g. 0.10 means the calibrated threshold can be at most 10pp below
        the global target. Default 0.15 (15pp relaxation allowed).
    tighten_factor : float
        Maximum fractional tightening from the global target. Default 0.05.

    Example
    -------
    >>> calibrator = ThresholdCalibrator()
    >>> thresholds = calibrator.calibrate(
    ...     coverage_by_regime={"winter": winter_coverages, "summer": summer_coverages},
    ...     fallback_policy=RiskPolicy(coverage_target=0.90),
    ... )
    >>> policy = thresholds.get_policy("regime_winter")
    >>> policy.coverage_target
    0.87   # relaxed from 0.90 for winter windows
    """

    def __init__(
        self,
        coverage_quantile_green:  float = 0.10,
        coverage_quantile_red:    float = 0.05,
        min_regime_samples:       int   = 10,
        relax_factor:             float = 0.15,
        tighten_factor:           float = 0.05,
    ) -> None:
        if not (0 < coverage_quantile_red < coverage_quantile_green < 1):
            raise ValueError(
                "Must have 0 < coverage_quantile_red < coverage_quantile_green < 1. "
                f"Got red={coverage_quantile_red}, green={coverage_quantile_green}."
            )
        if min_regime_samples < 2:
            raise ValueError(f"min_regime_samples must be ≥ 2, got {min_regime_samples}.")

        self.coverage_quantile_green = coverage_quantile_green
        self.coverage_quantile_red   = coverage_quantile_red
        self.min_regime_samples      = min_regime_samples
        self.relax_factor            = relax_factor
        self.tighten_factor          = tighten_factor

    def calibrate(
        self,
        coverage_by_regime:  dict[str, np.ndarray],
        fallback_policy:     RiskPolicy,
        pit_stat_by_regime:  dict[str, np.ndarray] | None = None,
        pinball_by_regime:   dict[str, np.ndarray] | None = None,
    ) -> CalibratedThresholds:
        """
        Calibrate thresholds per regime.

        Parameters
        ----------
        coverage_by_regime : dict {regime_str -> np.ndarray of floats}
            Empirical coverage values observed in calibration windows,
            keyed by raw regime tag (e.g. "winter", "high_vol").
        fallback_policy : RiskPolicy
            Global policy used for sparse regimes and as calibration anchor.
        pit_stat_by_regime : dict, optional
            Per-regime PIT KS statistics. Currently stored in stats
            but not used to modify coverage threshold (reserved for
            future multi-metric calibration).
        pinball_by_regime : dict, optional
            Per-regime mean pinball loss. Stored in stats, not threshold-modifying.

        Returns
        -------
        CalibratedThresholds
        """
        global_target = fallback_policy.coverage_target
        regime_policies: dict[str, RiskPolicy] = {}
        regime_stats:    dict[str, dict]        = {}

        n_sparse    = 0
        n_calibrated = 0

        for regime, coverages in coverage_by_regime.items():
            coverages = np.asarray(coverages, dtype=float)
            n = len(coverages)

            base_stats: dict = {
                "n_samples":       n,
                "mean_coverage":   float(np.mean(coverages)) if n > 0 else np.nan,
                "std_coverage":    float(np.std(coverages))  if n > 0 else np.nan,
                "calibrated":      False,
                "fallback_reason": None,
            }

            # Add optional stats
            if pit_stat_by_regime and regime in pit_stat_by_regime:
                pit = np.asarray(pit_stat_by_regime[regime], dtype=float)
                base_stats["mean_pit_ks"] = float(np.mean(pit)) if len(pit) > 0 else np.nan
            if pinball_by_regime and regime in pinball_by_regime:
                pb = np.asarray(pinball_by_regime[regime], dtype=float)
                base_stats["mean_pinball"] = float(np.mean(pb)) if len(pb) > 0 else np.nan

            if n < self.min_regime_samples:
                warnings.warn(
                    f"Regime '{regime}' has only {n} calibration samples "
                    f"(min={self.min_regime_samples}). Using fallback policy.",
                    UserWarning,
                    stacklevel=2,
                )
                base_stats["fallback_reason"] = f"sparse (n={n} < {self.min_regime_samples})"
                regime_stats[regime] = base_stats
                n_sparse += 1
                continue

            # Calibrate coverage threshold using quantile
            calibrated_target = float(np.quantile(coverages, self.coverage_quantile_green))

            # Clamp to [global_target - relax, global_target + tighten]
            lower_bound = global_target - self.relax_factor
            upper_bound = global_target + self.tighten_factor
            calibrated_target = float(np.clip(calibrated_target, lower_bound, upper_bound))

            # Build calibrated policy — only coverage_target changes
            regime_policy = RiskPolicy(
                coverage_target = calibrated_target,
            )
            regime_policies[regime] = regime_policy

            base_stats["calibrated"]         = True
            base_stats["calibrated_target"]  = round(calibrated_target, 4)
            base_stats["raw_quantile_value"] = round(
                float(np.quantile(coverages, self.coverage_quantile_green)), 4
            )
            base_stats["adjustment_pp"] = round(
                (calibrated_target - global_target) * 100, 2
            )
            regime_stats[regime] = base_stats
            n_calibrated += 1

        calibration_metrics = {
            "n_regimes_total":      len(coverage_by_regime),
            "n_regimes_calibrated": n_calibrated,
            "n_regimes_sparse":     n_sparse,
            "global_target":        global_target,
            "coverage_quantile_green": self.coverage_quantile_green,
            "coverage_quantile_red":   self.coverage_quantile_red,
            "relax_factor":         self.relax_factor,
            "tighten_factor":       self.tighten_factor,
        }

        return CalibratedThresholds(
            regime_policies          = regime_policies,
            fallback_policy          = fallback_policy,
            regime_stats             = regime_stats,
            calibration_metrics      = calibration_metrics,
            coverage_quantile_green  = self.coverage_quantile_green,
            coverage_quantile_red    = self.coverage_quantile_red,
            min_regime_samples       = self.min_regime_samples,
        )

    def calibrate_from_rolling_results(
        self,
        rolling_df:      "pd.DataFrame",
        regime_tags:     list[str],
        fallback_policy: RiskPolicy,
        coverage_col:    str = "empirical_coverage",
    ) -> CalibratedThresholds:
        """
        Calibrate directly from a rolling evaluation CSV DataFrame.

        Parameters
        ----------
        rolling_df : pd.DataFrame
            Rolling evaluation results with at least a coverage column.
        regime_tags : list[str]
            Regime tag per window row (length must match len(rolling_df)).
        fallback_policy : RiskPolicy
        coverage_col : str
            Column name for empirical coverage. Default "empirical_coverage".

        Returns
        -------
        CalibratedThresholds
        """
        import pandas as pd

        if len(regime_tags) != len(rolling_df):
            raise ValueError(
                f"regime_tags length ({len(regime_tags)}) must match "
                f"rolling_df length ({len(rolling_df)})."
            )

        if coverage_col not in rolling_df.columns:
            raise ValueError(
                f"Column '{coverage_col}' not found in rolling_df. "
                f"Available: {list(rolling_df.columns)}"
            )

        tags = np.asarray(regime_tags)
        # Strip "regime_" prefix
        raw_tags = np.array([t.removeprefix("regime_") for t in tags])
        coverage_vals = rolling_df[coverage_col].values.astype(float)

        coverage_by_regime: dict[str, np.ndarray] = {}
        for tag in np.unique(raw_tags):
            mask = raw_tags == tag
            coverage_by_regime[str(tag)] = coverage_vals[mask]

        return self.calibrate(
            coverage_by_regime=coverage_by_regime,
            fallback_policy=fallback_policy,
        )

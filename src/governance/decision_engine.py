"""
src/governance/decision_engine.py
===================================
DecisionEngine: top-level governance orchestrator that wires together
all diagnostic and classification components into a single .decide() call.

Architecture role (Image 6 — top-level orchestrator):
  INPUT  : DiagnosticsReadyObject (from Diagnostics_Input)
           optional: CalibratedThresholds (from ThresholdCalibrator)
           optional: regime_tag (from RegimeTagger)
  OUTPUT : GovernanceDecision
             final_label : "GREEN" | "YELLOW" | "RED"
             reason_codes : list[ReasonCode]
             metric_snapshot : dict of all computed diagnostic values
             policy_used : RiskPolicy
             regime_tag : str
             provenance : dict (full audit trail)

  DESIGN:
    The DecisionEngine does NOT reimplement diagnostics — it delegates to:
      - existing evaluate_distribution() / run_diagnostics_policy()
      - Anfuso interval backtest (from governance.anfuso)
      - Score_Pinball (from scoring.pinball)
      - Interval_Sharpness (from diagnostics.interval_sharpness)
      - classify_risk() + TrafficLight_Labeler (from governance.risk_classification)
      - CalibratedThresholds.get_policy() (from governance.threshold_calibrator)

    The engine computes only what the DiagnosticsReadyObject is capable of,
    using the .can_compute_* interface to gate each diagnostic branch.

  PROVENANCE:
    Every decision includes a full audit trail:
      - which metrics were computed
      - which were skipped (and why)
      - which policy was used
      - which reason codes fired
      - timestamp of decision

  BACKWARDS COMPATIBILITY:
    The existing run_diagnostics_policy() / write_run_artifacts() pipeline
    continues to work unchanged. DecisionEngine is an additive layer on top.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.diagnostics.diagnostics_input import DiagnosticsReadyObject
from src.governance.anfuso import anfuso_interval_backtest
from src.governance.reason_codes import ReasonCode
from src.governance.risk_classification import RiskPolicy, classify_risk


# ---------------------------------------------------------------------------
# GovernanceDecision — output
# ---------------------------------------------------------------------------

@dataclass
class GovernanceDecision:
    """
    Output of DecisionEngine.decide().

    Attributes
    ----------
    final_label : str
        "GREEN" | "YELLOW" | "RED"
    reason_codes : list[ReasonCode]
        Reason codes that contributed to the label.
    metric_snapshot : dict
        All computed diagnostic values (coverage, PIT stats, etc).
    policy_used : RiskPolicy
        The RiskPolicy that was applied (global or regime-calibrated).
    regime_tag : str
        Regime tag used for policy selection.
    model_id : str
    provenance : dict
        Full audit trail of what was computed, skipped, and why.
    decided_at : str
        ISO timestamp of decision.
    """
    final_label:     str
    reason_codes:    list[ReasonCode]
    metric_snapshot: dict
    policy_used:     RiskPolicy
    regime_tag:      str
    model_id:        str
    provenance:      dict
    decided_at:      str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    )

    @property
    def is_green(self) -> bool:
        return self.final_label == "GREEN"

    @property
    def is_yellow(self) -> bool:
        return self.final_label == "YELLOW"

    @property
    def is_red(self) -> bool:
        return self.final_label == "RED"

    def to_dict(self) -> dict:
        """JSON-serialisable summary."""
        return {
            "model_id":      self.model_id,
            "final_label":   self.final_label,
            "reason_codes":  [rc.value if hasattr(rc, 'value') else str(rc) for rc in self.reason_codes],
            "regime_tag":    self.regime_tag,
            "decided_at":    self.decided_at,
            "metric_snapshot": {
                k: (round(float(v), 6) if isinstance(v, (float, np.floating)) else v)
                for k, v in self.metric_snapshot.items()
            },
            "policy": {
                "coverage_target": self.policy_used.coverage_target,
            },
            "provenance": self.provenance,
        }


# ---------------------------------------------------------------------------
# DecisionEngine
# ---------------------------------------------------------------------------

class DecisionEngine:
    """
    Top-level governance orchestrator.

    Parameters
    ----------
    alpha : float
        Miscoverage level. Default 0.1.
    lb_lags : list[int]
        Ljung-Box lags for PIT independence tests. Default [5, 10, 20].
    global_policy : RiskPolicy or None
        Global fallback policy. If None, uses RiskPolicy(coverage_target=0.90).
    calibrated_thresholds : CalibratedThresholds or None
        Regime-conditioned thresholds. If None, global policy is always used.
    run_pinball : bool
        If True and DRO can compute pinball, run Score_Pinball. Default True.
    run_sharpness : bool
        If True and DRO can compute interval, run Interval_Sharpness. Default True.
    run_anfuso : bool
        If True and DRO can compute interval, run Anfuso backtest. Default True.

    Example
    -------
    >>> engine = DecisionEngine()
    >>> decision = engine.decide(dro, regime_tag="regime_winter")
    >>> decision.final_label
    'RED'
    >>> decision.reason_codes
    [ReasonCode.UNDERCOVERAGE, ReasonCode.ACF_DEPENDENCE_FAIL]
    """

    def __init__(
        self,
        alpha:                  float   = 0.1,
        lb_lags:                list[int] = None,
        global_policy:          RiskPolicy | None = None,
        calibrated_thresholds=None,  # CalibratedThresholds | None
        run_pinball:            bool    = True,
        run_sharpness:          bool    = True,
        run_anfuso:             bool    = True,
    ) -> None:
        self.alpha                 = alpha
        self.lb_lags               = lb_lags or [5, 10, 20]
        self.global_policy         = global_policy or RiskPolicy(coverage_target=0.90)
        self.calibrated_thresholds = calibrated_thresholds
        self.run_pinball           = run_pinball
        self.run_sharpness         = run_sharpness
        self.run_anfuso            = run_anfuso

    def decide(
        self,
        dro:        DiagnosticsReadyObject,
        regime_tag: str = "regime_normal",
    ) -> GovernanceDecision:
        """
        Run all available diagnostics and produce a GovernanceDecision.

        Parameters
        ----------
        dro : DiagnosticsReadyObject
            Must have at least one diagnostic capability.
        regime_tag : str
            SplitLabel regime tag for policy selection. Default "regime_normal".

        Returns
        -------
        GovernanceDecision
        """
        metrics:    dict[str, Any] = {}
        provenance: dict[str, Any] = {
            "computed":  [],
            "skipped":   [],
            "model_id":  dro.model_id,
            "regime_tag": regime_tag,
        }

        # ── Select policy ──────────────────────────────────────────────────
        if self.calibrated_thresholds is not None:
            policy = self.calibrated_thresholds.get_policy(regime_tag)
            provenance["policy_source"] = "calibrated"
        else:
            policy = self.global_policy
            provenance["policy_source"] = "global"

        # ── PIT diagnostics (requires samples or CDF) ──────────────────────
        if dro.can_compute_pit and dro.samples is not None:
            from src.calibration.pit import compute_pit
            from src.calibration.diagnostics import (
                pit_uniformity_tests,
                pit_autocorrelation_tests,
            )
            u = compute_pit(dro.y, dro.samples)
            metrics.update(pit_uniformity_tests(u))
            metrics.update(pit_autocorrelation_tests(u, lags=self.lb_lags))
            provenance["computed"].append("pit_uniformity")
            provenance["computed"].append("pit_autocorrelation")
        elif dro.can_compute_pit and dro.cdf_fn is not None:
            # CDF-based PIT
            u = np.clip(dro.cdf_fn(dro.y), 1e-12, 1 - 1e-12)
            from src.calibration.diagnostics import (
                pit_uniformity_tests,
                pit_autocorrelation_tests,
            )
            metrics.update(pit_uniformity_tests(u))
            metrics.update(pit_autocorrelation_tests(u, lags=self.lb_lags))
            provenance["computed"].append("pit_uniformity_cdf")
            provenance["computed"].append("pit_autocorrelation_cdf")
        else:
            provenance["skipped"].append(
                {"diagnostic": "pit", "reason": "no samples or CDF available"}
            )

        # ── Coverage (requires interval) ───────────────────────────────────
        if dro.can_compute_interval:
            from src.calibration.diagnostics import interval_coverage
            emp_cov = interval_coverage(dro.y, dro.lo, dro.hi)
            metrics["empirical_coverage"] = emp_cov
            provenance["computed"].append("empirical_coverage")
        else:
            provenance["skipped"].append(
                {"diagnostic": "coverage", "reason": "no interval available"}
            )

        # ── CRPS (requires samples) ────────────────────────────────────────
        if dro.can_compute_crps:
            from src.scoring.crps import crps_sample
            crps_vals = [
                crps_sample(dro.samples[i], float(dro.y[i]))
                for i in range(dro.n_obs)
            ]
            metrics["crps_mean"] = float(np.mean(crps_vals))
            provenance["computed"].append("crps")
        else:
            provenance["skipped"].append(
                {"diagnostic": "crps", "reason": "no samples available"}
            )

        # ── Pinball loss (requires quantiles) ─────────────────────────────
        if self.run_pinball and dro.can_compute_pinball:
            from src.scoring.pinball import Score_Pinball
            scorer = Score_Pinball()
            pb_result = scorer.compute(quantiles=dro.quantiles, y=dro.y)
            metrics["mean_pinball"] = pb_result.mean_pinball
            provenance["computed"].append("pinball")
        else:
            reason = "disabled" if not self.run_pinball else "no quantiles available"
            provenance["skipped"].append({"diagnostic": "pinball", "reason": reason})

        # ── Anfuso interval backtest (requires interval) ───────────────────
        if self.run_anfuso and dro.can_compute_interval:
            anf = anfuso_interval_backtest(
                dro.y, dro.lo, dro.hi, alpha=self.alpha
            )
            metrics.update({
                "anfuso_traffic_light_total":  anf["traffic_light_total"],
                "anfuso_traffic_light_lower":  anf["traffic_light_lower"],
                "anfuso_traffic_light_upper":  anf["traffic_light_upper"],
                "total_breach_rate":           anf["total_breach_rate"],
                "lower_breach_rate":           anf["lower_breach_rate"],
                "upper_breach_rate":           anf["upper_breach_rate"],
            })
            provenance["computed"].append("anfuso")
        else:
            reason = "disabled" if not self.run_anfuso else "no interval available"
            provenance["skipped"].append({"diagnostic": "anfuso", "reason": reason})

        # ── Sharpness (requires interval) ─────────────────────────────────
        if self.run_sharpness and dro.can_compute_interval:
            from src.diagnostics.interval_sharpness import Interval_Sharpness
            sharp = Interval_Sharpness(alpha=self.alpha)
            sr = sharp.compute(lo=dro.lo, hi=dro.hi, y=dro.y)
            metrics["mean_width"]          = sr.mean_width
            metrics["sharpness_label"]     = sr.sharpness_label
            metrics["coverage_error"]      = sr.coverage_error
            provenance["computed"].append("sharpness")
        else:
            reason = "disabled" if not self.run_sharpness else "no interval available"
            provenance["skipped"].append({"diagnostic": "sharpness", "reason": reason})

        # ── Classify risk ──────────────────────────────────────────────────
        governance = classify_risk(metrics, policy=policy)
        final_label = governance.get("traffic_light", "RED")
        reason_codes = self._extract_reason_codes(governance, metrics, policy)

        provenance["governance_raw"] = {
            k: v for k, v in governance.items()
            if not isinstance(v, (np.ndarray,))
        }

        return GovernanceDecision(
            final_label     = final_label,
            reason_codes    = reason_codes,
            metric_snapshot = metrics,
            policy_used     = policy,
            regime_tag      = regime_tag,
            model_id        = dro.model_id,
            provenance      = provenance,
        )

    # ── Private helpers ────────────────────────────────────────────────────

    def _extract_reason_codes(
        self,
        governance: dict,
        metrics:    dict,
        policy:     RiskPolicy,
    ) -> list[ReasonCode]:
        """Extract ReasonCode list from governance classification output."""
        codes: list[ReasonCode] = []

        label = governance.get("traffic_light", "RED")

        # Coverage
        emp_cov = metrics.get("empirical_coverage")
        if emp_cov is not None:
            target = policy.coverage_target
            if emp_cov < target - 0.02:
                codes.append(ReasonCode.UNDERCOVERAGE)
            elif emp_cov > target + 0.05:
                codes.append(ReasonCode.OVERCOVERAGE)

        # PIT uniformity
        ks_p = metrics.get("pit_ks_pvalue")
        if ks_p is not None and ks_p < 0.05:
            codes.append(ReasonCode.PIT_UNIFORMITY_FAIL)

        # PIT independence
        for lag in self.lb_lags:
            lb_p = metrics.get(f"pit_lb_pvalue_lag{lag}")
            if lb_p is not None and lb_p < 0.05:
                codes.append(ReasonCode.ACF_DEPENDENCE_FAIL)
                break  # one ACF code is enough

        # Anfuso
        anf_total = metrics.get("anfuso_traffic_light_total")
        if anf_total in ("RED", "YELLOW"):
            # Anfuso breach → undercoverage if coverage is below target
            emp_cov = metrics.get("empirical_coverage", 1.0)
            if emp_cov < (policy.coverage_target - 0.02):
                if ReasonCode.UNDERCOVERAGE not in codes:
                    codes.append(ReasonCode.UNDERCOVERAGE)

        # Clean: no issues
        if not codes and label == "GREEN":
            codes.append(ReasonCode.ALL_CLEAR)

        return codes

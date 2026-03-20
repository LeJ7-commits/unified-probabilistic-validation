"""
src/governance/risk_classification.py
=======================================
TrafficLight_Labeler: produces a final per-window governance label
from hierarchical diagnostic outputs.

Architecture
------------
  INPUT  : diagnostic metrics dict (from src.diagnostics.evaluator)
  OUTPUT : TrafficLightLabel dataclass
             .label        str   ∈ {GREEN, YELLOW, RED}
             .reason_codes list[ReasonCode]
             .signals      dict  (raw numeric signals for audit)

Sanity checks (per diagram):
  - always outputs a label
  - reason_codes is non-empty for YELLOW and RED

Policy
------
Classification follows a hierarchical rule:
  1. Any FAIL signal → RED   (most severe wins)
  2. Any WARN signal and no FAIL → YELLOW
  3. No signals triggered → GREEN

Backwards compatibility
-----------------------
The module-level `classify_risk()` function and `RiskPolicy` dataclass
are preserved so existing run scripts (run_001–run_004b) work without
modification. New code should use `TrafficLight_Labeler` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.governance.reason_codes import ReasonCode


# ---------------------------------------------------------------------------
# Policy configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RiskPolicy:
    """
    Governance policy thresholds for the TrafficLight_Labeler.

    All p-value thresholds follow the standard hypothesis-testing
    convention: a small p-value is evidence *against* the null
    (uniformity / independence), triggering YELLOW or RED.
    """
    # Uniformity / independence p-value thresholds
    pvalue_red:    float = 0.01   # strongly reject → RED
    pvalue_yellow: float = 0.05   # weakly reject   → YELLOW

    # Coverage tolerance (in probability units, e.g. 0.02 = 2pp)
    coverage_target:     float | None = None
    coverage_tol_red:    float = 0.05   # |error| > 5pp → RED
    coverage_tol_yellow: float = 0.02   # |error| > 2pp → YELLOW

    # Tail-specific breach rate thresholds (per tail, e.g. alpha/2 = 0.05)
    tail_nominal:     float | None = None   # e.g. 0.05 for α=0.1
    tail_tol_red:     float = 0.02          # breach_rate > nominal + tol → RED
    tail_tol_yellow:  float = 0.01          # breach_rate > nominal + tol → YELLOW

    # Legacy alias kept for backwards compatibility
    @property
    def pvalue_green(self) -> float:
        return self.pvalue_yellow


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrafficLightLabel:
    """
    Structured output of the TrafficLight_Labeler.

    Attributes
    ----------
    label : str
        Final governance classification ∈ {GREEN, YELLOW, RED}.
    reason_codes : list[ReasonCode]
        Structured reason codes explaining the label. Non-empty for
        YELLOW and RED; contains ReasonCode.ALL_CLEAR for GREEN.
    reason_messages : list[str]
        Human-readable messages corresponding to each reason code.
    signals : dict[str, Any]
        Raw numeric signals used in classification (for audit trail).
    """
    label:           str
    reason_codes:    list[ReasonCode] = field(default_factory=list)
    reason_messages: list[str]        = field(default_factory=list)
    signals:         dict[str, Any]   = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-compatible dict (preserves backwards compat)."""
        return {
            "risk_label":    self.label,
            "risk_reasons":  self.reason_messages,
            "reason_codes":  [rc.value for rc in self.reason_codes],
            "risk_signals":  self.signals,
        }


# ---------------------------------------------------------------------------
# TrafficLight_Labeler
# ---------------------------------------------------------------------------

class TrafficLight_Labeler:
    """
    Converts hierarchical diagnostic outputs into a final governance label.

    Parameters
    ----------
    policy : RiskPolicy, optional
        Governance policy thresholds. Defaults to RiskPolicy().

    Example
    -------
    >>> labeler = TrafficLight_Labeler(RiskPolicy(coverage_target=0.90))
    >>> label = labeler.label(metrics_dict)
    >>> print(label.label)          # "RED"
    >>> print(label.reason_codes)   # [ReasonCode.PIT_UNIFORMITY_FAIL, ...]
    """

    def __init__(self, policy: RiskPolicy | None = None) -> None:
        self.policy = policy or RiskPolicy()

    def label(self, metrics: dict[str, Any]) -> TrafficLightLabel:
        """
        Classify a single window or full-sample diagnostic metrics dict.

        Parameters
        ----------
        metrics : dict
            Keys expected (all optional):
              pit_ks_pvalue, pit_cvm_pvalue        — uniformity
              pit_lb_pvalue_lag{5,10,20,...}        — independence
              empirical_coverage                   — coverage
              lower_breach_rate, upper_breach_rate — tail-specific

        Returns
        -------
        TrafficLightLabel
        """
        p   = self.policy
        codes: list[ReasonCode] = []
        msgs:  list[str]        = []
        sigs:  dict[str, Any]   = {}

        # ── 1. Uniformity (PIT) ──────────────────────────────────────────────
        pit_pvals = [
            v for k, v in metrics.items()
            if k in ("pit_ks_pvalue", "pit_cvm_pvalue")
            and v is not None
        ]
        min_p_uniform = min(pit_pvals) if pit_pvals else None
        sigs["min_p_uniformity"] = min_p_uniform

        if min_p_uniform is not None:
            if min_p_uniform < p.pvalue_red:
                codes.append(ReasonCode.PIT_UNIFORMITY_FAIL)
                msgs.append(
                    f"Uniformity strongly rejected (min p={min_p_uniform:.3g})."
                )
            elif min_p_uniform < p.pvalue_yellow:
                codes.append(ReasonCode.PIT_UNIFORMITY_WARN)
                msgs.append(
                    f"Uniformity weakly rejected (min p={min_p_uniform:.3g})."
                )

        # ── 2. Serial independence (Ljung–Box) ───────────────────────────────
        lb_pvals = [
            v for k, v in metrics.items()
            if k.startswith("pit_lb_pvalue_lag") and v is not None
        ]
        min_p_lb = min(lb_pvals) if lb_pvals else None
        sigs["min_p_ljungbox"] = min_p_lb

        if min_p_lb is not None:
            if min_p_lb < p.pvalue_red:
                codes.append(ReasonCode.ACF_DEPENDENCE_FAIL)
                msgs.append(
                    f"Independence strongly rejected (min Ljung-Box p={min_p_lb:.3g})."
                )
            elif min_p_lb < p.pvalue_yellow:
                codes.append(ReasonCode.ACF_DEPENDENCE_WARN)
                msgs.append(
                    f"Independence weakly rejected (min Ljung-Box p={min_p_lb:.3g})."
                )

        # ── 3. Coverage ──────────────────────────────────────────────────────
        cov = metrics.get("empirical_coverage", None)
        sigs["empirical_coverage"] = cov

        if cov is not None and p.coverage_target is not None:
            err        = cov - p.coverage_target
            signed_err = err   # positive = over-covered, negative = under
            sigs["coverage_error"] = signed_err

            if abs(err) > p.coverage_tol_red:
                if err < 0:
                    codes.append(ReasonCode.UNDERCOVERAGE)
                    msgs.append(
                        f"Coverage off-target by {abs(err):.3f} "
                        f"(target={p.coverage_target:.3f})."
                    )
                else:
                    codes.append(ReasonCode.OVERCOVERAGE)
                    msgs.append(
                        f"Coverage off-target by {abs(err):.3f} "
                        f"(target={p.coverage_target:.3f})."
                    )
            elif abs(err) > p.coverage_tol_yellow:
                codes.append(ReasonCode.COVERAGE_WARN)
                msgs.append(
                    f"Coverage mildly off-target by {abs(err):.3f} "
                    f"(target={p.coverage_target:.3f})."
                )

        # ── 4. Tail-specific breach rates ────────────────────────────────────
        lower_br = metrics.get("lower_breach_rate", None)
        upper_br = metrics.get("upper_breach_rate", None)

        if p.tail_nominal is not None:
            for br, fail_code, warn_code, side in [
                (lower_br, ReasonCode.LOWER_TAIL_FAIL,
                 ReasonCode.LOWER_TAIL_WARN, "lower"),
                (upper_br, ReasonCode.UPPER_TAIL_FAIL,
                 ReasonCode.UPPER_TAIL_WARN, "upper"),
            ]:
                if br is None:
                    continue
                excess = br - p.tail_nominal
                if excess > p.tail_tol_red:
                    codes.append(fail_code)
                    msgs.append(
                        f"{side.capitalize()} tail breach rate {br:.4f} "
                        f"exceeds nominal {p.tail_nominal:.4f} by "
                        f"{excess:.4f}."
                    )
                elif excess > p.tail_tol_yellow:
                    codes.append(warn_code)
                    msgs.append(
                        f"{side.capitalize()} tail breach rate {br:.4f} "
                        f"mildly elevated above nominal {p.tail_nominal:.4f}."
                    )

        # ── 5. Aggregate label ───────────────────────────────────────────────
        FAIL_CODES = {
            ReasonCode.PIT_UNIFORMITY_FAIL,
            ReasonCode.ACF_DEPENDENCE_FAIL,
            ReasonCode.UNDERCOVERAGE,
            ReasonCode.OVERCOVERAGE,
            ReasonCode.LOWER_TAIL_FAIL,
            ReasonCode.UPPER_TAIL_FAIL,
        }
        WARN_CODES = {
            ReasonCode.PIT_UNIFORMITY_WARN,
            ReasonCode.ACF_DEPENDENCE_WARN,
            ReasonCode.COVERAGE_WARN,
            ReasonCode.LOWER_TAIL_WARN,
            ReasonCode.UPPER_TAIL_WARN,
        }

        code_set = set(codes)
        if code_set & FAIL_CODES:
            final_label = "RED"
        elif code_set & WARN_CODES:
            final_label = "YELLOW"
        else:
            final_label = "GREEN"
            codes.append(ReasonCode.ALL_CLEAR)
            msgs.append("All monitored diagnostic signals within policy thresholds.")

        return TrafficLightLabel(
            label=final_label,
            reason_codes=codes,
            reason_messages=msgs,
            signals=sigs,
        )


# ---------------------------------------------------------------------------
# Backwards-compatible module-level function
# ---------------------------------------------------------------------------

def classify_risk(
    metrics: dict[str, Any],
    policy: RiskPolicy | None = None,
) -> dict[str, Any]:
    """
    Module-level wrapper for backwards compatibility with run_001–run_004b.
    New code should use TrafficLight_Labeler directly.
    """
    labeler = TrafficLight_Labeler(policy)
    return labeler.label(metrics).to_dict()

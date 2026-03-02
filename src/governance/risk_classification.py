from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RiskPolicy:
    """
    Simple governance policy for traffic-light classification.

    Notes:
    - p-values are interpreted in the standard hypothesis-testing sense:
      small p-value => evidence against desired property (uniformity/independence).
    - Defaults are conservative and meant as starting points.
    """
    pvalue_green: float = 0.05   # pass
    pvalue_yellow: float = 0.01  # strong warning boundary

    # optional coverage tolerance if empirical_coverage is present
    coverage_target: float | None = None  # e.g. 0.90
    coverage_tol_green: float = 0.02      # within +/-2pp
    coverage_tol_yellow: float = 0.05     # within +/-5pp


def classify_risk(metrics: dict[str, Any], policy: RiskPolicy | None = None) -> dict[str, Any]:
    """
    Convert diagnostic metrics into a governance label.

    Returns
    -------
    {
      "risk_label": "GREEN"|"YELLOW"|"RED",
      "risk_reasons": [..],
      "risk_signals": {...}
    }
    """
    if policy is None:
        policy = RiskPolicy()

    reasons: list[str] = []
    signals: dict[str, Any] = {}

    # --- calibration / uniformity ---
    ks_p = metrics.get("pit_ks_pvalue", None)
    cvm_p = metrics.get("pit_cvm_pvalue", None)

    # choose the "most pessimistic" p-value among available ones
    pvals_uniformity = [p for p in [ks_p, cvm_p] if p is not None]
    min_p_uniform = min(pvals_uniformity) if pvals_uniformity else None
    signals["min_p_uniformity"] = min_p_uniform

    # --- independence (Ljung-Box on z) ---
    # if multiple lags exist, take min p-value across provided lags
    lb_pvals = [v for k, v in metrics.items() if k.startswith("pit_lb_pvalue_lag")]
    min_p_lb = min(lb_pvals) if lb_pvals else None
    signals["min_p_ljungbox"] = min_p_lb

    # --- optional coverage ---
    cov = metrics.get("empirical_coverage", None)
    signals["empirical_coverage"] = cov

    # Scoring the label:
    # RED if any strong failure; YELLOW if mild failure; else GREEN.
    label = "GREEN"

    # Uniformity signal
    if min_p_uniform is not None:
        if min_p_uniform < policy.pvalue_yellow:
            label = "RED"
            reasons.append(f"Uniformity strongly rejected (min p={min_p_uniform:.3g}).")
        elif min_p_uniform < policy.pvalue_green and label != "RED":
            label = "YELLOW"
            reasons.append(f"Uniformity weakly rejected (min p={min_p_uniform:.3g}).")

    # Independence signal
    if min_p_lb is not None:
        if min_p_lb < policy.pvalue_yellow:
            label = "RED"
            reasons.append(f"Independence strongly rejected (min Ljung-Box p={min_p_lb:.3g}).")
        elif min_p_lb < policy.pvalue_green and label != "RED":
            label = "YELLOW"
            reasons.append(f"Independence weakly rejected (min Ljung-Box p={min_p_lb:.3g}).")

    # Coverage signal (only if target provided or inferable)
    if cov is not None:
        target = policy.coverage_target
        if target is not None:
            err = abs(cov - target)
            signals["coverage_error"] = err
            if err > policy.coverage_tol_yellow:
                label = "RED"
                reasons.append(f"Coverage off-target by {err:.3f} (target={target:.3f}).")
            elif err > policy.coverage_tol_green and label != "RED":
                label = "YELLOW"
                reasons.append(f"Coverage mildly off-target by {err:.3f} (target={target:.3f}).")

    if not reasons:
        reasons.append("All monitored diagnostic signals within policy thresholds.")

    return {
        "risk_label": label,
        "risk_reasons": reasons,
        "risk_signals": signals,
    }
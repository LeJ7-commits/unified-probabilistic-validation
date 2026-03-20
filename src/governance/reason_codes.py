"""
src/governance/reason_codes.py
================================
Structured reason code enumeration for traffic-light governance.

ReasonCode is a str-Enum so values are directly JSON-serialisable
and can be compared against plain strings where needed.

Usage
-----
from src.governance.reason_codes import ReasonCode

rc = ReasonCode.PIT_UNIFORMITY_FAIL
print(rc)          # "PIT_uniformity_fail"
print(rc.value)    # "PIT_uniformity_fail"
rc == "PIT_uniformity_fail"   # True (str-Enum)
"""

from __future__ import annotations

from enum import Enum


class ReasonCode(str, Enum):
    """
    Standardised reason codes attached to traffic-light labels.

    Naming convention:
      <SIGNAL>_<SEVERITY>  where severity is FAIL (RED), WARN (YELLOW),
      or OK (GREEN-level confirmation).
    """

    # ── Distributional uniformity (PIT) ─────────────────────────────────────
    PIT_UNIFORMITY_FAIL  = "PIT_uniformity_fail"
    """KS / CvM / AD test strongly rejects U(0,1) — RED signal."""

    PIT_UNIFORMITY_WARN  = "PIT_uniformity_warn"
    """KS / CvM / AD test weakly rejects U(0,1) — YELLOW signal."""

    # ── Serial independence (Ljung–Box on z = Φ⁻¹(u)) ──────────────────────
    ACF_DEPENDENCE_FAIL  = "ACF_dependence_fail"
    """Ljung–Box strongly rejects serial independence — RED signal."""

    ACF_DEPENDENCE_WARN  = "ACF_dependence_warn"
    """Ljung–Box weakly rejects serial independence — YELLOW signal."""

    # ── Interval coverage ───────────────────────────────────────────────────
    UNDERCOVERAGE        = "undercoverage"
    """Empirical coverage significantly below nominal — RED signal."""

    OVERCOVERAGE         = "overcoverage"
    """Empirical coverage significantly above nominal — RED signal."""

    COVERAGE_WARN        = "coverage_warn"
    """Empirical coverage mildly off-target — YELLOW signal."""

    # ── Tail-specific signals ────────────────────────────────────────────────
    LOWER_TAIL_FAIL      = "lower_tail_fail"
    """Lower tail breach rate significantly exceeds nominal — RED signal."""

    UPPER_TAIL_FAIL      = "upper_tail_fail"
    """Upper tail breach rate significantly exceeds nominal — RED signal."""

    LOWER_TAIL_WARN      = "lower_tail_warn"
    """Lower tail breach rate mildly elevated — YELLOW signal."""

    UPPER_TAIL_WARN      = "upper_tail_warn"
    """Upper tail breach rate mildly elevated — YELLOW signal."""

    # ── Stability signals (from rolling / transition analysis) ───────────────
    ABSORBING_RED        = "absorbing_red"
    """Transition matrix shows RED as absorbing state (T_RR ≈ 1)."""

    HIGH_ENTROPY         = "high_entropy"
    """Stationary entropy > threshold — unstable governance classification."""

    INSUFFICIENT_WINDOWS = "insufficient_windows"
    """Too few rolling windows to estimate transition matrix reliably."""

    # ── All-clear ────────────────────────────────────────────────────────────
    ALL_CLEAR            = "all_clear"
    """All monitored diagnostic signals within policy thresholds — GREEN."""

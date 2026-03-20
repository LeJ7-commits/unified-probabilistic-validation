"""
src/governance/stability.py
============================
Stability meta-diagnostics for traffic-light governance sequences.

Two classes:

  Stability_TransitionMatrix
    Input : sequence of label strings {label_w} over time
    Output: 3×3 transition probability matrix P(label_t → label_{t+1}),
            per-state stability rates, stationary distribution,
            and absorbing-state detection

  Stability_Entropy
    Input : sequence of label strings {label_w} OR class probability
            matrix (n_windows × 3)
    Output: stationary Shannon entropy (bits), per-state row entropy,
            instability flag, and interpretation

Architecture (per diagram):
  - Stability_TransitionMatrix sanity checks: requires ≥ 2 windows
  - Stability_Entropy interpretation: unstable traffic lights may indicate
    regime shift, insufficient window size, or noisy diagnostic thresholds
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from src.governance.reason_codes import ReasonCode

# Canonical state ordering
STATES      = ("GREEN", "YELLOW", "RED")
STATE_INDEX = {s: i for i, s in enumerate(STATES)}
N_STATES    = len(STATES)

# Entropy thresholds for stability classification
_H_STABLE   = 0.5    # bits — absorbing or near-absorbing
_H_MODERATE = 1.2    # bits — some regime variation
# > 1.2 bits → unstable (max possible = log2(3) ≈ 1.585 bits)


# ---------------------------------------------------------------------------
# TransitionMatrix output dataclass
# ---------------------------------------------------------------------------

@dataclass
class TransitionResult:
    """
    Output of Stability_TransitionMatrix.

    Attributes
    ----------
    n_windows : int
        Number of windows in the input sequence.
    state_sequence : list[str]
        The input label sequence (validated and normalised).
    state_counts : dict[str, int]
        Raw count per state.
    state_freq : dict[str, float]
        Fraction of windows per state.
    transition_matrix : np.ndarray  shape (3, 3)
        Row-normalised transition probability matrix.
        T[i, j] = P(next = j | current = i).
    transition_counts : np.ndarray  shape (3, 3)
        Raw transition counts before normalisation.
    stability_rates : dict[str, float]
        Per-state P(same label next window) = T[i, i].
    stationary_dist : np.ndarray  shape (3,)
        Stationary distribution π such that π @ T ≈ π.
    reason_codes : list[ReasonCode]
        Stability-related reason codes (e.g. ABSORBING_RED,
        INSUFFICIENT_WINDOWS).
    """
    n_windows:         int
    state_sequence:    list[str]
    state_counts:      dict[str, int]
    state_freq:        dict[str, float]
    transition_matrix: np.ndarray
    transition_counts: np.ndarray
    stability_rates:   dict[str, float]
    stationary_dist:   np.ndarray
    reason_codes:      list[ReasonCode] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_windows":      self.n_windows,
            "state_counts":   self.state_counts,
            "state_freq_pct": {k: round(v * 100, 1)
                               for k, v in self.state_freq.items()},
            "transition_matrix": {
                STATES[i]: {STATES[j]: round(float(self.transition_matrix[i, j]), 4)
                            for j in range(N_STATES)}
                for i in range(N_STATES)
            },
            "stability_rates": {k: round(v, 4)
                                for k, v in self.stability_rates.items()},
            "stationary_dist": {STATES[i]: round(float(self.stationary_dist[i]), 4)
                                for i in range(N_STATES)},
            "reason_codes": [rc.value for rc in self.reason_codes],
        }


# ---------------------------------------------------------------------------
# Entropy output dataclass
# ---------------------------------------------------------------------------

@dataclass
class EntropyResult:
    """
    Output of Stability_Entropy.

    Attributes
    ----------
    stationary_entropy : float
        Shannon entropy of the stationary distribution (bits).
        Max possible = log2(3) ≈ 1.585 bits.
    row_entropy : dict[str, float]
        Shannon entropy of each row of the transition matrix.
        H_i = -∑_j T[i,j] * log2(T[i,j]).
    max_possible_entropy : float
        log2(N_STATES) — upper bound for reference.
    is_unstable : bool
        True if stationary_entropy > _H_MODERATE threshold.
    stability_label : str
        "stable" | "moderate" | "unstable"
    interpretation : str
        Human-readable interpretation of the entropy result.
    reason_codes : list[ReasonCode]
        HIGH_ENTROPY if unstable; empty otherwise.
    """
    stationary_entropy:    float
    row_entropy:           dict[str, float]
    max_possible_entropy:  float
    is_unstable:           bool
    stability_label:       str
    interpretation:        str
    reason_codes:          list[ReasonCode] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "stationary_entropy_bits":  round(self.stationary_entropy, 4),
            "row_entropy_bits":         {k: round(v, 4)
                                         for k, v in self.row_entropy.items()},
            "max_possible_entropy_bits": round(self.max_possible_entropy, 4),
            "is_unstable":              self.is_unstable,
            "stability_label":          self.stability_label,
            "interpretation":           self.interpretation,
            "reason_codes":             [rc.value for rc in self.reason_codes],
        }


# ---------------------------------------------------------------------------
# Stability_TransitionMatrix
# ---------------------------------------------------------------------------

class Stability_TransitionMatrix:
    """
    Builds a traffic-light transition probability matrix from a label sequence.

    Parameters
    ----------
    min_windows : int
        Minimum number of windows required to estimate transitions reliably.
        Sequences shorter than this trigger the INSUFFICIENT_WINDOWS reason
        code but still return results.

    Example
    -------
    >>> stm = Stability_TransitionMatrix(min_windows=5)
    >>> result = stm.fit(["RED", "RED", "YELLOW", "RED", "RED", "RED"])
    >>> result.stability_rates["RED"]    # 0.8
    >>> result.reason_codes              # [] (enough windows, no absorbing)
    """

    def __init__(self, min_windows: int = 5) -> None:
        self.min_windows = min_windows

    def fit(self, labels: Sequence[str]) -> TransitionResult:
        """
        Fit transition matrix from a sequence of label strings.

        Parameters
        ----------
        labels : sequence of str
            Each element ∈ {GREEN, YELLOW, RED}.

        Returns
        -------
        TransitionResult
        """
        seq = [str(s).upper().strip() for s in labels]
        n   = len(seq)

        reason_codes: list[ReasonCode] = []

        # Sanity check: enough windows
        if n < self.min_windows:
            reason_codes.append(ReasonCode.INSUFFICIENT_WINDOWS)

        # State counts and frequencies
        counts = {s: seq.count(s) for s in STATES}
        freqs  = {s: counts[s] / n if n > 0 else 0.0 for s in STATES}

        # Transition counts
        T_counts = np.zeros((N_STATES, N_STATES), dtype=float)
        for t in range(n - 1):
            i = STATE_INDEX.get(seq[t],   -1)
            j = STATE_INDEX.get(seq[t+1], -1)
            if i >= 0 and j >= 0:
                T_counts[i, j] += 1

        # Normalise rows
        row_sums = T_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1   # avoid division by zero for unvisited states
        T = T_counts / row_sums

        # Stability rates = diagonal of T
        stability_rates = {STATES[i]: float(T[i, i]) for i in range(N_STATES)}

        # Stationary distribution (dominant left eigenvector of T)
        pi = self._stationary_distribution(T)

        # Check for absorbing RED state
        if T[STATE_INDEX["RED"], STATE_INDEX["RED"]] >= 0.99 and counts["RED"] > 0:
            reason_codes.append(ReasonCode.ABSORBING_RED)

        return TransitionResult(
            n_windows=n,
            state_sequence=seq,
            state_counts=counts,
            state_freq=freqs,
            transition_matrix=T,
            transition_counts=T_counts,
            stability_rates=stability_rates,
            stationary_dist=pi,
            reason_codes=reason_codes,
        )

    @staticmethod
    def _stationary_distribution(T: np.ndarray) -> np.ndarray:
        """
        Compute stationary distribution π via dominant left eigenvector.
        Falls back to uniform if T is degenerate.
        """
        try:
            eigenvalues, eigenvectors = np.linalg.eig(T.T)
            idx = np.argmin(np.abs(eigenvalues - 1.0))
            pi  = np.real(eigenvectors[:, idx])
            pi  = np.abs(pi)
            if pi.sum() > 0:
                return pi / pi.sum()
        except Exception:
            pass
        return np.ones(N_STATES) / N_STATES


# ---------------------------------------------------------------------------
# Stability_Entropy
# ---------------------------------------------------------------------------

class Stability_Entropy:
    """
    Computes Shannon entropy of a traffic-light classification sequence.

    Entropy measures governance instability: high entropy means the
    classification fluctuates across states; low entropy means the
    classification is stable (absorbed into one state).

    Possible interpretations of high entropy (per diagram):
      - Regime shift in the underlying process
      - Insufficient rolling window size
      - Noisy diagnostic thresholds near a boundary

    Parameters
    ----------
    entropy_threshold_moderate : float
        Entropy above which governance is labelled "moderate" (default 0.5).
    entropy_threshold_unstable : float
        Entropy above which governance is labelled "unstable" (default 1.2).

    Example
    -------
    >>> se = Stability_Entropy()
    >>> result = se.compute_from_labels(["RED"] * 10)
    >>> result.stationary_entropy     # ≈ 0.0 (absorbing)
    >>> result.stability_label        # "stable"
    """

    def __init__(
        self,
        entropy_threshold_moderate: float = _H_STABLE,
        entropy_threshold_unstable: float = _H_MODERATE,
    ) -> None:
        self.h_moderate = entropy_threshold_moderate
        self.h_unstable = entropy_threshold_unstable
        self._max_h     = float(np.log2(N_STATES))

    def compute_from_labels(self, labels: Sequence[str]) -> EntropyResult:
        """
        Compute entropy from a raw label sequence.
        Derives empirical state probabilities then delegates to
        compute_from_probs.
        """
        seq   = [str(s).upper().strip() for s in labels]
        n     = len(seq)
        probs = np.array([seq.count(s) / n if n > 0 else 1/N_STATES
                          for s in STATES])
        return self._compute(probs, row_entropy_source=None)

    def compute_from_transition(
        self,
        transition_result: TransitionResult,
    ) -> EntropyResult:
        """
        Compute entropy using the stationary distribution from a
        TransitionResult, and per-state row entropies from the
        transition matrix.
        """
        return self._compute(
            stationary_probs=transition_result.stationary_dist,
            row_entropy_source=transition_result.transition_matrix,
        )

    def _compute(
        self,
        stationary_probs: np.ndarray,
        row_entropy_source: np.ndarray | None,
    ) -> EntropyResult:
        # Stationary entropy
        pi      = np.clip(stationary_probs, 0, None)
        nonzero = pi[pi > 0]
        h_stat  = float(-np.sum(nonzero * np.log2(nonzero))) if len(nonzero) > 0 else 0.0

        # Row entropy from transition matrix (if available)
        if row_entropy_source is not None:
            T = row_entropy_source
            row_h = {}
            for i, s in enumerate(STATES):
                row  = T[i]
                nz   = row[row > 0]
                row_h[s] = float(-np.sum(nz * np.log2(nz))) if len(nz) > 0 else 0.0
        else:
            row_h = {s: 0.0 for s in STATES}

        # Stability classification
        if h_stat <= self.h_moderate:
            stab_label = "stable"
            is_unstable = False
            interpretation = (
                "Classification is stable — the governance state is "
                "absorbed into one or two dominant zones. This indicates "
                "persistent structural miscalibration (if RED) or consistent "
                "model adequacy (if GREEN)."
            )
        elif h_stat <= self.h_unstable:
            stab_label = "moderate"
            is_unstable = False
            interpretation = (
                "Classification shows moderate variation across states. "
                "Possible causes: mild regime shifts, seasonal effects on "
                "calibration quality, or borderline diagnostic thresholds."
            )
        else:
            stab_label = "unstable"
            is_unstable = True
            interpretation = (
                "Classification is highly unstable. Possible causes: "
                "(1) genuine regime shifts in the underlying process, "
                "(2) rolling window too small relative to process memory, "
                "(3) diagnostic thresholds near a decision boundary."
            )

        reason_codes = [ReasonCode.HIGH_ENTROPY] if is_unstable else []

        return EntropyResult(
            stationary_entropy=h_stat,
            row_entropy=row_h,
            max_possible_entropy=self._max_h,
            is_unstable=is_unstable,
            stability_label=stab_label,
            interpretation=interpretation,
            reason_codes=reason_codes,
        )

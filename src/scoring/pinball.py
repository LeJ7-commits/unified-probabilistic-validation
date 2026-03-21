"""
src/scoring/pinball.py
=======================
Score_Pinball: computes pinball (quantile) loss for probabilistic forecasts.

Architecture role (Image 4 — Proper Scoring Rules branch):
  INPUT  : quantile forecasts q_t(p) for p in p-grid
           realizations y_t
           optional: regime tags per observation
  OUTPUT : vector loss per p (mean pinball loss at each quantile level)
           averaged pinball loss overall
           averaged pinball loss per regime (if regime tags provided)

Pinball loss definition:
  L_p(q, y) = (y - q) * p        if y >= q  (under-prediction)
            = (q - y) * (1 - p)  if y < q   (over-prediction)

  The mean pinball loss averaged over all p in the grid equals the CRPS
  when the grid is dense enough (connection to proper scoring rules).

Sanity check (per diagram):
  Quantile crossing fixed upstream (reference adapter check).
  Score_Pinball does NOT fix crossings — it reports them as a warning
  and computes the score regardless (the score is still valid even with
  mild crossings in practice).

Regime-stratified scoring:
  If regime_tags is provided (array of string labels, one per observation),
  the mean pinball loss is computed separately per regime, enabling
  diagnostic decomposition by market condition.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# PinballResult — output dataclass
# ---------------------------------------------------------------------------

@dataclass
class PinballResult:
    """
    Output of Score_Pinball.compute().

    Attributes
    ----------
    levels : np.ndarray, shape (K,)
        Quantile levels at which loss was computed.
    loss_per_level : np.ndarray, shape (K,)
        Mean pinball loss per quantile level.
    mean_pinball : float
        Mean pinball loss averaged over all levels and observations.
    loss_matrix : np.ndarray, shape (n_obs, K)
        Per-observation, per-level pinball loss.
    regime_losses : dict {str -> float}
        Mean pinball loss per regime tag (empty if no tags provided).
    regime_loss_per_level : dict {str -> np.ndarray(K,)}
        Per-regime, per-level mean loss (empty if no tags provided).
    n_obs : int
    n_crossing_pairs : int
        Number of (t, consecutive level pair) crossings detected.
    """
    levels:               np.ndarray
    loss_per_level:       np.ndarray
    mean_pinball:         float
    loss_matrix:          np.ndarray
    regime_losses:        dict[str, float]              = field(default_factory=dict)
    regime_loss_per_level: dict[str, np.ndarray]        = field(default_factory=dict)
    n_obs:                int                            = 0
    n_crossing_pairs:     int                            = 0

    def to_dict(self) -> dict:
        """JSON-serialisable summary."""
        return {
            "mean_pinball":       round(float(self.mean_pinball), 6),
            "loss_per_level":     {
                round(float(p), 4): round(float(v), 6)
                for p, v in zip(self.levels, self.loss_per_level)
            },
            "n_obs":              self.n_obs,
            "n_crossing_pairs":   self.n_crossing_pairs,
            "regime_losses":      {
                k: round(float(v), 6)
                for k, v in self.regime_losses.items()
            },
        }


# ---------------------------------------------------------------------------
# Score_Pinball
# ---------------------------------------------------------------------------

class Score_Pinball:
    """
    Computes pinball (quantile) loss for a set of quantile forecasts.

    Parameters
    ----------
    warn_on_crossings : bool
        If True, warn when quantile crossings are detected in the input.
        Crossings should be fixed upstream by Adapter_Quantiles — this
        is a defensive check only. Default True.

    Example
    -------
    >>> scorer = Score_Pinball()
    >>> result = scorer.compute(
    ...     quantiles={0.1: lo, 0.5: med, 0.9: hi},
    ...     y=actuals,
    ...     regime_tags=["winter"] * 50 + ["summer"] * 50,
    ... )
    >>> result.mean_pinball
    1.23
    >>> result.regime_losses
    {"winter": 1.10, "summer": 1.36}
    """

    def __init__(self, warn_on_crossings: bool = True) -> None:
        self.warn_on_crossings = warn_on_crossings

    def compute(
        self,
        quantiles:    dict[float, np.ndarray],
        y:            np.ndarray,
        regime_tags:  list[str] | np.ndarray | None = None,
    ) -> PinballResult:
        """
        Compute pinball loss.

        Parameters
        ----------
        quantiles : dict {float -> np.ndarray(n_obs,)}
            Quantile arrays keyed by probability level in (0, 1).
        y : np.ndarray, shape (n_obs,)
            Realizations.
        regime_tags : array-like of str, optional, length n_obs
            Regime label per observation. If provided, per-regime losses
            are computed in addition to the overall loss.

        Returns
        -------
        PinballResult
        """
        y = np.asarray(y, dtype=float)
        n = len(y)

        if len(quantiles) == 0:
            raise ValueError("quantiles dict must not be empty.")

        levels = np.array(sorted(float(p) for p in quantiles))
        K      = len(levels)

        # Build (n_obs, K) quantile matrix
        Q = np.stack(
            [np.asarray(quantiles[p], dtype=float) for p in levels],
            axis=1,
        )   # (n_obs, K)

        if Q.shape[0] != n:
            raise ValueError(
                f"All quantile arrays must have length {n} (same as y), "
                f"got length {Q.shape[0]}."
            )

        # Crossing check
        n_crossing_pairs = self._check_crossings(Q, levels)

        # Pinball loss: (n_obs, K)
        L = self._pinball_matrix(y, Q, levels)

        loss_per_level = L.mean(axis=0)          # (K,)
        mean_pinball   = float(L.mean())

        # Regime-stratified losses
        regime_losses:         dict[str, float]        = {}
        regime_loss_per_level: dict[str, np.ndarray]   = {}

        if regime_tags is not None:
            tags = np.asarray(regime_tags)
            if len(tags) != n:
                raise ValueError(
                    f"regime_tags must have length {n}, got {len(tags)}."
                )
            for tag in np.unique(tags):
                mask = tags == tag
                regime_L = L[mask]
                regime_losses[str(tag)]         = float(regime_L.mean())
                regime_loss_per_level[str(tag)] = regime_L.mean(axis=0)

        return PinballResult(
            levels               = levels,
            loss_per_level       = loss_per_level,
            mean_pinball         = mean_pinball,
            loss_matrix          = L,
            regime_losses        = regime_losses,
            regime_loss_per_level = regime_loss_per_level,
            n_obs                = n,
            n_crossing_pairs     = n_crossing_pairs,
        )

    def compute_from_dro(
        self,
        dro,
        regime_tags: list[str] | np.ndarray | None = None,
    ) -> PinballResult:
        """
        Convenience method: compute pinball loss from a DiagnosticsReadyObject.

        Parameters
        ----------
        dro : DiagnosticsReadyObject
            Must have can_compute_pinball == True.
        regime_tags : optional

        Returns
        -------
        PinballResult
        """
        dro.require("pinball")
        return self.compute(
            quantiles=dro.quantiles,
            y=dro.y,
            regime_tags=regime_tags,
        )

    # ── Static helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _pinball_matrix(
        y: np.ndarray,
        Q: np.ndarray,
        levels: np.ndarray,
    ) -> np.ndarray:
        """
        Compute (n_obs, K) pinball loss matrix.

        L[t, k] = (y[t] - Q[t,k]) * p_k   if y[t] >= Q[t,k]
                = (Q[t,k] - y[t]) * (1-p_k) if y[t] < Q[t,k]
        """
        y_col  = y[:, np.newaxis]          # (n_obs, 1) broadcast
        p_row  = levels[np.newaxis, :]     # (1, K) broadcast

        over   = (y_col - Q) * p_row       # when y >= q: positive contribution
        under  = (Q - y_col) * (1 - p_row) # when y < q: positive contribution

        L = np.where(y_col >= Q, over, under)
        return L

    def _check_crossings(
        self,
        Q:      np.ndarray,
        levels: np.ndarray,
    ) -> int:
        """
        Count crossing pairs (consecutive levels where Q[:,k] > Q[:,k+1]).
        Issues a warning if crossings are found.
        """
        K = Q.shape[1]
        n_crossings = 0
        for k in range(K - 1):
            crosses = np.sum(Q[:, k] > Q[:, k + 1] + 1e-8)
            n_crossings += int(crosses)

        if n_crossings > 0 and self.warn_on_crossings:
            warnings.warn(
                f"Score_Pinball detected {n_crossings} quantile crossing(s) "
                f"in the input. Crossings should be fixed upstream by "
                "Adapter_Quantiles (PAVA isotonic regression). "
                "The pinball score is computed regardless.",
                UserWarning,
                stacklevel=3,
            )
        return n_crossings


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def pinball_loss(
    y:         np.ndarray,
    q:         np.ndarray,
    level:     float,
) -> np.ndarray:
    """
    Compute element-wise pinball loss for a single quantile level.

    Parameters
    ----------
    y : np.ndarray, shape (n,)
    q : np.ndarray, shape (n,)
    level : float in (0, 1)

    Returns
    -------
    np.ndarray, shape (n,), non-negative
    """
    y = np.asarray(y, dtype=float)
    q = np.asarray(q, dtype=float)
    over  = (y - q) * level
    under = (q - y) * (1 - level)
    return np.where(y >= q, over, under)

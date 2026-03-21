"""
src/core/data_contract.py
==========================
DataContract (Canonical Schema) and StandardizedModelObject.

Architecture role (Image 1):
  All three input types — Monte Carlo Simulation, ENTSO-E Load Forecasts,
  and Renewable Generation Data — pass through the DataContract before
  anything downstream can consume them. The contract validates, normalises,
  and converts raw inputs into a StandardizedModelObject (t, y, predictive
  object) that all Adapters, BuildDist builders, and diagnostic layers
  share as their common input type.

Required fields (per diagram):
  t          : timestamp array (monotone, no gaps)
  y          : realization array (no NaN)
  model_id   : string identifier
  split      : SplitLabel — "train" | "test" | "window_{int}" | "regime_{tag}"

Optional fields (per diagram, depend on model type):
  h          : forecast horizon (int or array)
  y_hat      : point forecast array
  q_{p}      : quantile arrays for p in (0.0, 0.1, ..., 0.9, 1.0)
  S          : predictive samples (M × d) array
  x          : covariate array (optional)

Sanity checks (per diagram):
  ✓ No missing t or y
  ✓ Monotone timestamps per model_id
  ✓ Horizon consistency within rolling window
  ✓ Sample size M ≥ threshold (if S exists)
  ✓ Quantiles non-crossing (if q provided)

SplitLabel vocabulary (enforced by regex):
  "train"           — conformal calibration set
  "test"            — conformal evaluation set
  "window_{int}"    — rolling window identification (e.g. "window_3")
  "regime_{tag}"    — regime-tagged window (e.g. "regime_winter")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# SplitLabel
# ---------------------------------------------------------------------------

_SPLIT_PATTERN = re.compile(
    r"^(train|test|window_\d+|regime_[a-z0-9_]+)$"
)


def validate_split_label(label: str) -> str:
    """
    Validate and normalise a SplitLabel string.

    Valid forms:
      "train", "test",
      "window_{non-negative int}" e.g. "window_0", "window_42"
      "regime_{lowercase alphanum+underscore}" e.g. "regime_winter", "regime_high_vol"

    Returns the normalised (lowercased, stripped) label.
    Raises ValueError if the label does not match the vocabulary.
    """
    label = str(label).strip().lower()
    if not _SPLIT_PATTERN.match(label):
        raise ValueError(
            f"Invalid SplitLabel: '{label}'. "
            "Must be one of: 'train', 'test', 'window_{{int}}', 'regime_{{tag}}'. "
            "Example valid values: 'train', 'test', 'window_3', 'regime_winter', "
            "'regime_high_vol'."
        )
    return label


# ---------------------------------------------------------------------------
# DataContractError
# ---------------------------------------------------------------------------

class DataContractError(ValueError):
    """Raised when a DataContract sanity check fails."""


# ---------------------------------------------------------------------------
# DataContract
# ---------------------------------------------------------------------------

@dataclass
class DataContract:
    """
    Canonical schema validator for all model inputs.

    Validates required fields, enforces SplitLabel vocabulary, and runs
    sanity checks before constructing a StandardizedModelObject.

    Parameters
    ----------
    min_samples : int
        Minimum number of Monte Carlo paths required if S is provided.
        Per diagram: M ≥ threshold. Default 100.
    min_obs : int
        Minimum number of observations required. Default 2.
    allow_nan_x : bool
        If True, NaN values in covariate array x are tolerated.
        Default True (covariates are optional and may be sparse).

    Example
    -------
    >>> contract = DataContract()
    >>> obj = contract.validate(
    ...     t=timestamps, y=actuals, model_id="entsoe_load",
    ...     split="window_0", y_hat=forecasts
    ... )
    >>> obj.model_id
    'entsoe_load'
    """

    min_samples: int = 100
    min_obs:     int = 2
    allow_nan_x: bool = True

    def validate(
        self,
        *,
        t,
        y,
        model_id: str,
        split: str,
        h: int | np.ndarray | None = None,
        y_hat: np.ndarray | None = None,
        quantiles: dict[float, np.ndarray] | None = None,
        S: np.ndarray | None = None,
        x: np.ndarray | None = None,
    ) -> "StandardizedModelObject":
        """
        Validate inputs and return a StandardizedModelObject.

        Parameters
        ----------
        t : array-like of timestamps (datetime-like or numeric)
        y : array-like of float realizations, shape (n,)
        model_id : str
            Unique string identifier for the model/dataset.
        split : str
            SplitLabel — "train" | "test" | "window_{int}" | "regime_{tag}"
        h : int or array-like, optional
            Forecast horizon(s). If array, must have shape (n,).
        y_hat : array-like, optional
            Point forecast, shape (n,).
        quantiles : dict {float -> array(n,)}, optional
            Predictive quantiles. Keys must be in (0, 1).
        S : array-like, optional
            Predictive samples, shape (M, d) or (n, M).
            Convention: (n, M) — one row per observation, M paths.
        x : array-like, optional
            Covariates, shape (n, p) or (n,).

        Returns
        -------
        StandardizedModelObject
        """
        # ── 1. Required: t ────────────────────────────────────────────────
        t_arr = self._parse_timestamps(t)

        # ── 2. Required: y ────────────────────────────────────────────────
        y_arr = np.asarray(y, dtype=float)
        if y_arr.ndim != 1:
            raise DataContractError(
                f"y must be 1-dimensional, got shape {y_arr.shape}."
            )
        n = len(y_arr)
        if np.any(np.isnan(y_arr)):
            n_nan = int(np.isnan(y_arr).sum())
            raise DataContractError(
                f"y contains {n_nan} NaN value(s). All realizations must be finite."
            )

        # ── 3. Required: model_id ─────────────────────────────────────────
        model_id = str(model_id).strip()
        if not model_id:
            raise DataContractError("model_id must be a non-empty string.")

        # ── 4. Required: split ────────────────────────────────────────────
        split_validated = validate_split_label(split)

        # ── 5. Length consistency ─────────────────────────────────────────
        if len(t_arr) != n:
            raise DataContractError(
                f"t and y must have the same length. "
                f"Got len(t)={len(t_arr)}, len(y)={n}."
            )
        if n < self.min_obs:
            raise DataContractError(
                f"Too few observations: got {n}, minimum is {self.min_obs}."
            )

        # ── 6. Sanity: monotone timestamps ────────────────────────────────
        if not np.all(np.diff(t_arr.astype(np.int64)) > 0):
            raise DataContractError(
                "Timestamps must be strictly monotonically increasing "
                "(no duplicates or backwards steps)."
            )

        # ── 7. Optional: h ────────────────────────────────────────────────
        h_arr: np.ndarray | None = None
        if h is not None:
            if isinstance(h, (int, np.integer)):
                h_arr = np.full(n, int(h), dtype=int)
            else:
                h_arr = np.asarray(h, dtype=int)
                if h_arr.shape != (n,):
                    raise DataContractError(
                        f"h array must have shape ({n},), got {h_arr.shape}."
                    )
                if not np.all(h_arr >= 0):
                    raise DataContractError("All horizon values h must be ≥ 0.")

        # ── 8. Optional: y_hat ────────────────────────────────────────────
        y_hat_arr: np.ndarray | None = None
        if y_hat is not None:
            y_hat_arr = np.asarray(y_hat, dtype=float)
            if y_hat_arr.shape != (n,):
                raise DataContractError(
                    f"y_hat must have shape ({n},), got {y_hat_arr.shape}."
                )

        # ── 9. Optional: quantiles ────────────────────────────────────────
        q_validated: dict[float, np.ndarray] | None = None
        if quantiles is not None:
            q_validated = self._validate_quantiles(quantiles, n)

        # ── 10. Optional: S (samples) ─────────────────────────────────────
        S_arr: np.ndarray | None = None
        if S is not None:
            S_arr = np.asarray(S, dtype=float)
            # Accept (n, M) — standard convention in this codebase
            if S_arr.ndim == 2 and S_arr.shape[0] == n:
                M = S_arr.shape[1]
            elif S_arr.ndim == 2 and S_arr.shape[1] == n:
                # (M, n) — transpose to (n, M)
                S_arr = S_arr.T
                M = S_arr.shape[1]
            else:
                raise DataContractError(
                    f"S (samples) must have shape (n, M) or (M, n) where n={n}. "
                    f"Got shape {S_arr.shape}."
                )
            if M < self.min_samples:
                raise DataContractError(
                    f"Sample size M={M} is below minimum threshold {self.min_samples}. "
                    "Increase the number of Monte Carlo paths."
                )
            if not np.all(np.isfinite(S_arr)):
                raise DataContractError(
                    "S (samples) contains NaN or Inf values. "
                    "All sample paths must be finite."
                )

        # ── 11. Optional: x (covariates) ──────────────────────────────────
        x_arr: np.ndarray | None = None
        if x is not None:
            x_arr = np.asarray(x, dtype=float)
            if x_arr.ndim == 1:
                x_arr = x_arr.reshape(-1, 1)
            if x_arr.shape[0] != n:
                raise DataContractError(
                    f"x must have {n} rows (one per observation), "
                    f"got {x_arr.shape[0]}."
                )
            if not self.allow_nan_x and np.any(np.isnan(x_arr)):
                raise DataContractError(
                    "x (covariates) contains NaN values and allow_nan_x=False."
                )

        return StandardizedModelObject(
            t=t_arr,
            y=y_arr,
            model_id=model_id,
            split=split_validated,
            h=h_arr,
            y_hat=y_hat_arr,
            quantiles=q_validated,
            S=S_arr,
            x=x_arr,
            n_obs=n,
        )

    # ── Private helpers ────────────────────────────────────────────────────

    def _parse_timestamps(self, t) -> np.ndarray:
        """Parse timestamps to numpy datetime64[ns] array."""
        if isinstance(t, pd.DatetimeIndex):
            return t.values.astype("datetime64[ns]")
        arr = np.asarray(t)
        if arr.dtype.kind == "M":    # already datetime
            return arr.astype("datetime64[ns]")
        if arr.dtype.kind in ("i", "u", "f"):  # numeric — treat as ordinal
            return arr.astype(np.int64)
        try:
            return pd.to_datetime(arr).values.astype("datetime64[ns]")
        except Exception as exc:
            raise DataContractError(
                f"Cannot parse timestamps. Got dtype={arr.dtype}. "
                f"Original error: {exc}"
            )

    def _validate_quantiles(
        self,
        quantiles: dict[float, np.ndarray],
        n: int,
    ) -> dict[float, np.ndarray]:
        """Validate quantile dict: keys in (0,1), shapes (n,), non-crossing."""
        out: dict[float, np.ndarray] = {}
        for p, arr in quantiles.items():
            p = float(p)
            if not (0.0 < p < 1.0):
                raise DataContractError(
                    f"Quantile level {p} is out of (0, 1). "
                    "All quantile keys must be strictly between 0 and 1."
                )
            arr = np.asarray(arr, dtype=float)
            if arr.shape != (n,):
                raise DataContractError(
                    f"Quantile array for p={p} must have shape ({n},), "
                    f"got {arr.shape}."
                )
            out[p] = arr

        # Non-crossing check
        sorted_levels = sorted(out.keys())
        for i in range(len(sorted_levels) - 1):
            p_lo = sorted_levels[i]
            p_hi = sorted_levels[i + 1]
            if np.any(out[p_lo] > out[p_hi] + 1e-8):
                n_cross = int(np.sum(out[p_lo] > out[p_hi] + 1e-8))
                raise DataContractError(
                    f"Quantile crossing detected between p={p_lo} and p={p_hi} "
                    f"at {n_cross} observation(s). Quantiles must be non-decreasing."
                )
        return out


# ---------------------------------------------------------------------------
# StandardizedModelObject
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StandardizedModelObject:
    """
    Standardised (t, y, predictive object) produced by DataContract.validate().

    All fields are validated and normalised. Downstream components
    (Adapters, BuildDist builders, Diagnostics_Input) consume this
    object — never raw inputs.

    Attributes
    ----------
    t : np.ndarray
        Timestamp array, dtype datetime64[ns] or int64.
    y : np.ndarray, shape (n_obs,)
        Realisation array.
    model_id : str
        Unique model/dataset identifier.
    split : str
        Validated SplitLabel.
    n_obs : int
        Number of observations.
    h : np.ndarray or None, shape (n_obs,)
        Forecast horizons.
    y_hat : np.ndarray or None, shape (n_obs,)
        Point forecasts.
    quantiles : dict {float -> ndarray(n_obs,)} or None
        Non-crossing quantile arrays.
    S : np.ndarray or None, shape (n_obs, M)
        Predictive sample paths.
    x : np.ndarray or None, shape (n_obs, p)
        Covariates.
    """
    t:          np.ndarray
    y:          np.ndarray
    model_id:   str
    split:      str
    n_obs:      int
    h:          np.ndarray | None = None
    y_hat:      np.ndarray | None = None
    quantiles:  dict[float, np.ndarray] | None = None
    S:          np.ndarray | None = None
    x:          np.ndarray | None = None

    # ── Convenience properties ─────────────────────────────────────────────

    @property
    def has_samples(self) -> bool:
        return self.S is not None

    @property
    def has_quantiles(self) -> bool:
        return self.quantiles is not None and len(self.quantiles) > 0

    @property
    def has_point_forecast(self) -> bool:
        return self.y_hat is not None

    @property
    def n_samples(self) -> int | None:
        return self.S.shape[1] if self.S is not None else None

    @property
    def quantile_levels(self) -> list[float]:
        if self.quantiles is None:
            return []
        return sorted(self.quantiles.keys())

    @property
    def split_type(self) -> str:
        """Returns the prefix of the split label: 'train', 'test', 'window', 'regime'."""
        return self.split.split("_")[0]

    @property
    def split_index(self) -> int | None:
        """For 'window_{int}' splits, returns the integer index. Else None."""
        if self.split.startswith("window_"):
            return int(self.split.split("_")[1])
        return None

    @property
    def split_regime(self) -> str | None:
        """For 'regime_{tag}' splits, returns the tag string. Else None."""
        if self.split.startswith("regime_"):
            return "_".join(self.split.split("_")[1:])
        return None

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary dict for logging."""
        return {
            "model_id":        self.model_id,
            "split":           self.split,
            "n_obs":           self.n_obs,
            "has_samples":     self.has_samples,
            "n_samples":       self.n_samples,
            "has_quantiles":   self.has_quantiles,
            "quantile_levels": self.quantile_levels,
            "has_point_forecast": self.has_point_forecast,
            "has_covariates":  self.x is not None,
        }

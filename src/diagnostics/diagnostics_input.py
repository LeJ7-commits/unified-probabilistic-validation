"""
src/diagnostics/diagnostics_input.py
======================================
Diagnostics_Input: the gateway between adapter outputs and the four
diagnostic branches (Calibration via PIT, Proper Scoring Rules,
Interval/Coverage diagnostics, Multivariate diagnostics).

Architecture role (Image 3):
  INPUT  : Any one or more of:
             - CDF F_t(·)             from QuantileFunctionObject
             - samples X_t^{(m)}      from MarginalSamples / ResidualPool
             - quantile function Q_t  from QuantileFunctionObject
             - interval [l_t, U_t]    from conformal / any adapter
           PLUS realization y_t

  OUTPUT : DiagnosticsReadyObject — a standardised object that can supply
           whichever representations are available to downstream diagnostics.

  SANITY CHECKS (per diagram):
    - Object can provide at least ONE of: CDF / samples / quantiles / interval
    - Time alignment with y_t valid (no horizon mismatch)
    - Arrow from this gateway to the 4 diagnostic branches below

Intake formats (auto-detected):
  Format A — adapter output objects:
    ResidualPool          → extracts samples + interval + y
    MarginalSamples       → extracts samples + y
    QuantileFunctionObject → extracts quantiles + CDF callable + y
    JointSimulationObject → extracts per-variable marginals

  Format B — raw arrays:
    Pass y, samples, quantiles, lo/hi directly as numpy arrays.
    These are normalised and wrapped into a DiagnosticsReadyObject.

The DiagnosticsReadyObject exposes a capability interface:
    .can_compute_pit        → True if samples or CDF available
    .can_compute_crps       → True if samples available
    .can_compute_pinball    → True if quantiles available
    .can_compute_interval   → True if interval (lo, hi) available
    .can_compute_energy_score → True if joint samples (d > 1) available

This allows diagnostic branches to query capabilities before computing,
producing informative errors rather than silent missing-data failures.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

# Adapter output types — imported for isinstance checks
from src.adapters.point_forecast import ResidualPool
from src.adapters.simulation_joint import JointSimulationObject, MarginalSamples
from src.adapters.quantile_adapter import QuantileFunctionObject


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class DiagnosticsInputError(ValueError):
    """Raised when Diagnostics_Input cannot construct a valid object."""


# ---------------------------------------------------------------------------
# DiagnosticsReadyObject — output
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticsReadyObject:
    """
    Standardised diagnostics-ready object.

    All fields are optional except y and model_id. Downstream diagnostic
    components query .can_compute_* before attempting to use a representation.

    Attributes
    ----------
    model_id : str
    n_obs : int
    t : np.ndarray
        Timestamps.
    y : np.ndarray, shape (n_obs,)
        Realizations.
    samples : np.ndarray or None, shape (n_obs, M)
        Monte Carlo or bootstrap samples.
    quantiles : dict {float -> np.ndarray(n_obs,)} or None
        Quantile arrays keyed by probability level.
    lo : np.ndarray or None, shape (n_obs,)
        Lower interval bound.
    hi : np.ndarray or None, shape (n_obs,)
        Upper interval bound.
    alpha : float
        Miscoverage level for interval extraction.
    cdf_fn : callable or None
        CDF evaluator: cdf_fn(y_query: np.ndarray) -> np.ndarray in [0,1]
        Signature matches QuantileFunctionObject.interpolate_cdf().
    joint_samples : np.ndarray or None, shape (n_obs, M, d)
        Joint simulation paths for multivariate diagnostics (d > 1).
    variable_names : list[str] or None
        Variable names for joint samples.
    source_dist_type : str
        dist_type of the originating adapter output.
    meta : dict
        Provenance metadata.
    """
    model_id:         str
    n_obs:            int
    t:                np.ndarray
    y:                np.ndarray
    alpha:            float                          = 0.1
    samples:          np.ndarray | None             = None
    quantiles:        dict[float, np.ndarray] | None = None
    lo:               np.ndarray | None             = None
    hi:               np.ndarray | None             = None
    cdf_fn:           Callable | None               = None
    joint_samples:    np.ndarray | None             = None
    variable_names:   list[str] | None              = None
    source_dist_type: str                           = "unknown"
    meta:             dict                          = field(default_factory=dict)

    # ── Capability interface ───────────────────────────────────────────────

    @property
    def can_compute_pit(self) -> bool:
        """PIT requires samples (empirical CDF) or a CDF callable."""
        return self.samples is not None or self.cdf_fn is not None

    @property
    def can_compute_crps(self) -> bool:
        """CRPS sample-based approximation requires samples."""
        return self.samples is not None

    @property
    def can_compute_pinball(self) -> bool:
        """Pinball loss requires quantile arrays."""
        return self.quantiles is not None and len(self.quantiles) > 0

    @property
    def can_compute_interval(self) -> bool:
        """Interval / coverage diagnostics require lo and hi bounds."""
        return self.lo is not None and self.hi is not None

    @property
    def can_compute_energy_score(self) -> bool:
        """Energy Score requires joint samples (d ≥ 2)."""
        return (
            self.joint_samples is not None
            and self.joint_samples.ndim == 3
            and self.joint_samples.shape[2] >= 2
        )

    @property
    def capabilities(self) -> dict[str, bool]:
        """Return all capabilities as a dict."""
        return {
            "pit":          self.can_compute_pit,
            "crps":         self.can_compute_crps,
            "pinball":      self.can_compute_pinball,
            "interval":     self.can_compute_interval,
            "energy_score": self.can_compute_energy_score,
        }

    def require(self, capability: str) -> None:
        """
        Assert that a specific capability is available.
        Raises DiagnosticsInputError with an informative message if not.

        Parameters
        ----------
        capability : str
            One of: "pit", "crps", "pinball", "interval", "energy_score".
        """
        caps = self.capabilities
        if capability not in caps:
            raise DiagnosticsInputError(
                f"Unknown capability '{capability}'. "
                f"Valid options: {list(caps.keys())}"
            )
        if not caps[capability]:
            needed = {
                "pit":          "samples (n_obs, M) or a CDF callable",
                "crps":         "samples (n_obs, M)",
                "pinball":      "quantiles dict {float -> array(n_obs,)}",
                "interval":     "lo and hi arrays (n_obs,)",
                "energy_score": "joint_samples (n_obs, M, d) with d ≥ 2",
            }
            raise DiagnosticsInputError(
                f"Capability '{capability}' is not available for "
                f"model_id='{self.model_id}' (source: {self.source_dist_type}). "
                f"Required: {needed[capability]}."
            )

    def summary(self) -> dict:
        return {
            "model_id":         self.model_id,
            "n_obs":            self.n_obs,
            "source_dist_type": self.source_dist_type,
            "alpha":            self.alpha,
            "capabilities":     self.capabilities,
            "has_joint":        self.joint_samples is not None,
            "variable_names":   self.variable_names,
        }


# ---------------------------------------------------------------------------
# Diagnostics_Input — gateway class
# ---------------------------------------------------------------------------

class Diagnostics_Input:
    """
    Gateway that converts adapter outputs or raw arrays into a
    DiagnosticsReadyObject.

    Accepts two intake formats (auto-detected):
      Format A: adapter output objects
        - ResidualPool
        - MarginalSamples
        - QuantileFunctionObject
        - JointSimulationObject (extracts all marginals into one object)
      Format B: raw arrays (y, samples, quantiles, lo, hi)

    Parameters
    ----------
    alpha : float
        Default miscoverage level. Default 0.1.

    Example — Format A (from adapter):
    ------------------------------------
    >>> di = Diagnostics_Input()
    >>> obj = di.from_adapter(residual_pool)
    >>> obj.can_compute_pit
    True

    Example — Format B (raw arrays):
    ----------------------------------
    >>> di = Diagnostics_Input()
    >>> obj = di.from_arrays(
    ...     y=actuals, t=timestamps, model_id="entsoe",
    ...     samples=sample_array,
    ...     quantiles={0.05: lo, 0.95: hi},
    ...     lo=lo, hi=hi,
    ... )
    """

    def __init__(self, alpha: float = 0.1) -> None:
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        self.alpha = alpha

    # ── Format A: from adapter output ─────────────────────────────────────

    def from_adapter(
        self,
        adapter_output: Any,
        model_id:       str | None = None,
    ) -> DiagnosticsReadyObject:
        """
        Auto-detect adapter output type and convert to DiagnosticsReadyObject.

        Parameters
        ----------
        adapter_output : one of
            ResidualPool, MarginalSamples, QuantileFunctionObject,
            JointSimulationObject
        model_id : str, optional
            Override the model_id from the adapter output.

        Returns
        -------
        DiagnosticsReadyObject
        """
        if isinstance(adapter_output, ResidualPool):
            return self._from_residual_pool(adapter_output, model_id)
        elif isinstance(adapter_output, MarginalSamples):
            return self._from_marginal_samples(adapter_output, model_id)
        elif isinstance(adapter_output, QuantileFunctionObject):
            return self._from_quantile_function(adapter_output, model_id)
        elif isinstance(adapter_output, JointSimulationObject):
            return self._from_joint_simulation(adapter_output, model_id)
        else:
            raise DiagnosticsInputError(
                f"Unrecognised adapter output type: {type(adapter_output).__name__}. "
                "Expected one of: ResidualPool, MarginalSamples, "
                "QuantileFunctionObject, JointSimulationObject. "
                "For raw arrays, use Diagnostics_Input.from_arrays()."
            )

    # ── Format B: from raw arrays ──────────────────────────────────────────

    def from_arrays(
        self,
        *,
        y:              np.ndarray,
        t,
        model_id:       str,
        samples:        np.ndarray | None              = None,
        quantiles:      dict[float, np.ndarray] | None = None,
        lo:             np.ndarray | None              = None,
        hi:             np.ndarray | None              = None,
        cdf_fn:         Callable | None                = None,
        joint_samples:  np.ndarray | None              = None,
        variable_names: list[str] | None               = None,
        alpha:          float | None                   = None,
        source_dist_type: str                          = "raw_arrays",
    ) -> DiagnosticsReadyObject:
        """
        Build DiagnosticsReadyObject from raw arrays.

        Parameters
        ----------
        y : np.ndarray, shape (n_obs,)
        t : array-like of timestamps
        model_id : str
        samples : np.ndarray, optional, shape (n_obs, M)
        quantiles : dict, optional
        lo, hi : np.ndarray, optional, shape (n_obs,)
        cdf_fn : callable, optional
        joint_samples : np.ndarray, optional, shape (n_obs, M, d)
        variable_names : list[str], optional
        alpha : float, optional — overrides self.alpha
        source_dist_type : str

        Returns
        -------
        DiagnosticsReadyObject
        """
        a = alpha if alpha is not None else self.alpha
        y_arr, t_arr, n = self._validate_y_t(y, t, model_id)

        samples_arr   = self._validate_samples(samples, n, model_id)
        quantiles_val = self._validate_quantiles(quantiles, n, model_id)
        lo_arr, hi_arr = self._validate_interval(lo, hi, n, model_id)
        joint_arr     = self._validate_joint(joint_samples, n, model_id)

        return self._build(
            model_id        = model_id,
            n_obs           = n,
            t               = t_arr,
            y               = y_arr,
            alpha           = a,
            samples         = samples_arr,
            quantiles       = quantiles_val,
            lo              = lo_arr,
            hi              = hi_arr,
            cdf_fn          = cdf_fn,
            joint_samples   = joint_arr,
            variable_names  = variable_names,
            source_dist_type = source_dist_type,
            meta            = {"intake": "raw_arrays"},
        )

    # ── Private: adapter-specific converters ──────────────────────────────

    def _from_residual_pool(
        self,
        pool:     ResidualPool,
        model_id: str | None,
    ) -> DiagnosticsReadyObject:
        mid = model_id or pool.model_id
        n   = pool.n_obs

        # ResidualPool stores interval as pool_lo/pool_hi (absolute bounds)
        # and has no samples by default — samples are in entsoe_samples.npy
        # which the caller may pass separately. Here we extract what's available.
        return self._build(
            model_id         = mid,
            n_obs            = n,
            t                = pool.t_eval,
            y                = pool.y_eval,
            alpha            = pool.alpha,
            samples          = None,   # ResidualPool stores lo/hi not samples
            quantiles        = pool.to_quantiles(),
            lo               = pool.pool_lo,
            hi               = pool.pool_hi,
            cdf_fn           = None,
            joint_samples    = None,
            variable_names   = None,
            source_dist_type = "residual_reconstruction",
            meta             = {
                "intake":         "ResidualPool",
                "W":              pool.W,
                "bucket_fn_name": pool.bucket_fn_name,
            },
        )

    def _from_residual_pool_with_samples(
        self,
        pool:     ResidualPool,
        samples:  np.ndarray,
        model_id: str | None = None,
    ) -> DiagnosticsReadyObject:
        """
        Extended converter that also accepts an external samples array.
        Use when build scripts saved {asset}_samples.npy alongside lo/hi.
        """
        mid = model_id or pool.model_id
        n   = pool.n_obs
        samples_arr = self._validate_samples(samples, n, mid)

        return self._build(
            model_id         = mid,
            n_obs            = n,
            t                = pool.t_eval,
            y                = pool.y_eval,
            alpha            = pool.alpha,
            samples          = samples_arr,
            quantiles        = pool.to_quantiles(),
            lo               = pool.pool_lo,
            hi               = pool.pool_hi,
            cdf_fn           = None,
            joint_samples    = None,
            variable_names   = None,
            source_dist_type = "residual_reconstruction",
            meta             = {
                "intake":         "ResidualPool+samples",
                "W":              pool.W,
                "bucket_fn_name": pool.bucket_fn_name,
            },
        )

    def _from_marginal_samples(
        self,
        marginal: MarginalSamples,
        model_id: str | None,
    ) -> DiagnosticsReadyObject:
        mid = model_id or marginal.model_id
        q   = marginal.to_quantiles()
        lo  = q.get(marginal.alpha / 2)
        hi  = q.get(1 - marginal.alpha / 2)

        return self._build(
            model_id         = mid,
            n_obs            = marginal.n_timestamps,
            t                = marginal.t,
            y                = marginal.y,
            alpha            = marginal.alpha,
            samples          = marginal.samples,
            quantiles        = q,
            lo               = lo,
            hi               = hi,
            cdf_fn           = None,
            joint_samples    = None,
            variable_names   = [marginal.variable_name],
            source_dist_type = "empirical_joint",
            meta             = {"intake": "MarginalSamples",
                                "variable": marginal.variable_name},
        )

    def _from_quantile_function(
        self,
        qfo:      QuantileFunctionObject,
        model_id: str | None,
    ) -> DiagnosticsReadyObject:
        mid = model_id or qfo.model_id
        lo, hi = qfo.get_interval(alpha=qfo.alpha)

        return self._build(
            model_id         = mid,
            n_obs            = qfo.n_obs,
            t                = qfo.t,
            y                = qfo.y,
            alpha            = qfo.alpha,
            samples          = None,
            quantiles        = qfo.to_quantiles_dict(),
            lo               = lo,
            hi               = hi,
            cdf_fn           = qfo.interpolate_cdf,
            joint_samples    = None,
            variable_names   = None,
            source_dist_type = "quantile_function",
            meta             = {
                "intake":            "QuantileFunctionObject",
                "n_levels":          len(qfo.levels),
                "levels":            qfo.levels.tolist(),
                "n_crossings_fixed": qfo.n_crossings_fixed,
            },
        )

    def _from_joint_simulation(
        self,
        joint:    JointSimulationObject,
        model_id: str | None,
    ) -> DiagnosticsReadyObject:
        """
        For JointSimulationObject, use the first variable as the primary
        univariate series and attach joint_samples for Energy Score.
        For per-variable diagnostics, call from_adapter on each marginal.
        """
        mid = model_id or joint.model_id
        # Use first variable as primary univariate series
        first_var  = joint.variable_names[0]
        first_marg = joint.marginals[first_var]
        q          = first_marg.to_quantiles()
        lo         = q.get(first_marg.alpha / 2)
        hi         = q.get(1 - first_marg.alpha / 2)

        return self._build(
            model_id         = mid,
            n_obs            = joint.n_timestamps,
            t                = joint.t,
            y                = joint.y_joint[:, 0],
            alpha            = first_marg.alpha,
            samples          = first_marg.samples,
            quantiles        = q,
            lo               = lo,
            hi               = hi,
            cdf_fn           = None,
            joint_samples    = joint.samples_joint,
            variable_names   = joint.variable_names,
            source_dist_type = "empirical_joint",
            meta             = {
                "intake":         "JointSimulationObject",
                "d":              joint.d,
                "M":              joint.M,
                "primary_var":    first_var,
                "all_variables":  joint.variable_names,
            },
        )

    # ── Private: build and validate ────────────────────────────────────────

    def _build(
        self,
        *,
        model_id:         str,
        n_obs:            int,
        t:                np.ndarray,
        y:                np.ndarray,
        alpha:            float,
        samples:          np.ndarray | None,
        quantiles:        dict | None,
        lo:               np.ndarray | None,
        hi:               np.ndarray | None,
        cdf_fn:           Callable | None,
        joint_samples:    np.ndarray | None,
        variable_names:   list[str] | None,
        source_dist_type: str,
        meta:             dict,
    ) -> DiagnosticsReadyObject:
        """Core builder — enforces at-least-one-representation requirement."""
        has_any = any([
            samples is not None,
            quantiles is not None and len(quantiles) > 0,
            lo is not None and hi is not None,
            cdf_fn is not None,
        ])
        if not has_any:
            raise DiagnosticsInputError(
                f"Cannot build DiagnosticsReadyObject for model_id='{model_id}': "
                "at least one of samples / quantiles / (lo, hi) / cdf_fn "
                "must be provided."
            )

        # Time alignment check
        if len(t) != n_obs:
            raise DiagnosticsInputError(
                f"Time alignment error for model_id='{model_id}': "
                f"len(t)={len(t)} does not match n_obs={n_obs}."
            )
        if len(y) != n_obs:
            raise DiagnosticsInputError(
                f"Time alignment error for model_id='{model_id}': "
                f"len(y)={len(y)} does not match n_obs={n_obs}."
            )

        return DiagnosticsReadyObject(
            model_id         = model_id,
            n_obs            = n_obs,
            t                = t,
            y                = y,
            alpha            = alpha,
            samples          = samples,
            quantiles        = quantiles,
            lo               = lo,
            hi               = hi,
            cdf_fn           = cdf_fn,
            joint_samples    = joint_samples,
            variable_names   = variable_names,
            source_dist_type = source_dist_type,
            meta             = meta,
        )

    def _validate_y_t(
        self,
        y, t, model_id: str,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        import pandas as pd
        y_arr = np.asarray(y, dtype=float)
        if y_arr.ndim != 1:
            raise DiagnosticsInputError(
                f"y must be 1D for model_id='{model_id}', got shape {y_arr.shape}."
            )
        n = len(y_arr)
        if np.any(np.isnan(y_arr)):
            raise DiagnosticsInputError(
                f"y contains NaN values for model_id='{model_id}'."
            )
        # Parse timestamps
        if isinstance(t, pd.DatetimeIndex):
            t_arr = t.values.astype("datetime64[ns]")
        else:
            arr = np.asarray(t)
            if arr.dtype.kind == "M":
                t_arr = arr.astype("datetime64[ns]")
            elif arr.dtype.kind in ("i", "u", "f"):
                t_arr = arr.astype(np.int64)
            else:
                t_arr = pd.to_datetime(arr).values.astype("datetime64[ns]")
        return y_arr, t_arr, n

    def _validate_samples(
        self,
        samples, n: int, model_id: str,
    ) -> np.ndarray | None:
        if samples is None:
            return None
        s = np.asarray(samples, dtype=float)
        if s.ndim != 2 or s.shape[0] != n:
            raise DiagnosticsInputError(
                f"samples must have shape ({n}, M) for model_id='{model_id}', "
                f"got {s.shape}."
            )
        if not np.all(np.isfinite(s)):
            raise DiagnosticsInputError(
                f"samples contains NaN or Inf for model_id='{model_id}'."
            )
        return s

    def _validate_quantiles(
        self,
        quantiles, n: int, model_id: str,
    ) -> dict[float, np.ndarray] | None:
        if quantiles is None:
            return None
        out = {}
        for p, arr in quantiles.items():
            p = float(p)
            a = np.asarray(arr, dtype=float)
            if a.shape != (n,):
                raise DiagnosticsInputError(
                    f"quantile p={p} must have shape ({n},) for "
                    f"model_id='{model_id}', got {a.shape}."
                )
            out[p] = a
        return out if out else None

    def _validate_interval(
        self,
        lo, hi, n: int, model_id: str,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if lo is None and hi is None:
            return None, None
        if (lo is None) != (hi is None):
            raise DiagnosticsInputError(
                f"lo and hi must both be provided or both be None "
                f"for model_id='{model_id}'."
            )
        lo_arr = np.asarray(lo, dtype=float)
        hi_arr = np.asarray(hi, dtype=float)
        if lo_arr.shape != (n,) or hi_arr.shape != (n,):
            raise DiagnosticsInputError(
                f"lo and hi must have shape ({n},) for model_id='{model_id}'. "
                f"Got lo={lo_arr.shape}, hi={hi_arr.shape}."
            )
        if np.any(lo_arr > hi_arr + 1e-8):
            n_cross = int(np.sum(lo_arr > hi_arr + 1e-8))
            warnings.warn(
                f"{n_cross} observation(s) have lo > hi for "
                f"model_id='{model_id}'. Intervals may be invalid.",
                UserWarning,
                stacklevel=3,
            )
        return lo_arr, hi_arr

    def _validate_joint(
        self,
        joint_samples, n: int, model_id: str,
    ) -> np.ndarray | None:
        if joint_samples is None:
            return None
        j = np.asarray(joint_samples, dtype=float)
        if j.ndim != 3 or j.shape[0] != n:
            raise DiagnosticsInputError(
                f"joint_samples must have shape ({n}, M, d) for "
                f"model_id='{model_id}', got {j.shape}."
            )
        return j

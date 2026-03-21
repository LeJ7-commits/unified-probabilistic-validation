"""
src/adapters/simulation_joint.py
==================================
Adapter_SimulationJoint: converts joint Monte Carlo simulation output
into a JointSimulationObject suitable for both univariate marginal
diagnostics and multivariate joint evaluation.

Architecture role (Image 1):
  INPUT  : For each timestamp t:
             joint sample matrix S_t ∈ ℝ^(M×d)
             d = number of variables (e.g. price, temperature, gas, carbon)
             optional: weights w_t (scenario weights, shape (M,))
  OUTPUT : JointSimulationObject
             dist_type = "empirical_joint"
             samples   = S  (n_timestamps, M, d) array
             marginals = dict {variable_name -> MarginalSamples}
             meta      = {M, d, variable_names}

  ASSUMPTIONS (per diagram):
    - Samples represent predictive distribution at time t
    - Paths drawn conditionally on information available at t-1
    - Fixed M across timestamps

  SANITY CHECKS (per diagram):
    - M ≥ M_min (default 100, minimum for stable Energy Score)
    - No NaN or Inf values
    - Variance per dimension > ε (no degenerate distributions)
    - Dimensions consistent across timestamps

Input formats (auto-detected):
  Format A — sims_dict: dict[timestamp, dict[str, pd.DataFrame]]
    The structure produced by simulated_data.ipynb and
    simulated_data_extended.ipynb:
      sims_dict[asof_date][series_name] = DataFrame(index=hours, columns=paths)
    Realizations from realized_dict[series_name] are also accepted.

  Format B — 3D numpy array: shape (n_timestamps, M, d)
    Direct array input with variable_names list.

Output:
  JointSimulationObject contains both:
    - marginals: dict mapping each variable name to a MarginalSamples
      object compatible with existing univariate diagnostic pipeline
      (evaluate_distribution, run_diagnostics_policy)
    - joint: the full (n_timestamps, M, d) array for Energy Score and
      Dependence_Coherence diagnostics
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.core.data_contract import StandardizedModelObject, DataContract


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SimulationAdapterError(ValueError):
    """Raised when Adapter_SimulationJoint sanity checks fail."""


# ---------------------------------------------------------------------------
# MarginalSamples — per-variable output
# ---------------------------------------------------------------------------

@dataclass
class MarginalSamples:
    """
    Univariate marginal output for a single simulation variable.

    Compatible with evaluate_distribution() and run_diagnostics_policy()
    via the .to_samples() and .to_quantiles() methods.

    Attributes
    ----------
    variable_name : str
    model_id : str
    n_timestamps : int
        Number of evaluation timestamps.
    t : np.ndarray
        Timestamp array.
    y : np.ndarray, shape (n_timestamps,)
        Realizations.
    samples : np.ndarray, shape (n_timestamps, M)
        Simulation paths.
    weights : np.ndarray or None, shape (M,)
        Optional scenario weights (normalised to sum to 1).
    alpha : float
    """
    variable_name:  str
    model_id:       str
    n_timestamps:   int
    t:              np.ndarray
    y:              np.ndarray
    samples:        np.ndarray
    weights:        np.ndarray | None = None
    alpha:          float = 0.1

    def to_samples(self) -> np.ndarray:
        """Return samples array (n_timestamps, M) — input to evaluate_distribution."""
        return self.samples

    def to_quantiles(self) -> dict[float, np.ndarray]:
        """
        Compute empirical quantile bounds from samples.
        Returns {alpha/2: lo, 1-alpha/2: hi} compatible with run_diagnostics_policy.
        """
        lo = np.quantile(self.samples, self.alpha / 2, axis=1)
        hi = np.quantile(self.samples, 1 - self.alpha / 2, axis=1)
        return {self.alpha / 2: lo, 1 - self.alpha / 2: hi}

    def summary(self) -> dict:
        return {
            "variable_name": self.variable_name,
            "model_id":      self.model_id,
            "n_timestamps":  self.n_timestamps,
            "M":             self.samples.shape[1],
            "alpha":         self.alpha,
        }


# ---------------------------------------------------------------------------
# JointSimulationObject — combined output
# ---------------------------------------------------------------------------

@dataclass
class JointSimulationObject:
    """
    Output of Adapter_SimulationJoint.

    Contains both marginal per-variable outputs and the full joint
    sample array.

    Attributes
    ----------
    dist_type : str
        Always "empirical_joint".
    model_id : str
    n_timestamps : int
    M : int
        Number of Monte Carlo paths.
    d : int
        Number of variables (dimensions).
    variable_names : list[str]
    t : np.ndarray
        Timestamp array.
    y_joint : np.ndarray, shape (n_timestamps, d)
        Realizations for all variables.
    samples_joint : np.ndarray, shape (n_timestamps, M, d)
        Joint simulation paths.
    weights : np.ndarray or None, shape (M,)
        Optional normalised scenario weights.
    marginals : dict[str, MarginalSamples]
        Per-variable marginal objects, keyed by variable name.
    meta : dict
        Metadata: M, d, variable_names, sanity_flags.
    """
    dist_type:      str   = "empirical_joint"
    model_id:       str   = ""
    n_timestamps:   int   = 0
    M:              int   = 0
    d:              int   = 0
    variable_names: list[str] = field(default_factory=list)
    t:              np.ndarray = field(default_factory=lambda: np.array([]))
    y_joint:        np.ndarray = field(default_factory=lambda: np.array([]))
    samples_joint:  np.ndarray = field(default_factory=lambda: np.array([]))
    weights:        np.ndarray | None = None
    marginals:      dict[str, MarginalSamples] = field(default_factory=dict)
    meta:           dict = field(default_factory=dict)

    def get_marginal(self, variable_name: str) -> MarginalSamples:
        """Return marginal for a specific variable. Raises KeyError if missing."""
        if variable_name not in self.marginals:
            raise KeyError(
                f"Variable '{variable_name}' not found. "
                f"Available: {list(self.marginals.keys())}"
            )
        return self.marginals[variable_name]

    def summary(self) -> dict:
        return {
            "dist_type":      self.dist_type,
            "model_id":       self.model_id,
            "n_timestamps":   self.n_timestamps,
            "M":              self.M,
            "d":              self.d,
            "variable_names": self.variable_names,
        }


# ---------------------------------------------------------------------------
# Adapter_SimulationJoint
# ---------------------------------------------------------------------------

class Adapter_SimulationJoint:
    """
    Converts Monte Carlo simulation output into a JointSimulationObject.

    Accepts two input formats (auto-detected):
      Format A: sims_dict[asof][series_name] = pd.DataFrame (from notebooks)
      Format B: 3D numpy array, shape (n_timestamps, M, d)

    Parameters
    ----------
    variable_names : list[str], optional
        Required for Format B (3D array). For Format A, inferred from
        sims_dict keys.
    alpha : float
        Miscoverage level for marginal quantile bounds. Default 0.1.
    M_min : int
        Minimum number of paths. Default 100.
    var_min : float
        Minimum variance per dimension (degenerate check). Default 1e-8.
    model_id : str
        Identifier for the simulation model. Default "simulation".
    horizon_agg : str
        How to aggregate horizon steps within each as-of date.
        "mean" — average across all horizon steps (default)
        "first" — use only the first horizon step
        "all"  — keep all horizon steps (expands n_timestamps)

    Example — Format A (sims_dict from notebook):
    -----------------------------------------------
    >>> adapter = Adapter_SimulationJoint(model_id="sim_5d")
    >>> obj = adapter.from_sims_dict(
    ...     sims_dict=sims_dict,
    ...     realized_dict=realized_dict,
    ...     series_names=["price", "temp"],
    ... )

    Example — Format B (3D array):
    --------------------------------
    >>> adapter = Adapter_SimulationJoint(
    ...     variable_names=["price", "temp"],
    ...     model_id="sim_array"
    ... )
    >>> obj = adapter.from_array(
    ...     S=array_3d,           # (n_timestamps, M, d)
    ...     y=realizations_2d,    # (n_timestamps, d)
    ...     t=timestamps,
    ... )
    """

    def __init__(
        self,
        variable_names: list[str] | None = None,
        alpha:          float = 0.1,
        M_min:          int   = 100,
        var_min:        float = 1e-8,
        model_id:       str   = "simulation",
        horizon_agg:    str   = "mean",
    ) -> None:
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        if M_min < 1:
            raise ValueError(f"M_min must be ≥ 1, got {M_min}.")
        if horizon_agg not in ("mean", "first", "all"):
            raise ValueError(
                f"horizon_agg must be 'mean', 'first', or 'all', got '{horizon_agg}'."
            )

        self.variable_names = variable_names
        self.alpha          = alpha
        self.M_min          = M_min
        self.var_min        = var_min
        self.model_id       = model_id
        self.horizon_agg    = horizon_agg

    # ── Public API ──────────────────────────────────────────────────────────

    def from_sims_dict(
        self,
        sims_dict:     dict,
        realized_dict: dict[str, pd.DataFrame],
        series_names:  list[str] | None = None,
        weights:       np.ndarray | None = None,
    ) -> JointSimulationObject:
        """
        Build JointSimulationObject from sims_dict (Format A).

        Parameters
        ----------
        sims_dict : dict
            sims_dict[asof_date][series_name] = pd.DataFrame
            Index = horizon steps (1..n_horizons), columns = path_1..path_M
        realized_dict : dict[str, pd.DataFrame]
            realized_dict[series_name] = pd.DataFrame
            Index = asof_dates, columns = horizon steps
        series_names : list[str], optional
            Which series to extract. If None, uses all keys from first
            asof entry (sorted alphabetically for reproducibility).
        weights : np.ndarray, optional
            Scenario weights, shape (M,). Normalised internally.

        Returns
        -------
        JointSimulationObject
        """
        asof_dates = sorted(sims_dict.keys())
        if len(asof_dates) == 0:
            raise SimulationAdapterError("sims_dict is empty.")

        # Infer series names
        first_entry = sims_dict[asof_dates[0]]
        available   = sorted(first_entry.keys())
        if series_names is None:
            series_names = available
        else:
            missing = set(series_names) - set(available)
            if missing:
                raise SimulationAdapterError(
                    f"series_names {sorted(missing)} not found in sims_dict. "
                    f"Available: {available}"
                )

        d = len(series_names)
        n_ts = len(asof_dates)

        # Build (n_timestamps, M, d) array and (n_timestamps, d) realizations
        S_list: list[np.ndarray] = []   # each: (M, d) after horizon agg
        y_list: list[np.ndarray] = []   # each: (d,)
        M_val: int | None = None

        for asof in asof_dates:
            entry = sims_dict[asof]

            # Stack series into (n_horizons, M, d)
            series_arrays = []
            for s in series_names:
                df = entry[s]
                arr = df.values.astype(float)   # (n_horizons, M)
                series_arrays.append(arr)
            block = np.stack(series_arrays, axis=2)  # (n_horizons, M, d)

            # Aggregate horizons
            if self.horizon_agg == "mean":
                block_agg = block.mean(axis=0)   # (M, d)
            elif self.horizon_agg == "first":
                block_agg = block[0]              # (M, d)
            else:  # "all" — use first horizon for summary statistics
                block_agg = block.mean(axis=0)

            # Validate M consistency
            M_cur = block_agg.shape[0]
            if M_val is None:
                M_val = M_cur
            elif M_cur != M_val:
                raise SimulationAdapterError(
                    f"Inconsistent M across timestamps: expected {M_val}, "
                    f"got {M_cur} at asof={asof}."
                )

            S_list.append(block_agg)   # (M, d)

            # Realizations: mean across horizons
            y_row = []
            for s in series_names:
                real_df = realized_dict[s]
                if asof not in real_df.index:
                    raise SimulationAdapterError(
                        f"asof date {asof} not found in realized_dict['{s}']."
                    )
                vals = real_df.loc[asof].values.astype(float)
                if self.horizon_agg == "first":
                    y_row.append(float(vals[0]))
                else:
                    y_row.append(float(np.mean(vals)))
            y_list.append(y_row)

        # Shape: (n_timestamps, M, d) and (n_timestamps, d)
        S_arr = np.stack(S_list, axis=0)    # (n_timestamps, M, d)
        y_arr = np.array(y_list, dtype=float)   # (n_timestamps, d)

        t_arr = np.array(
            [pd.Timestamp(a).to_datetime64() for a in asof_dates],
            dtype="datetime64[ns]"
        )

        return self._build(
            S=S_arr, y=y_arr, t=t_arr,
            variable_names=series_names,
            weights=weights,
        )

    def from_array(
        self,
        S:              np.ndarray,
        y:              np.ndarray,
        t,
        variable_names: list[str] | None = None,
        weights:        np.ndarray | None = None,
    ) -> JointSimulationObject:
        """
        Build JointSimulationObject from a 3D numpy array (Format B).

        Parameters
        ----------
        S : np.ndarray, shape (n_timestamps, M, d)
        y : np.ndarray, shape (n_timestamps, d)  or  (n_timestamps,) if d=1
        t : array-like of timestamps, length n_timestamps
        variable_names : list[str], optional
            Names for each dimension. Defaults to ["var_0", "var_1", ...].
        weights : np.ndarray, optional
            Scenario weights, shape (M,).

        Returns
        -------
        JointSimulationObject
        """
        S = np.asarray(S, dtype=float)
        if S.ndim == 2:
            S = S[:, :, np.newaxis]   # treat as d=1
        if S.ndim != 3:
            raise SimulationAdapterError(
                f"S must have shape (n_timestamps, M, d), got {S.shape}."
            )

        n_ts, M, d = S.shape

        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y.shape != (n_ts, d):
            raise SimulationAdapterError(
                f"y must have shape ({n_ts}, {d}), got {y.shape}."
            )

        t_arr = self._parse_timestamps(t)
        if len(t_arr) != n_ts:
            raise SimulationAdapterError(
                f"t must have length {n_ts}, got {len(t_arr)}."
            )

        vnames = variable_names or self.variable_names
        if vnames is None:
            vnames = [f"var_{i}" for i in range(d)]
        if len(vnames) != d:
            raise SimulationAdapterError(
                f"variable_names must have length {d}, got {len(vnames)}."
            )

        return self._build(
            S=S, y=y, t=t_arr,
            variable_names=vnames,
            weights=weights,
        )

    # ── Internal builder ───────────────────────────────────────────────────

    def _build(
        self,
        S:              np.ndarray,
        y:              np.ndarray,
        t:              np.ndarray,
        variable_names: list[str],
        weights:        np.ndarray | None,
    ) -> JointSimulationObject:
        """Core builder — runs sanity checks and assembles output."""
        n_ts, M, d = S.shape

        # ── Sanity checks ─────────────────────────────────────────────────
        if M < self.M_min:
            raise SimulationAdapterError(
                f"M={M} is below minimum M_min={self.M_min}. "
                "Increase the number of Monte Carlo paths."
            )

        if not np.all(np.isfinite(S)):
            n_bad = int(np.sum(~np.isfinite(S)))
            raise SimulationAdapterError(
                f"S contains {n_bad} NaN or Inf value(s). "
                "All sample paths must be finite."
            )

        if not np.all(np.isfinite(y)):
            raise SimulationAdapterError(
                "y (realizations) contains NaN or Inf values."
            )

        # Variance check per dimension
        low_var_dims = []
        for i in range(d):
            var_i = float(np.var(S[:, :, i]))
            if var_i < self.var_min:
                low_var_dims.append((i, variable_names[i], var_i))

        if low_var_dims:
            names = [f"'{n}' (var={v:.2e})" for _, n, v in low_var_dims]
            raise SimulationAdapterError(
                f"Degenerate distribution detected in dimension(s): {names}. "
                f"Variance is below threshold var_min={self.var_min}."
            )

        # Weights
        w_arr: np.ndarray | None = None
        if weights is not None:
            w_arr = np.asarray(weights, dtype=float)
            if w_arr.shape != (M,):
                raise SimulationAdapterError(
                    f"weights must have shape ({M},), got {w_arr.shape}."
                )
            if np.any(w_arr < 0):
                raise SimulationAdapterError("weights must be non-negative.")
            w_sum = w_arr.sum()
            if w_sum < 1e-12:
                raise SimulationAdapterError("weights sum to zero.")
            w_arr = w_arr / w_sum   # normalise

        # ── Build marginals ───────────────────────────────────────────────
        marginals: dict[str, MarginalSamples] = {}
        for i, vname in enumerate(variable_names):
            marginals[vname] = MarginalSamples(
                variable_name = vname,
                model_id      = self.model_id,
                n_timestamps  = n_ts,
                t             = t,
                y             = y[:, i],
                samples       = S[:, :, i],   # (n_ts, M)
                weights       = w_arr,
                alpha         = self.alpha,
            )

        meta = {
            "M":              M,
            "d":              d,
            "variable_names": variable_names,
            "model_id":       self.model_id,
            "alpha":          self.alpha,
            "horizon_agg":    self.horizon_agg,
            "has_weights":    w_arr is not None,
        }

        return JointSimulationObject(
            dist_type      = "empirical_joint",
            model_id       = self.model_id,
            n_timestamps   = n_ts,
            M              = M,
            d              = d,
            variable_names = variable_names,
            t              = t,
            y_joint        = y,
            samples_joint  = S,
            weights        = w_arr,
            marginals      = marginals,
            meta           = meta,
        )

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _parse_timestamps(t) -> np.ndarray:
        if isinstance(t, pd.DatetimeIndex):
            return t.values.astype("datetime64[ns]")
        arr = np.asarray(t)
        if arr.dtype.kind == "M":
            return arr.astype("datetime64[ns]")
        try:
            return pd.to_datetime(arr).values.astype("datetime64[ns]")
        except Exception as exc:
            raise SimulationAdapterError(
                f"Cannot parse timestamps: {exc}"
            )

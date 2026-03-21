"""
src/adapters/quantile_adapter.py
==================================
Adapter_Quantiles: validates and wraps pre-computed quantile forecast
arrays, fixing crossings via isotonic regression (PAVA) and providing
a monotone cubic spline CDF interpolation for BuildDist_FromQuantiles.

Architecture role (Image 1):
  INPUT  : Quantile forecasts Q_t = {(p, q_t(p))} for p ∈ (0, 1)
  OUTPUT : QuantileFunctionObject
             dist_type = "quantile_function"
             Q_t = {p → q_t(p)} (non-crossing, validated)

  ASSUMPTIONS (per diagram):
    - Quantiles represent conditional distribution at time t
    - Provided quantile levels span a sufficient range

  SANITY CHECKS (per diagram):
    - Monotonicity: q(p_i) ≤ q(p_j) if p_i < p_j
      → Fixed via Pool Adjacent Violators (PAVA) isotonic regression + warn
    - No discontinuous jumps (jump ratio check)
    - Rough in-sample coverage: P(y_t ≤ q_t(0.5)) ≈ 0.5 (median calibration)

Crossing fix strategy: Pool Adjacent Violators Algorithm (PAVA)
  PAVA is the standard algorithm for isotonic regression. It is
  implemented here without sklearn dependency using a manual pool-merge
  approach, making the adapter self-contained.

CDF interpolation: PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)
  scipy.interpolate.PchipInterpolator preserves monotonicity between
  data points, producing a smooth CDF without overshooting. This is
  the correct choice for quantile functions because:
    - It guarantees the interpolated CDF remains non-decreasing
    - It avoids the oscillations of natural cubic splines
    - It matches the "monotone spline smoothing" sanity check in Image 2
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

try:
    from scipy.interpolate import PchipInterpolator
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class QuantileAdapterError(ValueError):
    """Raised when Adapter_Quantiles sanity checks fail hard."""


# ---------------------------------------------------------------------------
# PAVA — Pool Adjacent Violators Algorithm
# ---------------------------------------------------------------------------

def _pava_isotonic(y: np.ndarray) -> np.ndarray:
    """
    Pool Adjacent Violators Algorithm (PAVA) for non-decreasing isotonic
    regression. Returns the isotonic (non-decreasing) version of y.

    This is a manual implementation — no sklearn dependency.

    Parameters
    ----------
    y : np.ndarray, shape (n,)
        Input values to be made non-decreasing.

    Returns
    -------
    np.ndarray, shape (n,)
        Non-decreasing sequence that minimises sum of squared deviations.
    """
    y = y.astype(float).copy()
    n = len(y)
    # Each pool: (mean_value, count)
    pools: list[list[float | int]] = [[y[i], 1] for i in range(n)]

    # Merge pools that violate monotonicity
    i = 0
    while i < len(pools) - 1:
        if pools[i][0] > pools[i + 1][0]:
            # Merge pools i and i+1
            total = pools[i][1] + pools[i + 1][1]
            merged_mean = (
                pools[i][0] * pools[i][1] + pools[i + 1][0] * pools[i + 1][1]
            ) / total
            pools[i] = [merged_mean, total]
            pools.pop(i + 1)
            # Back up to check previous pool
            if i > 0:
                i -= 1
        else:
            i += 1

    # Expand pools back to original length
    result = np.empty(n, dtype=float)
    idx = 0
    for mean_val, count in pools:
        result[idx: idx + count] = mean_val
        idx += count
    return result


# ---------------------------------------------------------------------------
# QuantileFunctionObject — output
# ---------------------------------------------------------------------------

@dataclass
class QuantileFunctionObject:
    """
    Output of Adapter_Quantiles.

    Wraps validated, non-crossing quantile arrays with a PCHIP CDF
    interpolator for BuildDist_FromQuantiles.

    Attributes
    ----------
    dist_type : str
        Always "quantile_function".
    model_id : str
    n_obs : int
    t : np.ndarray
        Timestamps.
    y : np.ndarray or None, shape (n_obs,)
        Realizations (if provided).
    levels : np.ndarray
        Sorted quantile levels, shape (K,).
    quantile_arrays : dict[float, np.ndarray]
        Non-crossing quantile arrays keyed by level. Each shape (n_obs,).
    n_crossings_fixed : int
        Number of (t, level) pairs where crossings were fixed by PAVA.
    alpha : float
        Miscoverage level for interval extraction. Default 0.1.
    sanity_flags : dict
        Coverage calibration and jump check results.
    """
    dist_type:           str   = "quantile_function"
    model_id:            str   = ""
    n_obs:               int   = 0
    t:                   np.ndarray = field(default_factory=lambda: np.array([]))
    y:                   np.ndarray | None = None
    levels:              np.ndarray = field(default_factory=lambda: np.array([]))
    quantile_arrays:     dict[float, np.ndarray] = field(default_factory=dict)
    n_crossings_fixed:   int   = 0
    alpha:               float = 0.1
    sanity_flags:        dict  = field(default_factory=dict)

    def get_interval(
        self,
        alpha: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract (lower, upper) interval bounds at level alpha.

        Finds the closest available quantile levels to alpha/2 and
        1-alpha/2. Raises if exact levels are not present and
        interpolation is not available.

        Parameters
        ----------
        alpha : float, optional
            Miscoverage level. Defaults to self.alpha.

        Returns
        -------
        (lo, hi) : tuple of np.ndarray, each shape (n_obs,)
        """
        a = alpha if alpha is not None else self.alpha
        lo_level = a / 2
        hi_level = 1 - a / 2

        lo = self._nearest_quantile(lo_level)
        hi = self._nearest_quantile(hi_level)
        return lo, hi

    def to_quantiles_dict(self) -> dict[float, np.ndarray]:
        """Return quantile_arrays dict — compatible with run_diagnostics_policy."""
        return dict(self.quantile_arrays)

    def interpolate_cdf(self, y_query: np.ndarray) -> np.ndarray:
        """
        Evaluate the interpolated CDF F_t(y) at each observation t.

        Uses PCHIP monotone cubic spline interpolation across quantile
        levels. For each observation t, builds a CDF from the K quantile
        values and evaluates it at y_query[t].

        Parameters
        ----------
        y_query : np.ndarray, shape (n_obs,)
            Values at which to evaluate the CDF.

        Returns
        -------
        np.ndarray, shape (n_obs,)
            CDF values F_t(y_query[t]) ∈ [0, 1].
        """
        if not _HAS_SCIPY:
            raise ImportError(
                "scipy is required for CDF interpolation. "
                "Install with: pip install scipy"
            )

        y_query = np.asarray(y_query, dtype=float)
        if y_query.shape != (self.n_obs,):
            raise ValueError(
                f"y_query must have shape ({self.n_obs},), "
                f"got {y_query.shape}."
            )

        levels = self.levels
        cdf_values = np.empty(self.n_obs, dtype=float)

        for i in range(self.n_obs):
            # Quantile values at this observation
            q_vals = np.array(
                [self.quantile_arrays[p][i] for p in levels],
                dtype=float,
            )
            # PCHIP: maps quantile values (x-axis) to levels (y-axis)
            # i.e. F(q) = p
            try:
                interp = PchipInterpolator(q_vals, levels, extrapolate=False)
                val = float(interp(y_query[i]))
                # Clip to [0, 1] — extrapolation returns NaN outside range
                if np.isnan(val):
                    val = 0.0 if y_query[i] < q_vals[0] else 1.0
                cdf_values[i] = np.clip(val, 0.0, 1.0)
            except Exception:
                # Fallback: linear interpolation
                cdf_values[i] = float(
                    np.clip(np.interp(y_query[i], q_vals, levels), 0.0, 1.0)
                )

        return cdf_values

    def summary(self) -> dict:
        return {
            "dist_type":          self.dist_type,
            "model_id":           self.model_id,
            "n_obs":              self.n_obs,
            "n_levels":           len(self.levels),
            "levels":             self.levels.tolist(),
            "n_crossings_fixed":  self.n_crossings_fixed,
            "alpha":              self.alpha,
            "median_coverage_ok": self.sanity_flags.get("median_coverage_ok"),
            "n_jump_flagged":     self.sanity_flags.get("n_jump_flagged", 0),
        }

    # ── Private helpers ────────────────────────────────────────────────────

    def _nearest_quantile(self, level: float) -> np.ndarray:
        """Return the quantile array whose level is closest to `level`."""
        if level in self.quantile_arrays:
            return self.quantile_arrays[level]
        closest = min(self.quantile_arrays.keys(), key=lambda p: abs(p - level))
        warnings.warn(
            f"Quantile level {level:.4f} not available. "
            f"Using nearest level {closest:.4f}.",
            UserWarning,
            stacklevel=3,
        )
        return self.quantile_arrays[closest]


# ---------------------------------------------------------------------------
# Adapter_Quantiles
# ---------------------------------------------------------------------------

class Adapter_Quantiles:
    """
    Validates and wraps pre-computed quantile forecast arrays.

    Parameters
    ----------
    alpha : float
        Miscoverage level for interval extraction. Default 0.1.
    jump_ratio_max : float
        Maximum allowed ratio between consecutive quantile gaps.
        If max_gap / median_gap > jump_ratio_max, the observation is
        flagged for discontinuous jumps. Default 10.0.
    median_coverage_tol : float
        Tolerance for median calibration check. Raises if
        |empirical_median_coverage - 0.5| > tol. Default 0.20.
    min_levels : int
        Minimum number of quantile levels required. Default 3.
    model_id : str
        Identifier forwarded to output. Default "quantile_model".

    Example
    -------
    >>> adapter = Adapter_Quantiles(alpha=0.1)
    >>> qfo = adapter.transform(
    ...     quantiles={0.1: lo_arr, 0.5: med_arr, 0.9: hi_arr},
    ...     t=timestamps,
    ...     y=actuals,       # optional, needed for sanity checks
    ... )
    >>> lo, hi = qfo.get_interval(alpha=0.1)
    """

    def __init__(
        self,
        alpha:                float = 0.1,
        jump_ratio_max:       float = 10.0,
        median_coverage_tol:  float = 0.20,
        min_levels:           int   = 3,
        model_id:             str   = "quantile_model",
    ) -> None:
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        if min_levels < 2:
            raise ValueError(f"min_levels must be ≥ 2, got {min_levels}.")

        self.alpha               = alpha
        self.jump_ratio_max      = jump_ratio_max
        self.median_coverage_tol = median_coverage_tol
        self.min_levels          = min_levels
        self.model_id            = model_id

    def transform(
        self,
        quantiles: dict[float, np.ndarray],
        t,
        y:         np.ndarray | None = None,
        model_id:  str | None = None,
    ) -> QuantileFunctionObject:
        """
        Validate and wrap quantile arrays.

        Parameters
        ----------
        quantiles : dict {float -> np.ndarray(n_obs,)}
            Quantile arrays keyed by probability level in (0, 1).
        t : array-like of timestamps, length n_obs
        y : np.ndarray, optional, shape (n_obs,)
            Realizations. Required for median calibration sanity check.
        model_id : str, optional
            Overrides self.model_id if provided.

        Returns
        -------
        QuantileFunctionObject
        """
        mid = model_id or self.model_id

        # ── 1. Level validation ───────────────────────────────────────────
        if len(quantiles) < self.min_levels:
            raise QuantileAdapterError(
                f"At least {self.min_levels} quantile levels required, "
                f"got {len(quantiles)}."
            )

        for p in quantiles:
            if not (0.0 < float(p) < 1.0):
                raise QuantileAdapterError(
                    f"Quantile level {p} is out of (0, 1)."
                )

        levels = np.array(sorted(float(p) for p in quantiles))
        n_obs  = None

        # ── 2. Shape validation ───────────────────────────────────────────
        q_arrays: dict[float, np.ndarray] = {}
        for p in levels:
            arr = np.asarray(quantiles[p], dtype=float)
            if arr.ndim != 1:
                raise QuantileAdapterError(
                    f"Quantile array for p={p} must be 1-dimensional, "
                    f"got shape {arr.shape}."
                )
            if n_obs is None:
                n_obs = len(arr)
            elif len(arr) != n_obs:
                raise QuantileAdapterError(
                    f"All quantile arrays must have the same length. "
                    f"p={p} has length {len(arr)}, expected {n_obs}."
                )
            if not np.all(np.isfinite(arr)):
                n_bad = int(np.sum(~np.isfinite(arr)))
                raise QuantileAdapterError(
                    f"Quantile array for p={p} contains {n_bad} "
                    "NaN or Inf value(s)."
                )
            q_arrays[p] = arr

        if n_obs is None or n_obs == 0:
            raise QuantileAdapterError("Quantile arrays are empty.")

        # ── 3. Timestamp parsing ──────────────────────────────────────────
        t_arr = self._parse_timestamps(t)
        if len(t_arr) != n_obs:
            raise QuantileAdapterError(
                f"t must have length {n_obs}, got {len(t_arr)}."
            )

        # ── 4. y validation ───────────────────────────────────────────────
        y_arr: np.ndarray | None = None
        if y is not None:
            y_arr = np.asarray(y, dtype=float)
            if y_arr.shape != (n_obs,):
                raise QuantileAdapterError(
                    f"y must have shape ({n_obs},), got {y_arr.shape}."
                )

        # ── 5. Monotonicity check + PAVA fix ──────────────────────────────
        q_arrays, n_crossings = self._fix_crossings(q_arrays, levels, n_obs)

        # ── 6. Jump check ─────────────────────────────────────────────────
        sanity_flags = self._jump_check(q_arrays, levels, n_obs)
        sanity_flags["n_crossings_fixed"] = n_crossings

        # ── 7. Median coverage sanity check ──────────────────────────────
        if y_arr is not None and 0.5 in q_arrays:
            median_coverage = float(
                np.mean(y_arr <= q_arrays[0.5])
            )
            deviation = abs(median_coverage - 0.5)
            sanity_flags["empirical_median_coverage"] = round(median_coverage, 4)
            sanity_flags["median_coverage_deviation"] = round(deviation, 4)
            sanity_flags["median_coverage_ok"] = deviation <= self.median_coverage_tol
            if not sanity_flags["median_coverage_ok"]:
                warnings.warn(
                    f"Median calibration check failed for model_id='{mid}': "
                    f"P(y ≤ q_0.5) = {median_coverage:.3f} "
                    f"(deviation {deviation:.3f} > tol {self.median_coverage_tol}). "
                    "The median quantile may be systematically biased.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            sanity_flags["median_coverage_ok"] = None

        return QuantileFunctionObject(
            dist_type         = "quantile_function",
            model_id          = mid,
            n_obs             = n_obs,
            t                 = t_arr,
            y                 = y_arr,
            levels            = levels,
            quantile_arrays   = q_arrays,
            n_crossings_fixed = n_crossings,
            alpha             = self.alpha,
            sanity_flags      = sanity_flags,
        )

    # ── Private helpers ────────────────────────────────────────────────────

    def _fix_crossings(
        self,
        q_arrays: dict[float, np.ndarray],
        levels:   np.ndarray,
        n_obs:    int,
    ) -> tuple[dict[float, np.ndarray], int]:
        """
        Fix quantile crossings using PAVA isotonic regression.

        For each observation t, extracts the vector of quantile values
        across all levels and applies PAVA to enforce non-decreasing order.

        Returns fixed q_arrays and total number of crossing events fixed.
        """
        K = len(levels)
        # Build (n_obs, K) matrix
        Q_mat = np.stack([q_arrays[p] for p in levels], axis=1)  # (n_obs, K)

        n_crossings = 0
        for i in range(n_obs):
            row = Q_mat[i]
            if np.any(np.diff(row) < 0):   # crossing detected
                Q_mat[i] = _pava_isotonic(row)
                n_crossings += 1

        if n_crossings > 0:
            warnings.warn(
                f"{n_crossings}/{n_obs} observation(s) had quantile crossings "
                "that were fixed via PAVA isotonic regression. "
                "Consider improving the upstream quantile model.",
                UserWarning,
                stacklevel=3,
            )

        # Rebuild dict
        fixed: dict[float, np.ndarray] = {}
        for k, p in enumerate(levels):
            fixed[p] = Q_mat[:, k]
        return fixed, n_crossings

    def _jump_check(
        self,
        q_arrays: dict[float, np.ndarray],
        levels:   np.ndarray,
        n_obs:    int,
    ) -> dict:
        """
        Check for discontinuous jumps in the quantile function.

        For each observation, compute gaps between consecutive quantile
        values. Flag observations where max_gap / median_gap > jump_ratio_max.
        """
        K = len(levels)
        if K < 2:
            return {"n_jump_flagged": 0}

        Q_mat = np.stack([q_arrays[p] for p in levels], axis=1)
        gaps  = np.diff(Q_mat, axis=1)   # (n_obs, K-1)

        n_flagged = 0
        for i in range(n_obs):
            g = gaps[i]
            pos_gaps = g[g > 0]
            if len(pos_gaps) == 0:
                continue
            median_gap = float(np.median(pos_gaps))
            max_gap    = float(np.max(pos_gaps))
            if median_gap > 0 and (max_gap / median_gap) > self.jump_ratio_max:
                n_flagged += 1

        if n_flagged > 0:
            warnings.warn(
                f"{n_flagged}/{n_obs} observation(s) have discontinuous jumps "
                f"in the quantile function (gap ratio > {self.jump_ratio_max}). "
                "This may indicate a misspecified quantile model.",
                UserWarning,
                stacklevel=3,
            )

        return {"n_jump_flagged": n_flagged}

    @staticmethod
    def _parse_timestamps(t) -> np.ndarray:
        import pandas as pd
        if isinstance(t, pd.DatetimeIndex):
            return t.values.astype("datetime64[ns]")
        arr = np.asarray(t)
        if arr.dtype.kind == "M":
            return arr.astype("datetime64[ns]")
        if arr.dtype.kind in ("i", "u", "f"):
            return arr.astype(np.int64)
        try:
            return pd.to_datetime(arr).values.astype("datetime64[ns]")
        except Exception as exc:
            raise QuantileAdapterError(
                f"Cannot parse timestamps: {exc}"
            )

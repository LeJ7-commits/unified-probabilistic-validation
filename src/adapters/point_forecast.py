"""
src/adapters/point_forecast.py
================================
Adapter_PointForecast: converts a StandardizedModelObject with a point
forecast (y_hat) into a residual pool suitable for BuildDist_FromResiduals.

Architecture role (Image 1):
  INPUT  : StandardizedModelObject with y_hat present
  OUTPUT : ResidualPool dataclass
             dist_type = "residual_reconstruction"
             residual_pool = {e_{t-k} : k ∈ W}  (bucket-conditioned)
             optional: σ_t (robust scale estimate per observation)

  ASSUMPTIONS (per diagram):
    - Residuals approximately exchangeable within bucket
    - No structural drift within window W
    - No look-ahead bias (pool built from strictly past observations)

  SANITY CHECKS (per diagram):
    - residual_pool_size ≥ N_min_hard  (raises AdapterError if violated)
    - mean(residual_pool) ≈ 0          (bias flag, not error)
    - no extreme structural break inside W (variance ratio check)
    - no leakage from future timestamps   (enforced by strict past-only indexing)

Bucketing strategy — configurable via bucket_fn:
  The caller passes a callable `bucket_fn(timestamps) -> array of bucket ids`
  that assigns each observation to a bucket. Residuals are pooled within
  the same bucket to capture time-of-day or other conditional structure.

  Built-in bucket functions (importable from this module):
    bucket_hourly_24   — 24-bucket hour-of-day (renewables default)
    bucket_coarse_4    — 4-bucket coarse time-of-day (ENTSOE default)
    bucket_none        — single global bucket (no conditioning)

  Default: bucket_hourly_24 (most general; degrades gracefully for
  non-hourly data by collapsing to fewer effective buckets).

Design rationale:
  Making bucket_fn a parameter rather than hardcoding a strategy is the
  production-correct choice: different commodity classes have structurally
  different residual patterns (carbon has no intraday structure; gas has
  a morning peak; electricity has strong hour-of-day dependency). The
  framework adapts to each without modifying core adapter code.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from src.core.data_contract import StandardizedModelObject


# ---------------------------------------------------------------------------
# AdapterError
# ---------------------------------------------------------------------------

class AdapterError(ValueError):
    """Raised when an Adapter sanity check fails at the hard threshold."""


# ---------------------------------------------------------------------------
# Built-in bucket functions
# ---------------------------------------------------------------------------

def bucket_hourly_24(t: np.ndarray) -> np.ndarray:
    """
    24-bucket hour-of-day conditioning.
    Returns integer bucket ids in {0, 1, ..., 23}.

    Parameters
    ----------
    t : np.ndarray of datetime64[ns]

    Returns
    -------
    np.ndarray of int, shape (n,)
    """
    ts = pd.to_datetime(t)
    return ts.hour.to_numpy(dtype=int)


def bucket_coarse_4(t: np.ndarray) -> np.ndarray:
    """
    4-bucket coarse time-of-day conditioning (ENTSOE default).
      0 = Night     00:00–05:59
      1 = Morning   06:00–11:59
      2 = Afternoon 12:00–17:59
      3 = Evening   18:00–23:59

    Parameters
    ----------
    t : np.ndarray of datetime64[ns]

    Returns
    -------
    np.ndarray of int in {0, 1, 2, 3}, shape (n,)
    """
    hour = pd.to_datetime(t).hour.to_numpy(dtype=int)
    bucket = np.where(hour < 6, 0,
             np.where(hour < 12, 1,
             np.where(hour < 18, 2, 3)))
    return bucket


def bucket_none(t: np.ndarray) -> np.ndarray:
    """
    Single global bucket — no conditioning.
    All observations share bucket id 0.

    Parameters
    ----------
    t : np.ndarray

    Returns
    -------
    np.ndarray of zeros, shape (n,)
    """
    return np.zeros(len(t), dtype=int)


# Type alias for bucket functions
BucketFn = Callable[[np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
# ResidualPool output dataclass
# ---------------------------------------------------------------------------

@dataclass
class ResidualPool:
    """
    Output of Adapter_PointForecast.

    Produced per observation t — contains the conditional residual pool
    and optionally a per-observation robust scale estimate.

    Attributes
    ----------
    dist_type : str
        Always "residual_reconstruction".
    model_id : str
        Forwarded from the input StandardizedModelObject.
    n_obs : int
        Number of evaluable observations (after warmup burn).
    t_eval : np.ndarray
        Timestamps for evaluable observations.
    y_eval : np.ndarray, shape (n_obs,)
        Realizations for evaluable observations.
    y_hat_eval : np.ndarray, shape (n_obs,)
        Point forecasts for evaluable observations.
    residuals_eval : np.ndarray, shape (n_obs,)
        Residuals e_t = y_t − ŷ_t for evaluable observations.
    pool_lo : np.ndarray, shape (n_obs,)
        Lower quantile bound (alpha/2) of residual pool per observation.
    pool_hi : np.ndarray, shape (n_obs,)
        Upper quantile bound (1 - alpha/2) of residual pool per observation.
    pool_bias : np.ndarray, shape (n_obs,)
        Mean of residual pool per observation (bias estimate).
    pool_scale : np.ndarray, shape (n_obs,)
        Robust scale (1.4826 × MAD) of residual pool per observation.
    pool_sizes : np.ndarray of int, shape (n_obs,)
        Number of residuals in the pool for each evaluable observation.
    bucket_ids : np.ndarray of int, shape (n_obs,)
        Bucket assignment for each evaluable observation.
    alpha : float
        Miscoverage level used to compute pool_lo / pool_hi.
    W : int
        Trailing window size used.
    bucket_fn_name : str
        Name of the bucket function used (for reproducibility).
    sanity_flags : dict
        Per-observation sanity flags: bias_flag, break_flag.
    """
    dist_type:       str   = "residual_reconstruction"
    model_id:        str   = ""
    n_obs:           int   = 0
    t_eval:          np.ndarray = field(default_factory=lambda: np.array([]))
    y_eval:          np.ndarray = field(default_factory=lambda: np.array([]))
    y_hat_eval:      np.ndarray = field(default_factory=lambda: np.array([]))
    residuals_eval:  np.ndarray = field(default_factory=lambda: np.array([]))
    pool_lo:         np.ndarray = field(default_factory=lambda: np.array([]))
    pool_hi:         np.ndarray = field(default_factory=lambda: np.array([]))
    pool_bias:       np.ndarray = field(default_factory=lambda: np.array([]))
    pool_scale:      np.ndarray = field(default_factory=lambda: np.array([]))
    pool_sizes:      np.ndarray = field(default_factory=lambda: np.array([]))
    bucket_ids:      np.ndarray = field(default_factory=lambda: np.array([]))
    alpha:           float = 0.1
    W:               int   = 720
    bucket_fn_name:  str   = "bucket_hourly_24"
    sanity_flags:    dict  = field(default_factory=dict)

    def to_quantiles(self) -> dict[float, np.ndarray]:
        """
        Convert pool bounds to a quantiles dict compatible with
        evaluate_distribution() and run_diagnostics_policy().

        Returns
        -------
        {alpha/2: pool_lo, 1-alpha/2: pool_hi}
        """
        return {
            self.alpha / 2:       self.pool_lo,
            1 - self.alpha / 2:   self.pool_hi,
        }

    def summary(self) -> dict:
        """JSON-serialisable summary for logging."""
        return {
            "dist_type":       self.dist_type,
            "model_id":        self.model_id,
            "n_obs":           self.n_obs,
            "alpha":           self.alpha,
            "W":               self.W,
            "bucket_fn_name":  self.bucket_fn_name,
            "mean_pool_size":  float(np.mean(self.pool_sizes)) if len(self.pool_sizes) else 0,
            "mean_bias":       float(np.mean(np.abs(self.pool_bias))) if len(self.pool_bias) else 0,
            "n_bias_flagged":  int(self.sanity_flags.get("n_bias_flagged", 0)),
            "n_break_flagged": int(self.sanity_flags.get("n_break_flagged", 0)),
        }


# ---------------------------------------------------------------------------
# Adapter_PointForecast
# ---------------------------------------------------------------------------

class Adapter_PointForecast:
    """
    Converts a StandardizedModelObject (with y_hat) into a ResidualPool.

    Parameters
    ----------
    W : int
        Trailing window size — number of same-bucket past observations
        to include in the residual pool. Default 720 (30 days of hourly data).
    alpha : float
        Miscoverage level for pool quantile bounds (default 0.1 → 90%).
    bucket_fn : callable, optional
        Function mapping timestamps → bucket ids. Signature:
          bucket_fn(t: np.ndarray) -> np.ndarray of int
        Defaults to bucket_hourly_24.
        Built-ins: bucket_hourly_24, bucket_coarse_4, bucket_none.
    N_min_hard : int
        Hard minimum pool size. Raises AdapterError if any evaluable
        observation has fewer than this many residuals in its pool.
        Default 30.
    N_min_soft : int
        Soft minimum. Issues a warning (not an error) if the mean pool
        size falls below this. Default 60.
    bias_tol : float
        Bias tolerance. Observations where |pool_mean| > bias_tol × pool_std
        are flagged in sanity_flags["bias_flag"]. Default 0.3.
    break_var_ratio : float
        Structural break detection threshold. If the variance of the
        second half of the pool exceeds break_var_ratio × variance of the
        first half, the observation is flagged. Default 4.0.
    apply_bias_correction : bool
        If True, subtract pool mean from lo/hi bounds (bias correction).
        Default True.

    Example
    -------
    >>> adapter = Adapter_PointForecast(W=720, bucket_fn=bucket_hourly_24)
    >>> pool = adapter.transform(standardized_obj)
    >>> quantiles = pool.to_quantiles()
    """

    def __init__(
        self,
        W:                   int        = 720,
        alpha:               float      = 0.1,
        bucket_fn:           BucketFn   = bucket_hourly_24,
        N_min_hard:          int        = 30,
        N_min_soft:          int        = 60,
        bias_tol:            float      = 0.3,
        break_var_ratio:     float      = 4.0,
        apply_bias_correction: bool     = True,
    ) -> None:
        if W <= 0:
            raise ValueError(f"W must be positive, got {W}.")
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        if N_min_hard <= 0:
            raise ValueError(f"N_min_hard must be positive, got {N_min_hard}.")

        self.W                    = W
        self.alpha                = alpha
        self.bucket_fn            = bucket_fn
        self.N_min_hard           = N_min_hard
        self.N_min_soft           = N_min_soft
        self.bias_tol             = bias_tol
        self.break_var_ratio      = break_var_ratio
        self.apply_bias_correction = apply_bias_correction

    def transform(self, obj: StandardizedModelObject) -> ResidualPool:
        """
        Transform a StandardizedModelObject into a ResidualPool.

        Parameters
        ----------
        obj : StandardizedModelObject
            Must have y_hat present. Raises AdapterError otherwise.

        Returns
        -------
        ResidualPool
        """
        # ── Precondition: y_hat must exist ────────────────────────────────
        if not obj.has_point_forecast:
            raise AdapterError(
                f"Adapter_PointForecast requires y_hat, but model_id='{obj.model_id}' "
                "has no point forecast. Use Adapter_SimulationJoint or "
                "Adapter_Quantiles for other input types."
            )

        t       = obj.t
        y       = obj.y
        y_hat   = obj.y_hat
        n       = obj.n_obs

        # ── Compute residuals ─────────────────────────────────────────────
        residuals = y - y_hat   # e_t = y_t - ŷ_t, shape (n,)

        # ── Assign bucket ids ─────────────────────────────────────────────
        try:
            bucket_ids_all = self.bucket_fn(t)
        except Exception as exc:
            raise AdapterError(
                f"bucket_fn raised an error on timestamps of model_id='{obj.model_id}'. "
                f"Original error: {exc}"
            )
        bucket_ids_all = np.asarray(bucket_ids_all, dtype=int)
        if bucket_ids_all.shape != (n,):
            raise AdapterError(
                f"bucket_fn must return an array of shape ({n},), "
                f"got {bucket_ids_all.shape}."
            )

        # ── Rolling residual pool per observation ─────────────────────────
        lo_arr    = np.full(n, np.nan)
        hi_arr    = np.full(n, np.nan)
        bias_arr  = np.full(n, np.nan)
        scale_arr = np.full(n, np.nan)
        size_arr  = np.zeros(n, dtype=int)

        for t_idx in range(n):
            b = bucket_ids_all[t_idx]

            # Same-bucket past indices (strictly before t_idx — no leakage)
            past_same_bucket = np.where(
                (bucket_ids_all[:t_idx] == b)
            )[0]

            if len(past_same_bucket) < self.N_min_hard:
                # Not enough history yet — warmup period, skip
                continue

            # Take trailing W observations from same bucket
            pool_idx = past_same_bucket[-self.W:]
            pool     = residuals[pool_idx]

            # Bias correction
            bias = float(np.mean(pool))
            pool_c = pool - bias if self.apply_bias_correction else pool

            # Quantile bounds
            lo_arr[t_idx]    = float(np.quantile(pool_c, self.alpha / 2)) + bias
            hi_arr[t_idx]    = float(np.quantile(pool_c, 1 - self.alpha / 2)) + bias
            bias_arr[t_idx]  = bias
            scale_arr[t_idx] = self._robust_scale(pool_c)
            size_arr[t_idx]  = len(pool)

        # ── Trim to evaluable observations (warmup excluded) ──────────────
        mask   = ~np.isnan(lo_arr)
        n_eval = int(mask.sum())

        if n_eval == 0:
            raise AdapterError(
                f"No evaluable observations after warmup burn for model_id='{obj.model_id}'. "
                f"Total observations: {n}, W={self.W}, N_min_hard={self.N_min_hard}. "
                "Reduce W or N_min_hard, or provide more data."
            )

        t_eval         = t[mask]
        y_eval         = y[mask]
        y_hat_eval     = y_hat[mask]
        residuals_eval = residuals[mask]
        lo_eval        = lo_arr[mask]
        hi_eval        = hi_arr[mask]
        bias_eval      = bias_arr[mask]
        scale_eval     = scale_arr[mask]
        sizes_eval     = size_arr[mask]
        buckets_eval   = bucket_ids_all[mask]

        # ── Hard minimum check ────────────────────────────────────────────
        min_pool = int(sizes_eval.min())
        if min_pool < self.N_min_hard:
            raise AdapterError(
                f"Minimum pool size {min_pool} is below hard threshold "
                f"N_min_hard={self.N_min_hard} for model_id='{obj.model_id}'. "
                "Provide more historical data or reduce N_min_hard."
            )

        # ── Soft minimum warning ──────────────────────────────────────────
        mean_pool = float(sizes_eval.mean())
        if mean_pool < self.N_min_soft:
            warnings.warn(
                f"Mean pool size {mean_pool:.1f} is below soft threshold "
                f"N_min_soft={self.N_min_soft} for model_id='{obj.model_id}'. "
                "Quantile estimates may be unstable.",
                UserWarning,
                stacklevel=2,
            )

        # ── Sanity flags ──────────────────────────────────────────────────
        sanity_flags = self._compute_sanity_flags(
            bias_eval, scale_eval, residuals[mask], sizes_eval
        )

        # ── Add y_hat offset to lo/hi (interval = yhat + pool bounds) ────
        lo_final = y_hat_eval + lo_eval
        hi_final = y_hat_eval + hi_eval

        return ResidualPool(
            dist_type       = "residual_reconstruction",
            model_id        = obj.model_id,
            n_obs           = n_eval,
            t_eval          = t_eval,
            y_eval          = y_eval,
            y_hat_eval      = y_hat_eval,
            residuals_eval  = residuals_eval,
            pool_lo         = lo_final,
            pool_hi         = hi_final,
            pool_bias       = bias_eval,
            pool_scale      = scale_eval,
            pool_sizes      = sizes_eval,
            bucket_ids      = buckets_eval,
            alpha           = self.alpha,
            W               = self.W,
            bucket_fn_name  = getattr(self.bucket_fn, "__name__", "custom"),
            sanity_flags    = sanity_flags,
        )

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _robust_scale(x: np.ndarray) -> float:
        """1.4826 × MAD; falls back to std if MAD is degenerate."""
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        s   = 1.4826 * mad
        if not np.isfinite(s) or s <= 1e-12:
            s = float(np.std(x, ddof=1))
        return max(s, 1e-8)

    def _compute_sanity_flags(
        self,
        bias_arr:  np.ndarray,
        scale_arr: np.ndarray,
        resid_arr: np.ndarray,
        size_arr:  np.ndarray,
    ) -> dict:
        """
        Compute per-observation sanity flags.

        bias_flag : observations where |bias| > bias_tol × scale
        break_flag: observations where pool variance ratio > break_var_ratio
        """
        n = len(bias_arr)

        # Bias flag: |mean(pool)| > tol × scale
        bias_flag = np.abs(bias_arr) > (self.bias_tol * scale_arr)
        n_bias = int(bias_flag.sum())

        # Structural break flag (variance ratio of first vs second half)
        # Use residuals directly since we don't store per-obs pools
        # Approximate: rolling variance ratio over all residuals
        n_half = max(1, n // 2)
        var_first  = float(np.var(resid_arr[:n_half])) + 1e-12
        var_second = float(np.var(resid_arr[n_half:])) + 1e-12
        ratio = max(var_first / var_second, var_second / var_first)
        break_detected = ratio > self.break_var_ratio
        n_break = n if break_detected else 0

        if n_bias > 0:
            warnings.warn(
                f"{n_bias}/{n} observations have residual pool bias exceeding "
                f"tolerance ({self.bias_tol} × scale). Consider applying "
                "a longer warm-up window or regime-conditioned bucketing.",
                UserWarning,
                stacklevel=3,
            )
        if break_detected:
            warnings.warn(
                f"Structural break detected: variance ratio {ratio:.2f} exceeds "
                f"threshold {self.break_var_ratio}. Residuals show significant "
                "volatility shift across the evaluation period.",
                UserWarning,
                stacklevel=3,
            )

        return {
            "bias_flag":       bias_flag.tolist(),
            "n_bias_flagged":  n_bias,
            "break_flag":      break_detected,
            "variance_ratio":  round(ratio, 4),
            "n_break_flagged": n_break,
        }

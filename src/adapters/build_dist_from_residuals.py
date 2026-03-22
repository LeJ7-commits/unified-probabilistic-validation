"""
src/adapters/build_dist_from_residuals.py
==========================================
BuildDist_FromResiduals: reconstructs a predictive distribution from a
ResidualPool by generating Monte Carlo sample paths.

Architecture role (Image 2 — BuildDist layer):
  INPUT  : ResidualPool (from Adapter_PointForecast)
  OUTPUT : SampleMatrix — shape (n_obs, M)
             dist_type = "residual_reconstruction"
             samples[t, :] = y_hat[t] + bootstrap/parametric draws
                             from the residual pool at time t

Two reconstruction modes:
  non_parametric (default):
    Bootstrap resamples M residuals from the pool at each t.
    No distribution assumption. Preserves skewness, heavy tails,
    and any non-Gaussian structure in the residuals.
    Correct choice for energy forecasts with asymmetric errors.

  parametric (Gaussian):
    Fits N(mean, std) to the residual pool at each t.
    Faster. Appropriate only if residuals are approximately Gaussian.
    Understates tail risk for heavy-tailed commodities.

The output SampleMatrix is directly compatible with:
  - Diagnostics_Input.from_arrays(samples=...)
  - evaluate_distribution(samples=...)
  - run_diagnostics_policy(samples=...)
  - Score_Pinball, CRPS, PIT computation

This is the missing link between Adapter_PointForecast (which produces
lo/hi bounds) and PIT-based diagnostics (which require sample paths).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from src.adapters.point_forecast import ResidualPool


# ---------------------------------------------------------------------------
# SampleMatrix — output
# ---------------------------------------------------------------------------

@dataclass
class SampleMatrix:
    """
    Output of BuildDist_FromResiduals.

    Attributes
    ----------
    dist_type : str
        Always "residual_reconstruction".
    model_id : str
    n_obs : int
    M : int
        Number of sample paths.
    t : np.ndarray
        Timestamps for evaluable observations.
    y : np.ndarray, shape (n_obs,)
        Realizations.
    y_hat : np.ndarray, shape (n_obs,)
        Point forecasts.
    samples : np.ndarray, shape (n_obs, M)
        Predictive sample paths: y_hat[t] + residual_draws[t].
    mode : str
        "non_parametric" or "parametric".
    """
    dist_type:  str
    model_id:   str
    n_obs:      int
    M:          int
    t:          np.ndarray
    y:          np.ndarray
    y_hat:      np.ndarray
    samples:    np.ndarray
    mode:       str

    def to_quantiles(self, alpha: float = 0.1) -> dict[float, np.ndarray]:
        """Extract empirical quantile bounds from samples."""
        lo = np.quantile(self.samples, alpha / 2, axis=1)
        hi = np.quantile(self.samples, 1 - alpha / 2, axis=1)
        return {alpha / 2: lo, 1 - alpha / 2: hi}

    def summary(self) -> dict:
        return {
            "dist_type": self.dist_type,
            "model_id":  self.model_id,
            "n_obs":     self.n_obs,
            "M":         self.M,
            "mode":      self.mode,
        }


# ---------------------------------------------------------------------------
# BuildDist_FromResiduals
# ---------------------------------------------------------------------------

class BuildDist_FromResiduals:
    """
    Reconstructs predictive sample paths from a ResidualPool.

    Parameters
    ----------
    M : int
        Number of sample paths to generate. Default 500.
    mode : str
        "non_parametric" — bootstrap resampling (default, recommended)
        "parametric"     — Gaussian fit to residual pool
    seed : int or None
        Random seed for reproducibility. Default 42.
    clip_quantile : float or None
        If set, clips extreme residual samples beyond this quantile of the
        pool to prevent runaway tail samples in bootstrap mode.
        E.g. 0.999 clips the top/bottom 0.1% of draws. Default None.

    Example
    -------
    >>> builder = BuildDist_FromResiduals(M=500, mode="non_parametric")
    >>> matrix = builder.build(residual_pool)
    >>> # Use in Diagnostics_Input:
    >>> dro = di.from_arrays(y=matrix.y, t=matrix.t, model_id="m",
    ...                      samples=matrix.samples, lo=lo, hi=hi)
    """

    def __init__(
        self,
        M:             int   = 500,
        mode:          str   = "non_parametric",
        seed:          int | None = 42,
        clip_quantile: float | None = None,
    ) -> None:
        if mode not in ("non_parametric", "parametric"):
            raise ValueError(
                f"mode must be 'non_parametric' or 'parametric', got '{mode}'."
            )
        if M < 10:
            raise ValueError(f"M must be ≥ 10, got {M}.")

        self.M             = M
        self.mode          = mode
        self.rng           = np.random.default_rng(seed)
        self.clip_quantile = clip_quantile

    def build(self, pool: ResidualPool) -> SampleMatrix:
        """
        Build predictive sample paths from a ResidualPool.

        Parameters
        ----------
        pool : ResidualPool
            Output of Adapter_PointForecast.transform().

        Returns
        -------
        SampleMatrix
        """
        n = pool.n_obs
        samples = np.empty((n, self.M), dtype=float)

        if self.mode == "non_parametric":
            samples = self._bootstrap(pool, n, samples)
        else:
            samples = self._parametric(pool, n, samples)

        # Add y_hat offset: samples are residual draws, shift to predictive
        samples = pool.y_hat_eval[:, None] + samples

        return SampleMatrix(
            dist_type = "residual_reconstruction",
            model_id  = pool.model_id,
            n_obs     = n,
            M         = self.M,
            t         = pool.t_eval,
            y         = pool.y_eval,
            y_hat     = pool.y_hat_eval,
            samples   = samples,
            mode      = self.mode,
        )

    # ── Private ──────────────────────────────────────────────────────────

    def _bootstrap(
        self,
        pool:    ResidualPool,
        n:       int,
        out:     np.ndarray,
    ) -> np.ndarray:
        """
        Non-parametric bootstrap: resample from the empirical residual pool.

        For each observation t, draws M residuals with replacement from
        the bucket-conditioned residual pool. Since ResidualPool stores
        only lo/hi/bias/scale (not the full pool), we reconstruct an
        approximate pool from the pool statistics using the empirical
        quantile bounds plus the residuals_eval series.

        Fallback: if the pool is small, use the global residual series.
        """
        global_residuals = pool.residuals_eval  # shape (n,)

        # Clip threshold
        clip_lo = clip_hi = None
        if self.clip_quantile is not None:
            q = self.clip_quantile
            clip_lo = float(np.quantile(global_residuals, 1 - q))
            clip_hi = float(np.quantile(global_residuals, q))

        for t in range(n):
            # Use global residuals for bootstrap pool
            # (bucket-specific pool not stored in ResidualPool)
            pool_size = pool.pool_sizes[t]
            if pool_size >= 20:
                # Use the most recent pool_size residuals from the same
                # approximate bucket — approximate by using trailing window
                start = max(0, t - pool_size)
                local_pool = global_residuals[start:t]
                if len(local_pool) < 5:
                    local_pool = global_residuals
            else:
                local_pool = global_residuals

            draws = self.rng.choice(local_pool, size=self.M, replace=True)

            if clip_lo is not None:
                draws = np.clip(draws, clip_lo, clip_hi)

            out[t] = draws

        return out

    def _parametric(
        self,
        pool: ResidualPool,
        n:    int,
        out:  np.ndarray,
    ) -> np.ndarray:
        """
        Parametric Gaussian: fit N(bias_t, scale_t) per observation.

        Uses pool_bias and pool_scale stored in the ResidualPool.
        Faster than bootstrap but assumes Gaussian residuals.
        """
        for t in range(n):
            mu    = float(pool.pool_bias[t])
            sigma = float(pool.pool_scale[t])
            out[t] = self.rng.normal(mu, max(sigma, 1e-8), size=self.M)

        return out

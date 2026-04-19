"""
This module implements the Student-t copula.

The Student-t copula is parameterised by a correlation matrix ``R``
and a degrees-of-freedom parameter ``df``.  Given uniform pseudo-
observations ``u``, the link is the t-quantile transform
``x = t_ν⁻¹(u)``: the Student-t copula is the distribution of
``t_ν(X)`` where ``X`` is multivariate-t with shape matrix ``R`` and
degrees of freedom ``ν``.

Estimation follows the standard two-step recipe:

1. Estimate ``R`` by inverting Kendall's tau elementwise via
   ``ρ_ij = sin(π τ_ij / 2)`` and projecting to the PSD cone.
2. Estimate ``df`` by maximising the t-copula log-likelihood as a
   one-dimensional bounded optimisation.

References:
- Demarta, S. and McNeil, A. J. (2005). The t copula and related
  copulas. *International Statistical Review*, 73(1), 111-129.

Author:
- Jan Schlegel
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize_scalar
from scipy.stats import kendalltau, multivariate_t, t

from copulalib.copulas.base import Copula
from copulalib.utils.correlation import to_correlation


# =========================================================================== #
#                         Student-t Copula Class                              #
# =========================================================================== #
class StudentTCopula(Copula):
    """Student-t copula parameterised by a correlation matrix and df.

    Parameters
    ----------
    corr : array_like of shape (d, d), optional
        Pre-specified correlation matrix.  If ``None`` (default),
        estimated from data via Kendall's tau inversion during
        ``fit_copula``.
    df : float, optional
        Pre-specified degrees of freedom (must be > 2).  If ``None``
        (default), estimated by 1-D bounded MLE during
        ``fit_copula``.

    Attributes
    ----------
    corr : NDArray of shape (d, d) or None
        Fitted correlation matrix.
    df : float or None
        Fitted degrees of freedom.
    """

    def __init__(
        self,
        corr: ArrayLike | None = None,
        df: float | None = None,
    ) -> None:
        super().__init__()
        self.corr: NDArray[np.float64] | None = (
            None
            if corr is None
            else to_correlation(np.asarray(corr, dtype=np.float64))
        )
        self.df: float | None = df

    # -------------------------------------------------------------------------
    #  Fitting
    # -------------------------------------------------------------------------
    def _fit_copula(
        self,
        u: NDArray[np.float64],
        df_bounds: tuple[float, float] = (2.5, 50.0),
        **kwargs: Any,
    ) -> None:
        """Estimate ``corr`` (Kendall's tau inversion) and ``df``
        (1-D bounded MLE).

        Steps that have a pre-specified value are skipped.

        Parameters
        ----------
        u : NDArray of shape (n, d)
            Uniform pseudo-observations in the open unit hypercube.
        df_bounds : tuple of float, default (2.5, 50.0)
            Search interval for the degrees-of-freedom optimisation.
        """
        u_clip = np.clip(u, 1e-12, 1.0 - 1e-12)
        d = u_clip.shape[1]

        # ---------------------------------------------------------------------
        # Step 1 — correlation via Kendall's tau, then PSD projection
        # ---------------------------------------------------------------------
        if self.corr is None:
            tau = np.eye(d)
            for i in range(d):
                for j in range(i + 1, d):
                    tau_ij = float(
                        kendalltau(u_clip[:, i], u_clip[:, j]).statistic
                    )
                    tau[i, j] = tau_ij
                    tau[j, i] = tau_ij
            self.corr = to_correlation(np.sin(np.pi * tau / 2.0))

        # ---------------------------------------------------------------------
        # Step 2 — degrees of freedom via 1-D log-likelihood
        # ---------------------------------------------------------------------
        if self.df is None:
            corr = self.corr
            mean = np.zeros(d)

            def neg_log_lik(nu: float) -> float:
                x = t.ppf(u_clip, df=nu)  # (n, d)
                log_joint = multivariate_t.logpdf(
                    x, loc=mean, shape=corr, df=nu
                )
                log_marginals = np.sum(
                    t.logpdf(x, df=nu), axis=1
                )
                return float(-np.sum(log_joint - log_marginals))

            res = minimize_scalar(
                neg_log_lik, bounds=df_bounds, method="bounded"
            )
            self.df = float(res.x)

    # -------------------------------------------------------------------------
    #  Sampling
    # -------------------------------------------------------------------------
    def sample(self, n: int, **kwargs: Any) -> NDArray[np.float64]:
        """Draw ``n`` samples in uniform (copula) space.

        Uses the standard construction
        ``X = Z / sqrt(W / ν)`` with ``Z ~ N(0, R)`` and
        ``W ~ χ²(ν)``, then returns ``t_ν(X)``.

        Parameters
        ----------
        n : int
            Number of samples.
        rng : np.random.Generator, optional
            Random number generator (passed via ``**kwargs``).

        Returns
        -------
        NDArray of shape (n, d)
            Samples in the open unit hypercube.
        """
        self._check_fitted_params()
        assert self.corr is not None and self.df is not None
        rng: np.random.Generator = kwargs.get(
            "rng", np.random.default_rng()
        )
        d = self.corr.shape[0]
        z = rng.multivariate_normal(
            mean=np.zeros(d), cov=self.corr, size=n
        )  # (n, d)
        w = rng.chisquare(self.df, size=n)  # (n,)
        x = z / np.sqrt(w / self.df)[:, None]
        return np.asarray(t.cdf(x, df=self.df), dtype=np.float64)

    # -------------------------------------------------------------------------
    #  CDF / PDF / log-PDF
    # -------------------------------------------------------------------------
    def cdf(self, u: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Student-t copula CDF.

        Parameters
        ----------
        u : array_like of shape (n, d) or (d,)
            Points in the unit hypercube.

        Returns
        -------
        NDArray
            Copula CDF values in [0, 1].
        """
        self._check_fitted_params()
        assert self.corr is not None and self.df is not None
        u_arr = np.asarray(u, dtype=np.float64)
        x = t.ppf(np.clip(u_arr, 1e-12, 1.0 - 1e-12), df=self.df)
        d = self.corr.shape[0]
        return np.asarray(
            multivariate_t.cdf(
                x, loc=np.zeros(d), shape=self.corr, df=self.df
            ),
            dtype=np.float64,
        )

    def pdf(self, u: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Student-t copula density.

        Parameters
        ----------
        u : array_like of shape (n, d) or (d,)
            Points in the open unit hypercube.

        Returns
        -------
        NDArray
            Density values (non-negative).
        """
        return np.exp(self.logpdf(u))

    def logpdf(
        self, u: ArrayLike, **kwargs: Any
    ) -> NDArray[np.float64]:
        """Log Student-t copula density.

        Computed as the difference between the multivariate-t log
        density and the sum of the marginal univariate-t log
        densities, evaluated at ``x = t_ν⁻¹(u)``.

        Parameters
        ----------
        u : array_like of shape (n, d) or (d,)
            Points in the open unit hypercube.

        Returns
        -------
        NDArray
            Log-density values.
        """
        self._check_fitted_params()
        assert self.corr is not None and self.df is not None
        u_arr = np.atleast_2d(np.asarray(u, dtype=np.float64))
        x = t.ppf(np.clip(u_arr, 1e-12, 1.0 - 1e-12), df=self.df)
        d = self.corr.shape[0]
        log_joint = multivariate_t.logpdf(
            x, loc=np.zeros(d), shape=self.corr, df=self.df
        )
        log_marginals = np.sum(t.logpdf(x, df=self.df), axis=1)
        return np.asarray(log_joint - log_marginals, dtype=np.float64)

    # -------------------------------------------------------------------------
    #  Helpers
    # -------------------------------------------------------------------------
    def _check_fitted_params(self) -> None:
        """Raise if either ``corr`` or ``df`` has not been set."""
        if self.corr is None or self.df is None:
            raise RuntimeError(
                "StudentTCopula: call fit_copula() (or pass both "
                "`corr` and `df` to __init__) before this operation."
            )

    def __repr__(self) -> str:
        if self.corr is None or self.df is None:
            return "StudentTCopula(unfitted)"
        return (
            f"StudentTCopula(d={self.corr.shape[0]}, "
            f"df={self.df:.4g})"
        )

"""
This module implements the Gaussian (normal) copula.

The Gaussian copula is parameterised by a correlation matrix ``R``
(symmetric positive definite with unit diagonal).  Given uniform
pseudo-observations ``u``, the copula links them through the normal
score transform ``z = Φ⁻¹(u)``: the Gaussian copula is the
distribution of ``Φ(Z)`` where ``Z ~ N(0, R)``.

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
from scipy.stats import multivariate_normal, norm

from copulalib.copulas.base import Copula
from copulalib.utils.correlation import to_correlation


# =========================================================================== #
#                          Gaussian Copula Class                              #
# =========================================================================== #
class GaussianCopula(Copula):
    """Gaussian copula parameterised by a correlation matrix.

    Estimation is by maximum-likelihood on the normal scores: each
    column of ``u`` is mapped to ``z = Φ⁻¹(u)`` and the sample
    correlation matrix of ``z`` is computed.  The estimate is then
    projected onto the set of valid correlation matrices via
    eigenvalue clipping (see ``copulalib.utils.correlation``).

    Parameters
    ----------
    corr : array_like of shape (d, d), optional
        Pre-specified correlation matrix.  If ``None`` (default),
        estimated from data during ``fit_copula``.

    Attributes
    ----------
    corr : NDArray of shape (d, d) or None
        Fitted correlation matrix.  ``None`` until fitted.
    """

    def __init__(
        self,
        corr: ArrayLike | None = None,
    ) -> None:
        super().__init__()
        self.corr: NDArray[np.float64] | None = (
            None
            if corr is None
            else to_correlation(np.asarray(corr, dtype=np.float64))
        )

    # -------------------------------------------------------------------------
    #  Fitting
    # -------------------------------------------------------------------------
    def _fit_copula(
        self,
        u: NDArray[np.float64],
        **kwargs: Any,
    ) -> None:
        """Estimate the correlation matrix from uniform pseudo-
        observations.

        Skipped if ``corr`` was pre-specified in ``__init__``.

        Parameters
        ----------
        u : NDArray of shape (n, d)
            Uniform pseudo-observations in the open unit hypercube.
        """
        if self.corr is not None:
            return

        # ---------------------------------------------------------------------
        # Map to normal scores and take the sample correlation
        # ---------------------------------------------------------------------
        u_clip = np.clip(u, 1e-12, 1.0 - 1e-12)
        z = norm.ppf(u_clip)  # (n, d)
        cov = np.cov(z, rowvar=False, ddof=1)  # (d, d)
        d_inv_sqrt = 1.0 / np.sqrt(np.diag(cov))
        corr = cov * np.outer(d_inv_sqrt, d_inv_sqrt)
        self.corr = to_correlation(corr)

    # -------------------------------------------------------------------------
    #  Sampling
    # -------------------------------------------------------------------------
    def sample(self, n: int, **kwargs: Any) -> NDArray[np.float64]:
        """Draw ``n`` samples in uniform (copula) space.

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
        self._check_corr_fitted()
        assert self.corr is not None
        rng: np.random.Generator = kwargs.get(
            "rng", np.random.default_rng()
        )
        d = self.corr.shape[0]
        z = rng.multivariate_normal(
            mean=np.zeros(d), cov=self.corr, size=n
        )  # (n, d)
        return np.asarray(norm.cdf(z), dtype=np.float64)

    # -------------------------------------------------------------------------
    #  CDF / PDF / log-PDF
    # -------------------------------------------------------------------------
    def cdf(self, u: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Gaussian copula CDF.

        Parameters
        ----------
        u : array_like of shape (n, d) or (d,)
            Points in the unit hypercube.

        Returns
        -------
        NDArray
            Copula CDF values in [0, 1].
        """
        self._check_corr_fitted()
        assert self.corr is not None
        u_arr = np.asarray(u, dtype=np.float64)
        z = norm.ppf(np.clip(u_arr, 1e-12, 1.0 - 1e-12))
        d = self.corr.shape[0]
        return np.asarray(
            multivariate_normal.cdf(
                z, mean=np.zeros(d), cov=self.corr
            ),
            dtype=np.float64,
        )

    def pdf(self, u: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Gaussian copula density.

        Parameters
        ----------
        u : array_like of shape (n, d) or (d,)
            Points in the open unit hypercube.

        Returns
        -------
        NDArray
            Copula density values (non-negative).
        """
        return np.exp(self.logpdf(u))

    def logpdf(
        self, u: ArrayLike, **kwargs: Any
    ) -> NDArray[np.float64]:
        """Log Gaussian copula density.

        Uses the closed-form expression

            log c(u) = -½ log|R| - ½ z'(R⁻¹ - I) z

        where ``z = Φ⁻¹(u)``.

        Parameters
        ----------
        u : array_like of shape (n, d) or (d,)
            Points in the open unit hypercube.

        Returns
        -------
        NDArray
            Log-density values.
        """
        self._check_corr_fitted()
        assert self.corr is not None
        u_arr = np.atleast_2d(np.asarray(u, dtype=np.float64))
        z = norm.ppf(np.clip(u_arr, 1e-12, 1.0 - 1e-12))  # (n, d)
        r = self.corr
        d = r.shape[0]

        # ---------------------------------------------------------------------
        # log|R| and quadratic form z' (R⁻¹ - I) z
        # ---------------------------------------------------------------------
        _, logdet = np.linalg.slogdet(r)
        r_inv_minus_i = np.linalg.inv(r) - np.eye(d)
        quad = np.einsum("ni,ij,nj->n", z, r_inv_minus_i, z)
        return np.asarray(-0.5 * logdet - 0.5 * quad, dtype=np.float64)

    # -------------------------------------------------------------------------
    #  Helpers
    # -------------------------------------------------------------------------
    def _check_corr_fitted(self) -> None:
        """Raise if the correlation matrix has not been set."""
        if self.corr is None:
            raise RuntimeError(
                "GaussianCopula: call fit_copula() (or pass `corr` "
                "to __init__) before this operation."
            )

    def __repr__(self) -> str:
        if self.corr is None:
            return "GaussianCopula(unfitted)"
        return f"GaussianCopula(d={self.corr.shape[0]})"

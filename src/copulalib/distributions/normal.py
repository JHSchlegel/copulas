"""
This module implements the normal (Gaussian) marginal distribution.

Author:
    Jan Heinrich Schlegel 04.04.2026
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

from copulalib.distributions.base import Distribution


# =========================================================================== #
#                         Normal Distribution Class                           #
# =========================================================================== #
class NormalDistribution(Distribution):
    """Normal (Gaussian) distribution with parameters mu and sigma.

    Parameters are estimated via maximum-likelihood (equivalent to
    the sample mean and unbiased sample standard deviation) during
    ``fit``.  Either or both parameters may be pre-specified in
    ``__init__``; pre-specified values are kept as-is and never
    re-estimated.

    Parameters
    ----------
    mu : float, optional
        Mean.  If ``None`` (default), estimated from data during
        ``fit``.
    sigma : float, optional
        Standard deviation.  If ``None`` (default), estimated from
        data during ``fit``.
    """

    def __init__(
        self,
        mu: float | None = None,
        sigma: float | None = None,
    ) -> None:
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    # -------------------------------------------------------------------------
    #  Fitting
    # -------------------------------------------------------------------------
    def _fit(
        self, data: ArrayLike, **kwargs: Any
    ) -> NormalDistribution:
        """Estimate mu and/or sigma from data.

        Only parameters that are currently ``None`` are estimated;
        pre-specified values are left unchanged.

        Parameters
        ----------
        data : array_like
            1-D univariate sample.

        Returns
        -------
        self
        """
        arr = np.asarray(data, dtype=np.float64).ravel()
        if arr.size == 0:
            raise ValueError("data must be non-empty.")
        if self.mu is None:
            self.mu = float(np.mean(arr))
        if self.sigma is None:
            self.sigma = float(np.std(arr, ddof=1))
        return self

    # -------------------------------------------------------------------------
    #  CDF / PPF / PDF
    # -------------------------------------------------------------------------
    def cdf(self, x: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Normal CDF evaluated at x.

        Parameters
        ----------
        x : array_like
            Quantiles.

        Returns
        -------
        NDArray
            Probabilities in [0, 1].
        """
        self._check_fitted()
        assert self.mu is not None and self.sigma is not None
        return np.asarray(
            norm.cdf(x, loc=self.mu, scale=self.sigma),
            dtype=np.float64,
        )

    def ppf(self, q: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Normal quantile function (inverse CDF).

        Parameters
        ----------
        q : array_like
            Probabilities in [0, 1].

        Returns
        -------
        NDArray
            Corresponding quantiles.
        """
        self._check_fitted()
        assert self.mu is not None and self.sigma is not None
        return np.asarray(
            norm.ppf(q, loc=self.mu, scale=self.sigma),
            dtype=np.float64,
        )

    def pdf(self, x: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Normal probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles.

        Returns
        -------
        NDArray
            Density values (non-negative).
        """
        self._check_fitted()
        assert self.mu is not None and self.sigma is not None
        return np.asarray(
            norm.pdf(x, loc=self.mu, scale=self.sigma),
            dtype=np.float64,
        )

    def logpdf(
        self, x: ArrayLike, **kwargs: Any
    ) -> NDArray[np.float64]:
        """Log of the normal probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles.

        Returns
        -------
        NDArray
            Log-density values.
        """
        self._check_fitted()
        assert self.mu is not None and self.sigma is not None
        return np.asarray(
            norm.logpdf(x, loc=self.mu, scale=self.sigma),
            dtype=np.float64,
        )

    # -------------------------------------------------------------------------
    #  Sampling
    # -------------------------------------------------------------------------
    def sample(self, n: int, **kwargs: Any) -> NDArray[np.float64]:
        """Draw n samples from the fitted normal distribution.

        Parameters
        ----------
        n : int
            Number of samples.

        Returns
        -------
        NDArray of shape (n,)
        """
        self._check_fitted()
        assert self.mu is not None and self.sigma is not None
        rng: np.random.Generator = kwargs.get(
            "rng", np.random.default_rng()
        )
        return rng.normal(self.mu, self.sigma, size=n)

    # -------------------------------------------------------------------------
    #  Parameter interface
    # -------------------------------------------------------------------------
    @property
    def params(self) -> dict[str, float | None]:
        """Current parameter values.

        Returns
        -------
        dict
            Keys ``'mu'`` and ``'sigma'``.  Values are ``None``
            before fitting (unless pre-specified in ``__init__``).
        """
        return {"mu": self.mu, "sigma": self.sigma}

    # -------------------------------------------------------------------------
    #  Helpers
    # -------------------------------------------------------------------------
    def __repr__(self) -> str:
        mu = f"{self.mu:.4g}" if self.mu is not None else "None"
        sigma = (
            f"{self.sigma:.4g}" if self.sigma is not None else "None"
        )
        return f"NormalDistribution(mu={mu}, sigma={sigma})"

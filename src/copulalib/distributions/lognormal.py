"""
This module implements the log-normal marginal distribution.

The parameters ``mu`` and ``sigma`` are the mean and standard deviation
of log(X), i.e. the natural parameters of the underlying normal
distribution.  They are distinct from the data-space mean and variance
of X; see ``LogNormalDistribution.from_moments`` for conversion.

Author:
    Jan Heinrich Schlegel 05.04.2026
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import lognorm

from copulalib.distributions.base import Distribution


# =========================================================================== #
#                       Log-Normal Distribution Class                         #
# =========================================================================== #
class LogNormalDistribution(Distribution):
    """Log-normal distribution parameterised in log-space.

    If X ~ LogNormal(mu, sigma) then log(X) ~ Normal(mu, sigma).
    Parameters are estimated via MLE during ``fit``: the data are
    log-transformed and then the normal MLE is applied, i.e.

        mu    = mean(log X)
        sigma = std(log X, ddof=1)

    Either or both parameters may be pre-specified in ``__init__``;
    pre-specified values are kept as-is and not re-estimated.

    .. note::
        ``mu`` and ``sigma`` are **log-space** quantities and are not
        the data-space mean and standard deviation of X.  To construct
        a distribution from data-space moments use
        ``LogNormalDistribution.from_moments``.

    Parameters
    ----------
    mu : float, optional
        Mean of log(X).  If ``None`` (default), estimated from data.
    sigma : float, optional
        Standard deviation of log(X).  If ``None`` (default),
        estimated from data.
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
    #  Fitting  (MLE: log-transform then normal MLE)
    # -------------------------------------------------------------------------
    def _fit(self, data: ArrayLike, **kwargs: Any) -> LogNormalDistribution:
        """Estimate mu and/or sigma via MLE.

        Log-transforms the data and applies normal MLE.  Only
        parameters that are currently ``None`` are estimated;
        pre-specified values are left unchanged.

        Parameters
        ----------
        data : array_like
            1-D sample of strictly positive values.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If ``data`` is empty or contains non-positive values.
        """
        arr = np.asarray(data, dtype=np.float64).ravel()
        if arr.size == 0:
            raise ValueError("data must be non-empty.")
        if np.any(arr <= 0):
            raise ValueError(
                "LogNormalDistribution requires strictly positive "
                f"data; got min={arr.min():.6g}."
            )
        log_arr = np.log(arr)
        if self.mu is None:
            self.mu = float(np.mean(log_arr))
        if self.sigma is None:
            self.sigma = float(np.std(log_arr, ddof=1))
        return self

    # -------------------------------------------------------------------------
    #  CDF / PPF / PDF
    # -------------------------------------------------------------------------
    def cdf(self, x: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Log-normal CDF evaluated at x.

        Parameters
        ----------
        x : array_like
            Quantiles (must be >= 0).

        Returns
        -------
        NDArray
            Probabilities in [0, 1].
        """
        self._check_fitted()
        assert self.mu is not None and self.sigma is not None
        return np.asarray(
            lognorm.cdf(x, s=self.sigma, scale=np.exp(self.mu)),
            dtype=np.float64,
        )

    def ppf(self, q: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Log-normal quantile function (inverse CDF).

        Parameters
        ----------
        q : array_like
            Probabilities in [0, 1].

        Returns
        -------
        NDArray
            Corresponding quantiles (strictly positive).
        """
        self._check_fitted()
        assert self.mu is not None and self.sigma is not None
        return np.asarray(
            lognorm.ppf(q, s=self.sigma, scale=np.exp(self.mu)),
            dtype=np.float64,
        )

    def pdf(self, x: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Log-normal probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles (must be >= 0).

        Returns
        -------
        NDArray
            Density values (non-negative).
        """
        self._check_fitted()
        assert self.mu is not None and self.sigma is not None
        return np.asarray(
            lognorm.pdf(x, s=self.sigma, scale=np.exp(self.mu)),
            dtype=np.float64,
        )

    def logpdf(self, x: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Log of the log-normal probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles (must be >= 0).

        Returns
        -------
        NDArray
            Log-density values.
        """
        self._check_fitted()
        assert self.mu is not None and self.sigma is not None
        return np.asarray(
            lognorm.logpdf(x, s=self.sigma, scale=np.exp(self.mu)),
            dtype=np.float64,
        )

    # -------------------------------------------------------------------------
    #  Sampling
    # -------------------------------------------------------------------------
    def sample(self, n: int, **kwargs: Any) -> NDArray[np.float64]:
        """Draw n samples from the fitted log-normal distribution.

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
        rng: np.random.Generator = kwargs.get("rng", np.random.default_rng())
        return rng.lognormal(mean=self.mu, sigma=self.sigma, size=n)

    # -------------------------------------------------------------------------
    #  Parameter interface
    # -------------------------------------------------------------------------
    @property
    def params(self) -> dict[str, float | None]:
        """Current log-space parameter values.

        Returns
        -------
        dict
            Keys ``'mu'`` and ``'sigma'`` (both in log-space).
            Values are ``None`` before fitting unless pre-specified.
        """
        return {"mu": self.mu, "sigma": self.sigma}

    @classmethod
    def from_moments(
        cls,
        mean: float,
        variance: float,
    ) -> LogNormalDistribution:
        """Construct from data-space mean and variance via MOM.

        Useful for expert overrides: if you know the expected value
        and variance of X (not of log X), this converts them to the
        log-space parameters mu and sigma using the method-of-moments
        formulas:

            sigma² = log(variance / mean² + 1)
            mu     = log(mean) - sigma² / 2

        Parameters
        ----------
        mean : float
            Data-space mean E[X].  Must be strictly positive.
        variance : float
            Data-space variance Var[X].  Must be strictly positive.

        Returns
        -------
        LogNormalDistribution
            Instance with mu and sigma pre-specified (no fitting
            required).

        Raises
        ------
        ValueError
            If ``mean`` or ``variance`` are not strictly positive.
        """
        if mean <= 0:
            raise ValueError(f"mean must be strictly positive, got {mean}.")
        if variance <= 0:
            raise ValueError(
                f"variance must be strictly positive, got {variance}."
            )
        sigma_sq = float(np.log(variance / mean**2 + 1))
        mu = float(np.log(mean) - sigma_sq / 2)
        return cls(mu=mu, sigma=float(np.sqrt(sigma_sq)))

    # -------------------------------------------------------------------------
    #  Helpers
    # -------------------------------------------------------------------------
    def __repr__(self) -> str:
        mu = f"{self.mu:.4g}" if self.mu is not None else "None"
        sigma = f"{self.sigma:.4g}" if self.sigma is not None else "None"
        return f"LogNormalDistribution(mu={mu}, sigma={sigma})"

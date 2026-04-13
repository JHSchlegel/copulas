"""
This module implements the gamma marginal distribution.

The distribution is parameterised by a shape parameter ``alpha`` (k) and
a scale parameter ``beta`` (theta).  If X ~ Gamma(alpha, beta) then
E[X] = alpha * beta and Var[X] = alpha * beta^2.

Parameters are estimated via MLE (scipy) during ``fit``.

Author:
    Jan Heinrich Schlegel 13.04.2026
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import gamma as sp_gamma

from copulalib.distributions.base import Distribution


# =========================================================================== #
#                         Gamma Distribution Class                            #
# =========================================================================== #
class GammaDistribution(Distribution):
    """Gamma distribution parameterised by shape and scale.

    Parameters are estimated via MLE during ``fit``.  Either or both
    parameters may be pre-specified in ``__init__``; pre-specified
    values are kept as-is and not re-estimated.

    Parameters
    ----------
    alpha : float, optional
        Shape parameter (k > 0).  If ``None`` (default), estimated
        from data.
    beta : float, optional
        Scale parameter (theta > 0).  If ``None`` (default), estimated
        from data.
    """

    def __init__(
        self,
        alpha: float | None = None,
        beta: float | None = None,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    # -------------------------------------------------------------------------
    #  Fitting  (MLE via scipy)
    # -------------------------------------------------------------------------
    def _fit(self, data: ArrayLike, **kwargs: Any) -> GammaDistribution:
        """Estimate alpha and/or beta via MLE.

        Only parameters that are currently ``None`` are estimated;
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
                "GammaDistribution requires strictly positive "
                f"data; got min={arr.min():.6g}."
            )
        if self.alpha is None and self.beta is None:
            a, _, scale = sp_gamma.fit(arr, floc=0)
            self.alpha = float(a)
            self.beta = float(scale)
        elif self.alpha is None:
            a, _, _ = sp_gamma.fit(arr, floc=0, fscale=self.beta)
            self.alpha = float(a)
        elif self.beta is None:
            _, _, scale = sp_gamma.fit(arr, fa=self.alpha, floc=0)
            self.beta = float(scale)
        return self

    # -------------------------------------------------------------------------
    #  CDF / PPF / PDF
    # -------------------------------------------------------------------------
    def cdf(self, x: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Gamma CDF evaluated at x.

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
        assert self.alpha is not None and self.beta is not None
        return np.asarray(
            sp_gamma.cdf(x, a=self.alpha, scale=self.beta),
            dtype=np.float64,
        )

    def ppf(self, q: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Gamma quantile function (inverse CDF).

        Parameters
        ----------
        q : array_like
            Probabilities in [0, 1].

        Returns
        -------
        NDArray
            Corresponding quantiles (non-negative).
        """
        self._check_fitted()
        assert self.alpha is not None and self.beta is not None
        return np.asarray(
            sp_gamma.ppf(q, a=self.alpha, scale=self.beta),
            dtype=np.float64,
        )

    def pdf(self, x: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Gamma probability density function.

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
        assert self.alpha is not None and self.beta is not None
        return np.asarray(
            sp_gamma.pdf(x, a=self.alpha, scale=self.beta),
            dtype=np.float64,
        )

    def logpdf(self, x: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Log of the gamma probability density function.

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
        assert self.alpha is not None and self.beta is not None
        return np.asarray(
            sp_gamma.logpdf(x, a=self.alpha, scale=self.beta),
            dtype=np.float64,
        )

    # -------------------------------------------------------------------------
    #  Sampling
    # -------------------------------------------------------------------------
    def sample(self, n: int, **kwargs: Any) -> NDArray[np.float64]:
        """Draw n samples from the fitted gamma distribution.

        Parameters
        ----------
        n : int
            Number of samples.

        Returns
        -------
        NDArray of shape (n,)
        """
        self._check_fitted()
        assert self.alpha is not None and self.beta is not None
        rng: np.random.Generator = kwargs.get("rng", np.random.default_rng())
        return rng.gamma(shape=self.alpha, scale=self.beta, size=n)

    # -------------------------------------------------------------------------
    #  Parameter interface
    # -------------------------------------------------------------------------
    @property
    def params(self) -> dict[str, float | None]:
        """Current parameter values.

        Returns
        -------
        dict
            Keys ``'alpha'`` and ``'beta'``.  Values are ``None``
            before fitting unless pre-specified.
        """
        return {"alpha": self.alpha, "beta": self.beta}

    # -------------------------------------------------------------------------
    #  Helpers
    # -------------------------------------------------------------------------
    def __repr__(self) -> str:
        alpha = f"{self.alpha:.4g}" if self.alpha is not None else "None"
        beta = f"{self.beta:.4g}" if self.beta is not None else "None"
        return f"GammaDistribution(alpha={alpha}, beta={beta})"

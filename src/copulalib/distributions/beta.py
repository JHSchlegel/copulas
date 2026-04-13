"""
This module implements the beta marginal distribution.

The distribution is parameterised by two shape parameters ``alpha``
(a) and ``beta`` (b).  The support is the unit interval (0, 1).
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
from scipy.stats import beta as sp_beta

from copulalib.distributions.base import Distribution


# =========================================================================== #
#                          Beta Distribution Class                            #
# =========================================================================== #
class BetaDistribution(Distribution):
    """Beta distribution parameterised by two shape parameters.

    Parameters are estimated via MLE during ``fit``.  Either or both
    parameters may be pre-specified in ``__init__``; pre-specified
    values are kept as-is and not re-estimated.

    Parameters
    ----------
    alpha : float, optional
        First shape parameter (a > 0).  If ``None`` (default),
        estimated from data.
    beta : float, optional
        Second shape parameter (b > 0).  If ``None`` (default),
        estimated from data.
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
    def _fit(self, data: ArrayLike, **kwargs: Any) -> BetaDistribution:
        """Estimate alpha and/or beta via MLE.

        Only parameters that are currently ``None`` are estimated;
        pre-specified values are left unchanged.

        Parameters
        ----------
        data : array_like
            1-D sample of values in (0, 1).

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If ``data`` is empty or contains values outside (0, 1).
        """
        arr = np.asarray(data, dtype=np.float64).ravel()
        if arr.size == 0:
            raise ValueError("data must be non-empty.")
        if np.any(arr <= 0) or np.any(arr >= 1):
            raise ValueError(
                "BetaDistribution requires data in (0, 1); "
                f"got min={arr.min():.6g}, max={arr.max():.6g}."
            )
        if self.alpha is None and self.beta is None:
            a, b, _, _ = sp_beta.fit(arr, floc=0, fscale=1)
            self.alpha = float(a)
            self.beta = float(b)
        elif self.alpha is None:
            a, _, _, _ = sp_beta.fit(arr, fb=self.beta, floc=0, fscale=1)
            self.alpha = float(a)
        elif self.beta is None:
            _, b, _, _ = sp_beta.fit(arr, fa=self.alpha, floc=0, fscale=1)
            self.beta = float(b)
        return self

    # -------------------------------------------------------------------------
    #  CDF / PPF / PDF
    # -------------------------------------------------------------------------
    def cdf(self, x: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Beta CDF evaluated at x.

        Parameters
        ----------
        x : array_like
            Quantiles in [0, 1].

        Returns
        -------
        NDArray
            Probabilities in [0, 1].
        """
        self._check_fitted()
        assert self.alpha is not None and self.beta is not None
        return np.asarray(
            sp_beta.cdf(x, a=self.alpha, b=self.beta),
            dtype=np.float64,
        )

    def ppf(self, q: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Beta quantile function (inverse CDF).

        Parameters
        ----------
        q : array_like
            Probabilities in [0, 1].

        Returns
        -------
        NDArray
            Corresponding quantiles in [0, 1].
        """
        self._check_fitted()
        assert self.alpha is not None and self.beta is not None
        return np.asarray(
            sp_beta.ppf(q, a=self.alpha, b=self.beta),
            dtype=np.float64,
        )

    def pdf(self, x: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Beta probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles in [0, 1].

        Returns
        -------
        NDArray
            Density values (non-negative).
        """
        self._check_fitted()
        assert self.alpha is not None and self.beta is not None
        return np.asarray(
            sp_beta.pdf(x, a=self.alpha, b=self.beta),
            dtype=np.float64,
        )

    def logpdf(self, x: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Log of the beta probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles in [0, 1].

        Returns
        -------
        NDArray
            Log-density values.
        """
        self._check_fitted()
        assert self.alpha is not None and self.beta is not None
        return np.asarray(
            sp_beta.logpdf(x, a=self.alpha, b=self.beta),
            dtype=np.float64,
        )

    # -------------------------------------------------------------------------
    #  Sampling
    # -------------------------------------------------------------------------
    def sample(self, n: int, **kwargs: Any) -> NDArray[np.float64]:
        """Draw n samples from the fitted beta distribution.

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
        return rng.beta(a=self.alpha, b=self.beta, size=n)

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
        return f"BetaDistribution(alpha={alpha}, beta={beta})"

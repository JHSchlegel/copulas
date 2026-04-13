"""
This module implements the Pareto (Type I) marginal distribution.

The distribution is parameterised by a shape parameter ``alpha``
(tail index) and a scale parameter ``xm`` (minimum value).
If X ~ Pareto(alpha, xm) then P(X > x) = (xm / x)^alpha for x >= xm.

Parameters are estimated via MLE during ``fit``.

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
from scipy.stats import pareto as sp_pareto

from copulalib.distributions.base import Distribution


# =========================================================================== #
#                         Pareto Distribution Class                           #
# =========================================================================== #
class ParetoDistribution(Distribution):
    """Pareto (Type I) distribution parameterised by shape and scale.

    The MLE for the Pareto distribution is:

        xm    = min(data)
        alpha = n / sum(log(data / xm))

    Either or both parameters may be pre-specified in ``__init__``;
    pre-specified values are kept as-is and not re-estimated.

    Parameters
    ----------
    alpha : float, optional
        Shape / tail index (alpha > 0).  If ``None`` (default),
        estimated from data.
    xm : float, optional
        Scale / minimum value (xm > 0).  If ``None`` (default),
        estimated from data.
    """

    def __init__(
        self,
        alpha: float | None = None,
        xm: float | None = None,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.xm = xm

    # -------------------------------------------------------------------------
    #  Fitting  (MLE)
    # -------------------------------------------------------------------------
    def _fit(self, data: ArrayLike, **kwargs: Any) -> ParetoDistribution:
        """Estimate alpha and/or xm via MLE.

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
                "ParetoDistribution requires strictly positive "
                f"data; got min={arr.min():.6g}."
            )
        # closed-form MLE: xm = min(data), alpha = n / sum(log(x/xm))
        xm = float(np.min(arr)) if self.xm is None else self.xm
        if self.xm is None:
            self.xm = xm
        if self.alpha is None:
            n = arr.size
            self.alpha = float(n / np.sum(np.log(arr / xm)))
        return self

    # -------------------------------------------------------------------------
    #  CDF / PPF / PDF
    # -------------------------------------------------------------------------
    def cdf(self, x: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Pareto CDF evaluated at x.

        Parameters
        ----------
        x : array_like
            Quantiles (must be >= xm).

        Returns
        -------
        NDArray
            Probabilities in [0, 1].
        """
        self._check_fitted()
        assert self.alpha is not None and self.xm is not None
        return np.asarray(
            sp_pareto.cdf(x, b=self.alpha, scale=self.xm),
            dtype=np.float64,
        )

    def ppf(self, q: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Pareto quantile function (inverse CDF).

        Parameters
        ----------
        q : array_like
            Probabilities in [0, 1].

        Returns
        -------
        NDArray
            Corresponding quantiles (>= xm).
        """
        self._check_fitted()
        assert self.alpha is not None and self.xm is not None
        return np.asarray(
            sp_pareto.ppf(q, b=self.alpha, scale=self.xm),
            dtype=np.float64,
        )

    def pdf(self, x: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Pareto probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles (must be >= xm).

        Returns
        -------
        NDArray
            Density values (non-negative).
        """
        self._check_fitted()
        assert self.alpha is not None and self.xm is not None
        return np.asarray(
            sp_pareto.pdf(x, b=self.alpha, scale=self.xm),
            dtype=np.float64,
        )

    def logpdf(self, x: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Log of the Pareto probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles (must be >= xm).

        Returns
        -------
        NDArray
            Log-density values.
        """
        self._check_fitted()
        assert self.alpha is not None and self.xm is not None
        return np.asarray(
            sp_pareto.logpdf(x, b=self.alpha, scale=self.xm),
            dtype=np.float64,
        )

    # -------------------------------------------------------------------------
    #  Sampling
    # -------------------------------------------------------------------------
    def sample(self, n: int, **kwargs: Any) -> NDArray[np.float64]:
        """Draw n samples from the fitted Pareto distribution.

        Parameters
        ----------
        n : int
            Number of samples.

        Returns
        -------
        NDArray of shape (n,)
        """
        self._check_fitted()
        assert self.alpha is not None and self.xm is not None
        rng: np.random.Generator = kwargs.get("rng", np.random.default_rng())
        return (rng.pareto(a=self.alpha, size=n) + 1) * self.xm

    # -------------------------------------------------------------------------
    #  Parameter interface
    # -------------------------------------------------------------------------
    @property
    def params(self) -> dict[str, float | None]:
        """Current parameter values.

        Returns
        -------
        dict
            Keys ``'alpha'`` and ``'xm'``.  Values are ``None``
            before fitting unless pre-specified.
        """
        return {"alpha": self.alpha, "xm": self.xm}

    # -------------------------------------------------------------------------
    #  Helpers
    # -------------------------------------------------------------------------
    def __repr__(self) -> str:
        alpha = f"{self.alpha:.4g}" if self.alpha is not None else "None"
        xm = f"{self.xm:.4g}" if self.xm is not None else "None"
        return f"ParetoDistribution(alpha={alpha}, xm={xm})"

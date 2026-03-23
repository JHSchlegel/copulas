"""
This module implements the empirical marginal distribution, which has no
closed-form density.

Author:
    Jan Heinrich Schlegel 23.03.2026
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from copulalib.distributions.base import Distribution


# =========================================================================== #
#                        Empirical Distribution Class                         #
# =========================================================================== #
class EmpiricalDistribution(Distribution):
    """Non-parametric distribution defined by a sample.

    The CDF is the classical ECDF (right-continuous step function with
    jumps of 1/n at each order statistic).  The quantile function (ppf)
    is the left-continuous generalised inverse:

        Q(p) = x_{ceil(n*p)}   (1-based order statistic)

    Sampling uses the standard trick: draw U ~ Uniform(0,1) and return
    Q(U), which is equivalent to sampling with replacement from the
    original data.

    Attributes
    ----------
    _data : NDArray
        Sorted order statistics after fitting.
    _n : int
        Sample size.
    """

    def __init__(self) -> None:
        super().__init__()
        self._data: NDArray[np.float64] | None = None
        self._n: int = 0

    # ------------------------------------------------------------------ #
    #  Fitting                                                             #
    # ------------------------------------------------------------------ #
    def _fit(self, data: ArrayLike, **kwargs: Any) -> EmpiricalDistribution:
        """Sort and store the sample as order statistics.

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
        self._data = np.sort(arr)
        self._n = len(self._data)
        return self

    # ------------------------------------------------------------------ #
    #  CDF  —  F(x) = #{x_i <= x} / n                                     #
    # ------------------------------------------------------------------ #
    def cdf(self, x: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Empirical CDF evaluated at x.

        Uses binary search (right-side) so it is right-continuous, i.e.
        F(x) includes the jump at x itself.

        Parameters
        ----------
        x : array_like
            Quantiles at which to evaluate F.

        Returns
        -------
        NDArray
            Probabilities in [0, 1].
        """
        self._check_fitted()
        x = np.asarray(x, dtype=np.float64)
        # searchsorted(..., side='right') counts how many values are <= x
        return np.searchsorted(self._data, x, side="right") / self._n

    # ------------------------------------------------------------------ #
    #  PPF  —  Q(p) = x_{ceil(n*p)}                                        #
    # ------------------------------------------------------------------ #
    def ppf(self, q: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Empirical quantile function (inverse CDF).

        Implements the exact empirical quantile Q(p) = x_{ceil(n*p)},
        i.e. numpy's ``method='inverted_cdf'``, without interpolation.

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
        q = np.asarray(q, dtype=np.float64)
        if np.any((q < 0) | (q > 1)):
            raise ValueError("q must be in [0, 1].")
        # clip so that q=0 maps to x_(1) and q=1 maps to x_(n)
        idx = np.clip(np.ceil(self._n * q).astype(int) - 1, 0, self._n - 1)
        return self._data[idx]

    # -------------------------------------------------------------------------
    #  Sampling  —  sample with replacement
    # -------------------------------------------------------------------------
    def sample(self, n: int, **kwargs: Any) -> NDArray[np.float64]:
        """Draw n samples with replacement (bootstrap samples).

        Parameters
        ----------
        n : int
            Number of samples to draw.

        Returns
        -------
        NDArray of shape (n,)
        """
        self._check_fitted()
        rng: np.random.Generator = kwargs.get("rng", np.random.default_rng())
        return rng.choice(self._data, size=n, replace=True)

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #
    def _check_fitted(self) -> None:
        if self._data is None:
            raise RuntimeError("Call fit() before using this distribution.")

    def __repr__(self) -> str:
        if self._data is None:
            return "EmpiricalDistribution(unfitted)"
        return (
            f"EmpiricalDistribution(n={self._n}, "
            f"min={self._data[0]:.4g}, max={self._data[-1]:.4g})"
        )

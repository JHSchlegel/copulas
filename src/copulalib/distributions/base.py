"""
This module implements an asbstract Base class for univariate marginal
distributions.

Author:
    Jan Heinrich Schlegel 23.03.2026
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


# =========================================================================== #
#                            Base Distribution Class                          #
# =========================================================================== #
class Distribution(ABC):
    """Base class for univariate marginal distributions.

    Subclasses must implement ``fit``, ``cdf``, ``ppf``, and ``sample``.
    ``pdf`` and ``logpdf`` are optional — not every distribution has a
    closed-form density (e.g. truncated log-normal, empirical).
    """

    def __init__(self) -> None:
        self._fitted: bool = False

    def fit(self, data: ArrayLike, **kwargs: Any) -> Distribution:
        """Estimate parameters from data.

        Calls the subclass implementation via ``_fit`` and then sets
        ``self._fitted = True``.

        Parameters
        ----------
        data : array_like
            Univariate sample.

        Returns
        -------
        self
        """
        result = self._fit(data, **kwargs)
        self._fitted = True
        return result

    @abstractmethod
    def _fit(self, data: ArrayLike, **kwargs: Any) -> Distribution:
        """Estimate parameters from data.

        Parameters
        ----------
        data : array_like
            Univariate sample.

        Returns
        -------
        self
        """

    @abstractmethod
    def cdf(self, x: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Cumulative distribution function (probability integral
        transform).

        Parameters
        ----------
        x : array_like
            Quantiles.

        Returns
        -------
        NDArray
            Probabilities in [0, 1].
        """

    @abstractmethod
    def ppf(self, q: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Percent-point function (inverse CDF / quantile function).

        Parameters
        ----------
        q : array_like
            Probabilities in [0, 1].

        Returns
        -------
        NDArray
            Quantiles.
        """

    @abstractmethod
    def sample(self, n: int, **kwargs: Any) -> NDArray[np.float64]:
        """Draw random samples.

        Parameters
        ----------
        n : int
            Number of samples.

        Returns
        -------
        NDArray
            Array of shape ``(n,)``.
        """

    def pdf(self, x: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Probability density function.

        Optional — raises ``NotImplementedError`` by default.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement pdf."
        )

    def logpdf(self, x: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Log probability density function.

        Falls back to ``log(pdf(x))`` if not overridden.
        """
        return np.log(self.pdf(x))

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _check_fitted(self) -> None:
        """Raise if ``fit`` has not been called yet."""
        if not self._fitted:
            raise RuntimeError(
                f"{type(self).__name__}: call fit() before using "
                f"this distribution."
            )

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return f"{type(self).__name__}({status})"

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

    @abstractmethod
    def fit(self, data: ArrayLike) -> Distribution:
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
    def cdf(self, x: ArrayLike) -> NDArray[np.float64]:
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
    def ppf(self, q: ArrayLike) -> NDArray[np.float64]:
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
    def sample(self, n: int) -> NDArray[np.float64]:
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

    def pdf(self, x: ArrayLike) -> NDArray[np.float64]:
        """Probability density function.

        Optional — raises ``NotImplementedError`` by default.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement pdf."
        )

    def logpdf(self, x: ArrayLike) -> NDArray[np.float64]:
        """Log probability density function.

        Falls back to ``log(pdf(x))`` if not overridden.
        """
        return np.log(self.pdf(x))

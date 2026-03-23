"""
This module implements an abstract Base class for copulas and a dataclass
for marginal specification.

Author:
    Jan Heinrich Schlegel 23.03.2026
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from copulalib.distributions.base import Distribution


# =========================================================================== #
#                          Marginal Specification                             #
# =========================================================================== #
@dataclass
class Marginal:
    """Specification of a marginal distribution for one variable.

    Parameters
    ----------
    name : str
        Variable name.
    distribution : Distribution
        Univariate distribution instance.
    """

    name: str
    distribution: Distribution


# =========================================================================== #
#                            Base Copula Class                                #
# =========================================================================== #
class Copula(ABC):
    """Base class for copulas.

    Subclasses must implement ``fit``, ``sample``, and ``cdf``.
    ``pdf`` and ``logpdf`` are optional.
    """

    @abstractmethod
    def fit(
        self,
        data: ArrayLike,
        marginals: list[Marginal],
    ) -> Copula:
        """Fit the copula to data.

        Parameters
        ----------
        data : array_like
            Multivariate sample of shape ``(n, d)``.
        marginals : list[Marginal]
            Marginal specifications, one per dimension.

        Returns
        -------
        self
        """

    @abstractmethod
    def sample(self, n: int) -> NDArray[np.float64]:
        """Draw samples in uniform space.

        Parameters
        ----------
        n : int
            Number of samples.

        Returns
        -------
        NDArray
            Array of shape ``(n, d)`` with values in [0, 1].
        """

    @abstractmethod
    def cdf(self, u: ArrayLike) -> NDArray[np.float64]:
        """Copula CDF.

        Parameters
        ----------
        u : array_like
            Points in the unit hypercube, shape ``(n, d)``.

        Returns
        -------
        NDArray
            Copula CDF values.
        """

    def pdf(self, u: ArrayLike) -> NDArray[np.float64]:
        """Copula density.

        Optional — raises ``NotImplementedError`` by default.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement pdf."
        )

    def logpdf(self, u: ArrayLike) -> NDArray[np.float64]:
        """Log copula density.

        Falls back to ``log(pdf(u))`` if not overridden.
        """
        return np.log(self.pdf(u))

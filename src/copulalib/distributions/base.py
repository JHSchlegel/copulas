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
    # Parameter interface
    # -------------------------------------------------------------------------
    @property
    @abstractmethod
    def params(self) -> dict[str, float | None]:
        """Current parameter values as a name-to-value mapping.

        Returns
        -------
        dict
            Parameter names mapped to their current values.
            Returns an empty dict for non-parametric distributions.
            Values may be ``None`` before ``fit`` is called.
        """

    def set_params(self, params: dict[str, float]) -> None:
        """Override one or more parameters by name.

        Parameters
        ----------
        params : dict
            Mapping from parameter name to new value.  Keys must be
            a subset of those returned by ``self.params``.

        Raises
        ------
        ValueError
            If any key is not a recognised parameter of this
            distribution.
        """
        allowed = set(self.params)
        unknown = set(params) - allowed
        if unknown:
            raise ValueError(
                f"{type(self).__name__} has no parameters "
                f"{sorted(unknown)}."
            )
        for key, val in params.items():
            setattr(self, key, val)

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

"""
This module implements the independence copula.

The independence copula corresponds to mutually independent
marginals: ``C(u_1, ..., u_d) = u_1 * ... * u_d``.  It has no
parameters and ``fit_copula`` is a no-op.

Author:
- Jan Schlegel
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from copulalib.copulas.base import Copula


# =========================================================================== #
#                        Independence Copula Class                            #
# =========================================================================== #
class IndependenceCopula(Copula):
    """Product (independence) copula.

    Has no parameters: ``fit_copula`` is a no-op.  The dimension is
    inferred from the number of fitted marginals.
    """

    # -------------------------------------------------------------------------
    #  Fitting
    # -------------------------------------------------------------------------
    def _fit_copula(
        self, u: NDArray[np.float64], **kwargs: Any
    ) -> None:
        """No parameters to estimate — does nothing."""
        return

    # -------------------------------------------------------------------------
    #  Sampling
    # -------------------------------------------------------------------------
    def sample(self, n: int, **kwargs: Any) -> NDArray[np.float64]:
        """Draw ``n`` independent uniform samples.

        Parameters
        ----------
        n : int
            Number of samples.
        rng : np.random.Generator, optional
            Random number generator (passed via ``**kwargs``).

        Returns
        -------
        NDArray of shape (n, d)
            Samples in the open unit hypercube.
        """
        self._check_marginals_fitted()
        rng: np.random.Generator = kwargs.get(
            "rng", np.random.default_rng()
        )
        d = len(self._marginals)
        return rng.uniform(0.0, 1.0, size=(n, d))

    # -------------------------------------------------------------------------
    #  CDF / PDF / log-PDF
    # -------------------------------------------------------------------------
    def cdf(self, u: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Independence copula CDF: product of components."""
        u_arr = np.atleast_2d(np.asarray(u, dtype=np.float64))
        return np.asarray(np.prod(u_arr, axis=1), dtype=np.float64)

    def pdf(self, u: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Independence copula density: identically one."""
        u_arr = np.atleast_2d(np.asarray(u, dtype=np.float64))
        return np.ones(u_arr.shape[0], dtype=np.float64)

    def logpdf(
        self, u: ArrayLike, **kwargs: Any
    ) -> NDArray[np.float64]:
        """Log independence copula density: identically zero."""
        u_arr = np.atleast_2d(np.asarray(u, dtype=np.float64))
        return np.zeros(u_arr.shape[0], dtype=np.float64)

    def __repr__(self) -> str:
        return "IndependenceCopula()"

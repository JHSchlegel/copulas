"""
This module implements the comonotonic copula (Fréchet-Hoeffding
upper bound).

The comonotonic copula represents perfect positive dependence: all
components are deterministic monotone transformations of a single
underlying uniform random variable.  ``C(u_1, ..., u_d) = min(u_i)``.

Because the distribution is concentrated on the diagonal, no density
exists with respect to Lebesgue measure on the unit cube.

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
#                        Comonotonic Copula Class                             #
# =========================================================================== #
class ComonotonicCopula(Copula):
    """Comonotonic (perfect positive dependence) copula.

    Has no parameters.  ``pdf`` is undefined and raises
    ``NotImplementedError`` because the distribution is singular
    with respect to Lebesgue measure.
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
        """Draw ``n`` perfectly comonotonic samples.

        All ``d`` components share the same uniform draw.

        Parameters
        ----------
        n : int
            Number of samples.
        rng : np.random.Generator, optional
            Random number generator (passed via ``**kwargs``).

        Returns
        -------
        NDArray of shape (n, d)
            Samples in the open unit hypercube; rows are constant
            across columns.
        """
        self._check_marginals_fitted()
        rng: np.random.Generator = kwargs.get(
            "rng", np.random.default_rng()
        )
        d = len(self._marginals)
        u = rng.uniform(0.0, 1.0, size=n)
        return np.tile(u[:, None], (1, d))

    # -------------------------------------------------------------------------
    #  CDF
    # -------------------------------------------------------------------------
    def cdf(self, u: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Comonotonic CDF: minimum of components."""
        u_arr = np.atleast_2d(np.asarray(u, dtype=np.float64))
        return np.asarray(np.min(u_arr, axis=1), dtype=np.float64)

    def __repr__(self) -> str:
        return "ComonotonicCopula()"

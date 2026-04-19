"""
This module implements the empirical (non-parametric) copula.

Given a sample of pseudo-observations ``u_1, ..., u_n`` in the unit
hypercube, the empirical copula is the multivariate ECDF:

    C_n(u) = (1/n) * sum_i  1{u_{i,1} <= u_1, ..., u_{i,d} <= u_d}.

Sampling is bootstrap from the stored pseudo-observations: this
preserves the rank dependence structure in expectation but is
discrete with no density.

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
#                         Empirical Copula Class                              #
# =========================================================================== #
class EmpiricalCopula(Copula):
    """Empirical multivariate copula based on stored pseudo-obs.

    Attributes
    ----------
    pseudo_obs : NDArray of shape (n, d) or None
        Pseudo-observations (uniform pseudo-data) stored at fit time.
    """

    def __init__(self) -> None:
        super().__init__()
        self.pseudo_obs: NDArray[np.float64] | None = None

    # -------------------------------------------------------------------------
    #  Fitting
    # -------------------------------------------------------------------------
    def _fit_copula(
        self, u: NDArray[np.float64], **kwargs: Any
    ) -> None:
        """Store the pseudo-observations.

        Parameters
        ----------
        u : NDArray of shape (n, d)
            Uniform pseudo-observations.
        """
        self.pseudo_obs = np.asarray(u, dtype=np.float64).copy()

    # -------------------------------------------------------------------------
    #  Sampling  (bootstrap from stored pseudo-observations)
    # -------------------------------------------------------------------------
    def sample(self, n: int, **kwargs: Any) -> NDArray[np.float64]:
        """Draw ``n`` rows by sampling indices with replacement.

        Parameters
        ----------
        n : int
            Number of samples.
        rng : np.random.Generator, optional
            Random number generator (passed via ``**kwargs``).

        Returns
        -------
        NDArray of shape (n, d)
            Bootstrap sample of pseudo-observations.
        """
        self._check_obs_fitted()
        assert self.pseudo_obs is not None
        rng: np.random.Generator = kwargs.get(
            "rng", np.random.default_rng()
        )
        idx = rng.integers(0, self.pseudo_obs.shape[0], size=n)
        return np.asarray(self.pseudo_obs[idx], dtype=np.float64)

    # -------------------------------------------------------------------------
    #  CDF
    # -------------------------------------------------------------------------
    def cdf(self, u: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Empirical copula CDF.

        Parameters
        ----------
        u : array_like of shape (k, d) or (d,)
            Query points in the unit hypercube.

        Returns
        -------
        NDArray of shape (k,)
            Empirical CDF values in [0, 1].
        """
        self._check_obs_fitted()
        assert self.pseudo_obs is not None
        u_arr = np.atleast_2d(np.asarray(u, dtype=np.float64))
        # comp[k, i, j] = 1 if obs_{i,j} <= u_{k,j}
        comp = self.pseudo_obs[None, :, :] <= u_arr[:, None, :]
        return np.asarray(
            np.mean(np.all(comp, axis=2), axis=1), dtype=np.float64
        )

    # -------------------------------------------------------------------------
    #  Helpers
    # -------------------------------------------------------------------------
    def _check_obs_fitted(self) -> None:
        if self.pseudo_obs is None:
            raise RuntimeError(
                "EmpiricalCopula: call fit_copula() before this "
                "operation."
            )

    def __repr__(self) -> str:
        if self.pseudo_obs is None:
            return "EmpiricalCopula(unfitted)"
        return (
            f"EmpiricalCopula(n={self.pseudo_obs.shape[0]}, "
            f"d={self.pseudo_obs.shape[1]})"
        )

"""
This module implements the D-vine copula with Gaussian pair-copula
constructions (PCC).

A D-vine factorises a d-dimensional copula density as a product of
``d(d-1)/2`` bivariate (pair-) copulas, organised on ``d-1`` linear
trees.  At tree ``k`` (1-based) edge ``i`` corresponds to the
bivariate copula of ``(F(U_i | U_{i+1},...,U_{i+k-1}),
F(U_{i+k} | U_{i+1},...,U_{i+k-1}))``.

For Gaussian pair-copulas the parameter at edge ``(k, i)`` is the
partial correlation ``ρ_{i, i+k | i+1, ..., i+k-1}``.  These partial
correlations uniquely determine the full ``d × d`` correlation
matrix via the standard recursion, so the joint distribution is
multivariate-Gaussian on the latent normal scores.  We reconstruct
that correlation matrix at fit time and delegate sampling and
density evaluation to ``GaussianCopula``.

Only the Gaussian pair-copula family is implemented; the structure
extends straightforwardly to other families (Clayton, Frank, t)
once the corresponding h-functions are added.

References:
- Aas, K., Czado, C., Frigessi, A., and Bakken, H. (2009).
  Pair-copula constructions of multiple dependence.  *Insurance:
  Mathematics and Economics*, 44(2), 182-198.
- Joe, H. (2006). Generating random correlation matrices based on
  partial correlations.  *Journal of Multivariate Analysis*,
  97(10), 2177-2189.

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
from scipy.stats import norm

from copulalib.copulas.base import Copula
from copulalib.copulas.gaussian import GaussianCopula


# =========================================================================== #
#                          D-Vine Copula Class                                #
# =========================================================================== #
class DVineCopula(Copula):
    """D-vine copula with Gaussian pair-copulas.

    Parameters
    ----------
    family : str, default "gaussian"
        Pair-copula family.  Currently only ``"gaussian"`` is
        supported.

    Attributes
    ----------
    family : str
        Pair-copula family in use.
    thetas : list[list[float]] or None
        Fitted pair-copula parameters.  ``thetas[k][i]`` is the
        parameter (partial correlation) of the edge at tree
        ``k+1``, position ``i``.
    corr : NDArray of shape (d, d) or None
        Implied full correlation matrix reconstructed from
        ``thetas``.
    """

    def __init__(self, family: str = "gaussian") -> None:
        super().__init__()
        if family != "gaussian":
            raise NotImplementedError(
                f"DVineCopula only supports the Gaussian pair "
                f"family; got '{family}'."
            )
        self.family = family
        self.thetas: list[list[float]] | None = None
        self.corr: NDArray[np.float64] | None = None
        self._gaussian: GaussianCopula | None = None

    # -------------------------------------------------------------------------
    #  Fitting
    # -------------------------------------------------------------------------
    def _fit_copula(
        self,
        u: NDArray[np.float64],
        **kwargs: Any,
    ) -> None:
        """Sequentially fit pair-copulas and reconstruct the full
        correlation matrix.

        Parameters
        ----------
        u : NDArray of shape (n, d)
            Uniform pseudo-observations.
        """
        if u.ndim != 2 or u.shape[1] < 2:
            raise ValueError(
                f"DVineCopula requires d >= 2; got shape {u.shape}."
            )
        d = u.shape[1]
        self.thetas = self._fit_pair_copulas(u)
        self.corr = self._reconstruct_correlation(self.thetas, d)
        self._gaussian = GaussianCopula(corr=self.corr)

    # -------------------------------------------------------------------------
    #  Sampling and evaluation (delegated to internal GaussianCopula)
    # -------------------------------------------------------------------------
    def sample(self, n: int, **kwargs: Any) -> NDArray[np.float64]:
        """Draw ``n`` uniform samples via the induced Gaussian copula."""
        self._check_fitted()
        assert self._gaussian is not None
        return self._gaussian.sample(n, **kwargs)

    def cdf(self, u: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Copula CDF (Gaussian on the implied correlation matrix)."""
        self._check_fitted()
        assert self._gaussian is not None
        return self._gaussian.cdf(u, **kwargs)

    def pdf(self, u: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Copula density."""
        self._check_fitted()
        assert self._gaussian is not None
        return self._gaussian.pdf(u, **kwargs)

    def logpdf(
        self, u: ArrayLike, **kwargs: Any
    ) -> NDArray[np.float64]:
        """Log copula density."""
        self._check_fitted()
        assert self._gaussian is not None
        return self._gaussian.logpdf(u, **kwargs)

    # -------------------------------------------------------------------------
    #  Internal helpers
    # -------------------------------------------------------------------------
    def _check_fitted(self) -> None:
        if self._gaussian is None:
            raise RuntimeError(
                "DVineCopula: call fit_copula() before this "
                "operation."
            )

    @staticmethod
    def _fit_gaussian_pair(
        u: NDArray[np.float64],
        v: NDArray[np.float64],
    ) -> float:
        """Fit a bivariate Gaussian pair-copula by sample correlation
        of normal scores."""
        z1 = norm.ppf(np.clip(u, 1e-12, 1.0 - 1e-12))
        z2 = norm.ppf(np.clip(v, 1e-12, 1.0 - 1e-12))
        rho = float(np.corrcoef(z1, z2)[0, 1])
        return float(np.clip(rho, -0.999, 0.999))

    @staticmethod
    def _h_gaussian(
        u: NDArray[np.float64],
        v: NDArray[np.float64],
        rho: float,
    ) -> NDArray[np.float64]:
        """Gaussian pair-copula h-function ``h(u | v; ρ)``."""
        z1 = norm.ppf(np.clip(u, 1e-12, 1.0 - 1e-12))
        z2 = norm.ppf(np.clip(v, 1e-12, 1.0 - 1e-12))
        return np.asarray(
            norm.cdf((z1 - rho * z2) / np.sqrt(1.0 - rho * rho)),
            dtype=np.float64,
        )

    def _fit_pair_copulas(
        self, u: NDArray[np.float64]
    ) -> list[list[float]]:
        """Sequential pair-copula fitting (Aas et al. 2009)."""
        d = u.shape[1]
        left = [u[:, i].copy() for i in range(d - 1)]
        right = [u[:, i + 1].copy() for i in range(d - 1)]
        thetas: list[list[float]] = []

        for k in range(d - 1):
            n_edges = d - 1 - k
            thetas_k = [
                self._fit_gaussian_pair(left[i], right[i])
                for i in range(n_edges)
            ]
            thetas.append(thetas_k)

            # -----------------------------------------------------------------
            # Propagate pseudo-observations to the next tree
            # -----------------------------------------------------------------
            if k < d - 2:
                new_left = [
                    self._h_gaussian(left[i], right[i], thetas_k[i])
                    for i in range(n_edges - 1)
                ]
                new_right = [
                    self._h_gaussian(
                        right[i + 1], left[i + 1], thetas_k[i + 1]
                    )
                    for i in range(n_edges - 1)
                ]
                left, right = new_left, new_right
        return thetas

    @staticmethod
    def _reconstruct_correlation(
        thetas: list[list[float]],
        d: int,
    ) -> NDArray[np.float64]:
        """Recover the full correlation matrix from partial
        correlations along the D-vine."""
        r = np.eye(d, dtype=np.float64)
        for k in range(d - 1):
            for i in range(d - 1 - k):
                j = i + k + 1
                if k == 0:
                    r[i, j] = r[j, i] = thetas[0][i]
                    continue

                # -----------------------------------------------------------------
                # Partial-correlation recursion conditioning on {i+1, ..., j-1}
                # -----------------------------------------------------------------
                cond = list(range(i + 1, j))
                r_ss = r[np.ix_(cond, cond)]
                r_ss_inv = np.linalg.inv(r_ss)
                r_is = r[i, cond]
                r_js = r[j, cond]
                resid_i = 1.0 - r_is @ r_ss_inv @ r_is
                resid_j = 1.0 - r_js @ r_ss_inv @ r_js
                rho_partial = thetas[k][i]
                r[i, j] = r[j, i] = (
                    rho_partial * np.sqrt(resid_i * resid_j)
                    + r_is @ r_ss_inv @ r_js
                )
        return r

    def __repr__(self) -> str:
        if self.thetas is None:
            return "DVineCopula(unfitted)"
        return (
            f"DVineCopula(d={len(self.thetas) + 1}, "
            f"family='{self.family}')"
        )

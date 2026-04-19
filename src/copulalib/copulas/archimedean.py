"""
This module implements bivariate Archimedean copulas: Clayton,
Gumbel and Frank.

Each Archimedean copula has the form

    C(u_1, ..., u_d) = ψ⁻¹( ψ(u_1) + ... + ψ(u_d) )

for a strictly decreasing convex generator ψ.  The implementations
here are bivariate (``d = 2``); higher-dimensional Archimedean
copulas have known parameterisation limitations (single parameter
for all margins) and are best handled via vines.

Estimation uses Kendall's-tau inversion in all three cases.

References:
- Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed.,
  Springer.

Author:
- Jan Schlegel
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import kendalltau

from copulalib.copulas.base import Copula


# =========================================================================== #
#                       Bivariate Archimedean Base                            #
# =========================================================================== #
class _BivariateArchimedean(Copula):
    """Common scaffolding for bivariate Archimedean copulas.

    Subclasses provide ``cdf``, ``pdf``, ``sample``, the
    Kendall-tau-to-theta inversion via ``_theta_from_tau``, and
    parameter bounds.
    """

    _name: str = "Archimedean"

    def __init__(self, theta: float | None = None) -> None:
        super().__init__()
        self.theta: float | None = theta

    # -------------------------------------------------------------------------
    #  Fitting (Kendall's tau inversion)
    # -------------------------------------------------------------------------
    def _fit_copula(
        self,
        u: NDArray[np.float64],
        **kwargs: Any,
    ) -> None:
        """Estimate ``theta`` via inversion of Kendall's tau.

        Skipped if ``theta`` was pre-specified in ``__init__``.

        Parameters
        ----------
        u : NDArray of shape (n, 2)
            Bivariate uniform pseudo-observations.
        """
        self._require_bivariate(u)
        if self.theta is not None:
            return
        tau = float(kendalltau(u[:, 0], u[:, 1]).statistic)
        self.theta = self._theta_from_tau(tau)

    @abstractmethod
    def _theta_from_tau(self, tau: float) -> float:
        """Map a Kendall's tau value to the family parameter."""

    # -------------------------------------------------------------------------
    #  Helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _require_bivariate(arr: NDArray[np.float64]) -> None:
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                f"{_BivariateArchimedean._name} copulas require "
                f"bivariate data; got shape {arr.shape}."
            )

    def _check_fitted(self) -> None:
        if self.theta is None:
            raise RuntimeError(
                f"{type(self).__name__}: call fit_copula() (or "
                "pass `theta` to __init__) before this operation."
            )

    def __repr__(self) -> str:
        if self.theta is None:
            return f"{type(self).__name__}(unfitted)"
        return f"{type(self).__name__}(theta={self.theta:.4g})"


# =========================================================================== #
#                          Clayton Copula Class                               #
# =========================================================================== #
class ClaytonCopula(_BivariateArchimedean):
    """Bivariate Clayton copula with lower-tail dependence.

    Generator ``ψ(t) = (t^(-θ) - 1) / θ``, ``θ > 0``.
    Kendall's tau ``τ = θ / (θ + 2)``.

    Parameters
    ----------
    theta : float, optional
        Dependence parameter (``θ > 0``).  If ``None`` (default),
        estimated from data via tau inversion.
    """

    _name = "Clayton"

    def _theta_from_tau(self, tau: float) -> float:
        tau = float(np.clip(tau, 1e-6, 1.0 - 1e-6))
        return 2.0 * tau / (1.0 - tau)

    # -------------------------------------------------------------------------
    #  CDF / PDF
    # -------------------------------------------------------------------------
    def cdf(self, u: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Clayton bivariate CDF."""
        self._check_fitted()
        assert self.theta is not None
        arr = np.atleast_2d(np.asarray(u, dtype=np.float64))
        u1, u2 = arr[:, 0], arr[:, 1]
        return np.asarray(
            (u1 ** (-self.theta) + u2 ** (-self.theta) - 1.0)
            ** (-1.0 / self.theta),
            dtype=np.float64,
        )

    def pdf(self, u: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Clayton bivariate density."""
        self._check_fitted()
        assert self.theta is not None
        arr = np.atleast_2d(np.asarray(u, dtype=np.float64))
        u1, u2 = arr[:, 0], arr[:, 1]
        th = self.theta
        return np.asarray(
            (1.0 + th)
            * (u1 * u2) ** (-th - 1.0)
            * (u1 ** (-th) + u2 ** (-th) - 1.0) ** (-1.0 / th - 2.0),
            dtype=np.float64,
        )

    # -------------------------------------------------------------------------
    #  Sampling  (Marshall–Olkin: V ~ Gamma(1/θ, 1))
    # -------------------------------------------------------------------------
    def sample(self, n: int, **kwargs: Any) -> NDArray[np.float64]:
        """Draw ``n`` bivariate samples via Marshall–Olkin."""
        self._check_fitted()
        assert self.theta is not None
        rng: np.random.Generator = kwargs.get(
            "rng", np.random.default_rng()
        )
        v = rng.gamma(shape=1.0 / self.theta, scale=1.0, size=n)
        e = rng.exponential(scale=1.0, size=(n, 2))
        return np.asarray(
            (1.0 + e / v[:, None]) ** (-1.0 / self.theta),
            dtype=np.float64,
        )


# =========================================================================== #
#                           Gumbel Copula Class                               #
# =========================================================================== #
class GumbelCopula(_BivariateArchimedean):
    """Bivariate Gumbel copula with upper-tail dependence.

    Generator ``ψ(t) = (-log t)^θ``, ``θ ≥ 1``.
    Kendall's tau ``τ = 1 - 1/θ``.

    Parameters
    ----------
    theta : float, optional
        Dependence parameter (``θ ≥ 1``).  If ``None`` (default),
        estimated from data via tau inversion.
    """

    _name = "Gumbel"

    def _theta_from_tau(self, tau: float) -> float:
        tau = float(np.clip(tau, 1e-6, 1.0 - 1e-6))
        return 1.0 / (1.0 - tau)

    # -------------------------------------------------------------------------
    #  CDF / PDF
    # -------------------------------------------------------------------------
    def cdf(self, u: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Gumbel bivariate CDF."""
        self._check_fitted()
        assert self.theta is not None
        arr = np.atleast_2d(np.asarray(u, dtype=np.float64))
        u1, u2 = arr[:, 0], arr[:, 1]
        a = (-np.log(u1)) ** self.theta + (-np.log(u2)) ** self.theta
        return np.asarray(
            np.exp(-a ** (1.0 / self.theta)), dtype=np.float64
        )

    def pdf(self, u: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Gumbel bivariate density."""
        self._check_fitted()
        assert self.theta is not None
        arr = np.atleast_2d(np.asarray(u, dtype=np.float64))
        u1, u2 = arr[:, 0], arr[:, 1]
        th = self.theta
        lu, lv = -np.log(u1), -np.log(u2)
        a = lu**th + lv**th
        a_inv = a ** (1.0 / th)
        c = np.exp(-a_inv)
        return np.asarray(
            c
            * (lu * lv) ** (th - 1.0)
            / (u1 * u2)
            * a ** (-2.0 + 2.0 / th)
            * (a_inv + th - 1.0),
            dtype=np.float64,
        )

    # -------------------------------------------------------------------------
    #  Sampling  (Marshall–Olkin with positive α-stable mixing variable)
    # -------------------------------------------------------------------------
    def sample(self, n: int, **kwargs: Any) -> NDArray[np.float64]:
        """Draw ``n`` bivariate Gumbel samples.

        Uses the Marshall–Olkin construction with a positive
        α-stable mixing variable (Chambers–Mallows–Stuck algorithm)
        with ``α = 1/θ``.
        """
        self._check_fitted()
        assert self.theta is not None
        rng: np.random.Generator = kwargs.get(
            "rng", np.random.default_rng()
        )
        alpha = 1.0 / self.theta
        v = _positive_stable(alpha, n, rng)
        e = rng.exponential(scale=1.0, size=(n, 2))
        return np.asarray(
            np.exp(-((e / v[:, None]) ** alpha)), dtype=np.float64
        )


# =========================================================================== #
#                            Frank Copula Class                               #
# =========================================================================== #
class FrankCopula(_BivariateArchimedean):
    """Bivariate Frank copula (symmetric, no tail dependence).

    Generator ``ψ(t) = -log((e^(-θt) - 1) / (e^(-θ) - 1))``,
    ``θ ≠ 0``.  Kendall's tau is given by

        τ(θ) = 1 + (4 / θ) * (D₁(θ) - 1)

    where ``D₁(θ) = (1/θ) ∫₀^θ t / (e^t - 1) dt`` is the first
    Debye function.

    Parameters
    ----------
    theta : float, optional
        Dependence parameter (``θ ≠ 0``; positive for positive
        dependence).  If ``None`` (default), estimated from data
        via tau inversion.
    """

    _name = "Frank"

    def _theta_from_tau(self, tau: float) -> float:
        tau = float(np.clip(tau, -1.0 + 1e-6, 1.0 - 1e-6))
        if abs(tau) < 1e-6:
            return 1e-6  # near-independence; avoid θ = 0 singularity
        if tau > 0:
            return float(
                brentq(
                    lambda th: _frank_tau(th) - tau, 1e-6, 60.0
                )
            )
        return float(
            brentq(lambda th: _frank_tau(th) - tau, -60.0, -1e-6)
        )

    # -------------------------------------------------------------------------
    #  CDF / PDF
    # -------------------------------------------------------------------------
    def cdf(self, u: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Frank bivariate CDF."""
        self._check_fitted()
        assert self.theta is not None
        arr = np.atleast_2d(np.asarray(u, dtype=np.float64))
        u1, u2 = arr[:, 0], arr[:, 1]
        th = self.theta
        gu = np.expm1(-th * u1)  # e^(-θu) - 1
        gv = np.expm1(-th * u2)
        g1 = np.expm1(-th)
        return np.asarray(
            -np.log1p(gu * gv / g1) / th, dtype=np.float64
        )

    def pdf(self, u: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Frank bivariate density."""
        self._check_fitted()
        assert self.theta is not None
        arr = np.atleast_2d(np.asarray(u, dtype=np.float64))
        u1, u2 = arr[:, 0], arr[:, 1]
        th = self.theta
        eu = np.exp(-th * u1)
        ev = np.exp(-th * u2)
        e1 = np.exp(-th)
        denom = (1.0 - e1) - (1.0 - eu) * (1.0 - ev)
        return np.asarray(
            th * (1.0 - e1) * eu * ev / denom**2,
            dtype=np.float64,
        )

    # -------------------------------------------------------------------------
    #  Sampling  (closed-form conditional inversion)
    # -------------------------------------------------------------------------
    def sample(self, n: int, **kwargs: Any) -> NDArray[np.float64]:
        """Draw ``n`` bivariate Frank samples via conditional inversion."""
        self._check_fitted()
        assert self.theta is not None
        rng: np.random.Generator = kwargs.get(
            "rng", np.random.default_rng()
        )
        th = self.theta
        u1 = rng.uniform(0.0, 1.0, size=n)
        w = rng.uniform(0.0, 1.0, size=n)

        # ---------------------------------------------------------------------
        # Conditional CDF inverse:
        # u2 = -1/θ * log(1 + w(e^(-θ)-1) / (e^(-θu1) - w(e^(-θu1)-1)))
        # ---------------------------------------------------------------------
        eu1 = np.exp(-th * u1)
        e1 = np.exp(-th)
        u2 = -np.log1p(w * (e1 - 1.0) / (eu1 - w * (eu1 - 1.0))) / th
        return np.column_stack([u1, np.clip(u2, 1e-12, 1.0 - 1e-12)])


# =========================================================================== #
#                            Module Helpers                                   #
# =========================================================================== #
def _positive_stable(
    alpha: float,
    n: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Sample from a positive α-stable distribution (β=1, σ=1, μ=0).

    Uses the Chambers–Mallows–Stuck algorithm specialised to the
    totally-skewed positive case, which is the mixing variable
    needed for the Gumbel Marshall–Olkin construction.

    Parameters
    ----------
    alpha : float
        Stability index, ``alpha ∈ (0, 1)``.  For Gumbel,
        ``alpha = 1/θ``.
    n : int
        Number of samples.
    rng : np.random.Generator

    Returns
    -------
    NDArray of shape (n,)
        Strictly positive samples.
    """
    eps = 1e-8
    u = rng.uniform(-np.pi / 2.0 + eps, np.pi / 2.0 - eps, size=n)
    w = rng.exponential(scale=1.0, size=n)

    # ---------------------------------------------------------------------
    # CMS for totally-skewed positive stable (β = 1, α ∈ (0, 1))
    # B_{α,1} = π/2;  scale factor S_{α,1} = cos(πα/2)^(-1/α)
    # ---------------------------------------------------------------------
    scale = np.cos(np.pi * alpha / 2.0) ** (-1.0 / alpha)
    arg = u + np.pi / 2.0
    return scale * (
        np.sin(alpha * arg) / np.cos(u) ** (1.0 / alpha)
    ) * (np.cos(u - alpha * arg) / w) ** ((1.0 - alpha) / alpha)


def _frank_tau(theta: float) -> float:
    """Kendall's tau of the Frank copula at parameter ``theta``."""
    if abs(theta) < 1e-8:
        return 0.0

    def integrand(t: float) -> float:
        # Removable singularity at t=0 (limit equals 1)
        return 1.0 if abs(t) < 1e-10 else t / np.expm1(t)

    # quad handles negative upper limits with the right sign
    integral, _ = quad(integrand, 0.0, theta)
    debye_1 = integral / theta
    return 1.0 + 4.0 * (debye_1 - 1.0) / theta

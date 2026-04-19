"""
This module implements an abstract base class for copulas and a
dataclass for marginal specification.

Author:
    Jan Heinrich Schlegel 23.03.2026
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
from __future__ import annotations

import csv
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

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

    Subclasses must implement ``_fit_copula``, ``sample``, and
    ``cdf``.  ``pdf`` and ``logpdf`` are optional.

    The fitting workflow is split into two stages so that expert
    users can inspect and override marginal parameters between
    steps:

    .. code-block:: python

        copula.fit_marginals(data, marginals)
        copula.export_marginal_params("params.csv")
        # edit CSV in a spreadsheet
        copula.import_marginal_params("params.csv")
        copula.fit_copula(data)

    Alternatively, ``fit`` runs both stages in one call.
    """

    def __init__(self) -> None:
        self._marginals: list[Marginal] = []
        self._marginals_fitted: bool = False

    # -------------------------------------------------------------------------
    #  Marginal fitting
    # -------------------------------------------------------------------------
    def fit_marginals(
        self,
        data: ArrayLike,
        marginals: list[Marginal],
    ) -> None:
        """Fit each marginal distribution to its column of data.

        Parameters
        ----------
        data : array_like of shape (n, d)
            Multivariate sample.
        marginals : list[Marginal]
            One marginal per variable; length must equal ``d``.

        Raises
        ------
        ValueError
            If ``data`` is not 2-D or the number of marginals does
            not match the number of columns.
        """
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(
                f"data must be 2-D, got shape {arr.shape}."
            )
        if arr.shape[1] != len(marginals):
            raise ValueError(
                f"data has {arr.shape[1]} columns but "
                f"{len(marginals)} marginals were provided."
            )
        self._marginals = list(marginals)
        for i, m in enumerate(self._marginals):
            m.distribution.fit(arr[:, i])
        self._marginals_fitted = True

    # -------------------------------------------------------------------------
    #  Copula fitting
    # -------------------------------------------------------------------------
    @abstractmethod
    def _fit_copula(
        self,
        u: NDArray[np.float64],
        **kwargs: Any,
    ) -> None:
        """Fit copula structure to uniform pseudo-observations.

        Parameters
        ----------
        u : NDArray of shape (n, d)
            Pseudo-observations in the unit hypercube, obtained by
            applying each marginal CDF to the corresponding column.
        """

    def fit_copula(self, data: ArrayLike, **kwargs: Any) -> None:
        """Transform data to uniform space and fit copula structure.

        ``fit_marginals`` must be called first.

        Parameters
        ----------
        data : array_like of shape (n, d)
            Original multivariate sample (same array passed to
            ``fit_marginals``).

        Raises
        ------
        RuntimeError
            If ``fit_marginals`` has not been called yet.
        """
        self._check_marginals_fitted()
        arr = np.asarray(data, dtype=np.float64)
        u = self._to_uniform(arr)
        self._fit_copula(u, **kwargs)

    def fit(
        self,
        data: ArrayLike,
        marginals: list[Marginal],
        **kwargs: Any,
    ) -> Copula:
        """Fit marginals and copula structure in one step.

        Equivalent to calling ``fit_marginals`` followed by
        ``fit_copula``.

        Parameters
        ----------
        data : array_like of shape (n, d)
            Multivariate sample.
        marginals : list[Marginal]
            One marginal per variable.

        Returns
        -------
        self
        """
        self.fit_marginals(data, marginals)
        arr = np.asarray(data, dtype=np.float64)
        u = self._to_uniform(arr)
        self._fit_copula(u, **kwargs)
        return self

    # -------------------------------------------------------------------------
    #  Sampling and evaluation (to be implemented by subclasses)
    # -------------------------------------------------------------------------
    @abstractmethod
    def sample(self, n: int, **kwargs: Any) -> NDArray[np.float64]:
        """Draw samples in uniform (copula) space.

        Parameters
        ----------
        n : int
            Number of samples.

        Returns
        -------
        NDArray of shape (n, d)
            Values in [0, 1].
        """

    @abstractmethod
    def cdf(self, u: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Copula CDF.

        Parameters
        ----------
        u : array_like of shape (n, d)
            Points in the unit hypercube.

        Returns
        -------
        NDArray
            Copula CDF values.
        """

    def sample_data(
        self, n: int, **kwargs: Any
    ) -> NDArray[np.float64]:
        """Draw samples in the original data space.

        Calls ``sample`` to obtain uniform pseudo-observations, then
        applies the inverse CDF (PPF) of each fitted marginal.

        Parameters
        ----------
        n : int
            Number of samples.

        Returns
        -------
        NDArray of shape (n, d)
            Samples in the original data space.

        Raises
        ------
        RuntimeError
            If ``fit_marginals`` has not been called yet.
        """
        self._check_marginals_fitted()
        u = self.sample(n, **kwargs)
        return np.column_stack(
            [
                m.distribution.ppf(u[:, i])
                for i, m in enumerate(self._marginals)
            ]
        )

    def pdf(self, u: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Copula density.

        Optional — raises ``NotImplementedError`` by default.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement pdf."
        )

    def logpdf(self, u: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """Log copula density.

        Falls back to ``log(pdf(u))`` if not overridden.
        """
        return np.log(self.pdf(u))

    # -------------------------------------------------------------------------
    #  Parameter export / import
    # -------------------------------------------------------------------------
    def export_marginal_params(self, path: str) -> None:
        """Write fitted marginal parameters to a CSV file.

        The CSV has one row per marginal.  Columns are ``variable``,
        ``distribution``, and one column per unique parameter name
        across all marginals.  Parameters that do not apply to a
        given distribution are written as empty cells.

        Parameters
        ----------
        path : str
            Destination file path.

        Raises
        ------
        RuntimeError
            If ``fit_marginals`` has not been called yet.
        """
        self._check_marginals_fitted()

        # ---------------------------------------------------------------------
        # Collect ordered unique param names across all marginals
        # ---------------------------------------------------------------------
        param_names: list[str] = []
        seen: set[str] = set()
        for m in self._marginals:
            for k in m.distribution.params:
                if k not in seen:
                    param_names.append(k)
                    seen.add(k)

        fieldnames = ["variable", "distribution"] + param_names
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in self._marginals:
                row: dict[str, object] = {
                    "variable": m.name,
                    "distribution": type(m.distribution).__name__,
                }
                for k in param_names:
                    v = m.distribution.params.get(k)
                    row[k] = "" if v is None else v
                writer.writerow(row)

    def import_marginal_params(self, path: str) -> None:
        """Read marginal parameters from a CSV and apply overrides.

        Only non-empty cells in parameter columns are applied;
        empty cells leave the current value unchanged.  The
        ``distribution`` column is validated but never used to
        change the distribution type.

        Parameters
        ----------
        path : str
            CSV file previously created by
            ``export_marginal_params`` (and optionally edited).

        Raises
        ------
        RuntimeError
            If ``fit_marginals`` has not been called yet.
        """
        self._check_marginals_fitted()
        marginal_map = {
            m.name: m.distribution for m in self._marginals
        }

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("variable", "")
                if name not in marginal_map:
                    continue
                dist = marginal_map[name]

                # -------------------------------------------------------------
                # Warn if the CSV distribution type does not match
                # -------------------------------------------------------------
                csv_type = row.get("distribution", "")
                current_type = type(dist).__name__
                if csv_type and csv_type != current_type:
                    warnings.warn(
                        f"Variable '{name}': CSV specifies "
                        f"'{csv_type}' but current distribution is "
                        f"'{current_type}'. Applying params anyway.",
                        UserWarning,
                        stacklevel=2,
                    )

                # -------------------------------------------------------------
                # Parse and apply non-empty parameter cells
                # -------------------------------------------------------------
                params: dict[str, float] = {}
                for k, v in row.items():
                    if k in ("variable", "distribution") or not v:
                        continue
                    try:
                        params[k] = float(v)
                    except ValueError:
                        continue
                if params:
                    dist.set_params(params)

    # -------------------------------------------------------------------------
    #  Helpers
    # -------------------------------------------------------------------------
    def _to_uniform(
        self, arr: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Apply each marginal CDF to its column."""
        return np.column_stack(
            [
                m.distribution.cdf(arr[:, i])
                for i, m in enumerate(self._marginals)
            ]
        )

    def _check_marginals_fitted(self) -> None:
        """Raise if ``fit_marginals`` has not been called yet."""
        if not self._marginals_fitted:
            raise RuntimeError(
                f"{type(self).__name__}: call fit_marginals() "
                f"before this operation."
            )

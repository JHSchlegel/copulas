"""
Utilities for working with correlation matrices.

Author:
- Jan Schlegel
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


# =========================================================================== #
#                         Correlation Projection                              #
# =========================================================================== #
def to_correlation(
    matrix: ArrayLike,
    eig_floor: float = 1e-10,
) -> NDArray[np.float64]:
    """Project a symmetric matrix onto the set of valid correlation
    matrices via eigenvalue clipping and diagonal renormalisation.

    The procedure is: symmetrise, clip eigenvalues from below, and
    rescale by the inverse square root of the diagonal so that all
    diagonal entries equal one.  This is the simple eigenvalue
    projection — for the strict nearest-correlation problem use
    Higham's algorithm.

    Parameters
    ----------
    matrix : array_like of shape (d, d)
        Symmetric matrix to project.  Need not be positive
        semi-definite.
    eig_floor : float, default 1e-10
        Lower bound enforced on eigenvalues to ensure strict
        positive definiteness.

    Returns
    -------
    NDArray of shape (d, d)
        Symmetric positive-definite matrix with unit diagonal.

    Raises
    ------
    ValueError
        If ``matrix`` is not 2-D and square.
    """
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(
            f"matrix must be 2-D and square, got shape {arr.shape}."
        )

    # ---------------------------------------------------------------------
    # Symmetrise then clip eigenvalues from below
    # ---------------------------------------------------------------------
    sym = 0.5 * (arr + arr.T)
    w, v = np.linalg.eigh(sym)
    w = np.clip(w, eig_floor, None)
    psd = (v * w) @ v.T  # (d, d)

    # ---------------------------------------------------------------------
    # Rescale rows and columns so diagonal entries are exactly one
    # ---------------------------------------------------------------------
    d_inv_sqrt = 1.0 / np.sqrt(np.diag(psd))
    corr = psd * np.outer(d_inv_sqrt, d_inv_sqrt)
    np.fill_diagonal(corr, 1.0)
    return corr

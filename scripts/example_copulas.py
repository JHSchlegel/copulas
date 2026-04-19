"""
End-to-end demonstration of the copula library on synthetic data.

Generates a 2-D sample from a Student-t copula (heavy joint tails)
with a Beta(2, 5) marginal and a Gamma(2, 1) marginal.  Each copula
implementation is then fitted to the data, samples are drawn, and
the results are compared against the truth via Kendall's tau,
Spearman's rho, scatter plots in pseudo-observation space, and
density contour plots for the smooth parametric copulas.

Outputs
-------
- ``scripts/figures/copula_samples.png`` — pseudo-obs scatter grid
- ``scripts/figures/copula_density_contours.png`` — fitted densities
- ``scripts/figures/data_space_samples.png`` — samples in original
  marginal space

Author:
- Jan Schlegel
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.stats import kendalltau, spearmanr, t

from copulalib.copulas.archimedean import (
    ClaytonCopula,
    FrankCopula,
    GumbelCopula,
)
from copulalib.copulas.base import Copula, Marginal
from copulalib.copulas.comonotonic import ComonotonicCopula
from copulalib.copulas.empirical import EmpiricalCopula
from copulalib.copulas.gaussian import GaussianCopula
from copulalib.copulas.independence import IndependenceCopula
from copulalib.copulas.student_t import StudentTCopula
from copulalib.copulas.vine import DVineCopula
from copulalib.distributions.beta import BetaDistribution
from copulalib.distributions.gamma import GammaDistribution

FIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "figures"
)
os.makedirs(FIG_DIR, exist_ok=True)

PLT_STYLE: dict[str, Any] = {
    "scatter": {"s": 6, "alpha": 0.4, "edgecolors": "none"},
    "truth_color": "#222222",
    "fit_color": "#1f77b4",
    "contour_levels": 10,
}


# =========================================================================== #
#                         Synthetic Data Generation                           #
# =========================================================================== #
def generate_data(
    n: int = 4000,
    rho: float = 0.7,
    df: float = 4.0,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Draw ``n`` samples from a Student-t copula with a Beta and a
    Gamma marginal.

    Returns
    -------
    data : NDArray of shape (n, 2)
        Samples in the original (data) space.
    pseudo : NDArray of shape (n, 2)
        Corresponding pseudo-observations in the unit square
        (obtained from the true marginal CDFs).
    """
    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------------
    # Sample t copula in latent space, then map to uniforms via t CDF
    # ---------------------------------------------------------------------
    cov = np.array([[1.0, rho], [rho, 1.0]])
    z = rng.multivariate_normal([0.0, 0.0], cov, size=n)
    w = rng.chisquare(df, size=n)
    x = z / np.sqrt(w / df)[:, None]
    u = t.cdf(x, df=df)

    # ---------------------------------------------------------------------
    # Push uniforms through Beta(2, 5) and Gamma(2, 1) inverse CDFs
    # ---------------------------------------------------------------------
    from scipy.stats import beta as sp_beta
    from scipy.stats import gamma as sp_gamma

    col_0 = sp_beta.ppf(u[:, 0], a=2.0, b=5.0)
    col_1 = sp_gamma.ppf(u[:, 1], a=2.0, scale=1.0)
    data = np.column_stack([col_0, col_1])
    return data, u


# =========================================================================== #
#                            Copula Catalogue                                 #
# =========================================================================== #
def build_copulas() -> dict[str, Copula]:
    """Instantiate one copy of each copula model offered."""
    return {
        "Independence": IndependenceCopula(),
        "Comonotonic": ComonotonicCopula(),
        "Empirical": EmpiricalCopula(),
        "Gaussian": GaussianCopula(),
        "Student-t": StudentTCopula(),
        "Clayton": ClaytonCopula(),
        "Gumbel": GumbelCopula(),
        "Frank": FrankCopula(),
        "D-vine (Gaussian)": DVineCopula(),
    }


def make_marginals() -> list[Marginal]:
    """Fresh marginal specifications for one fit pass."""
    return [
        Marginal("x", BetaDistribution()),
        Marginal("y", GammaDistribution()),
    ]


# =========================================================================== #
#                              Visualisation                                  #
# =========================================================================== #
def _scatter_grid(
    sample_dict: dict[str, NDArray[np.float64]],
    truth: NDArray[np.float64],
    title: str,
    out_path: str,
    is_uniform: bool,
) -> None:
    """Plot a 3 × 3 grid of scatters, one per copula."""
    n_cop = len(sample_dict)
    n_cols = 3
    n_rows = (n_cop + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(11, 3.5 * n_rows), sharex=is_uniform,
        sharey=is_uniform,
    )
    axes = np.atleast_2d(axes).ravel()

    for ax, (name, sample) in zip(axes, sample_dict.items()):
        ax.scatter(
            truth[:, 0],
            truth[:, 1],
            color=PLT_STYLE["truth_color"],
            label="data",
            **PLT_STYLE["scatter"],
        )
        ax.scatter(
            sample[:, 0],
            sample[:, 1],
            color=PLT_STYLE["fit_color"],
            label="model",
            **PLT_STYLE["scatter"],
        )
        tau = float(kendalltau(sample[:, 0], sample[:, 1]).statistic)
        rho_s = float(spearmanr(sample[:, 0], sample[:, 1]).statistic)
        ax.set_title(
            f"{name}\nτ = {tau:+.2f}   ρ_S = {rho_s:+.2f}",
            fontsize=10,
        )
        if is_uniform:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
        ax.tick_params(labelsize=8)

    for ax in axes[len(sample_dict):]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper right", fontsize=9, frameon=True
    )
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_density_contours(
    copulas: dict[str, Copula],
    truth_pseudo: NDArray[np.float64],
    out_path: str,
) -> None:
    """Filled-contour plots of fitted densities for the smooth
    parametric copulas (those that implement ``pdf``)."""
    smooth = ["Gaussian", "Student-t", "Clayton", "Gumbel", "Frank"]
    fig, axes = plt.subplots(1, len(smooth), figsize=(15, 3.4))

    grid = np.linspace(0.02, 0.98, 80)
    uu, vv = np.meshgrid(grid, grid)
    points = np.column_stack([uu.ravel(), vv.ravel()])

    for ax, name in zip(axes, smooth):
        z = copulas[name].pdf(points).reshape(uu.shape)
        z = np.clip(z, 0, np.quantile(z, 0.99))  # cap extreme values
        ax.contourf(
            uu, vv, z, levels=PLT_STYLE["contour_levels"], cmap="viridis"
        )
        ax.scatter(
            truth_pseudo[:, 0],
            truth_pseudo[:, 1],
            s=2,
            color="white",
            alpha=0.25,
        )
        ax.set_title(name, fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=8)

    fig.suptitle(
        "Fitted copula densities (white dots = empirical pseudo-obs)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


# =========================================================================== #
#                             Reporting Helpers                               #
# =========================================================================== #
def report_dependence(
    name: str,
    sample: NDArray[np.float64],
    target_tau: float,
) -> None:
    """Print one summary line per fitted copula."""
    tau = float(kendalltau(sample[:, 0], sample[:, 1]).statistic)
    rho_s = float(spearmanr(sample[:, 0], sample[:, 1]).statistic)
    print(
        f"  {name:<18s}  tau = {tau:+.3f}  "
        f"(target {target_tau:+.3f})   rho_S = {rho_s:+.3f}"
    )


# =========================================================================== #
#                                  Main                                       #
# =========================================================================== #
def main() -> None:
    print("Generating synthetic data...")
    data, true_pseudo = generate_data()
    print(f"  data shape = {data.shape}")
    target_tau = float(
        kendalltau(true_pseudo[:, 0], true_pseudo[:, 1]).statistic
    )
    print(f"  target Kendall's tau = {target_tau:+.3f}")

    print("\nFitting copulas...")
    copulas = build_copulas()
    rng = np.random.default_rng(123)
    fit_samples_uniform: dict[str, NDArray[np.float64]] = {}
    fit_samples_data: dict[str, NDArray[np.float64]] = {}
    for name, cop in copulas.items():
        cop.fit(data, make_marginals())
        s_u = cop.sample(2000, rng=rng)
        s_d = cop.sample_data(2000, rng=rng)
        fit_samples_uniform[name] = s_u
        fit_samples_data[name] = s_d
        report_dependence(name, s_u, target_tau)

    print("\nDrawing figures...")
    _scatter_grid(
        fit_samples_uniform,
        true_pseudo,
        "Pseudo-obs (uniform space): black = data, blue = model",
        os.path.join(FIG_DIR, "copula_samples.png"),
        is_uniform=True,
    )
    _scatter_grid(
        fit_samples_data,
        data,
        "Data space: black = data, blue = model",
        os.path.join(FIG_DIR, "data_space_samples.png"),
        is_uniform=False,
    )
    plot_density_contours(
        copulas,
        true_pseudo,
        os.path.join(FIG_DIR, "copula_density_contours.png"),
    )


if __name__ == "__main__":
    main()

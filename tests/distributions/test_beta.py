"""
This module implements unit tests for the BetaDistribution class.

Author:
    Jan Heinrich Schlegel 13.04.2026
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
from __future__ import annotations

import unittest

import numpy as np

from copulalib.distributions.beta import BetaDistribution


# =========================================================================== #
#                          Test Beta Distribution                             #
# =========================================================================== #
class TestBetaDistribution(unittest.TestCase):
    """Unit tests for BetaDistribution."""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        self.data = self.rng.beta(a=2.0, b=5.0, size=5000)
        self.dist = BetaDistribution()
        self.dist.fit(self.data)

    # ------------------------------------------------------------------ #
    #  fit                                                                 #
    # ------------------------------------------------------------------ #
    def test_fit_recovers_alpha(self) -> None:
        self.assertAlmostEqual(self.dist.alpha, 2.0, delta=0.3)

    def test_fit_recovers_beta(self) -> None:
        self.assertAlmostEqual(self.dist.beta, 5.0, delta=0.5)

    def test_fit_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            BetaDistribution().fit([])

    def test_fit_out_of_range_raises(self) -> None:
        with self.assertRaises(ValueError):
            BetaDistribution().fit([0.0, 0.5, 1.0])

    def test_fit_preserves_preset_alpha(self) -> None:
        dist = BetaDistribution(alpha=3.0)
        dist.fit(self.data)
        self.assertEqual(dist.alpha, 3.0)

    def test_fit_preserves_preset_beta(self) -> None:
        dist = BetaDistribution(beta=2.0)
        dist.fit(self.data)
        self.assertEqual(dist.beta, 2.0)

    # ------------------------------------------------------------------ #
    #  cdf                                                                 #
    # ------------------------------------------------------------------ #
    def test_cdf_at_zero(self) -> None:
        self.assertAlmostEqual(float(self.dist.cdf(0.0)), 0.0)

    def test_cdf_at_one(self) -> None:
        self.assertAlmostEqual(float(self.dist.cdf(1.0)), 1.0)

    def test_cdf_monotone(self) -> None:
        x = np.linspace(0.01, 0.99, 50)
        vals = self.dist.cdf(x)
        self.assertTrue(np.all(np.diff(vals) >= 0))

    def test_cdf_unfitted_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            BetaDistribution().cdf(0.5)

    # ------------------------------------------------------------------ #
    #  ppf                                                                 #
    # ------------------------------------------------------------------ #
    def test_ppf_roundtrip(self) -> None:
        x = np.array([0.1, 0.3, 0.5, 0.8])
        np.testing.assert_allclose(
            self.dist.ppf(self.dist.cdf(x)), x, atol=1e-10
        )

    def test_ppf_unfitted_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            BetaDistribution().ppf(0.5)

    # ------------------------------------------------------------------ #
    #  pdf / logpdf                                                        #
    # ------------------------------------------------------------------ #
    def test_pdf_nonnegative(self) -> None:
        x = np.linspace(0.01, 0.99, 100)
        self.assertTrue(np.all(self.dist.pdf(x) >= 0))

    def test_logpdf_consistent_with_pdf(self) -> None:
        x = np.array([0.1, 0.3, 0.7])
        np.testing.assert_allclose(
            self.dist.logpdf(x), np.log(self.dist.pdf(x)), atol=1e-12
        )

    # ------------------------------------------------------------------ #
    #  sample                                                              #
    # ------------------------------------------------------------------ #
    def test_sample_shape(self) -> None:
        self.assertEqual(self.dist.sample(100).shape, (100,))

    def test_sample_in_unit_interval(self) -> None:
        samples = self.dist.sample(500, rng=np.random.default_rng(0))
        self.assertTrue(np.all((samples > 0) & (samples < 1)))

    def test_sample_reproducible(self) -> None:
        a = self.dist.sample(50, rng=np.random.default_rng(42))
        b = self.dist.sample(50, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(a, b)

    def test_sample_unfitted_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            BetaDistribution().sample(10)

    # ------------------------------------------------------------------ #
    #  params / set_params                                                 #
    # ------------------------------------------------------------------ #
    def test_params_keys(self) -> None:
        self.assertEqual(set(self.dist.params), {"alpha", "beta"})

    def test_set_params_unknown_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.dist.set_params({"gamma": 1.0})

    # ------------------------------------------------------------------ #
    #  repr                                                                #
    # ------------------------------------------------------------------ #
    def test_repr_unfitted(self) -> None:
        r = repr(BetaDistribution())
        self.assertIn("None", r)

    def test_repr_fitted(self) -> None:
        r = repr(self.dist)
        self.assertIn("alpha=", r)
        self.assertIn("beta=", r)


if __name__ == "__main__":
    unittest.main()

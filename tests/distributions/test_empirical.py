"""
This module implements unit tests for the EmpiricalDistribution class.

Author:
    Jan Heinrich Schlegel 24.03.2026
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
from __future__ import annotations

import unittest

import numpy as np

from copulalib.distributions.empirical import EmpiricalDistribution


# =========================================================================== #
#                        Test Empirical Distribution                          #
# =========================================================================== #
class TestEmpiricalDistribution(unittest.TestCase):
    """Unit tests for EmpiricalDistribution."""

    def setUp(self) -> None:
        self.dist = EmpiricalDistribution()
        self.dist.fit([3, 1, 5, 2, 4])

    # ------------------------------------------------------------------ #
    #  fit                                                                 #
    # ------------------------------------------------------------------ #
    def test_fit_sorts_data(self) -> None:
        np.testing.assert_array_equal(self.dist._data, [1, 2, 3, 4, 5])

    def test_fit_stores_length(self) -> None:
        self.assertEqual(self.dist._n, 5)

    def test_fit_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            EmpiricalDistribution().fit([])

    # ------------------------------------------------------------------ #
    #  cdf                                                                 #
    # ------------------------------------------------------------------ #
    def test_cdf_below_min(self) -> None:
        self.assertEqual(self.dist.cdf(0.0), 0.0)

    def test_cdf_above_max(self) -> None:
        self.assertEqual(self.dist.cdf(10.0), 1.0)

    def test_cdf_at_data_points(self) -> None:
        expected = np.array([1, 2, 3, 4, 5]) / 5
        np.testing.assert_array_equal(self.dist.cdf([1, 2, 3, 4, 5]), expected)

    def test_cdf_between_points(self) -> None:
        self.assertAlmostEqual(self.dist.cdf(2.5), 0.4)

    def test_cdf_unfitted_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            EmpiricalDistribution().cdf(0.5)

    # ------------------------------------------------------------------ #
    #  ppf                                                                 #
    # ------------------------------------------------------------------ #
    def test_ppf_at_zero(self) -> None:
        self.assertEqual(self.dist.ppf(0.0), 1.0)

    def test_ppf_at_one(self) -> None:
        self.assertEqual(self.dist.ppf(1.0), 5.0)

    def test_ppf_median(self) -> None:
        self.assertEqual(self.dist.ppf(0.5), 3.0)

    def test_ppf_out_of_range_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.dist.ppf(1.5)

    def test_ppf_unfitted_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            EmpiricalDistribution().ppf(0.5)

    # ------------------------------------------------------------------ #
    #  sample                                                              #
    # ------------------------------------------------------------------ #
    def test_sample_shape(self) -> None:
        self.assertEqual(self.dist.sample(100).shape, (100,))

    def test_sample_values_in_data(self) -> None:
        samples = self.dist.sample(200, rng=np.random.default_rng(0))
        self.assertTrue(set(samples).issubset({1, 2, 3, 4, 5}))

    def test_sample_reproducible(self) -> None:
        a = self.dist.sample(50, rng=np.random.default_rng(42))
        b = self.dist.sample(50, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(a, b)

    def test_sample_unfitted_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            EmpiricalDistribution().sample(10)

    # ------------------------------------------------------------------ #
    #  repr                                                                #
    # ------------------------------------------------------------------ #
    def test_repr_unfitted(self) -> None:
        self.assertIn("unfitted", repr(EmpiricalDistribution()))

    def test_repr_fitted(self) -> None:
        self.assertIn("n=5", repr(self.dist))


if __name__ == "__main__":
    unittest.main()

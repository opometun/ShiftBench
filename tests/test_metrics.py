from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shiftbench.metrics import (
    METRICS,
    centroid_distance,
    compute_covariance,
    compute_mean,
    frechet_distance,
    get_metric,
)


def _gaussian(n_samples: int, n_features: int, shift: float, scale: float, seed: int):
    rng = np.random.default_rng(seed)
    return (rng.normal(0.0, scale, (n_samples, n_features)) + shift).astype(np.float32)


class FeatureStatsTest(unittest.TestCase):
    def test_mean_is_per_feature_centroid(self) -> None:
        embeddings = np.array([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]])

        mu = compute_mean(embeddings)

        self.assertEqual(mu.shape, (2,))
        np.testing.assert_allclose(mu, [2.0, 20.0])

    def test_covariance_is_square_and_unbiased(self) -> None:
        embeddings = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 9.0]])

        sigma = compute_covariance(embeddings)

        self.assertEqual(sigma.shape, (2, 2))
        # Bessel-corrected: variance of [0, 2, 4] is 4.0, not 8/3.
        self.assertAlmostEqual(sigma[0, 0], 4.0)
        np.testing.assert_allclose(sigma, sigma.T)


class CentroidDistanceTest(unittest.TestCase):
    def test_identical_distributions_score_zero(self) -> None:
        mu = compute_mean(_gaussian(200, 8, 0.0, 1.0, seed=0))

        self.assertEqual(centroid_distance(mu, mu), 0.0)

    def test_matches_known_euclidean_distance(self) -> None:
        mu_a = np.array([0.0, 0.0])
        mu_b = np.array([3.0, 4.0])

        self.assertAlmostEqual(centroid_distance(mu_a, mu_b), 5.0)

    def test_is_symmetric(self) -> None:
        mu_a = compute_mean(_gaussian(200, 8, 0.0, 1.0, seed=1))
        mu_b = compute_mean(_gaussian(200, 8, 1.5, 1.0, seed=2))

        self.assertAlmostEqual(
            centroid_distance(mu_a, mu_b),
            centroid_distance(mu_b, mu_a),
        )


class FrechetDistanceTest(unittest.TestCase):
    def test_identical_distributions_score_zero(self) -> None:
        embeddings = _gaussian(200, 8, 0.0, 1.0, seed=0)
        mu = compute_mean(embeddings)
        sigma = compute_covariance(embeddings)

        # Rounding can push d2 slightly negative here; the clamp keeps it real.
        self.assertAlmostEqual(frechet_distance(mu, sigma, mu, sigma), 0.0, places=5)

    def test_equals_centroid_distance_when_covariances_match(self) -> None:
        # With sigma_a == sigma_b the covariance terms cancel exactly, so the
        # Frechet distance collapses to the distance between the means.
        sigma = compute_covariance(_gaussian(300, 6, 0.0, 1.0, seed=3))
        mu_a = np.zeros(6)
        mu_b = np.full(6, 2.0)

        self.assertAlmostEqual(
            frechet_distance(mu_a, sigma, mu_b, sigma),
            centroid_distance(mu_a, mu_b),
            places=4,
        )

    def test_catches_mode_collapse_that_centroid_distance_misses(self) -> None:
        # Same mean, wildly different spread: the case centroid distance
        # cannot see and the whole reason covariance is computed at all.
        diverse = _gaussian(400, 8, 0.0, 1.0, seed=4)
        collapsed = _gaussian(400, 8, 0.0, 0.001, seed=5)
        mu_a, sigma_a = compute_mean(diverse), compute_covariance(diverse)
        mu_b, sigma_b = compute_mean(collapsed), compute_covariance(collapsed)

        centroid = centroid_distance(mu_a, mu_b)
        frechet = frechet_distance(mu_a, sigma_a, mu_b, sigma_b)

        self.assertLess(centroid, 0.5)
        self.assertGreater(frechet, 2.0)
        self.assertGreater(frechet, centroid * 5)

    def test_is_symmetric(self) -> None:
        a = _gaussian(300, 6, 0.0, 1.0, seed=6)
        b = _gaussian(300, 6, 1.0, 2.0, seed=7)
        mu_a, sigma_a = compute_mean(a), compute_covariance(a)
        mu_b, sigma_b = compute_mean(b), compute_covariance(b)

        self.assertAlmostEqual(
            frechet_distance(mu_a, sigma_a, mu_b, sigma_b),
            frechet_distance(mu_b, sigma_b, mu_a, sigma_a),
            places=4,
        )

    def test_grows_with_separation(self) -> None:
        sigma = compute_covariance(_gaussian(300, 6, 0.0, 1.0, seed=8))
        mu = np.zeros(6)

        near = frechet_distance(mu, sigma, np.full(6, 0.5), sigma)
        far = frechet_distance(mu, sigma, np.full(6, 5.0), sigma)

        self.assertLess(near, far)

    def test_returns_plain_float(self) -> None:
        sigma = compute_covariance(_gaussian(100, 4, 0.0, 1.0, seed=9))
        mu = np.zeros(4)

        self.assertIsInstance(frechet_distance(mu, sigma, mu, sigma), float)
        self.assertIsInstance(centroid_distance(mu, mu), float)


class MetricRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        a = _gaussian(200, 6, 0.0, 1.0, seed=10)
        b = _gaussian(200, 6, 1.5, 2.0, seed=11)
        self.mu_a, self.sigma_a = compute_mean(a), compute_covariance(a)
        self.mu_b, self.sigma_b = compute_mean(b), compute_covariance(b)

    def _through_registry(self, name: str) -> float:
        return get_metric(name).compute(
            self.mu_a, self.sigma_a, self.mu_b, self.sigma_b
        )

    def test_registry_matches_calling_the_functions_directly(self) -> None:
        self.assertAlmostEqual(
            self._through_registry("centroid"),
            centroid_distance(self.mu_a, self.mu_b),
        )
        self.assertAlmostEqual(
            self._through_registry("frechet"),
            frechet_distance(self.mu_a, self.sigma_a, self.mu_b, self.sigma_b),
        )

    def test_centroid_ignores_covariance_it_is_handed(self) -> None:
        # Same means, wildly different covariances: centroid must not budge,
        # which is what lets both metrics share one call signature.
        garbage = np.eye(6) * 1000.0

        self.assertAlmostEqual(
            get_metric("centroid").compute(
                self.mu_a, self.sigma_a, self.mu_b, garbage
            ),
            self._through_registry("centroid"),
        )

    def test_uses_covariance_flag_is_accurate(self) -> None:
        self.assertFalse(METRICS["centroid"].uses_covariance)
        self.assertTrue(METRICS["frechet"].uses_covariance)

    def test_unknown_metric_lists_the_available_ones(self) -> None:
        with self.assertRaises(ValueError) as caught:
            get_metric("euclidean")

        message = str(caught.exception)
        self.assertIn("euclidean", message)
        self.assertIn("centroid", message)
        self.assertIn("frechet", message)

    def test_every_registered_metric_is_callable_and_self_named(self) -> None:
        for name, metric in METRICS.items():
            with self.subTest(metric=name):
                self.assertEqual(metric.name, name)
                self.assertIsInstance(self._through_registry(name), float)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shiftbench.metrics import (
    METRICS,
    SUMMARIES,
    centroid_distance,
    compute_covariance,
    compute_mean,
    frechet_distance,
    gaussian_summary,
    get_metric,
    get_summary,
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
        self.summary_a = gaussian_summary(a)
        self.summary_b = gaussian_summary(b)

    def test_registry_matches_calling_the_functions_directly(self) -> None:
        self.assertAlmostEqual(
            get_metric("centroid").compare(self.summary_a, self.summary_b),
            centroid_distance(self.summary_a["mu"], self.summary_b["mu"]),
        )
        self.assertAlmostEqual(
            get_metric("frechet").compare(self.summary_a, self.summary_b),
            frechet_distance(
                self.summary_a["mu"],
                self.summary_a["sigma"],
                self.summary_b["mu"],
                self.summary_b["sigma"],
            ),
        )

    def test_centroid_ignores_covariance_it_is_handed(self) -> None:
        # Same means, garbage covariance: centroid must not budge, which is
        # what lets all gaussian metrics share one summary artifact.
        doctored = dict(self.summary_b)
        doctored["sigma"] = np.eye(6) * 1000.0

        self.assertAlmostEqual(
            get_metric("centroid").compare(self.summary_a, doctored),
            get_metric("centroid").compare(self.summary_a, self.summary_b),
        )

    def test_gaussian_summary_matches_its_pieces(self) -> None:
        embeddings = _gaussian(100, 4, 0.5, 1.0, seed=12)

        summary = gaussian_summary(embeddings)

        np.testing.assert_allclose(summary["mu"], compute_mean(embeddings))
        np.testing.assert_allclose(summary["sigma"], compute_covariance(embeddings))

    def test_unknown_names_list_the_available_ones(self) -> None:
        with self.assertRaises(ValueError) as caught:
            get_metric("euclidean")
        message = str(caught.exception)
        self.assertIn("euclidean", message)
        self.assertIn("centroid", message)
        self.assertIn("class_frequency_js", message)

        with self.assertRaises(ValueError) as caught:
            get_summary("wavelet")
        self.assertIn("gaussian", str(caught.exception))

    def test_every_metric_is_self_named_and_consistent(self) -> None:
        # A summarizing metric must point at a registered Summary of the same
        # input kind; a pairwise metric has neither summary nor compare.
        for name, metric in METRICS.items():
            with self.subTest(metric=name):
                self.assertEqual(metric.name, name)
                if metric.summary is None:
                    self.assertIsNone(metric.compare)
                    self.assertIsNotNone(metric.pairwise)
                    self.assertEqual(metric.input, "image_dirs")
                else:
                    self.assertIn(metric.summary, SUMMARIES)
                    self.assertEqual(SUMMARIES[metric.summary].input, metric.input)
                    self.assertIsNotNone(metric.compare)
                    self.assertIsNone(metric.pairwise)

    def test_sadge_records_its_direction(self) -> None:
        # SADGE is a fused similarity: unlike every distance here, higher
        # means MORE alike. That must be recorded next to the score.
        self.assertTrue(METRICS["sadge"].params["higher_is_better"])
        self.assertEqual(METRICS["sadge"].params["K"], 10)

    def test_expected_metric_names_are_registered(self) -> None:
        self.assertEqual(
            sorted(METRICS),
            [
                "centroid",
                "class_frequency_js",
                "class_presence_js",
                "color_js",
                "frechet",
                "sadge",
                "scene_complexity_js",
                "texture_js",
            ],
        )


class HistogramEquivalenceTest(unittest.TestCase):
    """The summary-artifact path must give the same number as calling the
    original quantify_* functions directly — the Phase 5 no-drift guarantee."""

    def setUp(self) -> None:
        rng = np.random.default_rng(20)
        self.masks_a = [rng.integers(0, 5, (8, 8)) for _ in range(4)]
        self.masks_b = [rng.integers(0, 3, (8, 8)) for _ in range(4)]

    def _via_summaries(self, metric_name: str, a, b, **params) -> float:
        metric = get_metric(metric_name)
        make = SUMMARIES[metric.summary].make
        return metric.compare(make(a, **params), make(b, **params))

    def test_class_frequency_matches_quantify(self) -> None:
        from shiftbench.shift_quantification_metrics.label_based.class_frequency import (
            quantify_class_frequency_shift,
        )

        self.assertAlmostEqual(
            self._via_summaries(
                "class_frequency_js", self.masks_a, self.masks_b, num_classes=5
            ),
            quantify_class_frequency_shift(self.masks_a, self.masks_b, num_classes=5),
            places=12,
        )

    def test_class_presence_matches_quantify(self) -> None:
        from shiftbench.shift_quantification_metrics.label_based.class_presence import (
            quantify_class_presence_shift,
        )

        self.assertAlmostEqual(
            self._via_summaries(
                "class_presence_js", self.masks_a, self.masks_b, num_classes=5
            ),
            quantify_class_presence_shift(self.masks_a, self.masks_b, num_classes=5),
            places=12,
        )

    def test_scene_complexity_matches_quantify(self) -> None:
        from shiftbench.shift_quantification_metrics.label_based.scene_complexity import (
            quantify_scene_complexity_shift,
        )

        self.assertAlmostEqual(
            self._via_summaries(
                "scene_complexity_js", self.masks_a, self.masks_b, num_classes=5
            ),
            quantify_scene_complexity_shift(self.masks_a, self.masks_b, num_classes=5),
            places=12,
        )

    def test_color_and_texture_match_quantify(self) -> None:
        try:
            from shiftbench.shift_quantification_metrics.image_based.color import (
                quantify_color_shift,
            )
            from shiftbench.shift_quantification_metrics.image_based.texture import (
                quantify_texture_shift,
            )
        except ModuleNotFoundError:
            self.skipTest("requires opencv-python and scikit-image")

        rng = np.random.default_rng(21)
        imgs_a = [rng.integers(0, 255, (16, 16, 3)).astype(np.uint8) for _ in range(3)]
        imgs_b = [rng.integers(0, 128, (16, 16, 3)).astype(np.uint8) for _ in range(3)]

        self.assertAlmostEqual(
            self._via_summaries("color_js", imgs_a, imgs_b),
            quantify_color_shift(imgs_a, imgs_b),
            places=12,
        )
        self.assertAlmostEqual(
            self._via_summaries("texture_js", imgs_a, imgs_b),
            quantify_texture_shift(imgs_a, imgs_b),
            places=12,
        )


if __name__ == "__main__":
    unittest.main()

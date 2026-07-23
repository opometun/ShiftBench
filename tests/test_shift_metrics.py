"""Characterization tests for shift_quantification_metrics.

These pin the CURRENT numeric behavior of the pure functions before Phase 5
reshapes the package (docs/integration-plan.md). Where existing behavior is
questionable it is pinned anyway and marked with a comment, so any change to
it is a decision, not an accident.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shiftbench.shift_quantification_metrics.distances import js_distance
from shiftbench.shift_quantification_metrics.label_based.class_frequency import (
    class_frequency,
    quantify_class_frequency_shift,
)
from shiftbench.shift_quantification_metrics.label_based.class_presence import (
    class_presence_frequency,
    class_presence_per_image,
    quantify_class_presence_shift,
)
from shiftbench.shift_quantification_metrics.label_based.scene_complexity import (
    quantify_scene_complexity_shift,
    scene_complexity,
    scene_complexity_histogram,
)

try:
    from shiftbench.shift_quantification_metrics.image_based.color import (
        hsv_histogram,
        quantify_color_shift,
    )
    from shiftbench.shift_quantification_metrics.image_based.texture import (
        lbp_histogram,
        quantify_texture_shift,
    )

    HAS_IMAGE_DEPS = True
except ModuleNotFoundError:
    HAS_IMAGE_DEPS = False


class JsDistanceTest(unittest.TestCase):
    def test_identical_distributions_score_zero(self) -> None:
        p = np.array([0.5, 0.3, 0.2])

        self.assertAlmostEqual(js_distance(p, p), 0.0, places=6)

    def test_disjoint_distributions_score_one(self) -> None:
        self.assertAlmostEqual(
            js_distance(np.array([1.0, 0.0]), np.array([0.0, 1.0])), 1.0, places=5
        )

    def test_is_symmetric(self) -> None:
        a = np.array([0.7, 0.2, 0.1])
        b = np.array([0.1, 0.1, 0.8])

        self.assertAlmostEqual(js_distance(a, b), js_distance(b, a), places=12)

    def test_bounded_between_zero_and_one(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(20):
            a = rng.random(8)
            b = rng.random(8)

            self.assertGreaterEqual(js_distance(a, b), 0.0)
            self.assertLessEqual(js_distance(a, b), 1.0)

    def test_normalizes_unnormalized_inputs(self) -> None:
        # Callers pass raw counts; the eps-smoothing and division handle it.
        self.assertAlmostEqual(
            js_distance(np.array([2.0, 2.0]), np.array([500.0, 500.0])),
            0.0,
            places=6,
        )


class ClassFrequencyTest(unittest.TestCase):
    def test_known_mask_gives_exact_distribution(self) -> None:
        mask = np.array([[0, 0], [1, 2]])

        np.testing.assert_allclose(
            class_frequency(mask, num_classes=3), [0.5, 0.25, 0.25]
        )

    def test_distribution_sums_to_one(self) -> None:
        rng = np.random.default_rng(1)
        mask = rng.integers(0, 5, (16, 16))

        self.assertAlmostEqual(class_frequency(mask, num_classes=5).sum(), 1.0)

    def test_out_of_range_ids_grow_the_vector(self) -> None:
        # Pinned quirk: an id >= num_classes silently lengthens the output
        # (np.bincount minlength is a floor, not a cap), which would crash the
        # vstack in quantify_* on mixed datasets. Phase 5 should decide whether
        # to clip, error, or keep this.
        mask = np.array([[0, 5]])

        self.assertEqual(class_frequency(mask, num_classes=3).shape, (6,))

    def test_identical_datasets_shift_zero(self) -> None:
        masks = [np.array([[0, 1], [2, 2]]), np.array([[1, 1], [0, 2]])]

        self.assertAlmostEqual(
            quantify_class_frequency_shift(masks, list(masks), num_classes=3),
            0.0,
            places=6,
        )

    def test_area_imbalance_is_detected(self) -> None:
        road_heavy = [np.full((8, 8), 0)]
        car_heavy = [np.full((8, 8), 1)]

        shift = quantify_class_frequency_shift(road_heavy, car_heavy, num_classes=2)

        self.assertGreater(shift, 0.9)


class ClassPresenceTest(unittest.TestCase):
    def test_presence_is_binary_regardless_of_area(self) -> None:
        # One pixel of class 1 counts the same as half the image.
        mask = np.array([[1, 0], [0, 0]])

        np.testing.assert_allclose(
            class_presence_per_image(mask, num_classes=3), [1.0, 1.0, 0.0]
        )

    def test_frequency_counts_images_not_pixels(self) -> None:
        masks = [np.array([[0, 0]]), np.array([[0, 1]]), np.array([[1, 1]])]

        np.testing.assert_allclose(
            class_presence_frequency(masks, num_classes=2), [2.0, 2.0]
        )

    def test_identical_datasets_shift_zero(self) -> None:
        masks = [np.array([[0, 1]]), np.array([[1, 1]])]

        self.assertAlmostEqual(
            quantify_class_presence_shift(masks, list(masks), num_classes=2),
            0.0,
            places=6,
        )


class SceneComplexityTest(unittest.TestCase):
    def test_counts_distinct_classes(self) -> None:
        self.assertEqual(scene_complexity(np.array([[0, 3], [3, 7]]), 10), 3)

    def test_histogram_sums_to_one(self) -> None:
        masks = [np.array([[0, 1]]), np.array([[0, 0]]), np.array([[0, 1]])]

        hist = scene_complexity_histogram(masks, num_classes=2)

        self.assertAlmostEqual(hist.sum(), 1.0)
        # Bins are complexity values 0..num_classes: one 1-class image, two 2-class.
        np.testing.assert_allclose(hist, [0.0, 1 / 3, 2 / 3])

    def test_identical_datasets_shift_zero(self) -> None:
        masks = [np.array([[0, 1], [1, 1]]), np.array([[0, 0], [0, 0]])]

        self.assertAlmostEqual(
            quantify_scene_complexity_shift(masks, list(masks), num_classes=2),
            0.0,
            places=6,
        )

    def test_simpler_scenes_are_detected(self) -> None:
        varied = [np.array([[0, 1], [2, 3]]) for _ in range(4)]
        flat = [np.zeros((2, 2), dtype=int) for _ in range(4)]

        self.assertGreater(
            quantify_scene_complexity_shift(varied, flat, num_classes=4), 0.9
        )


@unittest.skipUnless(HAS_IMAGE_DEPS, "requires opencv-python and scikit-image")
class ImageMetricsTest(unittest.TestCase):
    def test_hsv_histogram_shape_and_mass_placement(self) -> None:
        solid_red = np.zeros((8, 8, 3), dtype=np.uint8)
        solid_red[..., 0] = 255

        hist = hsv_histogram(solid_red)

        # 32 hue + 16 saturation + 16 value bins. density=True normalizes per
        # bin WIDTH, so segments do not individually sum to 1 — pinned as-is.
        self.assertEqual(hist.shape, (64,))
        self.assertEqual(hist[:32].argmax(), 0)  # red hue in the first bin
        self.assertEqual(hist[32:48].argmax(), 15)  # fully saturated
        self.assertEqual(hist[48:].argmax(), 15)  # fully bright

    def test_lbp_histogram_is_a_distribution(self) -> None:
        rng = np.random.default_rng(2)
        gray = rng.integers(0, 255, (16, 16)).astype(np.uint8)

        hist = lbp_histogram(gray)

        self.assertEqual(hist.shape, (10,))  # P + 2 uniform-LBP bins
        self.assertAlmostEqual(hist.sum(), 1.0)  # bin width is 1 here

    def test_identical_datasets_shift_zero(self) -> None:
        rng = np.random.default_rng(3)
        imgs = [rng.integers(0, 255, (16, 16, 3)).astype(np.uint8) for _ in range(2)]

        self.assertAlmostEqual(quantify_color_shift(imgs, list(imgs)), 0.0, places=6)
        self.assertAlmostEqual(quantify_texture_shift(imgs, list(imgs)), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()

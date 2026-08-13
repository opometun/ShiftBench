"""Tests for decoding manifest-listed files into metric-ready arrays.

Needs pillow (the 'image' / 'features' extras); skipped where it is absent.
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from PIL import Image

    HAS_PIL = True
except ModuleNotFoundError:
    HAS_PIL = False

if HAS_PIL:
    from shiftbench.datasets.loaders import load_masks, load_rgb_images 


class NpyMaskLoaderTest(unittest.TestCase):
    """The .npy mask path needs only numpy, so it runs everywhere."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.directory = Path(self._directory.name)
        self.addCleanup(self._directory.cleanup)

    def test_npy_masks_load_without_pillow(self) -> None:
        from shiftbench.datasets.loaders import load_masks

        path = self.directory / "mask.npy"
        mask = np.array([[0, 1], [2, 2]], dtype=np.uint8)
        np.save(path, mask)

        masks = load_masks([str(path)], remap=None)

        self.assertTrue(np.issubdtype(masks[0].dtype, np.integer))
        np.testing.assert_array_equal(masks[0], mask)

    def test_3d_npy_in_the_mask_column_is_rejected(self) -> None:
        from shiftbench.datasets.loaders import load_masks

        path = self.directory / "bad.npy"
        np.save(path, np.zeros((2, 2, 3)))

        with self.assertRaisesRegex(ValueError, "single-channel"):
            load_masks([str(path)], remap=None)

    def test_custom_lut_remapping(self):
        from shiftbench.datasets.loaders import load_masks

        path = self.directory / "mask.png"
        raw_mask = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        Image.fromarray(raw_mask, mode="L").save(path)
        custom_lut = np.arange(256, dtype=np.uint8)
        custom_lut[0] = 10
        custom_lut[1] = 11
        custom_lut[2] = 12
        custom_lut[3] = 13
        masks = load_masks([str(path)], remap=custom_lut)

        expected = np.array([[10, 11], [12, 13]], dtype=np.uint8)
        np.testing.assert_array_equal(masks[0], expected)

    def test_default_cityscapes_lut_remapping(self):
        from shiftbench.datasets.loaders import load_masks

        path = self.directory / "mask.png"
        raw_mask = np.array([[7, 8], [11, 12]], dtype=np.uint8)
        Image.fromarray(raw_mask, mode="L").save(path)
        masks = load_masks([str(path)])

        expected = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        np.testing.assert_array_equal(masks[0], expected)


@unittest.skipUnless(HAS_PIL, "requires pillow")
class LoadersTest(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.directory = Path(self._directory.name)
        self.addCleanup(self._directory.cleanup)

    def test_images_decode_as_rgb_uint8(self) -> None:
        path = self.directory / "img.png"
        Image.fromarray(np.full((4, 6), 200, dtype=np.uint8), mode="L").save(path)

        images = load_rgb_images([str(path)])

        self.assertEqual(images[0].shape, (4, 6, 3))  # grayscale promoted to RGB
        self.assertEqual(images[0].dtype, np.uint8)

    def test_masks_decode_as_2d_integer_ids(self) -> None:
        path = self.directory / "mask.png"
        mask = np.array([[0, 1], [2, 2]], dtype=np.uint8)
        Image.fromarray(mask, mode="L").save(path)

        masks = load_masks([str(path)], remap=None)

        self.assertEqual(masks[0].shape, (2, 2))
        self.assertTrue(np.issubdtype(masks[0].dtype, np.integer))
        np.testing.assert_array_equal(masks[0], mask)

    def test_rgb_file_in_the_mask_column_is_rejected(self) -> None:
        path = self.directory / "not_a_mask.png"
        Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8), mode="RGB").save(path)

        with self.assertRaises(ValueError) as caught:
            load_masks([str(path)], remap=None)

        self.assertIn("single-channel", str(caught.exception))


if __name__ == "__main__":
    unittest.main()

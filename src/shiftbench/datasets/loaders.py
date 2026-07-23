"""Decode manifest-listed files into the arrays the shift metrics consume.

Kept separate from manifest.py so path handling stays dependency-free. PIL is
imported lazily: decoding image files needs pillow (the 'features' and 'image'
extras), but .npy masks load with numpy alone.
"""

from __future__ import annotations

import numpy as np


def load_rgb_images(paths: list[str]) -> list[np.ndarray]:
    """Decode images as (H, W, 3) uint8 RGB arrays.

    This is the input quantify_color_shift and quantify_texture_shift expect;
    they convert out of RGB themselves.
    """
    from PIL import Image

    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(np.array(image.convert("RGB")))
    return images


def load_masks(paths: list[str]) -> list[np.ndarray]:
    """Decode semantic masks as (H, W) integer class-id arrays.

    .npy masks load directly; anything else is decoded with pillow.

    Raises:
        ValueError: If a mask decodes to more than one channel — an RGB file
            in the mask column would otherwise flow into np.bincount and
            produce a wrong distribution instead of an error.
    """
    masks = []
    for path in paths:
        if str(path).endswith(".npy"):
            mask = np.load(path)
        else:
            from PIL import Image

            with Image.open(path) as mask_image:
                mask = np.array(mask_image)
        if mask.ndim != 2:
            raise ValueError(f"Mask is not single-channel: {path}")
        masks.append(mask.astype(np.int64, copy=False))
    return masks

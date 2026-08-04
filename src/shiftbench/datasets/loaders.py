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


# class id → trainId mapping
# There are 34 Cityscapes classes but only 19 are typically used in training.
# We set all unused class indices to 255. This index will be ignored.
# This mapping was taken from the official Cityscapes GitHub repo: 
# https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
ID_TO_TRAINID = {                                             # CATEGORY
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,   # void
    7: 0,   8: 1,                                             # flat
    9: 255, 10: 255,                                          # flat
    11: 2, 12: 3, 13: 4,                                      # construction
    14: 255, 15: 255, 16: 255,                                # construction
    17: 5, 18: 255,                                           # object
    19: 6, 20: 7,                                             # object
    21: 8, 22: 9,                                             # nature
    23: 10,                                                   # sky
    24: 11, 25: 12,                                           # human
    26: 13, 27: 14, 28: 15,                                   # vehicle
    29: 255, 30: 255,                                         # vehicle
    31: 16, 32: 17, 33: 18,                                   # vehicle
    -1: 255,                                                  # vehicle
}

# Build LUT once
LUT = np.full(256, 255, dtype=np.uint8)
for id_, trainId in ID_TO_TRAINID.items():
    LUT[id_] = trainId


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

        mask = mask.astype(np.uint8, copy=False)

        # Apply Cityscapes id → trainId mapping
        train_mask = LUT[mask]
    
        masks.append(train_mask)

    return masks

"""This file contains all functions relevant for the scene complexity shift quantification."""

import numpy as np

from shiftbench.metrics.distances import js_distance


def scene_complexity(mask:np.ndarray, num_classes:int) -> int:
    """Count how many distinct classes appear in a single semantic mask."""
    # Exclude ignore/void (255 after the trainId remap) and out-of-range labels.
    # Counting void as a distinct class adds +1 to every image that contains any
    # -- which is all of Cityscapes and GTA-V but almost none of Synscapes -- so
    # the complexity distributions would differ by a constant offset that has
    # nothing to do with how many real classes the scenes contain.
    labels = mask.ravel()
    labels = labels[labels < num_classes]
    counts = np.bincount(labels, minlength=num_classes)
    return int((counts > 0).sum())


def scene_complexity_histogram(masks:list, num_classes:int) -> np.ndarray:
    """Compute normalized scene complexity distribution of a dataset."""
    # Get scene complexity of each image
    complexities = [
        scene_complexity(mask, num_classes)
        for mask in masks
    ]

    # Get histogram over scene complexity values
    hist, _ = np.histogram(
        complexities,
        bins=np.arange(num_classes + 2),
        density=False
    )

    # Normalize to probability distribution
    hist = hist.astype(np.float64)
    hist /= hist.sum()

    return hist


def quantify_scene_complexity_shift(
    train_ds_masks: list,
    inference_ds_masks: list,
    num_classes: int = 19,
) -> float:
    """Compute JS-based scene complexity shift between two datasets."""
    # Get dataset-level scene complexity distributions
    train_dist = scene_complexity_histogram(train_ds_masks, num_classes)
    inference_dist = scene_complexity_histogram(inference_ds_masks, num_classes)

    # JS-distance
    shift = js_distance(a=train_dist, b=inference_dist)
    return shift

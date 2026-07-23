"""This file contains all functions relevant for the class presence-frequency shift quantification."""

import numpy as np

from shiftbench.metrics.distances import js_distance


def class_presence_per_image(mask:np.ndarray, num_classes:int) -> np.ndarray:
    """
    Compute a binary presence vector for a single mask.
    1 if class appears in the image, else 0.
    """
    counts = np.bincount(mask.ravel(), minlength=num_classes)
    return (counts > 0).astype(np.float64)


def class_presence_frequency(masks:list, num_classes:int) -> np.ndarray:
    """Count in how many images each class appears.

    Args:
        masks: list of HxW semantic label masks.
        num_classes: total number of classes in the dataset.

    Returns an array of shape (num_classes,).
    """
    per_image_presence = np.vstack([
        class_presence_per_image(mask, num_classes)
        for mask in masks
    ])
    return per_image_presence.sum(axis=0)


def quantify_class_presence_shift(
    train_ds_masks: list,
    inference_ds_masks: list,
    num_classes: int = 19,
) -> float:
    """Compute JS-based class presence-frequency shift between two datasets."""
    # Get dataset-level class presence frequency vectors
    train_counts = class_presence_frequency(train_ds_masks, num_classes)
    inference_counts = class_presence_frequency(inference_ds_masks, num_classes)

    # Normalize to probability distributions
    train_dist = train_counts / train_counts.sum()
    inference_dist = inference_counts / inference_counts.sum()

    # JS-distance
    shift = js_distance(a=train_dist, b=inference_dist)
    return shift

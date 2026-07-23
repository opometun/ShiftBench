"""This file contains all functions relevant for the class frequency-based shift quantification."""

import numpy as np

from shiftbench.metrics.distances import js_distance


def class_frequency(mask:np.ndarray, num_classes:int) -> np.ndarray:
    """Compute normalized class-frequency histogram from a semantic label mask.
    
    Args:
        mask: HxW integer array with class IDs in [0, num_classes-1].
        num_classes: Total number of classes in the dataset.
    
    Returns a probability distribution of shape (num_classes,).
    """
    # Flatten mask and count occurrences
    hist = np.bincount(mask.ravel(), minlength=num_classes).astype(np.float64)

    # Normalize to probability distribution
    hist /= hist.sum()

    return hist


def quantify_class_frequency_shift(
    train_ds_masks:list,
    inference_ds_masks:list, 
    num_classes:int=19,
) -> float:
    """Compute JS-based class frequency distribution shift between two datasets."""
    # Get dataset-level class distributions
    train_dist = np.vstack([class_frequency(mask, num_classes) for mask in train_ds_masks]).mean(axis=0)
    inference_dist = np.vstack([class_frequency(mask, num_classes) for mask in inference_ds_masks]).mean(axis=0)

    # JS-distance
    shift = js_distance(a=train_dist, b=inference_dist)
    return shift

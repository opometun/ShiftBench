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
    # Drop ignore/void pixels and any out-of-range label before counting.
    #
    # Masks arrive already remapped to trainIds, so void is 255. Passing that
    # to np.bincount returns a 256-long array whenever a mask contains any void
    # pixel, while a mask with none returns a 19-long one -- so np.vstack over
    # a dataset raises "array dimensions must match". Synscapes triggers this
    # because some of its masks are void-free; Cityscapes and GTA-V never are.
    #
    # Excluding void also fixes a quieter correctness problem: counted as a
    # class, void would be normalised into the distribution, and it accounts
    # for ~14% of Cityscapes pixels versus ~0.2% of Synscapes. The resulting
    # "class frequency shift" would then largely measure how much void each
    # dataset has rather than how its classes are distributed.
    labels = mask.ravel()
    labels = labels[labels < num_classes]
    hist = np.bincount(labels, minlength=num_classes).astype(np.float64)

    # Normalize to probability distribution
    total = hist.sum()
    if total == 0:
        return hist  # mask was entirely void; contributes nothing
    hist /= total

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

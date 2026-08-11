"""Assemble full datasets."""

import csv
import numpy as np

from shiftbench.config import ExperimentConfig
from shiftbench.datasets.loaders import (
    load_rgb_images, 
    load_masks,
)
from shiftbench.datasets.manifest import (
    load_image_paths_from_config,
    load_mask_paths_from_config,
)


def load_datasets_by_split(config:ExperimentConfig) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Load the datasets according to splits.
    
    Returns a dictionary containing a list of (img, mask) tuples for each split.
    """
    schema = config.dataset

    # Load all image and mask paths
    img_paths = load_image_paths_from_config(schema)
    mask_paths = load_mask_paths_from_config(schema)

    # Decode images and masks
    images = load_rgb_images(paths=img_paths)
    masks  = load_masks(paths=mask_paths)

    # Split by train/val/test
    with schema.path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    split_column = schema.split_column
    splits = {split: [] for split in schema.allowed_splits}

    for row, img, mask in zip(rows, images, masks):
        split = row[split_column]
        splits[split].append((img, mask))

    return splits
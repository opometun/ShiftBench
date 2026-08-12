"""Assemble full datasets."""

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


def load_datasets_by_split(
    config: ExperimentConfig,
    splits_to_load: list[str] | None = None,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Load datasets grouped by split, decoding only what is needed.

    Returns a dictionary containing a list of (img, mask) tuples for each split requested.
    """ 
    schema = config.dataset

    if splits_to_load is None:
        splits_to_load = schema.allowed_splits

    invalid = set(splits_to_load) - set(schema.allowed_splits)
    if invalid:
        raise ValueError(
            f"Invalid splits requested: {invalid}. "
            f"Allowed: {schema.allowed_splits}"
        )
    
    splits = {}
    for split in splits_to_load:
        img_paths  = load_image_paths_from_config(schema, split=split)
        mask_paths = load_mask_paths_from_config(schema, split=split)

        images = load_rgb_images(paths=img_paths)
        masks  = load_masks(paths=mask_paths)

        splits[split] = list(zip(images, masks))

    return splits
"""Scaffold for frozen DINOv2 feature extraction."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shiftbench.datasets.manifest import load_image_paths  # noqa: E402
from shiftbench.features.dinov2 import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    extract_features,
    get_device,
    load_frozen_dinov2,
)

BATCH_SIZE = 32


def parse_args():
    parser = argparse.ArgumentParser(description="Extract frozen DINOv2 CLS features.")
    parser.add_argument(
        "input_manifest",
        help="Path to the input dataset or mixture manifest.",
    )
    parser.add_argument(
        "output_path",
        help="Path to write the output .npy feature array.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face encoder id. Changing this changes every distance.",
    )
    parser.add_argument(
        "--image-column",
        default=None,
        help=(
            "Manifest column holding image paths. Defaults to guessing; set it "
            "to match the image_column of your experiment config."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    processor, model = load_frozen_dinov2(device, args.model)

    image_paths = load_image_paths(args.input_manifest, args.image_column)
    feature_batches = []
    for start_index in range(0, len(image_paths), BATCH_SIZE):
        image_batch = []
        for image_path in image_paths[start_index : start_index + BATCH_SIZE]:
            with Image.open(image_path) as image:
                image_batch.append(image.convert("RGB"))

        feature_batches.append(extract_features(processor, model, image_batch, device).to("cpu"))
    features = torch.cat(feature_batches, dim=0)
    np.save(args.output_path, features.numpy())


if __name__ == "__main__":
    main()

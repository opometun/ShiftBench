"""Extract frozen-encoder image features from a manifest."""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shiftbench.config import load_experiment_config  # noqa: E402
from shiftbench.datasets.manifest import (  # noqa: E402
    load_image_paths,
    load_image_paths_from_config,
)
from shiftbench.features.device import get_device  # noqa: E402
from shiftbench.features.registry import ENCODERS, get_encoder  # noqa: E402
from shiftbench.provenance import write_feature_provenance  # noqa: E402

BATCH_SIZE = 32


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract frozen image features for distribution-shift metrics."
    )
    parser.add_argument(
        "output_path",
        help="Path to write the output .npy feature array.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--manifest",
        help="Path to a dataset or mixture manifest, read without a config.",
    )
    source.add_argument(
        "--config",
        help=(
            "Experiment TOML supplying both the manifest path and its "
            "image_column. Preferred over --manifest."
        ),
    )
    parser.add_argument(
        "--encoder",
        default="dinov2",
        choices=sorted(ENCODERS),
        help=(
            "Which frozen encoder to run. Features from different encoders "
            "are not comparable to each other."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Hugging Face encoder id. Defaults to the chosen encoder's own "
            "default. Changing this changes every distance."
        ),
    )
    parser.add_argument(
        "--image-column",
        default=None,
        help=(
            "Manifest column holding image paths. Overrides the config's "
            "image_column. Without either, the column is guessed."
        ),
    )
    return parser.parse_args()


def resolve_image_paths(args):
    """Read image paths from whichever source the caller specified."""
    if args.config is None:
        return load_image_paths(args.manifest, args.image_column)

    config = load_experiment_config(Path(args.config))
    schema = config.dataset
    if args.image_column is not None:
        schema = replace(schema, image_column=args.image_column)
    return load_image_paths_from_config(schema)


def main():
    args = parse_args()
    encoder = get_encoder(args.encoder)
    model_name = args.model or encoder.default_model_name

    device = get_device()
    processor, model = encoder.load(device, model_name)

    image_paths = resolve_image_paths(args)
    feature_batches = []
    for start_index in range(0, len(image_paths), BATCH_SIZE):
        image_batch = []
        for image_path in image_paths[start_index : start_index + BATCH_SIZE]:
            with Image.open(image_path) as image:
                image_batch.append(image.convert("RGB"))

        batch_features = encoder.extract(processor, model, image_batch, device)
        feature_batches.append(batch_features.to("cpu"))

    features = torch.cat(feature_batches, dim=0).numpy()
    # np.save appends .npy itself, so resolve the real name the sidecar pairs with.
    features_path = args.output_path
    if not features_path.endswith(".npy"):
        features_path = f"{features_path}.npy"
    np.save(features_path, features)
    sidecar = write_feature_provenance(features_path, encoder.name, model_name)
    print(f"{features.shape} features from {encoder.name} ({model_name})")
    print(f"provenance written to {sidecar}")


if __name__ == "__main__":
    main()

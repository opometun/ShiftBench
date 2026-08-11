"""Extract frozen-encoder image features from a manifest."""
import argparse
import sys
import warnings
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
            "input_column. Preferred over --manifest."
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
            "input_column. Without either, the column is guessed."
        ),
    )
    parser.add_argument(
        "--split", 
        default=None,
        help=(
            "Dataset split to extract features for. "
            "If omitted, all rows of the manifest will be considered."
        ),
    )
    args = parser.parse_args()
    
    if (args.config is not None) and (args.split is not None):
        config = load_experiment_config(Path(args.config))
        allowed = config.dataset.allowed_splits
        if args.split not in allowed:
            raise ValueError(
                f"Invalid split '{args.split}'. Allowed: {allowed}"
            )
    return args


def resolve_image_paths(args) -> list[str]:
    """Read image paths from whichever source the caller specified."""
    # Read paths directly from manifest
    if args.config is None:
        paths = load_image_paths(
            input_manifest = args.manifest, 
            image_column = args.image_column, 
            split = args.split,
        )
    # Read paths via config information
    else:
        config = load_experiment_config(Path(args.config))
        schema = config.dataset
        if args.image_column is not None:
            schema = replace(schema, image_column=args.image_column)
        paths = load_image_paths_from_config(schema, split=args.split)
    return paths


def main():
    args = parse_args()

    if args.output_path is not None:
        features_path = Path(args.output_path)
        if features_path.suffix != ".npy":
            features_path = features_path.with_suffix(".npy")
        if features_path.exists():
            warnings.warn(
                f"Output file already exists and will be overwritten: {features_path}",
                category=UserWarning,
            )
        features_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        features_path = None
    
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
    # save the embedding in a .npy file
    np.save(features_path, features)
    # save the meta data (encoder, model) in a .json file (extended from .npy file)
    sidecar = write_feature_provenance(features_path, encoder.name, model_name)
    print(f"{features.shape} features from {encoder.name} ({model_name})")
    print(f"provenance written to {sidecar}")


if __name__ == "__main__":
    main()

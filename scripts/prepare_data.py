"""Prepare the data for an experiment. 

You only need to run this file once per experiment, because the resized images are saved on disk.
You do not need to run this file if you want to work with the original, potentially unequally sized images.
Note that different experiments might use the same data samples. So preparing the data for one experiment might have already modified (parts of) the data of another.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shiftbench.config import load_experiment_config
from shiftbench.datasets.prepare import resize_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-path",
        required=True,
        type=str,
        help="Path to experiment TOML file.",
    )
    args = parser.parse_args()

    config = load_experiment_config(Path(args.experiment_path))

    manifest_path = config.dataset.path
    input_column = config.dataset.input_column
    label_column = config.dataset.label_column

    # Resize input images
    resize_images(
        input_manifest=manifest_path,
        image_column=input_column,
        target_resolution=(1440, 720),
        interpolation_method="lanczos",
    )

    # Resize segmentation masks
    resize_images(
        input_manifest=manifest_path,
        image_column=label_column,
        target_resolution=(1440, 720),
        interpolation_method="nearest",
    )


if __name__ == "__main__":
    main()

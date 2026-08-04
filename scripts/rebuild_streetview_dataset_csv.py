"""Rebuild the dataset .csv for a street view image segmentation experiment."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shiftbench.config import load_experiment_config
from shiftbench.datasets.prepare import build_street_view_dataset_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-path",
        type=str,
        required=True,
        help="Path to experiment TOML file.",
    )
    parser.add_argument(
        "--ds-root-path",
        type=str,
        default="./streetViewData",
        help="Path to the folder that stores the data.",
    )
    args = parser.parse_args()

    config = load_experiment_config(Path(args.experiment_path))

    manifest_path = config.dataset.path
    train_selection = config.train_selection
    seed = config.seed

    build_street_view_dataset_csv(
        datasets_root=args.ds_root_path,
        output_csv=manifest_path,
        train_selection=train_selection,
        seed=seed,
    )


if __name__ == "__main__":
    main()

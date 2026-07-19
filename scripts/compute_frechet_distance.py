"""Compute Fréchet distance between two feature distributions."""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shiftbench.metrics import frechet_distance  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stats_a_path")
    parser.add_argument("stats_b_path")
    parser.add_argument(
        "-o",
        "--output-path",
        help="Path to write the computed Fréchet distance.",
    )
    args = parser.parse_args()
    stats_a = np.load(args.stats_a_path)
    stats_b = np.load(args.stats_b_path)
    distance = frechet_distance(
        stats_a["mu"],
        stats_a["sigma"],
        stats_b["mu"],
        stats_b["sigma"],
    )
    if args.output_path:
        with open(args.output_path, "w", encoding="utf-8") as file:
            file.write(f"{distance}\n")
    else:
        print(distance)


if __name__ == "__main__":
    main()
